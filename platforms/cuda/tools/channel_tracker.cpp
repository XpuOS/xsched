// Linux RM ABI subset from NVIDIA open-gpu-kernel-modules, tag 595.71.05:
// nvos.h and ctrl/ctrl2080/ctrl2080fifo.h (MIT). Keep layout assertions below.
#include "xsched/cuda/hal/level3/channel_control.h"
#include <cerrno>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <vector>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace xsched::cuda::channel;

namespace
{
struct Alloc21 { uint32_t client, parent, object, cls; uint64_t params; uint32_t size, status; };
struct Alloc64 { uint32_t client, parent, object, cls; uint64_t params, rights; uint32_t size, flags, status; };
struct Free { uint32_t client, parent, object, status; };
struct RmControl { uint32_t client, object, cmd, flags; uint64_t params; uint32_t size, status; };
struct Disable {
    uint8_t disable;
    uint32_t count;
    uint8_t only_scheduling, rewind;
    alignas(8) uint64_t event;
    uint32_t clients[64], channels[64];
};
static_assert(sizeof(Alloc21) == 32 && sizeof(Alloc64) == 48 && sizeof(Free) == 16);
static_assert(sizeof(RmControl) == 32 && sizeof(Disable) == 536);
static_assert(offsetof(Disable, event) == 16 && offsetof(Disable, clients) == 24);
using Key = std::pair<uint32_t, uint32_t>;
struct Object { uint32_t parent, cls; };
struct Owner {
    bool paused = false, uncertain = false;
    int fd = -1;
    unsigned subdevice = 0;
    std::vector<uint32_t> channels;
};
struct State {
    std::mutex mutex;
    std::map<Key, Object> objects;
    std::map<Key, Owner> owners;
};
State &GetState() { static auto *state = new State; return *state; }
#ifdef XSCHED_CHANNEL_TRACKER_TEST
int TestIoctl(int fd, unsigned long req, void *arg);
int RawIoctl(int fd, unsigned long req, void *arg) { return TestIoctl(fd, req, arg); }
#else
int RawIoctl(int fd, unsigned long req, void *arg) { return syscall(SYS_ioctl, fd, req, arg); }
#endif

bool Descendant(State &s, uint32_t client, uint32_t object, uint32_t ancestor)
{
    for (size_t depth = 0; depth <= s.objects.size(); ++depth) {
        if (object == ancestor) return true;
        auto it = s.objects.find({client, object});
        if (it == s.objects.end() || it->second.parent == object) return false;
        object = it->second.parent;
    }
    return false;
}
}

// This preload library must be loaded before CUDA initialization. Never discovers
// objects belonging to other processes; only successful allocations are recorded.
extern "C" int ioctl(int fd, unsigned long req, ...) noexcept
{
    va_list args;
    va_start(args, req);
    void *arg = va_arg(args, void *);
    va_end(args);
    bool alloc21 = req == _IOWR('F', 0x2b, Alloc21);
    bool alloc64 = req == _IOWR('F', 0x2b, Alloc64);
    bool freeing = req == _IOWR('F', 0x29, Free);
    if (!arg || (!alloc21 && !alloc64 && !freeing)) return RawIoctl(fd, req, arg);
    auto &s = GetState();
    std::lock_guard<std::mutex> lock(s.mutex);
    uint32_t client, parent, object, cls = 0;
    if (freeing) {
        auto *p = static_cast<Free *>(arg);
        client = p->client; parent = p->parent; object = p->object;
    } else if (alloc64) {
        auto *p = static_cast<Alloc64 *>(arg);
        client = p->client; parent = p->parent; object = p->object; cls = p->cls;
    } else {
        auto *p = static_cast<Alloc21 *>(arg);
        client = p->client; parent = p->parent; object = p->object; cls = p->cls;
    }
    for (const auto &[key, owner] : s.owners) {
        if (key.first != client) continue;
        // Context/group lifetime must outlive its registered queue. New channels
        // while paused would escape the disabled set, so reject that allocation.
        if ((freeing && Descendant(s, client, key.second, object)) ||
            ((owner.paused || owner.uncertain) &&
             Descendant(s, client, freeing ? object : parent, key.second))) {
            errno = EBUSY;
            return -1;
        }
    }
    int result = RawIoctl(fd, req, arg);
    int saved_errno = errno;
    uint32_t status = freeing ? static_cast<Free *>(arg)->status
        : alloc64 ? static_cast<Alloc64 *>(arg)->status : static_cast<Alloc21 *>(arg)->status;
    if (result == 0 && status == 0) {
        if (freeing) {
            std::vector<Key> removed;
            for (const auto &[key, value] : s.objects) {
                (void)value;
                if (key.first == client && Descendant(s, client, key.second, object)) removed.push_back(key);
            }
            for (auto key : removed) s.objects.erase(key);
        } else if (alloc64) {
            auto *p = static_cast<Alloc64 *>(arg);
            s.objects[{p->client, p->object}] = {parent, cls};
        } else {
            auto *p = static_cast<Alloc21 *>(arg);
            s.objects[{p->client, p->object}] = {parent, cls};
        }
    }
    errno = saved_errno;
    return result;
}

// op: 0 acquire (handle is a compute channel, returns its group), 1 pause,
// 2 resume, 3 release. Nonzero errors must never be interpreted as suspension.
extern "C" int xsched_cuda_channel_control_v2(int fd, unsigned client, unsigned subdevice,
                                             unsigned *handle, int op)
{
    if (!handle || fd < 0 || !client || !subdevice || op < kAcquire || op > kRelease) return EINVAL;
    auto &s = GetState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (op == kAcquire) {
        auto channel = s.objects.find({client, *handle});
        if (channel == s.objects.end() || channel->second.cls != 0xc96f) {
            fprintf(stderr, "CHANNEL tracker: missing compute channel %x (client %x, %zu tracked objects)\n", *handle, client, s.objects.size());
            return ENOENT;
        }
        Key key{client, channel->second.parent};
        auto group = s.objects.find(key);
        if (group == s.objects.end() || group->second.cls != 0xa06c) return ENOENT;
        Owner owner;
        owner.fd = fd;
        owner.subdevice = subdevice;
        if (!s.owners.emplace(key, std::move(owner)).second) return EBUSY;
        *handle = key.second;
        return 0;
    }
    auto it = s.owners.find({client, *handle});
    if (it == s.owners.end()) return ENOENT;
    auto &owner = it->second;
    if (owner.fd != fd || owner.subdevice != subdevice) return EINVAL;
    if (op == kRelease) {
        if (owner.paused || owner.uncertain) return EBUSY;
        s.owners.erase(it);
        return 0;
    }
    bool disable = op == kPause;
    if (disable && owner.uncertain) return EIO;
    if (!owner.uncertain && owner.paused == disable) return 0;
    if (disable) {
        owner.channels.clear();
        for (const auto &[key, obj] : s.objects) {
            if (key.first == client && obj.parent == *handle && obj.cls == 0xc96f)
                owner.channels.push_back(key.second);
        }
    }
    if (owner.channels.empty() || owner.channels.size() > 64) return EOVERFLOW;
    Disable p{};
    p.disable = disable;
    p.count = owner.channels.size();
    for (size_t i = 0; i < owner.channels.size(); ++i) {
        p.clients[i] = client;
        p.channels[i] = owner.channels[i];
    }
    RmControl c{client, subdevice, 0x2080110b, 0, reinterpret_cast<uint64_t>(&p), sizeof(p), 0};
    auto issue = [&]() {
        c.status = 0;
        int result = RawIoctl(fd, _IOWR('F', 0x2a, RmControl), &c);
        return result != 0 ? (errno ? errno : EIO) : int(c.status);
    };
    // A failed batch may have changed only some channels. Preserve the snapshot
    // and prohibit lifecycle changes until re-enabling the whole set succeeds.
    owner.uncertain = true;
    int error = issue();
    if (!disable)
        for (int retry = 0; error && retry < 2; ++retry) error = issue();
    if (error) {
        if (disable) {
            p.disable = false;
            int recovery = EIO;
            for (int retry = 0; recovery && retry < 3; ++retry) recovery = issue();
            if (!recovery) {
                owner.paused = false;
                owner.uncertain = false;
            }
            fprintf(stderr, "CHANNEL disable failed (%d), rollback=%d\n", error, recovery);
        }
        return error;
    }
    owner.paused = disable;
    owner.uncertain = false;
    return 0;
}

// XSched normally deep-binds the real driver, which bypasses ioctl interposition.
// Only this explicitly preloaded experimental helper changes libcuda loading.
#ifndef XSCHED_CHANNEL_TRACKER_TEST
extern "C" void *dlopen(const char *filename, int flags) noexcept
{
    using Open = void *(*)(const char *, int);
    static auto real_open = reinterpret_cast<Open>(dlsym(RTLD_NEXT, "dlopen"));
    if (!real_open) return nullptr;
    if (filename) {
        const char *name = std::strrchr(filename, '/');
        name = name ? name + 1 : filename;
        if (std::strcmp(name, "libcuda.so") == 0 || std::strncmp(name, "libcuda.so.", 11) == 0)
            flags &= ~RTLD_DEEPBIND;
    }
    return real_open(filename, flags);
}
#endif
