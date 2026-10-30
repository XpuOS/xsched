#include <cstring>
#include <cstdint>
#include <cstdio>
#include <map>
#include <cuxtra/cuxtra.h>
#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "xsched/cuda/hal/level3/channel.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

namespace
{
struct ContextOwners {
    std::mutex mutex;
    std::map<CUcontext, CUdevice> contexts;
};
ContextOwners &Owners() { static auto *owners = new ContextOwners; return *owners; }

#if defined(__linux__) && defined(__x86_64__)
bool ValidatedDriver()
{
    FILE *file = std::fopen("/proc/driver/nvidia/version", "r");
    if (!file) return false;
    char line[512] = {};
    char version[64];
    std::snprintf(version, sizeof(version), " %s ", channel::ValidatedDriverVersion);
    bool supported = std::fgets(line, sizeof(line), file) &&
                     std::strstr(line, version);
    std::fclose(file);
    return supported;
}
#endif
}

std::shared_ptr<HwQueue> CudaQueueLv3Channel::Create(CUstream stream)
{
#if defined(__linux__) && defined(__x86_64__)
    auto control = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, channel::ControlSymbol));
    if (!control || !ValidatedDriver()) {
        XWARN("CHANNEL requires the v2 tracker preloaded before CUDA initialization and driver %s",
              channel::ValidatedDriverVersion);
        return nullptr;
    }
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    CUcontext context = nullptr;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &context));
    if (owners.contexts.count(context)) {
        XWARN("CHANNEL context already has an owner; use separate contexts for independent scheduling");
        return nullptr;
    }
    // Private ToolsRm ABI used by the bundled cuXtra; validated on driver 595.71.05.
    const unsigned char bytes[16] = {0x14,0x06,0x18,0x0d,0x72,0xd6,0xcb,0x47,
                                     0xab,0x6e,0xb9,0x63,0xe5,0x1b,0xf4,0xc6};
    CUuuid uuid;
    std::memcpy(&uuid, bytes, sizeof(uuid));
    const void *raw = nullptr;
    if (Driver::GetExportTable(&raw, &uuid) != CUDA_SUCCESS || !raw) {
        XWARN("CHANNEL ToolsRm export table unavailable");
        return nullptr;
    }
    auto table = static_cast<const uintptr_t *>(raw);
    if (table[0] < 3 * sizeof(uintptr_t) || !table[1] || !table[2]) {
        XWARN("CHANNEL ToolsRm ABI unavailable");
        return nullptr;
    }
    unsigned handles[6] = {24, 0, 0, 0, 0, 0};
    unsigned channel_client = 0;
    auto get_handles = reinterpret_cast<CUresult (*)(void *, CUcontext)>(table[1]);
    auto get_channel = reinterpret_cast<CUresult (*)(unsigned *, unsigned *, CUcontext)>(table[2]);
    unsigned group = 0;
    if (get_handles(handles, context) != CUDA_SUCCESS ||
        get_channel(&channel_client, &group, context) != CUDA_SUCCESS) {
        XWARN("CHANNEL cannot query ToolsRm context handles");
        return nullptr;
    }
    unsigned client = handles[3], subdevice = handles[5];
    int fd = cuXtraGetRmControlFd();
    if (!client || !subdevice || !group || client != channel_client || fd < 0) return nullptr;
    int result = control(fd, client, subdevice, &group, channel::kAcquire);
    if (result != 0) {
        XWARN("CHANNEL registration failed (%d); no queue was registered", result);
        return nullptr;
    }
    auto queue = std::shared_ptr<CudaQueueLv3Channel>(
        new CudaQueueLv3Channel(stream, control, fd, client, subdevice, group));
    owners.contexts.emplace(context, queue->cudevice_);
    return queue;
#else
    UNUSED(stream);
    XWARN("CHANNEL requires Linux x86_64");
    return nullptr;
#endif
}

CudaQueueLv3Channel::CudaQueueLv3Channel(CUstream stream, Control control, int fd,
                                       unsigned client, unsigned subdevice, unsigned group)
    : CudaQueueLv1(stream), control_(control), fd_(fd), client_(client),
      subdevice_(subdevice), group_(group)
{
    XWARN("CHANNEL owns the context's compute group; independent copies may continue; DropCommands unsupported");
}

CudaQueueLv3Channel::~CudaQueueLv3Channel()
{
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    Resume();
    int result = control_(fd_, client_, subdevice_, &group_, channel::kRelease);
    XASSERT(result == 0, "CHANNEL release failed: %d", result);
    owners.contexts.erase(context_);
}

CUresult CudaQueueLv3Channel::DestroyContext(CUcontext context, bool legacy)
{
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    if (owners.contexts.count(context)) {
        XWARN("destroy CHANNEL XQueue and HwQueue before destroying its CUDA context");
        return CUDA_ERROR_NOT_READY;
    }
    return legacy ? Driver::CtxDestroy(context) : Driver::CtxDestroy_v2(context);
}

CUresult CudaQueueLv3Channel::ResetPrimaryContext(CUdevice device, bool legacy)
{
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    for (const auto &[context, owned_device] : owners.contexts) {
        UNUSED(context);
        if (owned_device == device) {
            XWARN("release CHANNEL queues on this device before primary context reset");
            return CUDA_ERROR_NOT_READY;
        }
    }
    return legacy ? Driver::DevicePrimaryCtxReset(device) : Driver::DevicePrimaryCtxReset_v2(device);
}

void CudaQueueLv3Channel::Pause()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_) return;
    int result = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        result = control_(fd_, client_, subdevice_, &group_, channel::kPause);
        if (!result) break;
        int recovery = control_(fd_, client_, subdevice_, &group_, channel::kResume);
        XASSERT(recovery == 0, "CHANNEL pause failed (%d), recovery failed (%d); hardware state unknown",
                result, recovery);
    }
    XASSERT(result == 0, "CHANNEL pause failed after bounded retries: %d; channels restored", result);
    paused_ = true;
}

void CudaQueueLv3Channel::Resume()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!paused_) return;
    int result = control_(fd_, client_, subdevice_, &group_, channel::kResume);
    XASSERT(result == 0, "CHANNEL resume failed: %d", result);
    paused_ = false;
}

void CudaQueueLv3Channel::Deactivate()
{
    // At Level 2 this backend uses the stronger, state-preserving Level 3 operation.
    Pause();
}

void CudaQueueLv3Channel::Interrupt() { Pause(); }
void CudaQueueLv3Channel::Restore(const CommandLog &) { Resume(); }
void CudaQueueLv3Channel::Reactivate(const CommandLog &) { Resume(); }

void CudaQueueLv3Channel::SynchronizeOnSuspend()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // The synchronous RM disable already stopped compute. Waiting for kernel
        // completion here would deadlock until another thread resumes the group.
        if (paused_) return;
    }
    CudaQueueLv1::Synchronize();
}

EXPORT_C_FUNC XResult CudaQueueGetChannelInfo(HwQueueHandle hwq, CudaChannelInfo *info)
{
    if (!info) return kXSchedErrorInvalidValue;
    auto queue = HwQueueManager::Get(hwq);
    if (!queue) return kXSchedErrorNotFound;
    auto channel = std::dynamic_pointer_cast<CudaQueueLv3Channel>(queue);
    if (!channel) return kXSchedErrorNotSupported;
    *info = {channel->GetContext(), 1, 1, 1, 0};
    return kXSchedSuccess;
}

void CudaQueueLv3Channel::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level >= kPreemptLevelBlock && level <= kPreemptLevelInterrupt,
            "unsupported CHANNEL level: %d", level);
    level_ = level;
}
