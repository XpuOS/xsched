// Compile the real tracker with a syscall substitute; no GPU or fault injection
// environment variables are exposed by the production shared library.
#define XSCHED_CHANNEL_TRACKER_TEST
#define ioctl TrackerIoctl
#include "../../tools/channel_tracker.cpp"
#undef ioctl
#include <cstdlib>
#include <deque>
#include <set>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace
{
std::deque<int> errors;
std::set<unsigned> disabled;
unsigned controls = 0;
unsigned allocated_object = 0;
int allocation_error = 0;
int TestIoctl(int, unsigned long req, void *arg)
{
    if (req == _IOWR('F', 0x2b, Alloc64)) {
        auto &p = *static_cast<Alloc64 *>(arg);
        if (allocation_error < 0) { errno = EIO; return -1; }
        p.status = allocation_error;
        if (!allocation_error && allocated_object) p.object = allocated_object;
        return 0;
    }
    if (req != _IOWR('F', 0x2a, RmControl)) return 0;
    ++controls;
    auto &c = *static_cast<RmControl *>(arg);
    auto &p = *reinterpret_cast<Disable *>(c.params);
    CHECK(c.cmd == 0x2080110b && !p.only_scheduling && !p.rewind && !p.event);
    int error = 0;
    if (!errors.empty()) { error = errors.front(); errors.pop_front(); }
    // A failing request can already have changed its first channel.
    unsigned count = error ? 1 : p.count;
    for (unsigned i = 0; i < count; ++i)
        if (p.disable) disabled.insert(p.channels[i]);
        else disabled.erase(p.channels[i]);
    if (error < 0) { errno = EIO; return -1; }
    c.status = error;
    return 0;
}

constexpr unsigned Client = 1, Group = 10, Subdevice = 3;
int Call(unsigned *handle, int op)
{
    return xsched_cuda_channel_control_v2(7, Client, Subdevice, handle, op);
}
void Allocate(unsigned parent, unsigned object, unsigned cls)
{
    Alloc21 p{Client, parent, object, cls, 0, 0, 0};
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc21), &p) == 0);
}
void Setup()
{
    auto &s = GetState();
    s.objects.clear(); s.owners.clear();
    errors.clear(); disabled.clear(); controls = 0;
    allocated_object = 0; allocation_error = 0;
    Allocate(0, Client, 0);
    Allocate(Client, Subdevice, 0x2080);
    Allocate(Client, Group, 0xa06c);
    Allocate(Group, 11, 0xc96f);
    Allocate(Group, 12, 0xc96f);
}
}

int main()
{
    Setup();
    // Use the driver's returned handle, and never record failed allocations.
    Alloc64 allocation{Client, Group, 0, 0xc96f, 0, 0, 0, 0, 0};
    allocated_object = 13;
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == 0);
    CHECK(allocation.object == 13);
    CHECK(GetState().objects.at({Client, 13}).parent == Group);
    CHECK(GetState().objects.at({Client, 13}).cls == 0xc96f);
    unsigned allocated_handle = 13;
    CHECK(Call(&allocated_handle, 0) == 0 && allocated_handle == Group);
    CHECK(Call(&allocated_handle, 1) == 0 && disabled.size() == 3);
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == -1 && errno == EBUSY);
    CHECK(Call(&allocated_handle, 2) == 0 && disabled.empty());
    CHECK(Call(&allocated_handle, 3) == 0);
    allocation.object = 14;
    allocated_object = 14;
    allocation_error = 0x40;
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == 0);
    CHECK(allocation.status == 0x40 && !GetState().objects.count({Client, 14}));
    allocation.status = 0;
    allocation_error = -1;
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == -1);
    CHECK(errno == EIO && !GetState().objects.count({Client, 14}));

    Setup();
    CHECK(Call(nullptr, 0) == EINVAL);
    unsigned handle = 11;
    CHECK(Call(&handle, 9) == EINVAL);
    CHECK(Call(&handle, 0) == 0 && handle == Group);
    unsigned duplicate = 12;
    CHECK(Call(&duplicate, 0) == EBUSY && duplicate == 12);
    CHECK(xsched_cuda_channel_control_v2(8, Client, Subdevice, &handle, 1) == EINVAL);
    CHECK(xsched_cuda_channel_control_v2(7, Client, 99, &handle, 1) == EINVAL);
    CHECK(controls == 0);
    Free parent{Client, 0, Client, 0};
    CHECK(TrackerIoctl(7, _IOWR('F', 0x29, Free), &parent) == -1 && errno == EBUSY);
    CHECK(Call(&handle, 1) == 0 && disabled.size() == 2);
    CHECK(Call(&handle, 1) == 0 && controls == 1);
    CHECK(Call(&handle, 3) == EBUSY);
    Alloc21 escaped{Client, Group, 13, 0xc96f, 0, 0, 0};
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc21), &escaped) == -1 && errno == EBUSY);
    Free child{Client, Group, 11, 0};
    CHECK(TrackerIoctl(7, _IOWR('F', 0x29, Free), &child) == -1 && errno == EBUSY);
    CHECK(Call(&handle, 2) == 0 && disabled.empty());
    CHECK(Call(&handle, 2) == 0 && controls == 2);

    // Partial disable with successful rollback must remain an error.
    errors = {-1, 0};
    CHECK(Call(&handle, 1) == EIO && disabled.empty());
    CHECK(!GetState().owners.at({Client, Group}).uncertain);
    CHECK(Call(&handle, 1) == 0 && disabled.size() == 2);
    // Transient enable failures are retried with the same snapshot.
    errors = {0x40, -1, 0};
    CHECK(Call(&handle, 2) == 0 && disabled.empty());

    errors = {0x40, 0x40, -1, 0x40};
    CHECK(Call(&handle, 1) == 0x40);
    CHECK(GetState().owners.at({Client, Group}).uncertain);
    CHECK(Call(&handle, 3) == EBUSY);
    unsigned before = controls;
    CHECK(Call(&handle, 1) == EIO && controls == before);
    CHECK(TrackerIoctl(7, _IOWR('F', 0x2b, Alloc21), &escaped) == -1 && errno == EBUSY);
    CHECK(Call(&handle, 2) == 0 && disabled.empty());
    CHECK(!GetState().owners.at({Client, Group}).uncertain);
    CHECK(Call(&handle, 1) == 0);
    errors = {-1, -1, -1};
    CHECK(Call(&handle, 2) == EIO);
    CHECK(GetState().owners.at({Client, Group}).uncertain);
    CHECK(Call(&handle, 3) == EBUSY);
    CHECK(Call(&handle, 2) == 0 && disabled.empty());
    CHECK(Call(&handle, 3) == 0);
    CHECK(TrackerIoctl(7, _IOWR('F', 0x29, Free), &parent) == 0);
    CHECK(GetState().objects.empty());
    handle = 11;
    CHECK(Call(&handle, 0) == ENOENT);

    Setup();
    for (unsigned i = 13; i < 76; ++i) Allocate(Group, i, 0xc96f);
    handle = 11;
    CHECK(Call(&handle, 0) == 0);
    CHECK(Call(&handle, 1) == EOVERFLOW && controls == 0);
    CHECK(Call(&handle, 3) == 0);
    puts("PASS: allocation ABIs, ownership, binding, lifecycle, partial failure, rollback, recovery, overflow");
}
