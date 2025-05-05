#include "xsched/protocol/device.h"

namespace xsched::protocol
{

XDeviceId GetDeviceId(XDevice dev)
{
    return (uint32_t)dev & 0xFFFFFFFF;
}

XDeviceType GetDeviceType(XDevice dev)
{
    return (XDeviceType)(dev >> 32);
}

XDevice MakeDevice(XDeviceType type, XDeviceId id)
{
    return ((uint64_t)type << 32) | (uint64_t)id;
}

} // namespace xsched::protocol
