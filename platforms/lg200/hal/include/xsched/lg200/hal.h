#pragma once

#include <stdint.h>

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XSCHED_LG200_DEVICE_NAME_MAX 64

typedef struct
{
    XDeviceId device_id;
    uint32_t drm_card;
    // PCI_ID from device/uevent, or cardN when PCI_ID is unavailable.
    char pci_id[XSCHED_LG200_DEVICE_NAME_MAX];
    char vendor[XSCHED_LG200_DEVICE_NAME_MAX];
    char driver[XSCHED_LG200_DEVICE_NAME_MAX];
    uint64_t local_memory_bytes;
} Lg200DeviceInfo;

typedef struct
{
    uint32_t drm_card;
    double gpu_utilization;
    uint64_t local_memory_used_bytes;
    uint64_t local_memory_total_bytes;
} Lg200DeviceStats;

// Enumerate DRM cards whose device/uevent DRIVER value starts with loonggpu.
// Passing devices as NULL queries the required count. If the supplied array is
// too small, count is updated to the required size and
// kXSchedErrorInvalidValue is returned.
XResult Lg200DeviceEnumerate(Lg200DeviceInfo *devices, uint32_t *count);

// Read optional statistics exported by the loonggpu DRM sysfs node. Missing
// utilization or memory attributes are reported as unknown values, while a
// non-LG200 card returns kXSchedErrorNotFound.
XResult Lg200DeviceReadStats(uint32_t drm_card, Lg200DeviceStats *stats);

#ifdef __cplusplus
}
#endif
