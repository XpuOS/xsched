#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t XDevice;
typedef uint32_t XDeviceId; // recommend to use PCI address (32-bit) for XDeviceId
typedef uint64_t XQueueHandle;
typedef uint64_t HwQueueHandle;
typedef uint64_t HwCommandHandle;

typedef enum {
    kXSchedSuccess           = 0,
    kXSchedErrorHardware     = 1,
    kXSchedErrorInvalidValue = 2,
    kXSchedErrorNotFound     = 3,
    kXSchedErrorNotAllowed   = 4,
    kXSchedErrorNotSupported = 5,
    kXSchedErrorUnknown      = 999,
} XResult;

typedef enum {
    kPlatformUnknown   = 0,
    kPlatformVPI       = 1,
    kPlatformCUDA      = 2,
    kPlatformCUDLA     = 3,
    kPlatformHIP       = 4,
    kPlatformAscend    = 5,
    kPlatformOpenCL    = 6,
    kPlatformLevelZero = 7,
} XPlatform;

typedef enum {
    kDeviceTypeUnknown = 0,
    kDeviceTypeCPU     = 1,
    kDeviceTypeGPU     = 2,
    kDeviceTypeNPU     = 3,
    kDeviceTypeFPGA    = 4,
    kDeviceTypeASIC    = 5,
    kDeviceTypeMCA     = 6, // Memory Copy Accelerator
} XDeviceType;

typedef enum {
    kPreemptLevelUnknown    = 0,
    kPreemptLevelBlock      = 1,
    kPreemptLevelDeactivate = 2,
    kPreemptLevelInterrupt  = 3,
    kPreemptLevelMax,
} XPreemptLevel;

typedef enum {
    kQueueStateUnknown = 0,
    kQueueStateIdle    = 1,
    kQueueStateReady   = 2,
} XQueueState;

typedef enum {
    kQueueFeatureNone               = 0x0000,
    kQueueFeatureAsyncSubmit        = 0x0001,
    kQueueFeatureDynamicLevel       = 0x0002,
    kQueueFeatureDynamicThreshold   = 0x0004,
    kQueueFeatureDynamicBatchSize   = 0x0008,
    kQueueFeatureSyncSuspend        = 0x0010,
    kQueueFeatureResumeDropCommands = 0x0020,
    kQueueFeatureMaskAll            = -1,
} XQueueFeature;

typedef enum {
    kQueueCreateFlagNone           = 0x0000,
    kQueueCreateFlagBlockingSubmit = 0x0001,
    kQueueCreateFlagMaskAll        = -1,
} XQueueCreateFlag;

typedef enum {
    kQueueSuspendFlagNone        = 0x0000,
    kQueueSuspendFlagSyncHwQueue = 0x0001,
    kQueueSuspendFlagMaskAll     = -1,
} XQueueSuspendFlag;

typedef enum {
    kQueueResumeFlagNone         = 0x0000,
    kQueueResumeFlagDropCommands = 0x0001,
    kQueueResumeFlagMaskAll      = -1,
} XQueueResumeFlag;

typedef XResult (*LaunchCallback)(HwQueueHandle, void *);

/// @brief Priority is a signed integer, with higher values indicating higher priority.
typedef int32_t Priority;
#define PRIORITY_NO_EXECUTE -256
#define PRIORITY_MIN        -255
#define PRIORITY_DEFAULT     000
#define PRIORITY_MAX         255

/// @brief Utilization is a percentage,
/// with 0 indicating no utilization and 100 indicating full utilization.
typedef int32_t Utilization;
#define UTILIZATION_MIN      0
#define UTILIZATION_DEFAULT  100
#define UTILIZATION_MAX      100

/// @brief Timeslice is a positive integer, indicating the number of microseconds.
typedef int64_t Timeslice;
#define TIMESLICE_MIN        100    // 0.1 ms
#define TIMESLICE_DEFAULT    5000   // 5 ms
#define TIMESLICE_MAX        100000 // 100 ms

/// @brief Laxity is a positive integer, indicating the number of microseconds.
typedef int64_t Laxity;
#define NO_LAXITY           -1

/// @brief Deadline is a positive integer, indicating the number of microseconds.
typedef int64_t Deadline;
#define NO_DEADLINE         -1

#ifdef __cplusplus
}
#endif
