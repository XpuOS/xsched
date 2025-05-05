import ctypes
from enum import IntEnum
from typing import Tuple

# Type definitions
XDevice         = int
XDeviceId       = int
XQueueHandle    = int
HwQueueHandle   = int
HwCommandHandle = int


# Enums
class XResult(IntEnum):
    kXSchedSuccess           = 0
    kXSchedErrorHardware     = 1
    kXSchedErrorInvalidValue = 2
    kXSchedErrorNotFound     = 3
    kXSchedErrorNotAllowed   = 4
    kXSchedErrorNotSupported = 5
    kXSchedErrorUnknown      = 999


class XPlatform(IntEnum):
    kPlatformUnknown   = 0
    kPlatformVPI       = 1
    kPlatformCUDA      = 2
    kPlatformCUDLA     = 3
    kPlatformHIP       = 4
    kPlatformAscend    = 5
    kPlatformOpenCL    = 6
    kPlatformLevelZero = 7


class XDeviceType(IntEnum):
    kDeviceTypeUnknown = 0
    kDeviceTypeCPU     = 1
    kDeviceTypeGPU     = 2
    kDeviceTypeNPU     = 3
    kDeviceTypeFPGA    = 4
    kDeviceTypeASIC    = 5
    kDeviceTypeMCA     = 6


class XPreemptLevel(IntEnum):
    kPreemptLevelUnknown    = 0
    kPreemptLevelBlock      = 1
    kPreemptLevelDeactivate = 2
    kPreemptLevelInterrupt  = 3
    kPreemptLevelMax        = 4


class XQueueState(IntEnum):
    kQueueStateUnknown = 0
    kQueueStateIdle    = 1
    kQueueStateReady   = 2


class XQueueFeature(IntEnum):
    kQueueFeatureNone               = 0x0000
    kQueueFeatureAsyncSubmit        = 0x0001
    kQueueFeatureDynamicLevel       = 0x0002
    kQueueFeatureDynamicThreshold   = 0x0004
    kQueueFeatureDynamicBatchSize   = 0x0008
    kQueueFeatureSyncSuspend        = 0x0010
    kQueueFeatureResumeDropCommands = 0x0020
    kQueueFeatureMaskAll            = -1


class XQueueCreateFlag(IntEnum):
    kQueueCreateFlagNone           = 0x0000
    kQueueCreateFlagBlockingSubmit = 0x0001
    kQueueCreateFlagMaskAll        = -1


class XQueueSuspendFlag(IntEnum):
    kQueueSuspendFlagNone        = 0x0000
    kQueueSuspendFlagSyncHwQueue = 0x0001
    kQueueSuspendFlagMaskAll     = -1


class XQueueResumeFlag(IntEnum):
    kQueueResumeFlagNone         = 0x0000
    kQueueResumeFlagDropCommands = 0x0001
    kQueueResumeFlagMaskAll      = -1


# Constants
Priority            = int
PRIORITY_NO_EXECUTE = -256
PRIORITY_MIN        = -255
PRIORITY_DEFAULT    = 0
PRIORITY_MAX        = 255

Utilization         = int
UTILIZATION_MIN     = 0
UTILIZATION_DEFAULT = 100
UTILIZATION_MAX     = 100

Timeslice           = int
TIMESLICE_MIN       = 100    # 0.1 ms
TIMESLICE_DEFAULT   = 5000   # 5 ms
TIMESLICE_MAX       = 100000 # 100 ms

Laxity              = int
NO_LAXITY           = -1

Deadline            = int
NO_DEADLINE         = -1


class XSched:
    __xres_ctype   = ctypes.c_int32
    __xqh_ctype    = ctypes.c_uint64
    __xqs_ctype    = ctypes.c_int32
    __hwqh_ctype   = ctypes.c_uint64
    __hwcmdh_ctype = ctypes.c_uint64
    __prio_ctype   = ctypes.c_int32
    __util_ctype   = ctypes.c_int32
    __ts_ctype     = ctypes.c_int64
    __lax_ctype    = ctypes.c_int64
    __ddl_ctype    = ctypes.c_int64

    try:
        __dll = ctypes.cdll.LoadLibrary("libpreempt.so")

        # XQueue functions
        __dll.XQueueCreate.argtypes = [ctypes.POINTER(__xqh_ctype), __hwqh_ctype, ctypes.c_int64, ctypes.c_int64]
        __dll.XQueueCreate.restype = __xres_ctype

        __dll.XQueueDestroy.argtypes = [__xqh_ctype]
        __dll.XQueueDestroy.restype = __xres_ctype

        __dll.XQueueSetPreemptLevel.argtypes = [__xqh_ctype, ctypes.c_int64]
        __dll.XQueueSetPreemptLevel.restype = __xres_ctype

        __dll.XQueueSetLaunchConfig.argtypes = [__xqh_ctype, ctypes.c_int64, ctypes.c_int64]
        __dll.XQueueSetLaunchConfig.restype = __xres_ctype

        __dll.XQueueSubmit.argtypes = [__xqh_ctype, __hwcmdh_ctype]
        __dll.XQueueSubmit.restype = __xres_ctype

        __dll.XQueueWait.argtypes = [__xqh_ctype, __hwcmdh_ctype]
        __dll.XQueueWait.restype = __xres_ctype

        __dll.XQueueWaitAll.argtypes = [__xqh_ctype]
        __dll.XQueueWaitAll.restype = __xres_ctype

        __dll.XQueueQuery.argtypes = [__xqh_ctype, ctypes.POINTER(__xqs_ctype)]
        __dll.XQueueQuery.restype = __xres_ctype

        __dll.XQueueSuspend.argtypes = [__xqh_ctype, ctypes.c_int64]
        __dll.XQueueSuspend.restype = __xres_ctype

        __dll.XQueueResume.argtypes = [__xqh_ctype, ctypes.c_int64]
        __dll.XQueueResume.restype = __xres_ctype

        __dll.XQueueProfileHwCommandCount.argtypes = [__xqh_ctype, ctypes.POINTER(ctypes.c_int64)]
        __dll.XQueueProfileHwCommandCount.restype = __xres_ctype

        # HwQueue functions
        __dll.HwQueueDestroy.argtypes = [__hwqh_ctype]
        __dll.HwQueueDestroy.restype = __xres_ctype

        __dll.HwQueueLaunch.argtypes = [__hwqh_ctype, __hwcmdh_ctype]
        __dll.HwQueueLaunch.restype = __xres_ctype

        __dll.HwQueueSynchronize.argtypes = [__hwqh_ctype]
        __dll.HwQueueSynchronize.restype = __xres_ctype

        __dll.HwCommandDestroy.argtypes = [__hwcmdh_ctype]
        __dll.HwCommandDestroy.restype = __xres_ctype

        # Hint functions
        __dll.XHintPriority.argtypes = [__xqh_ctype, __prio_ctype]
        __dll.XHintPriority.restype = __xres_ctype

        __dll.XHintUtilization.argtypes = [__xqh_ctype, __util_ctype]
        __dll.XHintUtilization.restype = __xres_ctype

        __dll.XHintTimeslice.argtypes = [__ts_ctype]
        __dll.XHintTimeslice.restype = __xres_ctype

        __dll.XHintLaxity.argtypes = [__xqh_ctype, __lax_ctype, __prio_ctype, __prio_ctype]
        __dll.XHintLaxity.restype = __xres_ctype

        __dll.XHintDeadline.argtypes = [__xqh_ctype, __ddl_ctype]
        __dll.XHintDeadline.restype = __xres_ctype

    except Exception as e:
        print(e)
        exit(1)

    @staticmethod
    def XQueueCreate(hwq: HwQueueHandle, level: XPreemptLevel, flags: int) -> Tuple[XResult, XQueueHandle]:
        xq = XSched.__xqh_ctype()
        res = XSched.__dll.XQueueCreate(ctypes.byref(xq), hwq, level, flags)
        return XResult(res), xq.value

    @staticmethod
    def XQueueDestroy(xq: XQueueHandle) -> XResult:
        res = XSched.__dll.XQueueDestroy(xq)
        return XResult(res)

    @staticmethod
    def XQueueSetPreemptLevel(xq: XQueueHandle, level: int) -> XResult:
        res = XSched.__dll.XQueueSetPreemptLevel(xq, level)
        return XResult(res)

    @staticmethod
    def XQueueSetLaunchConfig(xq: XQueueHandle, threshold: int, batch_size: int) -> XResult:
        res = XSched.__dll.XQueueSetLaunchConfig(xq, threshold, batch_size)
        return XResult(res)

    @staticmethod
    def XQueueSubmit(xq: XQueueHandle, hw_cmd: HwCommandHandle) -> XResult:
        res = XSched.__dll.XQueueSubmit(xq, hw_cmd)
        return XResult(res)

    @staticmethod
    def XQueueWait(xq: XQueueHandle, hw_cmd: HwCommandHandle) -> XResult:
        res = XSched.__dll.XQueueWait(xq, hw_cmd)
        return XResult(res)

    @staticmethod
    def XQueueWaitAll(xq: XQueueHandle) -> XResult:
        res = XSched.__dll.XQueueWaitAll(xq)
        return XResult(res)

    @staticmethod
    def XQueueQuery(xq: XQueueHandle, state: int) -> Tuple[XResult, XQueueState]:
        state = XSched.__xqs_ctype()
        res = XSched.__dll.XQueueQuery(xq, ctypes.byref(state))
        return XResult(res), XQueueState(state.value)

    @staticmethod
    def XQueueSuspend(xq: XQueueHandle, flags: int) -> XResult:
        res = XSched.__dll.XQueueSuspend(xq, flags)
        return XResult(res)

    @staticmethod
    def XQueueResume(xq: XQueueHandle, flags: int) -> XResult:
        res = XSched.__dll.XQueueResume(xq, flags)
        return XResult(res)

    @staticmethod
    def XQueueProfileHwCommandCount(xq: XQueueHandle) -> Tuple[XResult, int]:
        count = ctypes.c_int64()
        res = XSched.__dll.XQueueProfileHwCommandCount(xq, ctypes.byref(count))
        return XResult(res), count.value
    
    @staticmethod
    def HwQueueDestroy(hwq: HwQueueHandle) -> XResult:
        res = XSched.__dll.HwQueueDestroy(hwq)
        return XResult(res)

    @staticmethod
    def HwQueueLaunch(hwq: HwQueueHandle, hw_cmd: HwCommandHandle) -> XResult:
        res = XSched.__dll.HwQueueLaunch(hwq, hw_cmd)
        return XResult(res)

    @staticmethod
    def HwQueueSynchronize(hwq: HwQueueHandle) -> XResult:
        res = XSched.__dll.HwQueueSynchronize(hwq)
        return XResult(res)
    
    @staticmethod
    def HwCommandDestroy(hw_cmd: HwCommandHandle) -> XResult:
        res = XSched.__dll.HwCommandDestroy(hw_cmd)
        return XResult(res)

    @staticmethod
    def XHintPriority(xq: XQueueHandle, prio: Priority) -> XResult:
        res = XSched.__dll.XHintPriority(xq, prio)
        return XResult(res)

    @staticmethod
    def XHintUtilization(xq: XQueueHandle, util: Utilization) -> XResult:
        res = XSched.__dll.XHintUtilization(xq, util)
        return XResult(res)

    @staticmethod
    def XHintTimeslice(ts_us: Timeslice) -> XResult:
        res = XSched.__dll.XHintTimeslice(ts_us)
        return XResult(res)

    @staticmethod
    def XHintLaxity(xq: XQueueHandle, lax_us: Laxity, lax_prio: Priority, crit_prio: Priority) -> XResult:
        res = XSched.__dll.XHintLaxity(xq, lax_us, lax_prio, crit_prio)
        return XResult(res)

    @staticmethod
    def XHintDeadline(xq: XQueueHandle, ddl_us: Deadline) -> XResult:
        res = XSched.__dll.XHintDeadline(xq, ddl_us)
        return XResult(res)
