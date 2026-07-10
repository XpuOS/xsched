import ctypes
from typing import Tuple

from xsched import XResult, HwQueueHandle

hipStream_t = int


class XSchedHip:
    __xres_ctype       = ctypes.c_int32
    __hwqh_ctype       = ctypes.c_uint64
    __hip_stream_ctype = ctypes.c_int64

    try:
        __dll = ctypes.cdll.LoadLibrary("libhalhip.so")

        __dll.HipQueueCreate.argtypes = [ctypes.POINTER(__hwqh_ctype), __hip_stream_ctype]
        __dll.HipQueueCreate.restype = __xres_ctype

    except Exception as e:
        raise RuntimeError(f"failed to load libhalhip.so bindings: {e}") from e

    @staticmethod
    def HipQueueCreate(stream: hipStream_t) -> Tuple[XResult, HwQueueHandle]:
        hwq = XSchedHip.__hwqh_ctype()
        res = XSchedHip.__dll.HipQueueCreate(ctypes.byref(hwq), stream)
        return XResult(res), hwq.value
