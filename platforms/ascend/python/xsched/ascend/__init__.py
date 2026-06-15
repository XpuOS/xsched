import ctypes
from typing import Tuple

from xsched import HwQueueHandle, XResult

AclrtStream = int


class XSchedAscend:
    __xres_ctype = ctypes.c_int32
    __hwqh_ctype = ctypes.c_uint64
    __stream_ctype = ctypes.c_void_p

    try:
        __dll = ctypes.cdll.LoadLibrary("libhalascend.so")

        __dll.AclQueueCreate.argtypes = [
            ctypes.POINTER(__hwqh_ctype),
            __stream_ctype,
        ]
        __dll.AclQueueCreate.restype = __xres_ctype

    except Exception as e:
        raise RuntimeError(f"failed to load libhalascend.so bindings: {e}") from e

    @staticmethod
    def AclQueueCreate(stream: AclrtStream) -> Tuple[XResult, HwQueueHandle]:
        hwq = XSchedAscend.__hwqh_ctype()
        res = XSchedAscend.__dll.AclQueueCreate(
            ctypes.byref(hwq),
            ctypes.c_void_p(stream),
        )
        return XResult(res), hwq.value
