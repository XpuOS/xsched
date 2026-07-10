import ctypes
from typing import Tuple

from xsched import XResult, HwQueueHandle

CUstream = int


class XSchedCuda:
    __xres_ctype      = ctypes.c_int32
    __hwqh_ctype      = ctypes.c_uint64
    __cu_stream_ctype = ctypes.c_int64

    try:
        __dll = ctypes.cdll.LoadLibrary("libhalcuda.so")

        __dll.CudaQueueCreate.argtypes = [ctypes.POINTER(__hwqh_ctype), __cu_stream_ctype]
        __dll.CudaQueueCreate.restype = __xres_ctype

        __dll.CudaQueueGet.argtypes = [ctypes.POINTER(__hwqh_ctype), __cu_stream_ctype]
        __dll.CudaQueueGet.restype = __xres_ctype

    except Exception as e:
        raise RuntimeError(f"failed to load libhalcuda.so bindings: {e}") from e

    @staticmethod
    def CudaQueueCreate(stream: CUstream) -> Tuple[XResult, HwQueueHandle]:
        hwq = XSchedCuda.__hwqh_ctype()
        res = XSchedCuda.__dll.CudaQueueCreate(ctypes.byref(hwq), stream)
        return XResult(res), hwq.value

    @staticmethod
    def CudaQueueGet(stream: CUstream) -> Tuple[XResult, HwQueueHandle]:
        hwq = XSchedCuda.__hwqh_ctype()
        res = XSchedCuda.__dll.CudaQueueGet(ctypes.byref(hwq), stream)
        return XResult(res), hwq.value
