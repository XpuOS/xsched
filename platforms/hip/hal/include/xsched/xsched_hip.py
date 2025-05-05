from enum import IntEnum
import ctypes
from xsched import *

class XSchedHIP():
    try:
        __shim_dll = ctypes.cdll.LoadLibrary("libshimhip.so")
    except Exception as e:
        print(e)
        exit(1)

    @staticmethod
    def HIPQueueCreate(stream: ctypes.c_void_p):
        hwqueue = ctypes.c_uint64()
        res = XSchedHIP.__shim_dll.HipQueueCreate(ctypes.byref(hwqueue), stream)
        return XResult(res), hwqueue


