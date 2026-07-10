#pragma once

#include "xsched/protocol/def.h"
#include "xsched/utils/common.h"
#include "xsched/utils/symbol.h"
#include "xsched/utils/function.h"
#include "xsched/corex/hal/common/cuda.h"
#include "xsched/corex/hal/common/cudart.h"

namespace xsched::corex
{

class Driver
{
private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, XSCHED_COREX_LIB_ENV_NAME,
                           std::vector<std::string>({"libcuda.so.1", "libcuda.so"}),
                           std::vector<std::string>({"/usr/local/corex/lib", "/usr/local/corex/lib64"}));

public:
    STATIC_CLASS(Driver);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamGetCtx"), CUresult, StreamGetCtx, CUstream, hStream, CUcontext *,pctx);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuCtxGetCurrent"), CUresult, CtxGetCurrent, CUcontext *, pctx);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuCtxSetCurrent"), CUresult, CtxSetCurrent, CUcontext, ctx);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuKernelGetParamInfo"), CUresult, KernelGetParamInfo, CUkernel, kernel, size_t, paramIndex, size_t *, paramOffset, size_t *, paramSize);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamCreate"), CUresult, StreamCreate, CUstream *, phStream, unsigned int, Flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamCreateWithPriority"), CUresult, StreamCreateWithPriority, CUstream *, phStream, unsigned int, flags, int, priority);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamDestroy_v2"), CUresult, StreamDestroy_v2, CUstream, hStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamSynchronize"), CUresult, StreamSynchronize, CUstream, hStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamSynchronize_ptsz"), CUresult, StreamSynchronize_ptsz, CUstream, hStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamQuery"), CUresult, StreamQuery, CUstream, hStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamQuery_ptsz"), CUresult, StreamQuery_ptsz, CUstream, hStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamWaitEvent"), CUresult, StreamWaitEvent, CUstream, hStream, CUevent, hEvent, unsigned int, Flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cuStreamWaitEvent_ptsz"), CUresult, StreamWaitEvent_ptsz, CUstream, hStream, CUevent, hEvent, unsigned int, Flags);
};

class RtDriver
{
private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, XSCHED_COREX_RT_LIB_ENV_NAME,
                           std::vector<std::string>({"libcudart.so.10.2", "libcudart.so"}),
                           std::vector<std::string>({"/usr/local/corex/lib", "/usr/local/corex/lib64"}));

public:
    STATIC_CLASS(RtDriver);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaGetDevice"), cudaError_t, GetDevice, int *, device);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaSetDevice"), cudaError_t, SetDevice, int, device);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaGetDeviceProperties"), cudaError_t, GetDeviceProperties, cudaDeviceProp *, prop, int, device);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaFuncGetAttributes"), cudaError_t, FuncGetAttributes, cudaFuncAttributes *, attr, const void *, func);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaGetKernel"), cudaError_t, GetKernel, cudaKernel_t *, kernelPtr, const void *, entryFuncAddr);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaGetLastError"), cudaError_t, GetLastError);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaDeviceSynchronize"), cudaError_t, DeviceSynchronize);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaLaunchKernel"), cudaError_t, LaunchKernel, const void *, func, dim3, gridDim, dim3, blockDim, void **, args, size_t, sharedMem, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaLaunchKernel_ptsz"), cudaError_t, LaunchKernel_ptsz, const void *, func, dim3, gridDim, dim3, blockDim, void **, args, size_t, sharedMem, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaLaunchKernelExC"), cudaError_t, LaunchKernelExC, const cudaLaunchConfig_t *, config, const void *, func, void **, args);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpyAsync"), cudaError_t, MemcpyAsync, void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpyAsync_ptsz"), cudaError_t, MemcpyAsync_ptsz, void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpy2DAsync"), cudaError_t, Memcpy2DAsync, void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpy2DAsync_ptsz"), cudaError_t, Memcpy2DAsync_ptsz, void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpy3DAsync"), cudaError_t, Memcpy3DAsync, const cudaMemcpy3DParms *, p, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemcpy3DAsync_ptsz"), cudaError_t, Memcpy3DAsync_ptsz, const cudaMemcpy3DParms *, p, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemsetAsync"), cudaError_t, MemsetAsync, void *, devPtr, int, value, size_t, count, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMemsetAsync_ptsz"), cudaError_t, MemsetAsync_ptsz, void *, devPtr, int, value, size_t, count, cudaStream_t, stream);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaPointerGetAttributes"), cudaError_t, PointerGetAttributes, struct cudaPointerAttributes *, attributes, const void *, ptr);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaMallocHost"), cudaError_t, MallocHost, void **, ptr, size_t, size);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaFreeHost"), cudaError_t, FreeHost, void *, ptr);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaFree"), cudaError_t, Free, void *, devPtr);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamCreate"), cudaError_t, StreamCreate, cudaStream_t *, pStream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamCreateWithFlags"), cudaError_t, StreamCreateWithFlags, cudaStream_t *, pStream, unsigned int, flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamCreateWithPriority"), cudaError_t, StreamCreateWithPriority, cudaStream_t *, pStream, unsigned int, flags, int, priority);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamDestroy"), cudaError_t, StreamDestroy, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamQuery"), cudaError_t, StreamQuery, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamQuery_ptsz"), cudaError_t, StreamQuery_ptsz, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamSynchronize"), cudaError_t, StreamSynchronize, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamSynchronize_ptsz"), cudaError_t, StreamSynchronize_ptsz, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamWaitEvent"), cudaError_t, StreamWaitEvent, cudaStream_t, stream, cudaEvent_t, event, unsigned int, flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamWaitEvent_ptsz"), cudaError_t, StreamWaitEvent_ptsz, cudaStream_t, stream, cudaEvent_t, event, unsigned int, flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaStreamGetFlags"), cudaError_t, StreamGetFlags, cudaStream_t, stream, unsigned int *, flags);

    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventCreate"), cudaError_t, EventCreate, cudaEvent_t *, event);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventCreateWithFlags"), cudaError_t, EventCreateWithFlags, cudaEvent_t *, event, unsigned int, flags);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventRecord"), cudaError_t, EventRecord, cudaEvent_t, event, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventRecord_ptsz"), cudaError_t, EventRecord_ptsz, cudaEvent_t, event, cudaStream_t, stream);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventQuery"), cudaError_t, EventQuery, cudaEvent_t, event);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventSynchronize"), cudaError_t, EventSynchronize, cudaEvent_t, event);
    DEFINE_STATIC_ADDRESS_CALL(GetSymbol("cudaEventDestroy"), cudaError_t, EventDestroy, cudaEvent_t, event);

};

} // namespace xsched::corex
