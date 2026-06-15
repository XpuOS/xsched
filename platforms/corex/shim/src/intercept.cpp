#include "xsched/utils/function.h"
#include "xsched/corex/shim/shim.h"
#include "xsched/corex/hal/common/driver.h"

using namespace xsched::corex;

DEFINE_EXPORT_C_REDIRECT_CALL(XRtLaunchKernel, cudaError_t, cudaLaunchKernel, const void *, func, dim3, gridDim, dim3, blockDim, void **, args, size_t, sharedMem, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtLaunchKernel_ptsz, cudaError_t, cudaLaunchKernel_ptsz, const void *, func, dim3, gridDim, dim3, blockDim, void **, args, size_t, sharedMem, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtLaunchKernelExC, cudaError_t, cudaLaunchKernelExC, const cudaLaunchConfig_t *, config, const void *, func, void **, args);

DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpyAsync, cudaError_t, cudaMemcpyAsync, void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpyAsync_ptsz, cudaError_t, cudaMemcpyAsync_ptsz, void *, dst, const void *, src, size_t, count, cudaMemcpyKind, kind, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpy2DAsync, cudaError_t, cudaMemcpy2DAsync, void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpy2DAsync_ptsz, cudaError_t, cudaMemcpy2DAsync_ptsz, void *, dst, size_t, dpitch, const void *, src, size_t, spitch, size_t, width, size_t, height, cudaMemcpyKind, kind, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpy3DAsync, cudaError_t, cudaMemcpy3DAsync, const cudaMemcpy3DParms *, p, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemcpy3DAsync_ptsz, cudaError_t, cudaMemcpy3DAsync_ptsz, const cudaMemcpy3DParms *, p, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemsetAsync, cudaError_t, cudaMemsetAsync, void *, devPtr, int, value, size_t, count, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtMemsetAsync_ptsz, cudaError_t, cudaMemsetAsync_ptsz, void *, devPtr, int, value, size_t, count, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtFree, cudaError_t, cudaFree, void *, devPtr);

DEFINE_EXPORT_C_REDIRECT_CALL(XRtEventQuery, cudaError_t, cudaEventQuery, cudaEvent_t, event);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtEventRecord, cudaError_t, cudaEventRecord, cudaEvent_t, event, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtEventRecord_ptsz, cudaError_t, cudaEventRecord_ptsz, cudaEvent_t, event, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtEventSynchronize, cudaError_t, cudaEventSynchronize, cudaEvent_t, event);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamWaitEvent, cudaError_t, cudaStreamWaitEvent, cudaStream_t, stream, cudaEvent_t, event, unsigned int, flags);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamWaitEvent_ptsz, cudaError_t, cudaStreamWaitEvent_ptsz, cudaStream_t, stream, cudaEvent_t, event, unsigned int, flags);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtEventDestroy, cudaError_t, cudaEventDestroy, cudaEvent_t, event);

DEFINE_EXPORT_C_REDIRECT_CALL(XRtDeviceSynchronize, cudaError_t, cudaDeviceSynchronize);

DEFINE_EXPORT_C_REDIRECT_CALL(XStreamSynchronize, CUresult, cuStreamSynchronize, CUstream, hStream);
DEFINE_EXPORT_C_REDIRECT_CALL(XStreamSynchronize_ptsz, CUresult, cuStreamSynchronize_ptsz, CUstream, hStream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamSynchronize, cudaError_t, cudaStreamSynchronize, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamSynchronize_ptsz, cudaError_t, cudaStreamSynchronize_ptsz, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamQuery, cudaError_t, cudaStreamQuery, cudaStream_t, stream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamQuery_ptsz, cudaError_t, cudaStreamQuery_ptsz, cudaStream_t, stream);

DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamCreate, cudaError_t, cudaStreamCreate, cudaStream_t *, pStream);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamCreateWithFlags, cudaError_t, cudaStreamCreateWithFlags, cudaStream_t *, pStream, unsigned int, flags);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamCreateWithPriority, cudaError_t, cudaStreamCreateWithPriority, cudaStream_t *, pStream, unsigned int, flags, int, priority);
DEFINE_EXPORT_C_REDIRECT_CALL(XRtStreamDestroy, cudaError_t, cudaStreamDestroy, cudaStream_t, stream);

DEFINE_EXPORT_C_REDIRECT_CALL(XStreamCreate, CUresult, cuStreamCreate, CUstream *, phStream, unsigned int, Flags);
DEFINE_EXPORT_C_REDIRECT_CALL(XStreamCreateWithPriority, CUresult, cuStreamCreateWithPriority, CUstream *, phStream, unsigned int, flags, int, priority);
DEFINE_EXPORT_C_REDIRECT_CALL(XStreamDestroy_v2, CUresult, cuStreamDestroy_v2, CUstream, hStream);
