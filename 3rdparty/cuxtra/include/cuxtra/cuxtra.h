#pragma once

#include <cstddef>
#include <cstdint>

#define CUXTRA_CUDA_LIB_ENV_NAME       "CUXTRA_CUDA_LIB"
#define CUXTRA_CUDART_LIB_ENV_NAME     "CUXTRA_CUDART_LIB"
#define CUXTRA_CUDLA_LIB_ENV_NAME      "CUXTRA_CUDLA_LIB"

#ifdef _WIN32
    #ifdef __cplusplus
        #define CUXTRA_FUNC extern "C" __declspec(dllexport)
    #else
        #define CUXTRA_FUNC __declspec(dllexport)
    #endif
#else
    #ifdef __cplusplus
        #define CUXTRA_FUNC extern "C" __attribute__((visibility("default")))
    #else
        #define CUXTRA_FUNC __attribute__((visibility("default")))
    #endif
#endif

typedef int                 CUdevice;
typedef struct CUctx_st    *CUcontext;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long  CUdeviceptr;

typedef struct cudlaDevHandle_t *cudlaDevHandle;

// Memory operations
CUXTRA_FUNC size_t cuXtraMemcpyHtoD(CUdeviceptr dst, const void *src,
                                    const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraMemcpyDtoH(void *dst, const CUdeviceptr src,
                                    const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraMemcpyDtoD(CUdeviceptr dst, const CUdeviceptr src,
                                    const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraMemcpyFindHtoD(CUdeviceptr dst, const void *src,
                                        const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraMemcpyFindDtoH(void *dst, const CUdeviceptr src,
                                        const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraMemcpyFindDtoD(CUdeviceptr dst, const CUdeviceptr src,
                                        const size_t size, CUstream stream);

// Instruction memory operations
CUXTRA_FUNC CUdeviceptr cuXtraInstrMemBlockAlloc(CUcontext ctx, size_t size);
CUXTRA_FUNC void cuXtraInstrMemBlockFree(CUcontext ctx, CUdeviceptr base);
CUXTRA_FUNC size_t cuXtraInstrMemcpyHtoD(CUdeviceptr dst_instr, const void *src,
                                         const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraInstrMemcpyDtoH(void *dst, const CUdeviceptr src_instr,
                                         const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraInstrMemcpyDtoD(CUdeviceptr dst_instr,
                                         const CUdeviceptr src_instr,
                                         const size_t size, CUstream stream);
CUXTRA_FUNC size_t cuXtraInstrMemcpyHtoF(CUdeviceptr dst_instr, const void *src,
                                         const size_t size, CUstream stream,
                                         CUfunction dst_func);
CUXTRA_FUNC size_t cuXtraInstrMemcpyFtoH(void *dst, const CUdeviceptr src_instr,
                                         const size_t size, CUstream stream,
                                         CUfunction src_func);

// Kernel information
CUXTRA_FUNC void cuXtraGetFunctionName(CUcontext ctx, CUfunction func, const char **name);
CUXTRA_FUNC size_t cuXtraGetParamCount(CUfunction func);
CUXTRA_FUNC void cuXtraGetParamInfo(CUfunction func, size_t param_idx,
                                    size_t *offset, size_t *size, bool *in_shm);
CUXTRA_FUNC void cuXtraGetBinary(CUcontext ctx, CUfunction func,
                                 const void **binary, size_t *size, bool relocated);

// Kernel runtime control
CUXTRA_FUNC void cuXtraGetDebuggerParams(CUfunction func, void *params, size_t offset, size_t size);
CUXTRA_FUNC void cuXtraSetDebuggerParams(CUfunction func, const void *params, size_t size);
CUXTRA_FUNC CUdeviceptr cuXtraGetEntryPoint(CUfunction func);
CUXTRA_FUNC void cuXtraSetEntryPoint(CUfunction func, CUdeviceptr entry_point);
CUXTRA_FUNC size_t cuXtraGetLocalRegsPerThread(CUfunction func);
CUXTRA_FUNC void cuXtraSetLocalRegsPerThread(CUfunction func, size_t regs_cnt);
CUXTRA_FUNC size_t cuXtraGetBarrierCnt(CUfunction func);
CUXTRA_FUNC void cuXtraSetBarrierCnt(CUfunction func, size_t barrier_cnt);

// Cache and TLB operations
CUXTRA_FUNC void cuXtraInvalTLB(CUcontext ctx);
CUXTRA_FUNC void cuXtraInvalAllCache(CUcontext ctx);
CUXTRA_FUNC void cuXtraInvalInstrCache(CUcontext ctx);
CUXTRA_FUNC void cuXtraInvalConstCache(CUcontext ctx);
CUXTRA_FUNC void cuXtraInvalTextureCache(CUcontext ctx, bool wait_for_idle);
CUXTRA_FUNC void cuXtraFlushL2Cache(CUcontext ctx);
CUXTRA_FUNC void cuXtraFlushInvalL2Cache(CUcontext ctx);

// Trap handler operations
CUXTRA_FUNC void cuXtraGetTrapHandlerInfo(CUcontext ctx, CUdeviceptr *handler, size_t *size);
CUXTRA_FUNC void cuXtraTriggerTrap(CUcontext ctx);

// Timeslice group (TSG) operations
CUXTRA_FUNC size_t cuXtraGetTimeslice(CUcontext ctx);
CUXTRA_FUNC void cuXtraSetTimeslice(CUcontext ctx, size_t timeslice_us);

CUXTRA_FUNC void cuXtraSuspendContextJetson(CUcontext ctx);
CUXTRA_FUNC void cuXtraResumeContextJetson(CUcontext ctx);
CUXTRA_FUNC size_t cuXtraGetTimesliceJetson(CUcontext ctx);
CUXTRA_FUNC void cuXtraSetTimesliceJetson(CUcontext ctx, size_t timeslice_us);

// Set interleave level, low: 0, medium: 1, high: 2
CUXTRA_FUNC void cuXtraSetInterleaveLevelJetson(CUcontext ctx, int level);

// Register and create CUcontext for Jetson
CUXTRA_FUNC void cuXtraRegisterContextJetson(CUcontext ctx);
CUXTRA_FUNC CUcontext cuXtraCreateContextJetson(CUdevice dev, unsigned int flags);

// Dla queue operations
CUXTRA_FUNC void cuXtraSuspendDla(cudlaDevHandle dev_handle);
CUXTRA_FUNC void cuXtraResumeDla(cudlaDevHandle dev_handle);
CUXTRA_FUNC cudlaDevHandle cuXtraCreateDevHandleDla(uint64_t device, uint32_t flags);
