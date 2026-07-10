/*
 * Copyright 2010-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_H_)
#define _CUPTI_H_

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef NOMINMAX
#include <windows.h>
#else
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif
#endif

#include "xsched/cuda/hal/common/cuda.h"

/*
 * Copyright 2010-2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_RESULT_H_)
#define _CUPTI_RESULT_H_

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_RESULT_API CUPTI Result Codes
 * Error and result codes returned by CUPTI functions.
 * @{
 */

/**
 * \brief CUPTI result codes.
 *
 * Error and result codes returned by CUPTI functions.
 */
typedef enum {
    /**
     * No error.
     */
    CUPTI_SUCCESS                                       = 0,
    /**
     * One or more of the parameters is invalid.
     */
    CUPTI_ERROR_INVALID_PARAMETER                       = 1,
    /**
     * The device does not correspond to a valid CUDA device.
     */
    CUPTI_ERROR_INVALID_DEVICE                          = 2,
    /**
     * The context is NULL or not valid.
     */
    CUPTI_ERROR_INVALID_CONTEXT                         = 3,
    /**
     * The event domain id is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID                 = 4,
    /**
     * The event id is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_ID                        = 5,
    /**
     * The event name is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_NAME                      = 6,
    /**
     * The current operation cannot be performed due to dependency on
     * other factors.
     */
    CUPTI_ERROR_INVALID_OPERATION                       = 7,
    /**
     * Unable to allocate enough memory to perform the requested
     * operation.
     */
    CUPTI_ERROR_OUT_OF_MEMORY                           = 8,
    /**
     * An error occurred on the performance monitoring hardware.
     */
    CUPTI_ERROR_HARDWARE                                = 9,
    /**
     * The output buffer size is not sufficient to return all
     * requested data.
     */
    CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 10,
    /**
     * API is not implemented.
     */
    CUPTI_ERROR_API_NOT_IMPLEMENTED                     = 11,
    /**
     * The maximum limit is reached.
     */
    CUPTI_ERROR_MAX_LIMIT_REACHED                       = 12,
    /**
     * The object is not yet ready to perform the requested operation.
     */
    CUPTI_ERROR_NOT_READY                               = 13,
    /**
     * The current operation is not compatible with the current state
     * of the object
     */
    CUPTI_ERROR_NOT_COMPATIBLE                          = 14,
    /**
     * CUPTI is unable to initialize its connection to the CUDA
     * driver.
     */
    CUPTI_ERROR_NOT_INITIALIZED                         = 15,
    /**
     * The metric id is invalid.
     */
    CUPTI_ERROR_INVALID_METRIC_ID                        = 16,
    /**
     * The metric name is invalid.
     */
    CUPTI_ERROR_INVALID_METRIC_NAME                      = 17,
    /**
     * The queue is empty.
     */
    CUPTI_ERROR_QUEUE_EMPTY                              = 18,
    /**
     * Invalid handle (internal?).
     */
    CUPTI_ERROR_INVALID_HANDLE                           = 19,
    /**
     * Invalid stream.
     */
    CUPTI_ERROR_INVALID_STREAM                           = 20,
    /**
     * Invalid kind.
     */
    CUPTI_ERROR_INVALID_KIND                             = 21,
    /**
     * Invalid event value.
     */
    CUPTI_ERROR_INVALID_EVENT_VALUE                      = 22,
    /**
     * CUPTI is disabled due to conflicts with other enabled profilers
     */
    CUPTI_ERROR_DISABLED                                 = 23,
    /**
     * Invalid module.
     */
    CUPTI_ERROR_INVALID_MODULE                           = 24,
    /**
     * Invalid metric value.
     */
    CUPTI_ERROR_INVALID_METRIC_VALUE                     = 25,
    /**
     * The performance monitoring hardware is in use by other client.
     */
    CUPTI_ERROR_HARDWARE_BUSY                            = 26,
    /**
     * The attempted operation is not supported on the current
     * system or device.
     */
    CUPTI_ERROR_NOT_SUPPORTED                            = 27,
    /**
     * Unified memory profiling is not supported on the system.
     * Potential reason could be unsupported OS or architecture.
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED               = 28,
    /**
     * Unified memory profiling is not supported on the device
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE     = 29,
    /**
     * Unified memory profiling is not supported on a multi-GPU
     * configuration without P2P support between any pair of devices
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30,
    /**
     * Unified memory profiling is not supported under the
     * Multi-Process Service (MPS) environment. CUDA 7.5 removes this
     * restriction.
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS      = 31,
    /**
     * In CUDA 9.0, devices with compute capability 7.0 don't
     * support CDP tracing
     */
    CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED                = 32,
    /**
     * Profiling on virtualized GPU is not supported.
     */
    CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED         = 33,
    /**
     * Profiling results might be incorrect for CUDA applications
     * compiled with nvcc version older than 9.0 for devices with
     * compute capability 6.0 and 6.1.
     * Profiling session will continue and CUPTI will notify it using this error code.
     * User is advised to recompile the application code with nvcc version 9.0 or later.
     * Ignore this warning if code is already compiled with the recommended nvcc version.
     */
    CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE             = 34,
    /**
     * User doesn't have sufficient privileges which are required to
     * start the profiling session.
     * One possible reason for this may be that the NVIDIA driver or your system
     * administrator may have restricted access to the NVIDIA GPU performance counters.
     * To learn how to resolve this issue and find more information, please visit
     * https://developer.nvidia.com/CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
     */
    CUPTI_ERROR_INSUFFICIENT_PRIVILEGES                  = 35,
    /**
     * Legacy CUPTI Profiling API i.e. event API from the header cupti_events.h and
     * metric API from the header cupti_metrics.h are not compatible with the
     * Profiling API in the header cupti_profiler_target.h and Perfworks metrics API
     * in the headers nvperf_host.h and nvperf_target.h.
     */
    CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED             = 36,
    /**
     * Missing definition of the OpenACC API routine in the linked OpenACC library.
     *
     * One possible reason is that OpenACC library is linked statically in the
     * user application, which might not have the definition of all the OpenACC
     * API routines needed for the OpenACC profiling, as compiler might ignore
     * definitions for the functions not used in the application. This issue
     * can be mitigated by linking the OpenACC library dynamically.
     */
    CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE                = 37,
    /**
     * Legacy CUPTI Profiling API i.e. event API from the header cupti_events.h and
     * metric API from the header cupti_metrics.h are not supported on devices with
     * compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
     * These APIs are deprecated in the CUDA 12.8 release and will be removed in a future CUDA release.
     * These are replaced by the host profiling API in the header cupti_profiler_host.h and
     * target profiling API in the header cupti_range_profiler.h which are supported on
     * devices with compute capability 7.0 and higher (i.e. Volta and later GPU
     * architectures).
     */
    CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED            = 38,
    /**
     * CUPTI doesn't allow multiple callback subscribers. Only a single subscriber
     * can be registered at a time.
     * Same error code is used when application is launched using NVIDIA tools
     * like nvprof, Visual Profiler, Nsight Systems, Nsight Compute, cuda-gdb and
     * cuda-memcheck.
     */
    CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED       = 39,
    /**
     * Profiling on virtualized GPU is not allowed by hypervisor.
     */
    CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES = 40,
    /**
     * Profiling and tracing are not allowed when confidential computing mode
     * is enabled.
     */
    CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED = 41,
    /**
     * CUPTI does not support NVIDIA Crypto Mining Processors (CMP).
     * For more information, please visit https://developer.nvidia.com/ERR_NVCMPGPU
    */
    CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED = 42,
    /**
     * Profiling on Multi-instance GPU (MIG) is not supported.
     */
    CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED = 43,
    /**
     * Profiling on SLI device is not supported.
     */
    CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED = 44,
    /**
     * Profiling on WSL device is not supported.
     */
    CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED = 45,
    /**
     * An unknown internal error has occurred.
     */
    CUPTI_ERROR_UNKNOWN                                  = 999,
    CUPTI_ERROR_FORCE_INT                                = 0x7fffffff
} CUptiResult;

/**
 * \brief Get the descriptive string for a CUptiResult.
 *
 * Return the descriptive string for a CUptiResult in \p *str.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param result The result to get the string for
 * \param str Returns the string
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p str is NULL or \p
 * result is not a valid CUptiResult
 */
CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult result, const char **str);

/**
 * @brief Get the descriptive message corresponding to error codes returned
 * by CUPTI.
 * 
 * Return the descriptive error message for a CUptiResult in \p *str.
 * \note \b Thread-safety: this function is thread safe.
 * 
 * \param result The result to get the descriptive error message for
 * \param str Returns the error message string
 * 
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p str is NULL or \p
 * result is not a valid CUptiResult
 * 
 */

CUptiResult CUPTIAPI cuptiGetErrorMessage(CUptiResult result, const char **str);

/** @} */ /* END CUPTI_RESULT_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_RESULT_H_*/


/*
 * Copyright 2009-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __cuda_stdint_h__
#define __cuda_stdint_h__

// Compiler-specific treatment for C99's stdint.h
//
// By default, this header will use the standard headers (so it
// is your responsibility to make sure they are available), except
// on MSVC before Visual Studio 2010, when they were not provided.
// To support old MSVC, a few of the commonly-used definitions are
// provided here.  If more definitions are needed, add them here,
// or replace these definitions with a complete implementation,
// such as the ones available from Google, Boost, or MSVC10.  You
// can prevent the definition of any of these types (in order to
// use your own) by #defining CU_STDINT_TYPES_ALREADY_DEFINED.

#if !defined(CU_STDINT_TYPES_ALREADY_DEFINED)

// In VS including stdint.h forces the C++ runtime dep - provide an opt-out
// (CU_STDINT_VS_FORCE_NO_STDINT_H) for users that care (notably static
// cudart).
#if defined(_MSC_VER) && ((_MSC_VER < 1600) || defined(CU_STDINT_VS_FORCE_NO_STDINT_H))

// These definitions can be used with MSVC 8 and 9,
// which don't ship with stdint.h:

typedef unsigned   char   uint8_t;

typedef            short  int16_t;
typedef unsigned   short uint16_t;

// To keep it consistent with all MSVC build. define those types
// in the exact same way they are defined with the MSVC headers
#if defined(_MSC_VER)
typedef signed     char    int8_t;

typedef            int     int32_t;
typedef unsigned   int     uint32_t;

typedef long long          int64_t;
typedef unsigned long long uint64_t;
#else
typedef            char    int8_t;

typedef            long   int32_t;
typedef unsigned   long  uint32_t;

typedef          __int64  int64_t;
typedef unsigned __int64 uint64_t;
#endif

#elif defined(__DJGPP__)

// These definitions can be used when compiling
// C code with DJGPP, which only provides stdint.h
// when compiling C++ code with TR1 enabled.

typedef               char    int8_t;
typedef unsigned      char   uint8_t;

typedef               short  int16_t;
typedef unsigned      short uint16_t;

typedef               long   int32_t;
typedef unsigned      long  uint32_t;

typedef          long long   int64_t;
typedef unsigned long long  uint64_t;

#else

// Use standard headers, as specified by C99 and C++ TR1.
// Known to be provided by:
// - gcc/glibc, supported by all versions of glibc
// - djgpp, supported since 2001
// - MSVC, supported by Visual Studio 2010 and later

#include <stdint.h>

#endif

#endif // !defined(CU_STDINT_TYPES_ALREADY_DEFINED)


#endif // file guard


/*
 * Copyright 2010-2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_VERSION_H_)
#define _CUPTI_VERSION_H_

// #include <cuda_stdint.h>
// #include <cupti_result.h>

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_VERSION_API CUPTI Version
 * Function and macro to determine the CUPTI version.
 * @{
 */

/**
 * \brief The API version for this implementation of CUPTI.
 *
 * The API version for this implementation of CUPTI. This define along
 * with \ref cuptiGetVersion can be used to dynamically detect if the
 * version of CUPTI compiled against matches the version of the loaded
 * CUPTI library.
 *
 * v1 : CUDAToolsSDK 4.0
 * v2 : CUDAToolsSDK 4.1
 * v3 : CUDA Toolkit 5.0
 * v4 : CUDA Toolkit 5.5
 * v5 : CUDA Toolkit 6.0
 * v6 : CUDA Toolkit 6.5
 * v7 : CUDA Toolkit 6.5(with sm_52 support)
 * v8 : CUDA Toolkit 7.0
 * v9 : CUDA Toolkit 8.0
 * v10 : CUDA Toolkit 9.0
 * v11 : CUDA Toolkit 9.1
 * v12 : CUDA Toolkit 10.0, 10.1 and 10.2
 * v13 : CUDA Toolkit 11.0
 * v14 : CUDA Toolkit 11.1
 * v15 : CUDA Toolkit 11.2, 11.3 and 11.4
 * v16 : CUDA Toolkit 11.5
 * v17 : CUDA Toolkit 11.6
 * v18 : CUDA Toolkit 11.8
 * v19 : CUDA Toolkit 12.0
 * v20 : CUDA Toolkit 12.2
 * v21 : CUDA Toolkit 12.3
 * v22 : CUDA Toolkit 12.4
 * v23 : CUDA Toolkit 12.5
 * v24 : CUDA Toolkit 12.6
 * v26 : CUDA Toolkit 12.8
 * v27 : CUDA Toolkit 12.9
 * v28 : CUDA Toolkit 12.9 Update 1
 */
#define CUPTI_API_VERSION 28

/**
 * \brief Get the CUPTI API version.
 *
 * Return the API version in \p *version.
 *
 * \param version Returns the version
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p version is NULL
 * \sa CUPTI_API_VERSION
 */
CUptiResult CUPTIAPI cuptiGetVersion(uint32_t *version);

/** @} */ /* END CUPTI_VERSION_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_VERSION_H_*/


/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__DEVICE_TYPES_H__)
#define __DEVICE_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_TYPES_H__
#endif

#ifndef __DOXYGEN_ONLY__
// #include "crt/host_defines.h"

/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#if !defined(__HOST_DEFINES_H__)
#define __HOST_DEFINES_H__

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDADEVRT_INTERNAL__) && !defined(_ALLOW_UNSUPPORTED_LIBCPP)
#include <ctype.h>
#if ((defined(_MSC_VER ) && (defined(_M_X64) || defined(_M_AMD64))) ||\
     (defined(__x86_64__) || defined(__amd64__))) && defined(_LIBCPP_VERSION) && !(defined(__HORIZON__) || defined(__ANDROID__) || defined(__QNX__))
#error "libc++ is not supported on x86 system"
#endif
#endif

/* CUDA JIT mode (__CUDACC_RTC__) also uses GNU style attributes */
#if defined(__GNUC__) || (defined(__PGIC__) && defined(__linux__)) || defined(__CUDA_LIBDEVICE__) || defined(__CUDACC_RTC__)

#if defined(__CUDACC_RTC__)
#define __volatile__ volatile
#endif /* __CUDACC_RTC__ */

#define __no_return__ \
        __attribute__((noreturn))
        
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/   
#define __noinline__ \
        __attribute__((noinline))
#endif /* __CUDACC__  || __CUDA_ARCH__ || __CUDA_LIBDEVICE__ */

#undef __forceinline__
#define __forceinline__ \
        __inline__ __attribute__((always_inline))
#define __inline_hint__ \
        __attribute__((nv_inline_hint))
#define __align__(n) \
        __attribute__((aligned(n)))
#define __maxnreg__(a) \
        __attribute__((maxnreg(a)))
#define __local_maxnreg__(a) \
        __attribute__((local_maxnreg(a)))
#define __thread__ \
        __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) \
        __attribute__((a))
#define __location__(a) \
        __annotate__(a)
#define CUDARTAPI
#define CUDARTAPI_CDECL

#elif defined(_MSC_VER)

#if _MSC_VER >= 1400

#define __restrict__ \
        __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ \
        __inline
#define __no_return__ \
        __declspec(noreturn)
#define __noinline__ \
        __declspec(noinline)
#define __forceinline__ \
        __forceinline
#define __inline_hint__ \
        __declspec(nv_inline_hint)
#define __align__(n) \
        __declspec(align(n))
#define __maxnreg__(n) \
        __declspec(maxnreg(n))
#define __local_maxnreg__(n) \
        __declspec(local_maxnreg(n))
#define __thread__ \
        __declspec(thread)
#define __import__ \
        __declspec(dllimport)
#define __export__ \
        __declspec(dllexport)
#define __annotate__(a) \
        __declspec(a)
#define __location__(a) \
        __annotate__(__##a##__)
#define CUDARTAPI \
        __stdcall
#define CUDARTAPI_CDECL \
        __cdecl

#else /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#define __inline__

#if !defined(__align__)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

#endif /* !__align__ */

#if !defined(CUDARTAPI)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

#endif /* !CUDARTAPI */

#endif /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#if (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__)))) || \
    (defined(_MSC_VER) && _MSC_VER < 1900) || \
    (!defined(__GNUC__) && !defined(_MSC_VER))

#define __specialization_static \
        static

#else /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#define __specialization_static

#endif /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#if !defined(__CUDACC__) && !defined(__CUDA_LIBDEVICE__)

#undef __annotate__
#define __annotate__(a)

#else /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#define __launch_bounds__(...) \
        __annotate__(launch_bounds(__VA_ARGS__))

#endif /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#if defined(__CUDACC__) || defined(__CUDA_LIBDEVICE__) || \
    defined(__GNUC__) || defined(_WIN64)

#define __builtin_align__(a) \
        __align__(a)

#else /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__ || _WIN64 */

#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__  || _WIN64 */

#if defined(__CUDACC__) || !defined(__grid_constant__)
#define __grid_constant__ \
        __location__(grid_constant)
#endif /* defined(__CUDACC__) || !defined(__grid_constant__) */
        
#if defined(__CUDACC__) || !defined(__host__)
#define __host__ \
        __location__(host)
#endif /* defined(__CUDACC__) || !defined(__host__) */
#if defined(__CUDACC__) || !defined(__device__)
#define __device__ \
        __location__(device)
#endif /* defined(__CUDACC__) || !defined(__device__) */
#if defined(__CUDACC__) || !defined(__global__)
#define __global__ \
        __location__(global)
#endif /* defined(__CUDACC__) || !defined(__global__) */
#if defined(__CUDACC__) || !defined(__shared__)
#define __shared__ \
        __location__(shared)
#endif /* defined(__CUDACC__) || !defined(__shared__) */
#if defined(__CUDACC__) || !defined(__constant__)
#define __constant__ \
        __location__(constant)
#endif /* defined(__CUDACC__) || !defined(__constant__) */
#if defined(__CUDACC__) || !defined(__managed__)
#define __managed__ \
        __location__(managed)
#endif /* defined(__CUDACC__) || !defined(__managed__) */
#if defined(__CUDACC__) || !defined(__nv_pure__)
#define __nv_pure__ \
        __location__(nv_pure)
#endif /* defined(__CUDACC__) || !defined(__nv_pure__) */  
#if !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#else /* defined(__CUDACC__) */
#define __device_builtin__ \
        __location__(device_builtin)
#define __device_builtin_texture_type__ \
        __location__(device_builtin_texture_type)
#define __device_builtin_surface_type__ \
        __location__(device_builtin_surface_type)
#define __cudart_builtin__ \
        __location__(cudart_builtin)
#endif /* !defined(__CUDACC__) */

#if defined(__CUDACC__) || !defined(__cluster_dims__)
#if defined(_MSC_VER)        
#define __cluster_dims__(...) \
        __declspec(__cluster_dims__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __cluster_dims__(...) \
        __attribute__((cluster_dims(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__cluster_dims__) */

#if defined(__CUDACC__) || !defined(__block_size__)
#if defined(_MSC_VER)        
#define __block_size__(...) \
        __declspec(__block_size__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __block_size__(...) \
        __attribute__((block_size(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__block_size__) */

#define __CUDA_ARCH_HAS_FEATURE__(_FEAT) __CUDA_ARCH_FEAT_##_FEAT

#ifndef __CUDA_ARCH_FAMILY_SPECIFIC__
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N)  0
#else  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) */
#if (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) (__CUDA_ARCH_FAMILY_SPECIFIC__ == (N##0))
#else
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) \
    ( (((N) == 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)) || \
      (((N) != 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ >= (N##0) && __CUDA_ARCH_FAMILY_SPECIFIC__ < ((N##0) + 100))))
#endif  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010) */
#endif  /* __CUDA_ARCH_FAMILY_SPECIFIC__ */

#ifndef  __CUDA_ARCH_SPECIFIC__
#define __CUDA_HAS_ARCH_SPECIFIC(N) 0
#else
#define __CUDA_HAS_ARCH_SPECIFIC(N) (__CUDA_ARCH_SPECIFIC__ == (N##0))
#endif

#endif /* !__HOST_DEFINES_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

enum __device_builtin__ cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_TYPES_H__
#endif

#endif /* !__DEVICE_TYPES_H__ */

/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "device_types.h"
#if !defined(__CUDACC_RTC__)
#define EXCLUDE_FROM_RTC
// #include "driver_types.h"

/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__DRIVER_TYPES_H__)
#define __DRIVER_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#ifndef __DOXYGEN_ONLY__
// #include "crt/host_defines.h"

/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#if !defined(__HOST_DEFINES_H__)
#define __HOST_DEFINES_H__

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDADEVRT_INTERNAL__) && !defined(_ALLOW_UNSUPPORTED_LIBCPP)
#include <ctype.h>
#if ((defined(_MSC_VER ) && (defined(_M_X64) || defined(_M_AMD64))) ||\
     (defined(__x86_64__) || defined(__amd64__))) && defined(_LIBCPP_VERSION) && !(defined(__HORIZON__) || defined(__ANDROID__) || defined(__QNX__))
#error "libc++ is not supported on x86 system"
#endif
#endif

/* CUDA JIT mode (__CUDACC_RTC__) also uses GNU style attributes */
#if defined(__GNUC__) || (defined(__PGIC__) && defined(__linux__)) || defined(__CUDA_LIBDEVICE__) || defined(__CUDACC_RTC__)

#if defined(__CUDACC_RTC__)
#define __volatile__ volatile
#endif /* __CUDACC_RTC__ */

#define __no_return__ \
        __attribute__((noreturn))
        
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/   
#define __noinline__ \
        __attribute__((noinline))
#endif /* __CUDACC__  || __CUDA_ARCH__ || __CUDA_LIBDEVICE__ */

#undef __forceinline__
#define __forceinline__ \
        __inline__ __attribute__((always_inline))
#define __inline_hint__ \
        __attribute__((nv_inline_hint))
#define __align__(n) \
        __attribute__((aligned(n)))
#define __maxnreg__(a) \
        __attribute__((maxnreg(a)))
#define __local_maxnreg__(a) \
        __attribute__((local_maxnreg(a)))
#define __thread__ \
        __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) \
        __attribute__((a))
#define __location__(a) \
        __annotate__(a)
#define CUDARTAPI
#define CUDARTAPI_CDECL

#elif defined(_MSC_VER)

#if _MSC_VER >= 1400

#define __restrict__ \
        __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ \
        __inline
#define __no_return__ \
        __declspec(noreturn)
#define __noinline__ \
        __declspec(noinline)
#define __forceinline__ \
        __forceinline
#define __inline_hint__ \
        __declspec(nv_inline_hint)
#define __align__(n) \
        __declspec(align(n))
#define __maxnreg__(n) \
        __declspec(maxnreg(n))
#define __local_maxnreg__(n) \
        __declspec(local_maxnreg(n))
#define __thread__ \
        __declspec(thread)
#define __import__ \
        __declspec(dllimport)
#define __export__ \
        __declspec(dllexport)
#define __annotate__(a) \
        __declspec(a)
#define __location__(a) \
        __annotate__(__##a##__)
#define CUDARTAPI \
        __stdcall
#define CUDARTAPI_CDECL \
        __cdecl

#else /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#define __inline__

#if !defined(__align__)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

#endif /* !__align__ */

#if !defined(CUDARTAPI)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

#endif /* !CUDARTAPI */

#endif /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#if (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__)))) || \
    (defined(_MSC_VER) && _MSC_VER < 1900) || \
    (!defined(__GNUC__) && !defined(_MSC_VER))

#define __specialization_static \
        static

#else /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#define __specialization_static

#endif /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#if !defined(__CUDACC__) && !defined(__CUDA_LIBDEVICE__)

#undef __annotate__
#define __annotate__(a)

#else /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#define __launch_bounds__(...) \
        __annotate__(launch_bounds(__VA_ARGS__))

#endif /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#if defined(__CUDACC__) || defined(__CUDA_LIBDEVICE__) || \
    defined(__GNUC__) || defined(_WIN64)

#define __builtin_align__(a) \
        __align__(a)

#else /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__ || _WIN64 */

#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__  || _WIN64 */

#if defined(__CUDACC__) || !defined(__grid_constant__)
#define __grid_constant__ \
        __location__(grid_constant)
#endif /* defined(__CUDACC__) || !defined(__grid_constant__) */
        
#if defined(__CUDACC__) || !defined(__host__)
#define __host__ \
        __location__(host)
#endif /* defined(__CUDACC__) || !defined(__host__) */
#if defined(__CUDACC__) || !defined(__device__)
#define __device__ \
        __location__(device)
#endif /* defined(__CUDACC__) || !defined(__device__) */
#if defined(__CUDACC__) || !defined(__global__)
#define __global__ \
        __location__(global)
#endif /* defined(__CUDACC__) || !defined(__global__) */
#if defined(__CUDACC__) || !defined(__shared__)
#define __shared__ \
        __location__(shared)
#endif /* defined(__CUDACC__) || !defined(__shared__) */
#if defined(__CUDACC__) || !defined(__constant__)
#define __constant__ \
        __location__(constant)
#endif /* defined(__CUDACC__) || !defined(__constant__) */
#if defined(__CUDACC__) || !defined(__managed__)
#define __managed__ \
        __location__(managed)
#endif /* defined(__CUDACC__) || !defined(__managed__) */
#if defined(__CUDACC__) || !defined(__nv_pure__)
#define __nv_pure__ \
        __location__(nv_pure)
#endif /* defined(__CUDACC__) || !defined(__nv_pure__) */  
#if !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#else /* defined(__CUDACC__) */
#define __device_builtin__ \
        __location__(device_builtin)
#define __device_builtin_texture_type__ \
        __location__(device_builtin_texture_type)
#define __device_builtin_surface_type__ \
        __location__(device_builtin_surface_type)
#define __cudart_builtin__ \
        __location__(cudart_builtin)
#endif /* !defined(__CUDACC__) */

#if defined(__CUDACC__) || !defined(__cluster_dims__)
#if defined(_MSC_VER)        
#define __cluster_dims__(...) \
        __declspec(__cluster_dims__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __cluster_dims__(...) \
        __attribute__((cluster_dims(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__cluster_dims__) */

#if defined(__CUDACC__) || !defined(__block_size__)
#if defined(_MSC_VER)        
#define __block_size__(...) \
        __declspec(__block_size__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __block_size__(...) \
        __attribute__((block_size(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__block_size__) */

#define __CUDA_ARCH_HAS_FEATURE__(_FEAT) __CUDA_ARCH_FEAT_##_FEAT

#ifndef __CUDA_ARCH_FAMILY_SPECIFIC__
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N)  0
#else  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) */
#if (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) (__CUDA_ARCH_FAMILY_SPECIFIC__ == (N##0))
#else
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) \
    ( (((N) == 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)) || \
      (((N) != 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ >= (N##0) && __CUDA_ARCH_FAMILY_SPECIFIC__ < ((N##0) + 100))))
#endif  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010) */
#endif  /* __CUDA_ARCH_FAMILY_SPECIFIC__ */

#ifndef  __CUDA_ARCH_SPECIFIC__
#define __CUDA_HAS_ARCH_SPECIFIC(N) 0
#else
#define __CUDA_HAS_ARCH_SPECIFIC(N) (__CUDA_ARCH_SPECIFIC__ == (N##0))
#endif

#endif /* !__HOST_DEFINES_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#endif
// #include "vector_types.h"

/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__VECTOR_TYPES_H__)
#define __VECTOR_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__
#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#ifndef __DOXYGEN_ONLY__
// #include "crt/host_defines.h"

/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#if !defined(__HOST_DEFINES_H__)
#define __HOST_DEFINES_H__

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDADEVRT_INTERNAL__) && !defined(_ALLOW_UNSUPPORTED_LIBCPP)
#include <ctype.h>
#if ((defined(_MSC_VER ) && (defined(_M_X64) || defined(_M_AMD64))) ||\
     (defined(__x86_64__) || defined(__amd64__))) && defined(_LIBCPP_VERSION) && !(defined(__HORIZON__) || defined(__ANDROID__) || defined(__QNX__))
#error "libc++ is not supported on x86 system"
#endif
#endif

/* CUDA JIT mode (__CUDACC_RTC__) also uses GNU style attributes */
#if defined(__GNUC__) || (defined(__PGIC__) && defined(__linux__)) || defined(__CUDA_LIBDEVICE__) || defined(__CUDACC_RTC__)

#if defined(__CUDACC_RTC__)
#define __volatile__ volatile
#endif /* __CUDACC_RTC__ */

#define __no_return__ \
        __attribute__((noreturn))
        
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/   
#define __noinline__ \
        __attribute__((noinline))
#endif /* __CUDACC__  || __CUDA_ARCH__ || __CUDA_LIBDEVICE__ */

#undef __forceinline__
#define __forceinline__ \
        __inline__ __attribute__((always_inline))
#define __inline_hint__ \
        __attribute__((nv_inline_hint))
#define __align__(n) \
        __attribute__((aligned(n)))
#define __maxnreg__(a) \
        __attribute__((maxnreg(a)))
#define __local_maxnreg__(a) \
        __attribute__((local_maxnreg(a)))
#define __thread__ \
        __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) \
        __attribute__((a))
#define __location__(a) \
        __annotate__(a)
#define CUDARTAPI
#define CUDARTAPI_CDECL

#elif defined(_MSC_VER)

#if _MSC_VER >= 1400

#define __restrict__ \
        __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ \
        __inline
#define __no_return__ \
        __declspec(noreturn)
#define __noinline__ \
        __declspec(noinline)
#define __forceinline__ \
        __forceinline
#define __inline_hint__ \
        __declspec(nv_inline_hint)
#define __align__(n) \
        __declspec(align(n))
#define __maxnreg__(n) \
        __declspec(maxnreg(n))
#define __local_maxnreg__(n) \
        __declspec(local_maxnreg(n))
#define __thread__ \
        __declspec(thread)
#define __import__ \
        __declspec(dllimport)
#define __export__ \
        __declspec(dllexport)
#define __annotate__(a) \
        __declspec(a)
#define __location__(a) \
        __annotate__(__##a##__)
#define CUDARTAPI \
        __stdcall
#define CUDARTAPI_CDECL \
        __cdecl

#else /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#define __inline__

#if !defined(__align__)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

#endif /* !__align__ */

#if !defined(CUDARTAPI)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

#endif /* !CUDARTAPI */

#endif /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#if (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__)))) || \
    (defined(_MSC_VER) && _MSC_VER < 1900) || \
    (!defined(__GNUC__) && !defined(_MSC_VER))

#define __specialization_static \
        static

#else /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#define __specialization_static

#endif /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#if !defined(__CUDACC__) && !defined(__CUDA_LIBDEVICE__)

#undef __annotate__
#define __annotate__(a)

#else /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#define __launch_bounds__(...) \
        __annotate__(launch_bounds(__VA_ARGS__))

#endif /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#if defined(__CUDACC__) || defined(__CUDA_LIBDEVICE__) || \
    defined(__GNUC__) || defined(_WIN64)

#define __builtin_align__(a) \
        __align__(a)

#else /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__ || _WIN64 */

#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__  || _WIN64 */

#if defined(__CUDACC__) || !defined(__grid_constant__)
#define __grid_constant__ \
        __location__(grid_constant)
#endif /* defined(__CUDACC__) || !defined(__grid_constant__) */
        
#if defined(__CUDACC__) || !defined(__host__)
#define __host__ \
        __location__(host)
#endif /* defined(__CUDACC__) || !defined(__host__) */
#if defined(__CUDACC__) || !defined(__device__)
#define __device__ \
        __location__(device)
#endif /* defined(__CUDACC__) || !defined(__device__) */
#if defined(__CUDACC__) || !defined(__global__)
#define __global__ \
        __location__(global)
#endif /* defined(__CUDACC__) || !defined(__global__) */
#if defined(__CUDACC__) || !defined(__shared__)
#define __shared__ \
        __location__(shared)
#endif /* defined(__CUDACC__) || !defined(__shared__) */
#if defined(__CUDACC__) || !defined(__constant__)
#define __constant__ \
        __location__(constant)
#endif /* defined(__CUDACC__) || !defined(__constant__) */
#if defined(__CUDACC__) || !defined(__managed__)
#define __managed__ \
        __location__(managed)
#endif /* defined(__CUDACC__) || !defined(__managed__) */
#if defined(__CUDACC__) || !defined(__nv_pure__)
#define __nv_pure__ \
        __location__(nv_pure)
#endif /* defined(__CUDACC__) || !defined(__nv_pure__) */  
#if !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#else /* defined(__CUDACC__) */
#define __device_builtin__ \
        __location__(device_builtin)
#define __device_builtin_texture_type__ \
        __location__(device_builtin_texture_type)
#define __device_builtin_surface_type__ \
        __location__(device_builtin_surface_type)
#define __cudart_builtin__ \
        __location__(cudart_builtin)
#endif /* !defined(__CUDACC__) */

#if defined(__CUDACC__) || !defined(__cluster_dims__)
#if defined(_MSC_VER)        
#define __cluster_dims__(...) \
        __declspec(__cluster_dims__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __cluster_dims__(...) \
        __attribute__((cluster_dims(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__cluster_dims__) */

#if defined(__CUDACC__) || !defined(__block_size__)
#if defined(_MSC_VER)        
#define __block_size__(...) \
        __declspec(__block_size__(__VA_ARGS__))
        
#else  /* !defined(_MSC_VER) */
#define __block_size__(...) \
        __attribute__((block_size(__VA_ARGS__)))
#endif  /* defined(_MSC_VER) */
#endif  /* defined(__CUDACC__) || !defined(__block_size__) */

#define __CUDA_ARCH_HAS_FEATURE__(_FEAT) __CUDA_ARCH_FEAT_##_FEAT

#ifndef __CUDA_ARCH_FAMILY_SPECIFIC__
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N)  0
#else  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) */
#if (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) (__CUDA_ARCH_FAMILY_SPECIFIC__ == (N##0))
#else
#define __CUDA_HAS_ARCH_FAMILY_SPECIFIC(N) \
    ( (((N) == 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010)) || \
      (((N) != 101) && (__CUDA_ARCH_FAMILY_SPECIFIC__ >= (N##0) && __CUDA_ARCH_FAMILY_SPECIFIC__ < ((N##0) + 100))))
#endif  /* defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1010) */
#endif  /* __CUDA_ARCH_FAMILY_SPECIFIC__ */

#ifndef  __CUDA_ARCH_SPECIFIC__
#define __CUDA_HAS_ARCH_SPECIFIC(N) 0
#else
#define __CUDA_HAS_ARCH_SPECIFIC(N) (__CUDA_ARCH_SPECIFIC__ == (N##0))
#endif

#endif /* !__HOST_DEFINES_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#endif

/* NVRTC compiler defines these instead of in the header (to reduce compile time)
*/
#ifndef __CUDACC_RTC_BUILTIN_VECTOR_TYPES__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__) && \
    defined(_WIN32) && !defined(_WIN64)

#pragma warning(push)
#pragma warning(disable: 4201 4408)

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ tag                      \
{                                                  \
    union                                          \
    {                                              \
        struct { members };                        \
        struct { long long int :1,:0; };           \
    };                                             \
}

#else /* !__CUDACC__ && !__CUDACC_RTC__ && _WIN32 && !_WIN64 */

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ __align__(8) tag         \
{                                                  \
    members                                        \
}

#endif /* !__CUDACC__ && !__CUDACC_RTC__ && _WIN32 && !_WIN64 */

struct __device_builtin__ char1
{
    signed char x;
};

struct __device_builtin__ uchar1
{
    unsigned char x;
};


struct __device_builtin__ __align__(2) char2
{
    signed char x, y;
};

struct __device_builtin__ __align__(2) uchar2
{
    unsigned char x, y;
};

struct __device_builtin__ char3
{
    signed char x, y, z;
};

struct __device_builtin__ uchar3
{
    unsigned char x, y, z;
};

struct __device_builtin__ __align__(4) char4
{
    signed char x, y, z, w;
};

struct __device_builtin__ __align__(4) uchar4
{
    unsigned char x, y, z, w;
};

struct __device_builtin__ short1
{
    short x;
};

struct __device_builtin__ ushort1
{
    unsigned short x;
};

struct __device_builtin__ __align__(4) short2
{
    short x, y;
};

struct __device_builtin__ __align__(4) ushort2
{
    unsigned short x, y;
};

struct __device_builtin__ short3
{
    short x, y, z;
};

struct __device_builtin__ ushort3
{
    unsigned short x, y, z;
};

__cuda_builtin_vector_align8(short4, short x; short y; short z; short w;);
__cuda_builtin_vector_align8(ushort4, unsigned short x; unsigned short y; unsigned short z; unsigned short w;);

struct __device_builtin__ int1
{
    int x;
};

struct __device_builtin__ uint1
{
    unsigned int x;
};

__cuda_builtin_vector_align8(int2, int x; int y;);
__cuda_builtin_vector_align8(uint2, unsigned int x; unsigned int y;);

struct __device_builtin__ int3
{
    int x, y, z;
};

struct __device_builtin__ uint3
{
    unsigned int x, y, z;
};

struct __device_builtin__ __builtin_align__(16) int4
{
    int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) uint4
{
    unsigned int x, y, z, w;
};

struct __device_builtin__ long1
{
    long int x;
};

struct __device_builtin__ ulong1
{
    unsigned long x;
};

#if defined(_WIN32)
__cuda_builtin_vector_align8(long2, long int x; long int y;);
__cuda_builtin_vector_align8(ulong2, unsigned long int x; unsigned long int y;);
#else /* !_WIN32 */

struct __device_builtin__ __align__(2*sizeof(long int)) long2
{
    long int x, y;
};

struct __device_builtin__ __align__(2*sizeof(unsigned long int)) ulong2
{
    unsigned long int x, y;
};

#endif /* _WIN32 */

struct __device_builtin__ long3
{
    long int x, y, z;
};

struct __device_builtin__ ulong3
{
    unsigned long int x, y, z;
};

struct __device_builtin__ __builtin_align__(16) long4
{
    long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) ulong4
{
    unsigned long int x, y, z, w;
};

struct __device_builtin__ float1
{
    float x;
};

#if !defined(__CUDACC__) && defined(__arm__) && \
    defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-pedantic"

struct __device_builtin__ __attribute__((aligned(8))) float2
{
    float x; float y; float __cuda_gnu_arm_ice_workaround[0];
};

#pragma GCC poison __cuda_gnu_arm_ice_workaround
#pragma GCC diagnostic pop

#else /* !__CUDACC__ && __arm__ && __ARM_PCS_VFP &&
         __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

__cuda_builtin_vector_align8(float2, float x; float y;);

#endif /* !__CUDACC__ && __arm__ && __ARM_PCS_VFP &&
          __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

struct __device_builtin__ float3
{
    float x, y, z;
};

struct __device_builtin__ __builtin_align__(16) float4
{
    float x, y, z, w;
};

struct __device_builtin__ longlong1
{
    long long int x;
};

struct __device_builtin__ ulonglong1
{
    unsigned long long int x;
};

struct __device_builtin__ __builtin_align__(16) longlong2
{
    long long int x, y;
};

struct __device_builtin__ __builtin_align__(16) ulonglong2
{
    unsigned long long int x, y;
};

struct __device_builtin__ longlong3
{
    long long int x, y, z;
};

struct __device_builtin__ ulonglong3
{
    unsigned long long int x, y, z;
};

struct __device_builtin__ __builtin_align__(16) longlong4
{
    long long int x, y, z ,w;
};

struct __device_builtin__ __builtin_align__(16) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __device_builtin__ double1
{
    double x;
};

struct __device_builtin__ __builtin_align__(16) double2
{
    double x, y;
};

struct __device_builtin__ double3
{
    double x, y, z;
};

struct __device_builtin__ __builtin_align__(16) double4
{
    double x, y, z, w;
};

#if !defined(__CUDACC__) && defined(_WIN32) && !defined(_WIN64)

#pragma warning(pop)

#endif /* !__CUDACC__ && _WIN32 && !_WIN64 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __device_builtin__ struct char1 char1;
typedef __device_builtin__ struct uchar1 uchar1;
typedef __device_builtin__ struct char2 char2;
typedef __device_builtin__ struct uchar2 uchar2;
typedef __device_builtin__ struct char3 char3;
typedef __device_builtin__ struct uchar3 uchar3;
typedef __device_builtin__ struct char4 char4;
typedef __device_builtin__ struct uchar4 uchar4;
typedef __device_builtin__ struct short1 short1;
typedef __device_builtin__ struct ushort1 ushort1;
typedef __device_builtin__ struct short2 short2;
typedef __device_builtin__ struct ushort2 ushort2;
typedef __device_builtin__ struct short3 short3;
typedef __device_builtin__ struct ushort3 ushort3;
typedef __device_builtin__ struct short4 short4;
typedef __device_builtin__ struct ushort4 ushort4;
typedef __device_builtin__ struct int1 int1;
typedef __device_builtin__ struct uint1 uint1;
typedef __device_builtin__ struct int2 int2;
typedef __device_builtin__ struct uint2 uint2;
typedef __device_builtin__ struct int3 int3;
typedef __device_builtin__ struct uint3 uint3;
typedef __device_builtin__ struct int4 int4;
typedef __device_builtin__ struct uint4 uint4;
typedef __device_builtin__ struct long1 long1;
typedef __device_builtin__ struct ulong1 ulong1;
typedef __device_builtin__ struct long2 long2;
typedef __device_builtin__ struct ulong2 ulong2;
typedef __device_builtin__ struct long3 long3;
typedef __device_builtin__ struct ulong3 ulong3;
typedef __device_builtin__ struct long4 long4;
typedef __device_builtin__ struct ulong4 ulong4;
typedef __device_builtin__ struct float1 float1;
typedef __device_builtin__ struct float2 float2;
typedef __device_builtin__ struct float3 float3;
typedef __device_builtin__ struct float4 float4;
typedef __device_builtin__ struct longlong1 longlong1;
typedef __device_builtin__ struct ulonglong1 ulonglong1;
typedef __device_builtin__ struct longlong2 longlong2;
typedef __device_builtin__ struct ulonglong2 ulonglong2;
typedef __device_builtin__ struct longlong3 longlong3;
typedef __device_builtin__ struct ulonglong3 ulonglong3;
typedef __device_builtin__ struct longlong4 longlong4;
typedef __device_builtin__ struct ulonglong4 ulonglong4;
typedef __device_builtin__ struct double1 double1;
typedef __device_builtin__ struct double2 double2;
typedef __device_builtin__ struct double3 double3;
typedef __device_builtin__ struct double4 double4;

#undef  __cuda_builtin_vector_align8

#endif /* !defined(__CUDACC_RTC_BUILTIN_VECTOR_TYPES__) */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

struct __device_builtin__ dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
#if __cplusplus >= 201103L || ( defined(_MSC_VER) && _MSC_VER >= 1900 )
    /* MSVC 2015 introduced support for constexpr constructors. A check in addition to the _cpluscplus macro comparison
       that uses the _MSC_VER macro is required because by default, Visual Studio always returns the value 199711L for 
       the __cplusplus preprocessor macro. */
    __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ constexpr operator uint3(void) const { return uint3{x, y, z}; }
#else
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
#endif /* __cplusplus */
};

typedef __device_builtin__ struct dim3 dim3;

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__
#endif

#endif /* !__VECTOR_TYPES_H__ */


#ifndef __CUDACC_RTC_MINIMAL__
/**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDA_INTERNAL_COMPILATION__)


#if !defined(__CUDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__CUDACC_RTC__) */

#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define cudaStreamDefault                   0x00  /**< Default stream flag */
#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamLegacy                    ((cudaStream_t)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamPerThread                 ((cudaStream_t)0x2)

#define cudaEventDefault                    0x00  /**< Default event flag */
#define cudaEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define cudaEventDisableTiming              0x02  /**< Event will not record timing data */
#define cudaEventInterprocess               0x04  /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */

#define cudaEventRecordDefault              0x00  /**< Default event record flag */
#define cudaEventRecordExternal             0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaEventWaitDefault                0x00  /**< Default event wait flag */
#define cudaEventWaitExternal               0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define cudaDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define cudaDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define cudaDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define cudaDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                                    *  replaced with ::cudaDeviceScheduleBlockingSync. */
#define cudaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define cudaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define cudaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define cudaDeviceSyncMemops                0x80  /**< Device flag - Ensure synchronous memory operations on this context will synchronize */
#define cudaDeviceMask                      0xff  /**< Device flags mask */

#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */
#define cudaArrayColorAttachment            0x20  /**< Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API */
#define cudaArraySparse                     0x40  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array */
#define cudaArrayDeferredMapping            0x80  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array */

#define cudaIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define cudaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define cudaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define cudaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define cudaOccupancyDefault                0x00  /**< Default behavior */
#define cudaOccupancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define cudaCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define cudaInvalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */
#define cudaInitDeviceFlagsAreValid         0x01  /**< Tell the CUDA runtime that DeviceFlags is being set in cudaInitDevice call */
/**
 * If set, each kernel launched as part of ::cudaLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPreSync  0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::cudaLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPostSync 0x02

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

/** \cond impl_private */
#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif
/** \endcond impl_private */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * CUDA error types
 */
enum __device_builtin__ cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    8,
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,

    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidDevicePointer         =     17,
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,

    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read an unsupported data type as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,

    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    cudaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    cudaErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      52,

    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      53,
    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
     * launched grids at a greater depth successfully, the maximum nested
     * depth at which ::cudaDeviceSynchronize will be called must be specified
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime.
     * Keep in mind that additional levels of sync depth require the runtime
     * to reserve large amounts of device memory that cannot be used for
     * user allocations. Note that ::cudaDeviceSynchronize made from device
     * runtime is only supported on devices of compute capability < 9.0.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      98,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    cudaErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    cudaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the CUDA runtime may perform a minimal set of self-tests,
    * as well as CUDA driver tests, to establish the validity of both.
    * Introduced in CUDA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   cudaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    cudaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    cudaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    cudaErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    cudaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    cudaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    cudaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    cudaErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     214,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    cudaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    cudaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    cudaErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to cudaDeviceSynchronize.
     */
    cudaErrorUnsupportedDevSideSync       =     225,

    /**
     * This indicates that an exception occurred on the device that is now
     * contained by the GPU's error containment capability. Common causes are -
     * a. Certain types of invalid accesses of peer GPU memory over nvlink
     * b. Certain classes of hardware errors
     * This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must
     * be terminated and relaunched.
     */
    cudaErrorContained                    =     226,

    /**
     * This indicates that the device kernel source is invalid.
     */
    cudaErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    cudaErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    cudaErrorIllegalState                 =     401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    cudaErrorLossyQuery                   =     402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    cudaErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    cudaErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    cudaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     718,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     720,

    /**
     * An exception occurred on the device while exiting a kernel using tensor memory: the
     * tensor memory was not completely deallocated. This leaves the process in an inconsistent
     * state and any further CUDA work will return the same error. To continue using CUDA, the
     * process must be terminated and relaunched.
     */
    cudaErrorTensorMemoryLeak             =     721,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    cudaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    cudaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    cudaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    cudaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    cudaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    cudaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    cudaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    cudaErrorMpsMaxConnectionsReached     =     809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorMpsClientTerminated          =     810,

    /**
     * This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    cudaErrorCdpNotSupported              =     811,

    /**
     * This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
     */
    cudaErrorCdpVersionMismatch           =     812,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    cudaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    cudaErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    cudaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    cudaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    cudaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    cudaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    cudaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    cudaErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    cudaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    cudaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    cudaErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    cudaErrorExternalDevice               =    911,

    /**
     * This indicates that a kernel launch error has occurred due to cluster
     * misconfiguration.
     */
    cudaErrorInvalidClusterSize           =    912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
     */
    cudaErrorFunctionNotLoaded            =    913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
     */
    cudaErrorInvalidResourceType          =    914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
     */
    cudaErrorInvalidResourceConfiguration =    915,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =    999

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    , cudaErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum __device_builtin__ cudaChannelFormatKind
{
    cudaChannelFormatKindSigned                         =   0,      /**< Signed channel format */
    cudaChannelFormatKindUnsigned                       =   1,      /**< Unsigned channel format */
    cudaChannelFormatKindFloat                          =   2,      /**< Float channel format */
    cudaChannelFormatKindNone                           =   3,      /**< No channel format */
    cudaChannelFormatKindNV12                           =   4,      /**< Unsigned 8-bit integers, planar 4:2:0 YUV format */
    cudaChannelFormatKindUnsignedNormalized8X1          =   5,      /**< 1 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X2          =   6,      /**< 2 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X4          =   7,      /**< 4 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X1         =   8,      /**< 1 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X2         =   9,      /**< 2 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X4         =   10,     /**< 4 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X1            =   11,     /**< 1 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X2            =   12,     /**< 2 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X4            =   13,     /**< 4 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X1           =   14,     /**< 1 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X2           =   15,     /**< 2 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X4           =   16,     /**< 4 channel signed 16-bit normalized integer */
    cudaChannelFormatKindUnsignedBlockCompressed1       =   17,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB   =   18,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    cudaChannelFormatKindUnsignedBlockCompressed2       =   19,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB   =   20,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed3       =   21,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB   =   22,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed4       =   23,     /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindSignedBlockCompressed4         =   24,     /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed5       =   25,     /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindSignedBlockCompressed5         =   26,     /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed6H      =   27,     /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindSignedBlockCompressed6H        =   28,     /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7       =   29,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB   =   30,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedNormalized1010102      =   31      /**< 4 channel unsigned normalized (10-bit, 10-bit, 10-bit, 2-bit) format */

};

/**
 * CUDA Channel format descriptor
 */
struct __device_builtin__ cudaChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum cudaChannelFormatKind f; /**< Channel format kind */
};

/**
 * CUDA array
 */
typedef struct cudaArray *cudaArray_t;

/**
 * CUDA array (as source copy argument)
 */
typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;

/**
 * CUDA mipmapped array
 */
typedef struct cudaMipmappedArray *cudaMipmappedArray_t;

/**
 * CUDA mipmapped array (as source argument)
 */
typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;

/**
 * Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers
 */
#define cudaArraySparsePropertiesSingleMipTail   0x1

/**
 * Sparse CUDA array and CUDA mipmapped array properties
 */
struct __device_builtin__ cudaArraySparseProperties {
    struct {
        unsigned int width;             /**< Tile width in elements */
        unsigned int height;            /**< Tile height in elements */
        unsigned int depth;             /**< Tile depth in elements */
    } tileExtent;
    unsigned int miptailFirstLevel;     /**< First mip level at which the mip tail begins */   
    unsigned long long miptailSize;     /**< Total size of the mip tail. */
    unsigned int flags;                 /**< Flags will either be zero or ::cudaArraySparsePropertiesSingleMipTail */
    unsigned int reserved[4];
};

/**
 * CUDA array and CUDA mipmapped array memory requirements
 */
struct __device_builtin__ cudaArrayMemoryRequirements {
    size_t size;                    /**< Total size of the array. */
    size_t alignment;               /**< Alignment necessary for mapping the array. */
    unsigned int reserved[4];
};

/**
 * CUDA memory types
 */
enum __device_builtin__ cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
    cudaMemoryTypeHost         = 1, /**< Host memory */
    cudaMemoryTypeDevice       = 2, /**< Device memory */
    cudaMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * CUDA memory copy types
 */
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * CUDA Pitched memory pointer
 *
 * \sa ::make_cudaPitchedPtr
 */
struct __device_builtin__ cudaPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * CUDA extent
 *
 * \sa ::make_cudaExtent
 */
struct __device_builtin__ cudaExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * CUDA 3D position
 *
 * \sa ::make_cudaPos
 */
struct __device_builtin__ cudaPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * CUDA 3D memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
    enum cudaMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * Memcpy node parameters
 */
struct __device_builtin__ cudaMemcpyNodeParams {
    int flags;                            /**< Must be zero */
    int reserved[3];                      /**< Must be zero */
    struct cudaMemcpy3DParms copyParams;  /**< Parameters for the memory copy */
};

/**
 * CUDA 3D cross-device memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
};

/**
 * CUDA Memset node parameters
 */
struct __device_builtin__  cudaMemsetParams {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * CUDA Memset node parameters
 */
struct __device_builtin__  cudaMemsetParamsV2 {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
 */
enum __device_builtin__  cudaAccessProperty {
    cudaAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    cudaAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    cudaAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct __device_builtin__ cudaAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. CUDA driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum cudaAccessProperty hitProp;    /**< ::CUaccessProperty set for hit. */
    enum cudaAccessProperty missProp;   /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

#ifdef _WIN32
#define CUDART_CB __stdcall
#else
#define CUDART_CB
#endif

/**
 * CUDA host function
 * \param userData Argument value passed to the function
 */
typedef void (CUDART_CB *cudaHostFn_t)(void *userData);

/**
 * CUDA host node parameters
 */
struct __device_builtin__ cudaHostNodeParams {
    cudaHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * CUDA host node parameters
 */
struct __device_builtin__ cudaHostNodeParamsV2 {
    cudaHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * Possible stream capture statuses returned by ::cudaStreamIsCapturing
 */
enum __device_builtin__ cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    cudaStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    cudaStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
 */
enum __device_builtin__ cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
};

enum __device_builtin__ cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};

/**
 * Cluster scheduling policies. These may be passed to ::cudaFuncSetAttribute
 */
enum __device_builtin__ cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault       = 0, /**< the default policy */
    cudaClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
    cudaClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
};

/**
 * Flags for ::cudaStreamUpdateCaptureDependencies
 */
enum __device_builtin__ cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
    cudaStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
};

/**
 * Flags for user objects for graphs
 */
enum __device_builtin__ cudaUserObjectFlags {
    cudaUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor execution is not synchronized by any CUDA handle. */
};

/**
 * Flags for retaining user object references for graphs
 */
enum __device_builtin__ cudaUserObjectRetainFlags {
    cudaGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
};

/**
 * CUDA graphics interop resource
 */
struct cudaGraphicsResource;

/**
 * CUDA graphics interop register flags
 */
enum __device_builtin__ cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  /**< Default */
    cudaGraphicsRegisterFlagsReadOnly         = 1,  /**< CUDA will not write to this resource */ 
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  /**< CUDA will only write to and will not read from this resource */
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< CUDA will bind this resource to a surface reference */
    cudaGraphicsRegisterFlagsTextureGather    = 8   /**< CUDA will perform texture gather operations on this resource */
};

/**
 * CUDA graphics interop map flags
 */
enum __device_builtin__ cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    cudaGraphicsMapFlagsReadOnly     = 1,  /**< CUDA will not write to this resource */
    cudaGraphicsMapFlagsWriteDiscard = 2   /**< CUDA will only write to and will not read from this resource */
};

/**
 * CUDA graphics interop array indices for cube maps
 */
enum __device_builtin__ cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, /**< Positive X face of cubemap */
    cudaGraphicsCubeFaceNegativeX = 0x01, /**< Negative X face of cubemap */
    cudaGraphicsCubeFacePositiveY = 0x02, /**< Positive Y face of cubemap */
    cudaGraphicsCubeFaceNegativeY = 0x03, /**< Negative Y face of cubemap */
    cudaGraphicsCubeFacePositiveZ = 0x04, /**< Positive Z face of cubemap */
    cudaGraphicsCubeFaceNegativeZ = 0x05  /**< Negative Z face of cubemap */
};

/**
 * CUDA resource types
 */
enum __device_builtin__ cudaResourceType
{
    cudaResourceTypeArray          = 0x00, /**< Array resource */
    cudaResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    cudaResourceTypeLinear         = 0x02, /**< Linear resource */
    cudaResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * CUDA texture resource view formats
 */
enum __device_builtin__ cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    cudaResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    cudaResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    cudaResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    cudaResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    cudaResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    cudaResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    cudaResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    cudaResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    cudaResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    cudaResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    cudaResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    cudaResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    cudaResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    cudaResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    cudaResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    cudaResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    cudaResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    cudaResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * CUDA resource descriptor
 */
struct __device_builtin__ cudaResourceDesc {
    enum cudaResourceType resType;             /**< Resource type */
    
    union {
        struct {
            cudaArray_t array;                 /**< CUDA array */
        } array;
        struct {
            cudaMipmappedArray_t mipmap;       /**< CUDA mipmapped array */
        } mipmap;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t sizeInBytes;                /**< Size in bytes */
        } linear;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t width;                      /**< Width of the array in elements */
            size_t height;                     /**< Height of the array in elements */
            size_t pitchInBytes;               /**< Pitch between two rows in bytes */
        } pitch2D;
    } res;
};

/**
 * CUDA resource view descriptor
 */
struct __device_builtin__ cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
};

/**
 * CUDA pointer attributes
 */
struct __device_builtin__ cudaPointerAttributes
{
    /**
     * The type of memory - ::cudaMemoryTypeUnregistered, ::cudaMemoryTypeHost,
     * ::cudaMemoryTypeDevice or ::cudaMemoryTypeManaged.
     */
    enum cudaMemoryType type;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost or::cudaMemoryTypeManaged then
     * this identifies the device which was current when the memory was allocated
     * or registered (and if that device is deinitialized then this allocation
     * will vanish with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     *
     * \note CUDA doesn't check if unregistered memory is allocated so this field
     * may contain invalid pointer if an invalid pointer has been passed to CUDA.
     */
    void *hostPointer;
};

/**
 * CUDA function attributes
 */
struct __device_builtin__ cudaFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * Refer to ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    * See ::cudaFuncSetAttribute
    */
   int preferredShmemCarveout;

   /**
    * If this attribute is set, the kernel must launch with a valid cluster dimension
    * specified.
    */
   int clusterDimMustBeSet;

   /**
    * The required cluster width/height/depth in blocks. The values must either
    * all be 0 or all be positive. The validity of the cluster dimensions is
    * otherwise checked at launch time.
    *
    * If the value is set during compile time, it cannot be set at runtime.
    * Setting it at runtime should return cudaErrorNotPermitted.
    * See ::cudaFuncSetAttribute
    */
   int requiredClusterWidth;
   int requiredClusterHeight;
   int requiredClusterDepth;

   /**
    * The block scheduling policy of a function.
    * See ::cudaFuncSetAttribute
    */
   int clusterSchedulingPolicyPreference;

   /**
    * Whether the function can be launched with non-portable cluster size. 1 is
    * allowed, 0 is disallowed. A non-portable cluster size may only function
    * on the specific SKUs the program is tested on. The launch might fail if
    * the program is run on a different hardware platform.
    *
    * CUDA API provides ::cudaOccupancyMaxActiveClusters to assist with checking
    * whether the desired size can be launched on the current device.
    *
    * Portable Cluster Size
    *
    * A portable cluster size is guaranteed to be functional on all compute
    * capabilities higher than the target compute capability. The portable
    * cluster size for sm_90 is 8 blocks per cluster. This value may increase
    * for future compute capabilities.
    *
    * The specific hardware unit may support higher cluster sizes that’s not
    * guaranteed to be portable.
    * See ::cudaFuncSetAttribute
    */
   int nonPortableClusterSizeAllowed;

   /**
    * Reserved for future use.
    */
   int reserved[16];
};

/**
 * CUDA function attributes that can be set using ::cudaFuncSetAttribute
 */
enum __device_builtin__ cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
    cudaFuncAttributeClusterDimMustBeSet = 10, /**< Indicator to enforce valid cluster dimension specification on kernel launch */
    cudaFuncAttributeRequiredClusterWidth = 11, /**< Required cluster width */
    cudaFuncAttributeRequiredClusterHeight = 12, /**< Required cluster height */
    cudaFuncAttributeRequiredClusterDepth = 13, /**< Required cluster depth */
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14, /**< Whether non-portable cluster scheduling policy is supported */
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15, /**< Required cluster scheduling policy preference */
    cudaFuncAttributeMax
};

/**
 * CUDA function cache configurations
 */
enum __device_builtin__ cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    cudaFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    cudaFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    cudaFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * CUDA shared memory configuration
 * \deprecated
 */
enum __device_builtin__ cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations. These may be passed to cudaFuncSetAttribute
 */
enum __device_builtin__ cudaSharedCarveout {
    cudaSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
    cudaSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
    cudaSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
};

/**
 * CUDA device compute modes
 */
enum __device_builtin__ cudaComputeMode
{
    cudaComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
};

/**
 * CUDA Limits
 */
enum __device_builtin__ cudaLimit
{
    cudaLimitStackSize                    = 0x00, /**< GPU thread stack size */
    cudaLimitPrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
    cudaLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    cudaLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    cudaLimitDevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
    cudaLimitMaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    cudaLimitPersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
};

/**
 * CUDA Memory Advise values
 */
enum __device_builtin__ cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    cudaMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::cudaMemAdviseSetReadMostly */
    cudaMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    cudaMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    cudaMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    cudaMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * CUDA range attributes
 */
enum __device_builtin__ cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly               = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    cudaMemRangeAttributePreferredLocation        = 2, /**< The preferred location of the range */
    cudaMemRangeAttributeAccessedBy               = 3, /**< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device */
    cudaMemRangeAttributeLastPrefetchLocation     = 4, /**< The last location to which the range was prefetched */
    cudaMemRangeAttributePreferredLocationType    = 5, /**< The preferred location type of the range */
    cudaMemRangeAttributePreferredLocationId      = 6, /**< The preferred location id of the range */
    cudaMemRangeAttributeLastPrefetchLocationType = 7, /**< The last location type to which the range was prefetched */
    cudaMemRangeAttributeLastPrefetchLocationId   = 8  /**< The last location id to which the range was prefetched */
};

/**
 * CUDA GPUDirect RDMA flush writes APIs supported on the device
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesOptions {
    cudaFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::cudaDeviceFlushGPUDirectRDMAWrites() and its CUDA Driver API counterpart are supported on the device. */
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the CUDA device. */
};

/**
 * CUDA GPUDirect RDMA flush writes ordering features of the device
 */
enum __device_builtin__ cudaGPUDirectRDMAWritesOrdering {
    cudaGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::cudaFlushGPUDirectRDMAWrites() can be leveraged if supported. */
    cudaGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not. */
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device. */
};

/**
 * CUDA GPUDirect RDMA flush writes scopes
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesScope {
    cudaFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the CUDA device context owning the data. */
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all CUDA device contexts. */
};

/**
 * CUDA GPUDirect RDMA flush writes targets
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesTarget {
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice /**< Sets the target for ::cudaDeviceFlushGPUDirectRDMAWrites() to the currently active CUDA device context. */
};


/**
 * CUDA device attributes
 */
enum __device_builtin__ cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
    cudaDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    cudaDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    cudaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    cudaDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    cudaDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
    cudaDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
    cudaDevAttrSparseCudaArraySupported       = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    cudaDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< Deprecated, External timeline semaphore interop is supported on the device */
    cudaDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    cudaDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    cudaDevAttrClusterLaunch                  = 120, /**< Indicates device supports cluster launch */
    cudaDevAttrDeferredMappingCudaArraySupported = 121, /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    cudaDevAttrReserved122                    = 122,
    cudaDevAttrReserved123                    = 123,
    cudaDevAttrReserved124                    = 124,
    cudaDevAttrIpcEventSupport                = 125, /**< Device supports IPC Events. */ 
    cudaDevAttrMemSyncDomainCount             = 126, /**< Number of memory synchronization domains the device supports. */
    cudaDevAttrReserved127                    = 127,
    cudaDevAttrReserved128                    = 128,
    cudaDevAttrReserved129                    = 129,
    cudaDevAttrNumaConfig                     = 130, /**< NUMA configuration of a device: value is of type ::cudaDeviceNumaConfig enum */
    cudaDevAttrNumaId                         = 131, /**< NUMA node ID of the GPU memory */
    cudaDevAttrReserved132                    = 132,
    cudaDevAttrMpsEnabled                     = 133, /**< Contexts created on this device will be shared via MPS */
    cudaDevAttrHostNumaId                     = 134, /**< NUMA ID of the host node closest to the device or -1 when system does not support NUMA */
    cudaDevAttrD3D12CigSupported              = 135, /**< Device supports CIG with D3D12. */
    cudaDevAttrVulkanCigSupported             = 138, /**< Device supports CIG with Vulkan. */
    cudaDevAttrGpuPciDeviceId                 = 139, /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID. */
    cudaDevAttrGpuPciSubsystemId              = 140, /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID. */
    cudaDevAttrReserved141                    = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported   = 142, /**< Device supports HOST_NUMA location with the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrHostNumaMultinodeIpcSupported  = 143, /**< Device supports HostNuma location IPC between nodes in a multi-node system. */
    cudaDevAttrMax
};

/**
 * CUDA memory pool attributes
 */
enum __device_builtin__ cudaMemPoolAttr
{
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    cudaMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    cudaMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    cudaMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    cudaMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    cudaMemPoolAttrReservedMemCurrent         = 0x5,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    cudaMemPoolAttrUsedMemCurrent             = 0x7,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Specifies the type of location 
 */
enum __device_builtin__ cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,  /**< Location is a device location, thus id is a device ordinal */
    cudaMemLocationTypeHost = 2     /**< Location is host, id is ignored */
    , cudaMemLocationTypeHostNuma = 3 /**< Location is a host NUMA node, thus id is a host NUMA node id */
    , cudaMemLocationTypeHostNumaCurrent = 4 /**< Location is the host NUMA node closest to the current thread's CPU, id is ignored */
};

/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = ::cudaMemLocationTypeDevice and set id = the gpu's device ordinal.
 * To specify a cpu NUMA node, set type = ::cudaMemLocationTypeHostNuma and set id = host NUMA node id.
 */
struct __device_builtin__ cudaMemLocation {
    enum cudaMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
    int id;                         /**< identifier for a given this location's ::CUmemLocationType. */
};

/**
 * Specifies the memory protection flags for mapping.
 */
enum __device_builtin__ cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
    cudaMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
    cudaMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
};

/**
 * Memory access descriptor
 */
struct __device_builtin__ cudaMemAccessDesc {
    struct cudaMemLocation  location; /**< Location on which the request is to change it's accessibility */
    enum cudaMemAccessFlags flags;    /**< ::CUmemProt accessibility flags to set on the request */
};

/**
 * Defines the allocation types available
 */
enum __device_builtin__ cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    cudaMemAllocationTypePinned  = 0x1,
    cudaMemAllocationTypeMax     = 0x7FFFFFFF 
};

/**
 * Flags for specifying particular handle types
 */
enum __device_builtin__ cudaMemAllocationHandleType {
    cudaMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
    cudaMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    cudaMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    cudaMemHandleTypeWin32Kmt                = 0x4,   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
    cudaMemHandleTypeFabric                  = 0x8  /**< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t) */
};

/**
 * This flag, if set, indicates that the memory will be used as a buffer for
 * hardware accelerated decompression.
 */
#define cudaMemPoolCreateUsageHwDecompress 0x2

/**
 * Specifies the properties of allocations made from the pool.
 */
struct __device_builtin__ cudaMemPoolProps {
    enum cudaMemAllocationType         allocType;   /**< Allocation type. Currently must be specified as cudaMemAllocationTypePinned */
    enum cudaMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
    struct cudaMemLocation             location;    /**< Location allocations should reside. */
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::cudaMemHandleTypeWin32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void                              *win32SecurityAttributes;
    size_t                             maxSize;     /**< Maximum pool size. When set to 0, defaults to a system dependent value.*/
    unsigned short                     usage;        /**< Bitmask indicating intended usage for the pool. */
    unsigned char                      reserved[54]; /**< reserved for future use, must be 0 */
};

/**
 * Opaque data for exporting a pool allocation
 */
struct __device_builtin__ cudaMemPoolPtrExportData {
    unsigned char reserved[64];
};

/**
 * Memory allocation node parameters
 */
struct __device_builtin__ cudaMemAllocNodeParams {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
    */
    struct cudaMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct cudaMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by CUDA */
};

/**
 * Memory allocation node parameters
 */
struct __device_builtin__ cudaMemAllocNodeParamsV2 {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
    */
    struct cudaMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct cudaMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by CUDA */
};

/**
 * Memory free node parameters
 */
struct __device_builtin__ cudaMemFreeNodeParams {
    void *dptr; /**< in: the pointer to free */
};

/**
 * Graph memory attributes
 */
enum __device_builtin__ cudaGraphMemAttributeType {
    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently associated with graphs.
     */
    cudaGraphMemAttrUsedMemCurrent      = 0x0,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, associated with graphs since the
     * last time it was reset.  High watermark can only be reset to zero.
     */
    cudaGraphMemAttrUsedMemHigh         = 0x1,

    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemCurrent  = 0x2,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemHigh     = 0x3
};

/**
 * Flags to specify for copies within a batch. For more details see ::cudaMemcpyBatchAsync.
 */
enum __device_builtin__ cudaMemcpyFlags {
    cudaMemcpyFlagDefault                  = 0x0,

    /**
     * Hint to the driver to try and overlap the copy with compute work on the SMs.
     */
    cudaMemcpyFlagPreferOverlapWithCompute = 0x1
};

enum __device_builtin__ cudaMemcpySrcAccessOrder {
    /**
     * Default invalid.
     */
    cudaMemcpySrcAccessOrderInvalid       = 0x0,

    /**
     * Indicates that access to the source pointer must be in stream order.
     */
    cudaMemcpySrcAccessOrderStream        = 0x1,

    /**
     * Indicates that access to the source pointer can be out of stream order and all
     * accesses must be complete before the API call returns. This flag is suited for
     * ephemeral sources (ex., stack variables) when it's known that no prior operations
     * in the stream can be accessing the memory and also that the lifetime of the memory
     * is limited to the scope that the source variable was declared in. Specifying
     * this flag allows the driver to optimize the copy and removes the need for the user
     * to synchronize the stream after the API call.
     */
    cudaMemcpySrcAccessOrderDuringApiCall = 0x2,

    /**
     * Indicates that access to the source pointer can be out of stream order and the accesses
     * can happen even after the API call returns. This flag is suited for host pointers
     * allocated outside CUDA (ex., via malloc) when it's known that no prior operations
     * in the stream can be accessing the memory. Specifying this flag allows the driver
     * to optimize the copy on certain platforms.
     */
    cudaMemcpySrcAccessOrderAny           = 0x3,

    cudaMemcpySrcAccessOrderMax           = 0x7FFFFFFF
};

/**
 * Attributes specific to copies within a batch. For more details on usage see ::cudaMemcpyBatchAsync.
 */
struct __device_builtin__ cudaMemcpyAttributes {
    enum cudaMemcpySrcAccessOrder srcAccessOrder;  /**< Source access ordering to be observed for copies with this attribute. */
    struct cudaMemLocation srcLocHint;             /**< Hint location for the source operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
    struct cudaMemLocation dstLocHint;             /**< Hint location for the destination operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
    unsigned int flags;                            /**< Additional flags for copies with this attribute. See ::cudaMemcpyFlags. */
};

/**
 * These flags allow applications to convey the operand type for individual copies specified in ::cudaMemcpy3DBatchAsync.
 */
enum __device_builtin__ cudaMemcpy3DOperandType {
    cudaMemcpyOperandTypePointer = 0x1,            /**< Memcpy operand is a valid pointer. */
    cudaMemcpyOperandTypeArray = 0x2,              /**< Memcpy operand is a CUarray. */
    cudaMemcpyOperandTypeMax = 0x7FFFFFFF
};

/**
 * Struct representing offset into a ::cudaArray_t in elements
 */
struct __device_builtin__ cudaOffset3D {
    size_t x;
    size_t y;
    size_t z;
};

/**
 * Struct representing an operand for copy with ::cudaMemcpy3DBatchAsync
 */
struct __device_builtin__ cudaMemcpy3DOperand {
    enum cudaMemcpy3DOperandType type;
    union {
        /**
         * Struct representing an operand when ::cudaMemcpy3DOperand::type is ::cudaMemcpyOperandTypePointer
         */
        struct {
            void *ptr;
            size_t rowLength;                /**< Length of each row in elements. */ 
            size_t layerHeight;              /**< Height of each layer in elements. */ 
            struct cudaMemLocation locHint;  /**< Hint location for the operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
        } ptr;

        /**
         * Struct representing an operand when ::cudaMemcpy3DOperand::type is ::cudaMemcpyOperandTypeArray
         */
        struct {
            cudaArray_t array;
            struct cudaOffset3D offset;
        } array;
    } op;  
};

struct __device_builtin__ cudaMemcpy3DBatchOp {
    struct cudaMemcpy3DOperand src;                /**< Source memcpy operand. */
    struct cudaMemcpy3DOperand dst;                /**< Destination memcpy operand. */
    struct cudaExtent extent;                      /**< Extents of the memcpy between src and dst. The width, height and depth components must not be 0.*/
    enum cudaMemcpySrcAccessOrder srcAccessOrder;  /**< Source access ordering to be observed for copy from src to dst. */
    unsigned int flags;                            /**< Additional flags for copy from src to dst. See ::cudaMemcpyFlags. */
};

/**
 * CUDA device P2P attributes
 */

enum __device_builtin__ cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    cudaDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    cudaDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
    cudaDevP2PAttrCudaArrayAccessSupported     = 4  /**< Accessing CUDA arrays over the link supported */
};

/**
 * CUDA UUID types
 */
#ifndef CU_UUID_HAS_BEEN_DEFINED
#define CU_UUID_HAS_BEEN_DEFINED
struct __device_builtin__ CUuuid_st {     /**< CUDA definition of UUID */
    char bytes[16];
};
typedef __device_builtin__ struct CUuuid_st CUuuid;
#endif
typedef __device_builtin__ struct CUuuid_st cudaUUID_t;

/**
 * CUDA device properties
 */
struct __device_builtin__ cudaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Deprecated, Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Deprecated, Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          computeMode;                /**< Deprecated, Compute mode (See ::cudaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Deprecated, Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Deprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
    int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingCudaArraySupported; /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          reserved[63];               /**< Reserved for future use */
};

/**
 * CUDA IPC Handle Size
 */
#define CUDA_IPC_HANDLE_SIZE 64

/**
 * CUDA IPC event handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcEventHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcEventHandle_t;

/**
 * CUDA IPC memory handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcMemHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;

/*
 * CUDA Mem Fabric Handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaMemFabricHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaMemFabricHandle_t;

/**
 * External memory handle types
 */
enum __device_builtin__ cudaExternalMemoryHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalMemoryHandleTypeOpaqueFd         = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32      = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    cudaExternalMemoryHandleTypeD3D12Heap        = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    cudaExternalMemoryHandleTypeD3D12Resource    = 5,
    /**
    *  Handle is a shared NT handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11Resource    = 6,
    /**
    *  Handle is a globally shared handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    /**
    *  Handle is an NvSciBuf object
    */
    cudaExternalMemoryHandleTypeNvSciBuf         = 8
};

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define cudaExternalMemoryDedicated   0x1

/** When the /p flags parameter of ::cudaExternalSemaphoreSignalParams
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreSignalSkipNvSciBufMemSync     0x01

/** When the /p flags parameter of ::cudaExternalSemaphoreWaitParams
 * contains this flag, it indicates that waiting an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreWaitSkipNvSciBufMemSync       0x02

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need signaler specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrSignal       0x1

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need waiter specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrWait         0x2

/**
 * External memory handle descriptor
 */
struct __device_builtin__ cudaExternalMemoryHandleDesc {
    /**
     * Type of the handle
     */
    enum  cudaExternalMemoryHandleType type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::cudaExternalMemoryHandleTypeOpaqueFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalMemoryHandleTypeD3D12Heap 
         * - ::cudaExternalMemoryHandleTypeD3D12Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing NvSciBuf Object. Valid when type
         * is ::cudaExternalMemoryHandleTypeNvSciBuf
         */
        const void *nvSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::cudaExternalMemoryDedicated
     */
    unsigned int flags;
};

/**
 * External memory buffer descriptor
 */
struct __device_builtin__ cudaExternalMemoryBufferDesc {
    /**
     * Offset into the memory object where the buffer's base is
     */
    unsigned long long offset;
    /**
     * Size of the buffer
     */
    unsigned long long size;
    /**
     * Flags reserved for future use. Must be zero.
     */
    unsigned int flags;
};
 
/**
 * External memory mipmap descriptor
 */
struct __device_builtin__ cudaExternalMemoryMipmappedArrayDesc {
    /**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     */
    unsigned long long offset;
    /**
     * Format of base level of the mipmap chain
     */
    struct cudaChannelFormatDesc formatDesc;
    /**
     * Dimensions of base level of the mipmap chain
     */
    struct cudaExtent extent;
    /**
     * Flags associated with CUDA mipmapped arrays.
     * See ::cudaMallocMipmappedArray
     */
    unsigned int flags;
    /**
     * Total number of levels in the mipmap chain
     */
    unsigned int numLevels;
};
 
/**
 * External semaphore handle types
 */
enum __device_builtin__ cudaExternalSemaphoreHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalSemaphoreHandleTypeOpaqueFd       = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32    = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D12Fence     = 4,
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D11Fence     = 5,
    /**
     * Opaque handle to NvSciSync Object
     */
     cudaExternalSemaphoreHandleTypeNvSciSync     = 6,
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutex     = 7,
    /**
     * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt  = 8,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd  = 9,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
};

/**
 * External semaphore handle descriptor
 */
struct __device_builtin__ cudaExternalSemaphoreHandleDesc {
    /**
     * Type of the handle
     */
    enum cudaExternalSemaphoreHandleType type;
    union {
        /**
         * File descriptor referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueFd
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalSemaphoreHandleTypeD3D12Fence
         * - ::cudaExternalSemaphoreHandleTypeD3D11Fence
         * - ::cudaExternalSemaphoreHandleTypeKeyedMutex
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid synchronization primitive.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * Valid NvSciSyncObj. Must be non NULL
         */
        const void* nvSciSyncObj;
    } handle;
    /**
     * Flags reserved for the future. Must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters(deprecated)
 */
struct __device_builtin__ cudaExternalSemaphoreSignalParams_v1 {
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
* External semaphore wait parameters(deprecated)
*/
struct __device_builtin__ cudaExternalSemaphoreWaitParams_v1 {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreSignalParams{
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/**
 * External semaphore wait parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreWaitParams {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * CUDA Error types
 */
typedef __device_builtin__ enum cudaError cudaError_t;

/**
 * CUDA stream
 */
typedef __device_builtin__ struct CUstream_st *cudaStream_t;

/**
 * CUDA event types
 */
typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

/**
 * CUDA graphics resource types
 */
typedef __device_builtin__ struct cudaGraphicsResource *cudaGraphicsResource_t;

/**
 * CUDA external memory
 */
typedef __device_builtin__ struct CUexternalMemory_st *cudaExternalMemory_t;

/**
 * CUDA external semaphore
 */
typedef __device_builtin__ struct CUexternalSemaphore_st *cudaExternalSemaphore_t;

/**
 * CUDA graph
 */
typedef __device_builtin__ struct CUgraph_st *cudaGraph_t;

/**
 * CUDA graph node.
 */
typedef __device_builtin__ struct CUgraphNode_st *cudaGraphNode_t;

/**
 * CUDA user object for graphs
 */
typedef __device_builtin__ struct CUuserObject_st *cudaUserObject_t;

/**
 * CUDA handle for conditional graph nodes
 */
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

/**
 * CUDA function
 */
typedef __device_builtin__ struct CUfunc_st *cudaFunction_t;

/**
 * CUDA kernel
 */
typedef __device_builtin__ struct CUkern_st *cudaKernel_t;

/**
 * Online compiler and linker options
 */
enum __device_builtin__ cudaJitOption
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitMaxRegisters = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization of the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitThreadsPerBlock = 1,

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     */
    cudaJitWallTime = 2,

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::cudaJitInfoLogBufferSizeBytes)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    cudaJitInfoLogBuffer = 3,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    cudaJitInfoLogBufferSizeBytes = 4,

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::cudaJitErrorLogBufferSizeBytes)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    cudaJitErrorLogBuffer = 5,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    cudaJitErrorLogBufferSizeBytes = 6,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitOptimizationLevel = 7,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::cudaJit_Fallback.
     * Option type: unsigned int for enumerated type ::cudaJit_Fallback\n
     * Applies to: compiler only
     */
    cudaJitFallbackStrategy = 10,

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    cudaJitGenerateDebugInfo = 11,

    /**
     * Generate verbose log messages (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    cudaJitLogVerbose = 12,

    /**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    cudaJitGenerateLineInfo = 13,

    /**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::cudaJit_CacheMode.\n
     * Option type: unsigned int for enumerated type ::cudaJit_CacheMode\n
     * Applies to: compiler only
     */
    cudaJitCacheMode = 14,

    /**
     * Generate position independent code (0: false)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    cudaJitPositionIndependentCode = 30,

    /**
     * This option hints to the JIT compiler the minimum number of CTAs from the
     * kernel’s grid to be mapped to a SM. This option is ignored when used together
     * with ::cudaJitMaxRegisters or ::cudaJitThreadsPerBlock.
     * Optimizations based on this option need ::cudaJitMaxThreadsPerBlock to
     * be specified as well. For kernels already using PTX directive .minnctapersm,
     * this option will be ignored by default. Use ::cudaJitOverrideDirectiveValues
     * to let this option take precedence over the PTX directive.
     * Option type: unsigned int\n
     * Applies to: compiler only
    */
    cudaJitMinCtaPerSm = 31,

     /**
     * Maximum number threads in a thread block, computed as the product of
     * the maximum extent specifed for each dimension of the block. This limit
     * is guaranteed not to be exeeded in any invocation of the kernel. Exceeding
     * the the maximum number of threads results in runtime error or kernel launch
     * failure. For kernels already using PTX directive .maxntid, this option will
     * be ignored by default. Use ::cudaJitOverrideDirectiveValues to let this
     * option take precedence over the PTX directive.
     * Option type: int\n
     * Applies to: compiler only
    */
    cudaJitMaxThreadsPerBlock = 32,

    /**
     * This option lets the values specified using ::cudaJitMaxRegisters,
     * ::cudaJitThreadsPerBlock, ::cudaJitMaxThreadsPerBlock and
     * ::cudaJitMinCtaPerSm take precedence over any PTX directives.
     * (0: Disable, default; 1: Enable)
     * Option type: int\n
     * Applies to: compiler only
    */
    cudaJitOverrideDirectiveValues = 33,
};


/**
 * Library options to be specified with ::cudaLibraryLoadData() or ::cudaLibraryLoadFromFile()
 */
enum __device_builtin__ cudaLibraryOption
{
    cudaLibraryHostUniversalFunctionAndDataTable = 0,

    /**
     * Specifes that the argument \p code passed to ::cudaLibraryLoadData() will be preserved.
     * Specifying this option will let the driver know that \p code can be accessed at any point
     * until ::cudaLibraryUnload(). The default behavior is for the driver to allocate and
     * maintain its own copy of \p code. Note that this is only a memory usage optimization
     * hint and the driver can choose to ignore it if required.
     * Specifying this option with ::cudaLibraryLoadFromFile() is invalid and
     * will return ::cudaErrorInvalidValue.
     */
    cudaLibraryBinaryIsPreserved = 1,
};

struct __device_builtin__ cudalibraryHostUniversalFunctionAndDataTable
{
    void *functionTable;
    size_t functionWindowSize;
    void *dataTable;
    size_t dataWindowSize;
};

/**
 * Caching modes for dlcm
 */
enum __device_builtin__ cudaJit_CacheMode
{
    cudaJitCacheOptionNone = 0,   /**< Compile with no -dlcm flag specified */
    cudaJitCacheOptionCG,         /**< Compile with L1 cache disabled */
    cudaJitCacheOptionCA          /**< Compile with L1 cache enabled */
};

/**
 * Cubin matching fallback strategies
 */
enum __device_builtin__ cudaJit_Fallback
{
    cudaPreferPtx = 0,  /**< Prefer to compile ptx if exact binary match not found */

    cudaPreferBinary    /**< Prefer to fall back to compatible binary code if exact match not found */
};

/**
 * CUDA library
 */
typedef __device_builtin__ struct CUlib_st *cudaLibrary_t;

/**
 * CUDA memory pool
 */
typedef __device_builtin__ struct CUmemPoolHandle_st *cudaMemPool_t;

/**
 * CUDA cooperative group scope
 */
enum __device_builtin__ cudaCGScope {
    cudaCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
    cudaCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    cudaCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
};

/**
 * CUDA launch parameters
 */
struct __device_builtin__ cudaLaunchParams
{
    void *func;          /**< Device function symbol */
    dim3 gridDim;        /**< Grid dimentions */
    dim3 blockDim;       /**< Block dimentions */
    void **args;         /**< Arguments */
    size_t sharedMem;    /**< Shared memory */
    cudaStream_t stream; /**< Stream identifier */
};

/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParams {
    void* func;                     /**< Kernel to launch */
    dim3 gridDim;                   /**< Grid dimensions */
    dim3 blockDim;                  /**< Block dimensions */
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParamsV2 {
    void* func;                     /**< Kernel to launch */
    #if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;                   /**< Grid dimensions */
        dim3 blockDim;                  /**< Block dimensions */
    #else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;                  /**< Grid dimensions */
        uint3 blockDim;                 /**< Block dimensions */
    #endif
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreSignalNodeParams {
    cudaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreSignalNodeParamsV2 {
    cudaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreWaitNodeParams {
    cudaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreWaitNodeParamsV2 {
    cudaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

enum __device_builtin__ cudaGraphConditionalHandleFlags {
    cudaGraphCondAssignDefault = 1 /**< Apply default handle value when graph is launched. */
};

/**
 * CUDA conditional node types
 */
enum __device_builtin__ cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf  = 0,    /**< Conditional 'if/else' Node. Body[0] executed if condition is non-zero.  If \p size == 2, an optional ELSE graph is created and this is executed if the condition is zero. */
    cudaGraphCondTypeWhile = 1,  /**< Conditional 'while' Node. Body executed repeatedly while condition value is non-zero. */
    cudaGraphCondTypeSwitch = 2, /**< Conditional 'switch' Node. Body[n] is executed once, where 'n' is the value of the condition. If the condition does not match a body index, no body is launched. */
};

/**
 * CUDA conditional node parameters
 */
struct __device_builtin__ cudaConditionalNodeParams {
    cudaGraphConditionalHandle handle;       /**< Conditional node handle.
                                                  Handles must be created in advance of creating the node
                                                  using ::cudaGraphConditionalHandleCreate. */
    enum cudaGraphConditionalNodeType type;  /**< Type of conditional node. */
    unsigned int size;                       /**< Size of graph output array.  Allowed values are 1 for cudaGraphCondTypeWhile, 1 or 2
                                                  for cudaGraphCondTypeWhile, or any value greater than zero for cudaGraphCondTypeSwitch. */
    cudaGraph_t *phGraph_out;                /**< CUDA-owned array populated with conditional node child graphs during creation of the node.
                                                  Valid for the lifetime of the conditional node.
                                                  The contents of the graph(s) are subject to the following constraints:
                                                  
                                                  - Allowed node types are kernel nodes, empty nodes, child graphs, memsets,
                                                    memcopies, and conditionals. This applies recursively to child graphs and conditional bodies.
                                                  - All kernels, including kernels in nested conditionals or child graphs at any level,
                                                    must belong to the same CUDA context.
                                                  
                                                  These graphs may be populated using graph node creation APIs or ::cudaStreamBeginCaptureToGraph.
                                                  cudaGraphCondTypeIf:
                                                  phGraph_out[0] is executed when the condition is non-zero.  If \p size == 2, phGraph_out[1] will
                                                  be executed when the condition is zero.
                                                  cudaGraphCondTypeWhile:
                                                  phGraph_out[0] is executed as long as the condition is non-zero.
                                                  cudaGraphCondTypeSwitch:
                                                  phGraph_out[n] is executed when the condition is equal to n.  If the condition >= \p size,
                                                  no body graph is executed.
                                         */
};

/**
* CUDA Graph node types
*/
enum __device_builtin__ cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0x00, /**< GPU kernel node */
    cudaGraphNodeTypeMemcpy      = 0x01, /**< Memcpy node */
    cudaGraphNodeTypeMemset      = 0x02, /**< Memset node */
    cudaGraphNodeTypeHost        = 0x03, /**< Host (executable) node */
    cudaGraphNodeTypeGraph       = 0x04, /**< Node which executes an embedded graph */
    cudaGraphNodeTypeEmpty       = 0x05, /**< Empty (no-op) node */
    cudaGraphNodeTypeWaitEvent   = 0x06, /**< External event wait node */
    cudaGraphNodeTypeEventRecord = 0x07, /**< External event record node */
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08, /**< External semaphore signal node */
    cudaGraphNodeTypeExtSemaphoreWait = 0x09, /**< External semaphore wait node */
    cudaGraphNodeTypeMemAlloc    = 0x0a, /**< Memory allocation node */
    cudaGraphNodeTypeMemFree     = 0x0b, /**< Memory free node */
    cudaGraphNodeTypeConditional = 0x0d, /**< Conditional node
                                              
                                              May be used to implement a conditional execution path or loop
                                              inside of a graph. The graph(s) contained within the body of the conditional node
                                              can be selectively executed or iterated upon based on the value of a conditional
                                              variable.
                                              
                                              Handles must be created in advance of creating the node
                                              using ::cudaGraphConditionalHandleCreate.
                                              
                                              The following restrictions apply to graphs which contain conditional nodes:
                                                The graph cannot be used in a child node.
                                                Only one instantiation of the graph may exist at any point in time.
                                                The graph cannot be cloned.
                                              
                                              To set the control value, supply a default value when creating the handle and/or
                                              call ::cudaGraphSetConditional from device code.*/
    cudaGraphNodeTypeCount
};

/**
 * Child graph node ownership
 */
enum __device_builtin__ cudaGraphChildGraphNodeOwnership {
    cudaGraphChildGraphOwnershipClone = 0,      /**< Default behavior for a child graph node. Child graph is cloned
                                                     into the parent and memory allocation/free nodes can't be present
                                                     in the child graph. */
    cudaGraphChildGraphOwnershipMove  = 1,      /**< The child graph is moved to the parent. The handle to the child graph
                                                     is owned by the parent and will be destroyed when the parent is
                                                     destroyed.

                                                     The following restrictions apply to child graphs after they have been moved:
                                                      Cannot be independently instantiated or destroyed;
                                                      Cannot be added as a child graph of a separate parent graph;
                                                      Cannot be used as an argument to cudaGraphExecUpdate;
                                                      Cannot have additional memory allocation or free nodes added. */
};

/**
 * Child graph node parameters
 */
struct __device_builtin__ cudaChildGraphNodeParams {
    cudaGraph_t graph; /**< The child graph to clone into the node for node creation, or
                        *   a handle to the graph owned by the node for node query.
                        *   The graph must not contain conditional nodes. Graphs
                        *   containing memory allocation or memory free nodes must
                        *   set the ownership to be moved to the parent.
                        */
    enum cudaGraphChildGraphNodeOwnership ownership; /**< The ownership relationship of the child graph node. */
};

/**
 * Event record node parameters
 */
struct __device_builtin__ cudaEventRecordNodeParams {
    cudaEvent_t event; /**< The event to record when the node executes */
};

/**
 * Event wait node parameters
 */
struct __device_builtin__ cudaEventWaitNodeParams {
    cudaEvent_t event; /**< The event to wait on from the node */
};

/**
 * Graph node parameters.  See ::cudaGraphAddNode.
 */
struct __device_builtin__ cudaGraphNodeParams {
    enum cudaGraphNodeType type; /**< Type of the node */
    int reserved0[3];            /**< Reserved.  Must be zero. */

    union {
        long long                                      reserved1[29]; /**< Padding. Unused bytes must be zero. */
        struct cudaKernelNodeParamsV2                  kernel;        /**< Kernel node parameters. */
        struct cudaMemcpyNodeParams                    memcpy;        /**< Memcpy node parameters. */
        struct cudaMemsetParamsV2                      memset;        /**< Memset node parameters. */
        struct cudaHostNodeParamsV2                    host;          /**< Host node parameters. */
        struct cudaChildGraphNodeParams                graph;         /**< Child graph node parameters. */
        struct cudaEventWaitNodeParams                 eventWait;     /**< Event wait node parameters. */
        struct cudaEventRecordNodeParams               eventRecord;   /**< Event record node parameters. */
        struct cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal;  /**< External semaphore signal node parameters. */
        struct cudaExternalSemaphoreWaitNodeParamsV2   extSemWait;    /**< External semaphore wait node parameters. */
        struct cudaMemAllocNodeParamsV2                alloc;         /**< Memory allocation node parameters. */
        struct cudaMemFreeNodeParams                   free;          /**< Memory free node parameters. */
        struct cudaConditionalNodeParams               conditional;   /**< Conditional node parameters. */
    };

    long long reserved2; /**< Reserved bytes. Must be zero. */
};

/**
 * Type annotations that can be applied to graph edges as part of ::cudaGraphEdgeData.
 */
typedef __device_builtin__ enum cudaGraphDependencyType_enum {
    cudaGraphDependencyTypeDefault = 0, /**< This is an ordinary dependency. */
    cudaGraphDependencyTypeProgrammatic = 1  /**< This dependency type allows the downstream node to
                                                  use \c cudaGridDependencySynchronize(). It may only be used
                                                  between kernel nodes, and must be used with either the
                                                  ::cudaGraphKernelNodePortProgrammatic or
                                                  ::cudaGraphKernelNodePortLaunchCompletion outgoing port. */
} cudaGraphDependencyType;

/**
 * Optional annotation for edges in a CUDA graph. Note, all edges implicitly have annotations and
 * default to a zero-initialized value if not specified. A zero-initialized struct indicates a
 * standard full serialization of two nodes with memory visibility.
 */
typedef __device_builtin__ struct cudaGraphEdgeData_st {
    unsigned char from_port; /**< This indicates when the dependency is triggered from the upstream
                                  node on the edge. The meaning is specfic to the node type. A value
                                  of 0 in all cases means full completion of the upstream node, with
                                  memory visibility to the downstream node or portion thereof
                                  (indicated by \c to_port).
                                  <br>
                                  Only kernel nodes define non-zero ports. A kernel node
                                  can use the following output port types:
                                  ::cudaGraphKernelNodePortDefault, ::cudaGraphKernelNodePortProgrammatic,
                                  or ::cudaGraphKernelNodePortLaunchCompletion. */
    unsigned char to_port; /**< This indicates what portion of the downstream node is dependent on
                                the upstream node or portion thereof (indicated by \c from_port). The
                                meaning is specific to the node type. A value of 0 in all cases means
                                the entirety of the downstream node is dependent on the upstream work.
                                <br>
                                Currently no node types define non-zero ports. Accordingly, this field
                                must be set to zero. */
    unsigned char type; /**< This should be populated with a value from ::cudaGraphDependencyType. (It
                             is typed as char due to compiler-specific layout of bitfields.) See
                             ::cudaGraphDependencyType. */
    unsigned char reserved[5]; /**< These bytes are unused and must be zeroed. This ensures
                                    compatibility if additional fields are added in the future. */
} cudaGraphEdgeData;

/**
 * This port activates when the kernel has finished executing.
 */
#define cudaGraphKernelNodePortDefault 0
/**
 * This port activates when all blocks of the kernel have performed cudaTriggerProgrammaticLaunchCompletion()
 * or have terminated. It must be used with edge type ::cudaGraphDependencyTypeProgrammatic. See also
 * ::cudaLaunchAttributeProgrammaticEvent.
 */
#define cudaGraphKernelNodePortProgrammatic 1
/**
 * This port activates when all blocks of the kernel have begun execution. See also
 * ::cudaLaunchAttributeLaunchCompletionEvent.
 */
#define cudaGraphKernelNodePortLaunchCompletion 2

/**
 * CUDA executable (launchable) graph
 */
typedef struct CUgraphExec_st* cudaGraphExec_t;

/**
* CUDA Graph Update error types
*/
enum __device_builtin__ cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess                = 0x0, /**< The update succeeded */
    cudaGraphExecUpdateError                  = 0x1, /**< The update failed for an unexpected reason which is described in the return value of the function */
    cudaGraphExecUpdateErrorTopologyChanged   = 0x2, /**< The update failed because the topology changed */
    cudaGraphExecUpdateErrorNodeTypeChanged   = 0x3, /**< The update failed because a node type changed */
    cudaGraphExecUpdateErrorFunctionChanged   = 0x4, /**< The update failed because the function of a kernel node changed (CUDA driver < 11.2) */
    cudaGraphExecUpdateErrorParametersChanged = 0x5, /**< The update failed because the parameters changed in a way that is not supported */
    cudaGraphExecUpdateErrorNotSupported      = 0x6, /**< The update failed because something about the node is not supported */
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7, /**< The update failed because the function of a kernel node changed in an unsupported way */
    cudaGraphExecUpdateErrorAttributesChanged = 0x8 /**< The update failed because the node attributes changed in a way that is not supported */
};

/**
 * Graph instantiation results
*/
typedef __device_builtin__ enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,                       /**< Instantiation succeeded */
    cudaGraphInstantiateError = 1,                         /**< Instantiation failed for an unexpected reason which is described in the return value of the function */
    cudaGraphInstantiateInvalidStructure = 2,              /**< Instantiation failed due to invalid structure, such as cycles */
    cudaGraphInstantiateNodeOperationNotSupported = 3,     /**< Instantiation for device launch failed because the graph contained an unsupported operation */
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,   /**< Instantiation for device launch failed due to the nodes belonging to different contexts */
    cudaGraphInstantiateConditionalHandleUnused = 5        /**< One or more conditional handles are not associated with conditional nodes */
} cudaGraphInstantiateResult;

/**
 * Graph instantiation parameters
 */
typedef __device_builtin__ struct cudaGraphInstantiateParams_st
{
    unsigned long long flags;              /**< Instantiation flags */
    cudaStream_t uploadStream;             /**< Upload stream */
    cudaGraphNode_t errNode_out;           /**< The node which caused instantiation to fail, if any */
    cudaGraphInstantiateResult result_out; /**< Whether instantiation was successful.  If it failed, the reason why */
} cudaGraphInstantiateParams;

/**
 * Result information returned by cudaGraphExecUpdate
 */
typedef __device_builtin__ struct cudaGraphExecUpdateResultInfo_st {
    /**
     * Gives more specific detail when a cuda graph update fails. 
     */
    enum cudaGraphExecUpdateResult result;

    /**
     * The "to node" of the error edge when the topologies do not match.
     * The error node when the error is associated with a specific node.
     * NULL when the error is generic.
     */
    cudaGraphNode_t errorNode;

    /**
     * The from node of error edge when the topologies do not match. Otherwise NULL.
     */
    cudaGraphNode_t errorFromNode;
} cudaGraphExecUpdateResultInfo;

/**
 * CUDA device node handle for device-side node update
 */
typedef struct CUgraphDeviceUpdatableNode_st* cudaGraphDeviceNode_t;

/**
 * Specifies the field to update when performing multiple node updates from the device
 */
enum __device_builtin__ cudaGraphKernelNodeField
{
    cudaGraphKernelNodeFieldInvalid = 0, /**< Invalid field */
    cudaGraphKernelNodeFieldGridDim,     /**< Grid dimension update */
    cudaGraphKernelNodeFieldParam,       /**< Kernel parameter update */
    cudaGraphKernelNodeFieldEnabled      /**< Node enable/disable */
};

/**
 * Struct to specify a single node update to pass as part of a larger array to ::cudaGraphKernelNodeUpdatesApply
 */
struct __device_builtin__ cudaGraphKernelNodeUpdate {
    cudaGraphDeviceNode_t node;     /**< Node to update */
    enum cudaGraphKernelNodeField field; /**< Which type of update to apply. Determines how updateData is interpreted */
    union {
#if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;               /**< Grid dimensions */
#else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;              /**< Grid dimensions */
#endif
        struct {
            const void *pValue;     /**< Kernel parameter data to write in */
            size_t offset;          /**< Offset into the parameter buffer at which to apply the update */
            size_t size;            /**< Number of bytes to update */
        } param;                    /**< Kernel parameter data */
        unsigned int isEnabled;     /**< Node enable/disable data. Nonzero if the node should be enabled, 0 if it should be disabled */
    } updateData;                   /**< Update data to apply. Which field is used depends on field's value */
};

/**
 * Flags to specify search options to be used with ::cudaGetDriverEntryPoint
 * For more details see ::cuGetProcAddress
 */ 
enum __device_builtin__ cudaGetDriverEntryPointFlags {
    cudaEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
    cudaEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
    cudaEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
};

/**
 * Enum for status from obtaining driver entry points, used with ::cudaApiGetDriverEntryPoint
 */
enum __device_builtin__ cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess             = 0,  /**< Search for symbol found a match */
    cudaDriverEntryPointSymbolNotFound      = 1,  /**< Search for symbol was not found */
    cudaDriverEntryPointVersionNotSufficent = 2   /**< Search for symbol was found but version wasn't great enough */
};

/**
 * CUDA Graph debug write options
 */
enum __device_builtin__ cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose                  = 1<<0,  /**< Output all debug data as if every debug flag is enabled */
    cudaGraphDebugDotFlagsKernelNodeParams         = 1<<2,  /**< Adds cudaKernelNodeParams to output */
    cudaGraphDebugDotFlagsMemcpyNodeParams         = 1<<3,  /**< Adds cudaMemcpy3DParms to output */
    cudaGraphDebugDotFlagsMemsetNodeParams         = 1<<4,  /**< Adds cudaMemsetParams to output */
    cudaGraphDebugDotFlagsHostNodeParams           = 1<<5,  /**< Adds cudaHostNodeParams to output */
    cudaGraphDebugDotFlagsEventNodeParams          = 1<<6,  /**< Adds cudaEvent_t handle from record and wait nodes to output */
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7,  /**< Adds cudaExternalSemaphoreSignalNodeParams values to output */
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams   = 1<<8,  /**< Adds cudaExternalSemaphoreWaitNodeParams to output */
    cudaGraphDebugDotFlagsKernelNodeAttributes     = 1<<9,  /**< Adds cudaKernelNodeAttrID values to output */
    cudaGraphDebugDotFlagsHandles                  = 1<<10, /**< Adds node handles and every kernel function handle to output */
    cudaGraphDebugDotFlagsConditionalNodeParams    = 1<<15, /**< Adds cudaConditionalNodeParams to output */
};

/**
 * Flags for instantiating a graph
 */
enum __device_builtin__ cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1 /**< Automatically free memory allocated in a graph before relaunching. */
  , cudaGraphInstantiateFlagUpload           = 2 /**< Automatically upload the graph after instantiation. Only supported by                                                                                                                                                                                                                                                                                                     
                                                      ::cudaGraphInstantiateWithParams.  The upload will be performed using the                                                                                                                                                                                                                                                                                                   
                                                      stream provided in \p instantiateParams. */                                                                                                                                                                                                                                                                                                                               
  , cudaGraphInstantiateFlagDeviceLaunch     = 4 /**< Instantiate the graph to be launchable from the device. This flag can only                                                                                                                                                                                                                                                                                                
                                                      be used on platforms which support unified addressing. This flag cannot be                                                                                                                                                                                                                                                                                                
                                                      used in conjunction with cudaGraphInstantiateFlagAutoFreeOnLaunch. */                                                                                                                                                                                                                                                                                              
  , cudaGraphInstantiateFlagUseNodePriority  = 8 /**< Run the graph using the per-node priority attributes rather than the
                                                      priority of the stream it is launched into. */
};

/**
 * Memory Synchronization Domain
 *
 * A kernel can be launched in a specified memory synchronization domain that affects all memory operations issued by
 * that kernel. A memory barrier issued in one domain will only order memory operations in that domain, thus eliminating
 * latency increase from memory barriers ordering unrelated traffic.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomain, ::cudaStreamSetAttribute, ::cudaLaunchKernelEx,
 * ::cudaGraphKernelNodeSetAttribute.
 *
 * Memory operations done in kernels launched in different domains are considered system-scope distanced. In other
 * words, a GPU scoped memory synchronization is not sufficient for memory order to be observed by kernels in another
 * memory synchronization domain even if they are on the same GPU.
 */
typedef __device_builtin__ enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,    /**< Launch kernels in the default domain */
    cudaLaunchMemSyncDomainRemote  = 1     /**< Launch kernels in the remote domain */
} cudaLaunchMemSyncDomain;

/**
 * Memory Synchronization Domain map
 *
 * See ::cudaLaunchMemSyncDomain.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomainMap.
 *
 * Domain ID range is available through ::cudaDevAttrMemSyncDomainCount.
 */
typedef __device_builtin__ struct cudaLaunchMemSyncDomainMap_st {
    unsigned char default_;                /**< The default domain ID to use for designated kernels */
    unsigned char remote;                  /**< The remote domain ID to use for designated kernels */
} cudaLaunchMemSyncDomainMap;

/**
 * Launch attributes enum; used as id field of ::cudaLaunchAttribute
 */
typedef __device_builtin__ enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore                = 0 /**< Ignored entry, for convenient composition */
  , cudaLaunchAttributeAccessPolicyWindow    = 1 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::accessPolicyWindow. */
  , cudaLaunchAttributeCooperative           = 2 /**< Valid for graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::cooperative. */
  , cudaLaunchAttributeSynchronizationPolicy = 3 /**< Valid for streams. See ::cudaLaunchAttributeValue::syncPolicy. */
  , cudaLaunchAttributeClusterDimension                  = 4 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterDim. */
  , cudaLaunchAttributeClusterSchedulingPolicyPreference = 5 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterSchedulingPolicyPreference. */
  , cudaLaunchAttributeProgrammaticStreamSerialization   = 6 /**< Valid for launches. Setting
                                                                  ::cudaLaunchAttributeValue::programmaticStreamSerializationAllowed
                                                                  to non-0 signals that the kernel will use programmatic
                                                                  means to resolve its stream dependency, so that the
                                                                  CUDA runtime should opportunistically allow the grid's
                                                                  execution to overlap with the previous kernel in the
                                                                  stream, if that kernel requests the overlap. The
                                                                  dependent launches can choose to wait on the
                                                                  dependency using the programmatic sync
                                                                  (cudaGridDependencySynchronize() or equivalent PTX
                                                                  instructions). */
  , cudaLaunchAttributeProgrammaticEvent                 = 7 /**< Valid for launches. Set
                                                                  ::cudaLaunchAttributeValue::programmaticEvent to
                                                                  record the event. Event recorded through this launch
                                                                  attribute is guaranteed to only trigger after all
                                                                  block in the associated kernel trigger the event.  A
                                                                  block can trigger the event programmatically in a
                                                                  future CUDA release. A trigger can also be inserted at
                                                                  the beginning of each block's execution if
                                                                  triggerAtBlockStart is set to non-0. The dependent
                                                                  launches can choose to wait on the dependency using
                                                                  the programmatic sync (cudaGridDependencySynchronize()
                                                                  or equivalent PTX instructions). Note that dependents
                                                                  (including the CPU thread calling
                                                                  cudaEventSynchronize()) are not guaranteed to observe
                                                                  the release precisely when it is released. For
                                                                  example, cudaEventSynchronize() may only observe the
                                                                  event trigger long after the associated kernel has
                                                                  completed. This recording type is primarily meant for
                                                                  establishing programmatic dependency between device
                                                                  tasks. Note also this type of dependency allows, but
                                                                  does not guarantee, concurrent execution of tasks.
                                                                  <br>
                                                                  The event supplied must not be an interprocess or
                                                                  interop event. The event must disable timing (i.e.
                                                                  must be created with the ::cudaEventDisableTiming flag
                                                                  set). */
  , cudaLaunchAttributePriority              = 8 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::priority. */
  , cudaLaunchAttributeMemSyncDomainMap                  = 9 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomainMap. */
  , cudaLaunchAttributeMemSyncDomain                    = 10 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomain. */
  , cudaLaunchAttributePreferredClusterDimension = 11 /**< Valid for graph nodes and launches. Set
                                                           ::cudaLaunchAttributeValue::preferredClusterDim
                                                           to allow the kernel launch to specify a preferred substitute
                                                           cluster dimension. Blocks may be grouped according to either
                                                           the dimensions specified with this attribute (grouped into a
                                                           "preferred substitute cluster"), or the one specified with
                                                           ::cudaLaunchAttributeClusterDimension attribute (grouped
                                                           into a "regular cluster"). The cluster dimensions of a
                                                           "preferred substitute cluster" shall be an integer multiple
                                                           greater than zero of the regular cluster dimensions. The
                                                           device will attempt - on a best-effort basis - to group
                                                           thread blocks into preferred clusters over grouping them
                                                           into regular clusters. When it deems necessary (primarily
                                                           when the device temporarily runs out of physical resources
                                                           to launch the larger preferred clusters), the device may
                                                           switch to launch the regular clusters instead to attempt to
                                                           utilize as much of the physical device resources as possible.
                                                           <br>
                                                           Each type of cluster will have its enumeration / coordinate
                                                           setup as if the grid consists solely of its type of cluster.
                                                           For example, if the preferred substitute cluster dimensions
                                                           double the regular cluster dimensions, there might be
                                                           simultaneously a regular cluster indexed at (1,0,0), and a
                                                           preferred cluster indexed at (1,0,0). In this example, the
                                                           preferred substitute cluster (1,0,0) replaces regular
                                                           clusters (2,0,0) and (3,0,0) and groups their blocks.
                                                           <br>
                                                           This attribute will only take effect when a regular cluster
                                                           dimension has been specified. The preferred substitute cluster
                                                           dimension must be an integer multiple greater than zero of the
                                                           regular cluster dimension and must divide the grid. It must
                                                           also be no more than `maxBlocksPerCluster`, if it is set in
                                                           the kernel's `__launch_bounds__`. Otherwise it must be less
                                                           than the maximum value the driver can support. Otherwise,
                                                           setting this attribute to a value physically unable to fit on
                                                           any particular device is permitted. */
  , cudaLaunchAttributeLaunchCompletionEvent = 12 /**< Valid for launches. Set
                                                       ::cudaLaunchAttributeValue::launchCompletionEvent to record the
                                                       event.
                                                       <br>
                                                       Nominally, the event is triggered once all blocks of the kernel
                                                       have begun execution. Currently this is a best effort. If a kernel
                                                       B has a launch completion dependency on a kernel A, B may wait
                                                       until A is complete. Alternatively, blocks of B may begin before
                                                       all blocks of A have begun, for example if B can claim execution
                                                       resources unavailable to A (e.g. they run on different GPUs) or
                                                       if B is a higher priority than A.
                                                       Exercise caution if such an ordering inversion could lead
                                                       to deadlock.
                                                       <br>
                                                       A launch completion event is nominally similar to a programmatic
                                                       event with \c triggerAtBlockStart set except that it is not
                                                       visible to \c cudaGridDependencySynchronize() and can be used with
                                                       compute capability less than 9.0.
                                                       <br>
                                                       The event supplied must not be an interprocess or interop event.
                                                       The event must disable timing (i.e. must be created with the
                                                       ::cudaEventDisableTiming flag set). */
  , cudaLaunchAttributeDeviceUpdatableKernelNode = 13 /**< Valid for graph nodes, launches. This attribute is graphs-only,
                                                           and passing it to a launch in a non-capturing stream will result
                                                           in an error.
                                                           <br>
                                                           :cudaLaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can 
                                                           only be set to 0 or 1. Setting the field to 1 indicates that the
                                                           corresponding kernel node should be device-updatable. On success, a handle
                                                           will be returned via
                                                           ::cudaLaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be
                                                           passed to the various device-side update functions to update the node's
                                                           kernel parameters from within another kernel. For more information on the
                                                           types of device updates that can be made, as well as the relevant limitations
                                                           thereof, see ::cudaGraphKernelNodeUpdatesApply.
                                                           <br>
                                                           Nodes which are device-updatable have additional restrictions compared to
                                                           regular kernel nodes. Firstly, device-updatable nodes cannot be removed
                                                           from their graph via ::cudaGraphDestroyNode. Additionally, once opted-in
                                                           to this functionality, a node cannot opt out, and any attempt to set the
                                                           deviceUpdatable attribute to 0 will result in an error. Device-updatable
                                                           kernel nodes also cannot have their attributes copied to/from another kernel
                                                           node via ::cudaGraphKernelNodeCopyAttributes. Graphs containing one or more
                                                           device-updatable nodes also do not allow multiple instantiation, and neither
                                                           the graph nor its instantiated version can be passed to ::cudaGraphExecUpdate.
                                                           <br>
                                                           If a graph contains device-updatable nodes and updates those nodes from the device
                                                           from within the graph, the graph must be uploaded with ::cuGraphUpload before it
                                                           is launched. For such a graph, if host-side executable graph updates are made to the
                                                           device-updatable nodes, the graph must be uploaded before it is launched again. */
  , cudaLaunchAttributePreferredSharedMemoryCarveout = 14 /**< Valid for launches. On devices where the L1 cache and shared memory use the
                                                               same hardware resources, setting ::cudaLaunchAttributeValue::sharedMemCarveout 
                                                               to a percentage between 0-100 signals sets the shared memory carveout 
                                                               preference in percent of the total shared memory for that kernel launch. 
                                                               This attribute takes precedence over ::cudaFuncAttributePreferredSharedMemoryCarveout.
                                                               This is only a hint, and the driver can choose a different configuration if
                                                               required for the launch.*/  
} cudaLaunchAttributeID;

/**
 * Launch attributes union; used as value field of ::cudaLaunchAttribute
 */
typedef __device_builtin__ union cudaLaunchAttributeValue {
    char pad[64]; /* Pad to 64 bytes */
    struct cudaAccessPolicyWindow accessPolicyWindow; /**< Value of launch attribute ::cudaLaunchAttributeAccessPolicyWindow. */
    int cooperative; /**< Value of launch attribute ::cudaLaunchAttributeCooperative. Nonzero indicates a cooperative
                        kernel (see ::cudaLaunchCooperativeKernel). */
    enum cudaSynchronizationPolicy syncPolicy; /**< Value of launch attribute
                                                  ::cudaLaunchAttributeSynchronizationPolicy. ::cudaSynchronizationPolicy
                                                  for work queued up in this stream. */
    /**
     * Value of launch attribute ::cudaLaunchAttributeClusterDimension that
     * represents the desired cluster dimensions for the kernel. Opaque type
     * with the following fields:
     *     - \p x - The X dimension of the cluster, in blocks. Must be a divisor
     *              of the grid X dimension.
     *     - \p y - The Y dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Y dimension.
     *     - \p z - The Z dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Z dimension.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } clusterDim;
    enum cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; /**< Value of launch attribute
                                                                           ::cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
                                                                           scheduling policy preference for the kernel. */
    int programmaticStreamSerializationAllowed; /**< Value of launch attribute
                                                   ::cudaLaunchAttributeProgrammaticStreamSerialization. */

    /**
     * Value of launch attribute ::cudaLaunchAttributeProgrammaticEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when all blocks trigger it.
     *     - \p int flags;        - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                               ::cudaEventRecordExternal.
     *     - \p int triggerAtBlockStart - If this is set to non-0, each block launch will automatically trigger the event.
     */
    struct {
        cudaEvent_t event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
    int priority; /**< Value of launch attribute ::cudaLaunchAttributePriority. Execution priority of the kernel. */
    cudaLaunchMemSyncDomainMap memSyncDomainMap; /**< Value of launch attribute
                                                    ::cudaLaunchAttributeMemSyncDomainMap. See
                                                    ::cudaLaunchMemSyncDomainMap. */
    cudaLaunchMemSyncDomain memSyncDomain;       /**< Value of launch attribute ::cudaLaunchAttributeMemSyncDomain. See
                                                    ::cudaLaunchMemSyncDomain. */
    /**
     * Value of launch attribute ::cudaLaunchAttributePreferredClusterDimension
     * that represents the desired preferred cluster dimensions for the kernel.
     * Opaque type with the following fields:
     *     - \p x - The X dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid X dimension, and must be a multiple of
     *              the \p x field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p y - The Y dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid Y dimension, and must be a multiple of
     *              the \p y field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p z - The Z dimension of the preferred cluster, in blocks. Must be
     *              equal to the \p z field of ::cudaLaunchAttributeValue::clusterDim.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } preferredClusterDim;

    /**
     * Value of launch attribute ::cudaLaunchAttributeLaunchCompletionEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when the last block launches.
     *     - \p int flags - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                   ::cudaEventRecordExternal.
     */
    struct {
        cudaEvent_t event;
        int flags;
    } launchCompletionEvent;

    /**
     * Value of launch attribute ::cudaLaunchAttributeDeviceUpdatableKernelNode
     * with the following fields:
     *    - \p int deviceUpdatable - Whether or not the resulting kernel node should be device-updatable.
     *    - \p cudaGraphDeviceNode_t devNode - Returns a handle to pass to the various device-side update functions.
     */
    struct {
        int deviceUpdatable;
        cudaGraphDeviceNode_t devNode;
    } deviceUpdatableKernelNode;
    unsigned int sharedMemCarveout; /**< Value of launch attribute ::cudaLaunchAttributePreferredSharedMemoryCarveout. */
} cudaLaunchAttributeValue;

/**
 * Launch attribute
 */
typedef __device_builtin__ struct cudaLaunchAttribute_st {
    cudaLaunchAttributeID id; /**< Attribute to set */
    char pad[8 - sizeof(cudaLaunchAttributeID)];
    cudaLaunchAttributeValue val; /**< Value of the attribute */
} cudaLaunchAttribute;

/**
 * CUDA extensible launch configuration
 */
typedef __device_builtin__ struct cudaLaunchConfig_st {
    dim3 gridDim;               /**< Grid dimensions */
    dim3 blockDim;              /**< Block dimensions */
    size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    cudaStream_t stream;        /**< Stream identifier */
    cudaLaunchAttribute *attrs; /**< List of attributes; nullable if ::cudaLaunchConfig_t::numAttrs == 0 */
    unsigned int numAttrs;      /**< Number of attributes populated in ::cudaLaunchConfig_t::attrs */
} cudaLaunchConfig_t;

#define cudaStreamAttrID cudaLaunchAttributeID
#define cudaStreamAttributeAccessPolicyWindow    cudaLaunchAttributeAccessPolicyWindow
#define cudaStreamAttributeSynchronizationPolicy cudaLaunchAttributeSynchronizationPolicy
#define cudaStreamAttributeMemSyncDomainMap      cudaLaunchAttributeMemSyncDomainMap
#define cudaStreamAttributeMemSyncDomain         cudaLaunchAttributeMemSyncDomain
#define cudaStreamAttributePriority cudaLaunchAttributePriority

#define cudaStreamAttrValue cudaLaunchAttributeValue

#define cudaKernelNodeAttrID cudaLaunchAttributeID
#define cudaKernelNodeAttributeAccessPolicyWindow cudaLaunchAttributeAccessPolicyWindow
#define cudaKernelNodeAttributeCooperative        cudaLaunchAttributeCooperative
#define cudaKernelNodeAttributePriority           cudaLaunchAttributePriority
#define cudaKernelNodeAttributeClusterDimension                     cudaLaunchAttributeClusterDimension
#define cudaKernelNodeAttributeClusterSchedulingPolicyPreference    cudaLaunchAttributeClusterSchedulingPolicyPreference
#define cudaKernelNodeAttributeMemSyncDomainMap   cudaLaunchAttributeMemSyncDomainMap
#define cudaKernelNodeAttributeMemSyncDomain      cudaLaunchAttributeMemSyncDomain
#define cudaKernelNodeAttributePreferredSharedMemoryCarveout cudaLaunchAttributePreferredSharedMemoryCarveout
#define cudaKernelNodeAttributeDeviceUpdatableKernelNode cudaLaunchAttributeDeviceUpdatableKernelNode

#define cudaKernelNodeAttrValue cudaLaunchAttributeValue

/**
 * CUDA device NUMA config
 */
enum __device_builtin__  cudaDeviceNumaConfig {
    cudaDeviceNumaConfigNone  = 0, /**< The GPU is not a NUMA node */
    cudaDeviceNumaConfigNumaNode, /**< The GPU is a NUMA node, cudaDevAttrNumaId contains its NUMA ID */
};

/**
 * CUDA async callback handle
 */
typedef struct cudaAsyncCallbackEntry* cudaAsyncCallbackHandle_t;

struct cudaAsyncCallbackEntry;

/**
* Types of async notification that can occur
*/
typedef __device_builtin__ enum cudaAsyncNotificationType_enum {
    cudaAsyncNotificationTypeOverBudget = 0x1 /**< Sent when the process has exceeded its device memory budget */
} cudaAsyncNotificationType;

/**
* Information describing an async notification event
*/
typedef __device_builtin__ struct cudaAsyncNotificationInfo
{
    cudaAsyncNotificationType type; /**< The type of notification being sent */
    union {
        struct {
            unsigned long long bytesOverBudget; /**< The number of bytes that the process has allocated above its device memory budget */
        } overBudget; /**< Information about notifications of type \p cudaAsyncNotificationTypeOverBudget */
    } info; /**< Information about the notification. \p type must be checked in order to interpret this field. */
} cudaAsyncNotificationInfo_t;

typedef void (*cudaAsyncCallback)(cudaAsyncNotificationInfo_t*, void*, cudaAsyncCallbackHandle_t);


/** @} */
/** @} */ /* END CUDART_TYPES */

#endif  /* !__CUDACC_RTC_MINIMAL__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#undef __CUDA_DEPRECATED



#endif /* !__DRIVER_TYPES_H__ */

#undef EXCLUDE_FROM_RTC
#endif /* !__CUDACC_RTC__ */
// #include "surface_types.h"
// #include "texture_types.h"
// #include "vector_types.h"

/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__SURFACE_TYPES_H__)
#define __SURFACE_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "driver_types.h"

#ifndef __CUDACC_RTC_MINIMAL__

/**
 * \addtogroup CUDART_TYPES
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#define cudaSurfaceType1D              0x01
#define cudaSurfaceType2D              0x02
#define cudaSurfaceType3D              0x03
#define cudaSurfaceTypeCubemap         0x0C
#define cudaSurfaceType1DLayered       0xF1
#define cudaSurfaceType2DLayered       0xF2
#define cudaSurfaceTypeCubemapLayered  0xFC

/**
 * CUDA Surface boundary modes
 */
enum __device_builtin__ cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero  = 0,    /**< Zero boundary mode */
    cudaBoundaryModeClamp = 1,    /**< Clamp boundary mode */
    cudaBoundaryModeTrap  = 2     /**< Trap boundary mode */
};

/**
 * CUDA Surface format modes
 */
enum __device_builtin__  cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,     /**< Forced format mode */
    cudaFormatModeAuto = 1        /**< Auto format mode */
};

/**
 * An opaque value that represents a CUDA Surface object
 */
typedef __device_builtin__ unsigned long long cudaSurfaceObject_t;

/** @} */
/** @} */ /* END CUDART_TYPES */

#endif  /* !__CUDACC_RTC_MINIMAL__ */
#endif /* !__SURFACE_TYPES_H__ */


/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__TEXTURE_TYPES_H__)
#define __TEXTURE_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

// #include "driver_types.h"

#ifndef __CUDACC_RTC_MINIMAL__

/**
 * \addtogroup CUDART_TYPES
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#define cudaTextureType1D              0x01
#define cudaTextureType2D              0x02
#define cudaTextureType3D              0x03
#define cudaTextureTypeCubemap         0x0C
#define cudaTextureType1DLayered       0xF1
#define cudaTextureType2DLayered       0xF2
#define cudaTextureTypeCubemapLayered  0xFC

/**
 * CUDA texture address modes
 */
enum __device_builtin__ cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,    /**< Wrapping address mode */
    cudaAddressModeClamp  = 1,    /**< Clamp to edge address mode */
    cudaAddressModeMirror = 2,    /**< Mirror address mode */
    cudaAddressModeBorder = 3     /**< Border address mode */
};

/**
 * CUDA texture filter modes
 */
enum __device_builtin__ cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,     /**< Point filter mode */
    cudaFilterModeLinear = 1      /**< Linear filter mode */
};

/**
 * CUDA texture read modes
 */
enum __device_builtin__ cudaTextureReadMode
{
    cudaReadModeElementType     = 0,  /**< Read texture as specified element type */
    cudaReadModeNormalizedFloat = 1   /**< Read texture as normalized float */
};

/**
 * CUDA texture descriptor
 */
struct __device_builtin__ cudaTextureDesc
{
    /**
     * Texture address mode for up to 3 dimensions
     */
    enum cudaTextureAddressMode addressMode[3];
    /**
     * Texture filter mode
     */
    enum cudaTextureFilterMode  filterMode;
    /**
     * Texture read mode
     */
    enum cudaTextureReadMode    readMode;
    /**
     * Perform sRGB->linear conversion during texture read
     */
    int                         sRGB;
    /**
     * Texture Border Color
     */
    float                       borderColor[4];
    /**
     * Indicates whether texture reads are normalized or not
     */
    int                         normalizedCoords;
    /**
     * Limit to the anisotropy ratio
     */
    unsigned int                maxAnisotropy;
    /**
     * Mipmap filter mode
     */
    enum cudaTextureFilterMode  mipmapFilterMode;
    /**
     * Offset applied to the supplied mipmap level
     */
    float                       mipmapLevelBias;
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    float                       minMipmapLevelClamp;
    /**
     * Upper end of the mipmap level range to clamp access to
     */
    float                       maxMipmapLevelClamp;
    /**
     * Disable any trilinear filtering optimizations.
     */
    int                         disableTrilinearOptimization;
    /**
     * Enable seamless cube map filtering.
     */
    int                         seamlessCubemap;
};

/**
 * An opaque value that represents a CUDA texture object
 */
typedef __device_builtin__ unsigned long long cudaTextureObject_t;

/** @} */
/** @} */ /* END CUDART_TYPES */

#endif  /* !__CUDACC_RTC_MINIMAL__ */
#endif /* !__TEXTURE_TYPES_H__ */


/*
 * Copyright 2010-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUPTI_CALLBACKS_H__)
#define __CUPTI_CALLBACKS_H__

// #include <cuda.h>
// #include <builtin_types.h>
#include <string.h>
// #include <cuda_stdint.h>
// #include <cupti_result.h>

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_CALLBACK_API CUPTI Callback API
 * Functions, types, and enums that implement the CUPTI Callback API.
 * @{
 */

/**
 * \brief Specifies the point in an API call that a callback is issued.
 *
 * Specifies the point in an API call that a callback is issued. This
 * value is communicated to the callback function via \ref
 * CUpti_CallbackData::callbackSite.
 */
typedef enum {
  /**
   * The callback is at the entry of the API call.
   */
  CUPTI_API_ENTER                 = 0,
  /**
   * The callback is at the exit of the API call.
   */
  CUPTI_API_EXIT                  = 1,
  CUPTI_API_CBSITE_FORCE_INT     = 0x7fffffff
} CUpti_ApiCallbackSite;

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a
 * group of related API functions or CUDA driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  CUPTI_CB_DOMAIN_INVALID           = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  CUPTI_CB_DOMAIN_DRIVER_API        = 1,
  /**
   * Domain containing callback points for all runtime API
   * functions.
   */
  CUPTI_CB_DOMAIN_RUNTIME_API       = 2,
  /**
   * Domain containing callback points for CUDA resource tracking.
   */
  CUPTI_CB_DOMAIN_RESOURCE          = 3,
  /**
   * Domain containing callback points for CUDA synchronization.
   */
  CUPTI_CB_DOMAIN_SYNCHRONIZE       = 4,
  /**
   * Domain containing callback points for NVTX API functions.
   */
  CUPTI_CB_DOMAIN_NVTX              = 5,
  /**
   * Domain containing callback points for various states.
   */
  CUPTI_CB_DOMAIN_STATE             = 6,

  CUPTI_CB_DOMAIN_SIZE,

  CUPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
} CUpti_CallbackDomain;

/**
 * \brief Callback IDs for resource domain.
 *
 * Callback IDs for resource domain, CUPTI_CB_DOMAIN_RESOURCE.  This
 * value is communicated to the callback function via the \p cbid
 * parameter.
 */
typedef enum {
  /**
   * Invalid resource callback ID.
   */
  CUPTI_CBID_RESOURCE_INVALID                               = 0,
  /**
   * A new context has been created.
   */
  CUPTI_CBID_RESOURCE_CONTEXT_CREATED                       = 1,
  /**
   * A context is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING              = 2,
  /**
   * A new stream has been created.
   */
  CUPTI_CBID_RESOURCE_STREAM_CREATED                        = 3,
  /**
   * A stream is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING               = 4,
  /**
   * The driver has finished initializing.
   */
  CUPTI_CBID_RESOURCE_CU_INIT_FINISHED                      = 5,
  /**
   * A module has been loaded.
   */
  CUPTI_CBID_RESOURCE_MODULE_LOADED                         = 6,
  /**
   * A module is about to be unloaded.
   */
  CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING                = 7,
  /**
   * The current module which is being profiled.
   */
  CUPTI_CBID_RESOURCE_MODULE_PROFILED                       = 8,
  /**
   * CUDA graph has been created.
   */
  CUPTI_CBID_RESOURCE_GRAPH_CREATED                         = 9,
  /**
   * CUDA graph is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING                = 10,
  /**
   * CUDA graph is cloned.
   */
  CUPTI_CBID_RESOURCE_GRAPH_CLONED                          = 11,
  /**
   * CUDA graph node is about to be created
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING             = 12,
  /**
   * CUDA graph node is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED                     = 13,
  /**
   * CUDA graph node is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING            = 14,
  /**
   * Dependency on a CUDA graph node is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED          = 15,
  /**
   * Dependency on a CUDA graph node is destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16,
  /**
   * An executable CUDA graph is about to be created.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING             = 17,
  /**
   * An executable CUDA graph is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED                     = 18,
  /**
   * An executable CUDA graph is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING            = 19,
  /**
   * CUDA graph node is cloned.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED                      = 20,
  /**
   * CUDA stream attribute is changed.
   */
  CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED              = 21,

  CUPTI_CBID_RESOURCE_SIZE,
  CUPTI_CBID_RESOURCE_FORCE_INT                   = 0x7fffffff
} CUpti_CallbackIdResource;

/**
 * \brief Callback IDs for synchronization domain.
 *
 * Callback IDs for synchronization domain,
 * CUPTI_CB_DOMAIN_SYNCHRONIZE.  This value is communicated to the
 * callback function via the \p cbid parameter.
 */
typedef enum {
  /**
   * Invalid synchronize callback ID.
   */
  CUPTI_CBID_SYNCHRONIZE_INVALID                  = 0,
  /**
   * Stream synchronization has completed for the stream.
   */
  CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED      = 1,
  /**
   * Context synchronization has completed for the context.
   */
  CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED     = 2,
  CUPTI_CBID_SYNCHRONIZE_SIZE,
  CUPTI_CBID_SYNCHRONIZE_FORCE_INT                = 0x7fffffff
} CUpti_CallbackIdSync;


/**
 * \brief Callback IDs for state domain.
 *
 * Callback IDs for state domain,
 * CUPTI_CB_DOMAIN_STATE. This value is communicated to the
 * callback function via the \p cbid parameter.
 */
typedef enum {
  /**
   * Invalid state callback ID.
   */
  CUPTI_CBID_STATE_INVALID                        = 0,
  /**
   * Notification of fatal errors - high impact, non-recoverable
   * When encountered, CUPTI automatically invokes cuptiFinalize()
   * User can control behavior of the application in future from 
   * receiving this callback - such as continuing without profiling, or
   * terminating the whole application.
   */
  CUPTI_CBID_STATE_FATAL_ERROR                    = 1,
  /**
   * Notification of non fatal errors - high impact, but recoverable
   * This notification is not issued in the current release.
   */
  CUPTI_CBID_STATE_ERROR                          = 2,
  /**
   * Notification of warnings - low impact, recoverable.
   */
  CUPTI_CBID_STATE_WARNING                        = 3,

  CUPTI_CBID_STATE_SIZE,
  CUPTI_CBID_STATE_FORCE_INT         = 0x7fffffff
} CUpti_CallbackIdState;


/**
 * \brief Data passed into a runtime or driver API callback function.
 *
 * Data passed into a runtime or driver API callback function as the
 * \p cbdata argument to \ref CUpti_CallbackFunc. The \p cbdata will
 * be this type for \p domain equal to CUPTI_CB_DOMAIN_DRIVER_API or
 * CUPTI_CB_DOMAIN_RUNTIME_API. The callback data is valid only within
 * the invocation of the callback function that is passed the data. If
 * you need to retain some data for use outside of the callback, you
 * must make a copy of that data. For example, if you make a shallow
 * copy of CUpti_CallbackData within a callback, you cannot
 * dereference \p functionParams outside of that callback to access
 * the function parameters. \p functionName is an exception: the
 * string pointed to by \p functionName is a global constant and so
 * may be accessed outside of the callback.
 */
typedef struct {
  /**
   * Point in the runtime or driver function from where the callback
   * was issued.
   */
  CUpti_ApiCallbackSite callbackSite;

  /**
   * Name of the runtime or driver API function which issued the
   * callback. This string is a global constant and so may be
   * accessed outside of the callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the runtime or driver API
   * call. See generated_cuda_runtime_api_meta.h and
   * generated_cuda_meta.h for structure definitions for the
   * parameters for each runtime and driver API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the runtime or driver API
   * call. This field is only valid within the exit::CUPTI_API_EXIT
   * callback. For a runtime API \p functionReturnValue points to a
   * \p cudaError_t. For a driver API \p functionReturnValue points
   * to a \p CUresult.
   */
  void *functionReturnValue;

  /**
   * Name of the symbol operated on by the runtime or driver API
   * function which issued the callback. This entry is valid only for
   * driver and runtime launch callbacks, where it returns the name of
   * the kernel.
   */
  const char *symbolName;

  /**
   * Driver context current to the thread, or null if no context is
   * current. This value can change from the entry to exit callback
   * of a runtime API function if the runtime initializes a context.
   */
  CUcontext context;

  /**
   * Unique ID for the CUDA context associated with the thread. The
   * UIDs are assigned sequentially as contexts are created and are
   * unique within a process.
   */
  uint32_t contextUid;

  /**
   * Pointer to data shared between the entry and exit callbacks of
   * a given runtime or drive API function invocation. This field
   * can be used to pass 64-bit values from the entry callback to
   * the corresponding exit callback.
   */
  uint64_t *correlationData;

  /**
   * The activity record correlation ID for this callback. For a
   * driver domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID
   * in the CUpti_ActivityAPI record corresponding to the CUDA driver
   * function call. For a runtime domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_RUNTIME_API) this ID will equal the correlation
   * ID in the CUpti_ActivityAPI record corresponding to the CUDA
   * runtime function call. Within the callback, this ID can be
   * recorded to correlate user data with the activity record. This
   * field is new in 4.1.
   */
  uint32_t correlationId;

} CUpti_CallbackData;

/**
 * \brief Data passed into a resource callback function.
 *
 * Data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The callback
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * For CUPTI_CBID_RESOURCE_CONTEXT_CREATED and
   * CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING, the context being
   * created or destroyed. For CUPTI_CBID_RESOURCE_STREAM_CREATED and
   * CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the context
   * containing the stream being created or destroyed.
   */
  CUcontext context;

  union {
    /**
     * For CUPTI_CBID_RESOURCE_STREAM_CREATED and
     * CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the stream being
     * created or destroyed.
     */
    CUstream stream;
  } resourceHandle;

  /**
   * Reserved for future use.
   */
  void *resourceDescriptor;
} CUpti_ResourceData;


/**
 * \brief Module data passed into a resource callback function.
 *
 * CUDA module data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The module
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * Identifier to associate with the CUDA module.
   */
    uint32_t moduleId;

  /**
   * The size of the cubin.
   */
    size_t cubinSize;

  /**
   * Pointer to the associated cubin.
   */
    const char *pCubin;
} CUpti_ModuleResourceData;

/**
 * \brief CUDA graphs data passed into a resource callback function.
 *
 * CUDA graphs data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The graph
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * CUDA graph
   */
    CUgraph graph;
  /**
   * The original CUDA graph from which \param graph is cloned
   */
    CUgraph originalGraph;
  /**
   * CUDA graph node
   */
    CUgraphNode node;
  /**
   * The original CUDA graph node from which \param node is cloned
   */
    CUgraphNode originalNode;
  /**
   * Type of the \param node
   */
    CUgraphNodeType nodeType;
  /**
   * The dependent graph node
   * The size of the array is \param numDependencies.
   */
    CUgraphNode dependency;
  /**
   * CUDA executable graph
   */
    CUgraphExec graphExec;
} CUpti_GraphData;

/**
 * \brief Data passed into a synchronize callback function.
 *
 * Data passed into a synchronize callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_SYNCHRONIZE. The
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * The context of the stream being synchronized.
   */
  CUcontext context;
  /**
   * The stream being synchronized.
   */
  CUstream  stream;
} CUpti_SynchronizeData;

/**
 * \brief Data passed into a NVTX callback function.
 *
 * Data passed into a NVTX callback function as the \p cbdata argument
 * to \ref CUpti_CallbackFunc. The \p cbdata will be this type for \p
 * domain equal to CUPTI_CB_DOMAIN_NVTX. Unless otherwise notes, the
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * Name of the NVTX API function which issued the callback. This
   * string is a global constant and so may be accessed outside of the
   * callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the NVTX API call. See
   * generated_nvtx_meta.h for structure definitions for the
   * parameters for each NVTX API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the NVTX API call. See
   * nvToolsExt.h for each NVTX API function's return value.
   */
  const void *functionReturnValue;
} CUpti_NvtxData;

/**
 * \brief Stream attribute data passed into a resource callback function
 * for CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED callback

 * Data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The
 * stream attribute data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * The CUDA stream handle for the attribute
   */
  CUstream stream;

  /**
   * The type of the CUDA stream attribute
   */
  CUstreamAttrID attr;

  /**
   * The value of the CUDA stream attribute
   */
  const CUstreamAttrValue *value;
} CUpti_StreamAttrData;

/**
 * \brief Data passed into a State callback function.
 *
 * Data passed into a State callback function as the \p cbdata argument
 * to \ref CUpti_CallbackFunc. The \p cbdata will be this type for \p
 * domain equal to CUPTI_CB_DOMAIN_STATE and callback Ids belonging to CUpti_CallbackIdState. 
 * Unless otherwise noted, the callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  union {
    /**
     * Data passed along with the callback Ids 
     * Enum CUpti_CallbackIdState used to denote callback ids
     */
    struct {
      /**
       * Error code
       */
      CUptiResult result;
      /**
       * String containing more details. It can be NULL.
       */
      const char *message;
    } notification;
  };
} CUpti_StateData;

/**
 * \brief An ID for a driver API, runtime API, resource or
 * synchronization callback.
 *
 * An ID for a driver API, runtime API, resource or synchronization
 * callback. Within a driver API callback this should be interpreted
 * as a CUpti_driver_api_trace_cbid value (these values are defined in
 * cupti_driver_cbid.h). Within a runtime API callback this should be
 * interpreted as a CUpti_runtime_api_trace_cbid value (these values
 * are defined in cupti_runtime_cbid.h). Within a resource API
 * callback this should be interpreted as a \ref
 * CUpti_CallbackIdResource value. Within a synchronize API callback
 * this should be interpreted as a \ref CUpti_CallbackIdSync value.
 */
typedef uint32_t CUpti_CallbackId;

/**
 * \brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in \p cbdata depends on the \p domain. If \p domain is
 * CUPTI_CB_DOMAIN_DRIVER_API or CUPTI_CB_DOMAIN_RUNTIME_API the type
 * of \p cbdata will be CUpti_CallbackData. If \p domain is
 * CUPTI_CB_DOMAIN_RESOURCE the type of \p cbdata will be
 * CUpti_ResourceData. If \p domain is CUPTI_CB_DOMAIN_SYNCHRONIZE the
 * type of \p cbdata will be CUpti_SynchronizeData. If \p domain is
 * CUPTI_CB_DOMAIN_NVTX the type of \p cbdata will be CUpti_NvtxData.
 *
 * \param userdata User data supplied at subscription of the callback
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param cbdata Data passed to the callback.
 */
typedef void (CUPTIAPI *CUpti_CallbackFunc)(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *cbdata);

/**
 * \brief A callback subscriber.
 */
typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

/**
 * \brief Pointer to an array of callback domains.
 */
typedef CUpti_CallbackDomain *CUpti_DomainTable;

/**
 * \brief Get the available callback domains.
 *
 * Returns in \p *domainTable an array of size \p *domainCount of all
 * the available callback domains.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param domainCount Returns number of callback domains
 * \param domainTable Returns pointer to array of available callback domains
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialize CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p domainCount or \p domainTable are NULL
 */
CUptiResult CUPTIAPI cuptiSupportedDomains(size_t *domainCount,
                                           CUpti_DomainTable *domainTable);

/**
 * \brief Initialize a callback subscriber with a callback function
 * and user data.
 *
 * Initializes a callback subscriber with a callback function and
 * (optionally) a pointer to user data. The returned subscriber handle
 * can be used to enable and disable the callback for specific domains
 * and callback IDs.
 * \note Only a single subscriber can be registered at a time. To ensure
 * that no other CUPTI client interrupts the profiling session, it's the
 * responsibility of all the CUPTI clients to call this function before
 * starting the profling session. In case profiling session is already
 * started by another CUPTI client, this function returns the error code
 * CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.
 * Note that this function returns the same error when application is
 * launched using NVIDIA tools like nvprof, Visual Profiler, Nsight Systems,
 * Nsight Compute, cuda-gdb and cuda-memcheck.
 * \note This function does not enable any callbacks.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Returns handle to initialize subscriber
 * \param callback The callback function
 * \param userdata A pointer to user data. This data will be passed to
 * the callback function via the \p userdata parameter.
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialize CUPTI
 * \retval CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if there is already a CUPTI subscriber
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL
 */
CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                                    CUpti_CallbackFunc callback,
                                    void *userdata);

/**
 * \brief Unregister a callback subscriber.
 *
 * Removes a callback subscriber so that no future callbacks will be
 * issued to that subscriber.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Handle to the initialize subscriber
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL or not initialized
 */
CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);

/**
 * \brief Get the current enabled/disabled state of a callback for a specific
 * domain and function ID.
 *
 * Returns non-zero in \p *enable if the callback for a domain and
 * callback ID is enabled, and zero if not enabled.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, c) and cuptiEnableCallback(sub, d, c) are called concurrently,
 * the results are undefined.
 *
 * \param enable Returns non-zero if callback enabled, zero if not enabled
 * \param subscriber Handle to the initialize subscriber
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p enabled is NULL, or if \p
 * subscriber, \p domain or \p cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiGetCallbackState(uint32_t *enable,
                                           CUpti_SubscriberHandle subscriber,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid);

/**
 * \brief Enable or disabled callbacks for a specific domain and
 * callback ID.
 *
 * Enable or disabled callbacks for a subscriber for a specific domain
 * and callback ID.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, c) and cuptiEnableCallback(sub, d, c) are called concurrently,
 * the results are undefined.
 *
 * \param enable New enable state for the callback. Zero disables the
 * callback, non-zero enables the callback.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber, \p domain or \p
 * cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid);

/**
 * \brief Enable or disabled all callbacks for a specific domain.
 *
 * Enable or disabled all callbacks for a specific domain.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackEnabled(sub,
 * d, *) and cuptiEnableDomain(sub, d) are called concurrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in the
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber or \p domain is invalid
 */
CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain);

/**
 * \brief Enable or disable all callbacks in all domains.
 *
 * Enable or disable all callbacks in all domains.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, *) and cuptiEnableAllDomains(sub) are called concurrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in all
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is invalid
 */
CUptiResult CUPTIAPI cuptiEnableAllDomains(uint32_t enable,
                                           CUpti_SubscriberHandle subscriber);

/**
 * \brief Get the name of a callback for a specific domain and callback ID.
 *
 * Returns a pointer to the name c_string in \p **name.
 *
 * \note \b Names are available only for the DRIVER and RUNTIME domains.
 *
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param name Returns pointer to the name string on success, NULL otherwise
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p name is NULL, or if
 * \p domain or \p cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiGetCallbackName(CUpti_CallbackDomain domain,
                                          uint32_t cbid,
                                          const char **name);

/** @} */ /* END CUPTI_CALLBACK_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif  // file guard

/*
 * Copyright 2010-2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_EVENTS_H_)
#define _CUPTI_EVENTS_H_

// #include <cuda.h>
#include <string.h>
// #include <cuda_stdint.h>
// #include <cupti_result.h>

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_EVENT_API CUPTI Event API
 * Functions, types, and enums that implement the CUPTI Event API.
 *
 * \note The CUPTI event API from the header cupti_events.h is not supported on devices
 * with compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
 * This API is deprecated in CUDA 12.8 release and will be removed in a future CUDA release.
 * This is replaced by the host profiling API in the header cupti_profiler_host.h and
 * target profiling API in the header cupti_range_profiler.h which are supported on
 * devices with compute capability 7.0 and higher (i.e. Volta and later GPU architectures).
 *
 * @{
 */

/**
 * \brief ID for an event.
 *
 * An event represents a countable activity, action, or occurrence on
 * the device.
 */
typedef uint32_t CUpti_EventID;

/**
 * \brief ID for an event domain.
 *
 * ID for an event domain. An event domain represents a group of
 * related events. A device may have multiple instances of a domain,
 * indicating that the device can simultaneously record multiple
 * instances of each event within that domain.
 */
typedef uint32_t CUpti_EventDomainID;

/**
 * \brief A group of events.
 *
 * An event group is a collection of events that are managed
 * together. All events in an event group must belong to the same
 * domain.
 */
typedef void *CUpti_EventGroup;

/**
 * \brief Device class.
 *
 * Enumeration of device classes for device attribute
 * CUPTI_DEVICE_ATTR_DEVICE_CLASS.
 */
typedef enum {
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA              = 0,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO             = 1,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE            = 2,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA              = 3,
} CUpti_DeviceAttributeDeviceClass;

/**
 * \brief Device attributes.
 *
 * CUPTI device attributes. These attributes can be read using \ref
 * cuptiDeviceGetAttribute.
 */
typedef enum {
  /**
   * Number of event IDs for a device. Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_MAX_EVENT_ID                            = 1,
  /**
   * Number of event domain IDs for a device. Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID                     = 2,
  /**
   * Get global memory bandwidth in Kbytes/sec. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH                 = 3,
  /**
   * Get theoretical maximum number of instructions per cycle. Value
   * is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE                   = 4,
  /**
   * Get theoretical maximum number of single precision instructions
   * that can be executed per second. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION = 5,
  /**
   * Get number of frame buffers for device.  Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS                       = 6,
  /**
   * Get PCIE link rate in Mega bits/sec for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_LINK_RATE                          = 7,
  /**
   * Get PCIE link width for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH                         = 8,
  /**
   * Get PCIE generation for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_GEN                                = 9,
  /**
   * Get the class for the device. Value is a
   * CUpti_DeviceAttributeDeviceClass.
   */
  CUPTI_DEVICE_ATTR_DEVICE_CLASS                            = 10,
  /**
   * Get the peak single precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE                       = 11,
  /**
   * Get the peak double precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE                       = 12,
  /**
   * Get number of L2 units. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_L2_UNITS                           = 13,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_SHARED
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED = 14,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_L1
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 = 15,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_EQUAL
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL = 16,
  /**
   * Get the peak half precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE                       = 17,
  /**
   * Check if Nvlink is connected to device. Returns 1, if at least one
   * Nvlink is connected to the device, returns 0 otherwise.
   * Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_NVLINK_PRESENT                          = 18,
    /**
   * Check if Nvlink is present between GPU and CPU. Returns Bandwidth,
   * in Bytes/sec, if Nvlink is present, returns 0 otherwise.
   * Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW                       = 19,
  /**
   * Check if NVSwitch is present in the underlying topology.
   * Returns 1, if present, returns 0 otherwise.
   * Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT                        = 20,
  CUPTI_DEVICE_ATTR_FORCE_INT                               = 0x7fffffff,
} CUpti_DeviceAttribute;

/**
 * \brief Event domain attributes.
 *
 * Event domain attributes. Except where noted, all the attributes can
 * be read using either \ref cuptiDeviceGetEventDomainAttribute or
 * \ref cuptiEventDomainGetAttribute.
 */
typedef enum {
  /**
   * Event domain name. Value is a null terminated const c-string.
   */
  CUPTI_EVENT_DOMAIN_ATTR_NAME                 = 0,
  /**
   * Number of instances of the domain for which event counts will be
   * collected.  The domain may have additional instances that cannot
   * be profiled (see CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT).
   * Can be read only with \ref
   * cuptiDeviceGetEventDomainAttribute. Value is a uint32_t.
   */
  CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT       = 1,
  /**
   * Total number of instances of the domain, including instances that
   * cannot be profiled.  Use CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT
   * to get the number of instances that can be profiled. Can be read
   * only with \ref cuptiDeviceGetEventDomainAttribute. Value is a
   * uint32_t.
   */
  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT = 3,
  /**
   * Collection method used for events contained in the event domain.
   * Value is a \ref CUpti_EventCollectionMethod.
   */
  CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD    = 4,

  CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT      = 0x7fffffff,
} CUpti_EventDomainAttribute;

/**
 * \brief The collection method used for an event.
 *
 * The collection method indicates how an event is collected.
 */
typedef enum {
  /**
   * Event is collected using a hardware global performance monitor.
   */
  CUPTI_EVENT_COLLECTION_METHOD_PM                  = 0,
  /**
   * Event is collected using a hardware SM performance monitor.
   */
  CUPTI_EVENT_COLLECTION_METHOD_SM                  = 1,
  /**
   * Event is collected using software instrumentation.
   */
  CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED        = 2,
  /**
   * Event is collected using NvLink throughput counter method.
   */
  CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC           = 3,
  CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT           = 0x7fffffff
} CUpti_EventCollectionMethod;

/**
 * \brief Event group attributes.
 *
 * Event group attributes. These attributes can be read using \ref
 * cuptiEventGroupGetAttribute. Attributes marked [rw] can also be
 * written using \ref cuptiEventGroupSetAttribute.
 */
typedef enum {
  /**
   * The domain to which the event group is bound. This attribute is
   * set when the first event is added to the group.  Value is a
   * CUpti_EventDomainID.
   */
  CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID              = 0,
  /**
   * [rw] Profile all the instances of the domain for this
   * eventgroup. This feature can be used to get load balancing
   * across all instances of a domain. Value is an integer.
   */
  CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES = 1,
  /**
   * [rw] Reserved for user data.
   */
  CUPTI_EVENT_GROUP_ATTR_USER_DATA                    = 2,
  /**
   * Number of events in the group. Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS                   = 3,
  /**
   * Enumerates events in the group. Value is a pointer to buffer of
   * size sizeof(CUpti_EventID) * num_of_events in the eventgroup.
   * num_of_events can be queried using
   * CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS.
   */
  CUPTI_EVENT_GROUP_ATTR_EVENTS                       = 4,
  /**
   * Number of instances of the domain bound to this event group that
   * will be counted.  Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT               = 5,
  /**
   * Event group scope can be set to CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
   * CUPTI_EVENT_PROFILING_SCOPE_CONTEXT for an eventGroup, before
   * adding any event.
   * Sets the scope of eventgroup as CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
   * CUPTI_EVENT_PROFILING_SCOPE_CONTEXT when the scope of the events
   * that will be added is CUPTI_EVENT_PROFILING_SCOPE_BOTH.
   * If profiling scope of event is either
   * CUPTI_EVENT_PROFILING_SCOPE_DEVICE or CUPTI_EVENT_PROFILING_SCOPE_CONTEXT
   * then setting this attribute will not affect the default scope.
   * It is not allowed to add events of different scope to same eventgroup.
   * Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE               = 6,
  CUPTI_EVENT_GROUP_ATTR_FORCE_INT                     = 0x7fffffff,
} CUpti_EventGroupAttribute;

/**
* \brief Profiling scope for event.
*
* Profiling scope of event indicates if the event can be collected at context
* scope or device scope or both i.e. it can be collected at any of context or
* device scope.
*/
typedef enum {
  /**
   * Event is collected at context scope.
   */
  CUPTI_EVENT_PROFILING_SCOPE_CONTEXT                 = 0,
  /**
   * Event is collected at device scope.
   */
  CUPTI_EVENT_PROFILING_SCOPE_DEVICE                  = 1,
  /**
   * Event can be collected at device or context scope.
   * The scope can be set using \ref cuptiEventGroupSetAttribute API.
   */
  CUPTI_EVENT_PROFILING_SCOPE_BOTH                    = 2,
  CUPTI_EVENT_PROFILING_SCOPE_FORCE_INT               = 0x7fffffff
} CUpti_EventProfilingScope;

/**
 * \brief Event attributes.
 *
 * Event attributes. These attributes can be read using \ref
 * cuptiEventGetAttribute.
 */
typedef enum {
  /**
   * Event name. Value is a null terminated const c-string.
   */
  CUPTI_EVENT_ATTR_NAME              = 0,
  /**
   * Short description of event. Value is a null terminated const
   * c-string.
   */
  CUPTI_EVENT_ATTR_SHORT_DESCRIPTION = 1,
  /**
   * Long description of event. Value is a null terminated const
   * c-string.
   */
  CUPTI_EVENT_ATTR_LONG_DESCRIPTION  = 2,
  /**
   * Category of event. Value is CUpti_EventCategory.
   */
  CUPTI_EVENT_ATTR_CATEGORY          = 3,
  /**
   * Profiling scope of the events. It can be either device or context or both.
   * Value is a \ref CUpti_EventProfilingScope.
   */
  CUPTI_EVENT_ATTR_PROFILING_SCOPE   = 5,

  CUPTI_EVENT_ATTR_FORCE_INT         = 0x7fffffff,
} CUpti_EventAttribute;

/**
 * \brief Event collection modes.
 *
 * The event collection mode determines the period over which the
 * events within the enabled event groups will be collected.
 */
typedef enum {
  /**
   * Events are collected for the entire duration between the
   * cuptiEventGroupEnable and cuptiEventGroupDisable calls.
   * Event values are reset when the events are read.
   * For CUDA toolkit v6.0 and older this was the default mode.
   */
  CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS          = 0,
  /**
   * Events are collected only for the durations of kernel executions
   * that occur between the cuptiEventGroupEnable and
   * cuptiEventGroupDisable calls. Event collection begins when a
   * kernel execution begins, and stops when kernel execution
   * completes. Event values are reset to zero when each kernel
   * execution begins. If multiple kernel executions occur between the
   * cuptiEventGroupEnable and cuptiEventGroupDisable calls then the
   * event values must be read after each kernel launch if those
   * events need to be associated with the specific kernel launch.
   * Note that collection in this mode may significantly change the
   * overall performance characteristics of the application because
   * kernel executions that occur between the cuptiEventGroupEnable and
   * cuptiEventGroupDisable calls are serialized on the GPU.
   * This is the default mode from CUDA toolkit v6.5
   */
  CUPTI_EVENT_COLLECTION_MODE_KERNEL              = 1,
  CUPTI_EVENT_COLLECTION_MODE_FORCE_INT           = 0x7fffffff
} CUpti_EventCollectionMode;

/**
 * \brief An event category.
 *
 * Each event is assigned to a category that represents the general
 * type of the event. A event's category is accessed using \ref
 * cuptiEventGetAttribute and the CUPTI_EVENT_ATTR_CATEGORY attribute.
 */
typedef enum {
  /**
   * An instruction related event.
   */
  CUPTI_EVENT_CATEGORY_INSTRUCTION     = 0,
  /**
   * A memory related event.
   */
  CUPTI_EVENT_CATEGORY_MEMORY          = 1,
  /**
   * A cache related event.
   */
  CUPTI_EVENT_CATEGORY_CACHE           = 2,
  /**
   * A profile-trigger event.
   */
  CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER = 3,
  /**
   * A system event.
   */
  CUPTI_EVENT_CATEGORY_SYSTEM  = 4,
  CUPTI_EVENT_CATEGORY_FORCE_INT       = 0x7fffffff
} CUpti_EventCategory;

/**
 * \brief The overflow value for a CUPTI event.
 *
 * The CUPTI event value that indicates an overflow.
 */
#define CUPTI_EVENT_OVERFLOW ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

/**
 * \brief The value that indicates the event value is invalid
 */
#define CUPTI_EVENT_INVALID ((uint64_t)0xFFFFFFFFFFFFFFFEULL)

/**
 * \brief Flags for cuptiEventGroupReadEvent an
 * cuptiEventGroupReadAllEvents.
 *
 * Flags for \ref cuptiEventGroupReadEvent an \ref
 * cuptiEventGroupReadAllEvents.
 */
typedef enum {
  /**
   * No flags.
   */
  CUPTI_EVENT_READ_FLAG_NONE          = 0,
  CUPTI_EVENT_READ_FLAG_FORCE_INT     = 0x7fffffff,
} CUpti_ReadEventFlags;


/**
 * \brief A set of event groups.
 *
 * A set of event groups. When returned by \ref
 * cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
 * a set indicates that event groups that can be enabled at the same
 * time (i.e. all the events in the set can be collected
 * simultaneously).
 */
typedef struct {
  /**
   * The number of event groups in the set.
   */
  uint32_t numEventGroups;
  /**
   * An array of \p numEventGroups event groups.
   */
  CUpti_EventGroup *eventGroups;
} CUpti_EventGroupSet;

/**
 * \brief A set of event group sets.
 *
 * A set of event group sets. When returned by \ref
 * cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
 * a CUpti_EventGroupSets indicates the number of passes required to
 * collect all the events, and the event groups that should be
 * collected during each pass.
 */
typedef struct {
  /**
   * Number of event group sets.
   */
  uint32_t numSets;
  /**
   * An array of \p numSets event group sets.
   */
  CUpti_EventGroupSet *sets;
} CUpti_EventGroupSets;

/**
 * \brief Set the event collection mode.
 *
 * Set the event collection mode for a \p context.  The \p mode
 * controls the event collection behavior of all events in event
 * groups created in the \p context. This API is invalid in kernel
 * replay mode.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \param mode The event collection mode
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_OPERATION if called when replay mode is enabled
 * \retval CUPTI_ERROR_NOT_SUPPORTED if mode is not supported on the device
 */

CUptiResult CUPTIAPI cuptiSetEventCollectionMode(CUcontext context,
                                                 CUpti_EventCollectionMode mode);

/**
 * \brief Read a device attribute.
 *
 * Read a device attribute and return it in \p *value.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param attrib The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not a device attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiDeviceGetAttribute(CUdevice device,
                                             CUpti_DeviceAttribute attrib,
                                             size_t *valueSize,
                                             void *value);

/**
 * \brief Get the number of domains for a device.
 *
 * Returns the number of domains in \p numDomains for a device.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param numDomains Returns the number of domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numDomains is NULL
 */
CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains(CUdevice device,
                                                   uint32_t *numDomains);

/**
 * \brief Get the event domains for a device.
 *
 * Returns the event domains IDs in \p domainArray for a device.  The
 * size of the \p domainArray buffer is given by \p
 * *arraySizeBytes. The size of the \p domainArray buffer must be at
 * least \p numdomains * sizeof(CUpti_EventDomainID) or else all
 * domains will not be returned. The value returned in \p
 * *arraySizeBytes contains the number of bytes returned in \p
 * domainArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param arraySizeBytes The size of \p domainArray in bytes, and
 * returns the number of bytes written to \p domainArray
 * \param domainArray Returns the IDs of the event domains for the device
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p domainArray are NULL
 */
CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains(CUdevice device,
                                                 size_t *arraySizeBytes,
                                                 CUpti_EventDomainID *domainArray);

/**
 * \brief Read an event domain attribute.
 *
 * Returns an event domain attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param eventDomain ID of the event domain
 * \param attrib The event domain attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event domain attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiDeviceGetEventDomainAttribute(CUdevice device,
                                                        CUpti_EventDomainID eventDomain,
                                                        CUpti_EventDomainAttribute attrib,
                                                        size_t *valueSize,
                                                        void *value);

/**
 * \brief Get the number of event domains available on any device.
 *
 * Returns the total number of event domains available on any
 * CUDA-capable device.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param numDomains Returns the number of domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numDomains is NULL
 */
CUptiResult CUPTIAPI cuptiGetNumEventDomains(uint32_t *numDomains);

/**
 * \brief Get the event domains available on any device.
 *
 * Returns all the event domains available on any CUDA-capable device.
 * Event domain IDs are returned in \p domainArray. The size of the \p
 * domainArray buffer is given by \p *arraySizeBytes. The size of the
 * \p domainArray buffer must be at least \p numDomains *
 * sizeof(CUpti_EventDomainID) or all domains will not be
 * returned. The value returned in \p *arraySizeBytes contains the
 * number of bytes returned in \p domainArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param arraySizeBytes The size of \p domainArray in bytes, and
 * returns the number of bytes written to \p domainArray
 * \param domainArray Returns all the event domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p domainArray are NULL
 */
CUptiResult CUPTIAPI cuptiEnumEventDomains(size_t *arraySizeBytes,
                                           CUpti_EventDomainID *domainArray);

/**
 * \brief Read an event domain attribute.
 *
 * Returns an event domain attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param attrib The event domain attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event domain attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventDomainGetAttribute(CUpti_EventDomainID eventDomain,
                                                  CUpti_EventDomainAttribute attrib,
                                                  size_t *valueSize,
                                                  void *value);

/**
 * \brief Get number of events in a domain.
 *
 * Returns the number of events in \p numEvents for a domain.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param numEvents Returns the number of events in the domain
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numEvents is NULL
 */
CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents(CUpti_EventDomainID eventDomain,
                                                  uint32_t *numEvents);

/**
 * \brief Get the events in a domain.
 *
 * Returns the event IDs in \p eventArray for a domain.  The size of
 * the \p eventArray buffer is given by \p *arraySizeBytes. The size
 * of the \p eventArray buffer must be at least \p numdomainevents *
 * sizeof(CUpti_EventID) or else all events will not be returned. The
 * value returned in \p *arraySizeBytes contains the number of bytes
 * returned in \p eventArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param arraySizeBytes The size of \p eventArray in bytes, and
 * returns the number of bytes written to \p eventArray
 * \param eventArray Returns the IDs of the events in the domain
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or \p
 * eventArray are NULL
 */
CUptiResult CUPTIAPI cuptiEventDomainEnumEvents(CUpti_EventDomainID eventDomain,
                                                size_t *arraySizeBytes,
                                                CUpti_EventID *eventArray);

/**
 * \brief Get an event attribute.
 *
 * Returns an event attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param event ID of the event
 * \param attrib The event attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGetAttribute(CUpti_EventID event,
                                            CUpti_EventAttribute attrib,
                                            size_t *valueSize,
                                            void *value);

/**
 * \brief Find an event by name.
 *
 * Find an event by name and return the event ID in \p *event.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param eventName The name of the event to find
 * \param event Returns the ID of the found event or undefined if
 * unable to find the event
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_EVENT_NAME if unable to find an event
 * with name \p eventName. In this case \p *event is undefined
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventName or \p event are NULL
 */
CUptiResult CUPTIAPI cuptiEventGetIdFromName(CUdevice device,
                                             const char *eventName,
                                             CUpti_EventID *event);

/**
 * \brief Create a new event group for a context.
 *
 * Creates a new event group for \p context and returns the new group
 * in \p *eventGroup.
 * \note \p flags are reserved for future use and should be set to zero.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context for the event group
 * \param eventGroup Returns the new event group
 * \param flags Reserved - must be zero
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_OUT_OF_MEMORY
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupCreate(CUcontext context,
                                           CUpti_EventGroup *eventGroup,
                                           uint32_t flags);

/**
 * \brief Destroy an event group.
 *
 * Destroy an \p eventGroup and free its resources. An event group
 * cannot be destroyed if it is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group to destroy
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if the event group is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupDestroy(CUpti_EventGroup eventGroup);

/**
 * \brief Read an event group attribute.
 *
 * Read an event group attribute and return it in \p *value.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.).
 *
 * \param eventGroup The event group
 * \param attrib The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an eventgroup attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGroupGetAttribute(CUpti_EventGroup eventGroup,
                                                 CUpti_EventGroupAttribute attrib,
                                                 size_t *valueSize,
                                                 void *value);

/**
 * \brief Write an event group attribute.
 *
 * Write an event group attribute.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param attrib The attribute to write
 * \param valueSize The size, in bytes, of the value
 * \param value The attribute value to write
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event group attribute, or if
 * \p attrib is not a writable attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGroupSetAttribute(CUpti_EventGroup eventGroup,
                                                 CUpti_EventGroupAttribute attrib,
                                                 size_t valueSize,
                                                 void *value);

/**
 * \brief Add an event to an event group.
 *
 * Add an event to an event group. The event add can fail for a number of reasons:
 * \li The event group is enabled
 * \li The event does not belong to the same event domain as the
 * events that are already in the event group
 * \li Device limitations on the events that can belong to the same group
 * \li The event group is full
 *
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param event The event to add to the group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_OUT_OF_MEMORY
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p event belongs to a
 * different event domain than the events already in \p eventGroup, or
 * if a device limitation prevents \p event from being collected at
 * the same time as the events already in \p eventGroup
 * \retval CUPTI_ERROR_MAX_LIMIT_REACHED if \p eventGroup is full
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupAddEvent(CUpti_EventGroup eventGroup,
                                             CUpti_EventID event);

/**
 * \brief Remove an event from an event group.
 *
 * Remove \p event from the an event group. The event cannot be
 * removed if the event group is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param event The event to remove from the group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveEvent(CUpti_EventGroup eventGroup,
                                                CUpti_EventID event);

/**
 * \brief Remove all events from an event group.
 *
 * Remove all events from an event group. Events cannot be removed if
 * the event group is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveAllEvents(CUpti_EventGroup eventGroup);

/**
 * \brief Zero all the event counts in an event group.
 *
 * Zero all the event counts in an event group.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.).
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupResetAllEvents(CUpti_EventGroup eventGroup);

/**
 * \brief Enable an event group.
 *
 * Enable an event group. Enabling an event group zeros the value of
 * all the events in the group and then starts collection of those
 * events.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_NOT_READY if \p eventGroup does not contain any events
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p eventGroup cannot be
 * enabled due to other already enabled event groups
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 * \retval CUPTI_ERROR_HARDWARE_BUSY if another client is profiling
 * and hardware is busy
 */
CUptiResult CUPTIAPI cuptiEventGroupEnable(CUpti_EventGroup eventGroup);

/**
 * \brief Disable an event group.
 *
 * Disable an event group. Disabling an event group stops collection
 * of events contained in the group.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupDisable(CUpti_EventGroup eventGroup);

/**
 * \brief Read the value for an event in an event group.
 *
 * Read the value for an event in an event group. The event value is
 * returned in the \p eventValueBuffer buffer. \p
 * eventValueBufferSizeBytes indicates the size of the \p
 * eventValueBuffer buffer. The buffer must be at least sizeof(uint64)
 * if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set
 * on the group containing the event.  The buffer must be at least
 * (sizeof(uint64) * number of domain instances) if
 * ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is set on the
 * group.
 *
 * If any instance of an event counter overflows, the value returned
 * for that event instance will be ::CUPTI_EVENT_OVERFLOW.
 *
 * The only allowed value for \p flags is ::CUPTI_EVENT_READ_FLAG_NONE.
 *
 * Reading an event from a disabled event group is not allowed. After
 * being read, an event's value is reset to zero.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.). If \ref cuptiEventGroupResetAllEvents is
 * called simultaneously with this function, then returned event
 * values are undefined.
 *
 * \param eventGroup The event group
 * \param flags Flags controlling the reading mode
 * \param event The event to read
 * \param eventValueBufferSizeBytes The size of \p eventValueBuffer
 * in bytes, and returns the number of bytes written to \p
 * eventValueBuffer
 * \param eventValueBuffer Returns the event value(s)
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is disabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup, \p
 * eventValueBufferSizeBytes or \p eventValueBuffer is NULL
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if size of \p eventValueBuffer
 * is not sufficient
 */
CUptiResult CUPTIAPI cuptiEventGroupReadEvent(CUpti_EventGroup eventGroup,
                                              CUpti_ReadEventFlags flags,
                                              CUpti_EventID event,
                                              size_t *eventValueBufferSizeBytes,
                                              uint64_t *eventValueBuffer);

/**
 * \brief Read the values for all the events in an event group.
 *
 * Read the values for all the events in an event group. The event
 * values are returned in the \p eventValueBuffer buffer. \p
 * eventValueBufferSizeBytes indicates the size of \p
 * eventValueBuffer.  The buffer must be at least (sizeof(uint64) *
 * number of events in group) if
 * ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set on
 * the group containing the events.  The buffer must be at least
 * (sizeof(uint64) * number of domain instances * number of events in
 * group) if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is
 * set on the group.
 *
 * The data format returned in \p eventValueBuffer is:
 *    - domain instance 0: event0 event1 ... eventN
 *    - domain instance 1: event0 event1 ... eventN
 *    - ...
 *    - domain instance M: event0 event1 ... eventN
 *
 * The event order in \p eventValueBuffer is returned in \p
 * eventIdArray. The size of \p eventIdArray is specified in \p
 * eventIdArraySizeBytes. The size should be at least
 * (sizeof(CUpti_EventID) * number of events in group).
 *
 * If any instance of any event counter overflows, the value returned
 * for that event instance will be ::CUPTI_EVENT_OVERFLOW.
 *
 * The only allowed value for \p flags is ::CUPTI_EVENT_READ_FLAG_NONE.
 *
 * Reading events from a disabled event group is not allowed. After
 * being read, an event's value is reset to zero.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.). If \ref cuptiEventGroupResetAllEvents is
 * called simultaneously with this function, then returned event
 * values are undefined.
 *
 * \param eventGroup The event group
 * \param flags Flags controlling the reading mode
 * \param eventValueBufferSizeBytes The size of \p eventValueBuffer in
 * bytes, and returns the number of bytes written to \p
 * eventValueBuffer
 * \param eventValueBuffer Returns the event values
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes,
 * and returns the number of bytes written to \p eventIdArray
 * \param eventIdArray Returns the IDs of the events in the same order
 * as the values return in eventValueBuffer.
 * \param numEventIdsRead Returns the number of event IDs returned
 * in \p eventIdArray
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is disabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup, \p
 * eventValueBufferSizeBytes, \p eventValueBuffer, \p
 * eventIdArraySizeBytes, \p eventIdArray or \p numEventIdsRead is
 * NULL
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if size of \p eventValueBuffer
 * or \p eventIdArray is not sufficient
 */
CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents(CUpti_EventGroup       eventGroup,
                                                  CUpti_ReadEventFlags   flags,
                                                  size_t                 *eventValueBufferSizeBytes,
                                                  uint64_t               *eventValueBuffer,
                                                  size_t                 *eventIdArraySizeBytes,
                                                  CUpti_EventID          *eventIdArray,
                                                  size_t                 *numEventIdsRead);

/**
 * \brief For a set of events, get the grouping that indicates the
 * number of passes and the event groups necessary to collect the
 * events.
 *
 * The number of events that can be collected simultaneously varies by
 * device and by the type of the events. When events can be collected
 * simultaneously, they may need to be grouped into multiple event
 * groups because they are from different event domains. This function
 * takes a set of events and determines how many passes are required
 * to collect all those events, and which events can be collected
 * simultaneously in each pass.
 *
 * The CUpti_EventGroupSets returned in \p eventGroupPasses indicates
 * how many passes are required to collect the events with the \p
 * numSets field. Within each event group set, the \p sets array
 * indicates the event groups that should be collected on each pass.
 * \note \b Thread-safety: this function is thread safe, but client
 * must guard against another thread simultaneously destroying \p
 * context.
 *
 * \param context The context for event collection
 * \param eventIdArraySizeBytes Size of \p eventIdArray in bytes
 * \param eventIdArray Array of event IDs that need to be grouped
 * \param eventGroupPasses Returns a CUpti_EventGroupSets object that
 * indicates the number of passes required to collect the events and
 * the events to collect on each pass
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventIdArray or
 * \p eventGroupPasses is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetsCreate(CUcontext context,
                                               size_t eventIdArraySizeBytes,
                                               CUpti_EventID *eventIdArray,
                                               CUpti_EventGroupSets **eventGroupPasses);

/**
 * \brief Destroy a event group sets object.
 *
 * Destroy a CUpti_EventGroupSets object.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroupSets The object to destroy
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if any of the event groups
 * contained in the sets is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSets is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetsDestroy(CUpti_EventGroupSets *eventGroupSets);


/**
 * \brief Enable an event group set.
 *
 * Enable a set of event groups. Enabling a set of event groups zeros the value of
 * all the events in all the groups and then starts collection of those events.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroupSet The pointer to the event group set
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_NOT_READY if \p eventGroup does not contain any events
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p eventGroup cannot be
 * enabled due to other already enabled event groups
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSet is NULL
 * \retval CUPTI_ERROR_HARDWARE_BUSY if other client is profiling and hardware is
 * busy
 */
CUptiResult CUPTIAPI cuptiEventGroupSetEnable(CUpti_EventGroupSet *eventGroupSet);

/**
 * \brief Disable an event group set.
 *
 * Disable a set of event groups. Disabling a set of event groups
 * stops collection of events contained in the groups.
 * \note \b Thread-safety: this function is thread safe.
 * \note \b If this call fails, some of the event groups in the set may be disabled
 * and other event groups may remain enabled.
 *
 * \param eventGroupSet The pointer to the event group set
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSet is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetDisable(CUpti_EventGroupSet *eventGroupSet);

/**
 * \brief Enable kernel replay mode.
 *
 * Set profiling mode for the context to replay mode. In this mode,
 * any number of events can be collected in one run of the kernel. The
 * event collection mode will automatically switch to
 * CUPTI_EVENT_COLLECTION_MODE_KERNEL.  In this mode, \ref
 * cuptiSetEventCollectionMode will return
 * CUPTI_ERROR_INVALID_OPERATION.
 * \note \b Kernels might take longer to run if many events are enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiEnableKernelReplayMode(CUcontext context);

/**
 * \brief Disable kernel replay mode.
 *
 * Set profiling mode for the context to non-replay (default)
 * mode. Event collection mode will be set to
 * CUPTI_EVENT_COLLECTION_MODE_KERNEL.  All previously enabled
 * event groups and event group sets will be disabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiDisableKernelReplayMode(CUcontext context);

/**
 * \brief Function type for getting updates on kernel replay.
 *
 * \param kernelName The mangled kernel name
 * \param numReplaysDone Number of replays done so far
 * \param customData Pointer of any custom data passed in when subscribing
 */
typedef void (CUPTIAPI *CUpti_KernelReplayUpdateFunc)(
    const char *kernelName,
    int numReplaysDone,
    void *customData);

/**
 * \brief Subscribe to kernel replay updates.
 *
 * When subscribed, the function pointer passed in will be called each time a
 * kernel run is finished during kernel replay. Previously subscribed function
 * pointer will be replaced. Pass in NULL as the function pointer unsubscribes
 * the update.
 *
 * \param updateFunc The update function pointer
 * \param customData Pointer to any custom data
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiKernelReplaySubscribeUpdate(CUpti_KernelReplayUpdateFunc updateFunc, void *customData);

/** @} */ /* END CUPTI_EVENT_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_EVENTS_H_*/


/*
 * Copyright 2011-2024   NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_METRIC_H_)
#define _CUPTI_METRIC_H_

// #include <cuda.h>
#include <string.h>
// #include <cuda_stdint.h>
// #include <cupti_result.h>

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_METRIC_API CUPTI Metric API
 * Functions, types, and enums that implement the CUPTI Metric API.
 *
 * \note The CUPTI metric API from the header cupti_metrics.h is not supported on devices
 * with compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
 * This API is deprecated in CUDA 12.8 release and will be removed in a future CUDA release.
 * This is replaced by the host profiling API in the header cupti_profiler_host.h and
 * target profiling API in the header cupti_range_profiler.h which are supported on
 * devices with compute capability 7.0 and higher (i.e. Volta and later GPU architectures).
 *
 * @{
 */

/**
 * \brief ID for a metric.
 *
 * A metric provides a measure of some aspect of the device.
 */
typedef uint32_t CUpti_MetricID;

/**
 * \brief A metric category.
 *
 * Each metric is assigned to a category that represents the general
 * type of the metric. A metric's category is accessed using \ref
 * cuptiMetricGetAttribute and the CUPTI_METRIC_ATTR_CATEGORY
 * attribute.
 */
typedef enum {
  /**
   * A memory related metric.
   */
  CUPTI_METRIC_CATEGORY_MEMORY          = 0,
  /**
   * An instruction related metric.
   */
  CUPTI_METRIC_CATEGORY_INSTRUCTION     = 1,
  /**
   * A multiprocessor related metric.
   */
  CUPTI_METRIC_CATEGORY_MULTIPROCESSOR  = 2,
  /**
   * A cache related metric.
   */
  CUPTI_METRIC_CATEGORY_CACHE           = 3,
  /**
   * A texture related metric.
   */
  CUPTI_METRIC_CATEGORY_TEXTURE         = 4,
  /**
   *A Nvlink related metric.
   */
  CUPTI_METRIC_CATEGORY_NVLINK          = 5,
  /**
   *A PCIe related metric.
   */
  CUPTI_METRIC_CATEGORY_PCIE           = 6,
  CUPTI_METRIC_CATEGORY_FORCE_INT                         = 0x7fffffff,
} CUpti_MetricCategory;

/**
 * \brief A metric evaluation mode.
 *
 * A metric can be evaluated per hardware instance to know the load balancing
 * across instances of a domain or the metric can be evaluated in aggregate mode
 * when the events involved in metric evaluation are from different event
 * domains. It might be possible to evaluate some metrics in both
 * modes for convenience. A metric's evaluation mode is accessed using \ref
 * CUpti_MetricEvaluationMode and the CUPTI_METRIC_ATTR_EVALUATION_MODE
 * attribute.
 */
typedef enum {
  /**
   * If this bit is set, the metric can be profiled for each instance of the
   * domain. The event values passed to \ref cuptiMetricGetValue can contain
   * values for one instance of the domain. And \ref cuptiMetricGetValue can
   * be called for each instance.
   */
  CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE         = 1,
  /**
   * If this bit is set, the metric can be profiled over all instances. The
   * event values passed to \ref cuptiMetricGetValue can be aggregated values
   * of events for all instances of the domain.
   */
  CUPTI_METRIC_EVALUATION_MODE_AGGREGATE            = 1 << 1,
  CUPTI_METRIC_EVALUATION_MODE_FORCE_INT            = 0x7fffffff,
} CUpti_MetricEvaluationMode;

/**
 * \brief Kinds of metric values.
 *
 * Metric values can be one of several different kinds. Corresponding
 * to each kind is a member of the CUpti_MetricValue union. The metric
 * value returned by \ref cuptiMetricGetValue should be accessed using
 * the appropriate member of that union based on its value kind.
 */
typedef enum {
  /**
   * The metric value is a 64-bit double.
   */
  CUPTI_METRIC_VALUE_KIND_DOUBLE            = 0,
  /**
   * The metric value is a 64-bit unsigned integer.
   */
  CUPTI_METRIC_VALUE_KIND_UINT64            = 1,
  /**
   * The metric value is a percentage represented by a 64-bit
   * double. For example, 57.5% is represented by the value 57.5.
   */
  CUPTI_METRIC_VALUE_KIND_PERCENT           = 2,
  /**
   * The metric value is a throughput represented by a 64-bit
   * integer. The unit for throughput values is bytes/second.
   */
  CUPTI_METRIC_VALUE_KIND_THROUGHPUT        = 3,
  /**
   * The metric value is a 64-bit signed integer.
   */
  CUPTI_METRIC_VALUE_KIND_INT64             = 4,
  /**
   * The metric value is a utilization level, as represented by
   * CUpti_MetricValueUtilizationLevel.
   */
  CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL = 5,

  CUPTI_METRIC_VALUE_KIND_FORCE_INT  = 0x7fffffff
} CUpti_MetricValueKind;

/**
 * \brief Enumeration of utilization levels for metrics values of kind
 * CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL. Utilization values can
 * vary from IDLE (0) to MAX (10) but the enumeration only provides
 * specific names for a few values.
 */
typedef enum {
  CUPTI_METRIC_VALUE_UTILIZATION_IDLE      = 0,
  CUPTI_METRIC_VALUE_UTILIZATION_LOW       = 2,
  CUPTI_METRIC_VALUE_UTILIZATION_MID       = 5,
  CUPTI_METRIC_VALUE_UTILIZATION_HIGH      = 8,
  CUPTI_METRIC_VALUE_UTILIZATION_MAX       = 10,
  CUPTI_METRIC_VALUE_UTILIZATION_FORCE_INT = 0x7fffffff
} CUpti_MetricValueUtilizationLevel;

/**
 * \brief Metric attributes.
 *
 * Metric attributes describe properties of a metric. These attributes
 * can be read using \ref cuptiMetricGetAttribute.
 */
typedef enum {
  /**
   * Metric name. Value is a null terminated const c-string.
   */
  CUPTI_METRIC_ATTR_NAME              = 0,
  /**
   * Short description of metric. Value is a null terminated const c-string.
   */
  CUPTI_METRIC_ATTR_SHORT_DESCRIPTION = 1,
  /**
   * Long description of metric. Value is a null terminated const c-string.
   */
  CUPTI_METRIC_ATTR_LONG_DESCRIPTION  = 2,
  /**
   * Category of the metric. Value is of type CUpti_MetricCategory.
   */
  CUPTI_METRIC_ATTR_CATEGORY          = 3,
  /**
   * Value type of the metric. Value is of type CUpti_MetricValueKind.
   */
  CUPTI_METRIC_ATTR_VALUE_KIND          = 4,
  /**
   * Metric evaluation mode. Value is of type CUpti_MetricEvaluationMode.
   */
  CUPTI_METRIC_ATTR_EVALUATION_MODE     = 5,
  CUPTI_METRIC_ATTR_FORCE_INT         = 0x7fffffff,
} CUpti_MetricAttribute;

/**
 * \brief A metric value.
 *
 * Metric values can be one of several different kinds. Corresponding
 * to each kind is a member of the CUpti_MetricValue union. The metric
 * value returned by \ref cuptiMetricGetValue should be accessed using
 * the appropriate member of that union based on its value kind.
 */
typedef union {
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_DOUBLE.
   */
  double metricValueDouble;
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_UINT64.
   */
  uint64_t metricValueUint64;
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_INT64.
   */
  int64_t metricValueInt64;
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_PERCENT. For example, 57.5% is
   * represented by the value 57.5.
   */
  double metricValuePercent;
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_THROUGHPUT.  The unit for
   * throughput values is bytes/second.
   */
  uint64_t metricValueThroughput;
  /*
   * Value for CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL.
   */
  CUpti_MetricValueUtilizationLevel metricValueUtilizationLevel;
} CUpti_MetricValue;

/**
 * \brief Device class.
 *
 * Enumeration of device classes for metric property
 * CUPTI_METRIC_PROPERTY_DEVICE_CLASS.
 */
typedef enum {
  CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TESLA          = 0,
  CUPTI_METRIC_PROPERTY_DEVICE_CLASS_QUADRO         = 1,
  CUPTI_METRIC_PROPERTY_DEVICE_CLASS_GEFORCE        = 2,
  CUPTI_METRIC_PROPERTY_DEVICE_CLASS_TEGRA          = 3,
} CUpti_MetricPropertyDeviceClass;

/**
 * \brief Metric device properties.
 *
 * Metric device properties describe device properties which are needed for a metric.
 * Some of these properties can be collected using cuDeviceGetAttribute.
 */
typedef enum {
  /*
   * Number of multiprocessors on a device.  This can be collected
   * using value of \param CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT of
   * cuDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_MULTIPROCESSOR_COUNT,
  /*
   * Maximum number of warps on a multiprocessor. This can be
   * collected using ratio of value of \param
   * CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR and \param
   * CU_DEVICE_ATTRIBUTE_WARP_SIZE of cuDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_WARPS_PER_MULTIPROCESSOR,
  /*
   * GPU Time for kernel in ns. This should be profiled using CUPTI
   * Activity API.
   */
  CUPTI_METRIC_PROPERTY_KERNEL_GPU_TIME,
  /*
   * Clock rate for device in KHz.  This should be collected using
   * value of \param CU_DEVICE_ATTRIBUTE_CLOCK_RATE of
   * cuDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_CLOCK_RATE,
  /*
   * Number of Frame buffer units for device. This should be collected
   * using value of \param CUPTI_DEVICE_ATTRIBUTE_MAX_FRAME_BUFFERS of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_FRAME_BUFFER_COUNT,
  /*
   * Global memory bandwidth in KBytes/sec. This should be collected
   * using value of \param CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH
   * of cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_GLOBAL_MEMORY_BANDWIDTH,
  /*
   * PCIE link rate in Mega bits/sec. This should be collected using
   * value of \param CUPTI_DEVICE_ATTR_PCIE_LINK_RATE of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_PCIE_LINK_RATE,
  /*
   * PCIE link width for device. This should be collected using
   * value of \param CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_PCIE_LINK_WIDTH,
  /*
   * PCIE generation for device. This should be collected using
   * value of \param CUPTI_DEVICE_ATTR_PCIE_GEN of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_PCIE_GEN,
  /*
   * The device class. This should be collected using
   * value of \param CUPTI_DEVICE_ATTR_DEVICE_CLASS of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_DEVICE_CLASS,
  /*
   * Peak single precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE,
  /*
   * Peak double precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_FLOP_DP_PER_CYCLE,
  /*
   * Number of L2 units on a device. This can be collected
   * using value of \param CUPTI_DEVICE_ATTR_MAX_L2_UNITS of
   * cuDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_L2_UNITS,
  /*
   * Whether ECC support is enabled on the device. This can be
   * collected using value of \param CU_DEVICE_ATTRIBUTE_ECC_ENABLED of
   * cuDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_ECC_ENABLED,
  /*
   * Peak half precision floating point operations that
   * can be performed in one cycle by the device.
   * This should be collected using value of
   * \param CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_FLOP_HP_PER_CYCLE,
  /*
   * NVLINK Bandwitdh for device. This should be collected
   * using value of \param CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW of
   * cuptiDeviceGetAttribute.
   */
  CUPTI_METRIC_PROPERTY_GPU_CPU_NVLINK_BANDWIDTH,
} CUpti_MetricPropertyID;

/**
 * \brief Get the total number of metrics available on any device.
 *
 * Returns the total number of metrics available on any CUDA-capable
 * devices.
 *
 * \param numMetrics Returns the number of metrics
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numMetrics is NULL
*/
CUptiResult CUPTIAPI cuptiGetNumMetrics(uint32_t *numMetrics);

/**
 * \brief Get all the metrics available on any device.
 *
 * Returns the metric IDs in \p metricArray for all CUDA-capable
 * devices.  The size of the \p metricArray buffer is given by \p
 * *arraySizeBytes. The size of the \p metricArray buffer must be at
 * least \p numMetrics * sizeof(CUpti_MetricID) or all metric IDs will
 * not be returned. The value returned in \p *arraySizeBytes contains
 * the number of bytes returned in \p metricArray.
 *
 * \param arraySizeBytes The size of \p metricArray in bytes, and
 * returns the number of bytes written to \p metricArray
 * \param metricArray Returns the IDs of the metrics
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p metricArray are NULL
*/
CUptiResult CUPTIAPI cuptiEnumMetrics(size_t *arraySizeBytes,
                                      CUpti_MetricID *metricArray);

/**
 * \brief Get the number of metrics for a device.
 *
 * Returns the number of metrics available for a device.
 *
 * \param device The CUDA device
 * \param numMetrics Returns the number of metrics available for the
 * device
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numMetrics is NULL
 */
CUptiResult CUPTIAPI cuptiDeviceGetNumMetrics(CUdevice device,
                                              uint32_t *numMetrics);

/**
 * \brief Get the metrics for a device.
 *
 * Returns the metric IDs in \p metricArray for a device.  The size of
 * the \p metricArray buffer is given by \p *arraySizeBytes. The size
 * of the \p metricArray buffer must be at least \p numMetrics *
 * sizeof(CUpti_MetricID) or else all metric IDs will not be
 * returned. The value returned in \p *arraySizeBytes contains the
 * number of bytes returned in \p metricArray.
 *
 * \param device The CUDA device
 * \param arraySizeBytes The size of \p metricArray in bytes, and
 * returns the number of bytes written to \p metricArray
 * \param metricArray Returns the IDs of the metrics for the device
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p metricArray are NULL
 */
CUptiResult CUPTIAPI cuptiDeviceEnumMetrics(CUdevice device,
                                            size_t *arraySizeBytes,
                                            CUpti_MetricID *metricArray);

/**
 * \brief Get a metric attribute.
 *
 * Returns a metric attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 *
 * \param metric ID of the metric
 * \param attrib The metric attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not a metric attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiMetricGetAttribute(CUpti_MetricID metric,
                                             CUpti_MetricAttribute attrib,
                                             size_t *valueSize,
                                             void *value);

/**
 * \brief Find an metric by name.
 *
 * Find a metric by name and return the metric ID in \p *metric.
 *
 * \param device The CUDA device
 * \param metricName The name of metric to find
 * \param metric Returns the ID of the found metric or undefined if
 * unable to find the metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_METRIC_NAME if unable to find a metric
 * with name \p metricName. In this case \p *metric is undefined
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p metricName or \p
 * metric are NULL.
 */
CUptiResult CUPTIAPI cuptiMetricGetIdFromName(CUdevice device,
                                              const char *metricName,
                                              CUpti_MetricID *metric);

/**
 * \brief Get number of events required to calculate a metric.
 *
 * Returns the number of events in \p numEvents that are required to
 * calculate a metric.
 *
 * \param metric ID of the metric
 * \param numEvents Returns the number of events required for the metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numEvents is NULL
 */
CUptiResult CUPTIAPI cuptiMetricGetNumEvents(CUpti_MetricID metric,
                                             uint32_t *numEvents);

/**
 * \brief Get the events required to calculating a metric.
 *
 * Gets the event IDs in \p eventIdArray required to calculate a \p
 * metric. The size of the \p eventIdArray buffer is given by \p
 * *eventIdArraySizeBytes and must be at least \p numEvents *
 * sizeof(CUpti_EventID) or all events will not be returned. The value
 * returned in \p *eventIdArraySizeBytes contains the number of bytes
 * returned in \p eventIdArray.
 *
 * \param metric ID of the metric
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes,
 * and returns the number of bytes written to \p eventIdArray
 * \param eventIdArray Returns the IDs of the events required to
 * calculate \p metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventIdArraySizeBytes or \p
 * eventIdArray are NULL.
 */
CUptiResult CUPTIAPI cuptiMetricEnumEvents(CUpti_MetricID metric,
                                           size_t *eventIdArraySizeBytes,
                                           CUpti_EventID *eventIdArray);

/**
 * \brief Get number of properties required to calculate a metric.
 *
 * Returns the number of properties in \p numProp that are required to
 * calculate a metric.
 *
 * \param metric ID of the metric
 * \param numProp Returns the number of properties required for the
 * metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numProp is NULL
 */
CUptiResult CUPTIAPI cuptiMetricGetNumProperties(CUpti_MetricID metric,
                                                 uint32_t *numProp);

/**
 * \brief Get the properties required to calculating a metric.
 *
 * Gets the property IDs in \p propIdArray required to calculate a \p
 * metric. The size of the \p propIdArray buffer is given by \p
 * *propIdArraySizeBytes and must be at least \p numProp *
 * sizeof(CUpti_DeviceAttribute) or all properties will not be
 * returned. The value returned in \p *propIdArraySizeBytes contains
 * the number of bytes returned in \p propIdArray.
 *
 * \param metric ID of the metric
 * \param propIdArraySizeBytes The size of \p propIdArray in bytes,
 * and returns the number of bytes written to \p propIdArray
 * \param propIdArray Returns the IDs of the properties required to
 * calculate \p metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p propIdArraySizeBytes or \p
 * propIdArray are NULL.
 */
CUptiResult CUPTIAPI cuptiMetricEnumProperties(CUpti_MetricID metric,
                                               size_t *propIdArraySizeBytes,
                                               CUpti_MetricPropertyID *propIdArray);


/**
 * \brief For a metric get the groups of events that must be collected
 * in the same pass.
 *
 * For a metric get the groups of events that must be collected in the
 * same pass to ensure that the metric is calculated correctly. If the
 * events are not collected as specified then the metric value may be
 * inaccurate.
 *
 * The function returns NULL if a metric does not have any required
 * event group. In this case the events needed for the metric can be
 * grouped in any manner for collection.
 *
 * \param context The context for event collection
 * \param metric The metric ID
 * \param eventGroupSets Returns a CUpti_EventGroupSets object that
 * indicates the events that must be collected in the same pass to
 * ensure the metric is calculated correctly.  Returns NULL if no
 * grouping is required for metric
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 */
CUptiResult CUPTIAPI cuptiMetricGetRequiredEventGroupSets(CUcontext context,
                                                          CUpti_MetricID metric,
                                                          CUpti_EventGroupSets **eventGroupSets);

/**
 * \brief For a set of metrics, get the grouping that indicates the
 * number of passes and the event groups necessary to collect the
 * events required for those metrics.
 *
 * For a set of metrics, get the grouping that indicates the number of
 * passes and the event groups necessary to collect the events
 * required for those metrics.
 *
 * \see cuptiEventGroupSetsCreate for details on event group set
 * creation.
 *
 * \param context The context for event collection
 * \param metricIdArraySizeBytes Size of the metricIdArray in bytes
 * \param metricIdArray Array of metric IDs
 * \param eventGroupPasses Returns a CUpti_EventGroupSets object that
 * indicates the number of passes required to collect the events and
 * the events to collect on each pass
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p metricIdArray or
 * \p eventGroupPasses is NULL
 */
CUptiResult CUPTIAPI cuptiMetricCreateEventGroupSets(CUcontext context,
                                                     size_t metricIdArraySizeBytes,
                                                     CUpti_MetricID *metricIdArray,
                                                     CUpti_EventGroupSets **eventGroupPasses);

/**
 * \brief Calculate the value for a metric.
 *
 * Use the events collected for a metric to calculate the metric
 * value. Metric value evaluation depends on the evaluation mode
 * \ref CUpti_MetricEvaluationMode that the metric supports.
 * If a metric has evaluation mode as CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE,
 * then it assumes that the input event value is for one domain instance.
 * If a metric has evaluation mode as CUPTI_METRIC_EVALUATION_MODE_AGGREGATE,
 * it assumes that input event values are
 * normalized to represent all domain instances on a device. For the
 * most accurate metric collection, the events required for the metric
 * should be collected for all profiled domain instances. For example,
 * to collect all instances of an event, set the
 * CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES attribute on
 * the group containing the event to 1. The normalized value for the
 * event is then: (\p sum_event_values * \p totalInstanceCount) / \p
 * instanceCount, where \p sum_event_values is the summation of the
 * event values across all profiled domain instances, \p
 * totalInstanceCount is obtained from querying
 * CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT and \p instanceCount
 * is obtained from querying CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT (or
 * CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT).
 *
 * \param device The CUDA device that the metric is being calculated for
 * \param metric The metric ID
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes
 * \param eventIdArray The event IDs required to calculate \p metric
 * \param eventValueArraySizeBytes The size of \p eventValueArray in bytes
 * \param eventValueArray The normalized event values required to
 * calculate \p metric. The values must be order to match the order of
 * events in \p eventIdArray
 * \param timeDuration The duration over which the events were
 * collected, in ns
 * \param metricValue Returns the value for the metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_OPERATION
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if the
 * eventIdArray does not contain all the events needed for metric
 * \retval CUPTI_ERROR_INVALID_EVENT_VALUE if any of the
 * event values required for the metric is CUPTI_EVENT_OVERFLOW
 * \retval CUPTI_ERROR_INVALID_METRIC_VALUE if the computed metric value
 * cannot be represented in the metric's value type. For example,
 * if the metric value type is unsigned and the computed metric value is negative
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p metricValue,
 * \p eventIdArray or \p eventValueArray is NULL
 */
CUptiResult CUPTIAPI cuptiMetricGetValue(CUdevice device,
                                         CUpti_MetricID metric,
                                         size_t eventIdArraySizeBytes,
                                         CUpti_EventID *eventIdArray,
                                         size_t eventValueArraySizeBytes,
                                         uint64_t *eventValueArray,
                                         uint64_t timeDuration,
                                         CUpti_MetricValue *metricValue);

/**
 * \brief Calculate the value for a metric.
 *
 * Use the events and properties collected for a metric to calculate
 * the metric value. Metric value evaluation depends on the evaluation
 * mode \ref CUpti_MetricEvaluationMode that the metric supports.  If
 * a metric has evaluation mode as
 * CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE, then it assumes that the
 * input event value is for one domain instance.  If a metric has
 * evaluation mode as CUPTI_METRIC_EVALUATION_MODE_AGGREGATE, it
 * assumes that input event values are normalized to represent all
 * domain instances on a device. For the most accurate metric
 * collection, the events required for the metric should be collected
 * for all profiled domain instances. For example, to collect all
 * instances of an event, set the
 * CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES attribute on
 * the group containing the event to 1. The normalized value for the
 * event is then: (\p sum_event_values * \p totalInstanceCount) / \p
 * instanceCount, where \p sum_event_values is the summation of the
 * event values across all profiled domain instances, \p
 * totalInstanceCount is obtained from querying
 * CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT and \p instanceCount
 * is obtained from querying CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT (or
 * CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT).
 *
 * \param metric The metric ID
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes
 * \param eventIdArray The event IDs required to calculate \p metric
 * \param eventValueArraySizeBytes The size of \p eventValueArray in bytes
 * \param eventValueArray The normalized event values required to
 * calculate \p metric. The values must be order to match the order of
 * events in \p eventIdArray
 * \param propIdArraySizeBytes The size of \p propIdArray in bytes
 * \param propIdArray The metric property IDs required to calculate \p metric
 * \param propValueArraySizeBytes The size of \p propValueArray in bytes
 * \param propValueArray The metric property values required to
 * calculate \p metric. The values must be order to match the order of
 * metric properties in \p propIdArray
 * \param metricValue Returns the value for the metric
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_METRIC_ID
 * \retval CUPTI_ERROR_INVALID_OPERATION
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if the
 * eventIdArray does not contain all the events needed for metric
 * \retval CUPTI_ERROR_INVALID_EVENT_VALUE if any of the
 * event values required for the metric is CUPTI_EVENT_OVERFLOW
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the computed metric value
 * cannot be represented in the metric's value type. For example,
 * if the metric value type is unsigned and the computed metric value is negative
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p metricValue,
 * \p eventIdArray or \p eventValueArray is NULL
 */
CUptiResult CUPTIAPI cuptiMetricGetValue2(CUpti_MetricID metric,
                                          size_t eventIdArraySizeBytes,
                                          CUpti_EventID *eventIdArray,
                                          size_t eventValueArraySizeBytes,
                                          uint64_t *eventValueArray,
                                          size_t propIdArraySizeBytes,
                                          CUpti_MetricPropertyID *propIdArray,
                                          size_t propValueArraySizeBytes,
                                          uint64_t *propValueArray,
                                          CUpti_MetricValue *metricValue);

/** @} */ /* END CUPTI_METRIC_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_METRIC_H_*/


/*
 * Copyright 2011-2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_ACTIVITY_H_)
#define _CUPTI_ACTIVITY_H_

/**
 * Deprecated APIs and structures have been moved to the
 * header :doc: `cupti_activity_deprecated.h`, which is included at
 * the bottom of this file. Header cupti_activity.h contains
 * only the latest version of APIs and structures.
 */

// #include <cuda.h>
// #include <cupti_callbacks.h>
// #include <cupti_events.h>
// #include <cupti_metrics.h>
// #include <cupti_result.h>

#if defined(CUPTI_DIRECTIVE_SUPPORT)
// #include <Openacc/cupti_openacc.h>


/*
 * Copyright 2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// #include <cuda_stdint.h>

#if !defined(_CUPTI_OPENACC_H_)
#define _CUPTI_OPENACC_H_

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__LP64__)
#define CUPTILP64 1
#elif defined(_WIN64)
#define CUPTILP64 1
#else
#undef CUPTILP64
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \brief Initialize OpenACC support
 *
 * \param profRegister function of type acc_prof_reg as obtained from acc_register_library
 * \param profUnregister function of type acc_prof_reg as obtained from acc_register_library
 * \param profLookup function of type acc_prof_lookup as obtained from acc_register_library
 */
CUptiResult CUPTIAPI
cuptiOpenACCInitialize(void *profRegister, void *profUnregister, void *profLookup);

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_OPENACC_H_*/


// #include <Openmp/cupti_openmp.h>


/*
 * Copyright 2018 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// #include <cuda_stdint.h>
// #include "Openmp/omp-tools.h"

/*
 * include/50/omp-tools.h.var
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#ifndef __OMPT__
#define __OMPT__

/*****************************************************************************
 * system include files
 *****************************************************************************/

#include <stdint.h>
#include <stddef.h>

/*****************************************************************************
 * iteration macros
 *****************************************************************************/

#define FOREACH_OMPT_INQUIRY_FN(macro)      \
    macro (ompt_enumerate_states)           \
    macro (ompt_enumerate_mutex_impls)      \
                                            \
    macro (ompt_set_callback)               \
    macro (ompt_get_callback)               \
                                            \
    macro (ompt_get_state)                  \
                                            \
    macro (ompt_get_parallel_info)          \
    macro (ompt_get_task_info)              \
    macro (ompt_get_task_memory)            \
    macro (ompt_get_thread_data)            \
    macro (ompt_get_unique_id)              \
    macro (ompt_finalize_tool)              \
                                            \
    macro(ompt_get_num_procs)               \
    macro(ompt_get_num_places)              \
    macro(ompt_get_place_proc_ids)          \
    macro(ompt_get_place_num)               \
    macro(ompt_get_partition_place_nums)    \
    macro(ompt_get_proc_id)                 \
                                            \
    macro(ompt_get_target_info)             \
    macro(ompt_get_num_devices)

#define FOREACH_OMPT_STATE(macro)                                                                \
                                                                                                \
    /* first available state */                                                                 \
    macro (ompt_state_undefined, 0x102)      /* undefined thread state */                        \
                                                                                                \
    /* work states (0..15) */                                                                   \
    macro (ompt_state_work_serial, 0x000)    /* working outside parallel */                      \
    macro (ompt_state_work_parallel, 0x001)  /* working within parallel */                       \
    macro (ompt_state_work_reduction, 0x002) /* performing a reduction */                        \
                                                                                                \
    /* barrier wait states (16..31) */                                                          \
    macro (ompt_state_wait_barrier, 0x010)   /* waiting at a barrier */                          \
    macro (ompt_state_wait_barrier_implicit_parallel, 0x011)                                     \
                                            /* implicit barrier at the end of parallel region */\
    macro (ompt_state_wait_barrier_implicit_workshare, 0x012)                                    \
                                            /* implicit barrier at the end of worksharing */    \
    macro (ompt_state_wait_barrier_implicit, 0x013)  /* implicit barrier */                      \
    macro (ompt_state_wait_barrier_explicit, 0x014)  /* explicit barrier */                      \
                                                                                                \
    /* task wait states (32..63) */                                                             \
    macro (ompt_state_wait_taskwait, 0x020)  /* waiting at a taskwait */                         \
    macro (ompt_state_wait_taskgroup, 0x021) /* waiting at a taskgroup */                        \
                                                                                                \
    /* mutex wait states (64..127) */                                                           \
    macro (ompt_state_wait_mutex, 0x040)                                                         \
    macro (ompt_state_wait_lock, 0x041)      /* waiting for lock */                              \
    macro (ompt_state_wait_critical, 0x042)  /* waiting for critical */                          \
    macro (ompt_state_wait_atomic, 0x043)    /* waiting for atomic */                            \
    macro (ompt_state_wait_ordered, 0x044)   /* waiting for ordered */                           \
                                                                                                \
    /* target wait states (128..255) */                                                         \
    macro (ompt_state_wait_target, 0x080)        /* waiting for target region */                 \
    macro (ompt_state_wait_target_map, 0x081)    /* waiting for target data mapping operation */ \
    macro (ompt_state_wait_target_update, 0x082) /* waiting for target update operation */       \
                                                                                                \
    /* misc (256..511) */                                                                       \
    macro (ompt_state_idle, 0x100)           /* waiting for work */                              \
    macro (ompt_state_overhead, 0x101)       /* overhead excluding wait states */                \
                                                                                                \
    /* implementation-specific states (512..) */


#define FOREACH_KMP_MUTEX_IMPL(macro)                                                \
    macro (kmp_mutex_impl_none, 0)         /* unknown implementation */              \
    macro (kmp_mutex_impl_spin, 1)         /* based on spin */                       \
    macro (kmp_mutex_impl_queuing, 2)      /* based on some fair policy */           \
    macro (kmp_mutex_impl_speculative, 3)  /* based on HW-supported speculation */

#define FOREACH_OMPT_EVENT(macro)                                                                                        \
                                                                                                                         \
    /*--- Mandatory Events ---*/                                                                                         \
    macro (ompt_callback_thread_begin,      ompt_callback_thread_begin_t,       1) /* thread begin                    */ \
    macro (ompt_callback_thread_end,        ompt_callback_thread_end_t,         2) /* thread end                      */ \
                                                                                                                         \
    macro (ompt_callback_parallel_begin,    ompt_callback_parallel_begin_t,     3) /* parallel begin                  */ \
    macro (ompt_callback_parallel_end,      ompt_callback_parallel_end_t,       4) /* parallel end                    */ \
                                                                                                                         \
    macro (ompt_callback_task_create,       ompt_callback_task_create_t,        5) /* task begin                      */ \
    macro (ompt_callback_task_schedule,     ompt_callback_task_schedule_t,      6) /* task schedule                   */ \
    macro (ompt_callback_implicit_task,     ompt_callback_implicit_task_t,      7) /* implicit task                   */ \
                                                                                                                         \
    macro (ompt_callback_target,            ompt_callback_target_t,             8) /* target                          */ \
    macro (ompt_callback_target_data_op,    ompt_callback_target_data_op_t,     9) /* target data op                  */ \
    macro (ompt_callback_target_submit,     ompt_callback_target_submit_t,     10) /* target  submit                  */ \
                                                                                                                         \
    macro (ompt_callback_control_tool,      ompt_callback_control_tool_t,      11) /* control tool                    */ \
                                                                                                                         \
    macro (ompt_callback_device_initialize, ompt_callback_device_initialize_t, 12) /* device initialize               */ \
    macro (ompt_callback_device_finalize,   ompt_callback_device_finalize_t,   13) /* device finalize                 */ \
                                                                                                                         \
    macro (ompt_callback_device_load,       ompt_callback_device_load_t,       14) /* device load                     */ \
    macro (ompt_callback_device_unload,     ompt_callback_device_unload_t,     15) /* device unload                   */ \
                                                                                                                         \
    /* Optional Events */                                                                                                \
    macro (ompt_callback_sync_region_wait,  ompt_callback_sync_region_t,       16) /* sync region wait begin or end   */ \
                                                                                                                         \
    macro (ompt_callback_mutex_released,    ompt_callback_mutex_t,             17) /* mutex released                  */ \
                                                                                                                         \
    macro (ompt_callback_dependences,       ompt_callback_dependences_t,       18) /* report task dependences         */ \
    macro (ompt_callback_task_dependence,   ompt_callback_task_dependence_t,   19) /* report task dependence          */ \
                                                                                                                         \
    macro (ompt_callback_work,              ompt_callback_work_t,              20) /* task at work begin or end       */ \
                                                                                                                         \
    macro (ompt_callback_master,            ompt_callback_master_t,            21) /* task at master begin or end     */ \
                                                                                                                         \
    macro (ompt_callback_target_map,        ompt_callback_target_map_t,        22) /* target map                      */ \
                                                                                                                         \
    macro (ompt_callback_sync_region,       ompt_callback_sync_region_t,       23) /* sync region begin or end        */ \
                                                                                                                         \
    macro (ompt_callback_lock_init,         ompt_callback_mutex_acquire_t,     24) /* lock init                       */ \
    macro (ompt_callback_lock_destroy,      ompt_callback_mutex_t,             25) /* lock destroy                    */ \
                                                                                                                         \
    macro (ompt_callback_mutex_acquire,     ompt_callback_mutex_acquire_t,     26) /* mutex acquire                   */ \
    macro (ompt_callback_mutex_acquired,    ompt_callback_mutex_t,             27) /* mutex acquired                  */ \
                                                                                                                         \
    macro (ompt_callback_nest_lock,         ompt_callback_nest_lock_t,         28) /* nest lock                       */ \
                                                                                                                         \
    macro (ompt_callback_flush,             ompt_callback_flush_t,             29) /* after executing flush           */ \
                                                                                                                         \
    macro (ompt_callback_cancel,            ompt_callback_cancel_t,            30) /* cancel innermost binding region */ \
                                                                                                                         \
    macro (ompt_callback_reduction,         ompt_callback_sync_region_t,       31) /* reduction                       */ \
                                                                                                                         \
    macro (ompt_callback_dispatch,          ompt_callback_dispatch_t,          32) /* dispatch of work                */

/*****************************************************************************
 * implementation specific types
 *****************************************************************************/

typedef enum kmp_mutex_impl_t {
#define kmp_mutex_impl_macro(impl, code) impl = code,
    FOREACH_KMP_MUTEX_IMPL(kmp_mutex_impl_macro)
#undef kmp_mutex_impl_macro
} kmp_mutex_impl_t;

/*****************************************************************************
 * definitions generated from spec
 *****************************************************************************/

typedef enum ompt_callbacks_t {
  ompt_callback_thread_begin             = 1,
  ompt_callback_thread_end               = 2,
  ompt_callback_parallel_begin           = 3,
  ompt_callback_parallel_end             = 4,
  ompt_callback_task_create              = 5,
  ompt_callback_task_schedule            = 6,
  ompt_callback_implicit_task            = 7,
  ompt_callback_target                   = 8,
  ompt_callback_target_data_op           = 9,
  ompt_callback_target_submit            = 10,
  ompt_callback_control_tool             = 11,
  ompt_callback_device_initialize        = 12,
  ompt_callback_device_finalize          = 13,
  ompt_callback_device_load              = 14,
  ompt_callback_device_unload            = 15,
  ompt_callback_sync_region_wait         = 16,
  ompt_callback_mutex_released           = 17,
  ompt_callback_dependences              = 18,
  ompt_callback_task_dependence          = 19,
  ompt_callback_work                     = 20,
  ompt_callback_master                   = 21,
  ompt_callback_target_map               = 22,
  ompt_callback_sync_region              = 23,
  ompt_callback_lock_init                = 24,
  ompt_callback_lock_destroy             = 25,
  ompt_callback_mutex_acquire            = 26,
  ompt_callback_mutex_acquired           = 27,
  ompt_callback_nest_lock                = 28,
  ompt_callback_flush                    = 29,
  ompt_callback_cancel                   = 30,
  ompt_callback_reduction                = 31,
  ompt_callback_dispatch                 = 32
} ompt_callbacks_t;

typedef enum ompt_record_t {
  ompt_record_ompt               = 1,
  ompt_record_native             = 2,
  ompt_record_invalid            = 3
} ompt_record_t;

typedef enum ompt_record_native_t {
  ompt_record_native_info  = 1,
  ompt_record_native_event = 2
} ompt_record_native_t;

typedef enum ompt_set_result_t {
  ompt_set_error            = 0,
  ompt_set_never            = 1,
  ompt_set_impossible       = 2,
  ompt_set_sometimes        = 3,
  ompt_set_sometimes_paired = 4,
  ompt_set_always           = 5
} ompt_set_result_t;

typedef uint64_t ompt_id_t;

typedef uint64_t ompt_device_time_t;

typedef uint64_t ompt_buffer_cursor_t;

typedef enum ompt_thread_t {
  ompt_thread_initial                 = 1,
  ompt_thread_worker                  = 2,
  ompt_thread_other                   = 3,
  ompt_thread_unknown                 = 4
} ompt_thread_t;

typedef enum ompt_scope_endpoint_t {
  ompt_scope_begin                    = 1,
  ompt_scope_end                      = 2
} ompt_scope_endpoint_t;

typedef enum ompt_dispatch_t {
  ompt_dispatch_iteration             = 1,
  ompt_dispatch_section               = 2
} ompt_dispatch_t;

typedef enum ompt_sync_region_t {
  ompt_sync_region_barrier                = 1,
  ompt_sync_region_barrier_implicit       = 2,
  ompt_sync_region_barrier_explicit       = 3,
  ompt_sync_region_barrier_implementation = 4,
  ompt_sync_region_taskwait               = 5,
  ompt_sync_region_taskgroup              = 6,
  ompt_sync_region_reduction              = 7
} ompt_sync_region_t;

typedef enum ompt_target_data_op_t {
  ompt_target_data_alloc                = 1,
  ompt_target_data_transfer_to_device   = 2,
  ompt_target_data_transfer_from_device = 3,
  ompt_target_data_delete               = 4,
  ompt_target_data_associate            = 5,
  ompt_target_data_disassociate         = 6
} ompt_target_data_op_t;

typedef enum ompt_work_t {
  ompt_work_loop               = 1,
  ompt_work_sections           = 2,
  ompt_work_single_executor    = 3,
  ompt_work_single_other       = 4,
  ompt_work_workshare          = 5,
  ompt_work_distribute         = 6,
  ompt_work_taskloop           = 7
} ompt_work_t;

typedef enum ompt_mutex_t {
  ompt_mutex_lock                     = 1,
  ompt_mutex_test_lock                = 2,
  ompt_mutex_nest_lock                = 3,
  ompt_mutex_test_nest_lock           = 4,
  ompt_mutex_critical                 = 5,
  ompt_mutex_atomic                   = 6,
  ompt_mutex_ordered                  = 7
} ompt_mutex_t;

typedef enum ompt_native_mon_flag_t {
  ompt_native_data_motion_explicit    = 0x01,
  ompt_native_data_motion_implicit    = 0x02,
  ompt_native_kernel_invocation       = 0x04,
  ompt_native_kernel_execution        = 0x08,
  ompt_native_driver                  = 0x10,
  ompt_native_runtime                 = 0x20,
  ompt_native_overhead                = 0x40,
  ompt_native_idleness                = 0x80
} ompt_native_mon_flag_t;

typedef enum ompt_task_flag_t {
  ompt_task_initial                   = 0x00000001,
  ompt_task_implicit                  = 0x00000002,
  ompt_task_explicit                  = 0x00000004,
  ompt_task_target                    = 0x00000008,
  ompt_task_undeferred                = 0x08000000,
  ompt_task_untied                    = 0x10000000,
  ompt_task_final                     = 0x20000000,
  ompt_task_mergeable                 = 0x40000000,
  ompt_task_merged                    = 0x80000000
} ompt_task_flag_t;

typedef enum ompt_task_status_t {
  ompt_task_complete      = 1,
  ompt_task_yield         = 2,
  ompt_task_cancel        = 3,
  ompt_task_detach        = 4,
  ompt_task_early_fulfill = 5,
  ompt_task_late_fulfill  = 6,
  ompt_task_switch        = 7
} ompt_task_status_t;

typedef enum ompt_target_t {
  ompt_target                         = 1,
  ompt_target_enter_data              = 2,
  ompt_target_exit_data               = 3,
  ompt_target_update                  = 4
} ompt_target_t;

typedef enum ompt_parallel_flag_t {
  ompt_parallel_invoker_program = 0x00000001,
  ompt_parallel_invoker_runtime = 0x00000002,
  ompt_parallel_league          = 0x40000000,
  ompt_parallel_team            = 0x80000000
} ompt_parallel_flag_t;

typedef enum ompt_target_map_flag_t {
  ompt_target_map_flag_to             = 0x01,
  ompt_target_map_flag_from           = 0x02,
  ompt_target_map_flag_alloc          = 0x04,
  ompt_target_map_flag_release        = 0x08,
  ompt_target_map_flag_delete         = 0x10,
  ompt_target_map_flag_implicit       = 0x20
} ompt_target_map_flag_t;

typedef enum ompt_dependence_type_t {
  ompt_dependence_type_in              = 1,
  ompt_dependence_type_out             = 2,
  ompt_dependence_type_inout           = 3,
  ompt_dependence_type_mutexinoutset   = 4,
  ompt_dependence_type_source          = 5,
  ompt_dependence_type_sink            = 6
} ompt_dependence_type_t;

typedef enum ompt_cancel_flag_t {
  ompt_cancel_parallel       = 0x01,
  ompt_cancel_sections       = 0x02,
  ompt_cancel_loop           = 0x04,
  ompt_cancel_taskgroup      = 0x08,
  ompt_cancel_activated      = 0x10,
  ompt_cancel_detected       = 0x20,
  ompt_cancel_discarded_task = 0x40
} ompt_cancel_flag_t;

typedef uint64_t ompt_hwid_t;

typedef uint64_t ompt_wait_id_t;

typedef enum ompt_frame_flag_t {
  ompt_frame_runtime        = 0x00,
  ompt_frame_application    = 0x01,
  ompt_frame_cfa            = 0x10,
  ompt_frame_framepointer   = 0x20,
  ompt_frame_stackaddress   = 0x30
} ompt_frame_flag_t; 

typedef enum ompt_state_t {
  ompt_state_work_serial                      = 0x000,
  ompt_state_work_parallel                    = 0x001,
  ompt_state_work_reduction                   = 0x002,

  ompt_state_wait_barrier                     = 0x010,
  ompt_state_wait_barrier_implicit_parallel   = 0x011,
  ompt_state_wait_barrier_implicit_workshare  = 0x012,
  ompt_state_wait_barrier_implicit            = 0x013,
  ompt_state_wait_barrier_explicit            = 0x014,

  ompt_state_wait_taskwait                    = 0x020,
  ompt_state_wait_taskgroup                   = 0x021,

  ompt_state_wait_mutex                       = 0x040,
  ompt_state_wait_lock                        = 0x041,
  ompt_state_wait_critical                    = 0x042,
  ompt_state_wait_atomic                      = 0x043,
  ompt_state_wait_ordered                     = 0x044,

  ompt_state_wait_target                      = 0x080,
  ompt_state_wait_target_map                  = 0x081,
  ompt_state_wait_target_update               = 0x082,

  ompt_state_idle                             = 0x100,
  ompt_state_overhead                         = 0x101,
  ompt_state_undefined                        = 0x102
} ompt_state_t;

typedef uint64_t (*ompt_get_unique_id_t) (void);

typedef uint64_t ompd_size_t;

typedef uint64_t ompd_wait_id_t;

typedef uint64_t ompd_addr_t;
typedef int64_t  ompd_word_t;
typedef uint64_t ompd_seg_t;

typedef uint64_t ompd_device_t;

typedef uint64_t ompd_thread_id_t;

typedef enum ompd_scope_t {
  ompd_scope_global = 1,
  ompd_scope_address_space = 2,
  ompd_scope_thread = 3,
  ompd_scope_parallel = 4,
  ompd_scope_implicit_task = 5,
  ompd_scope_task = 6
} ompd_scope_t;

typedef uint64_t ompd_icv_id_t;

typedef enum ompd_rc_t {
  ompd_rc_ok = 0,
  ompd_rc_unavailable = 1,
  ompd_rc_stale_handle = 2,
  ompd_rc_bad_input = 3,
  ompd_rc_error = 4,
  ompd_rc_unsupported = 5,
  ompd_rc_needs_state_tracking = 6,
  ompd_rc_incompatible = 7,
  ompd_rc_device_read_error = 8,
  ompd_rc_device_write_error = 9,
  ompd_rc_nomem = 10,
} ompd_rc_t;

typedef void (*ompt_interface_fn_t) (void);

typedef ompt_interface_fn_t (*ompt_function_lookup_t) (
  const char *interface_function_name
);

typedef union ompt_data_t {
  uint64_t value;
  void *ptr;
} ompt_data_t;

typedef struct ompt_frame_t {
  ompt_data_t exit_frame;
  ompt_data_t enter_frame;
  int exit_frame_flags;
  int enter_frame_flags;
} ompt_frame_t;

typedef void (*ompt_callback_t) (void);

typedef void ompt_device_t;

typedef void ompt_buffer_t;

typedef void (*ompt_callback_buffer_request_t) (
  int device_num,
  ompt_buffer_t **buffer,
  size_t *bytes
);

typedef void (*ompt_callback_buffer_complete_t) (
  int device_num,
  ompt_buffer_t *buffer,
  size_t bytes,
  ompt_buffer_cursor_t begin,
  int buffer_owned
);

typedef void (*ompt_finalize_t) (
  ompt_data_t *tool_data
);

typedef int (*ompt_initialize_t) (
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t *tool_data
);

typedef struct ompt_start_tool_result_t {
  ompt_initialize_t initialize;
  ompt_finalize_t finalize;
  ompt_data_t tool_data;
} ompt_start_tool_result_t;

typedef struct ompt_record_abstract_t {
  ompt_record_native_t rclass;
  const char *type;
  ompt_device_time_t start_time;
  ompt_device_time_t end_time;
  ompt_hwid_t hwid;
} ompt_record_abstract_t;

typedef struct ompt_dependence_t {
  ompt_data_t variable;
  ompt_dependence_type_t dependence_type;
} ompt_dependence_t;

typedef int (*ompt_enumerate_states_t) (
  int current_state,
  int *next_state,
  const char **next_state_name
);

typedef int (*ompt_enumerate_mutex_impls_t) (
  int current_impl,
  int *next_impl,
  const char **next_impl_name
);

typedef ompt_set_result_t (*ompt_set_callback_t) (
  ompt_callbacks_t event,
  ompt_callback_t callback
);

typedef int (*ompt_get_callback_t) (
  ompt_callbacks_t event,
  ompt_callback_t *callback
);

typedef ompt_data_t *(*ompt_get_thread_data_t) (void);

typedef int (*ompt_get_num_procs_t) (void);

typedef int (*ompt_get_num_places_t) (void);

typedef int (*ompt_get_place_proc_ids_t) (
  int place_num,
  int ids_size,
  int *ids
);

typedef int (*ompt_get_place_num_t) (void);

typedef int (*ompt_get_partition_place_nums_t) (
  int place_nums_size,
  int *place_nums
);

typedef int (*ompt_get_proc_id_t) (void);

typedef int (*ompt_get_state_t) (
  ompt_wait_id_t *wait_id
);

typedef int (*ompt_get_parallel_info_t) (
  int ancestor_level,
  ompt_data_t **parallel_data,
  int *team_size
);

typedef int (*ompt_get_task_info_t) (
  int ancestor_level,
  int *flags,
  ompt_data_t **task_data,
  ompt_frame_t **task_frame,
  ompt_data_t **parallel_data,
  int *thread_num
);

typedef int (*ompt_get_task_memory_t)(
  void **addr,
  size_t *size,
  int block
);

typedef int (*ompt_get_target_info_t) (
  uint64_t *device_num,
  ompt_id_t *target_id,
  ompt_id_t *host_op_id
);

typedef int (*ompt_get_num_devices_t) (void);

typedef void (*ompt_finalize_tool_t) (void);

typedef int (*ompt_get_device_num_procs_t) (
  ompt_device_t *device
);

typedef ompt_device_time_t (*ompt_get_device_time_t) (
  ompt_device_t *device
);

typedef double (*ompt_translate_time_t) (
  ompt_device_t *device,
  ompt_device_time_t time
);

typedef ompt_set_result_t (*ompt_set_trace_ompt_t) (
  ompt_device_t *device,
  unsigned int enable,
  unsigned int etype
);

typedef ompt_set_result_t (*ompt_set_trace_native_t) (
  ompt_device_t *device,
  int enable,
  int flags
);

typedef int (*ompt_start_trace_t) (
  ompt_device_t *device,
  ompt_callback_buffer_request_t request,
  ompt_callback_buffer_complete_t complete
);

typedef int (*ompt_pause_trace_t) (
  ompt_device_t *device,
  int begin_pause
);

typedef int (*ompt_flush_trace_t) (
  ompt_device_t *device
);

typedef int (*ompt_stop_trace_t) (
  ompt_device_t *device
);

typedef int (*ompt_advance_buffer_cursor_t) (
  ompt_device_t *device,
  ompt_buffer_t *buffer,
  size_t size,
  ompt_buffer_cursor_t current,
  ompt_buffer_cursor_t *next
);

typedef ompt_record_t (*ompt_get_record_type_t) (
  ompt_buffer_t *buffer,
  ompt_buffer_cursor_t current
);

typedef void *(*ompt_get_record_native_t) (
  ompt_buffer_t *buffer,
  ompt_buffer_cursor_t current,
  ompt_id_t *host_op_id
);

typedef ompt_record_abstract_t *
(*ompt_get_record_abstract_t) (
  void *native_record
);

typedef void (*ompt_callback_thread_begin_t) (
  ompt_thread_t thread_type,
  ompt_data_t *thread_data
);

typedef struct ompt_record_thread_begin_t {
  ompt_thread_t thread_type;
} ompt_record_thread_begin_t;

typedef void (*ompt_callback_thread_end_t) (
  ompt_data_t *thread_data
);

typedef void (*ompt_callback_parallel_begin_t) (
  ompt_data_t *encountering_task_data,
  const ompt_frame_t *encountering_task_frame,
  ompt_data_t *parallel_data,
  unsigned int requested_parallelism,
  int flags,
  const void *codeptr_ra
);

typedef struct ompt_record_parallel_begin_t {
  ompt_id_t encountering_task_id;
  ompt_id_t parallel_id;
  unsigned int requested_parallelism;
  int flags;
  const void *codeptr_ra;
} ompt_record_parallel_begin_t;

typedef void (*ompt_callback_parallel_end_t) (
  ompt_data_t *parallel_data,
  ompt_data_t *encountering_task_data,
  int flags,
  const void *codeptr_ra
);

typedef struct ompt_record_parallel_end_t {
  ompt_id_t parallel_id;
  ompt_id_t encountering_task_id;
  int flags;
  const void *codeptr_ra;
} ompt_record_parallel_end_t;

typedef void (*ompt_callback_work_t) (
  ompt_work_t wstype,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  uint64_t count,
  const void *codeptr_ra
);

typedef struct ompt_record_work_t {
  ompt_work_t wstype;
  ompt_scope_endpoint_t endpoint;
  ompt_id_t parallel_id;
  ompt_id_t task_id;
  uint64_t count;
  const void *codeptr_ra;
} ompt_record_work_t;

typedef void (*ompt_callback_dispatch_t) (
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  ompt_dispatch_t kind,
  ompt_data_t instance 
);

typedef struct ompt_record_dispatch_t {
  ompt_id_t parallel_id;
  ompt_id_t task_id;
  ompt_dispatch_t kind;
  ompt_data_t instance; 
} ompt_record_dispatch_t;

typedef void (*ompt_callback_task_create_t) (
  ompt_data_t *encountering_task_data,
  const ompt_frame_t *encountering_task_frame,
  ompt_data_t *new_task_data,
  int flags,
  int has_dependences,
  const void *codeptr_ra
);

typedef struct ompt_record_task_create_t {
  ompt_id_t encountering_task_id;
  ompt_id_t new_task_id;
  int flags;
  int has_dependences;
  const void *codeptr_ra;
} ompt_record_task_create_t;

typedef void (*ompt_callback_dependences_t) (
  ompt_data_t *task_data,
  const ompt_dependence_t *deps,
  int ndeps
);

typedef struct ompt_record_dependences_t {
  ompt_id_t task_id;
  ompt_dependence_t dep;
  int ndeps;
} ompt_record_dependences_t;

typedef void (*ompt_callback_task_dependence_t) (
  ompt_data_t *src_task_data,
  ompt_data_t *sink_task_data
);

typedef struct ompt_record_task_dependence_t {
  ompt_id_t src_task_id;
  ompt_id_t sink_task_id;
} ompt_record_task_dependence_t;

typedef void (*ompt_callback_task_schedule_t) (
  ompt_data_t *prior_task_data,
  ompt_task_status_t prior_task_status,
  ompt_data_t *next_task_data
);

typedef struct ompt_record_task_schedule_t {
  ompt_id_t prior_task_id;
  ompt_task_status_t prior_task_status;
  ompt_id_t next_task_id;
} ompt_record_task_schedule_t;

typedef void (*ompt_callback_implicit_task_t) (
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  unsigned int actual_parallelism,
  unsigned int index,
  int flags
);

typedef struct ompt_record_implicit_task_t {
  ompt_scope_endpoint_t endpoint;
  ompt_id_t parallel_id;
  ompt_id_t task_id;
  unsigned int actual_parallelism;
  unsigned int index;
  int flags;
} ompt_record_implicit_task_t;

typedef void (*ompt_callback_master_t) (
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra
);

typedef struct ompt_record_master_t {
  ompt_scope_endpoint_t endpoint;
  ompt_id_t parallel_id;
  ompt_id_t task_id;
  const void *codeptr_ra;
} ompt_record_master_t;

typedef void (*ompt_callback_sync_region_t) (
  ompt_sync_region_t kind,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra
);

typedef struct ompt_record_sync_region_t {
  ompt_sync_region_t kind;
  ompt_scope_endpoint_t endpoint;
  ompt_id_t parallel_id;
  ompt_id_t task_id;
  const void *codeptr_ra;
} ompt_record_sync_region_t;

typedef void (*ompt_callback_mutex_acquire_t) (
  ompt_mutex_t kind,
  unsigned int hint,
  unsigned int impl,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra
);

typedef struct ompt_record_mutex_acquire_t {
  ompt_mutex_t kind;
  unsigned int hint;
  unsigned int impl;
  ompt_wait_id_t wait_id;
  const void *codeptr_ra;
} ompt_record_mutex_acquire_t;

typedef void (*ompt_callback_mutex_t) (
  ompt_mutex_t kind,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra
);

typedef struct ompt_record_mutex_t {
  ompt_mutex_t kind;
  ompt_wait_id_t wait_id;
  const void *codeptr_ra;
} ompt_record_mutex_t;

typedef void (*ompt_callback_nest_lock_t) (
  ompt_scope_endpoint_t endpoint,
  ompt_wait_id_t wait_id,
  const void *codeptr_ra
);

typedef struct ompt_record_nest_lock_t {
  ompt_scope_endpoint_t endpoint;
  ompt_wait_id_t wait_id;
  const void *codeptr_ra;
} ompt_record_nest_lock_t;

typedef void (*ompt_callback_flush_t) (
  ompt_data_t *thread_data,
  const void *codeptr_ra
);

typedef struct ompt_record_flush_t {
  const void *codeptr_ra;
} ompt_record_flush_t;

typedef void (*ompt_callback_cancel_t) (
  ompt_data_t *task_data,
  int flags,
  const void *codeptr_ra
);

typedef struct ompt_record_cancel_t {
  ompt_id_t task_id;
  int flags;
  const void *codeptr_ra;
} ompt_record_cancel_t;

typedef void (*ompt_callback_device_initialize_t) (
  int device_num,
  const char *type,
  ompt_device_t *device,
  ompt_function_lookup_t lookup,
  const char *documentation
);

typedef void (*ompt_callback_device_finalize_t) (
  int device_num
);

typedef void (*ompt_callback_device_load_t) (
  int device_num,
  const char *filename,
  int64_t offset_in_file,
  void *vma_in_file,
  size_t bytes,
  void *host_addr,
  void *device_addr,
  uint64_t module_id
);

typedef void (*ompt_callback_device_unload_t) (
  int device_num,
  uint64_t module_id
);

typedef void (*ompt_callback_target_data_op_t) (
  ompt_id_t target_id,
  ompt_id_t host_op_id,
  ompt_target_data_op_t optype,
  void *src_addr,
  int src_device_num,
  void *dest_addr,
  int dest_device_num,
  size_t bytes,
  const void *codeptr_ra
);

typedef struct ompt_record_target_data_op_t {
  ompt_id_t host_op_id;
  ompt_target_data_op_t optype;
  void *src_addr;
  int src_device_num;
  void *dest_addr;
  int dest_device_num;
  size_t bytes;
  ompt_device_time_t end_time;
  const void *codeptr_ra;
} ompt_record_target_data_op_t;

typedef void (*ompt_callback_target_t) (
  ompt_target_t kind,
  ompt_scope_endpoint_t endpoint,
  int device_num,
  ompt_data_t *task_data,
  ompt_id_t target_id,
  const void *codeptr_ra
);

typedef struct ompt_record_target_t {
  ompt_target_t kind;
  ompt_scope_endpoint_t endpoint;
  int device_num;
  ompt_id_t task_id;
  ompt_id_t target_id;
  const void *codeptr_ra;
} ompt_record_target_t;

typedef void (*ompt_callback_target_map_t) (
  ompt_id_t target_id,
  unsigned int nitems,
  void **host_addr,
  void **device_addr,
  size_t *bytes,
  unsigned int *mapping_flags,
  const void *codeptr_ra
);

typedef struct ompt_record_target_map_t {
  ompt_id_t target_id;
  unsigned int nitems;
  void **host_addr;
  void **device_addr;
  size_t *bytes;
  unsigned int *mapping_flags;
  const void *codeptr_ra;
} ompt_record_target_map_t;

typedef void (*ompt_callback_target_submit_t) (
  ompt_id_t target_id,
  ompt_id_t host_op_id,
  unsigned int requested_num_teams
);

typedef struct ompt_record_target_kernel_t {
  ompt_id_t host_op_id;
  unsigned int requested_num_teams;
  unsigned int granted_num_teams;
  ompt_device_time_t end_time;
} ompt_record_target_kernel_t;

typedef int (*ompt_callback_control_tool_t) (
  uint64_t command,
  uint64_t modifier,
  void *arg,
  const void *codeptr_ra
);

typedef struct ompt_record_control_tool_t {
  uint64_t command;
  uint64_t modifier;
  const void *codeptr_ra;
} ompt_record_control_tool_t;

typedef struct ompd_address_t {
  ompd_seg_t segment;
  ompd_addr_t address;
} ompd_address_t;

typedef struct ompd_frame_info_t {
  ompd_address_t frame_address;
  ompd_word_t frame_flag;
} ompd_frame_info_t;

typedef struct _ompd_aspace_handle ompd_address_space_handle_t;
typedef struct _ompd_thread_handle ompd_thread_handle_t;
typedef struct _ompd_parallel_handle ompd_parallel_handle_t;
typedef struct _ompd_task_handle ompd_task_handle_t;

typedef struct _ompd_aspace_cont ompd_address_space_context_t;
typedef struct _ompd_thread_cont ompd_thread_context_t;

typedef struct ompd_device_type_sizes_t {
  uint8_t sizeof_char;
  uint8_t sizeof_short;
  uint8_t sizeof_int;
  uint8_t sizeof_long;
  uint8_t sizeof_long_long;
  uint8_t sizeof_pointer;
} ompd_device_type_sizes_t;

typedef struct ompt_record_ompt_t {
  ompt_callbacks_t type;
  ompt_device_time_t time;
  ompt_id_t thread_id;
  ompt_id_t target_id;
  union {
    ompt_record_thread_begin_t thread_begin;
    ompt_record_parallel_begin_t parallel_begin;
    ompt_record_parallel_end_t parallel_end;
    ompt_record_work_t work;
    ompt_record_dispatch_t dispatch;
    ompt_record_task_create_t task_create;
    ompt_record_dependences_t dependences;
    ompt_record_task_dependence_t task_dependence;
    ompt_record_task_schedule_t task_schedule;
    ompt_record_implicit_task_t implicit_task;
    ompt_record_master_t master;
    ompt_record_sync_region_t sync_region;
    ompt_record_mutex_acquire_t mutex_acquire;
    ompt_record_mutex_t mutex;
    ompt_record_nest_lock_t nest_lock;
    ompt_record_flush_t flush;
    ompt_record_cancel_t cancel;
    ompt_record_target_t target;
    ompt_record_target_data_op_t target_data_op;
    ompt_record_target_map_t target_map;
    ompt_record_target_kernel_t target_kernel;
    ompt_record_control_tool_t control_tool;
  } record;
} ompt_record_ompt_t;

typedef ompt_record_ompt_t *(*ompt_get_record_ompt_t) (
  ompt_buffer_t *buffer,
  ompt_buffer_cursor_t current
);

#define ompt_id_none 0
#define ompt_data_none {0}
#define ompt_time_none 0
#define ompt_hwid_none 0
#define ompt_addr_none ~0
#define ompt_mutex_impl_none 0
#define ompt_wait_id_none 0

#define ompd_segment_none 0

#endif /* __OMPT__ */


#if !defined(_CUPTI_OPENMP_H_)
#define _CUPTI_OPENMP_H_

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__LP64__)
#define CUPTILP64 1
#elif defined(_WIN64)
#define CUPTILP64 1
#else
#undef CUPTILP64
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \brief Initialize OPENMP support (deprecated, used before OpenMP 5.0)
 *
 */
int CUPTIAPI cuptiOpenMpInitialize(ompt_function_lookup_t ompt_fn_lookup, const char *runtime_version, unsigned int ompt_version);

/**
 * \brief Initialize OPENMP support
 *
 */
int CUPTIAPI cuptiOpenMpInitialize_v2(ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t *tool_data);

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_OPENMP_H_*/

#endif

// #include <cupti_common.h>

/*
 * Copyright 2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 
#if !defined(__CUPTI_COMMON_H__)
#define __CUPTI_COMMON_H__

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#ifndef CUPTIUTILAPI
#ifdef _WIN32
#define CUPTIUTILAPI __stdcall
#else
#define CUPTIUTILAPI
#endif
#endif

#if defined(__LP64__)
#define CUPTILP64 1
#elif defined(_WIN64)
#define CUPTILP64 1
#else
#undef CUPTILP64
#endif

#define ACTIVITY_RECORD_ALIGNMENT 8
#if defined(_WIN32) // Windows 32- and 64-bit
#define START_PACKED_ALIGNMENT __pragma(pack(push,1)) // exact fit - no padding
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_RECORD_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__) // GCC
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__ ((__packed__)) __attribute__ ((aligned (ACTIVITY_RECORD_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else // all other compilers
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#endif /*__CUPTI_COMMON_H__*/


#define CUPTI_UNIFIED_MEMORY_CPU_DEVICE_ID ((uint32_t) 0xFFFFFFFFU)
#define CUPTI_INVALID_CONTEXT_ID ((uint32_t) 0xFFFFFFFFU)
#define CUPTI_INVALID_STREAM_ID ((uint32_t) 0xFFFFFFFFU)
#define CUPTI_INVALID_CHANNEL_ID ((uint32_t) 0xFFFFFFFFU)

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

#define invalidNumaId ((uint32_t) 0xFFFFFFFF)

/**
 * \defgroup CUPTI_ACTIVITY_API CUPTI Activity API
 * Functions, types, and enums that implement the CUPTI Activity API.
 * @{
 */

/**
 * \brief The kinds of activity records.
 *
 * Each activity record kind represents information about a GPU or an
 * activity occurring on a CPU or GPU. Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 * \see CUpti_Activity
 * \see CUpti_ActivityAPI
 * \see CUpti_ActivityContext
 * \see CUpti_ActivityContext2
 * \see CUpti_ActivityContext3
 * \see CUpti_ActivityDevice
 * \see CUpti_ActivityDevice2
 * \see CUpti_ActivityDevice3
 * \see CUpti_ActivityDevice4
 * \see CUpti_ActivityDeviceAttribute
 * \see CUpti_ActivityEvent
 * \see CUpti_ActivityEventInstance
 * \see CUpti_ActivityKernel
 * \see CUpti_ActivityKernel2
 * \see CUpti_ActivityKernel3
 * \see CUpti_ActivityKernel4
 * \see CUpti_ActivityKernel5
 * \see CUpti_ActivityKernel6
 * \see CUpti_ActivityKernel7
 * \see CUpti_ActivityKernel8
 * \see CUpti_ActivityKernel9
 * \see CUpti_ActivityCdpKernel
 * \see CUpti_ActivityPreemption
 * \see CUpti_ActivityMemcpy
 * \see CUpti_ActivityMemcpy3
 * \see CUpti_ActivityMemcpy4
 * \see CUpti_ActivityMemcpy5
 * \see CUpti_ActivityMemcpy6
 * \see CUpti_ActivityMemcpyPtoP
 * \see CUpti_ActivityMemcpyPtoP2
 * \see CUpti_ActivityMemcpyPtoP3
 * \see CUpti_ActivityMemcpyPtoP4
 * \see CUpti_ActivityMemset
 * \see CUpti_ActivityMemset2
 * \see CUpti_ActivityMemset3
 * \see CUpti_ActivityMemset4
 * \see CUpti_ActivityMemory
 * \see CUpti_ActivityMemory2
 * \see CUpti_ActivityMemory3
 * \see CUpti_ActivityMemory4
 * \see CUpti_ActivityMemoryPool
 * \see CUpti_ActivityMemoryPool2
 * \see CUpti_ActivityMetric
 * \see CUpti_ActivityMetricInstance
 * \see CUpti_ActivityName
 * \see CUpti_ActivityMarker
 * \see CUpti_ActivityMarker2
 * \see CUpti_ActivityMarkerData
 * \see CUpti_ActivitySourceLocator
 * \see CUpti_ActivityGlobalAccess
 * \see CUpti_ActivityGlobalAccess2
 * \see CUpti_ActivityGlobalAccess3
 * \see CUpti_ActivityBranch
 * \see CUpti_ActivityBranch2
 * \see CUpti_ActivityOverhead3
 * \see CUpti_ActivityEnvironment
 * \see CUpti_ActivityInstructionExecution
 * \see CUpti_ActivityUnifiedMemoryCounter
 * \see CUpti_ActivityFunction
 * \see CUpti_ActivityModule
 * \see CUpti_ActivitySharedAccess
 * \see CUpti_ActivityPCSampling
 * \see CUpti_ActivityPCSampling2
 * \see CUpti_ActivityPCSampling3
 * \see CUpti_ActivityPCSamplingRecordInfo
 * \see CUpti_ActivityCudaEvent2
 * \see CUpti_ActivityStream
 * \see CUpti_ActivitySynchronization2
 * \see CUpti_ActivityInstructionCorrelation
 * \see CUpti_ActivityExternalCorrelation
 * \see CUpti_ActivityUnifiedMemoryCounter3
 * \see CUpti_ActivityOpenAccData
 * \see CUpti_ActivityOpenAccLaunch
 * \see CUpti_ActivityOpenAccOther
 * \see CUpti_ActivityOpenMp
 * \see CUpti_ActivityNvLink
 * \see CUpti_ActivityNvLink2
 * \see CUpti_ActivityNvLink3
 * \see CUpti_ActivityNvLink4
 * \see CUpti_ActivityPcie
 * \see CUpti_ActivityConfidentialComputeRotation
 */

typedef enum {
  /**
   * The activity record is invalid.
   */
  CUPTI_ACTIVITY_KIND_INVALID  = 0,

  /**
   * A host<->host, host<->device, or device<->device memory copy.
   * For peer to peer memory copy, use the kind CUPTI_ACTIVITY_KIND_MEMCPY2.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemcpy6.
   */
  CUPTI_ACTIVITY_KIND_MEMCPY   = 1,

  /**
   * A memory set executing on the GPU. The corresponding activity
   * record structure is \ref CUpti_ActivityMemset4.
   */
  CUPTI_ACTIVITY_KIND_MEMSET   = 2,

  /**
   * A kernel executing on the GPU. This activity kind may significantly change
   * the overall performance characteristics of the application because all
   * kernel executions are serialized on the GPU. Other activity kind for kernel
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL doesn't break kernel concurrency.
   * The corresponding activity record structure is \ref CUpti_ActivityKernel9.
   */
  CUPTI_ACTIVITY_KIND_KERNEL   = 3,

  /**
   * A CUDA driver API function execution. The corresponding activity
   * record structure is \ref CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_DRIVER   = 4,

  /**
   * A CUDA runtime API function execution. The corresponding activity
   * record structure is \ref CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_RUNTIME  = 5,

  /**
   * A performance counter (aka event) value. The corresponding activity record
   * structure is \ref CUpti_ActivityEvent. This activity cannot be directly
   * enabled or disabled. Information collected using the Event API.
   * can be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_EVENT    = 6,

  /**
   * A performance metric value. The corresponding activity record structure is
   * \ref CUpti_ActivityMetric. This activity cannot be directly
   * enabled or disabled. Information collected using the Metric API.
   * can be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_METRIC   = 7,

  /**
   * Information about a CUDA device. The corresponding activity record
   * structure is \ref CUpti_ActivityDevice5.
   */
  CUPTI_ACTIVITY_KIND_DEVICE   = 8,

  /**
   * Information about a CUDA context. The corresponding activity record
   * structure is \ref CUpti_ActivityContext3.
   */
  CUPTI_ACTIVITY_KIND_CONTEXT  = 9,

  /**
   * A kernel executing on the GPU. This activity kind doesn't break
   * kernel concurrency. The corresponding activity record structure
   * is \ref CUpti_ActivityKernel9.
   */
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,

  /**
   * Resource naming done via NVTX APIs for thread, device, context, etc.
   * The corresponding activity record structure is \ref CUpti_ActivityName.
   */
  CUPTI_ACTIVITY_KIND_NAME     = 11,

  /**
   * Instantaneous, start, or end NVTX marker. The corresponding activity
   * record structure is \ref CUpti_ActivityMarker2.
   */
  CUPTI_ACTIVITY_KIND_MARKER = 12,

  /**
   * Extended, optional, data about a NVTX marker. User must enable
   * CUPTI_ACTIVITY_KIND_MARKER as well to get records for marker data.
   * The corresponding activity record structure is \ref CUpti_ActivityMarkerData.
   */
  CUPTI_ACTIVITY_KIND_MARKER_DATA = 13,

  /**
   * Source information about source level result. The corresponding
   * activity record structure is \ref CUpti_ActivitySourceLocator.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR = 14,

  /**
   * Results for source-level global access. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityGlobalAccess3.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS = 15,

  /**
   * Results for source-level branch. The corresponding
   * activity record structure is \ref CUpti_ActivityBranch2.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_BRANCH = 16,

  /**
   * Overhead added by CUPTI, Compiler, CUDA driver etc. The
   * corresponding activity record structure is
   * \ref CUpti_ActivityOverhead3.
   */
  CUPTI_ACTIVITY_KIND_OVERHEAD = 17,

  /**
   * A CDP (CUDA Dynamic Parallel) kernel executing on the GPU. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityCdpKernel. This activity cannot be directly
   * enabled or disabled. It is enabled and disabled through
   * concurrent kernel activity i.e. _CONCURRENT_KERNEL.
   */
  CUPTI_ACTIVITY_KIND_CDP_KERNEL = 18,
  /**
   * Preemption activity record indicating a preemption of a CDP (CUDA
   * Dynamic Parallel) kernel executing on the GPU. The corresponding
   * activity record structure is \ref CUpti_ActivityPreemption.
   */
  CUPTI_ACTIVITY_KIND_PREEMPTION = 19,

  /**
   * Environment activity records indicating power, clock, thermal,
   * etc. levels of the GPU. The corresponding activity record
   * structure is \ref CUpti_ActivityEnvironment.
   */
  CUPTI_ACTIVITY_KIND_ENVIRONMENT = 20,

  /**
   * An performance counter value associated with a specific event domain
   * instance. The corresponding activity record structure is \ref
   * CUpti_ActivityEventInstance. This activity cannot be directly
   * enabled or disabled. Information collected using the Event API.
   * can be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_EVENT_INSTANCE = 21,

  /**
   * A peer to peer memory copy. The corresponding activity record
   * structure is \ref CUpti_ActivityMemcpyPtoP4.
   */
  CUPTI_ACTIVITY_KIND_MEMCPY2 = 22,

  /**
   * A performance metric value associated with a specific metric domain
   * instance. The corresponding activity record structure is \ref
   * CUpti_ActivityMetricInstance. This activity cannot be directly
   * enabled or disabled. Information collected using the Metric API.
   * can be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_METRIC_INSTANCE = 23,

  /**
   * Results for source-level instruction execution.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstructionExecution.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION = 24,

  /**
   * Unified Memory counter record. The corresponding activity
   * record structure is \ref CUpti_ActivityUnifiedMemoryCounter3.
   */
  CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25,

  /**
   * Device global/function record. The corresponding activity
   * record structure is \ref CUpti_ActivityFunction.
   */
  CUPTI_ACTIVITY_KIND_FUNCTION = 26,

  /**
   * CUDA Module record. The corresponding activity
   * record structure is \ref CUpti_ActivityModule.
   * This activity cannot be directly enabled or disabled.
   * Information collected using the module callback can be
   * be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_MODULE = 27,

  /**
   * A device attribute value. The corresponding activity record
   * structure is \ref CUpti_ActivityDeviceAttribute.
   * This activity cannot be directly enabled or disabled.
   * Information collected using attributes CUpti_DeviceAttribute
   * or CUdevice_attribute can be stored in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE   = 28,

  /**
   * Results for source-level shared access. The
   * corresponding activity record structure is \ref
   * CUpti_ActivitySharedAccess.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_SHARED_ACCESS = 29,

  /**
   * PC sampling information for kernels. This will serialize
   * kernels. The corresponding activity record structure
   * is \ref CUpti_ActivityPCSampling3. In CUDA 12.5, this kind
   * is deprecated for Volta and later GPU architectures in favor
   * of PC Sampling APIs from the header cupti_pcsampling.h which
   * allows concurrent kernel execution.
   */
  CUPTI_ACTIVITY_KIND_PC_SAMPLING = 30,

  /**
   * Summary information about PC sampling records. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityPCSamplingRecordInfo. In CUDA 12.5, this kind
   * is deprecated for Volta and later GPU architectures in favor
   * of PC Sampling APIs from the header cupti_pcsampling.h.
   */
  CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO = 31,

  /**
   * SASS/Source line-by-line correlation record.
   * This will generate sass/source correlation for functions that have source
   * level analysis or pc sampling results. The records will be generated only
   * when either of source level analysis or pc sampling activity is enabled.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstructionCorrelation.
   * In CUDA 12.6, this kind is deprecated for Volta and later GPU architectures
   * in favor of SASS Metric APIs from the header cupti_sass_metrics.h.
   */
  CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION = 32,

  /**
   * OpenACC data events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccData.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_DATA = 33,

  /**
   * OpenACC launch events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccLaunch.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH = 34,

  /**
   * OpenACC other events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenAccOther.
   */
  CUPTI_ACTIVITY_KIND_OPENACC_OTHER = 35,

  /**
   * Information about a CUDA event (cudaEvent).
   * The corresponding activity record structure is \ref
   * CUpti_ActivityCudaEvent2.
   */
  CUPTI_ACTIVITY_KIND_CUDA_EVENT = 36,

  /**
   * Information about a CUDA stream. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityStream.
   */
  CUPTI_ACTIVITY_KIND_STREAM = 37,

  /**
   * Records for CUDA synchronization primitives. The
   * corresponding activity record structure is \ref
   * CUpti_ActivitySynchronization2.
   */
  CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38,

  /**
   * Records for correlation of different programming APIs. The
   * corresponding activity record structure is \ref
   * CUpti_ActivityExternalCorrelation.
   */
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39,

  /**
   * NVLink topology information.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityNvLink4.
   */
  CUPTI_ACTIVITY_KIND_NVLINK = 40,

  /**
   * Instantaneous Event information.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousEvent.
   * This activity can not be directly enabled or disabled.
   * Information collected using the Event API can be stored
   * in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT = 41,

  /**
   * Instantaneous Event information for a specific event
   * domain instance.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousEventInstance.
   * This activity can not be directly enabled or disabled.
   * Information collected using the Event API can be stored
   * in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE = 42,

  /**
   * Instantaneous Metric information
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousMetric.
   * This activity cannot be directly enabled or disabled.
   * Information collected using the Metric API can be stored
   * in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC = 43,

  /**
   * Instantaneous Metric information for a specific metric
   * domain instance.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityInstantaneousMetricInstance.
   * This activity cannot be directly enabled or disabled.
   * Information collected using the Metric API can be stored
   * in the corresponding activity record.
   */
  CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE = 44,

  /**
   * Memory activity tracking allocation and freeing of the memory
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemory.
   */
  CUPTI_ACTIVITY_KIND_MEMORY = 45,

  /**
   * PCI devices information used for PCI topology.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityPcie.
   */
  CUPTI_ACTIVITY_KIND_PCIE = 46,

  /**
   * OpenMP parallel events.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityOpenMp.
   */
  CUPTI_ACTIVITY_KIND_OPENMP = 47,

  /**
   * A CUDA driver kernel launch occurring outside of any
   * public API function execution. Tools can handle these
   * like records for driver API launch functions, although
   * the cbid field is not used here.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityAPI.
   */
  CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API = 48,

  /**
   * Memory activity tracking allocation and freeing of the memory
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemory4.
   */
  CUPTI_ACTIVITY_KIND_MEMORY2 = 49,

  /**
   * Memory pool activity tracking creation, destruction and
   * trimming of the memory pool.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemoryPool2.
   */
  CUPTI_ACTIVITY_KIND_MEMORY_POOL = 50,

  /**
   * Activity record for graph-level information.
   * The corresponding activity record structure is
   * \ref CUpti_ActivityGraphTrace2.
   */
  CUPTI_ACTIVITY_KIND_GRAPH_TRACE = 51,

  /**
   * JIT (Just-in-time) operation tracking.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityJit.
   */
  CUPTI_ACTIVITY_KIND_JIT = 52,

  /**
   * This activity can not be directly enabled or disabled.
   * It is enabled when CUPTI_ACTIVITY_KIND_GRAPH_TRACE is enabled
   * and device graph trace is enabled through API cuptiActivityEnableDeviceGraph().
   * The corresponding activity record structure is
   * \ref CUpti_ActivityDeviceGraphTrace.
   */
  CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE = 53,

  /**
   * Tracing batches of copies that are to be decompressed.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityMemDecompress.
   */
  CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS = 54,

  /**
   * Tracing new overheads introduced on some hardware due when
   * confidential computing is enabled.
   * The corresponding activity record structure is \ref
   * CUpti_ActivityConfidentialComputeRotation.
   */
  CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION ,


  /**
   * Count of supported activity kinds.
   */
  CUPTI_ACTIVITY_KIND_COUNT,

  CUPTI_ACTIVITY_KIND_FORCE_INT     = 0x7fffffff
} CUpti_ActivityKind;

/**
 * \brief The kinds of activity objects.
 * \see CUpti_ActivityObjectKindId
 */
typedef enum {
  /**
   * The object kind is not known.
   */
  CUPTI_ACTIVITY_OBJECT_UNKNOWN  = 0,

  /**
   * A process.
   */
  CUPTI_ACTIVITY_OBJECT_PROCESS  = 1,

  /**
   * A thread.
   */
  CUPTI_ACTIVITY_OBJECT_THREAD   = 2,

  /**
   * A device.
   */
  CUPTI_ACTIVITY_OBJECT_DEVICE   = 3,

  /**
   * A context.
   */
  CUPTI_ACTIVITY_OBJECT_CONTEXT  = 4,

  /**
   * A stream.
   */
  CUPTI_ACTIVITY_OBJECT_STREAM   = 5,

  CUPTI_ACTIVITY_OBJECT_FORCE_INT = 0x7fffffff
} CUpti_ActivityObjectKind;

/**
 * \brief Identifiers for object kinds as specified by
 * CUpti_ActivityObjectKind.
 * \see CUpti_ActivityObjectKind
 */
typedef union {
  /**
   * A process object requires that we identify the process ID. A
   * thread object requires that we identify both the process and
   * thread ID.
   */
  struct {
    uint32_t processId;
    uint32_t threadId;
  } pt;

  /**
   * A device object requires that we identify the device ID. A
   * context object requires that we identify both the device and
   * context ID. A stream object requires that we identify device,
   * context, and stream ID.
   */
  struct {
    uint32_t deviceId;
    uint32_t contextId;
    uint32_t streamId;
  } dcs;
} CUpti_ActivityObjectKindId;

/**
 * \brief The structure to provide additional data for CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL.
 */
typedef struct {
  /**
   * The remaining space in the command buffer. This field will always be zero
   * when the command buffer is full, making it not useful in such cases.
   *
   */
  uint32_t commandBufferLength;
  /**
   * The channel ID of the command buffer.
   *
   */
  uint32_t channelID;
  /**
   * The channel type of the command buffer.
   *
   */
  uint32_t channelType;
} CUpti_ActivityOverheadCommandBufferFullData;

/**
 * \brief The kinds of activity overhead.
 */
typedef enum {
  /**
   * The overhead kind is not known.
   */
  CUPTI_ACTIVITY_OVERHEAD_UNKNOWN               = 0,

  /**
   * Compiler overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER       = 1,

  /**
   * Activity buffer flush overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH               = 1<<16,

  /**
   * CUPTI instrumentation overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION            = 2<<16,

  /**
   * CUPTI resource creation and destruction overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE                   = 3<<16,

  /**
   * CUDA Runtime triggered module loading overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING = 4<<16,

  /**
   * Lazy function loading overhead.
   */
  CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING            = 5<<16,

  /**
   * Overhead due to lack of command buffer space.
   * Refer CUpti_ActivityOverheadCommandBufferFullData for more details.
   */
  CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL              = 6<<16,

  /**
   * Overhead due to activity buffer request.
   */
  CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST          = 7<<16,

  /**
    * Overhead due to UVM activity initialization.
    */
   CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT                = 8<<16,

  CUPTI_ACTIVITY_OVERHEAD_FORCE_INT             = 0x7fffffff
} CUpti_ActivityOverheadKind;

/**
 * \brief The kind of a compute API.
 */
typedef enum {
  /**
   * The compute API is not known.
   */
  CUPTI_ACTIVITY_COMPUTE_API_UNKNOWN    = 0,

  /**
   * The compute APIs are for CUDA.
   */
  CUPTI_ACTIVITY_COMPUTE_API_CUDA       = 1,

  /**
   * The compute APIs are for CUDA running
   * in MPS (Multi-Process Service) environment.
   */
  CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS   = 2,

  CUPTI_ACTIVITY_COMPUTE_API_FORCE_INT  = 0x7fffffff
} CUpti_ActivityComputeApiKind;

/**
 * \brief Flags associated with activity records.
 *
 * Activity record flags. Flags can be combined by bitwise OR to
 * associated multiple flags with an activity record. Each flag is
 * specific to a certain activity kind, as noted below.
 */
typedef enum {
  /**
   * Indicates the activity record has no flags.
   */
  CUPTI_ACTIVITY_FLAG_NONE          = 0,

  /**
   * Indicates the activity represents a device that supports
   * concurrent kernel execution. Valid for
   * CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUPTI_ACTIVITY_FLAG_DEVICE_CONCURRENT_KERNELS  = 1 << 0,

  /**
   * Indicates if the activity represents a CUdevice_attribute value
   * or a CUpti_DeviceAttribute value. Valid for
   * CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
   */
  CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE  = 1 << 0,

  /**
   * Indicates the activity represents an asynchronous memcpy
   * operation. Valid for CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC  = 1 << 0,

  /**
   * Indicates the activity represents an instantaneous marker. Valid
   * for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS  = 1 << 0,

  /**
   * Indicates the activity represents a region start marker. Valid
   * for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_START  = 1 << 1,

  /**
   * Indicates the activity represents a region end marker. Valid for
   * CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_END  = 1 << 2,

  /**
   * Indicates the activity represents an attempt to acquire a user
   * defined synchronization object.
   * Valid for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE = 1 << 3,

  /**
   * Indicates the activity represents success in acquiring the
   * user defined synchronization object.
   * Valid for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS = 1 << 4,

  /**
   * Indicates the activity represents failure in acquiring the
   * user defined synchronization object.
   * Valid for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED = 1 << 5,

  /**
   * Indicates the activity represents releasing a reservation on
   * user defined synchronization object.
   * Valid for CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE = 1 << 6,

  /**
   * Indicates the activity represents a marker that does not specify
   * a color. Valid for CUPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_COLOR_NONE  = 1 << 0,

  /**
   * Indicates the activity represents a marker that specifies a color
   * in alpha-red-green-blue format. Valid for
   * CUPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  CUPTI_ACTIVITY_FLAG_MARKER_COLOR_ARGB  = 1 << 1,

  /**
   * The number of bytes requested by each thread
   * Valid for CUpti_ActivityGlobalAccess3.
   */
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_SIZE_MASK  = 0xFF << 0,

  /**
   * If bit in this flag is set, the access was load, else it is a
   * store access. Valid for CUpti_ActivityGlobalAccess3.
   */
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_LOAD       = 1 << 8,

  /**
   * If this bit in flag is set, the load access was cached else it is
   * uncached. Valid for CUpti_ActivityGlobalAccess3.
   */
  CUPTI_ACTIVITY_FLAG_GLOBAL_ACCESS_KIND_CACHED     = 1 << 9,

  /**
   * If this bit in flag is set, the metric value overflowed. Valid
   * for CUpti_ActivityMetric and CUpti_ActivityMetricInstance.
   */
  CUPTI_ACTIVITY_FLAG_METRIC_OVERFLOWED     = 1 << 0,

  /**
   * If this bit in flag is set, the metric value couldn't be
   * calculated. This occurs when a value(s) required to calculate the
   * metric is missing.  Valid for CUpti_ActivityMetric and
   * CUpti_ActivityMetricInstance.
   */
  CUPTI_ACTIVITY_FLAG_METRIC_VALUE_INVALID  = 1 << 1,

  /**
   * If this bit in flag is set, the source level metric value couldn't be
   * calculated. This occurs when a value(s) required to calculate the
   * source level metric cannot be evaluated.
   * Valid for CUpti_ActivityInstructionExecution.
   */
  CUPTI_ACTIVITY_FLAG_INSTRUCTION_VALUE_INVALID  = 1 << 0,

  /**
   * The mask for the instruction class, \ref CUpti_ActivityInstructionClass
   * Valid for CUpti_ActivityInstructionExecution and
   * CUpti_ActivityInstructionCorrelation
   */
  CUPTI_ACTIVITY_FLAG_INSTRUCTION_CLASS_MASK    = 0xFF << 1,

  /**
   * When calling cuptiActivityFlushAll, this flag
   * can be set to force CUPTI to flush all records in the buffer, whether
   * finished or not
   */
  CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1 << 0,

  /**
   * The number of bytes requested by each thread
   * Valid for CUpti_ActivitySharedAccess.
   */
  CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_SIZE_MASK  = 0xFF << 0,

  /**
   * If bit in this flag is set, the access was load, else it is a
   * store access.  Valid for CUpti_ActivitySharedAccess.
   */
  CUPTI_ACTIVITY_FLAG_SHARED_ACCESS_KIND_LOAD       = 1 << 8,

  /**
   * Indicates the activity represents an asynchronous memset
   * operation. Valid for CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC  = 1 << 0,

  /**
   * Indicates the activity represents thrashing in CPU.
   * Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING in
   * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUPTI_ACTIVITY_FLAG_THRASHING_IN_CPU = 1 << 0,

  /**
   * Indicates the activity represents page throttling in CPU.
   * Valid for counter of kind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING in
   * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUPTI_ACTIVITY_FLAG_THROTTLING_IN_CPU = 1 << 0,

  CUPTI_ACTIVITY_FLAG_FORCE_INT = 0x7fffffff
} CUpti_ActivityFlag;

/**
 * \brief The stall reason for PC sampling activity.
 */
typedef enum {
  /**
   * Invalid reason
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID      = 0,

  /**
   * No stall, instruction is selected for issue
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE         = 1,

  /**
   * Warp is blocked because next instruction is not yet available,
   * because of instruction cache miss, or because of branching effects
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH   = 2,

  /**
   * Instruction is waiting on an arithmetic dependency
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY   = 3,

  /**
   * Warp is blocked because it is waiting for a memory access to complete.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY   = 4,

  /**
   * Texture sub-system is fully utilized or has too many outstanding requests.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE   = 5,

  /**
   * Warp is blocked as it is waiting at __syncthreads() or at memory barrier.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC   = 6,

  /**
   * Warp is blocked waiting for __constant__ memory and immediate memory access to complete.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY   = 7,

  /**
   * Compute operation cannot be performed due to the required resources not
   * being available.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY   = 8,

  /**
   * Warp is blocked because there are too many pending memory operations.
   * In Kepler architecture it often indicates high number of memory replays.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE   = 9,

  /**
   * Warp was ready to issue, but some other warp issued instead.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED   = 10,

  /**
   * Miscellaneous reasons
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER   = 11,

  /**
   * Sleeping.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING   = 12,

  CUPTI_ACTIVITY_PC_SAMPLING_STALL_FORCE_INT  = 0x7fffffff
} CUpti_ActivityPCSamplingStallReason;

/**
 * \brief Sampling period for PC sampling method
 *
 * Sampling period can be set using \ref cuptiActivityConfigurePCSampling
 */
typedef enum {
  /**
   * The PC sampling period is not set.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_INVALID = 0,

  /**
   * Minimum sampling period available on the device.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN = 1,

  /**
   * Sampling period in lower range.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW = 2,

  /**
   * Medium sampling period.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID = 3,

  /**
   * Sampling period in higher range.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH = 4,

  /**
   * Maximum sampling period available on the device.
   */
  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX = 5,

  CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_FORCE_INT = 0x7fffffff
} CUpti_ActivityPCSamplingPeriod;

/**
 * \brief The kind of a memory copy, indicating the source and
 * destination targets of the copy.
 *
 * Each kind represents the source and destination targets of a memory
 * copy. Targets are host, device, and array.
 */
typedef enum {
  /**
   * The memory copy kind is not known.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0,

  /**
   * A host to device memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOD    = 1,

  /**
   * A device to host memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOH    = 2,

  /**
   * A host to device array memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOA    = 3,

  /**
   * A device array to host memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOH    = 4,

  /**
   * A device array to device array memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOA    = 5,

  /**
   * A device array to device memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_ATOD    = 6,

  /**
   * A device to device array memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOA    = 7,

  /**
   * A device to device memory copy on the same device.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_DTOD    = 8,

  /**
   * A host to host memory copy.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_HTOH    = 9,

  /**
   * A peer to peer memory copy across different devices.
   */
  CUPTI_ACTIVITY_MEMCPY_KIND_PTOP    = 10,

  CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 0x7fffffff
} CUpti_ActivityMemcpyKind;

/**
 * \brief The kinds of memory accessed by a memory operation/copy.
 *
 * Each kind represents the type of the memory
 * accessed by a memory operation/copy.
 */
typedef enum {
  /**
   * The memory kind is unknown.
   */
  CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN            = 0,

  /**
   * The memory is pageable.
   */
  CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE           = 1,

  /**
   * The memory is pinned.
   */
  CUPTI_ACTIVITY_MEMORY_KIND_PINNED             = 2,

  /**
   * The memory is on the device.
   */
  CUPTI_ACTIVITY_MEMORY_KIND_DEVICE             = 3,

  /**
   * The memory is an array.
   */
  CUPTI_ACTIVITY_MEMORY_KIND_ARRAY              = 4,

  /**
   * The memory is managed
   */
  CUPTI_ACTIVITY_MEMORY_KIND_MANAGED            = 5,

  /**
   * The memory is device static
   */
  CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC      = 6,

  /**
   * The memory is managed static
   */
  CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC     = 7,

  CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT          = 0x7fffffff
} CUpti_ActivityMemoryKind;

/**
 * \brief The kind of a preemption activity.
 */
typedef enum {
  /**
   * The preemption kind is not known.
   */
  CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWN    = 0,

  /**
   * Preemption to save CDP block.
   */
  CUPTI_ACTIVITY_PREEMPTION_KIND_SAVE       = 1,

  /**
   * Preemption to restore CDP block.
   */
  CUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE    = 2,

  CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT  = 0x7fffffff
} CUpti_ActivityPreemptionKind;

/**
 * \brief The kind of environment data. Used to indicate what type of
 * data is being reported by an environment activity record.
 */
typedef enum {
  /**
   * Unknown data.
   */
  CUPTI_ACTIVITY_ENVIRONMENT_UNKNOWN = 0,

  /**
   * The environment data is related to speed.
   */
  CUPTI_ACTIVITY_ENVIRONMENT_SPEED = 1,

  /**
   * The environment data is related to temperature.
   */
  CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE = 2,

  /**
   * The environment data is related to power.
   */
  CUPTI_ACTIVITY_ENVIRONMENT_POWER = 3,

  /**
   * The environment data is related to cooling.
   */
  CUPTI_ACTIVITY_ENVIRONMENT_COOLING = 4,

  CUPTI_ACTIVITY_ENVIRONMENT_COUNT,

  CUPTI_ACTIVITY_ENVIRONMENT_KIND_FORCE_INT    = 0x7fffffff
} CUpti_ActivityEnvironmentKind;

/**
 * \brief Reasons for clock throttling.
 *
 * The possible reasons that a clock can be throttled. There can be
 * more than one reason that a clock is being throttled so these types
 * can be combined by bitwise OR.  These are used in the
 * clocksThrottleReason field in the Environment Activity Record.
 */
typedef enum {
  /**
   * Nothing is running on the GPU and the clocks are dropping to idle
   * state.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_GPU_IDLE              = 0x00000001,

  /**
   * The GPU clocks are limited by a user specified limit.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_USER_DEFINED_CLOCKS   = 0x00000002,

  /**
   * A software power scaling algorithm is reducing the clocks below
   * requested clocks.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_SW_POWER_CAP          = 0x00000004,

  /**
   * Hardware slowdown to reduce the clock by a factor of two or more
   * is engaged.  This is an indicator of one of the following: 1)
   * Temperature is too high, 2) External power brake assertion is
   * being triggered (e.g. by the system power supply), 3) Change in
   * power state.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN           = 0x00000008,

  /**
   * Some unspecified factor is reducing the clocks.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_UNKNOWN               = 0x80000000,

  /**
   * Throttle reason is not supported for this GPU.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_UNSUPPORTED           = 0x40000000,

  /**
   * No clock throttling.
   */
  CUPTI_CLOCKS_THROTTLE_REASON_NONE                  = 0x00000000,

  CUPTI_CLOCKS_THROTTLE_REASON_FORCE_INT             = 0x7fffffff
} CUpti_EnvironmentClocksThrottleReason;

/**
 * \brief Scope of the unified memory counter (deprecated in CUDA 7.0)
 */
typedef enum {
  /**
   * The unified memory counter scope is not known.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN = 0,

  /**
   * Collect unified memory counter for single process on one device
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE = 1,

  /**
   * Collect unified memory counter for single process across all devices
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES = 2,

  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_COUNT,

  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_FORCE_INT = 0x7fffffff
} CUpti_ActivityUnifiedMemoryCounterScope;

/**
 * \brief Kind of the Unified Memory counter
 *
 * Many activities are associated with Unified Memory mechanism; among them
 * are transfers from host to device, device to host, page fault at
 * host side.
 */
typedef enum {
  /**
   * The unified memory counter kind is not known.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN = 0,

  /**
   * Number of bytes transferred from host to device
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD = 1,

  /**
   * Number of bytes transferred from device to host
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH = 2,

  /**
   * Number of CPU page faults, this is only supported on 64 bit
   * Linux and Mac platforms
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT = 3,

  /**
   * Number of GPU page faults, this is only supported on devices with
   * compute capability 6.0 and higher and 64 bit Linux platforms
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT = 4,

  /**
   * Thrashing occurs when data is frequently accessed by
   * multiple processors and has to be constantly migrated around
   * to achieve data locality. In this case the overhead of migration
   * may exceed the benefits of locality.
   * This is only supported on 64 bit Linux platforms.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING = 5,

  /**
   * Throttling is a prevention technique used by the driver to avoid
   * further thrashing. Here, the driver doesn't service the fault for
   * one of the contending processors for a specific period of time,
   * so that the other processor can run at full-speed.
   * This is only supported on 64 bit Linux platforms.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING = 6,

  /**
   * In case throttling does not help, the driver tries to pin the memory
   * to a processor for a specific period of time. One of the contending
   * processors will have slow  access to the memory, while the other will
   * have fast access.
   * This is only supported on 64 bit Linux platforms.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP = 7,

  /**
   * Number of bytes transferred from one device to another device.
   * This is only supported on 64 bit Linux platforms.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD = 8,

  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_COUNT,

  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_FORCE_INT = 0x7fffffff
} CUpti_ActivityUnifiedMemoryCounterKind;

/**
 * \brief Memory access type for unified memory page faults
 *
 * This is valid for \ref CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
 * and \ref CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
 */
typedef enum {
  /**
   * The unified memory access type is not known
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWN = 0,

  /**
   * The page fault was triggered by read memory instruction
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READ = 1,

  /**
   * The page fault was triggered by write memory instruction
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITE = 2,

  /**
   * The page fault was triggered by atomic memory instruction
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMIC = 3,

  /**
   * The page fault was triggered by memory prefetch operation
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH = 4
} CUpti_ActivityUnifiedMemoryAccessType;

/**
 * \brief Migration cause of the Unified Memory counter
 *
 * This is valid for \ref CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
 * \ref CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH
 */
typedef enum {
  /**
   * The unified memory migration cause is not known
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_UNKNOWN = 0,

  /**
   * The unified memory migrated due to an explicit call from
   * the user e.g. cudaMemPrefetchAsync
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_USER = 1,

  /**
   * The unified memory migrated to guarantee data coherence
   * e.g. CPU/GPU faults on Pascal+ and kernel launch on pre-Pascal GPUs
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_COHERENCE = 2,

  /**
   * The unified memory was speculatively migrated by the UVM driver
   * before being accessed by the destination processor to improve
   * performance
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_PREFETCH = 3,

  /**
   * The unified memory migrated to the CPU because it was evicted to make
   * room for another block of memory on the GPU
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_EVICTION = 4,

  /**
    * The unified memory migrated to another processor because of access counter
    * notifications. Only frequently accessed pages are migrated between CPU and GPU, or
    * between peer GPUs.
    */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_MIGRATION_CAUSE_ACCESS_COUNTERS = 5,
} CUpti_ActivityUnifiedMemoryMigrationCause;

/**
 * \brief Remote memory map cause of the Unified Memory counter
 *
 * This is valid for \ref CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP
 */
typedef enum {
  /**
   * The cause of mapping to remote memory was unknown
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_UNKNOWN = 0,

  /**
   * Mapping to remote memory was added to maintain data coherence.
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_COHERENCE = 1,

  /**
   * Mapping to remote memory was added to prevent further thrashing
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_THRASHING = 2,

  /**
   * Mapping to remote memory was added to enforce the hints
   * specified by the programmer or by performance heuristics of the
   * UVM driver
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_POLICY = 3,

  /**
   * Mapping to remote memory was added because there is no more
   * memory available on the processor and eviction was not
   * possible
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_OUT_OF_MEMORY = 4,

  /**
   * Mapping to remote memory was added after the memory was
   * evicted to make room for another block of memory on the GPU
   */
  CUPTI_ACTIVITY_UNIFIED_MEMORY_REMOTE_MAP_CAUSE_EVICTION = 5,
} CUpti_ActivityUnifiedMemoryRemoteMapCause;

/**
 * \brief SASS instruction classification.
 *
 * The sass instruction are broadly divided into different class. Each enum represents a classification.
 */
typedef enum {
  /**
   * The instruction class is not known.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNKNOWN = 0,

  /**
   * Represents a 32 bit floating point operation.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_32 = 1,

  /**
   * Represents a 64 bit floating point operation.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_64 = 2,

  /**
   * Represents an integer operation.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTEGER = 3,

  /**
   * Represents a bit conversion operation.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_BIT_CONVERSION = 4,

  /**
   * Represents a control flow instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONTROL_FLOW = 5,

  /**
   * Represents a global load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL = 6,

  /**
   * Represents a shared load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED = 7,

  /**
   * Represents a local load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_LOCAL = 8,

  /**
   * Represents a generic load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_GENERIC = 9,

  /**
   * Represents a surface load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE = 10,

  /**
   * Represents a constant load instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_CONSTANT = 11,

  /**
   * Represents a texture load-store instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_TEXTURE = 12,

  /**
   * Represents a global atomic instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_GLOBAL_ATOMIC = 13,

  /**
   * Represents a shared atomic instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_SHARED_ATOMIC = 14,

  /**
   * Represents a surface atomic instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_SURFACE_ATOMIC = 15,

  /**
   * Represents a inter-thread communication instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_INTER_THREAD_COMMUNICATION = 16,

  /**
   * Represents a barrier instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_BARRIER = 17,

  /**
   * Represents some miscellaneous instructions which do not fit in the above classification.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_MISCELLANEOUS = 18,

  /**
   * Represents a 16 bit floating point operation.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_FP_16 = 19,

  /**
   * Represents uniform instruction.
   */
  CUPTI_ACTIVITY_INSTRUCTION_CLASS_UNIFORM = 20,

  CUPTI_ACTIVITY_INSTRUCTION_CLASS_KIND_FORCE_INT     = 0x7fffffff
} CUpti_ActivityInstructionClass;

/**
 * \brief Partitioned global caching option
 */
typedef enum {
  /**
   * Partitioned global cache config unknown.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN       = 0,

  /**
   * Partitioned global cache not supported.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED = 1,

  /**
   * Partitioned global cache config off.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF           = 2,

  /**
   * Partitioned global cache config on.
   */
  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON            = 3,

  CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_FORCE_INT     = 0x7fffffff
} CUpti_ActivityPartitionedGlobalCacheConfig;

/**
 * \brief Synchronization type.
 *
 * The types of synchronization to be used with
 * CUpti_ActivitySynchronization2.
 */

typedef enum {
  /**
   * Unknown data.
   */
  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN             = 0,

  /**
   * Event synchronize API.
   */
  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE   = 1,

  /**
   * Stream wait event API.
   */
  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT   = 2,

  /**
   * Stream synchronize API.
   */
  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE  = 3,

  /**
   * Context synchronize API.
   */
  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE = 4,

  CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_FORCE_INT           = 0x7fffffff
} CUpti_ActivitySynchronizationType;

/**
 * \brief stream type.
 *
 * The types of stream to be used with CUpti_ActivityStream.
 */

typedef enum {
  /**
   * Unknown data.
   */
  CUPTI_ACTIVITY_STREAM_CREATE_FLAG_UNKNOWN      = 0,

  /**
   * Default stream.
   */
  CUPTI_ACTIVITY_STREAM_CREATE_FLAG_DEFAULT      = 1,

  /**
   * Non-blocking stream.
   */
  CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NON_BLOCKING = 2,

  /**
   * Null stream.
   */
  CUPTI_ACTIVITY_STREAM_CREATE_FLAG_NULL         = 3,

  /**
   * Stream create Mask
   */
  CUPTI_ACTIVITY_STREAM_CREATE_MASK              = 0xFFFF,

  CUPTI_ACTIVITY_STREAM_CREATE_FLAG_FORCE_INT    = 0x7fffffff
} CUpti_ActivityStreamFlag;

/**
* \brief Link flags.
*
* Describes link properties, to be used with CUpti_ActivityNvLink.
*/

typedef enum {
  /**
   * The flag is invalid.
   */
  CUPTI_LINK_FLAG_INVALID        = 0,

  /**
  * Is peer to peer access supported by this link.
  */
  CUPTI_LINK_FLAG_PEER_ACCESS    = (1 << 1),

  /**
  * Is system memory access supported by this link.
  */
  CUPTI_LINK_FLAG_SYSMEM_ACCESS  = (1 << 2),

  /**
  * Is peer atomic access supported by this link.
  */
  CUPTI_LINK_FLAG_PEER_ATOMICS   = (1 << 3),

  /**
  * Is system memory atomic access supported by this link.
  */
  CUPTI_LINK_FLAG_SYSMEM_ATOMICS = (1 << 4),

  CUPTI_LINK_FLAG_FORCE_INT = 0x7fffffff
} CUpti_LinkFlag;

/**
* \brief Memory operation types.
*
* Describes the type of memory operation, to be used with CUpti_ActivityMemory4.
*/

typedef enum {
  /**
   * The operation is invalid.
   */
  CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID   = 0,

  /**
  * Memory is allocated.
  */
  CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION = 1,

  /**
  * Memory is released.
  */
  CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE    = 2,

  CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_FORCE_INT  = 0x7fffffff
} CUpti_ActivityMemoryOperationType;

/**
* \brief Memory pool types.
*
* Describes the type of memory pool, to be used with CUpti_ActivityMemory4.
*/

typedef enum {
  /**
   * The operation is invalid.
   */
  CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID   = 0,

  /**
  * Memory pool is local to the process.
  */
  CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL     = 1,

  /**
  * Memory pool is imported by the process.
  */
  CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED  = 2,

  CUPTI_ACTIVITY_MEMORY_POOL_TYPE_FORCE_INT = 0x7fffffff
} CUpti_ActivityMemoryPoolType;

/**
* \brief Memory pool operation types.
*
* Describes the type of memory pool operation, to be used with CUpti_ActivityMemoryPool2.
*/

typedef enum {
  /**
   * The operation is invalid.
   */
  CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_INVALID   = 0,

  /**
  * Memory pool is created.
  */
  CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_CREATED   = 1,

  /**
  * Memory pool is destroyed.
  */
  CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_DESTROYED = 2,

  /**
  * Memory pool is trimmed.
  */
  CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED   = 3,

  CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_FORCE_INT = 0x7fffffff
} CUpti_ActivityMemoryPoolOperationType;

typedef enum {
  CUPTI_CHANNEL_TYPE_INVALID      = 0,

  /**
   * Channel is used for standard work launch and tracking
   */
  CUPTI_CHANNEL_TYPE_COMPUTE      = 1,

  /**
   * Channel is used by an asynchronous copy engine
   * For confidential compute configurations, work launch and
   * completion are done using the copy engines.
   */
  CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY = 2,


  /**
   * Channel is used for memory decompression operations
   */
    CUPTI_CHANNEL_TYPE_DECOMP ,

  CUPTI_CHANNEL_TYPE_FORCE_INT = 0x7fffffff
} CUpti_ChannelType;

/**
* \brief CIG (CUDA in Graphics) Modes.
*
* Describes the CIG modes associated with the CUDA context.
*/

typedef enum
{
  /**
   * Regular (non-CIG) mode
   */
  CUPTI_CONTEXT_CIG_MODE_NONE         = 0,
  /**
   * CIG mode
   */
  CUPTI_CONTEXT_CIG_MODE_CIG          = 1,
  /**
   * CIG fallback mode
   */
  CUPTI_CONTEXT_CIG_MODE_CIG_FALLBACK = 2,

  CUPTI_CONTEXT_CIG_MODE_FORCE_INT    = 0x7fffffff
} CUpti_ContextCigMode;

/**
 * The source-locator ID that indicates an unknown source
 * location. There is not an actual CUpti_ActivitySourceLocator object
 * corresponding to this value.
 */
#define CUPTI_SOURCE_LOCATOR_ID_UNKNOWN 0

/**
 * An invalid function index ID.
 */
#define CUPTI_FUNCTION_INDEX_ID_INVALID 0

/**
 * An invalid/unknown correlation ID. A correlation ID of this value
 * indicates that there is no correlation for the activity record.
 */
#define CUPTI_CORRELATION_ID_UNKNOWN 0

/**
 * An invalid/unknown grid ID.
 */
#define CUPTI_GRID_ID_UNKNOWN 0LL

/**
 * An invalid/unknown timestamp for a start, end, queued, submitted,
 * or completed time.
 */
#define CUPTI_TIMESTAMP_UNKNOWN 0LL

/**
 * An invalid/unknown value.
 */
#define CUPTI_SYNCHRONIZATION_INVALID_VALUE ((uint32_t) 0xFFFFFFFFU)

/**
 * An invalid/unknown process id.
 */
#define CUPTI_AUTO_BOOST_INVALID_CLIENT_PID 0

/**
 * Invalid/unknown NVLink port number.
*/
#define CUPTI_NVLINK_INVALID_PORT -1

/**
 * Maximum NVLink port numbers.
*/
#define CUPTI_MAX_NVLINK_PORTS 32

/**
 * An invalid/unknown value for decompressed bytes.
*/
#define CUPTI_DECOMPRESSED_BYTES_UNKNOWN 0LL

START_PACKED_ALIGNMENT
/**
 * \brief Unified Memory counters configuration structure
 *
 * This structure controls the enable/disable of the various
 * Unified Memory counters consisting of scope, kind and other parameters.
 * See function \ref cuptiActivityConfigureUnifiedMemoryCounter
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Unified Memory counter Counter scope. (deprecated in CUDA 7.0)
   */
  CUpti_ActivityUnifiedMemoryCounterScope scope;

  /**
   * Unified Memory counter Counter kind
   */
  CUpti_ActivityUnifiedMemoryCounterKind kind;

  /**
   * Device id of the target device. This is relevant only
   * for single device scopes. (deprecated in CUDA 7.0)
   */
  uint32_t deviceId;

  /**
   * Control to enable/disable the counter. To enable the counter
   * set it to non-zero value while disable is indicated by zero.
   */
  uint32_t enable;
} CUpti_ActivityUnifiedMemoryCounterConfig;

/**
 * \brief Device auto boost state structure
 *
 * This structure defines auto boost state for a device.
 * See function \ref cuptiGetAutoBoostState
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Returned auto boost state. 1 is returned in case auto boost is enabled, 0
   * otherwise
   */
  uint32_t enabled;

  /**
   * Id of process that has set the current boost state. The value will be
   * CUPTI_AUTO_BOOST_INVALID_CLIENT_PID if the user does not have the
   * permission to query process ids or there is an error in querying the
   * process id.
   */
  uint32_t pid;

} CUpti_ActivityAutoBoostState;

/**
 * \brief PC sampling configuration structure
 *
 * This structure defines the pc sampling configuration.
 *
 * See function \ref cuptiActivityConfigurePCSampling
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * Size of configuration structure.
   * CUPTI client should set the size of the structure. It will be used in CUPTI to check what fields are
   * available in the structure. Used to preserve backward compatibility.
   */
  uint32_t size;

  /**
   * There are 5 level provided for sampling period. The level
   * internally maps to a period in terms of cycles. Same level can
   * map to different number of cycles on different gpus. No of
   * cycles will be chosen to minimize information loss. The period
   * chosen will be given by samplingPeriodInCycles in
   * \ref CUpti_ActivityPCSamplingRecordInfo for each kernel instance.
   */
  CUpti_ActivityPCSamplingPeriod samplingPeriod;

  /**
   * This will override the period set by samplingPeriod. Value 0 in samplingPeriod2 will be
   * considered as samplingPeriod2 should not be used and samplingPeriod should be used.
   * Valid values for samplingPeriod2 are between 5 to 31 both inclusive.
   * This will set the sampling period to (2^samplingPeriod2) cycles.
   */
  uint32_t samplingPeriod2;
} CUpti_ActivityPCSamplingConfig;

/**
 * \brief The base activity record.
 *
 * The activity API uses a CUpti_Activity as a generic representation
 * for any activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the CUpti_Activity object can
 * be cast to the specific activity record type appropriate for that kind.
 *
 * Note that all activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;
} CUpti_Activity;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   *  Reserved for internal use.
   */
  uint32_t pad2;

  /**
   * The total number of memcopy operations traced in this record.
   * This field is valid for memcpy operations happening using
   * MemcpyBatchAsync APIs in CUDA.
   * In MemcpyBatchAsync APIs, multiple memcpy operations are batched
   * together for optimization purposes based on certain heuristics.
   * For other memcpy operations, this field will be 1.
   */
   uint64_t copyCount;
} CUpti_ActivityMemcpy6;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;
} CUpti_ActivityMemcpyPtoP4;

/**
 * \brief The activity record for memset.
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint32_t graphId;

  /**
   * The ID of the HW channel on which the memory set is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   *  Undefined. Reserved for internal use
   */
  uint32_t pad2;
} CUpti_ActivityMemset4;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY).
 * This activity record provides a single record for the memory
 * allocation and memory release operations.
 *
 * Note: It is recommended to move to the new activity record \ref CUpti_ActivityMemory4
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY2.
 * \ref CUpti_ActivityMemory4 provides separate records for memory
 * allocation and memory release operations. This allows to correlate the
 * corresponding driver and runtime API activity record with the memory operation.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY
   */
  CUpti_ActivityKind kind;

  /**
   * The memory kind requested by the user
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The virtual address of the allocation
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, i.e.
   * the time when memory was allocated, in ns.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory operation, i.e.
   * the time when memory was freed, in ns.
   * This will be 0 if memory is not freed in the application
   */
  uint64_t end;

  /**
   * The program counter of the allocation of memory
   */
  uint64_t allocPC;

  /**
   * The program counter of the freeing of memory. This will
   * be 0 if memory is not freed in the application
   */
  uint64_t freePC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory allocation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;
} CUpti_ActivityMemory;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY2).
 * This activity record provides separate records for memory allocation and
 * memory release operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory operation.
 *
 * Note: This activity record is an upgrade over \ref CUpti_ActivityMemory
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY.
 * \ref CUpti_ActivityMemory provides a single record for the memory
 * allocation and memory release operations.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY2
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryOperationType.
   */
  CUpti_ActivityMemoryOperationType memoryOperationType;

  /**
   * The memory kind requested by the user, \ref CUpti_ActivityMemoryKind.
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The correlation ID of the memory operation. Each memory operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The program counter of the memory operation.
   */
  uint64_t PC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory operation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

  /**
   * The ID of the stream. If memory operation is not async, \p streamId is set to CUPTI_INVALID_STREAM_ID.
   */
  uint32_t streamId;

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;

  /**
   * \p isAsync is set if memory operation happens through async memory APIs.
   */
  uint32_t isAsync;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * The memory pool configuration used for the memory operations.
   */
  struct PACKED_ALIGNMENT {
    /**
     * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
     */
    CUpti_ActivityMemoryPoolType memoryPoolType;

#ifdef CUPTILP64
    /**
     * Undefined. Reserved for internal use.
     */
    uint32_t pad2;
#endif

    /**
     * The base address of the memory pool.
     */
    uint64_t address;

    /**
     * The release threshold of the memory pool in bytes. \p releaseThreshold is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t releaseThreshold;

    /**
     * The size of memory pool in bytes and the processId of the memory pools
     * \p size is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     * \p processId is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED, \ref CUpti_ActivityMemoryPoolType
     */
    union {
      uint64_t size;
      uint64_t processId;
    } pool;

    /**
     * The utilized size of the memory pool. \p utilizedSize is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t utilizedSize;
  } memoryPoolConfig;

    /**
     * The shared object or binary that the memory allocation request comes from.
     */
    const char* source;
} CUpti_ActivityMemory4;

/**
 * \brief The activity record for memory pool.
 *
 * This activity record represents a memory pool creation, destruction and
 * trimming (CUPTI_ACTIVITY_KIND_MEMORY_POOL).
 * This activity record provides separate records for memory pool creation,
 * destruction and trimming operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory pool operation.
 *
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY_POOL
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryPoolOperationType.
   */
  CUpti_ActivityMemoryPoolOperationType memoryPoolOperationType;

  /**
   * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
   */
  CUpti_ActivityMemoryPoolType memoryPoolType;

  /**
   * The correlation ID of the memory pool operation. Each memory pool
   * operation is assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory pool is created.
   */
  uint32_t deviceId;

  /**
   * The minimum bytes to keep of the memory pool. \p minBytesToKeep is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED,
   * \ref CUpti_ActivityMemoryPoolOperationType
   */
  size_t minBytesToKeep;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The size of the memory pool operation in bytes. \p size is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t size;

  /**
   * The release threshold of the memory pool. \p releaseThreshold is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t releaseThreshold;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The utilized size of the memory pool. \p utilizedSize is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t utilizedSize;
} CUpti_ActivityMemoryPool2;

/**
 * \brief The type of the CUDA kernel launch.
 */
typedef enum {
  /**
  * The kernel was launched via a regular kernel call
  */
  CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR = 0,

  /**
  * The kernel was launched via API \ref cudaLaunchCooperativeKernel() or
  * \ref cuLaunchCooperativeKernel()
  */
  CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE = 1,

  /**
  * The kernel was launched via API \ref cudaLaunchCooperativeKernelMultiDevice() or
  * \ref cuLaunchCooperativeKernelMultiDevice()
  */
  CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE = 2,

  /**
  * The kernel was launched as a CBL commandlist
  */
  CUPTI_ACTIVITY_LAUNCH_TYPE_CBL_COMMANDLIST = 3,
} CUpti_ActivityLaunchType;

/**
 * \brief The shared memory limit per block config for a kernel
 * This should be used to set 'cudaOccFuncShmemConfig' field in occupancy calculator API
 */
typedef enum  {
    /** The shared memory limit config is default
     */
    CUPTI_FUNC_SHMEM_LIMIT_DEFAULT              = 0x00,

    /** User has opted for a higher dynamic shared memory limit using function attribute
     * 'cudaFuncAttributeMaxDynamicSharedMemorySize' for runtime API or
     * CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES for driver API
     */
    CUPTI_FUNC_SHMEM_LIMIT_OPTIN                = 0x01,

    CUPTI_FUNC_SHMEM_LIMIT_FORCE_INT            = 0x7fffffff
} CUpti_FuncShmemLimitConfig;

/**
 * \brief The activity record for kernel.
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes (deprecated in CUDA 11.8).
   * Refer field localMemoryTotal_v2
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;

  /**
   * The ID of the HW channel on which the kernel is launched.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   * The X-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterX;

  /**
   * The Y-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterY;

  /**
   * The Z-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterZ;

  /**
   * The cluster scheduling policy for the kernel. Refer CUclusterSchedulingPolicy
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterSchedulingPolicy;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint64_t localMemoryTotal_v2;

  /**
   * The maximum cluster size for the kernel
   */
  uint32_t maxPotentialClusterSize;

  /**
   * The maximum clusters that could co-exist on the target device for the kernel
   */
  uint32_t maxActiveClusters;


} CUpti_ActivityKernel9;

/**
 * \brief The activity record for CDP (CUDA Dynamic Parallelism)
 * kernel.
 *
 * This activity record represents a CDP kernel execution.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CDP_KERNEL
   */
  CUpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel execution
   * is assigned a unique grid ID.
   */
  int64_t gridId;

  /**
   * The grid ID of the parent kernel.
   */
  int64_t parentGridId;

  /**
   * The timestamp when kernel is queued up, in ns. A value of
   * CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time is
   * unknown.
   */
  uint64_t queued;

  /**
   * The timestamp when kernel is submitted to the gpu, in ns. A value
   * of CUPTI_TIMESTAMP_UNKNOWN indicates that the submission time is
   * unknown.
   */
  uint64_t submitted;

  /**
   * The timestamp when kernel is marked as completed, in ns. A value
   * of CUPTI_TIMESTAMP_UNKNOWN indicates that the completion time is
   * unknown.
   */
  uint64_t completed;

  /**
   * The X-dimension of the parent block.
   */
  uint32_t parentBlockX;

  /**
   * The Y-dimension of the parent block.
   */
  uint32_t parentBlockY;

  /**
   * The Z-dimension of the parent block.
   */
  uint32_t parentBlockZ;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityCdpKernel;

/**
 * \brief The activity record for a preemption of a CDP kernel.
 *
 * This activity record represents a preemption of a CDP kernel.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PREEMPTION
   */
  CUpti_ActivityKind kind;

  /**
  * kind of the preemption
  */
  CUpti_ActivityPreemptionKind preemptionKind;

  /**
   * The timestamp of the preemption, in ns. A value of 0 indicates
   * that timestamp information could not be collected for the
   * preemption.
   */
  uint64_t timestamp;

  /**
  * The grid-id of the block that is preempted
  */
  int64_t gridId;

  /**
   * The X-dimension of the block that is preempted
   */
  uint32_t blockX;

  /**
   * The Y-dimension of the block that is preempted
   */
  uint32_t blockY;

  /**
   * The Z-dimension of the block that is preempted
   */
  uint32_t blockZ;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityPreemption;

/**
 * \brief The activity record for a driver or runtime API invocation.
 *
 * This activity record represents an invocation of a driver or
 * runtime API (CUPTI_ACTIVITY_KIND_DRIVER and
 * CUPTI_ACTIVITY_KIND_RUNTIME).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DRIVER,
   * CUPTI_ACTIVITY_KIND_RUNTIME, or CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID of the driver or runtime function.
   */
  CUpti_CallbackId cbid;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The ID of the process where the driver or runtime CUDA function
   * is executing.
   */
  uint32_t processId;

  /**
   * The ID of the thread where the driver or runtime CUDA function is
   * executing.
   */
  uint32_t threadId;

  /**
   * The correlation ID of the driver or runtime CUDA function. Each
   * function invocation is assigned a unique correlation ID that is
   * identical to the correlation ID in the memcpy, memset, or kernel
   * activity record that is associated with this function.
   */
  uint32_t correlationId;

  /**
   * The return value for the function. For a CUDA driver function
   * with will be a CUresult value, and for a CUDA runtime function
   * this will be a cudaError_t value.
   */
  uint32_t returnValue;
} CUpti_ActivityAPI;

/**
 * \brief The activity record for a CUPTI event.
 *
 * This activity record represents a CUPTI event value
 * (CUPTI_ACTIVITY_KIND_EVENT). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of CUPTI that collect
 * event data may choose to use this type to store the collected event
 * data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_EVENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The event ID.
   */
  CUpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The event domain ID.
   */
  CUpti_EventDomainID domain;

  /**
   * The correlation ID of the event. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the event was gathered.
   */
  uint32_t correlationId;
} CUpti_ActivityEvent;

/**
 * \brief The activity record for a CUPTI event with instance
 * information.
 *
 * This activity record represents the a CUPTI event value for a
 * specific event domain instance
 * (CUPTI_ACTIVITY_KIND_EVENT_INSTANCE). This activity record kind is
 * not produced by the activity API but is included for completeness
 * and ease-of-use. Profile frameworks built on top of CUPTI that
 * collect event data may choose to use this type to store the
 * collected event data. This activity record should be used when
 * event domain instance information needs to be associated with the
 * event.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * CUPTI_ACTIVITY_KIND_EVENT_INSTANCE.
   */
  CUpti_ActivityKind kind;

  /**
   * The event ID.
   */
  CUpti_EventID id;

  /**
   * The event domain ID.
   */
  CUpti_EventDomainID domain;

  /**
   * The event domain instance.
   */
  uint32_t instance;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The correlation ID of the event. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the event was gathered.
   */
  uint32_t correlationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityEventInstance;

/**
 * \brief The activity record for a CUPTI metric.
 *
 * This activity record represents the collection of a CUPTI metric
 * value (CUPTI_ACTIVITY_KIND_METRIC). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of CUPTI that collect
 * metric data may choose to use this type to store the collected metric
 * data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_METRIC.
   */
  CUpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  CUpti_MetricID id;

  /**
   * The metric value.
   */
  CUpti_MetricValue value;

  /**
   * The correlation ID of the metric. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the metric was gathered.
   */
  uint32_t correlationId;

  /**
   * The properties of this metric. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t pad[3];
} CUpti_ActivityMetric;

/**
 * \brief The activity record for a CUPTI metric with instance
 * information.
 *
 * This activity record represents a CUPTI metric value
 * for a specific metric domain instance
 * (CUPTI_ACTIVITY_KIND_METRIC_INSTANCE).  This activity record kind
 * is not produced by the activity API but is included for
 * completeness and ease-of-use. Profile frameworks built on top of
 * CUPTI that collect metric data may choose to use this type to store
 * the collected metric data. This activity record should be used when
 * metric domain instance information needs to be associated with the
 * metric.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * CUPTI_ACTIVITY_KIND_METRIC_INSTANCE.
   */
  CUpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  CUpti_MetricID id;

  /**
   * The metric value.
   */
  CUpti_MetricValue value;

  /**
   * The metric domain instance.
   */
  uint32_t instance;

  /**
   * The correlation ID of the metric. Use of this ID is user-defined,
   * but typically this ID value will equal the correlation ID of the
   * kernel for which the metric was gathered.
   */
  uint32_t correlationId;

  /**
   * The properties of this metric. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t pad[7];
} CUpti_ActivityMetricInstance;

/**
 * \brief The activity record for source locator.
 *
 * This activity record represents a source locator
 * (CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID for the source path, will be used in all the source level
   * results.
   */
  uint32_t id;

  /**
   * The line number in the source .
   */
  uint32_t lineNumber;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The path for the file.
   */
  const char *fileName;
} CUpti_ActivitySourceLocator;

/**
 * \brief The activity record for source-level global
 * access.
 *
 * This activity records the locations of the global
 * accesses in the source (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * The pc offset for the access.
   */
  uint64_t pcOffset;

  /**
   * This increments each time when this instruction is executed by number of
   * threads that executed this instruction with predicate and condition code
   * evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this
     access
   */
  uint64_t l2_transactions;

  /**
   * The minimum number of L2 transactions possible based on the access pattern.
   */
  uint64_t theoreticalL2Transactions;
} CUpti_ActivityGlobalAccess3;

/**
 * \brief The activity record for source level result
 * branch.
 *
 * This activity record the locations of the branches in the
 * source (CUPTI_ACTIVITY_KIND_BRANCH).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_BRANCH.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the branch.
   */
  uint32_t pcOffset;

  /**
   * Number of times this branch diverged
   */
  uint32_t diverged;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction
   */
  uint64_t threadsExecuted;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * regardless of predicate or condition code.
   */
  uint32_t executed;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityBranch2;

/**
 * \brief The activity record for a device. (CUDA 11.6 onwards)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Flag to indicate whether the device is visible to CUDA. Users can
   * set the device visibility using CUDA_VISIBLE_DEVICES environment
   */
  uint8_t isCudaVisible;

  /**
   * MIG enabled flag for device
   */
  uint8_t isMigEnabled;

  uint8_t reserved[6];

  /**
   * GPU Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t gpuInstanceId;

  /**
   * Compute Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t computeInstanceId;

  /**
   * The MIG UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid migUuid;

  /**
   * Numa (Non-uniform memory access) information for device
   * GPU is a NUMA node or not
  */
  uint32_t isNumaNode;

  /**
   * Numa (Non-uniform memory access) information for device
   * NUMA node ID of the GPU memory
   * if GPU is not a NUMA node, it returns invalidNumaId
  */
  uint32_t numaId;
} CUpti_ActivityDevice5;

/**
 * \brief The activity record for a device attribute.
 *
 * This activity record represents information about a GPU device:
 * either a CUpti_DeviceAttribute or CUdevice_attribute value
 * (CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID of the device that this attribute applies to.
   */
  uint32_t deviceId;

  /**
   * The attribute, either a CUpti_DeviceAttribute or
   * CUdevice_attribute. Flag
   * CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE is used to indicate
   * what kind of attribute this is. If
   * CUPTI_ACTIVITY_FLAG_DEVICE_ATTRIBUTE_CUDEVICE is 1 then
   * CUdevice_attribute field is value, otherwise
   * CUpti_DeviceAttribute field is valid.
   */
  union {
    CUdevice_attribute cu;
    CUpti_DeviceAttribute cupti;
  } attribute;

  /**
   * The value for the attribute. See CUpti_DeviceAttribute and
   * CUdevice_attribute for the type of the value for a given
   * attribute.
   */
  union {
    double vDouble;
    uint32_t vUint32;
    uint64_t vUint64;
    int32_t vInt32;
    int64_t vInt64;
  } value;
} CUpti_ActivityDeviceAttribute;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (CUPTI_ACTIVITY_KIND_CONTEXT).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONTEXT.
   */
  CUpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see CUpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;

  /**
   * The ID of the parent context. It would be 0 if
   * context does not have parent
   */
  uint32_t parentContextId;

  /**
   * This field indicates whether the context is a green context
   */
  uint8_t isGreenContext;

  uint8_t padding;

  /**
   * Number of multiprocessors assigned to the green context
   * Invalid if the field 'isGreenContext' is 0
   */
  uint16_t numMultiprocessors;

  /**
   * This field indicates the CIG mode
   */
  CUpti_ContextCigMode cigMode;

  uint32_t padding2;

} CUpti_ActivityContext3;

/**
 * \brief The activity record providing a name.
 *
 * This activity record provides a name for a device, context, thread,
 * etc. and other resource naming done via NVTX APIs
 * (CUPTI_ACTIVITY_KIND_NAME).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NAME.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of activity object being named.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name.
   */
  const char *name;

} CUpti_ActivityName;

/**
 * \brief The activity record providing a marker which is an
 * instantaneous point in time.
 *
 * The marker is specified with a descriptive name and unique id
 * (CUPTI_ACTIVITY_KIND_MARKER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The timestamp for the marker, in ns. A value of 0 indicates that
   * timestamp information could not be collected for the marker.
   */
  uint64_t timestamp;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * The kind of activity object associated with this marker.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object associated with this
   * marker. 'objectKind' indicates which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;


  /**
   * The marker name for an instantaneous or start marker. This will
   * be NULL for an end marker.
   */
  const char *name;

  /**
   * The name of the domain to which this marker belongs to.
   * This will be NULL for default domain.
   */
  const char *domain;

} CUpti_ActivityMarker2;

/**
 * \brief The activity record providing detailed information for a marker.
 *
 * User must enable CUPTI_ACTIVITY_KIND_MARKER as well
 * to get records for marker data.
 * The marker data contains color, payload, and category.
 * (CUPTI_ACTIVITY_KIND_MARKER_DATA).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be
   * CUPTI_ACTIVITY_KIND_MARKER_DATA.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * Defines the payload format for the value associated with the marker.
   */
  CUpti_MetricValueKind payloadKind;

  /**
   * The payload value.
   */
  CUpti_MetricValue payload;

  /**
   * The color for the marker.
   */
  uint32_t color;

  /**
   * The category for the marker.
   */
  uint32_t category;

} CUpti_ActivityMarkerData;

/**
 * \brief The activity record for CUPTI and driver overheads.
 *
 * This activity record provides CUPTI and driver overhead information
 * (CUPTI_ACTIVITY_KIND_OVERHEAD).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of overhead, CUPTI, DRIVER, COMPILER etc.
   */
  CUpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;

  /**
   * The correlation ID of the overhead operation to which
   * records belong to. This ID is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the overhead operation.
   * In some cases, it can be zero, such as for CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH records.
   */
  uint32_t correlationId;

  /**
   * Reserved for internal use.
   */
  uint32_t reserved0;

  /**
   * Pointer to the struct with additional details about the overhead.
   * Refer CUpti_ActivityOverheadKind enum and the corresponding structure to typecast and access additional overhead data.
   * Client is responsible for freeing this memory using the free function when done.
   */
  void *overheadData;

} CUpti_ActivityOverhead3;

/**
 * \brief The activity record for CUPTI environmental data.
 *
 * This activity record provides CUPTI environmental data, include
 * power, clocks, and thermals.  This information is sampled at
 * various rates and returned in this activity record.  The consumer
 * of the record needs to check the environmentKind field to figure
 * out what kind of environmental record this is.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_ENVIRONMENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID of the device
   */
  uint32_t deviceId;

  /**
   * The timestamp when this sample was retrieved, in ns. A value of 0
   * indicates that timestamp information could not be collected for
   * the marker.
   */
  uint64_t timestamp;

  /**
   * The kind of data reported in this record.
   */
  CUpti_ActivityEnvironmentKind environmentKind;

  union {
    /**
     * Data returned for CUPTI_ACTIVITY_ENVIRONMENT_SPEED environment
     * kind.
     */
    struct {
      /**
       * The SM frequency in MHz
       */
      uint32_t smClock;

      /**
       * The memory frequency in MHz
       */
      uint32_t memoryClock;

      /**
       * The PCIe link generation.
       */
      uint32_t pcieLinkGen;

      /**
       * The PCIe link width.
       */
      uint32_t pcieLinkWidth;

      /**
       * The clocks throttle reasons.
       */
      CUpti_EnvironmentClocksThrottleReason clocksThrottleReasons;
    } speed;

    /**
     * Data returned for CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE
     * environment kind.
     */
    struct {
      /**
       * The GPU temperature in degrees C.
       */
      uint32_t gpuTemperature;
    } temperature;

    /**
     * Data returned for CUPTI_ACTIVITY_ENVIRONMENT_POWER environment kind.
     * The power in milliwatts consumed by GPU and associated circuitry.
     * The power in milliwatts that will trigger power management algorithm.
     */
    struct {

      uint32_t power;
      uint32_t powerLimit;
    } power;

    /**
     * Data returned for CUPTI_ACTIVITY_ENVIRONMENT_COOLING
     * environment kind.
     */
    struct {
      /**
       * The fan speed as percentage of maximum.
       */
      uint32_t fanSpeed;
    } cooling;
  } data;
} CUpti_ActivityEnvironment;

/**
 * \brief The activity record for source-level instruction execution.
 *
 * This activity records result for source level instruction execution.
 * (CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction execution.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction, regardless of predicate or condition code.
   */
  uint64_t threadsExecuted;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t notPredOffThreadsExecuted;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * regardless of predicate or condition code.
   */
  uint32_t executed;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityInstructionExecution;

/**
 * \brief The activity record for PC sampling.
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * These samples indicate that no instruction was issued in that cycle from
   * the warp scheduler from where the warp was sampled.
   * Field is valid for devices with compute capability 6.0 and higher
   */
  uint32_t latencySamples;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons. The count includes
   * latencySamples.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref CUpti_ActivityPCSamplingStallReason
   */
  CUpti_ActivityPCSamplingStallReason stallReason;

  /**
   * The pc offset for the instruction.
   */
  uint64_t pcOffset;
} CUpti_ActivityPCSampling3;

/**
 * \brief The activity record for record status for PC sampling.
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO.
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * Number of times the PC was sampled for this kernel instance including all
   * dropped samples.
   */
  uint64_t totalSamples;

  /**
   * Number of samples that were dropped by hardware due to backpressure/overflow.
   */
  uint64_t droppedSamples;
  /**
   * Sampling period in terms of number of cycles .
   */
  uint64_t samplingPeriodInCycles;
} CUpti_ActivityPCSamplingRecordInfo;

/**
 * \brief The activity record for Unified Memory counters (CUDA 7.0 and beyond)
 *
 * This activity record represents a Unified Memory counter
 * (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind
   */
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Value of the counter
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THREASHING and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP, it is the size of the
   * memory region in bytes.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, it
   * is the number of page fault groups for the same page.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
   * it is the program counter for the instruction that caused fault.
   */
  uint64_t value;

  /**
   * The start timestamp of the counter, in ns.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity starts on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT, timestamp is
   * captured when CUDA driver started processing the fault.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, timestamp
   * is captured when CUDA driver detected thrashing of memory region.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING,
   * timestamp is captured when throttling operation was started by CUDA driver.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP,
   * timestamp is captured when CUDA driver has pushed all required operations
   * to the processor specified by dstId.
   */
  uint64_t start;

  /**
   * The end timestamp of the counter, in ns.
   * Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity finishes on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, timestamp is
   * captured when CUDA driver queues the replay of faulting memory accesses on the GPU
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, timestamp
   * is captured when throttling operation was finished by CUDA driver
   */
  uint64_t end;

  /**
   * This is the virtual base address of the page/s being transferred. For cpu and
   * gpu faults, the virtual address for the page that faulted.
   */
  uint64_t address;

  /**
   * The ID of the source CPU/device involved in the memory transfer, page fault, thrashing,
   * throttling or remote map operation. For counterKind
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, it is a bitwise ORing of the
   * device IDs fighting for the memory region, ONLY if there are less than 32 devices. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
   */
  uint32_t srcId;

  /**
   * The ID of the destination CPU/device involved in the memory transfer or remote map
   * operation. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t dstId;

  /**
   * The ID of the stream causing the transfer.
   * This value of this field is invalid.
   */
  uint32_t streamId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The flags associated with this record. See enums \ref CUpti_ActivityUnifiedMemoryAccessType
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
   * and \ref CUpti_ActivityUnifiedMemoryMigrationCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
   * and \ref CUpti_ActivityUnifiedMemoryRemoteMapCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP and \ref CUpti_ActivityFlag
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * \brief The bitmask of devices involved in the operation.
   *
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, it is a bitwise ORing of the
   * device IDs fighting for the memory region. processors[0] represents the device ID of the device 0 to device 63,
   * processors[1] represents device ID of device 64 to device 127 and so on.
   * Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_DTOD or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_FAULT_REPLAY
   */
  uint64_t processors[5];
} CUpti_ActivityUnifiedMemoryCounter3;

/**
 * \brief The activity record for global/device functions.
 *
 * This activity records function name and corresponding module
 * information.
 * (CUPTI_ACTIVITY_KIND_FUNCTION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_FUNCTION.
   */
  CUpti_ActivityKind kind;

  /**
  * ID to uniquely identify the record
  */
  uint32_t id;

  /**
   * The ID of the context where the function is launched.
   */
  uint32_t contextId;

  /**
   * The module ID in which this global/device function is present.
   */
  uint32_t moduleId;

  /**
   * The function's unique symbol index in the module.
   */
  uint32_t functionIndex;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The name of the function. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityFunction;

/**
 * \brief The activity record for a CUDA module.
 *
 * This activity record represents a CUDA module
 * (CUPTI_ACTIVITY_KIND_MODULE). This activity record kind is not
 * produced by the activity API but is included for completeness and
 * ease-of-use. Profile frameworks built on top of CUPTI that collect
 * module data from the module callback may choose to use this type to
 * store the collected module data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MODULE.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID of the context where the module is loaded.
   */
  uint32_t contextId;

  /**
   * The module ID.
   */
  uint32_t id;

  /**
   * The cubin size.
   */
  uint32_t cubinSize;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The pointer to cubin.
   */
  const void *cubin;
} CUpti_ActivityModule;

/**
 * \brief The activity record for source-level shared
 * access.
 *
 * This activity records the locations of the shared
 * accesses in the source
 * (CUPTI_ACTIVITY_KIND_SHARED_ACCESS).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_SHARED_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this shared access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of shared memory transactions generated by this access
   */
  uint64_t sharedTransactions;

  /**
   * The minimum number of shared memory transactions possible based on the access pattern.
   */
  uint64_t theoreticalSharedTransactions;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivitySharedAccess;

/**
 * \brief The activity record for CUDA event.
 *
 * This activity is used to track recorded events.
 * (CUPTI_ACTIVITY_KIND_CUDA_EVENT).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CUDA_EVENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context where the event was recorded.
   */
  uint32_t contextId;

  /**
   * The compute stream where the event was recorded.
   */
  uint32_t streamId;

  /**
   * A unique event ID to identify the event record.
   */
  uint32_t eventId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * The ID of the device where the event was recorded.
   */
  uint32_t deviceId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad2;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The device-side timestamp on CUDA event record.
   * Timestamp is in nanoseconds.
   */
  uint64_t deviceTimestamp;
  /**
   * A unique ID to associate event synchronization records
   * with the latest CUDA Event record. Similar field is added
   * in CUpti_ActivitySynchronization2 to associate CUDA Event
   * record to the synchronization record.
   *
   * The same CUDA event can be used multiple times, so the
   * event id will not be unique to correlate the synchronization
   * record with the latest CUDA Event record.
   * This field will be unique and can be used to do the required
   * correlation.
   */
  uint64_t cudaEventSyncId;
} CUpti_ActivityCudaEvent2;

/**
 * \brief The activity record for CUDA stream.
 *
 * This activity is used to track created streams.
 * (CUPTI_ACTIVITY_KIND_STREAM).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_STREAM.
   */
  CUpti_ActivityKind kind;
  /**
   * The ID of the context where the stream was created.
   */
  uint32_t contextId;

  /**
   * A unique stream ID to identify the stream.
   */
  uint32_t streamId;

  /**
   * The clamped priority for the stream.
   */
  uint32_t priority;

  /**
   * Flags associated with the stream.
   */
  CUpti_ActivityStreamFlag flag;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;
} CUpti_ActivityStream;

/**
 * \brief The activity record for synchronization management.
 *
 * This activity is used to track various CUDA synchronization APIs.
 * (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.
   */
  CUpti_ActivityKind kind;

  /**
   * The type of record.
   */
  CUpti_ActivitySynchronizationType type;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context for which the synchronization API is called.
   * In case of context synchronization API it is the context id for which the API is called.
   * In case of stream/event synchronization it is the ID of the context where the stream/event was created.
   */
  uint32_t contextId;

  /**
   * The compute stream for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuEventSynchronize.
   */
  uint32_t streamId;

  /**
   * The event ID for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuStreamSynchronize.
   */
  uint32_t cudaEventId;

  /**
   * A unique ID to associate event synchronization records
   * with the latest CUDA Event record. Similar field is added
   * in CUpti_ActivityCudaEvent2 to associate synchronization
   * record to the CUDA Event record.
   *
   * The same CUDA event can be used multiple times, so the
   * event id will not be unique to correlate the synchronization
   * record with the latest CUDA Event record.
   * This field will be unique and can be used to do the required
   * correlation.
   *
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicates that
   * the field is not applicable for this record.
   * Valid only for synchronization records related to CUDA Events.
   */
  uint64_t cudaEventSyncId;

  /**
   * The return value for the synchronization record.
   * Use cuptiActivityEnableAllSyncRecords API to enable/disable
   * collection of synchronization records with return value being
   * non-zero. This will be a CUresult value.
   */
  uint32_t returnValue;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivitySynchronization2;

/**
 * \brief The activity record for source-level sass/source
 * line-by-line correlation.
 *
 * This activity records source level sass/source correlation
 * information.
 * (CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

 /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityInstructionCorrelation;

/**
 * \brief The OpenAcc event kind for OpenAcc activity records.
 *
 * \see CUpti_ActivityKindOpenAcc
 */
typedef enum {
  CUPTI_OPENACC_EVENT_KIND_INVALID              = 0,
  CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT          = 1,
  CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN      = 2,
  CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN     = 3,
  CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH       = 4,
  CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD       = 5,
  CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD     = 6,
  CUPTI_OPENACC_EVENT_KIND_WAIT                 = 7,
  CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT        = 8,
  CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT    = 9,
  CUPTI_OPENACC_EVENT_KIND_UPDATE               = 10,
  CUPTI_OPENACC_EVENT_KIND_ENTER_DATA           = 11,
  CUPTI_OPENACC_EVENT_KIND_EXIT_DATA            = 12,
  CUPTI_OPENACC_EVENT_KIND_CREATE               = 13,
  CUPTI_OPENACC_EVENT_KIND_DELETE               = 14,
  CUPTI_OPENACC_EVENT_KIND_ALLOC                = 15,
  CUPTI_OPENACC_EVENT_KIND_FREE                 = 16,
  CUPTI_OPENACC_EVENT_KIND_FORCE_INT            = 0x7fffffff
} CUpti_OpenAccEventKind;

/**
 * \brief The OpenAcc parent construct kind for OpenAcc activity records.
 */
typedef enum {
  CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN          = 0,
  CUPTI_OPENACC_CONSTRUCT_KIND_PARALLEL         = 1,
  CUPTI_OPENACC_CONSTRUCT_KIND_KERNELS          = 2,
  CUPTI_OPENACC_CONSTRUCT_KIND_LOOP             = 3,
  CUPTI_OPENACC_CONSTRUCT_KIND_DATA             = 4,
  CUPTI_OPENACC_CONSTRUCT_KIND_ENTER_DATA       = 5,
  CUPTI_OPENACC_CONSTRUCT_KIND_EXIT_DATA        = 6,
  CUPTI_OPENACC_CONSTRUCT_KIND_HOST_DATA        = 7,
  CUPTI_OPENACC_CONSTRUCT_KIND_ATOMIC           = 8,
  CUPTI_OPENACC_CONSTRUCT_KIND_DECLARE          = 9,
  CUPTI_OPENACC_CONSTRUCT_KIND_INIT             = 10,
  CUPTI_OPENACC_CONSTRUCT_KIND_SHUTDOWN         = 11,
  CUPTI_OPENACC_CONSTRUCT_KIND_SET              = 12,
  CUPTI_OPENACC_CONSTRUCT_KIND_UPDATE           = 13,
  CUPTI_OPENACC_CONSTRUCT_KIND_ROUTINE          = 14,
  CUPTI_OPENACC_CONSTRUCT_KIND_WAIT             = 15,
  CUPTI_OPENACC_CONSTRUCT_KIND_RUNTIME_API      = 16,
  CUPTI_OPENACC_CONSTRUCT_KIND_FORCE_INT        = 0x7fffffff

} CUpti_OpenAccConstructKind;

typedef enum {
  CUPTI_OPENMP_EVENT_KIND_INVALID               = 0,
  CUPTI_OPENMP_EVENT_KIND_PARALLEL              = 1,
  CUPTI_OPENMP_EVENT_KIND_TASK                  = 2,
  CUPTI_OPENMP_EVENT_KIND_THREAD                = 3,
  CUPTI_OPENMP_EVENT_KIND_IDLE                  = 4,
  CUPTI_OPENMP_EVENT_KIND_WAIT_BARRIER          = 5,
  CUPTI_OPENMP_EVENT_KIND_WAIT_TASKWAIT         = 6,
  CUPTI_OPENMP_EVENT_KIND_FORCE_INT             = 0x7fffffff
} CUpti_OpenMpEventKind;

/**
 * \brief The base activity record for OpenAcc records.
 *
 * The OpenACC activity API part uses a CUpti_ActivityOpenAcc as a generic
 * representation for any OpenACC activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the CUpti_ActivityOpenAcc object can
 * be cast to the specific OpenACC activity record type appropriate for that kind.
 *
 * Note that all OpenACC activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;

  /**
   * CUPTI OpenACC event kind (\see CUpti_OpenAccEventKind)
   */
  CUpti_OpenAccEventKind eventKind;

  /**
   * CUPTI OpenACC parent construct kind (\see CUpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  CUpti_OpenAccConstructKind parentConstruct;

  /**
   * Version number
   */
  uint32_t version;

  /**
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /**
   * Device type
   */
  uint32_t deviceType;

  /**
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /**
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /**
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /**
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A zero value means the line number is not known.
   */
  uint32_t lineNo;

  /**
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /**
   * The line number of the first line of the function named in funcName.
   * A zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /**
   * The last line number of the function named in funcName.
   * A zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * CUPTI start timestamp
   */
  uint64_t start;

  /**
   * CUPTI end timestamp
   */
  uint64_t end;

  /**
   * CUDA device id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuDeviceId;

  /**
   * CUDA context id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuContextId;

  /**
   * CUDA stream id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuStreamId;

  /**
   * The ID of the process where the OpenACC activity is executing.
   */
  uint32_t cuProcessId;

  /**
   * The ID of the thread where the OpenACC activity is executing.
   */
  uint32_t cuThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_nvidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceding external correlation record of type
   * CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event occurred.
   */
  const char *funcName;
} CUpti_ActivityOpenAcc;

/**
 * \brief The activity record for OpenACC data.
 *
 * (CUPTI_ACTIVITY_KIND_OPENACC_DATA).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_OPENACC_DATA.
   */
  CUpti_ActivityKind kind;

  /**
   * CUPTI OpenACC event kind (\see CUpti_OpenAccEventKind)
   */
  CUpti_OpenAccEventKind eventKind;

  /*
   * CUPTI OpenACC parent construct kind (\see CUpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  CUpti_OpenAccConstructKind parentConstruct;

  /*
   * Version number
   */
  uint32_t version;

  /*
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /*
   * Device type
   */
  uint32_t deviceType;

  /*
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /*
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /*
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /*
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /*
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /*
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /*
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * CUPTI start timestamp
   */
  uint64_t start;

  /**
   * CUPTI end timestamp
   */
  uint64_t end;

  /**
   * CUDA device id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuDeviceId;

  /**
   * CUDA context id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuContextId;

  /**
   * CUDA stream id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuStreamId;

  /**
   * The ID of the process where the OpenACC activity is executing.
   */
  uint32_t cuProcessId;

  /**
   * The ID of the thread where the OpenACC activity is executing.
   */
  uint32_t cuThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_nvidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceding external correlation record of type
   * CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /*
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /*
   * A pointer to a null-terminated string containing the name of the
   * function in which the event occurred.
   */
  const char *funcName;

  /* --- end of common CUpti_ActivityOpenAcc part --- */

  /**
   * Number of bytes
   */
  uint64_t bytes;

  /**
   * Host pointer if available
   */
  uint64_t hostPtr;

  /**
   * Device pointer if available
   */
  uint64_t devicePtr;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /*
   * A pointer to null-terminated string containing the name of the variable
   * for which this event is triggered, if known, or a null pointer if not.
   */
  const char *varName;

} CUpti_ActivityOpenAccData;

/**
 * \brief The activity record for OpenACC launch.
 *
 * (CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH.
   */
  CUpti_ActivityKind kind;

  /**
   * CUPTI OpenACC event kind (\see CUpti_OpenAccEventKind)
   */
  CUpti_OpenAccEventKind eventKind;

  /**
   * CUPTI OpenACC parent construct kind (\see CUpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  CUpti_OpenAccConstructKind parentConstruct;

  /**
   * Version number
   */
  uint32_t version;

  /**
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /**
   * Device type
   */
  uint32_t deviceType;

  /**
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /**
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /**
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /**
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /**
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /**
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /**
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * CUPTI start timestamp
   */
  uint64_t start;

  /**
   * CUPTI end timestamp
   */
  uint64_t end;

  /**
   * CUDA device id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuDeviceId;

  /**
   * CUDA context id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuContextId;

  /**
   * CUDA stream id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuStreamId;

  /**
   * The ID of the process where the OpenACC activity is executing.
   */
  uint32_t cuProcessId;

  /**
   * The ID of the thread where the OpenACC activity is executing.
   */
  uint32_t cuThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_nvidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceding external correlation record of type
   * CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /**
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /**
   * A pointer to a null-terminated string containing the name of the
   * function in which the event occurred.
   */
  const char *funcName;

  /* --- end of common CUpti_ActivityOpenAcc part --- */

  /**
   * The number of gangs created for this kernel launch
   */
  uint64_t numGangs;

  /**
   * The number of workers created for this kernel launch
   */
  uint64_t numWorkers;

  /**
   * The number of vector lanes created for this kernel launch
   */
  uint64_t vectorLength;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * A pointer to null-terminated string containing the name of the
   * kernel being launched, if known, or a null pointer if not.
   */
  const char *kernelName;

} CUpti_ActivityOpenAccLaunch;

/**
 * \brief The activity record for OpenACC other.
 *
 * (CUPTI_ACTIVITY_KIND_OPENACC_OTHER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_OPENACC_OTHER.
   */
  CUpti_ActivityKind kind;

  /**
   * CUPTI OpenACC event kind (\see CUpti_OpenAccEventKind)
   */
  CUpti_OpenAccEventKind eventKind;

  /**
   * CUPTI OpenACC parent construct kind (\see CUpti_OpenAccConstructKind)
   *
   * Note that for applications using PGI OpenACC runtime < 16.1, this
   * will always be CUPTI_OPENACC_CONSTRUCT_KIND_UNKNOWN.
   */
  CUpti_OpenAccConstructKind parentConstruct;

  /**
   * Version number
   */
  uint32_t version;

  /**
   * 1 for any implicit event, such as an implicit wait at a synchronous data construct
   * 0 otherwise
   */
  uint32_t implicit;

  /**
   * Device type
   */
  uint32_t deviceType;

  /**
   * Device number
   */
  uint32_t deviceNumber;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /**
   * Value of async() clause of the corresponding directive
   */
  uint64_t async;

  /**
   * Internal asynchronous queue number used
   */
  uint64_t asyncMap;

  /**
   * The line number of the directive or program construct or the starting line
   * number of the OpenACC construct corresponding to the event.
   * A negative or zero value means the line number is not known.
   */
  uint32_t lineNo;

  /**
   * For an OpenACC construct, this contains the line number of the end
   * of the construct. A negative or zero value means the line number is not known.
   */
  uint32_t endLineNo;

  /**
   * The line number of the first line of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcLineNo;

  /**
   * The last line number of the function named in func_name.
   * A negative or zero value means the line number is not known.
   */
  uint32_t funcEndLineNo;

  /**
   * CUPTI start timestamp
   */
  uint64_t start;

  /**
   * CUPTI end timestamp
   */
  uint64_t end;

  /**
   * CUDA device id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuDeviceId;

  /**
   * CUDA context id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuContextId;

  /**
   * CUDA stream id
   * Valid only if deviceType is acc_device_nvidia.
   */
  uint32_t cuStreamId;

  /**
   * The ID of the process where the OpenACC activity is executing.
   */
  uint32_t cuProcessId;

  /**
   * The ID of the thread where the OpenACC activity is executing.
   */
  uint32_t cuThreadId;

  /**
   * The OpenACC correlation ID.
   * Valid only if deviceType is acc_device_nvidia.
   * If not 0, it uniquely identifies this record. It is identical to the
   * externalId in the preceding external correlation record of type
   * CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC.
   */
  uint32_t externalId;

  /**
   * A pointer to null-terminated string containing the name of or path to
   * the source file, if known, or a null pointer if not.
   */
  const char *srcFile;

  /**
   * A pointer to a null-terminated string containing the name of the
   * function in which the event occurred.
   */
  const char *funcName;

  /* --- end of common CUpti_ActivityOpenAcc part --- */
} CUpti_ActivityOpenAccOther;

/**
 * \brief The base activity record for OpenMp records.
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {

  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;

  /**
   * CUPTI OpenMP event kind (\see CUpti_OpenMpEventKind)
   */
  CUpti_OpenMpEventKind eventKind;

  /**
   * Version number
   */
  uint32_t version;

  /**
   * ThreadId
   */
  uint32_t threadId;

  /**
   * CUPTI start timestamp
   */
  uint64_t start;

  /**
   * CUPTI end timestamp
   */
  uint64_t end;

  /**
   * The ID of the process where the OpenMP activity is executing.
   */
  uint32_t cuProcessId;

  /**
   * The ID of the thread where the OpenMP activity is executing.
   */
  uint32_t cuThreadId;
} CUpti_ActivityOpenMp;

/**
 * \brief The kind of external APIs supported for correlation.
 *
 * Custom correlation kinds are reserved for usage in external tools.
 *
 * \see CUpti_ActivityExternalCorrelation
 */
typedef enum {
    CUPTI_EXTERNAL_CORRELATION_KIND_INVALID              = 0,

    /**
     * The external API is unknown to CUPTI
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN              = 1,

    /**
     * The external API is OpenACC
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC              = 2,

    /**
     * The external API is custom0
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0              = 3,

    /**
     * The external API is custom1
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1              = 4,

    /**
     * The external API is custom2
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2              = 5,

    /**
     * Add new kinds before this line
     */
    CUPTI_EXTERNAL_CORRELATION_KIND_SIZE,

    CUPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT            = 0x7fffffff
} CUpti_ExternalCorrelationKind;

/**
 * \brief The activity record for correlation with external records
 *
 * This activity record correlates native CUDA records (e.g. CUDA Driver API,
 * kernels, memcpys, ...) with records from external APIs such as OpenACC.
 * (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION).
 *
 * \see CUpti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of external API this record correlated to.
   */
  CUpti_ExternalCorrelationKind externalKind;

  /**
   * The correlation ID of the associated non-CUDA API record.
   * The exact field in the associated external record depends
   * on that record's activity kind (\see externalKind).
   */
  uint64_t externalId;

  /**
   * The correlation ID of the associated CUDA driver or runtime API record.
   */
  uint32_t correlationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t reserved;
} CUpti_ActivityExternalCorrelation;

/**
* \brief The device type for device connected to NVLink.
*/
typedef enum {
    CUPTI_DEV_TYPE_INVALID = 0,

    /**
    * The device type is GPU.
    */
    CUPTI_DEV_TYPE_GPU = 1,

    /**
    * The device type is NVLink processing unit in CPU.
    */
    CUPTI_DEV_TYPE_NPU = 2,

    CUPTI_DEV_TYPE_FORCE_INT = 0x7fffffff
} CUpti_DevType;

/**
* \brief NVLink information.
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
*/

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
   */
  CUpti_ActivityKind kind;

  /**
   * NvLink version.
   */
  uint32_t nvlinkVersion;

  /**
   * Type of device 0 \ref CUpti_DevType
   */
  CUpti_DevType typeDev0;

  /**
   * Type of device 1 \ref CUpti_DevType
   */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {

      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
   * Flag gives capabilities of the link \see CUpti_LinkFlag
   */
  uint32_t flag;

  /**
   * Number of physical NVLinks present between two devices.
   */
  uint32_t  physicalNvLinkCount;

  /**
   * Port numbers for maximum 32 NVLinks connected to device 0.
   * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t  portDev0[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Port numbers for maximum 32 NVLinks connected to device 1.
   * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t  portDev1[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Bandwidth of NVLink in kbytes/sec
   */
  uint64_t  bandwidth;

  /**
   * NVSwitch is connected as an intermediate node.
   */
  uint8_t nvswitchConnected;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[7];
} CUpti_ActivityNvLink4;

#define CUPTI_MAX_GPUS 32
/**
 * Field to differentiate whether PCIE Activity record
 * is of a GPU or a PCI Bridge
 */
typedef enum {
    /**
     * PCIE GPU record
     */
    CUPTI_PCIE_DEVICE_TYPE_GPU       = 0,

    /**
     * PCIE Bridge record
     */
    CUPTI_PCIE_DEVICE_TYPE_BRIDGE    = 1,

    CUPTI_PCIE_DEVICE_TYPE_FORCE_INT = 0x7fffffff
} CUpti_PcieDeviceType;

/**
 * \brief PCI devices information required to construct topology
 *
 * This structure gives capabilities of GPU and PCI bridge connected to the PCIE bus
 * which can be used to understand the topology.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PCIE.
   */
  CUpti_ActivityKind kind;

  /**
   * Type of device in topology, \ref CUpti_PcieDeviceType. If type is
   * CUPTI_PCIE_DEVICE_TYPE_GPU use devId for id and gpuAttr and if type is
   * CUPTI_PCIE_DEVICE_TYPE_BRIDGE use bridgeId for id and bridgeAttr.
   */
  CUpti_PcieDeviceType type;

  /**
   * A unique identifier for GPU or Bridge in Topology
   */
  union {
    /**
     * GPU device ID
     */
    CUdevice devId;

    /**
     * A unique identifier for Bridge in the Topology
     */
    uint32_t bridgeId;
  } id;

  /**
   * Domain for the GPU or Bridge, required to identify which PCIE bus it belongs to in
   * multiple NUMA systems.
   */
  uint32_t domain;

  /**
   * PCIE Generation of GPU or Bridge.
   */
  uint16_t pcieGeneration;

  /**
   * Link rate of the GPU or bridge in gigatransfers per second (GT/s)
   */
  uint16_t linkRate;

  /**
   * Link width of the GPU or bridge
   */
  uint16_t linkWidth;

  /**
   * Upstream bus ID for the GPU or PCI bridge. Required to identify which bus it is
   * connected to in the topology.
   */
  uint16_t upstreamBus;

  /**
   * Attributes for more information about GPU (gpuAttr) or PCI Bridge (bridgeAttr)
   */
  union {
    struct {
      /**
       * UUID for the device. \ref CUpti_ActivityDevice5.
       */
      CUuuid uuidDev;

      /**
       * CUdevice with which this device has P2P capability.
       * This can also be obtained by querying cuDeviceCanAccessPeer or
       * cudaDeviceCanAccessPeer APIs
       */
      CUdevice peerDev[CUPTI_MAX_GPUS];
    } gpuAttr;

    struct {
      /**
       * The downstream bus number, used to search downstream devices/bridges connected
       * to this bridge.
       */
      uint16_t secondaryBus;

      /**
       * Device ID of the bridge
       */
      uint16_t deviceId;

      /**
       * Vendor ID of the bridge
       */
      uint16_t vendorId;

      /**
       * Padding for alignment
       */
      uint16_t pad0;
    } bridgeAttr;
  } attr;
} CUpti_ActivityPcie;

/**
 * \brief PCIE Generation.
 *
 * Enumeration of PCIE Generation for
 * pcie activity attribute pcieGeneration
 */
typedef enum {
  /**
  * PCIE Generation 1
  */
  CUPTI_PCIE_GEN_GEN1       = 1,

  /**
  * PCIE Generation 2
  */
  CUPTI_PCIE_GEN_GEN2       = 2,

  /**
  * PCIE Generation 3
  */
  CUPTI_PCIE_GEN_GEN3       = 3,

  /**
  * PCIE Generation 4
  */
  CUPTI_PCIE_GEN_GEN4       = 4,

  /**
  * PCIE Generation 5
  */
  CUPTI_PCIE_GEN_GEN5       = 5,

  /**
  * PCIE Generation 6
  */
  CUPTI_PCIE_GEN_GEN6       = 6,

  CUPTI_PCIE_GEN_FORCE_INT  = 0x7fffffff
} CUpti_PcieGen;


/*
 * \brief Confidential Computing Rotation Events
 *
 * Define event type for confidential compute tracing
 */
typedef enum {
    CUPTI_CONFIDENTIAL_COMPUTE_INVALID_ROTATION_EVENT   = 0,
    /**
     * This channel has been blocked from accepting new CUDA work so a key rotation can be done.
     */
    CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_BLOCKED   = 1,
    /**
     * This channel remains blocked and all queued CUDA work has completed.
     * Other clients or channels may cause delays in starting the key rotation.
     */
    CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_DRAINED = 2,
    /**
     * Key rotations have completed and this channel is unblocked.
     */
    CUPTI_CONFIDENTIAL_COMPUTE_KEY_ROTATION_CHANNEL_UNBLOCKED     = 3,


    CUPTI_CONFIDENTIAL_COMPUTE_EVENT_TYPE_FORCE_INT    = 0x7fffffff
} CUpti_ConfidentialComputeRotationEventType;

/**
 * \brief Event related to confidential compute encryption rotation
 *
 * This structure gives timestamps for stages of encryption rotation
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONFIDENTIAL_COMPUTE_ROTATION.
   */
  CUpti_ActivityKind kind;

  /**
   * Type of event \ref CUpti_ConfidentialComputeRotationEventType
   */
  CUpti_ConfidentialComputeRotationEventType eventType;

  /**
   * Device ID
   */
  uint32_t deviceId;

  /**
   * Context ID
   */
  uint32_t contextId;

  /**
   * Channel ID
   */
  uint32_t channelId;

  /**
   * Channel Type \ref CUpti_ChannelType
   */
  CUpti_ChannelType channelType;

  /**
   * Timestamp in ns
   */
  uint64_t timestamp;
} CUpti_ActivityConfidentialComputeRotation;


/**
 * \brief The activity record for an instantaneous CUPTI event.
 *
 * This activity record represents a CUPTI event value
 * (CUPTI_ACTIVITY_KIND_EVENT) sampled at a particular instant.
 * This activity record kind is not produced by the activity API but is
 * included for completeness and ease-of-use. Profiler frameworks built on
 * top of CUPTI that collect event data at a particular time may choose to
 * use this type to store the collected event data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The event ID.
   */
  CUpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The timestamp at which event is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * Undefined. reserved for internal use
   */
  uint32_t reserved;
} CUpti_ActivityInstantaneousEvent;

/**
 * \brief The activity record for an instantaneous CUPTI event
 * with event domain instance information.
 *
 * This activity record represents the a CUPTI event value for a
 * specific event domain instance
 * (CUPTI_ACTIVITY_KIND_EVENT_INSTANCE) sampled at a particular instant.
 * This activity record kind is not produced by the activity API but is
 * included for completeness and ease-of-use. Profiler frameworks built on
 * top of CUPTI that collect event data may choose to use this type to store the
 * collected event data. This activity record should be used when
 * event domain instance information needs to be associated with the
 * event.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE.
   */
  CUpti_ActivityKind kind;

  /**
   * The event ID.
   */
  CUpti_EventID id;

  /**
   * The event value.
   */
  uint64_t value;

  /**
   * The timestamp at which event is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * The event domain instance
   */
  uint8_t instance;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[3];
} CUpti_ActivityInstantaneousEventInstance;

/**
 * \brief The activity record for an instantaneous CUPTI metric.
 *
 * This activity record represents the collection of a CUPTI metric
 * value (CUPTI_ACTIVITY_KIND_METRIC) at a particular instance.
 * This activity record kind is not produced by the activity API but
 * is included for completeness and ease-of-use. Profiler frameworks built
 * on top of CUPTI that collect metric data may choose to use this type to
 * store the collected metric data.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC.
   */
  CUpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  CUpti_MetricID id;

  /**
   * The metric value.
   */
  CUpti_MetricValue value;

  /**
   * The timestamp at which metric is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * The properties of this metric. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[3];
} CUpti_ActivityInstantaneousMetric;

/**
 * \brief The instantaneous activity record for a CUPTI metric with instance
 * information.

 * This activity record represents a CUPTI metric value
 * for a specific metric domain instance
 * (CUPTI_ACTIVITY_KIND_METRIC_INSTANCE) sampled at a particular time. This
 * activity record kind is not produced by the activity API but is included for
 * completeness and ease-of-use. Profiler frameworks built on top of
 * CUPTI that collect metric data may choose to use this type to store
 * the collected metric data. This activity record should be used when
 * metric domain instance information needs to be associated with the
 * metric.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE.
   */
  CUpti_ActivityKind kind;

  /**
   * The metric ID.
   */
  CUpti_MetricID id;

  /**
   * The metric value.
   */
  CUpti_MetricValue value;

  /**
   * The timestamp at which metric is sampled
   */
  uint64_t timestamp;

  /**
   * The device id
   */
  uint32_t deviceId;

  /**
   * The properties of this metric. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The metric domain instance
   */
  uint8_t instance;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[2];
} CUpti_ActivityInstantaneousMetricInstance;

/**
 * \brief The types of JIT entry.
 *
 * To be used in CUpti_ActivityJit.
 */
typedef enum {
  CUPTI_ACTIVITY_JIT_ENTRY_INVALID= 0,

  /**
  * PTX to CUBIN.
  */
  CUPTI_ACTIVITY_JIT_ENTRY_PTX_TO_CUBIN = 1,

  /**
  * NVVM-IR to PTX
  */
  CUPTI_ACTIVITY_JIT_ENTRY_NVVM_IR_TO_PTX = 2,

  CUPTI_ACTIVITY_JIT_ENTRY_TYPE_FORCE_INT = 0x7fffffff
} CUpti_ActivityJitEntryType;

/**
 * \brief The types of JIT compilation operations.
 *
 * To be used in CUpti_ActivityJit.
 */

typedef enum {
  CUPTI_ACTIVITY_JIT_OPERATION_INVALID = 0,
  /**
  * Loaded from the compute cache.
  */
  CUPTI_ACTIVITY_JIT_OPERATION_CACHE_LOAD = 1,

  /**
  * Stored in the compute cache.
  */
  CUPTI_ACTIVITY_JIT_OPERATION_CACHE_STORE = 2,

  /**
  * JIT compilation.
  */
  CUPTI_ACTIVITY_JIT_OPERATION_COMPILE = 3,

  CUPTI_ACTIVITY_JIT_OPERATION_TYPE_FORCE_INT = 0x7fffffff
} CUpti_ActivityJitOperationType;

/**
 * \brief The activity record for JIT operations.
 * This activity represents the JIT operations (compile, load, store) of a CUmodule
 * from the Compute Cache.
 * Gives the exact hashed path of where the cached module is loaded from,
 * or where the module will be stored after Just-In-Time (JIT) compilation.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind must be CUPTI_ACTIVITY_KIND_JIT.
   */
  CUpti_ActivityKind kind;

  /**
    * The JIT entry type.
    */
  CUpti_ActivityJitEntryType jitEntryType;

  /**
   * The JIT operation type.
   */
  CUpti_ActivityJitOperationType jitOperationType;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The start timestamp for the JIT operation, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the JIT operation.
   */
  uint64_t start;

  /**
   * The end timestamp for the JIT operation, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the JIT operation.
   */
  uint64_t end;

  /**
   * The correlation ID of the JIT operation to which
   * records belong to. Each JIT operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the JIT operation.
   */
  uint32_t correlationId;

  /**
   * Internal use.
   */
  uint32_t padding;

  /**
   * The correlation ID to correlate JIT compilation, load and store operations.
   * Each JIT compilation unit is assigned a unique correlation ID
   * at the time of the JIT compilation. This correlation id can be used
   * to find the matching JIT cache load/store records.
   */
  uint64_t jitOperationCorrelationId;

  /**
   * The size of compute cache.
   */
  uint64_t cacheSize;

  /**
   * The path where the fat binary is cached.
   */
  const char* cachePath;

  /**
   * The ID of the process where the JIT operation is executing.
   */
  uint32_t processId;

  /**
   * The ID of the thread where the JIT operation is executing.
   */
  uint32_t threadId;
} CUpti_ActivityJit2;


/**
 * \brief The activity record for trace of graph execution.
 *
 * This activity record represents execution for a graph without giving visibility
 * about the execution of its nodes. This is intended to reduce overheads in tracing
 * each node. The activity kind is CUPTI_ACTIVITY_KIND_GRAPH_TRACE
 */
typedef struct {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GRAPH_TRACE
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the graph launch. Each graph launch is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the graph.
   */
  uint32_t correlationId;

  /**
   * The start timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t start;

  /**
   * The end timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t end;

  /**
   * The ID of the device where the first node of the graph is executed.
   * If this is INT_MAX, then the start is on the host.
   */
  uint32_t deviceId;

  /**
   * The unique ID of the graph that is launched.
   */
  uint32_t graphId;

  /**
   * The ID of the context where the first node of the graph is executed.
   * If this is INT_MAX, then the start is on the host.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the graph is being launched.
   */
  uint32_t streamId;

  /**
   * This field is reserved for internal use
   */
  void *reserved;

  /**
   * The ID of the device where last node of the graph is executed
   */
  uint32_t endDeviceId;

  /**
   * The ID of the context where the last node of the graph is executed.
   */
  uint32_t endContextId;
} CUpti_ActivityGraphTrace2;

/**
 * \brief The launch mode for device graph execution.
 */
typedef enum {
    CUPTI_DEVICE_GRAPH_LAUNCH_MODE_INVALID = 0,
    CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET = 1,
    CUPTI_DEVICE_GRAPH_LAUNCH_MODE_TAIL = 2,
    CUPTI_DEVICE_GRAPH_LAUNCH_MODE_FIRE_AND_FORGET_AS_SIBLING = 3,
} CUpti_DeviceGraphLaunchMode;

/**
 * \brief The activity record for trace of device graph execution.
 *
 * This activity record represents execution for a device launched graph without giving visibility
 * about the execution of its nodes. This is intended to reduce overheads in tracing
 * each node. The activity kind is CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE
 */
typedef struct {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE
   */
  CUpti_ActivityKind kind;

  /**
   * The ID of the device where the first node of the graph is executed.
   */
  uint32_t deviceId;

  /**
   * The start timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t start;

  /**
   * The end timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t end;

  /**
   * The unique ID of the graph that is launched.
   */
  uint32_t graphId;

  /**
   * The unique ID of the graph that has launched this graph.
   */
  uint32_t launcherGraphId;

  /**
   * The type of launch. See \ref CUpti_DeviceGraphLaunchMode
   */
  uint32_t deviceLaunchMode;

  /**
   * The ID of the context where the first node of the graph is executed.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the graph is being launched.
   */
  uint64_t streamId;

  /**
   * This field is reserved for internal use
   */
  void *reserved;

} CUpti_ActivityDeviceGraphTrace;

/**
 * \brief The activity record for trace of decompression operations.
 *
 * This activity record represents execution for a batch of decompression operatios.
 * The activity kind is CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS
 */
typedef struct {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS
   */
  CUpti_ActivityKind kind;

  /**
   * The ID of the device.
   */
  uint32_t deviceId;

  /**
   * The ID of the context.
   */
  uint32_t contextId;

  /**
   * The ID of the stream.
   */
  uint32_t streamId;

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   * The correlation ID of the decompression operations. Each operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the operation.
   */
  uint32_t correlationId;

  /**
   * The number of operations in the batch.
   */
  uint32_t numberOfOperations;

  /**
   * The number of bytes to be read and decompressed in the
   * batch operation.
   */
  uint64_t sourceBytes;

  /**
   * This field is reserved for internal use
   */
  void *reserved0;

  /**
   * The start timestamp.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the start time is unknown.
   */
  uint64_t start;

  /**
   * The end timestamp.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the start time is unknown.
   */
  uint64_t end;
} CUpti_ActivityMemDecompress;

END_PACKED_ALIGNMENT

/**
 * \brief Activity attributes.
 *
 * These attributes are used to control the behavior of the activity
 * API.
 */
typedef enum {
    /**
     * The device memory size (in bytes) reserved for storing profiling data for concurrent
     * kernels (activity kind \ref CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), memcopies and memsets
     * for each buffer on a context. The value is a size_t.
     *
     * There is a limit on how many device buffers can be allocated per context. User
     * can query and set this limit using the attribute
     * \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
     * CUPTI doesn't pre-allocate all the buffers, it pre-allocates only those many
     * buffers as set by the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
     * When all of the data in a buffer is consumed, it is added in the reuse pool, and
     * CUPTI picks a buffer from this pool when a new buffer is needed. Thus memory
     * footprint does not scale with the kernel count. Applications with the high density
     * of kernels, memcopies and memsets might result in having CUPTI to allocate more device buffers.
     * CUPTI allocates another buffer only when it runs out of the buffers in the
     * reuse pool.
     *
     * Since buffer allocation happens in the main application thread, this might result
     * in stalls in the critical path. CUPTI pre-allocates 3 buffers of the same size to
     * mitigate this issue. User can query and set the pre-allocation limit using the
     * attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE.
     *
     * Having larger buffer size leaves less device memory for the application.
     * Having smaller buffer size increases the risk of dropping timestamps for
     * records if too many kernels or memcopies or memsets are launched at one time.
     *
     * This value only applies to new buffer allocations. Set this value before initializing
     * CUDA or before creating a context to ensure it is considered for the following allocations.
     *
     * The default value is 3200000 (~3MB) which can accommodate profiling data
     * up to 100,000 kernels, memcopies and memsets combined.
     *
     * Note: Starting with the CUDA 12.0 Update 1 release, CUPTI allocates profiling buffer in the
     * device memory by default as this might help in improving the performance of the
     * tracing run. Refer to the description of the attribute
     * \ref CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED for more details.
     * Size of the memory and maximum number of pools are still controlled by the attributes
     * \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE and \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT.
     *
     * Note: The actual amount of device memory per buffer reserved by CUPTI might be larger.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE                      = 0,

    /**
     * The device memory size (in bytes) reserved for storing profiling
     * data for CDP operations for each buffer on a context. The
     * value is a size_t.
     *
     * Having larger buffer size means less flush operations but
     * consumes more device memory. This value only applies to new
     * allocations.
     *
     * Set this value before initializing CUDA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 8388608 (8MB).
     *
     * Note: The actual amount of device memory per context reserved by
     * CUPTI might be larger.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP          = 1,

    /**
     * The maximum number of device memory buffers per context. The value is a size_t.
     *
     * For an application with high rate of kernel launches, memcopies and memsets having a bigger pool
     * limit helps in timestamp collection for all these activities at the expense of a larger memory footprint.
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
     * for more details.
     *
     * Setting this value will not modify the number of memory buffers
     * currently stored.
     *
     * Set this value before initializing CUDA to ensure the limit is
     * not exceeded.
     *
     * The default value is 250.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT                = 2,

    /**
     * This attribute is not supported starting with CUDA 12.3
     * CUPTI no longer uses profiling semaphore pool to store profiling data.
     *
     * There is a limit on how many semaphore pools can be allocated per context. User
     * can query and set this limit using the attribute
     * \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT.
     * CUPTI doesn't pre-allocate all the semaphore pools, it pre-allocates only those many
     * semaphore pools as set by the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
     * When all of the data in a semaphore pool is consumed, it is added in the reuse pool, and
     * CUPTI picks a semaphore pool from the reuse pool when a new semaphore pool is needed. Thus memory
     * footprint does not scale with the kernel count. Applications with the high density
     * of kernels might result in having CUPTI to allocate more semaphore pools.
     * CUPTI allocates another semaphore pool only when it runs out of the semaphore pools in the
     * reuse pool.
     *
     * Since semaphore pool allocation happens in the main application thread, this might result
     * in stalls in the critical path. CUPTI pre-allocates 3 semaphore pools of the same size to
     * mitigate this issue. User can query and set the pre-allocation limit using the
     * attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE.
     *
     * Having larger semaphore pool size leaves less device memory for the application.
     * Having smaller semaphore pool size increases the risk of dropping timestamps for
     * kernel records if too many kernels are issued/launched at one time.
     *
     * This value only applies to new semaphore pool allocations. Set this value before initializing
     * CUDA or before creating a context to ensure it is considered for the following allocations.
     *
     * The default value is 25000 which can accommodate profiling data for upto 25,000 kernels.
     *
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE           = 3,

    /**
     * This attribute is not supported starting with CUDA 12.3
     * CUPTI no longer uses profiling semaphore pool to store profiling data.
     *
     * The maximum number of profiling semaphore pools per context. The value is a size_t.
     *
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
     * for more details.
     *
     * Set this value before initializing CUDA to ensure the limit is not exceeded.
     *
     * The default value is 250.
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT          = 4,

    /**
     * The flag to indicate whether user should provide activity buffer of zero value.
     * The value is a uint8_t.
     *
     * If the value of this attribute is non-zero, user should provide
     * a zero value buffer in the \ref CUpti_BuffersCallbackRequestFunc.
     * If the user does not provide a zero value buffer after setting this to non-zero,
     * the activity buffer may contain some uninitialized values when CUPTI returns it in
     * \ref CUpti_BuffersCallbackCompleteFunc
     *
     * If the value of this attribute is zero, CUPTI will initialize the user buffer
     * received in the \ref CUpti_BuffersCallbackRequestFunc to zero before filling it.
     * If the user sets this to zero, a few stalls may appear in critical path because CUPTI
     * will zero out the buffer in the main thread.
     * Set this value before returning from \ref CUpti_BuffersCallbackRequestFunc to
     * ensure it is considered for all the subsequent user buffers.
     *
     * The default value is 0.
     */
    CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER              = 5,

    /**
     * Number of device buffers to pre-allocate for a context during the initialization phase.
     * The value is a size_t.
     *
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE
     * for details.
     *
     * This value must be less than the maximum number of device buffers set using
     * the attribute \ref CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT
     *
     * Set this value before initializing CUDA or before creating a context to ensure it
     * is considered by the CUPTI.
     *
     * The default value is set to 3 to ping pong between these buffers (if possible).
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE        = 6,

    /**
     * This attribute is not supported starting with CUDA 12.3
     * CUPTI no longer uses profiling semaphore pool to store profiling data.
     *
     * Number of profiling semaphore pools to pre-allocate for a context during the
     * initialization phase. The value is a size_t.
     *
     * Refer to the description of the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE
     * for details.
     *
     * This value must be less than the maximum number of profiling semaphore pools set
     * using the attribute \ref CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT
     *
     * Set this value before initializing CUDA or before creating a context to ensure it
     * is considered by the CUPTI.
     *
     * The default value is set to 3 to ping pong between these pools (if possible).
     */
    CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_PRE_ALLOCATE_VALUE  = 7,

    /**
     * Allocate page-locked (pinned) host memory for storing profiling data for concurrent
     * kernels, memcopies and memsets for each buffer on a context. The value is a uint8_t.
     *
     * Starting with the CUDA 11.2 release, CUPTI allocates profiling buffer in the pinned host
     * memory by default as this might help in improving the performance of the tracing run.
     * Allocating excessive amounts of pinned memory may degrade system performance, since it
     * reduces the amount of memory available to the system for paging. For this reason user
     * might want to change the location from pinned host memory to device memory by setting
     * value of this attribute to 0.
     *
     * Using page-locked (pinned) host memory buffers is not supported on confidential computing
     * devices. On setting this attribute to 1, CUPTI will return CUPTI_ERROR_NOT_SUPPORTED.
     *
     * The default value is 1.
     */
    CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED         = 8,

    /**
     * Request activity buffers per-thread to store CUPTI activity records
     * in the activity buffer on per-thread basis. The value is a uint8_t.
     *
     * The attribute should be set before registering the buffer callbacks using
     * cuptiActivityRegisterCallbacks API and before any of the CUPTI activity kinds are enabled.
     * This makes sure that all the records are stored in activity buffers allocated per-thread.
     * Changing this attribute in the middle of the profiling session will result in undefined behavior.
     *
     * The default value is 0.
     */
    CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER,


    /**
     * The device memory size (in bytes) reserved for storing profiling
     * data for device graph operations for each buffer on a context. The
     * value is a size_t.
     *
     * Having larger buffer size means less flush operations but
     * consumes more device memory. This value only applies to new
     * allocations.
     *
     * Set this value before initializing CUDA or before creating a
     * context to ensure it is considered for the following allocations.
     *
     * The default value is 16777216 (16MB).
     *
     * Note: The actual amount of device memory per context reserved by
     * CUPTI might be larger.
     */
    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_DEVICE_GRAPHS,


    CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_FORCE_INT                 = 0x7fffffff
} CUpti_ActivityAttribute;

/**
 * \brief Thread-Id types.
 *
 * CUPTI uses different methods to obtain the thread-id depending on the
 * support and the underlying platform. This enum documents these methods
 * for each type. APIs \ref cuptiSetThreadIdType and \ref cuptiGetThreadIdType
 * can be used to set and get the thread-id type.
 */
typedef enum {
    /**
     * Default type
     * Windows uses API GetCurrentThreadId()
     * Linux/Mac/Android/QNX use POSIX pthread API pthread_self()
     */
    CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT       = 0,

    /**
     * This type is based on the system API available on the underlying platform
     * and thread-id obtained is supposed to be unique for the process lifetime.
     * Windows uses API GetCurrentThreadId()
     * Linux uses syscall SYS_gettid
     * Mac uses syscall SYS_thread_selfid
     * Android/QNX use gettid()
     */
    CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM        = 1,

    /**
     * Add new enums before this field.
     */
    CUPTI_ACTIVITY_THREAD_ID_TYPE_SIZE          = 2,

    CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT     = 0x7fffffff
} CUpti_ActivityThreadIdType;

/**
 * \brief Get the CUPTI timestamp.
 *
 * Returns a timestamp normalized to correspond with the start and end
 * timestamps reported in the CUPTI activity records. The timestamp is
 * reported in nanoseconds.
 *
 * \param timestamp Returns the CUPTI timestamp
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p timestamp is NULL
 */
CUptiResult CUPTIAPI cuptiGetTimestamp(uint64_t *timestamp);

/**
 * \brief Get the ID of a context.
 *
 * Get the ID of a context.
 *
 * \param context The context
 * \param contextId Returns a process-unique ID for the context
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT The context is NULL or not valid.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p contextId is NULL
 */
CUptiResult CUPTIAPI cuptiGetContextId(CUcontext context, uint32_t *contextId);

/**
 * \brief Get the ID of a stream.
 *
 * Get the ID of a stream. The stream ID is unique within a context
 * (i.e. all streams within a context will have unique stream
 * IDs).
 *
 * \param context If non-NULL then the stream is checked to ensure
 * that it belongs to this context. Typically this parameter should be
 * null.
 * \param stream The stream
 * \param streamId Returns a context-unique ID for the stream
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_STREAM if unable to get stream ID, or
 * if \p context is non-NULL and \p stream does not belong to the
 * context
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p streamId is NULL
 *
 * **DEPRECATED** This method is deprecated as of CUDA 8.0.
 * Use method cuptiGetStreamIdEx instead.
 */
CUptiResult CUPTIAPI cuptiGetStreamId(CUcontext context, CUstream stream, uint32_t *streamId);

/**
* \brief Get the ID of a stream.
*
* Get the ID of a stream. The stream ID is unique within a context
* (i.e. all streams within a context will have unique stream
* IDs).
*
* \param context If non-NULL then the stream is checked to ensure
* that it belongs to this context. Typically this parameter should be
* null.
* \param stream The stream
* \param perThreadStream Flag to indicate if program is compiled for per-thread streams
* \param streamId Returns a context-unique ID for the stream
*
* \retval CUPTI_SUCCESS
* \retval CUPTI_ERROR_NOT_INITIALIZED
* \retval CUPTI_ERROR_INVALID_STREAM if unable to get stream ID, or
* if \p context is non-NULL and \p stream does not belong to the
* context
* \retval CUPTI_ERROR_INVALID_PARAMETER if \p streamId is NULL
*/
CUptiResult CUPTIAPI cuptiGetStreamIdEx(CUcontext context, CUstream stream, uint8_t perThreadStream, uint32_t *streamId);

/**
 * \brief Get the ID of a device
 *
 * If \p context is NULL, returns the ID of the device that contains
 * the currently active context. If \p context is non-NULL, returns
 * the ID of the device which contains that context. Operates in a
 * similar manner to cudaGetDevice() or cuCtxGetDevice() but may be
 * called from within callback functions.
 *
 * \param context The context, or NULL to indicate the current context.
 * \param deviceId Returns the ID of the device that is current for
 * the calling thread.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE if unable to get device ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p deviceId is NULL
 */
CUptiResult CUPTIAPI cuptiGetDeviceId(CUcontext context, uint32_t *deviceId);

/**
 * \brief Get the unique ID of a graph node
 *
 * Returns the unique ID of the CUDA graph node.
 *
 * \param node The graph node.
 * \param nodeId Returns the unique ID of the node
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p node is NULL
 */
CUptiResult CUPTIAPI cuptiGetGraphNodeId(CUgraphNode node, uint64_t *nodeId);

/**
 * \brief Get the unique ID of graph
 *
 * Returns the unique ID of CUDA graph.
 *
 * \param graph The graph.
 * \param pId Returns the unique ID of the graph
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p graph is NULL
 */
CUptiResult CUPTIAPI cuptiGetGraphId(CUgraph graph, uint32_t *pId);

/**
 * \brief Get the unique ID of executable graph
 *
 * Returns the unique ID of executable CUDA graph.
 *
 * \param graphExec The executable graph.
 * \param pId Returns the unique ID of the executable graph
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p graph is NULL
 */
CUptiResult CUPTIAPI cuptiGetGraphExecId(CUgraphExec graphExec, uint32_t *pId);

/**
 * \brief Enable collection of a specific kind of activity record.
 *
 * Enable collection of a specific kind of activity record. Multiple
 * kinds can be enabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to collect
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind kind);

/**
 * \brief Enable collection of a specific kind of activity record. For certain activity kinds
 * it dumps existing records.
 *
 * In general, the behavior of this API is similar to the API \ref cuptiActivityEnable i.e. it
 * enables the collection of a specific kind of activity record.
 * Additionally, this API can help in dumping the records for activities which happened in
 * the past before enabling the corresponding activity kind.
 * The API allows to get records for the current resource allocations done in CUDA
 * For CUPTI_ACTIVITY_KIND_DEVICE, existing device records are dumped
 * For CUPTI_ACTIVITY_KIND_CONTEXT, existing context records are dumped
 * For CUPTI_ACTIVITY_KIND_STREAM, existing stream records are dumped
 * For CUPTI_ACTIVITY_KIND_ NVLINK, existing NVLINK records are dumped
 * For CUPTI_ACTIVITY_KIND_PCIE, existing PCIE records are dumped
 * For other activities, the behavior is similar to the API \ref cuptiActivityEnable
 *
 * Device records are emitted in CUPTI on CUDA driver initialization. Those records
 * can only be retrieved by the user if CUPTI is attached before CUDA initialization.
 * Context and stream records are emitted on context and stream creation.
 * The use case of the API is to provide the records for CUDA resources
 * (contexts/streams/devices) that are currently active if user late attaches CUPTI.
 *
 * Before calling this function, the user must register buffer callbacks
 * to get the activity records by calling \ref cuptiActivityRegisterCallbacks.
 * If the user does not register the buffers and calls API \ref cuptiActivityEnableAndDump,
 * then CUPTI will enable the activity kind but not provide any records for that
 * activity kind.
 *
 * \param kind The kind of activity record to collect
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_UNKNOWN if buffer is not initialized.
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityEnableAndDump(CUpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record.
 *
 * Disable collection of a specific kind of activity record. Multiple
 * kinds can be disabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to stop collecting
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind kind);

/**
 * \brief Enable collection of a specific kind of activity record for
 * a context.
 *
 * Enable collection of a specific kind of activity record for a
 * context.  This setting done by this API will supersede the global
 * settings for activity records enabled by \ref cuptiActivityEnable.
 * Multiple kinds can be enabled by calling this function multiple
 * times.
 *
 * \param context The context for which activity is to be enabled
 * \param kind The kind of activity record to collect
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityEnableContext(CUcontext context, CUpti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record for
 * a context.
 *
 * Disable collection of a specific kind of activity record for a context.
 * This setting done by this API will supersede the global settings
 * for activity records.
 * Multiple kinds can be enabled by calling this function multiple times.
 *
 * \param context The context for which activity is to be disabled
 * \param kind The kind of activity record to stop collecting
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
CUptiResult CUPTIAPI cuptiActivityDisableContext(CUcontext context, CUpti_ActivityKind kind);

/**
 * \brief Get the number of activity records that were dropped of
 * insufficient buffer space.
 *
 * Get the number of records that were dropped because of insufficient
 * buffer space.  The dropped count includes records that could not be
 * recorded because CUPTI did not have activity buffer space available
 * for the record (because the CUpti_BuffersCallbackRequestFunc
 * callback did not return an empty buffer of sufficient size) and
 * also CDP records that could not be record because the device-size
 * buffer was full (size is controlled by the
 * CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP attribute). The dropped
 * count maintained for the queue is reset to zero when this function
 * is called.
 *
 * \param context The context, or NULL to get dropped count from global queue
 * \param streamId The stream ID
 * \param dropped The number of records that were dropped since the last call
 * to this function.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p dropped is NULL
 */
CUptiResult CUPTIAPI cuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t streamId,
                                                       size_t *dropped);

/**
 * \brief Iterate over the activity records in a buffer.
 *
 * This is a helper function to iterate over the activity records in a
 * buffer. A buffer of activity records is typically obtained by
 * receiving a CUpti_BuffersCallbackCompleteFunc callback. Stop iterating
 * the buffer when an error occurs.
 *
 * An example of typical usage:
 * \code
 * CUpti_Activity *record = NULL;
 * CUptiResult status = CUPTI_SUCCESS;
 *   do {
 *      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
 *      if(status == CUPTI_SUCCESS) {
 *           // Use record here...
 *      }
 *      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
 *          break;
 *      else if (status == CUPTI_ERROR_INVALID_KIND)
 *          break;
 *      else {
 *          goto Error;
 *      }
 *    } while (1);
 * \endcode
 *
 * \param buffer The buffer containing activity records
 * \param record Inputs the previous record returned by
 * cuptiActivityGetNextRecord and returns the next activity record
 * from the buffer. If input value is NULL, returns the first activity
 * record in the buffer. Records of certain kinds like CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
 * may contain invalid (0) timestamps, indicating that no timing information could
 * be collected for lack of device memory.
 * \param validBufferSizeBytes The number of valid bytes in the buffer.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p buffer is NULL.
 * \retval CUPTI_ERROR_INVALID_KIND if activity record is either incomplete or invalid
 */
CUptiResult CUPTIAPI cuptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes,
                                                CUpti_Activity **record);

/**
 * \brief Function type for callback used by CUPTI to request an empty
 * buffer for storing activity records.
 *
 * This callback function signals the CUPTI client that an activity
 * buffer is needed by CUPTI. The activity buffer is used by CUPTI to
 * store activity records. The callback function can decline the
 * request by setting \p *buffer to NULL. In this case CUPTI may drop
 * activity records.
 *
 * \param buffer Returns the new buffer. If set to NULL then no buffer
 * is returned.
 * \param size Returns the size of the returned buffer.
 * \param maxNumRecords Returns the maximum number of records that
 * should be placed in the buffer. If 0 then the buffer is filled with
 * as many records as possible. If > 0 the buffer is filled with at
 * most that many records before it is returned.
 */
typedef void (CUPTIAPI *CUpti_BuffersCallbackRequestFunc)(
    uint8_t **buffer,
    size_t *size,
    size_t *maxNumRecords);

/**
 * \brief Function type for callback used by CUPTI to return a buffer
 * of activity records.
 *
 * This callback function returns to the CUPTI client a buffer
 * containing activity records.  The buffer contains \p validSize
 * bytes of activity records which should be read using
 * cuptiActivityGetNextRecord. The number of dropped records can be
 * read using cuptiActivityGetNumDroppedRecords. After this call CUPTI
 * relinquished ownership of the buffer and will not use it
 * anymore. The client may return the buffer to CUPTI using the
 * CUpti_BuffersCallbackRequestFunc callback.
 * Note: CUDA 6.0 onwards, all buffers returned by this callback are
 * global buffers i.e. there is no context/stream specific buffer.
 * User needs to parse the global buffer to extract the context/stream
 * specific activity records.
 *
 * \param context The context this buffer is associated with. If NULL, the
 * buffer is associated with the global activities. This field is deprecated
 * as of CUDA 6.0 and will always be NULL.
 * \param streamId The stream id this buffer is associated with.
 * This field is deprecated as of CUDA 6.0 and will always be NULL.
 * \param buffer The activity record buffer.
 * \param size The total size of the buffer in bytes as set in
 * CUpti_BuffersCallbackRequestFunc.
 * \param validSize The number of valid bytes in the buffer.
 */
typedef void (CUPTIAPI *CUpti_BuffersCallbackCompleteFunc)(
    CUcontext context,
    uint32_t streamId,
    uint8_t *buffer,
    size_t size,
    size_t validSize);

/**
 * \brief Registers callback functions with CUPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested/completed callbacks from CUPTI.
 *
 * Registering these callbacks prevents the client from using CUPTI's
 * blocking enqueue/dequeue functions.
 *
 * \param funcBufferRequested callback which is invoked when an empty
 * buffer is requested by CUPTI
 * \param funcBufferCompleted callback which is invoked when a buffer
 * containing activity records is available from CUPTI
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if either \p
 * funcBufferRequested or \p funcBufferCompleted is NULL
 */
CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
        CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

/**
 * \brief Wait for all activity records to be delivered via the
 * completion callback.
 *
 * This function does not return until all activity records associated
 * with the specified context/stream are returned to the CUPTI client
 * using the callback registered in cuptiActivityRegisterCallbacks. To
 * ensure that all activity records are complete, the requested
 * stream(s), if any, are synchronized.
 *
 * If \p context is NULL, the global activity records (i.e. those not
 * associated with a particular stream) are flushed (in this case no
 * streams are synchronized).  If \p context is a valid CUcontext and
 * \p streamId is 0, the buffers of all streams of this context are
 * flushed.  Otherwise, the buffers of the specified stream in this
 * context is flushed.
 *
 * Before calling this function, the buffer handling callback api
 * must be activated by calling cuptiActivityRegisterCallbacks.
 *
 * \param context A valid CUcontext or NULL.
 * \param streamId The stream ID.
 * \param flag The flag can be set to indicate a forced flush. See CUpti_ActivityFlag
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_CUPTI_ERROR_INVALID_OPERATION if not preceded
 * by a successful call to cuptiActivityRegisterCallbacks
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred
 *
 * **DEPRECATED** This method is deprecated
 * CONTEXT and STREAMID will be ignored. Use cuptiActivityFlushAll
 * to flush all data.
 */
CUptiResult CUPTIAPI cuptiActivityFlush(CUcontext context, uint32_t streamId, uint32_t flag);

/**
 * \brief Request to deliver activity records via the buffer completion callback.
 *
 * This function returns the activity records associated with all contexts/streams
 * (and the global buffers not associated with any stream) to the CUPTI client
 * using the callback registered in cuptiActivityRegisterCallbacks.
 *
 * This is a blocking call but it doesn't issue any CUDA synchronization calls
 * implicitly thus it's not guaranteed that all activities are completed on the
 * underlying devices. Activity record is considered as completed if it has all
 * the information filled up including the timestamps if any. It is the client's
 * responsibility to issue necessary CUDA synchronization calls before calling
 * this function if all activity records with complete information are expected
 * to be delivered.
 *
 * Behavior of the function based on the input flag:
 * (-) ::For default flush i.e. when flag is set as 0, it returns all the
 * activity buffers which have all the activity records completed, buffers need not
 * to be full though. It doesn't return buffers which have one or more incomplete
 * records. Default flush can be done at a regular interval in a separate thread.
 * (-) ::For forced flush i.e. when flag CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is passed
 * to the function, it returns all the activity buffers including the ones which have
 * one or more incomplete activity records. It's suggested for clients to do the
 * force flush before the termination of the profiling session to allow remaining
 * buffers to be delivered. In general, it can be done in the at-exit handler.
 *
 * Before calling this function, the buffer handling callback api must be activated
 * by calling cuptiActivityRegisterCallbacks.
 *
 * \param flag The flag can be set to indicate a forced flush. See CUpti_ActivityFlag
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if not preceded by a
 * successful call to cuptiActivityRegisterCallbacks
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred
 *
 * \see cuptiActivityFlushPeriod
 */
CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t flag);

/**
 * \brief Read an activity API attribute.
 *
 * Read an activity API attribute and return it in \p *value.
 *
 * \param attr The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value is NULL, or
 * if \p attr is not an activity attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiActivityGetAttribute(CUpti_ActivityAttribute attr,
        size_t *valueSize, void* value);

/**
 * \brief Write an activity API attribute.
 *
 * Write an activity API attribute.
 *
 * \param attr The attribute to write
 * \param valueSize The size, in bytes, of the value
 * \param value The attribute value to write
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value is NULL, or
 * if \p attr is not an activity attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiActivitySetAttribute(CUpti_ActivityAttribute attr,
        size_t *valueSize, void* value);


/**
 * \brief Set Unified Memory Counter configuration.
 *
 * Set the configuration before enabling the corresponding activity kind
 * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER.
 * The API should be called after CUDA driver initialization.
 *
 * \param config A pointer to \ref CUpti_ActivityUnifiedMemoryCounterConfig structures
 * containing Unified Memory counter configuration.
 * \param count Number of Unified Memory counter configuration structures
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p config is NULL or
 * any parameter in the \p config structures is not a valid value
 * \retval CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED One potential reason is that
 * platform (OS/arch) does not support the unified memory counters
 * \retval CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE Indicates that the device
 * does not support the unified memory counters
 * \retval CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES Indicates that
 * multi-GPU configuration without P2P support between any pair of devices
 * does not support the unified memory counters
 */
CUptiResult CUPTIAPI cuptiActivityConfigureUnifiedMemoryCounter(CUpti_ActivityUnifiedMemoryCounterConfig *config, uint32_t count);

/**
 * \brief Get auto boost state
 *
 * The profiling results can be inconsistent in case auto boost is enabled.
 * CUPTI tries to disable auto boost while profiling. It can fail to disable in
 * cases where user does not have the permissions or CUDA_AUTO_BOOST env
 * variable is set. The function can be used to query whether auto boost is
 * enabled.
 *
 * \param context A valid CUcontext.
 * \param state A pointer to \ref CUpti_ActivityAutoBoostState structure which
 * contains the current state and the id of the process that has requested the
 * current state
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p CUcontext or \p state is NULL
 * \retval CUPTI_ERROR_NOT_SUPPORTED Indicates that the device does not support auto boost
 * \retval CUPTI_ERROR_UNKNOWN an internal error occurred
 */
CUptiResult CUPTIAPI cuptiGetAutoBoostState(CUcontext context, CUpti_ActivityAutoBoostState *state);

/**
 * \brief Set PC sampling configuration.
 *
 * For Pascal and older GPU architectures this API must be called before enabling
 * activity kind CUPTI_ACTIVITY_KIND_PC_SAMPLING. There is no such requirement
 * for Volta and newer GPU architectures.
 *
 * For Volta and newer GPU architectures if this API is called in the middle of
 * execution, PC sampling configuration will be updated for subsequent kernel launches.
 *
 * \param ctx The context
 * \param config A pointer to \ref CUpti_ActivityPCSamplingConfig structure
 * containing PC sampling configuration.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_OPERATION if this api is called while
 * some valid event collection method is set.
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p config is NULL or
 * any parameter in the \p config structures is not a valid value
 * \retval CUPTI_ERROR_NOT_SUPPORTED Indicates that the system/device
 * does not support the unified memory counters
 */
CUptiResult CUPTIAPI cuptiActivityConfigurePCSampling(CUcontext ctx, CUpti_ActivityPCSamplingConfig *config);

/**
 * \brief Returns the last error from a cupti call or callback
 *
 * Returns the last error that has been produced by any of the cupti api calls
 * or the callback in the same host thread and resets it to CUPTI_SUCCESS.
 */
CUptiResult CUPTIAPI cuptiGetLastError(void);

/**
 * \brief Set the thread-id type
 *
 * CUPTI uses the method corresponding to set type to generate the thread-id.
 * See enum \ref CUpti_ActivityThreadIdType for the list of methods.
 * Activity records having thread-id field contain the same value.
 * Thread id type must not be changed during the profiling session to
 * avoid thread-id value mismatch across activity records.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_SUPPORTED if \p type is not supported on the platform
 */
CUptiResult CUPTIAPI cuptiSetThreadIdType(CUpti_ActivityThreadIdType type);

/**
 * \brief Get the thread-id type
 *
 * Returns the thread-id type used in CUPTI
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p type is NULL
  */
CUptiResult CUPTIAPI cuptiGetThreadIdType(CUpti_ActivityThreadIdType *type);

/**
* \brief Check support for a compute capability
*
* This function is used to check the support for a device based on
* it's compute capability. It sets the \p support when the compute
* capability is supported by the current version of CUPTI, and clears
* it otherwise. This version of CUPTI might not support all GPUs sharing
* the same compute capability. It is suggested to use API \ref
* cuptiDeviceSupported which provides correct information.
*
* \param major The major revision number of the compute capability
* \param minor The minor revision number of the compute capability
* \param support Pointer to an integer to return the support status
*
* \retval CUPTI_SUCCESS
* \retval CUPTI_ERROR_INVALID_PARAMETER if \p support is NULL
*
* \sa ::cuptiDeviceSupported
*/
CUptiResult CUPTIAPI cuptiComputeCapabilitySupported(int major, int minor, int *support);

/**
* \brief Check support for a compute device
*
* This function is used to check the support for a compute device.
* It sets the \p support when the device is supported by the current
* version of CUPTI, and clears it otherwise.
*
* \param dev The device handle returned by CUDA Driver API cuDeviceGet
* \param support Pointer to an integer to return the support status
*
* \retval CUPTI_SUCCESS
* \retval CUPTI_ERROR_INVALID_PARAMETER if \p support is NULL
* \retval CUPTI_ERROR_INVALID_DEVICE if \p dev is not a valid device
*
* \sa ::cuptiComputeCapabilitySupported
*/
CUptiResult CUPTIAPI cuptiDeviceSupported(CUdevice dev, int *support);

/**
 * This indicates the virtualization mode in which CUDA device is running
 */
typedef enum {
  /**
   * No virtualization mode is associated with the device
   * i.e. it's a baremetal GPU
   */
  CUPTI_DEVICE_VIRTUALIZATION_MODE_NONE = 0,
  /**
   * The device is associated with the pass-through GPU.
   * In this mode, an entire physical GPU is directly assigned
   * to one virtual machine (VM).
   */
  CUPTI_DEVICE_VIRTUALIZATION_MODE_PASS_THROUGH = 1,
  /**
   * The device is associated with the virtual GPU (vGPU).
   * In this mode multiple virtual machines (VMs) have simultaneous,
   * direct access to a single physical GPU.
   */
  CUPTI_DEVICE_VIRTUALIZATION_MODE_VIRTUAL_GPU = 2,

  CUPTI_DEVICE_VIRTUALIZATION_MODE_FORCE_INT = 0x7fffffff
} CUpti_DeviceVirtualizationMode;

/**
 * \brief Query the virtualization mode of the device
 *
 * This function is used to query the virtualization mode of the CUDA device.
 *
 * \param dev The device handle returned by CUDA Driver API cuDeviceGet
 * \param mode Pointer to an CUpti_DeviceVirtualizationMode to return the virtualization mode
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_DEVICE if \p dev is not a valid device
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p mode is NULL
 *
 */
CUptiResult CUPTIAPI cuptiDeviceVirtualizationMode(CUdevice dev, CUpti_DeviceVirtualizationMode *mode);

/**
 * \brief Detach CUPTI from the running process
 *
 * This API detaches the CUPTI from the running process. It destroys and cleans up all the
 * resources associated with CUPTI in the current process. After CUPTI detaches from the process,
 * the process will keep on running with no CUPTI attached to it.
 * For safe operation of the API, it is recommended this API is invoked from the exit callsite
 * of any of the CUDA Driver or Runtime API. Otherwise CUPTI client needs to make sure that
 * required CUDA synchronization and CUPTI activity buffer flush is done before calling the API.
 * Sample code showing the usage of the API in the cupti callback handler code:
 * \code
  void CUPTIAPI
  cuptiCallbackHandler(void *userdata, CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, void *cbdata)
  {
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;

    // Take this code path when CUPTI detach is requested
    if (detachCupti) {
      switch(domain)
      {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
        case CUPTI_CB_DOMAIN_DRIVER_API:
          if (cbInfo->callbackSite == CUPTI_API_EXIT) {
              // call the CUPTI detach API
              cuptiFinalize();
          }
          break;
        default:
          break;
      }
    }
  }
 \endcode
 */
CUptiResult CUPTIAPI cuptiFinalize(void);

/**
 * \brief Push an external correlation id for the calling thread
 *
 * This function notifies CUPTI that the calling thread is entering an external API region.
 * When a CUPTI activity API record is created while within an external API region and
 * CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION is enabled, the activity API record will
 * be preceded by a CUpti_ActivityExternalCorrelation record for each \ref CUpti_ExternalCorrelationKind.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param id External correlation id.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER The external API kind is invalid
 */
CUptiResult CUPTIAPI cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t id);

/**
 * \brief Pop an external correlation id for the calling thread
 *
 * This function notifies CUPTI that the calling thread is leaving an external API region.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param lastId If the function returns successful, contains the last external correlation id for this \p kind, can be NULL.
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER The external API kind is invalid.
 * \retval CUPTI_ERROR_QUEUE_EMPTY No external id is currently associated with \p kind.
 */
CUptiResult CUPTIAPI cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind, uint64_t *lastId);

/**
 * \brief Controls the collection of queued and submitted timestamps for kernels.
 *
 * This API is used to control the collection of queued and submitted timestamps
 * for kernels whose records are provided through the struct \ref CUpti_ActivityKernel9.
 * Default value is 0, i.e. these timestamps are not collected. This API needs
 * to be called before initialization of CUDA and this setting should not be
 * changed during the profiling session.
 *
 * This API is not supported if the HW trace is enabled through the API \ref cuptiActivityEnableHWTrace.
 * \param enable is a boolean, denoting whether these timestamps should be
 * collected
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityEnableLatencyTimestamps(uint8_t enable);

/**
 * \brief Sets the flush period for the worker thread
 *
 * CUPTI creates a worker thread to minimize the perturbance for the application created
 * threads. CUPTI offloads certain operations from the application threads to the worker
 * thread, this includes synchronization of profiling resources between host and device,
 * delivery of the activity buffers to the client using the callback registered in
 * cuptiActivityRegisterCallbacks. For performance reasons, CUPTI wakes up the worker
 * thread based on certain heuristics.
 *
 * This API is used to control the flush period of the worker thread. This setting will
 * override the CUPTI heuristics. Setting time to zero disables the periodic flush and
 * restores the default behavior.
 *
 * Periodic flush can return only those activity buffers which are full and have all the
 * activity records completed.
 *
 * It's allowed to use the API \ref cuptiActivityFlushAll to flush the data on-demand, even
 * when client sets the periodic flush.
 *
 * \param time flush period in milliseconds (ms)
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 *
 * \see cuptiActivityFlushAll
 */
CUptiResult CUPTIAPI cuptiActivityFlushPeriod(uint32_t time);

/**
 * \brief Controls the collection of launch attributes for kernels.
 *
 * This API is used to control the collection of launch attributes for kernels whose
 * records are provided through the struct \ref CUpti_ActivityKernel9.
 * Default value is 0, i.e. these attributes are not collected.
 *
 * \param enable is a boolean denoting whether these launch attributes should be collected
 */
CUptiResult CUPTIAPI cuptiActivityEnableLaunchAttributes(uint8_t enable);

/**
 * \brief Function type for callback used by CUPTI to request a timestamp
 * to be used in activity records.
 *
 * This callback function signals the CUPTI client that a timestamp needs
 * to be returned. This timestamp would be treated as normalized timestamp
 * to be used for various purposes in CUPTI. For example to store start and
 * end timestamps reported in the CUPTI activity records.
 * The returned timestamp must be in nanoseconds.
 *
 * \sa ::cuptiActivityRegisterTimestampCallback
 */
typedef uint64_t (CUPTIAPI *CUpti_TimestampCallbackFunc)(void);

/**
 * \brief Registers callback function with CUPTI for providing timestamp.
 *
 * This function registers a callback function to obtain timestamp of user's
 * choice instead of using CUPTI provided timestamp.
 * By default CUPTI uses different methods, based on the underlying platform,
 * to retrieve the timestamp
 * Linux and Android use clock_gettime(CLOCK_REALTIME, ..)
 * Windows uses QueryPerformanceCounter()
 * QNX uses ClockCycles()
 * Timestamps retrieved using these methods are converted to nanosecond if needed
 * before usage.
 *
 * Timestamps for GPU activities such as kernels, memory copies and memset operations are
 * recorded directly on the GPU. To provide a unified and normalized view of these timestamps
 * in relation to CPU time, CUPTI performs a linear interpolation to convert GPU timestamps
 * into CPU timestamps during post-processing.
 * For activities where timestamps are captured on the GPU, the timestamp callback is invoked
 * during the post-processing phase, while converting GPU timestamps into CPU timestamps.
 * For activities for which timestamps are captured directly on the CPU, the timestamp callback
 * is invoked immediately at the time of the activity.
 *
 * The registration of timestamp callback should be done before any of the CUPTI
 * activity kinds are enabled to make sure that all the records report the timestamp using
 * the callback function registered through cuptiActivityRegisterTimestampCallback API.
 *
 * Changing the timestamp callback function in CUPTI through
 * cuptiActivityRegisterTimestampCallback API in the middle of the profiling
 * session can cause records generated prior to the change to report
 * timestamps through previous timestamp method.
 *
 * \param funcTimestamp callback which is invoked when a timestamp is
 * needed by CUPTI
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p funcTimestamp is NULL
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityRegisterTimestampCallback(CUpti_TimestampCallbackFunc funcTimestamp);

/**
 * \brief Controls the collection of records for device launched graphs.
 *
 * This API is used to control the collection of records for device launched graphs.
 * Default value is 0, i.e. these records are not collected. This API needs
 * to be called before initialization of CUDA and this setting should not be
 * changed during the profiling session.
 *
 * \param enable is a boolean, denoting whether these records should be
 * collected
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityEnableDeviceGraph(uint8_t enable);

/**
 * \brief Controls the collection of activity records for specific CUDA Driver APIs.
 *
 * Activity kind CUPTI_ACTIVITY_KIND_DRIVER controls the collection of either all
 * CUDA Driver APIs or none. API cuptiActivityEnableDriverApi can be used for fine-grained
 * control, it allows enabling/disabling tracing of a specific set of CUDA Driver APIs.
 * To disable collection of a small set of CUDA Driver APIs, user can
 * first enable the collection of all Driver APIs using the activity kind
 * CUPTI_ACTIVITY_KIND_DRIVER and call this API to disable specific Driver APIs.
 * And to enable the collection of a small set of CUDA Driver APIs, user can
 * call this API without using the activity kind CUPTI_ACTIVITY_KIND_DRIVER.
 *
 * Note: Activity kind CUPTI_ACTIVITY_KIND_DRIVER overrides the settings done by this API
 * if it is called after the API.
 *
 * \param cbid callback id of the CUDA Driver API. This can be found in the header cupti_driver_cbid.h.
 * \param enable is a boolean, denoting whether to enable or disable the collection
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityEnableDriverApi(CUpti_CallbackId cbid, uint8_t enable);

/**
 * \brief Controls the collection of activity records for specific CUDA Runtime APIs.
 *
 * Activity kind CUPTI_ACTIVITY_KIND_RUNTIME controls the collection of either all
 * CUDA Runtime APIs or none. API cuptiActivityEnableRuntimeApi can be used for fine-grained
 * control, it allows enabling/disabling tracing of a specific set of CUDA Runtime APIs.
 * To disable collection of a small set of CUDA Runtime APIs, user can
 * first enable the collection of all Runtime APIs using the activity kind
 * CUPTI_ACTIVITY_KIND_RUNTIME and call this API to disable specific Runtime APIs.
 * And to enable the collection of a small set of CUDA Runtime APIs, user can
 * call this API without using the activity kind CUPTI_ACTIVITY_KIND_RUNTIME.
 *
 * Note: Activity kind CUPTI_ACTIVITY_KIND_RUNTIME overrides the settings done by this API
 * if it is called after the API.
 *
 * \param cbid callback id of the CUDA Runtime API. This can be found in the header cupti_runtime_cbid.h.
 * \param enable is a boolean, denoting whether to enable or disable the collection
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityEnableRuntimeApi(CUpti_CallbackId cbid, uint8_t enable);

/**
 * \brief Enables the collection of CUDA kernel timestamps through Hardware Event System(HES).
 *
 * This API enables the collection of CUDA kernel timestamps through HW events instead
 * of the traditional SW instrumentation and semaphore based approach.
 * This option is only available on Blackwell architecture.
 * This API should be called after driver is initialized.
 *
 * \param enable is a boolean, denoting whether to enable or disable the collection through HW events
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED if CUPTI is not initialized or the CUDA driver is not initialized
 * \retval CUPTI_ERROR_NOT_SUPPORTED if HW trace cannot be enabled on the current platform
 * \retval CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED
 * \retval CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED
 * \retval CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED
 * \retval CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED
 * \retval CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED
 * \retval CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED
 */
CUptiResult CUPTIAPI cuptiActivityEnableHWTrace(uint8_t enable);


/**
 *  \brief Enables tracking the source library for memory allocation requests.
 *
 * This API is used to control whether or not we track the source library of
 * memory allocation requests. Default value is 0, i.e. it is not tracked. The
 * activity kind CUPTI_ACTIVITY_KIND_MEMORY2 needs to be enabled, and if this flag is
 * set, we get the full path of the shared object responsible for the GPU memory allocation
 * request in the member source in the CUpti_ActivityMemory4 records. Also note that this feature
 * adds runtime overhead.
 *
 * \param enable is a boolean, denoting whether the source library of the memory allocation
 * request needs to be tracked
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
*/
CUptiResult CUPTIAPI cuptiActivityEnableAllocationSource (uint8_t enable);

/**
 * \brief Enables collecting records for all synchronization operations.
 *
 * CUPTI provides CUDA event query and stream query records via CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.
 * Using this API, CUPTI client can disable to record CUDA event query and stream query records
 * for queries for which the operations have not yet been completed on the CUDA event/stream.
 *
 * By default, the record is generated for all CUDA events and stream irrespective of whether the
 * operations have been completed on the CUDA event/stream.
 *
 * \param enable is a boolean, denoting whether to enable or disable the collection of all CUDA event query
 * and stream query records
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 */
CUptiResult CUPTIAPI cuptiActivityEnableAllSyncRecords(uint8_t enable);

/** @} */ /* END CUPTI_ACTIVITY_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

// Including deprecated structures of CUPTI_ACTIVITY_API
// #include "cupti_activity_deprecated.h"

/*
 * Copyright 2011-2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(_CUPTI_ACTIVITY_DEPRECATED_H_)
#define _CUPTI_ACTIVITY_DEPRECATED_H_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \brief The kinds of activity records.
 *
 * Each activity record kind represents information about a GPU or an
 * activity occurring on a CPU or GPU. Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 * \see CUpti_ActivityOverhead
 * \see CUpti_ActivityOverhead2
 * \see CUpti_ActivityDevice
 * \see CUpti_ActivityDevice2
 * \see CUpti_ActivityDevice3
 * \see CUpti_ActivityDevice4
 * \see CUpti_ActivityKernel
 * \see CUpti_ActivityKernel2
 * \see CUpti_ActivityKernel3
 * \see CUpti_ActivityKernel4
 * \see CUpti_ActivityKernel5
 * \see CUpti_ActivityKernel6
 * \see CUpti_ActivityKernel7
 * \see CUpti_ActivityKernel8
 * \see CUpti_ActivityMemcpy
 * \see CUpti_ActivityMemcpy3
 * \see CUpti_ActivityMemcpy4
 * \see CUpti_ActivityMemcpyPtoP
 * \see CUpti_ActivityMemcpyPtoP2
 * \see CUpti_ActivityMemcpyPtoP3
 * \see CUpti_ActivityMemset
 * \see CUpti_ActivityMemset2
 * \see CUpti_ActivityMemset3
 * \see CUpti_ActivityMemory2
 * \see CUpti_ActivityMemory3
 * \see CUpti_ActivityMemoryPool
 * \see CUpti_ActivityMarker
 * \see CUpti_ActivityGlobalAccess
 * \see CUpti_ActivityGlobalAccess2
 * \see CUpti_ActivityBranch
 * \see CUpti_ActivityPCSampling
 * \see CUpti_ActivityPCSampling2
 * \see CUpti_ActivityUnifiedMemoryCounter
 * \see CUpti_ActivityUnifiedMemoryCounter2
 * \see CUpti_ActivityNvLink
 * \see CUpti_ActivityNvLink2
 * \see CUpti_ActivityNvLink3
 */

/**
 * \brief The activity record for CUPTI and driver overheads.
 * (Deprecated in CUDA 12.2)
 *
 * This activity record provides CUPTI and driver overhead information
 * (CUPTI_ACTIVITY_OVERHEAD). These records are now reported using
 * CUpti_ActivityOverhead3
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of overhead, CUPTI, DRIVER, COMPILER etc.
   */
  CUpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;
} CUpti_ActivityOverhead;

/**
 * \brief The activity record for CUPTI and driver overheads.
 *
 * This activity record provides CUPTI and driver overhead information
 * (CUPTI_ACTIVITY_OVERHEAD).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_OVERHEAD.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of overhead, CUPTI, DRIVER, COMPILER etc.
   */
  CUpti_ActivityOverheadKind overheadKind;

  /**
   * The kind of activity object that the overhead is associated with.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object. 'objectKind' indicates
   * which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

  /**
   * The start timestamp for the overhead, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the overhead.
   */
  uint64_t start;

  /**
   * The end timestamp for the overhead, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the overhead.
   */
  uint64_t end;

  /**
   * The correlation ID of the overhead operation to which
   * records belong to. This ID is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the overhead operation.
   * In some cases, it can be zero, such as for CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH records.
   */
  uint32_t correlationId;

  /**
   * Reserved for internal use.
   */
  uint32_t reserved0;
} CUpti_ActivityOverhead2;

/**
 * \brief The activity record for a device. (deprecated)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityDevice;

/**
 * \brief The activity record for a device. (deprecated)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;
} CUpti_ActivityDevice2;

/**
 * \brief The activity record for a device. (CUDA 7.0 onwards)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Flag to indicate whether the device is visible to CUDA. Users can
   * set the device visibility using CUDA_VISIBLE_DEVICES environment
   */
  uint8_t isCudaVisible;

  uint8_t reserved[7];
} CUpti_ActivityDevice3;

/**
 * \brief The activity record for a device. (CUDA 11.6 onwards)
 *
 * This activity record represents information about a GPU device
 * (CUPTI_ACTIVITY_KIND_DEVICE).
 * Device activity is now reported using the
 * CUpti_ActivityDevice5 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_DEVICE.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the device. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The global memory bandwidth available on the device, in
   * kBytes/sec.
   */
  uint64_t globalMemoryBandwidth;

  /**
   * The amount of global memory on the device, in bytes.
   */
  uint64_t globalMemorySize;

  /**
   * The amount of constant memory on the device, in bytes.
   */
  uint32_t constantMemorySize;

  /**
   * The size of the L2 cache on the device, in bytes.
   */
  uint32_t l2CacheSize;

  /**
   * The number of threads per warp on the device.
   */
  uint32_t numThreadsPerWarp;

  /**
   * The core clock rate of the device, in kHz.
   */
  uint32_t coreClockRate;

  /**
   * Number of memory copy engines on the device.
   */
  uint32_t numMemcpyEngines;

  /**
   * Number of multiprocessors on the device.
   */
  uint32_t numMultiprocessors;

  /**
   * The maximum "instructions per cycle" possible on each device
   * multiprocessor.
   */
  uint32_t maxIPC;

  /**
   * Maximum number of warps that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxWarpsPerMultiprocessor;

  /**
   * Maximum number of blocks that can be present on a multiprocessor
   * at any given time.
   */
  uint32_t maxBlocksPerMultiprocessor;

  /**
   * Maximum amount of shared memory available per multiprocessor, in bytes.
   */
  uint32_t maxSharedMemoryPerMultiprocessor;

  /**
   * Maximum number of 32-bit registers available per multiprocessor.
   */
  uint32_t maxRegistersPerMultiprocessor;

  /**
   * Maximum number of registers that can be allocated to a block.
   */
  uint32_t maxRegistersPerBlock;

  /**
   * Maximum amount of shared memory that can be assigned to a block,
   * in bytes.
   */
  uint32_t maxSharedMemoryPerBlock;

  /**
   * Maximum number of threads allowed in a block.
   */
  uint32_t maxThreadsPerBlock;

  /**
   * Maximum allowed X dimension for a block.
   */
  uint32_t maxBlockDimX;

  /**
   * Maximum allowed Y dimension for a block.
   */
  uint32_t maxBlockDimY;

  /**
   * Maximum allowed Z dimension for a block.
   */
  uint32_t maxBlockDimZ;

  /**
   * Maximum allowed X dimension for a grid.
   */
  uint32_t maxGridDimX;

  /**
   * Maximum allowed Y dimension for a grid.
   */
  uint32_t maxGridDimY;

  /**
   * Maximum allowed Z dimension for a grid.
   */
  uint32_t maxGridDimZ;

  /**
   * Compute capability for the device, major number.
   */
  uint32_t computeCapabilityMajor;

  /**
   * Compute capability for the device, minor number.
   */
  uint32_t computeCapabilityMinor;

  /**
   * The device ID.
   */
  uint32_t id;

  /**
   * ECC enabled flag for device
   */
  uint32_t eccEnabled;

  /**
   * The device UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid uuid;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The device name. This name is shared across all activity records
   * representing instances of the device, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Flag to indicate whether the device is visible to CUDA. Users can
   * set the device visibility using CUDA_VISIBLE_DEVICES environment
   */
  uint8_t isCudaVisible;

  /**
   * MIG enabled flag for device
   */
  uint8_t isMigEnabled;

  uint8_t reserved[6];

  /**
   * GPU Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t gpuInstanceId;

  /**
   * Compute Instance id for MIG enabled devices.
   * If mig mode is disabled value is set to UINT32_MAX
   */
  uint32_t computeInstanceId;

  /**
   * The MIG UUID. This value is the globally unique immutable
   * alphanumeric identifier of the device.
   */
  CUuuid migUuid;

} CUpti_ActivityDevice4;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL
   * or CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * The cache configuration requested by the kernel. The value is one
   * of the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigRequested;

  /**
   * The cache configuration used for the kernel. The value is one of
   * the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigExecuted;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the kernel. Each kernel execution
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the kernel.
   */
  uint32_t runtimeCorrelationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel2;

/**
 * \brief The activity record for a kernel (CUDA 6.5(with sm_52 support) onwards).
 * (deprecated in CUDA 9.0)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL).
 * Kernel activities are now reported using the CUpti_ActivityKernel9 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityKernel3;

/**
 * \brief The activity record for a kernel (CUDA 9.0(with sm_70 support) onwards).
 * (deprecated in CUDA 11.0)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL).
 * Kernel activities are now reported using the CUpti_ActivityKernel9 activity
 * record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;
} CUpti_ActivityKernel4;

/**
 * \brief The activity record for a kernel (CUDA 11.0(with sm_80 support) onwards).
 * (deprecated in CUDA 11.2)
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;
} CUpti_ActivityKernel5;

/**
 * \brief The activity record for kernel. (deprecated in CUDA 11.6)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;
} CUpti_ActivityKernel6;

/**
 * \brief The activity record for kernel. (deprecated in CUDA 11.8)
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by CUPTI. Kernel activities are now reported using the
 * CUpti_ActivityKernel9 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;

  /**
   * The ID of the HW channel on which the kernel is launched.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;
} CUpti_ActivityKernel7;

/**
 * \brief The activity record for kernel.
 *
 * This activity record represents a kernel execution
 * (CUPTI_ACTIVITY_KIND_KERNEL and
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_KERNEL or
   * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  CUpti_ActivityKind kind;

  /**
   * For devices with compute capability 7.0+ cacheConfig values are not updated
   * in case field isSharedMemoryCarveoutRequested is set
   */
  union {
    uint8_t both;
    struct {
      /**
       * The cache configuration requested by the kernel. The value is one
       * of the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t requested:4;

      /**
       * The cache configuration used for the kernel. The value is one of
       * the CUfunc_cache enumeration values from cuda.h.
       */
      uint8_t executed:4;
    } config;
  } cacheConfig;

  /**
   * The shared memory configuration used for the kernel. The value is one of
   * the CUsharedconfig enumeration values from cuda.h.
   */
  uint8_t sharedMemoryConfig;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The partitioned global caching requested for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheRequested;

  /**
   * The partitioned global caching executed for the kernel. Partitioned
   * global caching is required to enable caching on certain chips, such as
   * devices with compute capability 5.2. Partitioned global caching can be
   * automatically disabled if the occupancy requirement of the launch cannot
   * support caching.
   */
  CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns.  It
   * represents the completion of all it's child kernels and the
   * kernel itself. A value of CUPTI_TIMESTAMP_UNKNOWN indicates that
   * the completion time is unknown.
   */
  uint64_t completed;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes (deprecated in CUDA 11.8).
   * Refer field localMemoryTotal_v2
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the kernel.
   */
  uint32_t correlationId;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique
   * grid ID at runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The timestamp when the kernel is queued up in the command buffer, in ns.
   * A value of CUPTI_TIMESTAMP_UNKNOWN indicates that the queued time
   * could not be collected for the kernel. This timestamp is not collected
   * by default. Use API \ref cuptiActivityEnableLatencyTimestamps() to
   * enable collection.
   *
   * Command buffer is a buffer written by CUDA driver to send commands
   * like kernel launch, memory copy etc to the GPU. All launches of CUDA
   * kernels are asynchronous with respect to the host, the host requests
   * the launch by writing commands into the command buffer, then returns
   * without checking the GPU's progress.
   */
  uint64_t queued;

  /**
   * The timestamp when the command buffer containing the kernel launch
   * is submitted to the GPU, in ns. A value of CUPTI_TIMESTAMP_UNKNOWN
   * indicates that the submitted time could not be collected for the kernel.
   * This timestamp is not collected by default. Use API \ref
   * cuptiActivityEnableLatencyTimestamps() to enable collection.
   */
  uint64_t submitted;

  /**
   * The indicates if the kernel was executed via a regular launch or via a
   * single/multi device cooperative launch. \see CUpti_ActivityLaunchType
   */
  uint8_t launchType;

  /**
   * This indicates if CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT was
   * updated for the kernel launch
   */
  uint8_t isSharedMemoryCarveoutRequested;

  /**
   * Shared memory carveout value requested for the function in percentage of
   * the total resource. The value will be updated only if field
   * isSharedMemoryCarveoutRequested is set.
   */
  uint8_t sharedMemoryCarveoutRequested;

  /**
   * Undefined. Reserved for internal use.
   */
  uint8_t padding;

 /**
  * Shared memory size set by the driver.
  */
  uint32_t sharedMemoryExecuted;

  /**
   * The unique ID of the graph node that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint64_t graphNodeId;

  /**
   * The shared memory limit config for the kernel. This field shows whether user has opted for a
   * higher per block limit of dynamic shared memory.
   */
  CUpti_FuncShmemLimitConfig shmemLimitConfig;

  /**
   * The unique ID of the graph that launched this kernel through graph launch APIs.
   * This field will be 0 if the kernel is not launched through graph launch APIs.
   */
  uint32_t graphId;

  /**
   * The pointer to the access policy window. The structure CUaccessPolicyWindow is
   * defined in cuda.h.
   */
  CUaccessPolicyWindow *pAccessPolicyWindow;

  /**
   * The ID of the HW channel on which the kernel is launched.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   * The X-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterX;

  /**
   * The Y-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterY;

  /**
   * The Z-dimension cluster size for the kernel.
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterZ;

  /**
   * The cluster scheduling policy for the kernel. Refer CUclusterSchedulingPolicy
   * Field is valid for devices with compute capability 9.0 and higher
   */
  uint32_t clusterSchedulingPolicy;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint64_t localMemoryTotal_v2;
} CUpti_ActivityKernel8;

/**
 * \brief The activity record for memory copies. (deprecated)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemcpy;

/**
 * \brief The activity record for memory copies. (deprecated in CUDA 11.1)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemcpy3;

/**
 * \brief The activity record for memory copies. (deprecated in CUDA 11.6)
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemcpy4;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2) but is no longer generated
 * by CUPTI. Peer-to-peer memory copy activities are now reported using the
 * CUpti_ActivityMemcpyPtoP2 activity record..
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemcpyPtoP;

typedef CUpti_ActivityMemcpyPtoP CUpti_ActivityMemcpy2;

/**
 * \brief The activity record for peer-to-peer memory copies.
 * (deprecated in CUDA 11.1)
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemcpyPtoP2;

/**
 * \brief The activity record for peer-to-peer memory copies.
 * (deprecated in CUDA 11.6)
 *
 * This activity record represents a peer-to-peer memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY2).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY2.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed the memcpy through graph launch.
   * This field will be 0 if memcpy is not done using graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemcpyPtoP3;

/**
 * \brief The activity record for memset. (deprecated)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} CUpti_ActivityMemset;

/**
 * \brief The activity record for memset. (deprecated in CUDA 11.1)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint64_t graphNodeId;
} CUpti_ActivityMemset2;

/**
 * \brief The activity record for memset. (deprecated in CUDA 11.6)
 *
 * This activity record represents a memory set operation
 * (CUPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMSET.
   */
  CUpti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see CUpti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see CUpti_ActivityMemoryKind
   */
  uint16_t memoryKind;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memset through graph launch.
   * This field will be 0 if the memset is not executed through graph launch.
   */
  uint32_t graphId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t padding;
} CUpti_ActivityMemset3;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY2).
 * This activity record provides separate records for memory allocation and
 * memory release operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory operation.
 *
 * Note: This activity record is an upgrade over \ref CUpti_ActivityMemory
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY.
 * \ref CUpti_ActivityMemory provides a single record for the memory
 * allocation and memory release operations.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY2
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryOperationType.
   */
  CUpti_ActivityMemoryOperationType memoryOperationType;

  /**
   * The memory kind requested by the user, \ref CUpti_ActivityMemoryKind.
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The correlation ID of the memory operation. Each memory operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The program counter of the memory operation.
   */
  uint64_t PC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory operation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

  /**
   * The ID of the stream. If memory operation is not async, \p streamId is set to CUPTI_INVALID_STREAM_ID.
   */
  uint32_t streamId;

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;

  /**
   * \p isAsync is set if memory operation happens through async memory APIs.
   */
  uint32_t isAsync;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * The memory pool configuration used for the memory operations.
   */
  struct {
    /**
     * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
     */
    CUpti_ActivityMemoryPoolType memoryPoolType;

#ifdef CUPTILP64
    /**
     * Undefined. Reserved for internal use.
     */
    uint32_t pad2;
#endif

    /**
     * The base address of the memory pool.
     */
    uint64_t address;

    /**
     * The release threshold of the memory pool in bytes. \p releaseThreshold is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t releaseThreshold;

   /**
   * The size of the memory pool in bytes and the processID of the memory pool.
   * \p size is valid if \p memoryPoolType is
   * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   * \p processId is valid if \p memoryPoolType is
   * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED, \ref CUpti_ActivityMemoryPoolType.
   */
   union {
      uint64_t size;
      uint64_t processId;
    } pool;
  } memoryPoolConfig;

} CUpti_ActivityMemory2;

/**
 * \brief The activity record for memory.
 *
 * This activity record represents a memory allocation and free operation
 * (CUPTI_ACTIVITY_KIND_MEMORY2).
 * This activity record provides separate records for memory allocation and
 * memory release operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory operation.
 *
 * Note: This activity record is an upgrade over \ref CUpti_ActivityMemory2
 * enabled using the kind \ref CUPTI_ACTIVITY_KIND_MEMORY.
 * \ref CUpti_ActivityMemory provides a single record for the memory
 * allocation and memory release operations.
 */

typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY2
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryOperationType.
   */
  CUpti_ActivityMemoryOperationType memoryOperationType;

  /**
   * The memory kind requested by the user, \ref CUpti_ActivityMemoryKind.
   */
  CUpti_ActivityMemoryKind memoryKind;

  /**
   * The correlation ID of the memory operation. Each memory operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The number of bytes of memory allocated.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;

  /**
   * The program counter of the memory operation.
   */
  uint64_t PC;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory operation is taking place.
   */
  uint32_t deviceId;

  /**
   * The ID of the context. If context is NULL, \p contextId is set to CUPTI_INVALID_CONTEXT_ID.
   */
  uint32_t contextId;

  /**
   * The ID of the stream. If memory operation is not async, \p streamId is set to CUPTI_INVALID_STREAM_ID.
   */
  uint32_t streamId;

  /**
   * Variable name. This name is shared across all activity
   * records representing the same symbol, and so should not be
   * modified.
   */
  const char* name;

  /**
   * \p isAsync is set if memory operation happens through async memory APIs.
   */
  uint32_t isAsync;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad1;
#endif

  /**
   * The memory pool configuration used for the memory operations.
   */
  struct PACKED_ALIGNMENT {
    /**
     * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
     */
    CUpti_ActivityMemoryPoolType memoryPoolType;

#ifdef CUPTILP64
    /**
     * Undefined. Reserved for internal use.
     */
    uint32_t pad2;
#endif

    /**
     * The base address of the memory pool.
     */
    uint64_t address;

    /**
     * The release threshold of the memory pool in bytes. \p releaseThreshold is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t releaseThreshold;

    /**
     * The size of memory pool in bytes and the processId of the memory pools
     * \p size is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     * \p processId is valid if \p memoryPoolType is
     * CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED, \ref CUpti_ActivityMemoryPoolType
     */
    union {
      uint64_t size;
      uint64_t processId;
    } pool;

    /**
     * The utilized size of the memory pool. \p utilizedSize is
     * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
     */
    uint64_t utilizedSize;
  } memoryPoolConfig;

} CUpti_ActivityMemory3;

/**
 * \brief The activity record for memory pool.
 *
 * This activity record represents a memory pool creation, destruction and
 * trimming (CUPTI_ACTIVITY_KIND_MEMORY_POOL).
 * This activity record provides separate records for memory pool creation,
 * destruction and trimming operations.
 * This allows to correlate the corresponding driver and runtime API
 * activity record with the memory pool operation.
 *
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMORY_POOL
   */
  CUpti_ActivityKind kind;

  /**
   * The memory operation requested by the user, \ref CUpti_ActivityMemoryPoolOperationType.
   */
  CUpti_ActivityMemoryPoolOperationType memoryPoolOperationType;

  /**
   * The type of the memory pool, \ref CUpti_ActivityMemoryPoolType
   */
  CUpti_ActivityMemoryPoolType memoryPoolType;

  /**
   * The correlation ID of the memory pool operation. Each memory pool
   * operation is assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory operation.
   */
  uint32_t correlationId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The ID of the device where the memory pool is created.
   */
  uint32_t deviceId;

  /**
   * The minimum bytes to keep of the memory pool. \p minBytesToKeep is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_OPERATION_TYPE_TRIMMED,
   * \ref CUpti_ActivityMemoryPoolOperationType
   */
  size_t minBytesToKeep;

#ifndef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The virtual address of the allocation.
   */
  uint64_t address;

  /**
   * The size of the memory pool operation in bytes. \p size is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t size;

  /**
   * The release threshold of the memory pool. \p releaseThreshold is
   * valid for CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL, \ref CUpti_ActivityMemoryPoolType.
   */
  uint64_t releaseThreshold;

  /**
   * The start timestamp for the memory operation, in ns.
   */
  uint64_t timestamp;
} CUpti_ActivityMemoryPool;

/**
 * \brief The activity record providing a marker which is an
 * instantaneous point in time. (deprecated in CUDA 8.0)
 *
 * The marker is specified with a descriptive name and unique id
 * (CUPTI_ACTIVITY_KIND_MARKER).
 * Marker activity is now reported using the
 * CUpti_ActivityMarker2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MARKER.
   */
  CUpti_ActivityKind kind;

  /**
   * The flags associated with the marker. \see CUpti_ActivityFlag
   */
  CUpti_ActivityFlag flags;

  /**
   * The timestamp for the marker, in ns. A value of 0 indicates that
   * timestamp information could not be collected for the marker.
   */
  uint64_t timestamp;

  /**
   * The marker ID.
   */
  uint32_t id;

  /**
   * The kind of activity object associated with this marker.
   */
  CUpti_ActivityObjectKind objectKind;

  /**
   * The identifier for the activity object associated with this
   * marker. 'objectKind' indicates which ID is valid for this record.
   */
  CUpti_ActivityObjectKindId objectId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * The marker name for an instantaneous or start marker. This will
   * be NULL for an end marker.
   */
  const char *name;

} CUpti_ActivityMarker;

/**
 * \brief The activity record for source-level global
 * access. (deprecated)
 *
 * This activity records the locations of the global
 * accesses in the source (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * CUpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;
} CUpti_ActivityGlobalAccess;

/**
 * \brief The activity record for source-level global
 * access. (deprecated in CUDA 9.0)
 *
 * This activity records the locations of the global
 * accesses in the source (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS).
 * Global access activities are now reported using the
 * CUpti_ActivityGlobalAccess3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this global access.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the access.
   */
  uint32_t pcOffset;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction with predicate and condition code evaluating to true.
   */
  uint64_t threadsExecuted;

  /**
   * The total number of 32 bytes transactions to L2 cache generated by this access
   */
  uint64_t l2_transactions;

  /**
   * The minimum number of L2 transactions possible based on the access pattern.
   */
  uint64_t theoreticalL2Transactions;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * when at least one of thread among warp is active with predicate and condition code
   * evaluating to true.
   */
  uint32_t executed;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityGlobalAccess2;

/**
 * \brief The activity record for source level result
 * branch. (deprecated)
 *
 * This activity record the locations of the branches in the
 * source (CUPTI_ACTIVITY_KIND_BRANCH).
 * Branch activities are now reported using the
 * CUpti_ActivityBranch2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_BRANCH.
   */
  CUpti_ActivityKind kind;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The pc offset for the branch.
   */
  uint32_t pcOffset;

  /**
   * The number of times this instruction was executed per warp. It will be incremented
   * regardless of predicate or condition code.
   */
  uint32_t executed;

  /**
   * Number of times this branch diverged
   */
  uint32_t diverged;

  /**
   * This increments each time when this instruction is executed by number
   * of threads that executed this instruction
   */
  uint64_t threadsExecuted;
} CUpti_ActivityBranch;

/**
 * \brief The activity record for PC sampling. (deprecated in CUDA 8.0)
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * CUpti_ActivityPCSampling2 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref CUpti_ActivityPCSamplingStallReason
   */
  CUpti_ActivityPCSamplingStallReason stallReason;
} CUpti_ActivityPCSampling;

/**
 * \brief The activity record for PC sampling. (deprecated in CUDA 9.0)
 *
 * This activity records information obtained by sampling PC
 * (CUPTI_ACTIVITY_KIND_PC_SAMPLING).
 * PC sampling activities are now reported using the
 * CUpti_ActivityPCSampling3 activity record.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_PC_SAMPLING.
   */
  CUpti_ActivityKind kind;

  /**
   * The properties of this instruction.
   */
  CUpti_ActivityFlag flags;

  /**
   * The ID for source locator.
   */
  uint32_t sourceLocatorId;

  /**
   * The correlation ID of the kernel to which this result is associated.
   */
  uint32_t correlationId;

  /**
  * Correlation ID with global/device function name
  */
  uint32_t functionId;

  /**
   * The pc offset for the instruction.
   */
  uint32_t pcOffset;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * These samples indicate that no instruction was issued in that cycle from
   * the warp scheduler from where the warp was sampled.
   * Field is valid for devices with compute capability 6.0 and higher
   */
  uint32_t latencySamples;

  /**
   * Number of times the PC was sampled with the stallReason in the record.
   * The same PC can be sampled with different stall reasons. The count includes
   * latencySamples.
   */
  uint32_t samples;

  /**
   * Current stall reason. Includes one of the reasons from
   * \ref CUpti_ActivityPCSamplingStallReason
   */
  CUpti_ActivityPCSamplingStallReason stallReason;

  uint32_t pad;
} CUpti_ActivityPCSampling2;

/**
 * \brief The activity record for Unified Memory counters (deprecated in CUDA 7.0)
 *
 * This activity record represents a Unified Memory counter
 * (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind. See \ref CUpti_ActivityUnifiedMemoryCounterKind
   */
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Scope of the Unified Memory counter. See \ref CUpti_ActivityUnifiedMemoryCounterScope
   */
  CUpti_ActivityUnifiedMemoryCounterScope scope;

  /**
   * The ID of the device involved in the memory transfer operation.
   * It is not relevant if the scope of the counter is global (all devices).
   */
  uint32_t deviceId;

  /**
   * Value of the counter
   *
   */
  uint64_t value;

  /**
   * The timestamp when this sample was retrieved, in ns. A value of 0
   * indicates that timestamp information could not be collected
   */
  uint64_t timestamp;

  /**
   * The ID of the process to which this record belongs to. In case of
   * global scope, processId is undefined.
   */
  uint32_t processId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityUnifiedMemoryCounter;

/**
 * \brief The activity record for Unified Memory counters (deprecated in 12.8)
 *
 * This activity record represents a Unified Memory counter
 * (CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
   */
  CUpti_ActivityKind kind;

  /**
   * The Unified Memory counter kind
   */
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;

  /**
   * Value of the counter
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THREASHING and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP, it is the size of the
   * memory region in bytes.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, it
   * is the number of page fault groups for the same page.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
   * it is the program counter for the instruction that caused fault.
   */
  uint64_t value;

  /**
   * The start timestamp of the counter, in ns.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity starts on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT, timestamp is
   * captured when CUDA driver started processing the fault.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, timestamp
   * is captured when CUDA driver detected thrashing of memory region.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING,
   * timestamp is captured when throttling operation was started by CUDA driver.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP,
   * timestamp is captured when CUDA driver has pushed all required operations
   * to the processor specified by dstId.
   */
  uint64_t start;

  /**
   * The end timestamp of the counter, in ns.
   * Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD and
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, timestamp is
   * captured when activity finishes on GPU.
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, timestamp is
   * captured when CUDA driver queues the replay of faulting memory accesses on the GPU
   * For counterKind CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, timestamp
   * is captured when throttling operation was finished by CUDA driver
   */
  uint64_t end;

  /**
   * This is the virtual base address of the page/s being transferred. For cpu and
   * gpu faults, the virtual address for the page that faulted.
   */
  uint64_t address;

  /**
   * The ID of the source CPU/device involved in the memory transfer, page fault, thrashing,
   * throttling or remote map operation. For counterKind
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, it is a bitwise ORing of the
   * device IDs fighting for the memory region. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT
   */
  uint32_t srcId;

  /**
   * The ID of the destination CPU/device involved in the memory transfer or remote map
   * operation. Ignore this field if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t dstId;

  /**
   * The ID of the stream causing the transfer.
   * This value of this field is invalid.
   */
  uint32_t streamId;

  /**
   * The ID of the process to which this record belongs to.
   */
  uint32_t processId;

  /**
   * The flags associated with this record. See enums \ref CUpti_ActivityUnifiedMemoryAccessType
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
   * and \ref CUpti_ActivityUnifiedMemoryMigrationCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD
   * and \ref CUpti_ActivityUnifiedMemoryRemoteMapCause if counterKind is
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP and \ref CUpti_ActivityFlag
   * if counterKind is CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING or
   * CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
   */
  uint32_t flags;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityUnifiedMemoryCounter2;

/**
* \brief NVLink information. (deprecated in CUDA 9.0)
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NVLink information are now reported using the
* CUpti_ActivityNvLink2 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
  * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
  */
  CUpti_ActivityKind kind;

  /**
  * NVLink version.
  */
  uint32_t nvlinkVersion;

  /**
  * Type of device 0 \ref CUpti_DevType
  */
  CUpti_DevType typeDev0;

  /**
  * Type of device 1 \ref CUpti_DevType
  */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
      * Index of the NPU. First index will always be zero.
      */
      uint32_t index;

      /**
      * Domain ID of NPU. On Linux, this can be queried using lspci.
      */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
      * Index of the NPU. First index will always be zero.
      */
      uint32_t index;

      /**
      * Domain ID of NPU. On Linux, this can be queried using lspci.
      */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
  * Flag gives capabilities of the link \see CUpti_LinkFlag
  */
  uint32_t flag;

  /**
  * Number of physical NVLinks present between two devices.
  */
  uint32_t physicalNvLinkCount;

  /**
  * Port numbers for maximum 4 NVLinks connected to device 0.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
  * In case of invalid/unknown port number, this field will be set
  * to value CUPTI_NVLINK_INVALID_PORT.
  * This will be used to correlate the metric values to individual
  * physical link and attribute traffic to the logical NVLink in
  * the topology.
  */
  int8_t portDev0[4];

  /**
  * Port numbers for maximum 4 NVLinks connected to device 1.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
  * In case of invalid/unknown port number, this field will be set
  * to value CUPTI_NVLINK_INVALID_PORT.
  * This will be used to correlate the metric values to individual
  * physical link and attribute traffic to the logical NVLink in
  * the topology.
  */
  int8_t portDev1[4];

  /**
  * Bandwidth of NVLink in kbytes/sec
  */
  uint64_t bandwidth;
} CUpti_ActivityNvLink;

/**
* \brief NVLink information. (deprecated in CUDA 10.0)
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NvLink information are now reported using the
* CUpti_ActivityNvLink4 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
   */
  CUpti_ActivityKind kind;

  /**
   * NvLink version.
   */
  uint32_t nvlinkVersion;

  /**
   * Type of device 0 \ref CUpti_DevType
   */
  CUpti_DevType typeDev0;

  /**
   * Type of device 1 \ref CUpti_DevType
   */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
   * Flag gives capabilities of the link \see CUpti_LinkFlag
   */
  uint32_t flag;

  /**
   * Number of physical NVLinks present between two devices.
   */
  uint32_t physicalNvLinkCount;

  /**
   * Port numbers for maximum 16 NVLinks connected to device 0.
   * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev0[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Port numbers for maximum 16 NVLinks connected to device 1.
   * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev1[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Bandwidth of NVLink in kbytes/sec
   */
  uint64_t  bandwidth;
} CUpti_ActivityNvLink2;

/**
* \brief NVLink information.
*
* This structure gives capabilities of each logical NVLink connection between two devices,
* gpu<->gpu or gpu<->CPU which can be used to understand the topology.
* NvLink information are now reported using the
* CUpti_ActivityNvLink4 activity record.
*/
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_NVLINK.
   */
  CUpti_ActivityKind kind;
  /**
   * NvLink version.
   */
  uint32_t nvlinkVersion;

  /**
   * Type of device 0 \ref CUpti_DevType
   */
  CUpti_DevType typeDev0;

  /**
   * Type of device 1 \ref CUpti_DevType
   */
  CUpti_DevType typeDev1;

  /**
  * If typeDev0 is CUPTI_DEV_TYPE_GPU, UUID for device 0. \ref CUpti_ActivityDevice5.
  * If typeDev0 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev0;

  /**
  * If typeDev1 is CUPTI_DEV_TYPE_GPU, UUID for device 1. \ref CUpti_ActivityDevice5.
  * If typeDev1 is CUPTI_DEV_TYPE_NPU, struct npu for NPU.
  */
  union {
    CUuuid uuidDev;
    struct {
      /**
       * Index of the NPU. First index will always be zero.
       */
      uint32_t index;

      /**
       * Domain ID of NPU. On Linux, this can be queried using lspci.
       */
      uint32_t domainId;
    } npu;
  } idDev1;

  /**
   * Flag gives capabilities of the link \see CUpti_LinkFlag
   */
  uint32_t flag;

  /**
   * Number of physical NVLinks present between two devices.
   */
  uint32_t physicalNvLinkCount;

  /**
   * Port numbers for maximum 16 NVLinks connected to device 0.
   * If typeDev0 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev0[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Port numbers for maximum 16 NVLinks connected to device 1.
   * If typeDev1 is CUPTI_DEV_TYPE_NPU, ignore this field.
   * In case of invalid/unknown port number, this field will be set
   * to value CUPTI_NVLINK_INVALID_PORT.
   * This will be used to correlate the metric values to individual
   * physical link and attribute traffic to the logical NVLink in
   * the topology.
   */
  int8_t portDev1[CUPTI_MAX_NVLINK_PORTS];

  /**
   * Bandwidth of NVLink in kbytes/sec
   */
  uint64_t bandwidth;

  /**
   * NVSwitch is connected as an intermediate node.
   */
  uint8_t nvswitchConnected;

  /**
   * Undefined. reserved for internal use
   */
  uint8_t pad[7];
} CUpti_ActivityNvLink3;

/**
 * \brief The activity record for trace of graph execution.
 *
 * This activity record represents execution for a graph without giving visibility
 * about the execution of its nodes. This is intended to reduce overheads in tracing
 * each node. The activity kind is CUPTI_ACTIVITY_KIND_GRAPH_TRACE
 * Graph trace activity is now reported using CUpti_ActivityGraphTrace2 record.
 */
typedef struct {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_GRAPH_TRACE
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the graph launch. Each graph launch is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the graph.
   */
  uint32_t correlationId;

  /**
   * The start timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t start;

  /**
   * The end timestamp for the graph execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the graph.
   */
  uint64_t end;

  /**
   * The ID of the device where the graph execution is occurring.
   */
  uint32_t deviceId;

  /**
   * The unique ID of the graph that is launched.
   */
  uint32_t graphId;

  /**
   * The ID of the context where the graph is being launched.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the graph is being launched.
   */
  uint32_t streamId;

  /**
   * This field is reserved for internal use
   */
  void *reserved;
} CUpti_ActivityGraphTrace;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (CUPTI_ACTIVITY_KIND_CONTEXT).
 * Context activity is now reported using CUpti_ActivityContext3 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONTEXT.
   */
  CUpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see CUpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;
} CUpti_ActivityContext;

/**
 * \brief The activity record for a context.
 *
 * This activity record represents information about a context
 * (CUPTI_ACTIVITY_KIND_CONTEXT).
 * Context activity is now reported using CUpti_ActivityContext3 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CONTEXT.
   */
  CUpti_ActivityKind kind;

  /**
   * The context ID.
   */
  uint32_t contextId;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The compute API kind. \see CUpti_ActivityComputeApiKind
   */
  uint16_t computeApiKind;

  /**
   * The ID for the NULL stream in this context
   */
  uint16_t nullStreamId;

  /**
   * The ID of the parent context. It would be 0 if
   * context does not have parent
   */
  uint32_t parentContextId;

  /**
   * This field indicates whether the context is a green context
   */
  uint8_t isGreenContext;

  uint8_t padding;

  /**
   * Number of multiprocessors assigned to the green context
   * Invalid if the field 'isGreenContext' is 0
   */
  uint16_t numMultiprocessors;
} CUpti_ActivityContext2;

/**
 * \brief The activity record for JIT operations.
 * This activity represents the JIT operations (compile, load, store) of a CUmodule
 * from the Compute Cache.
 * Gives the exact hashed path of where the cached module is loaded from,
 * or where the module will be stored after Just-In-Time (JIT) compilation.
 *
 * JIT activity is now reported using CUpti_ActivityJit2 record
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind must be CUPTI_ACTIVITY_KIND_JIT.
   */
  CUpti_ActivityKind kind;

  /**
    * The JIT entry type.
    */
  CUpti_ActivityJitEntryType jitEntryType;

  /**
   * The JIT operation type.
   */
  CUpti_ActivityJitOperationType jitOperationType;

  /**
   * The device ID.
   */
  uint32_t deviceId;

  /**
   * The start timestamp for the JIT operation, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the JIT operation.
   */
  uint64_t start;

  /**
   * The end timestamp for the JIT operation, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the JIT operation.
   */
  uint64_t end;

  /**
   * The correlation ID of the JIT operation to which
   * records belong to. Each JIT operation is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver or runtime API activity record that
   * launched the JIT operation.
   */
  uint32_t correlationId;

  /**
   * Internal use.
   */
  uint32_t padding;

  /**
   * The correlation ID to correlate JIT compilation, load and store operations.
   * Each JIT compilation unit is assigned a unique correlation ID
   * at the time of the JIT compilation. This correlation id can be used
   * to find the matching JIT cache load/store records.
   */
  uint64_t jitOperationCorrelationId;

  /**
   * The size of compute cache.
   */
  uint64_t cacheSize;

  /**
   * The path where the fat binary is cached.
   */
  const char* cachePath;
} CUpti_ActivityJit;

/**
 * \brief The activity record for CUDA event.
 *
 * This activity is used to track recorded events.
 * (CUPTI_ACTIVITY_KIND_CUDA_EVENT).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivityCudaEvent2
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_CUDA_EVENT.
   */
  CUpti_ActivityKind kind;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context where the event was recorded.
   */
  uint32_t contextId;

  /**
   * The compute stream where the event was recorded.
   */
  uint32_t streamId;

  /**
   * A unique event ID to identify the event record.
   */
  uint32_t eventId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
} CUpti_ActivityCudaEvent;

/**
 * \brief The activity record for synchronization management.
 *
 * This activity is used to track various CUDA synchronization APIs.
 * (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivitySynchronization2
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.
   */
  CUpti_ActivityKind kind;

  /**
   * The type of record.
   */
  CUpti_ActivitySynchronizationType type;

  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The correlation ID of the API to which this result is associated.
   */
  uint32_t correlationId;

  /**
   * The ID of the context for which the synchronization API is called.
   * In case of context synchronization API it is the context id for which the API is called.
   * In case of stream/event synchronization it is the ID of the context where the stream/event was created.
   */
  uint32_t contextId;

  /**
   * The compute stream for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuEventSynchronize.
   */
  uint32_t streamId;

  /**
   * The event ID for which the synchronization API is called.
   * A CUPTI_SYNCHRONIZATION_INVALID_VALUE value indicate the field is not applicable for this record.
   * Not valid for cuCtxSynchronize, cuStreamSynchronize.
   */
  uint32_t cudaEventId;
} CUpti_ActivitySynchronization;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy
 * (CUPTI_ACTIVITY_KIND_MEMCPY).
 *
 * Structure deprecated in CUDA 12.8: Refer to CUpti_ActivityMemcpy6
 * for the latest structure.
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be CUPTI_ACTIVITY_KIND_MEMCPY.
   */
  CUpti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see CUpti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see CUpti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see CUpti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef CUPTILP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;

  /**
   * The unique ID of the graph node that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint64_t graphNodeId;

  /**
   * The unique ID of the graph that executed this memcpy through graph launch.
   * This field will be 0 if the memcpy is not done through graph launch.
   */
  uint32_t graphId;

  /**
   * The ID of the HW channel on which the memory copy is occurring.
   */
  uint32_t channelID;

  /**
   * The type of the channel
   */
  CUpti_ChannelType channelType;

  /**
   *  Reserved for internal use.
   */
  uint32_t pad2;
} CUpti_ActivityMemcpy5;

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_ACTIVITY_DEPRECATED_H_*/


#endif /*_CUPTI_ACTIVITY_H_*/



// *************************************************************************
//      Definitions of indices for API functions, unique across entire API
// *************************************************************************

// This file is generated.  Any changes you make will be lost during the next clean build.
// CUDA public interface, for type definitions and cu* function prototypes

#if !defined(_CUPTI_DRIVER_CBID_H_)
#define _CUPTI_DRIVER_CBID_H_

typedef enum CUpti_driver_api_trace_cbid_enum {
    CUPTI_DRIVER_TRACE_CBID_INVALID                                                        = 0,
    CUPTI_DRIVER_TRACE_CBID_cuInit                                                         = 1,
    CUPTI_DRIVER_TRACE_CBID_cuDriverGetVersion                                             = 2,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGet                                                    = 3,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCount                                               = 4,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetName                                                = 5,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceComputeCapability                                      = 6,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem                                               = 7,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetProperties                                          = 8,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetAttribute                                           = 9,
    CUPTI_DRIVER_TRACE_CBID_cuCtxCreate                                                    = 10,
    CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy                                                   = 11,
    CUPTI_DRIVER_TRACE_CBID_cuCtxAttach                                                    = 12,
    CUPTI_DRIVER_TRACE_CBID_cuCtxDetach                                                    = 13,
    CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent                                               = 14,
    CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent                                                = 15,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice                                                 = 16,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize                                               = 17,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoad                                                   = 18,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData                                               = 19,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx                                             = 20,
    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary                                          = 21,
    CUPTI_DRIVER_TRACE_CBID_cuModuleUnload                                                 = 22,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction                                            = 23,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal                                              = 24,
    CUPTI_DRIVER_TRACE_CBID_cu64ModuleGetGlobal                                            = 25,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetTexRef                                              = 26,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo                                                   = 27,
    CUPTI_DRIVER_TRACE_CBID_cu64MemGetInfo                                                 = 28,
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc                                                     = 29,
    CUPTI_DRIVER_TRACE_CBID_cu64MemAlloc                                                   = 30,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch                                                = 31,
    CUPTI_DRIVER_TRACE_CBID_cu64MemAllocPitch                                              = 32,
    CUPTI_DRIVER_TRACE_CBID_cuMemFree                                                      = 33,
    CUPTI_DRIVER_TRACE_CBID_cu64MemFree                                                    = 34,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange                                           = 35,
    CUPTI_DRIVER_TRACE_CBID_cu64MemGetAddressRange                                         = 36,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost                                                 = 37,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost                                                  = 38,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc                                                 = 39,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer                                      = 40,
    CUPTI_DRIVER_TRACE_CBID_cu64MemHostGetDevicePointer                                    = 41,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostGetFlags                                              = 42,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD                                                   = 43,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoD                                                 = 44,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH                                                   = 45,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoH                                                 = 46,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD                                                   = 47,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoD                                                 = 48,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA                                                   = 49,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoA                                                 = 50,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD                                                   = 51,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyAtoD                                                 = 52,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA                                                   = 53,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH                                                   = 54,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA                                                   = 55,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D                                                     = 56,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned                                            = 57,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D                                                     = 58,
    CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3D                                                   = 59,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync                                              = 60,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoDAsync                                            = 61,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync                                              = 62,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoHAsync                                            = 63,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync                                              = 64,
    CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoDAsync                                            = 65,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync                                              = 66,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync                                              = 67,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync                                                = 68,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync                                                = 69,
    CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3DAsync                                              = 70,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8                                                     = 71,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8                                                   = 72,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16                                                    = 73,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16                                                  = 74,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32                                                    = 75,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32                                                  = 76,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8                                                   = 77,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8                                                 = 78,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16                                                  = 79,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16                                                = 80,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32                                                  = 81,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32                                                = 82,
    CUPTI_DRIVER_TRACE_CBID_cuFuncSetBlockShape                                            = 83,
    CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedSize                                            = 84,
    CUPTI_DRIVER_TRACE_CBID_cuFuncGetAttribute                                             = 85,
    CUPTI_DRIVER_TRACE_CBID_cuFuncSetCacheConfig                                           = 86,
    CUPTI_DRIVER_TRACE_CBID_cuArrayCreate                                                  = 87,
    CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor                                           = 88,
    CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy                                                 = 89,
    CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate                                                = 90,
    CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor                                         = 91,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefCreate                                                 = 92,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefDestroy                                                = 93,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetArray                                               = 94,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress                                             = 95,
    CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress                                           = 96,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D                                           = 97,
    CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress2D                                         = 98,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFormat                                              = 99,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddressMode                                         = 100,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFilterMode                                          = 101,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFlags                                               = 102,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress                                             = 103,
    CUPTI_DRIVER_TRACE_CBID_cu64TexRefGetAddress                                           = 104,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetArray                                               = 105,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddressMode                                         = 106,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFilterMode                                          = 107,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFormat                                              = 108,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFlags                                               = 109,
    CUPTI_DRIVER_TRACE_CBID_cuParamSetSize                                                 = 110,
    CUPTI_DRIVER_TRACE_CBID_cuParamSeti                                                    = 111,
    CUPTI_DRIVER_TRACE_CBID_cuParamSetf                                                    = 112,
    CUPTI_DRIVER_TRACE_CBID_cuParamSetv                                                    = 113,
    CUPTI_DRIVER_TRACE_CBID_cuParamSetTexRef                                               = 114,
    CUPTI_DRIVER_TRACE_CBID_cuLaunch                                                       = 115,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid                                                   = 116,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync                                              = 117,
    CUPTI_DRIVER_TRACE_CBID_cuEventCreate                                                  = 118,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecord                                                  = 119,
    CUPTI_DRIVER_TRACE_CBID_cuEventQuery                                                   = 120,
    CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize                                             = 121,
    CUPTI_DRIVER_TRACE_CBID_cuEventDestroy                                                 = 122,
    CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime                                             = 123,
    CUPTI_DRIVER_TRACE_CBID_cuStreamCreate                                                 = 124,
    CUPTI_DRIVER_TRACE_CBID_cuStreamQuery                                                  = 125,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize                                            = 126,
    CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy                                                = 127,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnregisterResource                                   = 128,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsSubResourceGetMappedArray                            = 129,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer                             = 130,
    CUPTI_DRIVER_TRACE_CBID_cu64GraphicsResourceGetMappedPointer                           = 131,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags                                  = 132,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources                                         = 133,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources                                       = 134,
    CUPTI_DRIVER_TRACE_CBID_cuGetExportTable                                               = 135,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetLimit                                                  = 136,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetLimit                                                  = 137,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevice                                               = 138,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate                                               = 139,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D10RegisterResource                                = 140,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10RegisterResource                                        = 141,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10UnregisterResource                                      = 142,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10MapResources                                            = 143,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10UnmapResources                                          = 144,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceSetMapFlags                                     = 145,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedArray                                  = 146,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer                                = 147,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize                                   = 148,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch                                  = 149,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions                            = 150,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevice                                               = 151,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate                                               = 152,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D11RegisterResource                                = 153,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevice                                                = 154,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate                                                = 155,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D9RegisterResource                                 = 156,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDirect3DDevice                                        = 157,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterResource                                         = 158,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterResource                                       = 159,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9MapResources                                             = 160,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapResources                                           = 161,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceSetMapFlags                                      = 162,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions                             = 163,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedArray                                   = 164,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer                                 = 165,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize                                    = 166,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch                                   = 167,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9Begin                                                    = 168,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9End                                                      = 169,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterVertexBuffer                                     = 170,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer                                          = 171,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapVertexBuffer                                        = 172,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterVertexBuffer                                   = 173,
    CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate                                                  = 174,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterBuffer                                     = 175,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterImage                                      = 176,
    CUPTI_DRIVER_TRACE_CBID_cuWGLGetDevice                                                 = 177,
    CUPTI_DRIVER_TRACE_CBID_cuGLInit                                                       = 178,
    CUPTI_DRIVER_TRACE_CBID_cuGLRegisterBufferObject                                       = 179,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject                                            = 180,
    CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObject                                          = 181,
    CUPTI_DRIVER_TRACE_CBID_cuGLUnregisterBufferObject                                     = 182,
    CUPTI_DRIVER_TRACE_CBID_cuGLSetBufferObjectMapFlags                                    = 183,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync                                       = 184,
    CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObjectAsync                                     = 185,
    CUPTI_DRIVER_TRACE_CBID_cuVDPAUGetDevice                                               = 186,
    CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate                                               = 187,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterVideoSurface                            = 188,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterOutputSurface                           = 189,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetSurfRef                                             = 190,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefCreate                                                = 191,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefDestroy                                               = 192,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetFormat                                             = 193,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetArray                                              = 194,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetFormat                                             = 195,
    CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetArray                                              = 196,
    CUPTI_DRIVER_TRACE_CBID_cu64DeviceTotalMem                                             = 197,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPointer                              = 198,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedSize                                 = 199,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPitch                                = 200,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetSurfaceDimensions                          = 201,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetSurfaceDimensions                           = 202,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPointer                               = 203,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedSize                                  = 204,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPitch                                 = 205,
    CUPTI_DRIVER_TRACE_CBID_cu64D3D9MapVertexBuffer                                        = 206,
    CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject                                          = 207,
    CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObjectAsync                                     = 208,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevices                                              = 209,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreateOnDevice                                       = 210,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevices                                              = 211,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreateOnDevice                                       = 212,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevices                                               = 213,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreateOnDevice                                        = 214,
    CUPTI_DRIVER_TRACE_CBID_cu64MemHostAlloc                                               = 215,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async                                                = 216,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8Async                                              = 217,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async                                               = 218,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16Async                                             = 219,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async                                               = 220,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32Async                                             = 221,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async                                              = 222,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8Async                                            = 223,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async                                             = 224,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16Async                                           = 225,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async                                             = 226,
    CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32Async                                           = 227,
    CUPTI_DRIVER_TRACE_CBID_cu64ArrayCreate                                                = 228,
    CUPTI_DRIVER_TRACE_CBID_cu64ArrayGetDescriptor                                         = 229,
    CUPTI_DRIVER_TRACE_CBID_cu64Array3DCreate                                              = 230,
    CUPTI_DRIVER_TRACE_CBID_cu64Array3DGetDescriptor                                       = 231,
    CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2D                                                   = 232,
    CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DUnaligned                                          = 233,
    CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DAsync                                              = 234,
    CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2                                                 = 235,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate_v2                                            = 236,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate_v2                                            = 237,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate_v2                                             = 238,
    CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate_v2                                               = 239,
    CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate_v2                                            = 240,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2                                           = 241,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo_v2                                                = 242,
    CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2                                                  = 243,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2                                             = 244,
    CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2                                                   = 245,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange_v2                                        = 246,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer_v2                                   = 247,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2                                                    = 248,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2                                                  = 249,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2                                                 = 250,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2                                                 = 251,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2                                                = 252,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2                                               = 253,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2                                               = 254,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress_v2                                          = 255,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v2                                        = 256,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress_v2                                          = 257,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer_v2                          = 258,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem_v2                                            = 259,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer_v2                             = 260,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize_v2                                = 261,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch_v2                               = 262,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions_v2                         = 263,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions_v2                          = 264,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer_v2                              = 265,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize_v2                                 = 266,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch_v2                                = 267,
    CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer_v2                                       = 268,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2                                         = 269,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2                                    = 270,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2                                              = 271,
    CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2                                               = 272,
    CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor_v2                                        = 273,
    CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2                                             = 274,
    CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor_v2                                      = 275,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2                                                = 276,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2                                           = 277,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2                                                = 278,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2                                           = 279,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2                                                = 280,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2                                           = 281,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2                                                = 282,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2                                           = 283,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2                                                = 284,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2                                                = 285,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2                                                = 286,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2                                                  = 287,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2                                         = 288,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2                                             = 289,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2                                                  = 290,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2                                             = 291,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2                                                = 292,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2                                           = 293,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2                                              = 294,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent                                              = 295,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetApiVersion                                             = 296,
    CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDirect3DDevice                                       = 297,
    CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDirect3DDevice                                       = 298,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetCacheConfig                                            = 299,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetCacheConfig                                            = 300,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister                                              = 301,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister                                            = 302,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent                                                = 303,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent                                                = 304,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy                                                       = 305,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync                                                  = 306,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel                                                 = 307,
    CUPTI_DRIVER_TRACE_CBID_cuProfilerStart                                                = 308,
    CUPTI_DRIVER_TRACE_CBID_cuProfilerStop                                                 = 309,
    CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttribute                                          = 310,
    CUPTI_DRIVER_TRACE_CBID_cuProfilerInitialize                                           = 311,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceCanAccessPeer                                          = 312,
    CUPTI_DRIVER_TRACE_CBID_cuCtxEnablePeerAccess                                          = 313,
    CUPTI_DRIVER_TRACE_CBID_cuCtxDisablePeerAccess                                         = 314,
    CUPTI_DRIVER_TRACE_CBID_cuMemPeerRegister                                              = 315,
    CUPTI_DRIVER_TRACE_CBID_cuMemPeerUnregister                                            = 316,
    CUPTI_DRIVER_TRACE_CBID_cuMemPeerGetDevicePointer                                      = 317,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer                                                   = 318,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync                                              = 319,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer                                                 = 320,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync                                            = 321,
    CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2                                                = 322,
    CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent_v2                                            = 323,
    CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent_v2                                             = 324,
    CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2                                              = 325,
    CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy_v2                                             = 326,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v3                                        = 327,
    CUPTI_DRIVER_TRACE_CBID_cuIpcGetMemHandle                                              = 328,
    CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle                                             = 329,
    CUPTI_DRIVER_TRACE_CBID_cuIpcCloseMemHandle                                            = 330,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetByPCIBusId                                          = 331,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetPCIBusId                                            = 332,
    CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices                                                 = 333,
    CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle                                            = 334,
    CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle                                           = 335,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetSharedMemConfig                                        = 336,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetSharedMemConfig                                        = 337,
    CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedMemConfig                                       = 338,
    CUPTI_DRIVER_TRACE_CBID_cuTexObjectCreate                                              = 339,
    CUPTI_DRIVER_TRACE_CBID_cuTexObjectDestroy                                             = 340,
    CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceDesc                                     = 341,
    CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetTextureDesc                                      = 342,
    CUPTI_DRIVER_TRACE_CBID_cuSurfObjectCreate                                             = 343,
    CUPTI_DRIVER_TRACE_CBID_cuSurfObjectDestroy                                            = 344,
    CUPTI_DRIVER_TRACE_CBID_cuSurfObjectGetResourceDesc                                    = 345,
    CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback                                            = 346,
    CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayCreate                                         = 347,
    CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetLevel                                       = 348,
    CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayDestroy                                        = 349,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmappedArray                                      = 350,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapFilterMode                                    = 351,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelBias                                     = 352,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelClamp                                    = 353,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMaxAnisotropy                                       = 354,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmappedArray                                      = 355,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapFilterMode                                    = 356,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelBias                                     = 357,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelClamp                                    = 358,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMaxAnisotropy                                       = 359,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedMipmappedArray                      = 360,
    CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceViewDesc                                 = 361,
    CUPTI_DRIVER_TRACE_CBID_cuLinkCreate                                                   = 362,
    CUPTI_DRIVER_TRACE_CBID_cuLinkAddData                                                  = 363,
    CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile                                                  = 364,
    CUPTI_DRIVER_TRACE_CBID_cuLinkComplete                                                 = 365,
    CUPTI_DRIVER_TRACE_CBID_cuLinkDestroy                                                  = 366,
    CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority                                     = 367,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority                                            = 368,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags                                               = 369,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetStreamPriorityRange                                    = 370,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged                                              = 371,
    CUPTI_DRIVER_TRACE_CBID_cuGetErrorString                                               = 372,
    CUPTI_DRIVER_TRACE_CBID_cuGetErrorName                                                 = 373,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessor                    = 374,
    CUPTI_DRIVER_TRACE_CBID_cuCompilePtx                                                   = 375,
    CUPTI_DRIVER_TRACE_CBID_cuBinaryFree                                                   = 376,
    CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync                                         = 377,
    CUPTI_DRIVER_TRACE_CBID_cuPointerSetAttribute                                          = 378,
    CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2                                           = 379,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags_v2                               = 380,
    CUPTI_DRIVER_TRACE_CBID_cuLinkCreate_v2                                                = 381,
    CUPTI_DRIVER_TRACE_CBID_cuLinkAddData_v2                                               = 382,
    CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile_v2                                               = 383,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSize                               = 384,
    CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices_v2                                              = 385,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRetain                                       = 386,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease                                      = 387,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags                                     = 388,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset                                        = 389,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsEGLRegisterImage                                     = 390,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetFlags                                                  = 391,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxGetState                                     = 392,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnect                                     = 393,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerDisconnect                                  = 394,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerAcquireFrame                                = 395,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerReleaseFrame                                = 396,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2_ptds                                           = 397,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2_ptds                                           = 398,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2_ptds                                           = 399,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2_ptds                                           = 400,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2_ptds                                           = 401,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2_ptds                                           = 402,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2_ptds                                           = 403,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2_ptds                                           = 404,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2_ptds                                             = 405,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2_ptds                                    = 406,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2_ptds                                             = 407,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy_ptds                                                  = 408,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer_ptds                                              = 409,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer_ptds                                            = 410,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2_ptds                                             = 411,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2_ptds                                            = 412,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2_ptds                                            = 413,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2_ptds                                           = 414,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2_ptds                                          = 415,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2_ptds                                          = 416,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2_ptds                                    = 417,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz                                             = 418,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2_ptsz                                      = 419,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2_ptsz                                      = 420,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz                                      = 421,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz                                      = 422,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz                                      = 423,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz                                        = 424,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2_ptsz                                        = 425,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync_ptsz                                         = 426,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync_ptsz                                       = 427,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz                                           = 428,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async_ptsz                                          = 429,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz                                          = 430,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async_ptsz                                         = 431,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async_ptsz                                        = 432,
    CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async_ptsz                                        = 433,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority_ptsz                                       = 434,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags_ptsz                                          = 435,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz                                         = 436,
    CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback_ptsz                                       = 437,
    CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync_ptsz                                    = 438,
    CUPTI_DRIVER_TRACE_CBID_cuStreamQuery_ptsz                                             = 439,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz                                       = 440,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz                                             = 441,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz                                            = 442,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources_ptsz                                    = 443,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources_ptsz                                  = 444,
    CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2_ptsz                               = 445,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerConnect                                     = 446,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerDisconnect                                  = 447,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerPresentFrame                                = 448,
    CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedEglFrame                            = 449,
    CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttributes                                         = 450,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags           = 451,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSizeWithFlags                      = 452,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerReturnFrame                                 = 453,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetP2PAttribute                                        = 454,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefSetBorderColor                                         = 455,
    CUPTI_DRIVER_TRACE_CBID_cuTexRefGetBorderColor                                         = 456,
    CUPTI_DRIVER_TRACE_CBID_cuMemAdvise                                                    = 457,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32                                            = 458,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_ptsz                                       = 459,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32                                           = 460,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_ptsz                                      = 461,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp                                             = 462,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_ptsz                                        = 463,
    CUPTI_DRIVER_TRACE_CBID_cuNVNbufferGetPointer                                          = 464,
    CUPTI_DRIVER_TRACE_CBID_cuNVNtextureGetArray                                           = 465,
    CUPTI_DRIVER_TRACE_CBID_cuNNSetAllocator                                               = 466,
    CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync                                             = 467,
    CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_ptsz                                        = 468,
    CUPTI_DRIVER_TRACE_CBID_cuEventCreateFromNVNSync                                       = 469,
    CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnectWithFlags                            = 470,
    CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttribute                                         = 471,
    CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttributes                                        = 472,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64                                            = 473,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_ptsz                                       = 474,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64                                           = 475,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_ptsz                                      = 476,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel                                      = 477,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz                                 = 478,
    CUPTI_DRIVER_TRACE_CBID_cuEventCreateFromEGLSync                                       = 479,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice                           = 480,
    CUPTI_DRIVER_TRACE_CBID_cuFuncSetAttribute                                             = 481,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid                                                = 482,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx                                                 = 483,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_ptsz                                            = 484,
    CUPTI_DRIVER_TRACE_CBID_cuImportExternalMemory                                         = 485,
    CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedBuffer                                = 486,
    CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedMipmappedArray                        = 487,
    CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalMemory                                        = 488,
    CUPTI_DRIVER_TRACE_CBID_cuImportExternalSemaphore                                      = 489,
    CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync                                = 490,
    CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync_ptsz                           = 491,
    CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync                                  = 492,
    CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync_ptsz                             = 493,
    CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalSemaphore                                     = 494,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture                                           = 495,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz                                      = 496,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture                                             = 497,
    CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz                                        = 498,
    CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing                                            = 499,
    CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing_ptsz                                       = 500,
    CUPTI_DRIVER_TRACE_CBID_cuGraphCreate                                                  = 501,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode                                           = 502,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams                                     = 503,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemcpyNode                                           = 504,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeGetParams                                     = 505,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemsetNode                                           = 506,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeGetParams                                     = 507,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeSetParams                                     = 508,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetType                                             = 509,
    CUPTI_DRIVER_TRACE_CBID_cuGraphGetRootNodes                                            = 510,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependencies                                     = 511,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependentNodes                                   = 512,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate                                             = 513,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch                                                  = 514,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz                                             = 515,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy                                             = 516,
    CUPTI_DRIVER_TRACE_CBID_cuGraphDestroy                                                 = 517,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddDependencies                                         = 518,
    CUPTI_DRIVER_TRACE_CBID_cuGraphRemoveDependencies                                      = 519,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeSetParams                                     = 520,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams                                     = 521,
    CUPTI_DRIVER_TRACE_CBID_cuGraphDestroyNode                                             = 522,
    CUPTI_DRIVER_TRACE_CBID_cuGraphClone                                                   = 523,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeFindInClone                                         = 524,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddChildGraphNode                                       = 525,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddEmptyNode                                            = 526,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc                                               = 527,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz                                          = 528,
    CUPTI_DRIVER_TRACE_CBID_cuGraphChildGraphNodeGetGraph                                  = 529,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddHostNode                                             = 530,
    CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeGetParams                                       = 531,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetLuid                                                = 532,
    CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeSetParams                                       = 533,
    CUPTI_DRIVER_TRACE_CBID_cuGraphGetNodes                                                = 534,
    CUPTI_DRIVER_TRACE_CBID_cuGraphGetEdges                                                = 535,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo                                         = 536,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_ptsz                                    = 537,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams                                 = 538,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2                                        = 539,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz                                   = 540,
    CUPTI_DRIVER_TRACE_CBID_cuThreadExchangeStreamCaptureMode                              = 541,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetNvSciSyncAttributes                                 = 542,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyAvailableDynamicSMemPerBlock                        = 543,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease_v2                                   = 544,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset_v2                                     = 545,
    CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags_v2                                  = 546,
    CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve                                            = 547,
    CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree                                               = 548,
    CUPTI_DRIVER_TRACE_CBID_cuMemCreate                                                    = 549,
    CUPTI_DRIVER_TRACE_CBID_cuMemRelease                                                   = 550,
    CUPTI_DRIVER_TRACE_CBID_cuMemMap                                                       = 551,
    CUPTI_DRIVER_TRACE_CBID_cuMemUnmap                                                     = 552,
    CUPTI_DRIVER_TRACE_CBID_cuMemSetAccess                                                 = 553,
    CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle                                   = 554,
    CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle                                 = 555,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationGranularity                                  = 556,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationPropertiesFromHandle                         = 557,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetAccess                                                 = 558,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags                                               = 559,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags_ptsz                                          = 560,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate                                              = 561,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemcpyNodeSetParams                                 = 562,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemsetNodeSetParams                                 = 563,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecHostNodeSetParams                                   = 564,
    CUPTI_DRIVER_TRACE_CBID_cuMemRetainAllocationHandle                                    = 565,
    CUPTI_DRIVER_TRACE_CBID_cuFuncGetModule                                                = 566,
    CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle_v2                                          = 567,
    CUPTI_DRIVER_TRACE_CBID_cuCtxResetPersistingL2Cache                                    = 568,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeCopyAttributes                                = 569,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetAttribute                                  = 570,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetAttribute                                  = 571,
    CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes                                         = 572,
    CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes_ptsz                                    = 573,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute                                           = 574,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute_ptsz                                      = 575,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute                                           = 576,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute_ptsz                                      = 577,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2                                          = 578,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetTexture1DLinearMaxWidth                             = 579,
    CUPTI_DRIVER_TRACE_CBID_cuGraphUpload                                                  = 580,
    CUPTI_DRIVER_TRACE_CBID_cuGraphUpload_ptsz                                             = 581,
    CUPTI_DRIVER_TRACE_CBID_cuArrayGetSparseProperties                                     = 582,
    CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetSparseProperties                            = 583,
    CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync                                             = 584,
    CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync_ptsz                                        = 585,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams                             = 586,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags                                         = 587,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz                                    = 588,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventRecordNode                                      = 589,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventWaitNode                                        = 590,
    CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeGetEvent                                 = 591,
    CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeGetEvent                                   = 592,
    CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeSetEvent                                 = 593,
    CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeSetEvent                                   = 594,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventRecordNodeSetEvent                             = 595,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventWaitNodeSetEvent                               = 596,
    CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane                                                = 597,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync                                                = 598,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz                                           = 599,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync                                                 = 600,
    CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz                                            = 601,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolTrimTo                                                = 602,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAttribute                                          = 603,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAttribute                                          = 604,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAccess                                             = 605,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetDefaultMemPool                                      = 606,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate                                                = 607,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolDestroy                                               = 608,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceSetMemPool                                             = 609,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetMemPool                                             = 610,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync                                        = 611,
    CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz                                   = 612,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportToShareableHandle                               = 613,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportFromShareableHandle                             = 614,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportPointer                                         = 615,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportPointer                                         = 616,
    CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAccess                                             = 617,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresSignalNode                         = 618,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeGetParams                   = 619,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeSetParams                   = 620,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresWaitNode                           = 621,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeGetParams                     = 622,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeSetParams                     = 623,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresSignalNodeSetParams               = 624,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresWaitNodeSetParams                 = 625,
    CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress                                               = 626,
    CUPTI_DRIVER_TRACE_CBID_cuFlushGPUDirectRDMAWrites                                     = 627,
    CUPTI_DRIVER_TRACE_CBID_cuGraphDebugDotPrint                                           = 628,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2                                      = 629,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2_ptsz                                 = 630,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies                              = 631,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz                         = 632,
    CUPTI_DRIVER_TRACE_CBID_cuUserObjectCreate                                             = 633,
    CUPTI_DRIVER_TRACE_CBID_cuUserObjectRetain                                             = 634,
    CUPTI_DRIVER_TRACE_CBID_cuUserObjectRelease                                            = 635,
    CUPTI_DRIVER_TRACE_CBID_cuGraphRetainUserObject                                        = 636,
    CUPTI_DRIVER_TRACE_CBID_cuGraphReleaseUserObject                                       = 637,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemAllocNode                                         = 638,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemFreeNode                                          = 639,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGraphMemTrim                                           = 640,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetGraphMemAttribute                                   = 641,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceSetGraphMemAttribute                                   = 642,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags                                    = 643,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetExecAffinitySupport                                 = 644,
    CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v3                                                 = 645,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetExecAffinity                                           = 646,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid_v2                                             = 647,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemAllocNodeGetParams                                   = 648,
    CUPTI_DRIVER_TRACE_CBID_cuGraphMemFreeNodeGetParams                                    = 649,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetEnabled                                          = 650,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetEnabled                                          = 651,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx                                               = 652,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz                                          = 653,
    CUPTI_DRIVER_TRACE_CBID_cuArrayGetMemoryRequirements                                   = 654,
    CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetMemoryRequirements                          = 655,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams                                   = 656,
    CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz                              = 657,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecGetFlags                                            = 658,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2                                         = 659,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2_ptsz                                    = 660,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2                                         = 661,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2_ptsz                                    = 662,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2                                        = 663,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2_ptsz                                   = 664,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2                                        = 665,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2_ptsz                                   = 666,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2                                          = 667,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2_ptsz                                     = 668,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddBatchMemOpNode                                       = 669,
    CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeGetParams                                 = 670,
    CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeSetParams                                 = 671,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecBatchMemOpNodeSetParams                             = 672,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetLoadingMode                                         = 673,
    CUPTI_DRIVER_TRACE_CBID_cuMemGetHandleForAddressRange                                  = 674,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialClusterSize                             = 675,
    CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveClusters                                   = 676,
    CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress_v2                                            = 677,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData                                              = 678,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile                                          = 679,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryUnload                                                = 680,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel                                             = 681,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetModule                                             = 682,
    CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction                                            = 683,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetGlobal                                             = 684,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetManaged                                            = 685,
    CUPTI_DRIVER_TRACE_CBID_cuKernelGetAttribute                                           = 686,
    CUPTI_DRIVER_TRACE_CBID_cuKernelSetAttribute                                           = 687,
    CUPTI_DRIVER_TRACE_CBID_cuKernelSetCacheConfig                                         = 688,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2                                        = 689,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams_v2                                  = 690,
    CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2                                  = 691,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2                              = 692,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetId                                                  = 693,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetId_ptsz                                             = 694,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetId                                                     = 695,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2                                           = 696,
    CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled                                         = 697,
    CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeIm2col                                        = 698,
    CUPTI_DRIVER_TRACE_CBID_cuTensorMapReplaceAddress                                      = 699,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetUnifiedFunction                                    = 700,
    CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttribute                                         = 701,
    CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttributeGlobal                                   = 702,
    CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttribute                                         = 703,
    CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttributeGlobal                                   = 704,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetFlags                                                  = 705,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastCreate                                              = 706,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastAddDevice                                           = 707,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem                                             = 708,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr                                            = 709,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastUnbind                                              = 710,
    CUPTI_DRIVER_TRACE_CBID_cuMulticastGetGranularity                                      = 711,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode                                                 = 712,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams                                           = 713,
    CUPTI_DRIVER_TRACE_CBID_cuGraphExecNodeSetParams                                       = 714,
    CUPTI_DRIVER_TRACE_CBID_cuMemAdvise_v2                                                 = 715,
    CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2                                          = 716,
    CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2_ptsz                                     = 717,
    CUPTI_DRIVER_TRACE_CBID_cuFuncGetName                                                  = 718,
    CUPTI_DRIVER_TRACE_CBID_cuKernelGetName                                                = 719,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph                                    = 720,
    CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz                               = 721,
    CUPTI_DRIVER_TRACE_CBID_cuGraphConditionalHandleCreate                                 = 722,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode_v2                                              = 723,
    CUPTI_DRIVER_TRACE_CBID_cuGraphGetEdges_v2                                             = 724,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependencies_v2                                  = 725,
    CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependentNodes_v2                                = 726,
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddDependencies_v2                                      = 727,
    CUPTI_DRIVER_TRACE_CBID_cuGraphRemoveDependencies_v2                                   = 728,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v3                                      = 729,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v3_ptsz                                 = 730,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2                           = 731,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2_ptsz                      = 732,
    CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo                                             = 733,
    CUPTI_DRIVER_TRACE_CBID_cuKernelGetParamInfo                                           = 734,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceRegisterAsyncNotification                              = 735,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceUnregisterAsyncNotification                            = 736,
    CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunctionCount                                       = 737,
    CUPTI_DRIVER_TRACE_CBID_cuModuleEnumerateFunctions                                     = 738,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernelCount                                        = 739,
    CUPTI_DRIVER_TRACE_CBID_cuLibraryEnumerateKernels                                      = 740,
    CUPTI_DRIVER_TRACE_CBID_cuFuncIsLoaded                                                 = 741,
    CUPTI_DRIVER_TRACE_CBID_cuFuncLoad                                                     = 742,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxCreate                                               = 743,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxDestroy                                              = 744,
    CUPTI_DRIVER_TRACE_CBID_cuDeviceGetDevResource                                         = 745,
    CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevResource                                            = 746,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxGetDevResource                                       = 747,
    CUPTI_DRIVER_TRACE_CBID_cuDevResourceGenerateDesc                                      = 748,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxRecordEvent                                          = 749,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxWaitEvent                                            = 750,
    CUPTI_DRIVER_TRACE_CBID_cuDevSmResourceSplitByCount                                    = 751,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetGreenCtx                                            = 752,
    CUPTI_DRIVER_TRACE_CBID_cuCtxFromGreenCtx                                              = 753,
    CUPTI_DRIVER_TRACE_CBID_cuKernelGetLibrary                                             = 754,
    CUPTI_DRIVER_TRACE_CBID_cuCtxRecordEvent                                               = 755,
    CUPTI_DRIVER_TRACE_CBID_cuCtxWaitEvent                                                 = 756,
    CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v4                                                 = 757,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxStreamCreate                                         = 758,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_v2                                              = 759,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_v2_ptsz                                         = 760,
    CUPTI_DRIVER_TRACE_CBID_cuMemBatchDecompressAsync                                      = 761,
    CUPTI_DRIVER_TRACE_CBID_cuMemBatchDecompressAsync_ptsz                                 = 762,
    CUPTI_DRIVER_TRACE_CBID_cuLogsRegisterCallback                                         = 763,
    CUPTI_DRIVER_TRACE_CBID_cuLogsUnregisterCallback                                       = 764,
    CUPTI_DRIVER_TRACE_CBID_cuLogsCurrent                                                  = 765,
    CUPTI_DRIVER_TRACE_CBID_cuLogsDumpToFile                                               = 766,
    CUPTI_DRIVER_TRACE_CBID_cuLogsDumpToMemory                                             = 767,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessGetRestoreThreadId                          = 768,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessGetState                                    = 769,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessLock                                        = 770,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessCheckpoint                                  = 771,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessRestore                                     = 772,
    CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessUnlock                                      = 773,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevice                                              = 774,
    CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevice_ptsz                                         = 775,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync                                             = 776,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync_ptsz                                        = 777,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync                                           = 778,
    CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync_ptsz                                      = 779,
    CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime_v2                                          = 780,
    CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeIm2colWide                                    = 781,
    CUPTI_DRIVER_TRACE_CBID_cuGreenCtxGetId                                                = 782,
    CUPTI_DRIVER_TRACE_CBID_cuStreamCreateForCaptureToCig                                  = 783,
    CUPTI_DRIVER_TRACE_CBID_SIZE                                                           = 784,
    CUPTI_DRIVER_TRACE_CBID_FORCE_INT                                                      = 0x7fffffff
} CUpti_driver_api_trace_cbid;

#endif



// *************************************************************************
//      Definitions of indices for API functions, unique across entire API
// *************************************************************************

// This file is generated.  Any changes you make will be lost during the next clean build.
// CUDA public interface, for type definitions and cu* function prototypes

#if !defined(_CUPTI_RUNTIME_CBID_H)
#define _CUPTI_RUNTIME_CBID_H

typedef enum CUpti_runtime_api_trace_cbid_enum {
    CUPTI_RUNTIME_TRACE_CBID_INVALID                                                       = 0,
    CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020                                    = 1,
    CUPTI_RUNTIME_TRACE_CBID_cudaRuntimeGetVersion_v3020                                   = 2,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020                                      = 3,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020                                 = 4,
    CUPTI_RUNTIME_TRACE_CBID_cudaChooseDevice_v3020                                        = 5,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetChannelDesc_v3020                                      = 6,
    CUPTI_RUNTIME_TRACE_CBID_cudaCreateChannelDesc_v3020                                   = 7,
    CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020                                       = 8,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020                                       = 9,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020                                        = 10,
    CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020                                     = 11,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorString_v3020                                      = 12,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020                                              = 13,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetCacheConfig_v3020                                  = 14,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetAttributes_v3020                                   = 15,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020                                           = 16,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020                                           = 17,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetValidDevices_v3020                                     = 18,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetDeviceFlags_v3020                                      = 19,
    CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020                                              = 20,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020                                         = 21,
    CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020                                                = 22,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020                                         = 23,
    CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020                                           = 24,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020                                          = 25,
    CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020                                            = 26,
    CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020                                           = 27,
    CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020                                = 28,
    CUPTI_RUNTIME_TRACE_CBID_cudaHostGetFlags_v3020                                        = 29,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemGetInfo_v3020                                          = 30,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020                                              = 31,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020                                            = 32,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020                                       = 33,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_v3020                                     = 34,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020                                     = 35,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_v3020                                   = 36,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020                                  = 37,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_v3020                                = 38,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020                                      = 39,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020                                    = 40,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020                                         = 41,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020                                  = 42,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020                                = 43,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020                                       = 44,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_v3020                                = 45,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_v3020                              = 46,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020                                 = 47,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020                               = 48,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020                                              = 49,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020                                            = 50,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020                                         = 51,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020                                       = 52,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolAddress_v3020                                    = 53,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolSize_v3020                                       = 54,
    CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture_v3020                                         = 55,
    CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture2D_v3020                                       = 56,
    CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToArray_v3020                                  = 57,
    CUPTI_RUNTIME_TRACE_CBID_cudaUnbindTexture_v3020                                       = 58,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureAlignmentOffset_v3020                           = 59,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureReference_v3020                                 = 60,
    CUPTI_RUNTIME_TRACE_CBID_cudaBindSurfaceToArray_v3020                                  = 61,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceReference_v3020                                 = 62,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLSetGLDevice_v3020                                       = 63,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLRegisterBufferObject_v3020                              = 64,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObject_v3020                                   = 65,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObject_v3020                                 = 66,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLUnregisterBufferObject_v3020                            = 67,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLSetBufferObjectMapFlags_v3020                           = 68,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObjectAsync_v3020                              = 69,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObjectAsync_v3020                            = 70,
    CUPTI_RUNTIME_TRACE_CBID_cudaWGLGetDevice_v3020                                        = 71,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterImage_v3020                             = 72,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterBuffer_v3020                            = 73,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnregisterResource_v3020                          = 74,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceSetMapFlags_v3020                         = 75,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsMapResources_v3020                                = 76,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnmapResources_v3020                              = 77,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedPointer_v3020                    = 78,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsSubResourceGetMappedArray_v3020                   = 79,
    CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUGetDevice_v3020                                      = 80,
    CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUSetVDPAUDevice_v3020                                 = 81,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterVideoSurface_v3020                   = 82,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterOutputSurface_v3020                  = 83,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevice_v3020                                      = 84,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevices_v3020                                     = 85,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D11SetDirect3DDevice_v3020                              = 86,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D11RegisterResource_v3020                       = 87,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevice_v3020                                      = 88,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevices_v3020                                     = 89,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10SetDirect3DDevice_v3020                              = 90,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D10RegisterResource_v3020                       = 91,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10RegisterResource_v3020                               = 92,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnregisterResource_v3020                             = 93,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10MapResources_v3020                                   = 94,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnmapResources_v3020                                 = 95,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceSetMapFlags_v3020                            = 96,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetSurfaceDimensions_v3020                   = 97,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedArray_v3020                         = 98,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPointer_v3020                       = 99,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedSize_v3020                          = 100,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPitch_v3020                         = 101,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevice_v3020                                       = 102,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevices_v3020                                      = 103,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9SetDirect3DDevice_v3020                               = 104,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDirect3DDevice_v3020                               = 105,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D9RegisterResource_v3020                        = 106,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterResource_v3020                                = 107,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterResource_v3020                              = 108,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapResources_v3020                                    = 109,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapResources_v3020                                  = 110,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceSetMapFlags_v3020                             = 111,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetSurfaceDimensions_v3020                    = 112,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedArray_v3020                          = 113,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPointer_v3020                        = 114,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedSize_v3020                           = 115,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPitch_v3020                          = 116,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9Begin_v3020                                           = 117,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9End_v3020                                             = 118,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterVertexBuffer_v3020                            = 119,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterVertexBuffer_v3020                          = 120,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapVertexBuffer_v3020                                 = 121,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapVertexBuffer_v3020                               = 122,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020                                          = 123,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForDevice_v3020                                  = 124,
    CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForHost_v3020                                    = 125,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020                                   = 126,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetLimit_v3020                                      = 127,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetLimit_v3020                                      = 128,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020                                        = 129,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020                                       = 130,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020                                   = 131,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020                                         = 132,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020                                         = 133,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020                                = 134,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020                                         = 135,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020                                        = 136,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020                                    = 137,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020                                          = 138,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020                                    = 139,
    CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020                                            = 140,
    CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020                                       = 141,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020                                            = 142,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020                                       = 143,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020                                            = 144,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020                                       = 145,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetCacheConfig_v3020                                = 146,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020                                     = 147,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDirect3DDevice_v3020                              = 148,
    CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDirect3DDevice_v3020                              = 149,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetCacheConfig_v3020                                = 150,
    CUPTI_RUNTIME_TRACE_CBID_cudaPointerGetAttributes_v4000                                = 151,
    CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000                                        = 152,
    CUPTI_RUNTIME_TRACE_CBID_cudaHostUnregister_v4000                                      = 153,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceCanAccessPeer_v4000                                 = 154,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceEnablePeerAccess_v4000                              = 155,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceDisablePeerAccess_v4000                             = 156,
    CUPTI_RUNTIME_TRACE_CBID_cudaPeerRegister_v4000                                        = 157,
    CUPTI_RUNTIME_TRACE_CBID_cudaPeerUnregister_v4000                                      = 158,
    CUPTI_RUNTIME_TRACE_CBID_cudaPeerGetDevicePointer_v4000                                = 159,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000                                          = 160,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000                                     = 161,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_v4000                                        = 162,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_v4000                                   = 163,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020                                         = 164,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020                                   = 165,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetLimit_v3020                                      = 166,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetLimit_v3020                                      = 167,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetCacheConfig_v3020                                = 168,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetCacheConfig_v3020                                = 169,
    CUPTI_RUNTIME_TRACE_CBID_cudaProfilerInitialize_v4000                                  = 170,
    CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000                                       = 171,
    CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000                                        = 172,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetByPCIBusId_v4010                                 = 173,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetPCIBusId_v4010                                   = 174,
    CUPTI_RUNTIME_TRACE_CBID_cudaGLGetDevices_v4010                                        = 175,
    CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010                                   = 176,
    CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010                                  = 177,
    CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010                                     = 178,
    CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010                                    = 179,
    CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010                                   = 180,
    CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetInfo_v4010                                        = 181,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetSharedMemConfig_v4020                              = 182,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetSharedMemConfig_v4020                            = 183,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetSharedMemConfig_v4020                            = 184,
    CUPTI_RUNTIME_TRACE_CBID_cudaCreateTextureObject_v5000                                 = 185,
    CUPTI_RUNTIME_TRACE_CBID_cudaDestroyTextureObject_v5000                                = 186,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceDesc_v5000                        = 187,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectTextureDesc_v5000                         = 188,
    CUPTI_RUNTIME_TRACE_CBID_cudaCreateSurfaceObject_v5000                                 = 189,
    CUPTI_RUNTIME_TRACE_CBID_cudaDestroySurfaceObject_v5000                                = 190,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceObjectResourceDesc_v5000                        = 191,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000                                = 192,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetMipmappedArrayLevel_v5000                              = 193,
    CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000                                  = 194,
    CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToMipmappedArray_v5000                         = 195,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedMipmappedArray_v5000             = 196,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_v5000                                   = 197,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000                               = 198,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceViewDesc_v5000                    = 199,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000                                  = 200,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050                                       = 201,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050                            = 202,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_v5050                                   = 203,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_v5050                                      = 204,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetStreamPriorityRange_v5050                        = 205,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000                                       = 206,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000           = 207,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_v6000                                = 208,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorName_v6050                                        = 209,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050           = 210,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000                                        = 211,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceFlags_v7000                                      = 212,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000                                         = 213,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000                                   = 214,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000                                         = 215,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000                                       = 216,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_ptds_v7000                                  = 217,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_ptds_v7000                                = 218,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_ptds_v7000                                = 219,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_ptds_v7000                              = 220,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_ptds_v7000                             = 221,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_ptds_v7000                           = 222,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_ptds_v7000                                 = 223,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_ptds_v7000                               = 224,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000                                    = 225,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_ptsz_v7000                             = 226,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_ptsz_v7000                           = 227,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000                                  = 228,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_ptsz_v7000                           = 229,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_ptsz_v7000                         = 230,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_ptsz_v7000                            = 231,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_ptsz_v7000                          = 232,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset_ptds_v7000                                         = 233,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_ptds_v7000                                       = 234,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000                                    = 235,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_ptsz_v7000                                  = 236,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_ptsz_v7000                              = 237,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_ptsz_v7000                                 = 238,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000                              = 239,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000                                    = 240,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_ptsz_v7000                           = 241,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_ptsz_v7000                                    = 242,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_ptds_v7000                                       = 243,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_ptsz_v7000                                  = 244,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000                                       = 245,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000                                  = 246,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_ptsz_v7000                                = 247,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_ptsz_v7000                              = 248,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_ptds_v7000                                   = 249,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_ptsz_v7000                              = 250,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000  = 251,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v8000                                    = 252,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_ptsz_v8000                               = 253,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v8000                                           = 254,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetP2PAttribute_v8000                               = 255,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsEGLRegisterImage_v7000                            = 256,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnect_v7000                            = 257,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerDisconnect_v7000                         = 258,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerAcquireFrame_v7000                       = 259,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerReleaseFrame_v7000                       = 260,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerConnect_v7000                            = 261,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerDisconnect_v7000                         = 262,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerPresentFrame_v7000                       = 263,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerReturnFrame_v7000                        = 264,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedEglFrame_v7000                   = 265,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttribute_v8000                                = 266,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttributes_v8000                               = 267,
    CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnectWithFlags_v7000                   = 268,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000                             = 269,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000                        = 270,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateFromEGLSync_v9000                              = 271,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000                  = 272,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetAttribute_v9000                                    = 273,
    CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalMemory_v10000                               = 274,
    CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedBuffer_v10000                      = 275,
    CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedMipmappedArray_v10000              = 276,
    CUPTI_RUNTIME_TRACE_CBID_cudaDestroyExternalMemory_v10000                              = 277,
    CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalSemaphore_v10000                            = 278,
    CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_v10000                      = 279,
    CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_ptsz_v10000                 = 280,
    CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v10000                        = 281,
    CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_ptsz_v10000                   = 282,
    CUPTI_RUNTIME_TRACE_CBID_cudaDestroyExternalSemaphore_v10000                           = 283,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000                                     = 284,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000                                = 285,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphCreate_v10000                                        = 286,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeGetParams_v10000                           = 287,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000                           = 288,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddKernelNode_v10000                                 = 289,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNode_v10000                                 = 290,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeGetParams_v10000                           = 291,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParams_v10000                           = 292,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemsetNode_v10000                                 = 293,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemsetNodeGetParams_v10000                           = 294,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemsetNodeSetParams_v10000                           = 295,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddHostNode_v10000                                   = 296,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphHostNodeGetParams_v10000                             = 297,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddChildGraphNode_v10000                             = 298,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphChildGraphNodeGetGraph_v10000                        = 299,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEmptyNode_v10000                                  = 300,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphClone_v10000                                         = 301,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeFindInClone_v10000                               = 302,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetType_v10000                                   = 303,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetRootNodes_v10000                                  = 304,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependencies_v10000                           = 305,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependentNodes_v10000                         = 306,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddDependencies_v10000                               = 307,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphRemoveDependencies_v10000                            = 308,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphDestroyNode_v10000                                   = 309,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v10000                                   = 310,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000                                        = 311,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000                                   = 312,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecDestroy_v10000                                   = 313,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphDestroy_v10000                                       = 314,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000                                 = 315,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_ptsz_v10000                            = 316,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamIsCapturing_v10000                                  = 317,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamIsCapturing_ptsz_v10000                             = 318,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_v10000                                   = 319,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_ptsz_v10000                              = 320,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphHostNodeSetParams_v10000                             = 321,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetNodes_v10000                                      = 322,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetEdges_v10000                                      = 323,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v10010                               = 324,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_ptsz_v10010                          = 325,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010                       = 326,
    CUPTI_RUNTIME_TRACE_CBID_cudaThreadExchangeStreamCaptureMode_v10010                    = 327,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetNvSciSyncAttributes_v10020                       = 328,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyAvailableDynamicSMemPerBlock_v10200              = 329,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetFlags_v10200                                     = 330,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetFlags_ptsz_v10200                                = 331,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParams_v10020                       = 332,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemsetNodeSetParams_v10020                       = 333,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecHostNodeSetParams_v10020                         = 334,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020                                    = 335,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetFuncBySymbol_v11000                                    = 336,
    CUPTI_RUNTIME_TRACE_CBID_cudaCtxResetPersistingL2Cache_v11000                          = 337,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeCopyAttributes_v11000                      = 338,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeGetAttribute_v11000                        = 339,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetAttribute_v11000                        = 340,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamCopyAttributes_v11000                               = 341,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamCopyAttributes_ptsz_v11000                          = 342,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetAttribute_v11000                                 = 343,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetAttribute_ptsz_v11000                            = 344,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_v11000                                 = 345,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_ptsz_v11000                            = 346,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetTexture1DLinearMaxWidth_v11010                   = 347,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_v10000                                        = 348,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_ptsz_v10000                                   = 349,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNodeToSymbol_v11010                         = 350,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNodeFromSymbol_v11010                       = 351,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNode1D_v11010                               = 352,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParamsToSymbol_v11010                   = 353,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParamsFromSymbol_v11010                 = 354,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParams1D_v11010                         = 355,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParamsToSymbol_v11010               = 356,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParamsFromSymbol_v11010             = 357,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParams1D_v11010                     = 358,
    CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetSparseProperties_v11010                           = 359,
    CUPTI_RUNTIME_TRACE_CBID_cudaMipmappedArrayGetSparseProperties_v11010                  = 360,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecChildGraphNodeSetParams_v11010                   = 361,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEventRecordNode_v11010                            = 362,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventRecordNodeGetEvent_v11010                       = 363,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventRecordNodeSetEvent_v11010                       = 364,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEventWaitNode_v11010                              = 365,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventWaitNodeGetEvent_v11010                         = 366,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventWaitNodeSetEvent_v11010                         = 367,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecEventRecordNodeSetEvent_v11010                   = 368,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecEventWaitNodeSetEvent_v11010                     = 369,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_v11010                               = 370,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_ptsz_v11010                          = 371,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetDefaultMemPool_v11020                            = 372,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020                                        = 373,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020                                   = 374,
    CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020                                          = 375,
    CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020                                     = 376,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolTrimTo_v11020                                      = 377,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolSetAttribute_v11020                                = 378,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolGetAttribute_v11020                                = 379,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolSetAccess_v11020                                   = 380,
    CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetPlane_v11020                                      = 381,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolGetAccess_v11020                                   = 382,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolCreate_v11020                                      = 383,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolDestroy_v11020                                     = 384,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetMemPool_v11020                                   = 385,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetMemPool_v11020                                   = 386,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolExportToShareableHandle_v11020                     = 387,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolImportFromShareableHandle_v11020                   = 388,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolExportPointer_v11020                               = 389,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolImportPointer_v11020                               = 390,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_v11020                                = 391,
    CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_ptsz_v11020                           = 392,
    CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_v2_v11020                   = 393,
    CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_v2_ptsz_v11020              = 394,
    CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v2_v11020                     = 395,
    CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v2_ptsz_v11020                = 396,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddExternalSemaphoresSignalNode_v11020               = 397,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresSignalNodeGetParams_v11020         = 398,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresSignalNodeSetParams_v11020         = 399,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddExternalSemaphoresWaitNode_v11020                 = 400,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresWaitNodeGetParams_v11020           = 401,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresWaitNodeSetParams_v11020           = 402,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecExternalSemaphoresSignalNodeSetParams_v11020     = 403,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecExternalSemaphoresWaitNodeSetParams_v11020       = 404,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceFlushGPUDirectRDMAWrites_v11030                     = 405,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPoint_v11030                                = 406,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPoint_ptsz_v11030                           = 407,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphDebugDotPrint_v11030                                 = 408,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_v11030                            = 409,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_ptsz_v11030                       = 410,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v11030                    = 411,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_ptsz_v11030               = 412,
    CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectCreate_v11030                                   = 413,
    CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectRetain_v11030                                   = 414,
    CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectRelease_v11030                                  = 415,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphRetainUserObject_v11030                              = 416,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphReleaseUserObject_v11030                             = 417,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040                          = 418,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemAllocNode_v11040                               = 419,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemAllocNodeGetParams_v11040                         = 420,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemFreeNode_v11040                                = 421,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemFreeNodeGetParams_v11040                          = 422,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGraphMemTrim_v11040                                 = 423,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetGraphMemAttribute_v11040                         = 424,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetGraphMemAttribute_v11040                         = 425,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetEnabled_v11060                                = 426,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetEnabled_v11060                                = 427,
    CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetMemoryRequirements_v11060                         = 428,
    CUPTI_RUNTIME_TRACE_CBID_cudaMipmappedArrayGetMemoryRequirements_v11060                = 429,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060                                    = 430,
    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060                               = 431,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxPotentialClusterSize_v11070                   = 432,
    CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveClusters_v11070                         = 433,
    CUPTI_RUNTIME_TRACE_CBID_cudaCreateTextureObject_v2_v11080                             = 434,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectTextureDesc_v2_v11080                     = 435,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000                         = 436,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000                    = 437,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecGetFlags_v12000                                  = 438,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetKernel_v12000                                          = 439,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v2_v12000                             = 440,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetId_v12000                                        = 441,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetId_ptsz_v12000                                   = 442,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000                                   = 443,
    CUPTI_RUNTIME_TRACE_CBID_cudaInitDevice_v12000                                         = 444,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddNode_v12020                                       = 445,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetParams_v12020                                 = 446,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecNodeSetParams_v12020                             = 447,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v2_v12020                                       = 448,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v2_v12020                                = 449,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v2_ptsz_v12020                           = 450,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetName_v12030                                        = 451,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_v12030                          = 452,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_ptsz_v12030                     = 453,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphConditionalHandleCreate_v12030                       = 454,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetEdges_v2_v12030                                   = 455,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependencies_v2_v12030                        = 456,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependentNodes_v2_v12030                      = 457,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddDependencies_v2_v12030                            = 458,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphRemoveDependencies_v2_v12030                         = 459,
    CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddNode_v2_v12030                                    = 460,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v3_v12030                            = 461,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v3_ptsz_v12030                       = 462,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v2_v12030                 = 463,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v2_ptsz_v12030            = 464,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceRegisterAsyncNotification_v12040                    = 465,
    CUPTI_RUNTIME_TRACE_CBID_cudaDeviceUnregisterAsyncNotification_v12040                  = 466,
    CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetParamInfo_v12040                                   = 467,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPointByVersion_v12050                       = 468,
    CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPointByVersion_ptsz_v12050                  = 469,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryLoadData_v12060                                    = 470,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryLoadFromFile_v12060                                = 471,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryUnload_v12060                                      = 472,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetKernel_v12060                                   = 473,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetGlobal_v12060                                   = 474,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetManaged_v12060                                  = 475,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetUnifiedFunction_v12060                          = 476,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetKernelCount_v12060                              = 477,
    CUPTI_RUNTIME_TRACE_CBID_cudaLibraryEnumerateKernels_v12060                            = 478,
    CUPTI_RUNTIME_TRACE_CBID_cudaKernelSetAttributeForDevice_v12060                        = 479,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevice_v12080                                    = 480,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevice_ptsz_v12080                               = 481,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_v12080                                   = 482,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_ptsz_v12080                              = 483,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_v12080                                 = 484,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_ptsz_v12080                            = 485,
    CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v2_v12080                                = 486,
    CUPTI_RUNTIME_TRACE_CBID_SIZE                                                          = 487,
    CUPTI_RUNTIME_TRACE_CBID_FORCE_INT                                                     = 0x7fffffff
} CUpti_runtime_api_trace_cbid;

#endif


/*
 * Copyright 2013-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

typedef enum {
  CUPTI_CBID_NVTX_INVALID                               = 0,
  CUPTI_CBID_NVTX_nvtxMarkA                             = 1,
  CUPTI_CBID_NVTX_nvtxMarkW                             = 2,
  CUPTI_CBID_NVTX_nvtxMarkEx                            = 3,
  CUPTI_CBID_NVTX_nvtxRangeStartA                       = 4,
  CUPTI_CBID_NVTX_nvtxRangeStartW                       = 5,
  CUPTI_CBID_NVTX_nvtxRangeStartEx                      = 6,
  CUPTI_CBID_NVTX_nvtxRangeEnd                          = 7,
  CUPTI_CBID_NVTX_nvtxRangePushA                        = 8,
  CUPTI_CBID_NVTX_nvtxRangePushW                        = 9,
  CUPTI_CBID_NVTX_nvtxRangePushEx                       = 10,
  CUPTI_CBID_NVTX_nvtxRangePop                          = 11,
  CUPTI_CBID_NVTX_nvtxNameCategoryA                     = 12,
  CUPTI_CBID_NVTX_nvtxNameCategoryW                     = 13,
  CUPTI_CBID_NVTX_nvtxNameOsThreadA                     = 14,
  CUPTI_CBID_NVTX_nvtxNameOsThreadW                     = 15,
  CUPTI_CBID_NVTX_nvtxNameCuDeviceA                     = 16,
  CUPTI_CBID_NVTX_nvtxNameCuDeviceW                     = 17,
  CUPTI_CBID_NVTX_nvtxNameCuContextA                    = 18,
  CUPTI_CBID_NVTX_nvtxNameCuContextW                    = 19,
  CUPTI_CBID_NVTX_nvtxNameCuStreamA                     = 20,
  CUPTI_CBID_NVTX_nvtxNameCuStreamW                     = 21,
  CUPTI_CBID_NVTX_nvtxNameCuEventA                      = 22,
  CUPTI_CBID_NVTX_nvtxNameCuEventW                      = 23,
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceA                   = 24,
  CUPTI_CBID_NVTX_nvtxNameCudaDeviceW                   = 25,
  CUPTI_CBID_NVTX_nvtxNameCudaStreamA                   = 26,
  CUPTI_CBID_NVTX_nvtxNameCudaStreamW                   = 27,
  CUPTI_CBID_NVTX_nvtxNameCudaEventA                    = 28,
  CUPTI_CBID_NVTX_nvtxNameCudaEventW                    = 29,
  CUPTI_CBID_NVTX_nvtxDomainMarkEx                      = 30,
  CUPTI_CBID_NVTX_nvtxDomainRangeStartEx                = 31,
  CUPTI_CBID_NVTX_nvtxDomainRangeEnd                    = 32,
  CUPTI_CBID_NVTX_nvtxDomainRangePushEx                 = 33,
  CUPTI_CBID_NVTX_nvtxDomainRangePop                    = 34,
  CUPTI_CBID_NVTX_nvtxDomainResourceCreate              = 35,
  CUPTI_CBID_NVTX_nvtxDomainResourceDestroy             = 36,
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryA               = 37,
  CUPTI_CBID_NVTX_nvtxDomainNameCategoryW               = 38,
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringA             = 39,
  CUPTI_CBID_NVTX_nvtxDomainRegisterStringW             = 40,
  CUPTI_CBID_NVTX_nvtxDomainCreateA                     = 41,
  CUPTI_CBID_NVTX_nvtxDomainCreateW                     = 42,
  CUPTI_CBID_NVTX_nvtxDomainDestroy                     = 43,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserCreate              = 44,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserDestroy             = 45,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireStart        = 46,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireFailed       = 47,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserAcquireSuccess      = 48,
  CUPTI_CBID_NVTX_nvtxDomainSyncUserReleasing           = 49,
  CUPTI_CBID_NVTX_SIZE,
  CUPTI_CBID_NVTX_FORCE_INT                             = 0x7fffffff
} CUpti_nvtx_api_trace_cbid;

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif    


/* To support function parameter structures for obsoleted API. See
   cuda.h for the actual definition of these structures. */
// typedef unsigned int CUdeviceptr_v1;
// typedef struct CUDA_MEMCPY2D_v1_st { int dummy; } CUDA_MEMCPY2D_v1;
// typedef struct CUDA_MEMCPY3D_v1_st { int dummy; } CUDA_MEMCPY3D_v1;
// typedef struct CUDA_ARRAY_DESCRIPTOR_v1_st { int dummy; } CUDA_ARRAY_DESCRIPTOR_v1;
// typedef struct CUDA_ARRAY3D_DESCRIPTOR_v1_st { int dummy; } CUDA_ARRAY3D_DESCRIPTOR_v1;

/* Function parameter structures */
// #include <generated_cuda_runtime_api_meta.h>
// #include <generated_cuda_meta.h>

/* The following parameter structures cannot be included unless a
   header that defines GL_VERSION is included before including them.
   If these are needed then make sure such a header is included
   already. */
#ifdef GL_VERSION
// #include <generated_cuda_gl_interop_meta.h>
// #include <generated_cudaGL_meta.h>
#endif

//#include <generated_nvtx_meta.h>

/* The following parameter structures cannot be included by default as
   they are not guaranteed to be available on all systems. Uncomment
   the includes that are available, or use the include explicitly. */
#if defined(__linux__)
//#include <generated_cuda_vdpau_interop_meta.h>
//#include <generated_cudaVDPAU_meta.h>
#endif

#ifdef _WIN32
//#include <generated_cuda_d3d9_interop_meta.h>
//#include <generated_cuda_d3d10_interop_meta.h>
//#include <generated_cuda_d3d11_interop_meta.h>
//#include <generated_cudaD3D9_meta.h>
//#include <generated_cudaD3D10_meta.h>
//#include <generated_cudaD3D11_meta.h>
#endif

#endif /*_CUPTI_H_*/


