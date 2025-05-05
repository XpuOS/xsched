/*
 * Copyright 2020-2021 NVIDIA Corporation.  All rights reserved.
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

#pragma once

#include <stdint.h>
#include <stddef.h>

/**
 * \file cudla.h
 * \brief Header file for the cuDLA application programming interface.
 */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup CUDLA_TYPES Data types used by cuDLA driver
 * @{
 */

#define CUDLA_VER_MAJOR 1U
#define CUDLA_VER_MINOR 4U
#define CUDLA_VER_PATCH 0U

#define CUDLA_DATA_FORMAT_UNKNOWN   0U
#define CUDLA_DATA_FORMAT_NCHW      1U
#define CUDLA_DATA_FORMAT_NHWC      2U
#define CUDLA_DATA_FORMAT_NCxHWx    3U

#define CUDLA_DATA_TYPE_UNKNOWN 0U
#define CUDLA_DATA_TYPE_FLOAT   1U
#define CUDLA_DATA_TYPE_HALF    2U
#define CUDLA_DATA_TYPE_INT16   3U
#define CUDLA_DATA_TYPE_INT8    4U
#define CUDLA_DATA_TYPE_UINT8   5U
#define CUDLA_DATA_TYPE_UINT16  6U

#define CUDLA_DATA_CATEGORY_IMAGE   0U
#define CUDLA_DATA_CATEGORY_WEIGHT  1U
#define CUDLA_DATA_CATEGORY_FEATURE 2U
#define CUDLA_DATA_CATEGORY_PLANAR  3U
#define CUDLA_DATA_CATEGORY_BIAS    4U

#define CUDLA_PIXEL_FORMAT_R8 0U
#define CUDLA_PIXEL_FORMAT_R10 1U
#define CUDLA_PIXEL_FORMAT_R12 2U
#define CUDLA_PIXEL_FORMAT_R16 3U
#define CUDLA_PIXEL_FORMAT_R16_I 4U
#define CUDLA_PIXEL_FORMAT_R16_F 5U
#define CUDLA_PIXEL_FORMAT_A16B16G16R16 6U
#define CUDLA_PIXEL_FORMAT_X16B16G16R16 7U
#define CUDLA_PIXEL_FORMAT_A16B16G16R16_F 8U
#define CUDLA_PIXEL_FORMAT_A16Y16U16V16 9U
#define CUDLA_PIXEL_FORMAT_V16U16Y16A16 10U
#define CUDLA_PIXEL_FORMAT_A16Y16U16V16_F 11U
#define CUDLA_PIXEL_FORMAT_A8B8G8R8 12U
#define CUDLA_PIXEL_FORMAT_A8R8G8B8 13U
#define CUDLA_PIXEL_FORMAT_B8G8R8A8 14U
#define CUDLA_PIXEL_FORMAT_R8G8B8A8 15U
#define CUDLA_PIXEL_FORMAT_X8B8G8R8 16U
#define CUDLA_PIXEL_FORMAT_X8R8G8B8 17U
#define CUDLA_PIXEL_FORMAT_B8G8R8X8 18U
#define CUDLA_PIXEL_FORMAT_R8G8B8X8 19U
#define CUDLA_PIXEL_FORMAT_A2B10G10R10 20U
#define CUDLA_PIXEL_FORMAT_A2R10G10B10 21U
#define CUDLA_PIXEL_FORMAT_B10G10R10A2 22U
#define CUDLA_PIXEL_FORMAT_R10G10B10A2 23U
#define CUDLA_PIXEL_FORMAT_A2Y10U10V10 24U
#define CUDLA_PIXEL_FORMAT_V10U10Y10A2 25U
#define CUDLA_PIXEL_FORMAT_A8Y8U8V8 26U
#define CUDLA_PIXEL_FORMAT_V8U8Y8A8 27U
#define CUDLA_PIXEL_FORMAT_Y8___U8V8_N444 28U
#define CUDLA_PIXEL_FORMAT_Y8___V8U8_N444 29U
#define CUDLA_PIXEL_FORMAT_Y10___U10V10_N444 30U
#define CUDLA_PIXEL_FORMAT_Y10___V10U10_N444 31U
#define CUDLA_PIXEL_FORMAT_Y12___U12V12_N444 32U
#define CUDLA_PIXEL_FORMAT_Y12___V12U12_N444 33U
#define CUDLA_PIXEL_FORMAT_Y16___U16V16_N444 34U
#define CUDLA_PIXEL_FORMAT_Y16___V16U16_N444 35U
#define CUDLA_PIXEL_FORMAT_FEATURE 36U
#define CUDLA_PIXEL_FORMAT_UNKNOWN 37U

#define CUDLA_PIXEL_MAPPING_PITCH_LINEAR 0U
#define CUDLA_PIXEL_MAPPING_BLOCK_LINEAR 1U

#define CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES 8U
#define CUDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN 80U

/**
 * Error codes.
 */
enum cudlaStatus {
    /**
     * The API call returned with no errors.
     */
    cudlaSuccess                                           = 0,
    /**
     * This indicates that one or more parameters passed
     * to the API is/are incorrect.
     */
    cudlaErrorInvalidParam                                 = 1,
    /**
     * This indicates that the API call failed due to
     * lack of underlying resources.
     */
    cudlaErrorOutOfResources                               = 2,
    /**
     * This indicates that an internal error occurred
     * during creation of device handle.
     */
    cudlaErrorCreationFailed                               = 3,
    /**
     * This indicates that the memory object being 
     * passed in the API call has not been registered
     * before.
     */
    cudlaErrorInvalidAddress                               = 4,
    /**
     * This indicates that an OS error occurred.
     */
    cudlaErrorOs                                           = 5,
    /**
     * This indicates that there was an error in a
     * CUDA operation as part of the API call.
     */
    cudlaErrorCuda                                         = 6,
    /**
     * This indicates that there was an error in the
     * DLA runtime for the API call.
     */
    cudlaErrorUmd                                          = 7,
    /**
     * This indicates that the device handle passed
     * to the API call is invalid.
     */
    cudlaErrorInvalidDevice                                = 8,
    /**
     * This indicates that an invalid attribute is
     * being requested.
     */
    cudlaErrorInvalidAttribute                             = 9,
    /**
     * This indicates that the underlying DLA runtime
     * is incompatible with the current cuDLA version.
     */
    cudlaErrorIncompatibleDlaSWVersion                     = 10,
    /**
     * This indicates that the memory object is
     * already registered.
     */
    cudlaErrorMemoryRegistered                             = 11,
    /**
     * This indicates that the module being passed
     * is invalid.
     */
    cudlaErrorInvalidModule                                = 12,
    /**
     * This indicates that the operation being
     * requested by the API call is unsupported.
     */
    cudlaErrorUnsupportedOperation                         = 13,
    /**
     * This indicates that the NvSci operation
     * requested by the API call failed.
     */
    cudlaErrorNvSci                                        = 14,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrInvalidInput                           = 0x40000001,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrInvalidPreAction                       = 0x40000002,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrNoMem                                  = 0x40000003,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrProcessorBusy                          = 0x40000004,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrTaskStatusMismatch                     = 0x40000005,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrEngineTimeout                          = 0x40000006,
    /**
     * DLA HW Error.
     */
    cudlaErrorDlaErrDataMismatch                           = 0x40000007,
    /**
     * This indicates that an unknown error has
     * occurred.
     */
    cudlaErrorUnknown                                      = 0x7fffffff
};

typedef enum cudlaStatus cudlaStatus;

/**
 * Device creation modes.
 */
enum cudlaMode {
    /**
     * Hyrbid mode.
     */
    CUDLA_CUDA_DLA         = 0,
    /**
     * Standalone mode.
     */
    CUDLA_STANDALONE       = 1
};

typedef enum cudlaMode cudlaMode;

/**
 * cuDLA NvSciSync attributes.
 */
enum cudlaNvSciSyncAttributes {
    /**
     * Wait attribute.
     */
    CUDLA_NVSCISYNC_ATTR_WAIT   = 1,
    /**
     * Signal attribute.
     */
    CUDLA_NVSCISYNC_ATTR_SIGNAL = 2
};

typedef enum cudlaNvSciSyncAttributes cudlaNvSciSyncAttributes;

/**
 * External memory handle descriptor.
 */
struct cudlaExternalMemoryHandleDesc_t {
    /**
     * A handle representing an external memory object.
     */
    const void* extBufObject;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
};
typedef struct cudlaExternalMemoryHandleDesc_t cudlaExternalMemoryHandleDesc;

/**
 * External semaphore handle descriptor.
 */
struct cudlaExternalSemaphoreHandleDesc_t {
    /**
     * A handle representing an external synchronization object.
     */
    const void* extSyncObject;
};
typedef struct cudlaExternalSemaphoreHandleDesc_t cudlaExternalSemaphoreHandleDesc;

struct cudlaDevHandle_t;
typedef struct cudlaDevHandle_t* cudlaDevHandle;

struct cudlaModule_t;
typedef struct cudlaModule_t* cudlaModule;

/**
 * Module attribute types.
 */
enum cudlaModuleAttributeType {
    /**
     * Flag to retrieve number of input tensors.
     */
    CUDLA_NUM_INPUT_TENSORS = 0,
    /**
     * Flag to retrieve number of output tensors.
     */
    CUDLA_NUM_OUTPUT_TENSORS = 1,
    /**
     * Flag to retrieve all the input tensor descriptors.
     */
    CUDLA_INPUT_TENSOR_DESCRIPTORS = 2,
    /**
     * Flag to retrieve all the output tensor descriptors.
     */
    CUDLA_OUTPUT_TENSOR_DESCRIPTORS = 3,
    /**
     * Flag to retrieve total number of output task statistics buffer.
     */
    CUDLA_NUM_OUTPUT_TASK_STATISTICS = 4,
    /**
     * Flag to retrieve all the output task statistics descriptors.
     */
    CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS = 5,
};

typedef enum cudlaModuleAttributeType cudlaModuleAttributeType;

/**
 * Tensor descriptor.
 */
typedef struct cudlaModuleTensorDescriptor_t {
    char name[CUDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN + 1];
    uint64_t size;
    uint64_t n;
    uint64_t c;
    uint64_t h;
    uint64_t w;
    uint8_t dataFormat;
    uint8_t dataType;
    uint8_t dataCategory;
    uint8_t pixelFormat;
    uint8_t pixelMapping;
    uint32_t stride[CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES];
} cudlaModuleTensorDescriptor;

/**
 * Module attribute.
 */
typedef union cudlaModuleAttribute_t {
    /**
     * Returns the number of input tensors.
     */
    uint32_t numInputTensors;
    /**
     * Returns the number of output tensors.
     */
    uint32_t numOutputTensors;
    /**
     * Returns an array of input tensor descriptors.
     */
    cudlaModuleTensorDescriptor* inputTensorDesc;
    /**
     * Returns an array of output tensor descriptors.
     */
    cudlaModuleTensorDescriptor* outputTensorDesc;
} cudlaModuleAttribute;

/**
 * Supported fence types.
 */
enum cudlaFenceType {
    /**
     * NvSciSync fence type for EOF.
     */
    CUDLA_NVSCISYNC_FENCE = 1,

    /*
     * NvSciSync fence type for SOF.
     */
    CUDLA_NVSCISYNC_FENCE_SOF = 2
};

typedef enum cudlaFenceType cudlaFenceType;

/**
 * Fence description.
 */
typedef struct CudlaFence {
    /**
     * Fence.
     */
    void* fence;
    /**
     * Fence type.
     */
    cudlaFenceType type;
} CudlaFence;

/**
 * Wait events for ::cudlaSubmitTask.
 */
typedef struct cudlaWaitEvents {
    /**
     * Array of fence pointers for all the wait events.
     */
    const CudlaFence* preFences;
    /**
     * Total number of wait events.
     */
    uint32_t numEvents;
} cudlaWaitEvents;

/**
 * Signal events for ::cudlaSubmitTask
 */
typedef struct cudlaSignalEvents {
    /**
     * Array of registered synchronization objects (via ::cudlaImportExternalSemaphore).
     */
    uint64_t* const * devPtrs;
    /**
     * Array of fences pointers for all the signal events
     * corresponding to the synchronization objects.
     */
    CudlaFence* eofFences;
    /**
     * Total number of signal events.
     */
    uint32_t numEvents;
} cudlaSignalEvents;

/**
 * Module load flags for ::cudlaModuleLoadFromMemory.
 */
enum cudlaModuleLoadFlags {
    /**
     * Default flag.
     */
    CUDLA_MODULE_DEFAULT = 0,
	/**
	 * Flag to load a module that is used to perform
	 * permanent fault diagnostics for DLA HW.
	 */
	CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS = 1
};

typedef enum cudlaModuleLoadFlags cudlaModuleLoadFlags;

/**
 * Task submission flags for ::cudlaSubmitTask.
 */
enum cudlaSubmissionFlags {
    /**
     * Flag to specify that the submitted task must be 
     * bypassed for execution.
     */
    CUDLA_SUBMIT_NOOP = 1,
    /**
     * Flag to specify that the global lock acquire must
     * be skipped.
     */
    CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE = 1<<1,
	/**
	 * Flag to specify that the submitted task is to
     * run permanent fault diagnostics for DLA HW.
	 */
	CUDLA_SUBMIT_DIAGNOSTICS_TASK = 1<<2
};

typedef enum cudlaSubmissionFlags cudlaSubmissionFlags;

/**
 * Access permission flags for importing NvSciBuffers
 */
enum cudlaAccessPermissionFlags {
    /**
     * Flag to import memory with read-write permission
     */
    CUDLA_READ_WRITE_PERM = 0,
    /**
     * Flag to import memory with read-only permission
     */
    CUDLA_READ_ONLY_PERM = 1,
    /**
     * Flag to indicate buffer as layerwise statistics buffer.
     */
    CUDLA_TASK_STATISTICS = 1<<1
};

typedef enum cudlaAccessPermissionFlags cudlaAccessPermissionFlags;

/**
 * Structure of Task.
 */
typedef struct cudlaTask_t {
    /**
     * cuDLA module handle.
     */
    cudlaModule moduleHandle;
    /**
     * Array of output tensors.
     */
    uint64_t* const * outputTensor;
    /**
     * Number of output tensors.
     */
    uint32_t numOutputTensors;
    /**
     * Number of input tensors.
     */
    uint32_t numInputTensors;
    /**
     * Array of input tensors.
     */
    const uint64_t* const * inputTensor;
    /**
     * Wait events.
     */
    const cudlaWaitEvents* waitEvents;
    /**
     * Signal events.
     */
    cudlaSignalEvents* signalEvents;
} cudlaTask;

/**
 * Device attribute type.
 */
enum cudlaDevAttributeType {
    /**
     * Flag to check for support for UVA.
     */
    CUDLA_UNIFIED_ADDRESSING = 0,
    /**
     * Flag to check for DLA HW version.
     */
    CUDLA_DEVICE_VERSION = 1
};

typedef enum cudlaDevAttributeType cudlaDevAttributeType;

/**
 * Device attribute.
 */
typedef union cudlaDevAttribute_t {
    /**
     *  Returns 0 if unified addressing is not supported.
     */
    uint8_t unifiedAddressingSupported;
    /**
     *  DLA device version. Xavier has 1.0 and Orin has 2.0.
     */
    uint32_t deviceVersion;
} cudlaDevAttribute;

/** @} */ /* END CUDLA_TYPES */

/**
 * \defgroup CUDLA_API cuDLA API
 *
 * ___MANBRIEF___ cuDLA API
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the application programming interface
 * of the cuDLA driver.
 *
 * @{
 */

/**
 * \brief Returns the version number of the library.
 *
 * cuDLA is semantically versioned. This function will return the
 * version as 1000000*major + 1000*minor + patch.
 *
 * \param[out] version - cuDLA library version will be available
 *                    in this variable upon successful execution.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam
 */
cudlaStatus cudlaGetVersion(uint64_t * const version);

/**
 * \brief Get device count.
 *
 * Get number of DLA devices available to use.
 *
 * \param[out] pNumDevices - The number of DLA devices will be
 *                           available in this variable upon 
 *                           successful completion.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorUmd,
 * ::cudlaErrorIncompatibleDlaSWVersion
*/
cudlaStatus cudlaDeviceGetCount(uint64_t * const pNumDevices);

/**
 * \brief Create a device handle.
 *
 * Creates an instance of a cuDLA device  which can be used to 
 * submit DLA operations. The application can create the handle
 * in hybrid or standalone mode. In hybrid mode, the current
 * set GPU device is used by this API to decide the association
 * of the created DLA device handle. This function returns 
 * ::cudlaErrorUnsupportedOperation if the current set GPU device
 * is a dGPU as cuDLA is not supported on dGPU presently.
 *
 * \param[in] device - Device number (can be 0 or 1).
 * \param[in] flags - Flags controlling device creation.
 * \param[out] devHandle - Pointer to hold the created cuDLA device handle.
 *
 * Valid values for \p flags are:
 * - ::CUDLA_CUDA_DLA - In this mode, cuDLA serves as a programming 
 *                    model extension of CUDA wherein DLA work can be 
 *                    submitted using CUDA constructs.
 * - ::CUDLA_STANDALONE - In this mode, cuDLA works standalone without 
 *                      any interaction with CUDA.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorOutOfResources,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorIncompatibleDlaSWVersion,
 * ::cudlaErrorCreationFailed,
 * ::cudlaErrorCuda,
 * ::cudlaErrorUmd,
 * ::cudlaErrorUnsupportedOperation
 */
cudlaStatus cudlaCreateDevice(uint64_t const device,
                              cudlaDevHandle * const devHandle,
                              uint32_t const flags);

/**
 * \brief Registers the CUDA memory to DLA engine.
 *
 * As part of registration, a system mapping is created whereby 
 * the DLA HW can access the underlying CUDA memory. The resultant
 * mapping is available in devPtr and applications must use this
 * mapping while referring this memory in submit operations.
 *
 * This function will return ::cudlaErrorInvalidAddress if the pointer
 * or size to be registered is invalid. In addition, if the input
 * pointer was already registered, then this function will return
 * ::cudlaErrorMemoryRegistered. Attempting to re-register memory
 * does not cause any irrecoverable error in cuDLA and applications
 * can continue to use cuDLA APIs even after this error has occurred.
 *
 * \param[in] devHandle - A valid cuDLA device handle create by 
 *                        a previous call to ::cudlaCreateDevice.
 * \param[in] ptr - The CUDA pointer to be registered.
 * \param[in] size - The size of the mapping i.e the number of 
 *                   bytes from ptr that must be mapped.
 * \param[out] devPtr - The output pointer where the mapping will 
 *                      be available.
 * \param[in] flags - Applications can use this flag to control several
 *                    aspects of the registration process.
 *
 * The valid values of \p flags can be one of the following (or an OR of these values):
 * - 0, default
 * - ::CUDLA_TASK_STATISTICS, specifies that the external memory needs to be registered 
 *                            with DLA for layerwise statistics.
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidAddress,
 * ::cudlaErrorCuda,
 * ::cudlaErrorUmd,
 * ::cudlaErrorOutOfResources,
 * ::cudlaErrorMemoryRegistered,
 * ::cudlaErrorUnsupportedOperation
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaMemRegister(cudlaDevHandle const devHandle,
                             const uint64_t* const ptr,
                             size_t const size,
                             uint64_t** const devPtr,
                             uint32_t const flags);

/**
 * \brief Load a DLA module.
 *
 * Loads the module into the current device handle. Currently,
 * DLA supports only 1 loadable per device handle. So, attempting
 * to load another loadable in the same device handle would return
 * with an error code of ::cudlaErrorUnsupportedOperation.
 *
 * \param[in] devHandle - The input cuDLA device handle. The module 
 *                        will be loaded in the context of this
 *                        handle.
 * \param[in] pModule - A pointer to an in-memory module.
 * \param[in] moduleSize - The size of the module.
 * \param[out] hModule - The address in which the loaded module handle 
 *                      will be available upon successful execution.
 * \param[in] flags - Applications can use this flag to specify how the
 *					  module is going to be used.
 *
 * The valid values of \p flags can be one of the following:
 * - :: CUDLA_MODULE_DEFAULT, Default value which is 0.
 * - :: CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS, Application can specify this flag to load a
 *									   module that is used for performing fault
 *									   diagnostics for DLA HW. With this flag set, the
 *									   \p pModule and \p moduleSize parameters shall be
 *								       NULL and 0 as the diagnostics module is loaded
 *									   internally.
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorOutOfResources,
 * ::cudlaErrorUnsupportedOperation,
 * ::cudlaErrorUmd
 */
cudlaStatus cudlaModuleLoadFromMemory(cudlaDevHandle const devHandle,
                                      const uint8_t * const pModule,
                                      size_t const moduleSize,
                                      cudlaModule* const hModule,
                                      uint32_t const flags);

/**
 * \brief Get DLA module attributes.
 *
 * Get module attributes from the loaded module. This API returns
 * ::cudlaErrorInvalidDevice if the module is not loaded in any
 * device.
 *
 * \param[in] hModule - The input DLA module.
 * \param[in] attrType - The attribute type that is being requested.
 * \param[out] attribute - The output pointer where the attribute
 *                         will be available.
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidModule,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorUmd,
 * ::cudlaErrorInvalidAttribute,
 * ::cudlaErrorUnsupportedOperation
 */
cudlaStatus cudlaModuleGetAttributes(cudlaModule const hModule,
                                     cudlaModuleAttributeType const attrType,
                                     cudlaModuleAttribute* const attribute);

/**
 * \brief Unload a DLA module.
 *
 * Unload the module from the device handle that it was loaded into. This
 * API returns ::cudlaErrorInvalidDevice if the module is not loaded into
 * a valid device.
 *
 * \param[in] hModule - Handle to the loaded module.
 * \param[in] flags - Reserved for future. Must be set to 0.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorInvalidModule,
 * ::cudlaErrorUmd
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaModuleUnload(cudlaModule const hModule, uint32_t const flags);

/**
 * \brief Submits the inference operation on DLA.
 *
 * This operation takes in a sequence of tasks and submits them
 * to the DLA HW for execution in the same sequence as they appear
 * in the input task array. The input and output tensors (and statistics 
 * buffer if used) are assumed to be pre-registered using ::cudlaMemRegister 
 * (in hybrid mode) or ::cudlaImportExternalMemory (in standalone mode). 
 * Failure to do so can result in this function returning ::cudlaErrorInvalidAddress.
 *
 * The \p stream parameter must be specified as the CUDA stream on 
 * which the DLA task is submitted for execution in hybrid mode.
 * In standalone mode, this parameter must be passed as NULL and
 * failure to do so will result in this function returning
 * ::cudlaErrorInvalidParam.
 *
 * The ::cudlaTask structure has a provision to specify wait and 
 * signal events that cuDLA must wait on and signal respectively as 
 * part of cudlaSubmitTask(). Each submitted task will wait for all its 
 * wait events to be signaled before beginning execution and will provide 
 * a signal event (if one is requested for during ::cudlaSubmitTask) that 
 * the application (or any other entity) can wait on to ensure that 
 * the submitted task has completed execution. In cuDLA 1.0, only 
 * NvSciSync fences are supported as part of wait events. Furthermore, 
 * only NvSciSync objects (registered as part of ::cudlaImportExternalSemaphore) 
 * can be signaled as part of signal events and the fence corresponding to 
 * the signaled event is returned as part of ::cudlaSubmitTask.
 *
 * In standalone mode, if inputTensor and outputTensor fields are set to NULL
 * inside the ::cudlaTask structure, the task submission is interpreted as an
 * enqueue of wait and signal events that must be considered for subsequent task
 * submissions. No actual task submission is done. Multiple such subsequent task
 * submissions with NULL fields in the input/outputTensor fields will overwrite
 * the list of wait and signal events to be considered. In other words, the wait
 * and signal events considered are effectively what are specified in the last
 * submit call with NULL data fields.
 * During an actual task submit in standalone mode, the effective wait events and
 * signal events that will be considered are what the application sets using NULL
 * data submissions and what is set for that particular task submission in the
 * waitEvents and signalEvents fields. The wait events set as part of NULL data
 * submission are considered as dependencies for only the first task and the signal
 * events set as part of NULL data submission are signaled when the last task of
 * task list is complete.
 * All constraints that apply to waitEvents and signalEvents individually
 * (as described below) are also applicable to the combined list.
 * 
 * For wait events, applications are expected to 
 * - register their synchronization objects using ::cudlaImportExternalSemaphore.
 * - create the required number of fence placeholders using ::CudlaFence.
 * - fill in the placeholders with the relevant fences from the application.
 * - list out all the fences in ::cudlaWaitEvents.
 *
 * For signal events, applications are expected to
 * - register their synchronization objects using ::cudlaImportExternalSemaphore.
 * - create the required number of placeholder fences using ::CudlaFence.
 *   cuDLA supports 2 kinds of Fences, SOF and EOF Fence.
 *   - SOF(Start Of Frame) Fence is the type of fence which is signaled
 *     before the task execution on DLA starts. Use cudlaFenceType as
 *     CUDLA_NVSCISYNC_FENCE_SOF to mark a fence as SOF fence.
 *   - EOF(End Of Frame) Fence is the type of fence which is signaled after the
 *     task execution on DLA is complete. Use cudlaFenceType as
 *     CUDLA_NVSCISYNC_FENCE to mark a fence as EOF fence.
 * - place the registered objects and the corresponding fences in ::cudlaSignalEvents.
 * In case ofdeterministic semaphore, fence is not required to be passed in
 * ::cudlaSignalEvents.\n
 * When ::cudlaSubmitTask returns successfully, the fences present in 
 * ::cudlaSignalEvents can be used to wait for the particular task to be completed.
 * cuDLA supports 1 sync point and any number of semaphores as part of
 * ::cudlaSignalEvents. If more than 1 sync point is specified,
 * ::cudlaErrorInvalidParam is returned.
 *
 * During submission, users have an option to enable layerwise statistics profiling 
 * for the individual layers of the network. This option needs to be exercised by 
 * specifying additional output buffers that would contain the profiling information. Specifically,
 * - "cudlaTask::numOutputTensors" should be the sum of value returned by 
 * cudlaModuleGetAttributes(...,CUDLA_NUM_OUTPUT_TENSORS,...) and
 * cudlaModuleGetAttributes(...,CUDLA_NUM_OUTPUT_TASK_STATISTICS,...) 
 * - "cudlaTask::outputTensor" should contain the array of output tensors appended 
 * with array of statistics output buffer.
 *
 * This function can return ::cudlaErrorUnsupportedOperation if
 * - stream being used in hybrid mode is in capturing state.
 * - application attempts to use NvSci functionalities in hybrid mode.
 * - loading of NvSci libraries failed for a particular platform.
 * - fence type other than ::CUDLA_NVSCISYNC_FENCE is specified.
 * - waitEvents or signaEvents is not NULL in hybrid mode.
 * - inputTensor or outputTensor is NULL in hybrid mode and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
 * - inputTensor is NULL and outputTensor is not NULL and vice versa in standalone mode
 *   and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
 * - inputTensor and outputTensor is NULL and number of tasks is not equal to 1 in standalone mode
 *   and the flags are not CUDLA_SUBMIT_DIAGNOSTICS_TASK.
 * - inputTensor is not NULL or output tensor is NULL and the flags are CUDLA_SUBMIT_DIAGNOSTICS_TASK.
 * - the effective signal events list has multiple sync points to signal.
 * - if layerwise feature is unsupported.
 *
 * This function can return ::cudlaErrorNvSci or ::cudlaErrorInvalidAttribute
 * in certain cases when the underlying NvSci operation fails.
 *
 * This function can return ::cudlaErrorOs if an internal system operation
 * fails.
 *
 * \param[in] devHandle - A valid cuDLA device handle.
 * \param[in,out] ptrToTasks - A list of inferencing tasks.
 * \param[in] numTasks - The number of tasks.
 * \param[in] stream - The stream on which the DLA task has to be submitted.
 * \param[in] flags - Applications can use this flag to control several
 *                    aspects of the submission process.
 *
 * The valid values of \p flags can be one of the following (or an OR of these values):
 * - 0, default
 * - ::CUDLA_SUBMIT_NOOP, specifies that the submitted task must be skipped
 *                        during execution on the DLA. However, all the waitEvents
 *                        and signalEvents dependencies must be satisfied. This
 *                        flag is ignored when NULL data submissions are being done
 *                        as in that case only the wait and signal events are internally
 *                        stored for the next task submission.
 * - ::CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE, specifies that the submitted task is being
 *                              enqueued in a device handle and that no
 *                              other task is being enqueued in that device handle
 *                              at that time in any other thread. This is a flag
 *                              that apps can use as an optimization. Ordinarily,
 *                              the cuDLA APIs acquire a global lock internally
 *                              to guarantee thread safety. However, this lock
 *                              causes unwanted serialization in cases where the
 *                              the applications are submitting tasks to different
 *                              device handles. If an application was submitting 
 *                              one or more tasks in multiple threads and if these
 *                              submissions are to different device handles and if
 *                              there is no shared data being provided as part of
 *                              the task information in the respective submissions
 *                              then applications can specify this flag during submission
 *                              so that the internal lock acquire is skipped. Shared
 *                              data also includes the input stream in hybrid mode
 *                              operation. Therefore, if the same stream is being used
 *                              to submit two different tasks and even if the two
 *                              device handles are different, the usage of this
 *                              flag is invalid.
 * - ::CUDLA_SUBMIT_DIAGNOSTICS_TASK, specifies that the submitted task is to run
 *								permanent fault diagnostics for DLA HW. User can
 *								use this task to probe the state of DLA HW. With this
 *								flag set, in standalone mode user is not allowed to
 *								do event only submissions, where tensor information
 *								is NULL and only events (wait/signal or both) are present
 *								in task. This is because the task always runs on a internally 
 *								loaded diagnostic module. This diagnostic module does not
 *								expect any input tensors and so input tensor memory, however
 *								user is expected to query no. of output tensors, allocate the
 *								output tensor memory and pass the same while using the submit
 *								task.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorInvalidModule,
 * ::cudlaErrorCuda,
 * ::cudlaErrorUmd,
 * ::cudlaErrorOutOfResources,
 * ::cudlaErrorInvalidAddress,
 * ::cudlaErrorUnsupportedOperation,
 * ::cudlaErrorInvalidAttribute,
 * ::cudlaErrorNvSci
 * ::cudlaErrorOs
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaSubmitTask(cudlaDevHandle const devHandle,
                            const cudlaTask * const ptrToTasks,
                            uint32_t const numTasks,
                            void* const stream,
                            uint32_t const flags);

/**
 * \brief Get cuDLA device attributes.
 *
 * UVA addressing between CUDA and DLA requires special support in
 * the underlying kernel mode drivers. Applications are expected to
 * query the cuDLA runtime to check if the current version of
 * cuDLA supports UVA addressing.
 *
 * \param[in] devHandle - The input cuDLA device handle.
 * \param[in] attrib - The attribute that is being requested.
 * \param[out] pAttribute - The output pointer where the attribute
 *                         will be available.
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorUmd,
 * ::cudlaErrorInvalidAttribute
 */
cudlaStatus cudlaDeviceGetAttribute(cudlaDevHandle const devHandle,
                                    cudlaDevAttributeType const attrib,
                                    cudlaDevAttribute* const pAttribute);

/**
 * \brief Unregisters the input memory from DLA engine.
 *
 * The system mapping that enables the DLA HW to access the memory
 * is removed. This mapping could have been created by a previous
 * call to ::cudlaMemRegister , ::cudlaImportExternalMemory or
 * ::cudlaImportExternalSemaphore.
 *
 * \param[in] devHandle - A valid cuDLA device handle create by 
 *                        a previous call to ::cudlaCreateDevice.
 * \param[in] devPtr - The pointer to be unregistered.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorInvalidAddress,
 * ::cudlaErrorUmd
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaMemUnregister(cudlaDevHandle const devHandle,
                               const uint64_t * const devPtr);

/**
 * \brief Gets the last asynchronous error in task execution.
 *
 * The DLA tasks execute asynchronously on the DLA HW. As a 
 * result, the status of the task execution is not known at
 * the time of task submission. The status of the task 
 * executed by the DLA HW most recently for the particular 
 * device handle can be queried using this interface.
 *
 * Note that a return code of ::cudlaSuccess from this function
 * does not necessarily imply that most recent task executed
 * successfully. Since this function returns immediately, it 
 * can only report the status of the tasks at the snapshot
 * of time when it is called. To be guaranteed of task completion,
 * applications must synchronize on the submitted tasks in 
 * hybrid or standalone modes and then call this API to check
 * for errors.
 *
 * \param[in] devHandle - A valid device handle.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorDlaErrInvalidInput,
 * ::cudlaErrorDlaErrInvalidPreAction,
 * ::cudlaErrorDlaErrNoMem,
 * ::cudlaErrorDlaErrProcessorBusy,
 * ::cudlaErrorDlaErrTaskStatusMismatch,
 * ::cudlaErrorDlaErrEngineTimeout,
 * ::cudlaErrorDlaErrDataMismatch,
 * ::cudlaErrorUnknown
 */
cudlaStatus cudlaGetLastError(cudlaDevHandle const devHandle);

/**
 * \brief Destroy device handle.
 *
 * Destroys the instance of the cuDLA device which was created with
 * cudlaCreateDevice. Before destroying the handle, it is important
 * to ensure that all the tasks submitted previously to the device
 * are completed. Failure to do so can lead to application crashes.
 *
 * In hybrid mode, cuDLA internally performs memory allocations with
 * CUDA using the primary context.  As a result, before destroying 
 * or resetting a CUDA primary context, it is mandatory that all 
 * cuDLA device initializations are destroyed.
 *
 * \param[in] devHandle - A valid device handle.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorCuda,
 * ::cudlaErrorUmd
 */
cudlaStatus cudlaDestroyDevice(cudlaDevHandle const devHandle);

/**
 * \brief Imports external memory into cuDLA.
 *
 * Imports the allocated external memory by registering it with DLA. 
 * After successful registration, the returned pointer can be used in a 
 * task submit.
 *
 * On Tegra, cuDLA supports importing NvSciBuf objects in standalone mode
 * only. In the event of failed NvSci initialization (either due to 
 * usage of this API in hybrid mode or an issue in the NvSci library
 * initialization), this function would return ::cudlaErrorUnsupportedOperation.
 * This function can return ::cudlaErrorNvSci or ::cudlaErrorInvalidAttribute
 * in certain cases when the underlying NvSci operation fails.
 *
 * \param[in] devHandle - A valid device handle.
 * \param[in] desc - Contains description about allocated
 *                   external memory.
 * \param[out] devPtr - The output pointer where the mapping will
 *                      be available.
 * \param[in] flags - Application can use this flag to specify the memory access
 *                    permissions of the memory that needs to be registered with
 *                    DLA.
 * The valid values of \p flags can be one of the following:
 * - ::CUDLA_READ_WRITE_PERM, specifies that the external memory needs to be 
 *                            registered with DLA as read-write memory.
 * - ::CUDLA_READ_ONLY_PERM, specifies that the external memory needs to be 
 *                           registered with DLA as read-only memory.
 * - ::CUDLA_TASK_STATISTICS, specifies that the external memory needs to be registered 
 *                            with DLA for layerwise statistics.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorUnsupportedOperation,
 * ::cudlaErrorNvSci,
 * ::cudlaErrorInvalidAttribute,
 * ::cudlaErrorMemoryRegistered,
 * ::cudlaErrorUmd
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaImportExternalMemory(cudlaDevHandle const devHandle,
                                      const cudlaExternalMemoryHandleDesc* const desc,
                                      uint64_t** const devPtr,
                                      uint32_t const flags);

/**
 * \brief Get cuDLA's NvSciSync attributes.
 *
 * Gets the NvSciSync's attributes in the attribute list
 * created by the application.
 *
 * cuDLA supports two types of NvSciSync object primitives -
 * - Sync point
 * - Deterministic semaphore
 * cuDLA prioritizes sync point primitive over deterministic
 * semaphore primitive by default and sets these priorities in the
 * NvSciSync attribute list.
 *
 * For Deterministic semaphore, NvSciSync attribute list used to create
 * the NvSciSync object must have value of
 * NvSciSyncAttrKey_RequireDeterministicFences key set to true.
 *
 * cuDLA also supports Timestamp feature on NvSciSync objects. Waiter can
 * request for this by setting NvSciSync attribute
 * "NvSciSyncAttrKey_WaiterRequireTimestamps" as true.
 *
 * In the event of failed NvSci initialization this function would return 
 * ::cudlaErrorUnsupportedOperation. This function can return ::cudlaErrorNvSci 
 * or ::cudlaErrorInvalidAttribute in certain cases when the underlying 
 * NvSci operation fails.
 *
 * \param[out] attrList - Attribute list created by the application.
 * \param[in] flags - Applications can use this flag to specify how
 *                    they intend to use the NvSciSync object created from
 *                    the \p attrList.
 *
 * The valid values of \p flags can be one of the following (or an OR of these values):
 * - ::CUDLA_NVSCISYNC_ATTR_WAIT, specifies that the application intend to use the 
 *     NvSciSync object created using this attribute list as a waiter in cuDLA and 
 *     therefore needs cuDLA to fill waiter specific NvSciSyncAttr.
 * - ::CUDLA_NVSCISYNC_ATTR_SIGNAL, specifies that the application intend to use the 
 *     NvSciSync object created using this attribute list as a signaler in cuDLA and 
 *     therefore needs cuDLA to fill signaler specific NvSciSyncAttr.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorUnsupportedOperation,
 * ::cudlaErrorInvalidAttribute,
 * ::cudlaErrorNvSci
 */
cudlaStatus cudlaGetNvSciSyncAttributes(uint64_t* attrList,
                                        uint32_t const flags);

/**
 * \brief Imports external semaphore into cuDLA.
 *
 * Imports the allocated external semaphore by registering it with DLA.
 * After successful registration, the returned pointer can be used in a 
 * task submission to signal synchronization objects.
 *
 * On Tegra, cuDLA supports importing NvSciSync objects in standalone mode
 * only. NvSciSync object primitives that cuDLA supports are sync point and
 * deterministic semaphore.
 *
 * cuDLA also supports Timestamp feature on NvSciSync objects, using which
 * the user can get a snapshot of DLA clock at which a particular fence 
 * is signaled. At any point in time there are only 512 valid timestamp buffers 
 * that can be associated with fences.
 * For example,
 *    If User has created 513 fences from a single NvSciSync object with
 *    timestamp enabled then the timestamp buffer associated with 1st fence
 *    is same as with 513th fence.
 *
 * In the event of failed NvSci initialization (either
 * due to usage of this API in hybrid mode or an issue in the NvSci library
 * initialization), this function would return ::cudlaErrorUnsupportedOperation.
 * This function can return ::cudlaErrorNvSci or ::cudlaErrorInvalidAttribute
 * in certain cases when the underlying NvSci operation fails.
 *
 * \param[in] devHandle - A valid device handle.
 * \param[in] desc - Contains sempahore object.
 * \param[out] devPtr - The output pointer where the mapping will
 *                      be available.
 * \param[in] flags - Reserved for future. Must be set to 0.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam,
 * ::cudlaErrorInvalidDevice,
 * ::cudlaErrorUnsupportedOperation,
 * ::cudlaErrorNvSci,
 * ::cudlaErrorInvalidAttribute,
 * ::cudlaErrorMemoryRegistered
 *
 * \note This API can return task execution errors from previous DLA
 * task submissions.
 */
cudlaStatus cudlaImportExternalSemaphore(cudlaDevHandle const devHandle,
                                         const cudlaExternalSemaphoreHandleDesc* const desc,
                                         uint64_t** const devPtr,
                                         uint32_t const flags);

/**
 * \brief Set task timeout in millisecond.
 *
 * Set task timeout in ms for each device handle
 *
 * In case , device handle is invalid or timeout is 0
 * or timeout is greater than 1000 sec, this function
 * would return cudlaErrorInvalidParam otherwise cudlaSuccess
 *
 * \param[in] devHandle - A valid device handle.
 * \param[in] timeout - task timeout value in ms.
 *
 * \return
 * ::cudlaSuccess,
 * ::cudlaErrorInvalidParam
 */
cudlaStatus cudlaSetTaskTimeoutInMs(cudlaDevHandle const devHandle,
                                    uint32_t const timeout);

/** @} */ /* END CUDLA_API */
#if defined(__cplusplus)
}

#endif /* __cplusplus */
