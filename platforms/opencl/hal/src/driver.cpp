#include "xsched/opencl/hal/driver.h"

using namespace xsched::opencl;

#define SAVE_OPENCL_EXT_FUNC_ADDR(func)     \
    if (strcmp(func_name, #func) == 0) { \
        if (func##_ptr == nullptr) func##_ptr = reinterpret_cast<func##_fn>(ret); \
        return ret;                      \
    }

#define INIT_OPENCL_EXT_FUNC_ADDR(func) func##_fn Driver::func##_ptr = nullptr;

void *Driver::GetExtensionFunctionAddressForPlatform(cl_platform_id plat, const char *func_name)
{
    using GetExtFuncAddrPtr = void *(*)(cl_platform_id, const char *);
    static const auto get_ext_func_addr = reinterpret_cast<GetExtFuncAddrPtr>(
        GetSymbol("clGetExtensionFunctionAddressForPlatform"));
    void *ret = get_ext_func_addr(plat, func_name);
    if (ret == nullptr) return ret;

    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clFinalizeCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clRetainCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clReleaseCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandBarrierWithWaitListKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferRectKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferToImageKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandCopyImageKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandCopyImageToBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandFillBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandFillImageKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandNDRangeKernelKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetCommandBufferInfoKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandSVMMemcpyKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCommandSVMMemFillKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clRemapCommandBufferKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clUpdateMutableCommandsKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetMutableCommandInfoKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSetMemObjectDestructorAPPLE);
    SAVE_OPENCL_EXT_FUNC_ADDR(clLogMessagesToSystemLogAPPLE);
    SAVE_OPENCL_EXT_FUNC_ADDR(clLogMessagesToStdoutAPPLE);
    SAVE_OPENCL_EXT_FUNC_ADDR(clLogMessagesToStderrAPPLE);
    SAVE_OPENCL_EXT_FUNC_ADDR(clIcdGetPlatformIDsKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateProgramWithILKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clTerminateContextKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateCommandQueueWithPropertiesKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clReleaseDeviceEXT);
    SAVE_OPENCL_EXT_FUNC_ADDR(clRetainDeviceEXT);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateSubDevicesEXT);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMigrateMemObjectEXT);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetDeviceImageInfoQCOM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueAcquireGrallocObjectsIMG);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueReleaseGrallocObjectsIMG);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueGenerateMipmapIMG);
    // SAVE_OPENCL_EXT_FUNC_ADDR(clGetKernelSubGroupInfoKHR); // deprecated
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetKernelSuggestedLocalWorkSizeKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueAcquireExternalMemObjectsKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueReleaseExternalMemObjectsKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetSemaphoreHandleForTypeKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clReImportSemaphoreSyncFdKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateSemaphoreWithPropertiesKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueWaitSemaphoresKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSignalSemaphoresKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetSemaphoreInfoKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clReleaseSemaphoreKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clRetainSemaphoreKHR);
    SAVE_OPENCL_EXT_FUNC_ADDR(clImportMemoryARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSVMAllocARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSVMFreeARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMFreeARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMemcpyARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMemFillARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMapARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMUnmapARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSetKernelArgSVMPointerARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSetKernelExecInfoARM);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateAcceleratorINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetAcceleratorInfoINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clRetainAcceleratorINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clReleaseAcceleratorINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clHostMemAllocINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clDeviceMemAllocINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSharedMemAllocINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clMemFreeINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clMemBlockingFreeINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetMemAllocInfoINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSetKernelArgMemPointerINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMemFillINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMemcpyINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMemAdviseINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMigrateMemINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueMemsetINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCreateBufferWithPropertiesINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueReadHostPipeINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clEnqueueWriteHostPipeINTEL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetImageRequirementsInfoEXT);
    SAVE_OPENCL_EXT_FUNC_ADDR(clGetICDLoaderInfoOCLICD);
    SAVE_OPENCL_EXT_FUNC_ADDR(clSetContentSizeBufferPoCL);
    SAVE_OPENCL_EXT_FUNC_ADDR(clCancelCommandsIMG);

    return ret;
}

INIT_OPENCL_EXT_FUNC_ADDR(clCreateCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clFinalizeCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clRetainCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clReleaseCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandBarrierWithWaitListKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferRectKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandCopyBufferToImageKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandCopyImageKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandCopyImageToBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandFillBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandFillImageKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandNDRangeKernelKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clGetCommandBufferInfoKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandSVMMemcpyKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCommandSVMMemFillKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clRemapCommandBufferKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clUpdateMutableCommandsKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clGetMutableCommandInfoKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clSetMemObjectDestructorAPPLE);
INIT_OPENCL_EXT_FUNC_ADDR(clLogMessagesToSystemLogAPPLE);
INIT_OPENCL_EXT_FUNC_ADDR(clLogMessagesToStdoutAPPLE);
INIT_OPENCL_EXT_FUNC_ADDR(clLogMessagesToStderrAPPLE);
INIT_OPENCL_EXT_FUNC_ADDR(clIcdGetPlatformIDsKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateProgramWithILKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clTerminateContextKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateCommandQueueWithPropertiesKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clReleaseDeviceEXT);
INIT_OPENCL_EXT_FUNC_ADDR(clRetainDeviceEXT);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateSubDevicesEXT);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMigrateMemObjectEXT);
INIT_OPENCL_EXT_FUNC_ADDR(clGetDeviceImageInfoQCOM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueAcquireGrallocObjectsIMG);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueReleaseGrallocObjectsIMG);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueGenerateMipmapIMG);
// INIT_OPENCL_EXT_FUNC_ADDR(clGetKernelSubGroupInfoKHR); // deprecated
INIT_OPENCL_EXT_FUNC_ADDR(clGetKernelSuggestedLocalWorkSizeKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueAcquireExternalMemObjectsKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueReleaseExternalMemObjectsKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clGetSemaphoreHandleForTypeKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clReImportSemaphoreSyncFdKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateSemaphoreWithPropertiesKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueWaitSemaphoresKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSignalSemaphoresKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clGetSemaphoreInfoKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clReleaseSemaphoreKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clRetainSemaphoreKHR);
INIT_OPENCL_EXT_FUNC_ADDR(clImportMemoryARM);
INIT_OPENCL_EXT_FUNC_ADDR(clSVMAllocARM);
INIT_OPENCL_EXT_FUNC_ADDR(clSVMFreeARM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMFreeARM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMemcpyARM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMemFillARM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMMapARM);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueSVMUnmapARM);
INIT_OPENCL_EXT_FUNC_ADDR(clSetKernelArgSVMPointerARM);
INIT_OPENCL_EXT_FUNC_ADDR(clSetKernelExecInfoARM);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateAcceleratorINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clGetAcceleratorInfoINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clRetainAcceleratorINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clReleaseAcceleratorINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clHostMemAllocINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clDeviceMemAllocINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clSharedMemAllocINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clMemFreeINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clMemBlockingFreeINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clGetMemAllocInfoINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clSetKernelArgMemPointerINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMemFillINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMemcpyINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMemAdviseINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMigrateMemINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueMemsetINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clCreateBufferWithPropertiesINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueReadHostPipeINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clEnqueueWriteHostPipeINTEL);
INIT_OPENCL_EXT_FUNC_ADDR(clGetImageRequirementsInfoEXT);
INIT_OPENCL_EXT_FUNC_ADDR(clGetICDLoaderInfoOCLICD);
INIT_OPENCL_EXT_FUNC_ADDR(clSetContentSizeBufferPoCL);
INIT_OPENCL_EXT_FUNC_ADDR(clCancelCommandsIMG);
