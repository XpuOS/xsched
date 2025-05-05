#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "xsched/hip/hal/hip.h"

namespace xsched::hip {

struct KernelParamInfo {
    size_t size;
    size_t offset;
};
using kernel_names_params_map = std::unordered_map<std::string, std::vector<KernelParamInfo>>;


class KernelParamManager {
public:
    static KernelParamManager* Instance();

    // For static code objects
    void RegisterStaticCodeObject(const void* data);
    void RegisterStaticFunction(const void* func, const char* name);

    // For dynamic code objects
    void RegisterDynamicCodeObject(const char* file_path, hipModule_t mod);
    void RegisterDynamicFunction(hipModule_t mod, hipFunction_t func, const char* func_name);

    // Query functions
    void GetStaticKernelParams(const void* func, uint32_t* numParameters, uint32_t* allParamsSize);
    void GetDynamicKernelParams(hipFunction_t func, uint32_t* numParameters, uint32_t* allParamsSize);
    
    void GetStaticKernelParamInfo(const void* func, uint32_t index, size_t* offset, size_t* size);
    void GetDynamicKernelParamInfo(hipFunction_t func, uint32_t index, size_t* offset, size_t* size);
    
    const char* GetStaticFunctionName(const void* func);
    const char* GetDynamicFunctionName(hipFunction_t func);

private:
    KernelParamManager() = default;
    ~KernelParamManager() = default;


    // For static code objects
    kernel_names_params_map static_kernel_names_params_;
    std::unordered_map<const void*, std::vector<KernelParamInfo>> static_kernel_ptrs_params_;
    std::unordered_map<const void*, std::string> static_kernel_ptrs_names_;

    // For dynamic code objects
    std::unordered_map<hipModule_t, kernel_names_params_map> module_kernel_params_;
    std::unordered_map<hipFunction_t, std::vector<KernelParamInfo>> dynamic_kernel_ptrs_params_;
    std::unordered_map<hipFunction_t, std::string> dynamic_kernel_ptrs_names_;
};

} // namespace xsched::hal::hip 