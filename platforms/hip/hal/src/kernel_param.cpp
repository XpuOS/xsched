#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <unordered_map>

#include "xsched/utils/log.h"
#include "xsched/hip/hal/kernel_param.h"
#include "xsched/hip/hal/amd_comgr.h"
#include "xsched/hip/hal/driver.h"

namespace xsched::hip
{

struct FatBinStructure
{
    uint32_t magic;
    uint32_t version;
    void *image;
    void *dummy;
};

// Copy from rocclr/hipamd/src/hip_fatbin.cpp
static bool FindFileNameFromAddress(const void *image, std::string *fname_ptr, size_t *foffset_ptr)
{
    // Get the list of mapped file list
    bool ret_value = false;
    std::ifstream proc_maps;
    proc_maps.open("/proc/self/maps", std::ifstream::in);
    if (!proc_maps.is_open() || !proc_maps.good()) return ret_value;

    // For every line on the list map find out low, high address
    std::string line;
    while (std::getline(proc_maps, line)) {
        char dash;
        std::stringstream tokens(line);
        uintptr_t low_address, high_address;
        tokens >> std::hex >> low_address >> std::dec >> dash >> std::hex >> high_address >> std::dec;
        if (dash != '-') continue;

        // If address is > low_address and < high_address, then this
        // is the mapped file. Get the URI path and offset.
        uintptr_t address = reinterpret_cast<uintptr_t>(image);
        if ((address >= low_address) && (address < high_address)) {
            std::string permissions, device, uri_file_path;
            size_t offset;
            uint64_t inode;
            tokens >> permissions >> std::hex >> offset >> std::dec >> device >> inode >> uri_file_path;

            if (inode == 0 || uri_file_path.empty()) return ret_value;

            *fname_ptr = uri_file_path;
            *foffset_ptr = offset + address - low_address;
            ret_value = true;
            break;
        }
    }

    return ret_value;
}

// Copy from rocclr/hipamd/src/hip_fatbin.cpp
static bool GetFileHandle(const char *fname, int *fd_ptr, size_t *sz_ptr)
{
    if ((fd_ptr == nullptr) || (sz_ptr == nullptr)) return false;
    // open system function call, return false on fail
    struct stat stat_buf;
    *fd_ptr = open(fname, O_RDONLY);
    if (*fd_ptr < 0) return false;

    // Retrieve stat info and size
    if (fstat(*fd_ptr, &stat_buf) != 0) {
        close(*fd_ptr);
        return false;
    }

    *sz_ptr = stat_buf.st_size;
    return true;
}

#define ASSERT_COMGR_STATUS(expr) \
    do { \
        amd_comgr_status_t status = expr; \
        if (status != AMD_COMGR_STATUS_SUCCESS) { \
            XERRO("Failed to execute AMD COMGR API: %d", status); \
            return; \
        } \
    } while (0)


struct KernelParamCallbackData
{
    KernelParamInfo info;
    bool is_valid; // for value_kind == "global_buffer" or "by_value"
};

static amd_comgr_status_t kernelParamCallback(amd_comgr_metadata_node_t key,
                                              amd_comgr_metadata_node_t value,
                                              void *_user_data)
{

    KernelParamCallbackData *data = reinterpret_cast<KernelParamCallbackData *>(_user_data);

    // Get key name
    size_t string_length;
    CodeObjectManager::get_metadata_string(key, &string_length, nullptr);
    char* key_name = new char[string_length];
    CodeObjectManager::get_metadata_string(key, &string_length, key_name);


    // Get value kind
    CodeObjectManager::get_metadata_string(value, &string_length, nullptr);
    char* value_string = new char[string_length];
    CodeObjectManager::get_metadata_string(value, &string_length, value_string);

    if (strcmp(key_name, ".size") == 0) {
        data->info.size = std::stoul(value_string);
    } else if (strcmp(key_name, ".offset") == 0) {
        data->info.offset = std::stoul(value_string);
    } else if (strcmp(key_name, ".value_kind") == 0) {
        if (strcmp(value_string, "global_buffer") == 0 || strcmp(value_string, "by_value") == 0) {
            data->is_valid = true;
        } else {
            data->is_valid = false;
        }
    }
    // XDEBG("key: %s, value: %s", key_name, value_string);

    delete[] key_name;
    delete[] value_string;
    return AMD_COMGR_STATUS_SUCCESS;
}


static void registerForOneISA(amd_comgr_data_t code_object, kernel_names_params_map& kernel_names_params)
{
    // Get the metadata node for kernels
    amd_comgr_metadata_node_t metadata;
    ASSERT_COMGR_STATUS(CodeObjectManager::get_data_metadata(code_object, &metadata));

    amd_comgr_metadata_node_t kernels_metadata;
    ASSERT_COMGR_STATUS(CodeObjectManager::metadata_lookup(metadata, "amdhsa.kernels", &kernels_metadata));

    // Iterate through the kernels and get the parameters.
    // The ultimate goal is to get the kernel name and the parameters.

    size_t num_kernels;
    ASSERT_COMGR_STATUS(CodeObjectManager::get_metadata_list_size(kernels_metadata, &num_kernels));

    for (size_t i = 0; i < num_kernels; i++) {
        amd_comgr_metadata_node_t kernel_metadata;
        ASSERT_COMGR_STATUS(CodeObjectManager::index_list_metadata(kernels_metadata, i, &kernel_metadata));

        // Get the kernel name
        amd_comgr_metadata_node_t kernel_name_node;
        
        size_t kernel_name_size;
        ASSERT_COMGR_STATUS(CodeObjectManager::metadata_lookup(kernel_metadata, ".name", &kernel_name_node));
        ASSERT_COMGR_STATUS(CodeObjectManager::get_metadata_string(kernel_name_node, &kernel_name_size, nullptr));
        char* kernel_name = new char[kernel_name_size];
        ASSERT_COMGR_STATUS(CodeObjectManager::get_metadata_string(kernel_name_node, &kernel_name_size, kernel_name));

        kernel_names_params[kernel_name] = std::vector<KernelParamInfo>();

        amd_comgr_metadata_node_t kernel_args_node;
        // TODO: what if the kernel has no arguments?
        ASSERT_COMGR_STATUS(CodeObjectManager::metadata_lookup(kernel_metadata, ".args", &kernel_args_node));

        size_t num_args;
        ASSERT_COMGR_STATUS(CodeObjectManager::get_metadata_list_size(kernel_args_node, &num_args));
        // XDEBG("kernel %s has %zu arguments", kernel_name, num_args);
        for (size_t j = 0; j < num_args; j++) {
            amd_comgr_metadata_node_t arg_node;
            ASSERT_COMGR_STATUS(CodeObjectManager::index_list_metadata(kernel_args_node, j, &arg_node));

            // skip if the argument is not a map
            amd_comgr_metadata_kind_t kind;
            ASSERT_COMGR_STATUS(CodeObjectManager::get_metadata_kind(arg_node, &kind));
            if (kind != AMD_COMGR_METADATA_KIND_MAP) continue;

            // iterate through the map
            KernelParamCallbackData data;
            data.is_valid = false;
            ASSERT_COMGR_STATUS(CodeObjectManager::iterate_map_metadata(arg_node, kernelParamCallback, &data));

            if (data.is_valid) {
                // XDEBG("kernel %s, param %zu, size %zu, offset %zu", kernel_name, j, data.info.size, data.info.offset);
                kernel_names_params[std::string(kernel_name)].push_back(data.info);
            }

            ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(arg_node));
        }

        delete[] kernel_name;
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(kernel_args_node));
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(kernel_name_node));
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(kernel_metadata));
    }

    ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(kernels_metadata));
    ASSERT_COMGR_STATUS(CodeObjectManager::destroy_metadata(metadata));
}

static void registerForDataObject(amd_comgr_data_t data_object, const void* image, kernel_names_params_map& kernel_names_params)
{
    // Get a list of ISAs
    size_t isa_count;
    ASSERT_COMGR_STATUS(CodeObjectManager::get_isa_count(&isa_count));
    if (isa_count == 0) XERRO("No ISAs found");

    amd_comgr_code_object_info_t *info_list = new amd_comgr_code_object_info_t[isa_count];
    for (size_t i = 0; i < isa_count; i++) {
        ASSERT_COMGR_STATUS(CodeObjectManager::get_isa_name(i, &info_list[i].isa));
    }
    ASSERT_COMGR_STATUS(CodeObjectManager::lookup_code_object(data_object, info_list, isa_count));

    // 1.4 Process each ISA
    for (size_t i = 0; i < isa_count; i++) {
        // XDEBG("RegisterForDataObject: ISA %s, size = %zu, offset = %zu", info_list[i].isa, info_list[i].size, info_list[i].offset);
        if (info_list[i].size == 0) continue;
        amd_comgr_data_t code_object;
        ASSERT_COMGR_STATUS(CodeObjectManager::create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &code_object));
        ASSERT_COMGR_STATUS(CodeObjectManager::set_data(code_object, info_list[i].size, ((const char*)image+info_list[i].offset)));
        registerForOneISA(code_object, kernel_names_params);
        ASSERT_COMGR_STATUS(CodeObjectManager::release_data(code_object));
    }    
    
    delete[] info_list;
    return;
}

void KernelParamManager::RegisterStaticCodeObject(const void* data)
{
    //  Parse the binary and get metadata object
    const FatBinStructure *fatbin = reinterpret_cast<const FatBinStructure *>(data);
    const void *image = fatbin->image;

    // Now, we need to find where the data is from
    // so we can call amd_comgr_set_data_from_file_slice() later
    std::string fname;
    size_t foffset;
    if (!FindFileNameFromAddress(image, &fname, &foffset)) XERRO("Failed to find file name from address");

    int fd;
    size_t fsize;
    if (!GetFileHandle(fname.c_str(), &fd, &fsize)) XERRO("Failed to get file handle and size");

    // Create the data object and set the file slice
    amd_comgr_data_t data_object;
    ASSERT_COMGR_STATUS(CodeObjectManager::create_data(AMD_COMGR_DATA_KIND_FATBIN, &data_object));
    ASSERT_COMGR_STATUS(CodeObjectManager::set_data_from_file_slice(data_object, fd, foffset, fsize));
    registerForDataObject(data_object, image, this->static_kernel_names_params_);
    ASSERT_COMGR_STATUS(CodeObjectManager::release_data(data_object));
}

void KernelParamManager::RegisterStaticFunction(const void* func, const char* name)
{
    static_kernel_ptrs_params_[func] = static_kernel_names_params_[std::string(name)];
    static_kernel_ptrs_names_[func] = std::string(name);
}

void KernelParamManager::RegisterDynamicCodeObject(const char* file_path, hipModule_t mod)
{
    // Similar to RegisterStaticCodeObject but read from file_path directly
    // Store results in module_kernel_params_[mod]
    int fd;
    size_t fsize;
    if (!GetFileHandle(file_path, &fd, &fsize)) XERRO("Failed to get file handle for dynamic code object");

    // const char* file_content = nullptr; 
    // mmap the fd to the file_content
    char* file_content = new char[fsize];
    std::ifstream file(file_path, std::ios::binary);
    file.read(file_content, fsize);
    file.close();

    amd_comgr_data_t data_object;
    ASSERT_COMGR_STATUS(CodeObjectManager::create_data(AMD_COMGR_DATA_KIND_FATBIN, &data_object));
    ASSERT_COMGR_STATUS(CodeObjectManager::set_data_from_file_slice(data_object, fd, 0, fsize));

    // XDEBG("RegisterDynamicCodeObject for %s, fsize = %zu, fd = %d", file_path, fsize, fd);

    this->module_kernel_params_[mod] = kernel_names_params_map();
    registerForDataObject(data_object, file_content, this->module_kernel_params_[mod]);
    for (auto iter : this->module_kernel_params_[mod]) {
        // XDEBG("Dynamic module %p has kernel %s with %zu parameters", mod, iter.first.c_str(), iter.second.size());
    }
    ASSERT_COMGR_STATUS(CodeObjectManager::release_data(data_object));
    delete[] file_content;
}

void KernelParamManager::RegisterDynamicFunction(hipModule_t mod, hipFunction_t func, const char* func_name)
{
    // Register the function pointer and the function name
    auto it = module_kernel_params_.find(mod);
    if (it != module_kernel_params_.end()) {
        auto& kernel_map = it->second;
        auto kernel_it = kernel_map.find(func_name);
        if (kernel_it != kernel_map.end()) {
            dynamic_kernel_ptrs_params_[func] = kernel_it->second;
            dynamic_kernel_ptrs_names_[func] = func_name;
        } else {
            XERRO("Dynamic function %p with name %s not found in module %p", func, func_name, mod);
        }
    } else {
        XERRO("Dynamic module %p not found", mod);
    }
}

void KernelParamManager::GetStaticKernelParams(const void* func, uint32_t* numParameters, uint32_t* allParamsSize)
{
    auto static_it = static_kernel_ptrs_params_.find(func);
    if (static_it == static_kernel_ptrs_params_.end()) return;
    *numParameters = static_it->second.size();
    uint32_t maxOffset = 0;
    for (const auto& param : static_it->second) {
        if (param.offset + param.size > maxOffset) maxOffset = param.offset + param.size;
    }
    *allParamsSize = maxOffset;
}

constexpr size_t func_kernel_offset = 136;
constexpr size_t kernel_parameters_offset = 72;
constexpr size_t parameters_signature_offset = 0;
constexpr size_t signature_numParameters_offset = 56;
constexpr size_t signature_allParamsSize_offset = 60;
constexpr size_t signature_params_offset = 0;

constexpr size_t signatureParam_size_offset = 16;
constexpr size_t signatureParam_offset_offset = 8;

constexpr size_t signatureParam_size = 120;

template<typename T>
T* offset_ptr(void* base, size_t offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(base) + offset);
}

static void HipFunctionGetParamSize(hipFunction_t func, size_t index, size_t* offset, size_t* size)
{
    // Navigate through: func->kernel->parameters_->signature_->params_[index].size_
    void* kernel = *offset_ptr<void*>(func, func_kernel_offset);
    void* parameters = *offset_ptr<void*>(kernel, kernel_parameters_offset);
    void* signature = *offset_ptr<void*>(parameters, parameters_signature_offset);
    std::vector<char[signatureParam_size]>* params = offset_ptr<std::vector<char[signatureParam_size]>>(signature, signature_params_offset);
    void* data_entry = params->data()[index];
    *offset = *offset_ptr<size_t>(data_entry, signatureParam_offset_offset);
    *size = *offset_ptr<size_t>(data_entry, signatureParam_size_offset);
}

static void HipFunctionGetNumParameters(hipFunction_t func, uint32_t* numParameters, uint32_t* allParamsSize)
{
    // Navigate through: func->kernel->parameters_->signature_->numParameters_
    void* kernel = *offset_ptr<void*>(func, func_kernel_offset);
    void* parameters = *offset_ptr<void*>(kernel, kernel_parameters_offset);
    void* signature = *offset_ptr<void*>(parameters, parameters_signature_offset);
    *numParameters = *offset_ptr<uint32_t>(signature, signature_numParameters_offset);
    // *allParamsSize = *offset_ptr<uint32_t>(signature, signature_allParamsSize_offset);

    size_t lastSize = 0;
    size_t lastOffset = 0;
    HipFunctionGetParamSize(func, *numParameters - 1, &lastOffset, &lastSize);
    *allParamsSize = lastOffset + lastSize;
}


void KernelParamManager::GetDynamicKernelParams(hipFunction_t func, uint32_t* numParameters, uint32_t* allParamsSize)
{
    return HipFunctionGetNumParameters(func, numParameters, allParamsSize);
}

const char* KernelParamManager::GetStaticFunctionName(const void* func)
{
    auto it = static_kernel_ptrs_names_.find(func);
    if (it == static_kernel_ptrs_names_.end()) return nullptr;
    return it->second.c_str();
}

const char* KernelParamManager::GetDynamicFunctionName(hipFunction_t func)
{
    auto it = dynamic_kernel_ptrs_names_.find(func);
    if (it == dynamic_kernel_ptrs_names_.end()) return nullptr;
    return it->second.c_str();
}

void KernelParamManager::GetStaticKernelParamInfo(const void* func, uint32_t index, size_t* offset, size_t* size)
{
    auto it = static_kernel_ptrs_params_.find(func);
    if (it == static_kernel_ptrs_params_.end()) return;
    *offset = it->second[index].offset;
    *size = it->second[index].size;
}

void KernelParamManager::GetDynamicKernelParamInfo(hipFunction_t func, uint32_t index, size_t* offset, size_t* size)
{
    return HipFunctionGetParamSize(func, index, offset, size);
}

KernelParamManager* KernelParamManager::Instance()
{
    static KernelParamManager *instance = new KernelParamManager();
    return instance;
}

} // namespace xsched::hal::hip
