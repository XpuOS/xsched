#include "xsched/hip/hal/amd_comgr.h"
#include "xsched/hip/hal/driver.h"
#include "xsched/hip/hal/hip_assert.h"
#include "xsched/utils/log.h"
#include "xsched/hip/hal/kernel_param.h"

#define ASSERT_COMGR_STATUS(expr) \
    do { \
        amd_comgr_status_t status = expr; \
        if (status != AMD_COMGR_STATUS_SUCCESS) { \
            XERRO("Failed to execute AMD COMGR API: %d", status); \
            return; \
        } \
    } while (0)

namespace xsched::hip {

/***************partially copied from clr/hipamd/src/hip_code_object.cpp*****************/
// In uncompressed mode
static constexpr char kOffloadBundleUncompressedMagicStr[] = "__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t kOffloadBundleUncompressedMagicStrSize =
    sizeof(kOffloadBundleUncompressedMagicStr);

// In compressed mode
static constexpr char kOffloadBundleCompressedMagicStr[] = "CCOB";
static constexpr size_t kOffloadBundleCompressedMagicStrSize =
    sizeof(kOffloadBundleCompressedMagicStr);
// static constexpr char kOffloadKindHip[] = "hip";
// static constexpr char kOffloadKindHipv4[] = "hipv4";
// static constexpr char kOffloadKindHcc[] = "hcc";
static constexpr char kAmdgcnTargetTriple[] = "amdgcn-amd-amdhsa-";
static constexpr char kHipFatBinName[] = "hipfatbin";
// static constexpr char kHipFatBinName_[] = "hipfatbin-";
static constexpr char kOffloadKindHipv4_[] = "hipv4-";  // bundled code objects need the prefix
// static constexpr char kOffloadHipV4FatBinName_[] = "hipfatbin-hipv4-";

// Clang Offload bundler description & Header in uncompressed mode.
struct __ClangOffloadBundleInfo {
  uint64_t offset;
  uint64_t size;
  uint64_t bundleEntryIdSize;
  const char bundleEntryId[1];
};

struct __ClangOffloadBundleUncompressedHeader {
  const char magic[kOffloadBundleUncompressedMagicStrSize - 1];
  uint64_t numOfCodeObjects;
  __ClangOffloadBundleInfo desc[1];
};

// Clang Offload bundler description & Header in compressed mode.
struct __ClangOffloadBundleCompressedHeader {
  const char magic[kOffloadBundleCompressedMagicStrSize - 1];
  uint16_t versionNumber;
  uint16_t compressionMethod;
  uint32_t totalSize;
  uint32_t uncompressedBinarySize;
  uint64_t Hash;
  const char compressedBinarydesc[1];
};

static bool IsClangOffloadMagicBundle(const void* data, bool& isCompressed) {
  std::string magic(reinterpret_cast<const char*>(data),
                    kOffloadBundleUncompressedMagicStrSize - 1);
  if (!magic.compare(kOffloadBundleUncompressedMagicStr)) {
    isCompressed = false;
    return true;
  }
  std::string magic1(reinterpret_cast<const char*>(data),
                    kOffloadBundleCompressedMagicStrSize - 1);
  if (!magic1.compare(kOffloadBundleCompressedMagicStr)) {
    isCompressed = true;
    return true;
  }
  return false;
}

static size_t getFatbinSize(const void* data, const bool isCompressed) {
  if (isCompressed) {
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleCompressedHeader*>(data);
    return obheader->totalSize;
  } else {
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
    const __ClangOffloadBundleInfo* desc = &obheader->desc[0];
    uint64_t i = 0;
    while (++i < obheader->numOfCodeObjects) {
      desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
          reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) + desc->bundleEntryIdSize);
    }
    return desc->offset + desc->size;
  }
}
/***************partially copied from clr/hipamd/src/hip_code_object.cpp*****************/
void registerForOneISA(amd_comgr_data_t code_object, kernel_names_params_map& kernel_names_params);

void registerForDataObjectV2_8(amd_comgr_data_t data_object, const void* image, kernel_names_params_map& kernel_names_params)
{
    /*****code logic is based on function ExtractFatBinaryUsingCOMGR in clr/hipamd/src/hip_fatbin.cpp*****/
    (void) data_object; // used to surpress warning
    // construct BundleEntryIDs
    int deviceID;
    HIP_ASSERT(Driver::GetDevice(&deviceID));
    hipDeviceProp_t deviceProp;
    HIP_ASSERT(Driver::GetDeviceProperties(&deviceProp, deviceID));
    std::string BundleEntryID = kOffloadKindHipv4_;
    BundleEntryID += kAmdgcnTargetTriple;
    BundleEntryID += deviceProp.gcnArchName;
    std::vector<const char *> BundleEntryIDs{BundleEntryID.c_str()};

    // unbundle image
    bool isCompressed = false;
    size_t size;
    if(!IsClangOffloadMagicBundle(image, isCompressed)) {
        XDEBG("image is not a ClangOffloadBundle");
        return;
    }
    size = getFatbinSize(image, isCompressed);
    amd_comgr_data_t dataCodeObj{0};
    amd_comgr_data_set_t dataSetBundled{0};
    amd_comgr_data_set_t dataSetUnbundled{0};
    amd_comgr_action_info_t actionInfoUnbundle{0};
    amd_comgr_data_t item{0};
    // amd_comgr_status_t comgrStatus = AMD_COMGR_STATUS_SUCCESS;
    XINFO("Magic %s",isCompressed? kOffloadBundleCompressedMagicStr: kOffloadBundleUncompressedMagicStr);
    // Create Bundled dataset
    ASSERT_COMGR_STATUS(CodeObjectManager::create_data_set(&dataSetBundled));
    // CodeObject
    ASSERT_COMGR_STATUS(CodeObjectManager::create_data(AMD_COMGR_DATA_KIND_OBJ_BUNDLE, &dataCodeObj));
    ASSERT_COMGR_STATUS(CodeObjectManager::set_data(dataCodeObj, size, static_cast<const char*>(image)));
    ASSERT_COMGR_STATUS(CodeObjectManager::set_data_name(dataCodeObj, kHipFatBinName)); // this step might be skipped
    ASSERT_COMGR_STATUS(CodeObjectManager::data_set_add(dataSetBundled, dataCodeObj));
    // Set up ActionInfo
    ASSERT_COMGR_STATUS(CodeObjectManager::create_action_info(&actionInfoUnbundle));
    ASSERT_COMGR_STATUS(CodeObjectManager::action_info_set_language(actionInfoUnbundle, AMD_COMGR_LANGUAGE_HIP));
    ASSERT_COMGR_STATUS(CodeObjectManager::action_info_set_bundle_entry_ids(actionInfoUnbundle, BundleEntryIDs.data(), BundleEntryIDs.size()));
    // Unbundle
    ASSERT_COMGR_STATUS(CodeObjectManager::create_data_set(&dataSetUnbundled));
    ASSERT_COMGR_STATUS(CodeObjectManager::do_action(AMD_COMGR_ACTION_UNBUNDLE, actionInfoUnbundle, dataSetBundled, dataSetUnbundled));
    // register code object
    size_t count = 0;
    ASSERT_COMGR_STATUS(CodeObjectManager::action_data_count(dataSetUnbundled, AMD_COMGR_DATA_KIND_EXECUTABLE, &count));
    // XDEBG("CO count after unbundle %lu",count);
    for(size_t i = 0; i < count; i++) {
        ASSERT_COMGR_STATUS(CodeObjectManager::action_data_get_data(dataSetUnbundled, AMD_COMGR_DATA_KIND_EXECUTABLE, i, &item));
        size_t itemSize;
        CodeObjectManager::get_data(item, &itemSize, nullptr);
        if(itemSize == 0) {
            continue;
        }
        registerForOneISA(item, kernel_names_params);
    }
    // Cleanup
    if(actionInfoUnbundle.handle) {
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_action_info(actionInfoUnbundle));
    }
    if(dataSetBundled.handle) {
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_data_set(dataSetBundled));
    }
    if(dataSetUnbundled.handle) {
        ASSERT_COMGR_STATUS(CodeObjectManager::destroy_data_set(dataSetUnbundled));
    }
    if(dataCodeObj.handle) {
        ASSERT_COMGR_STATUS(CodeObjectManager::release_data(dataCodeObj));
    }
    if(item.handle) {
        ASSERT_COMGR_STATUS(CodeObjectManager::release_data(item));
    }
    return;
}

}