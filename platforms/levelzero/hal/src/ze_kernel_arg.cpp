#include "xsched/levelzero/hal/ze_kernel_arg.h"
#include "xsched/levelzero/hal/ze_assert.h"

using namespace xsched::levelzero;

void ZeKernelGroupSize::Set(ze_kernel_handle_t hKernel)
{
    ZE_ASSERT(Driver::KernelSetGroupSize(hKernel, groupSizeX_, groupSizeY_, groupSizeZ_));
}

ZeKernelArgumentValue::ZeKernelArgumentValue(uint32_t argIndex, size_t argSize, const void *pArgValue)
    : ZeKernelArg(kArgumentValue), argIndex_(argIndex), argSize_(argSize)
{
    if (pArgValue != nullptr) {
        pArgValue_ = malloc(argSize);
        std::memcpy(pArgValue_, pArgValue, argSize);
    }
}

ZeKernelArgumentValue::~ZeKernelArgumentValue()
{
    if (pArgValue_) free(pArgValue_);
}

void ZeKernelArgumentValue::Set(ze_kernel_handle_t hKernel)
{
    ZE_ASSERT(Driver::KernelSetArgumentValue(hKernel, argIndex_, argSize_, pArgValue_));
}

void ZeKernelIndirectAccessFlags::Set(ze_kernel_handle_t hKernel)
{
    ZE_ASSERT(Driver::KernelSetIndirectAccess(hKernel, flags_));
}

void ZeKernelCacheConfigFlags::Set(ze_kernel_handle_t hKernel)
{
    ZE_ASSERT(Driver::KernelSetCacheConfig(hKernel, flags_));
}

void ZeKernelGlobalOffset::Set(ze_kernel_handle_t hKernel)
{
    ZE_ASSERT(Driver::KernelSetGlobalOffsetExp(hKernel, offsetX_, offsetY_, offsetZ_));
}

ze_result_t KernelArgsManager::AddGroupSize(ze_kernel_handle_t hKernel, uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ)
{
    if (group_size_.find(hKernel) == group_size_.end()) {
        group_size_[hKernel] = std::make_shared<std::list<ZeKernelGroupSize>>();
    }
    group_size_[hKernel]->emplace_back(groupSizeX, groupSizeY, groupSizeZ);
    XDEBG("KernelSetGroupSize(kernel: %p) deferred", hKernel);
    return ZE_RESULT_SUCCESS;
}

ze_result_t KernelArgsManager::AddArgumentValue(ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize, const void *pArgValue)
{
    if (argument_value_.find(hKernel) == argument_value_.end()) {
        argument_value_[hKernel] = std::make_shared<std::list<ZeKernelArgumentValue>>();
    }
    argument_value_[hKernel]->emplace_back(argIndex, argSize, pArgValue);
    XDEBG("KernelSetArgumentValue(kernel: %p) deferred", hKernel);
    return ZE_RESULT_SUCCESS;
}

ze_result_t KernelArgsManager::AddIndirectAccess(ze_kernel_handle_t hKernel, ze_kernel_indirect_access_flags_t flags)
{
    if (indirect_access_flags_.find(hKernel) == indirect_access_flags_.end()) {
        indirect_access_flags_[hKernel] = std::make_shared<std::list<ZeKernelIndirectAccessFlags>>();
    }
    indirect_access_flags_[hKernel]->emplace_back(flags);
    XDEBG("KernelSetIndirectAccess(kernel: %p) deferred", hKernel);
    return ZE_RESULT_SUCCESS;
}

ze_result_t KernelArgsManager::AddCacheConfig(ze_kernel_handle_t hKernel, ze_cache_config_flags_t flags)
{
    if (cache_config_flags_.find(hKernel) == cache_config_flags_.end()) {
        cache_config_flags_[hKernel] = std::make_shared<std::list<ZeKernelCacheConfigFlags>>();
    }
    cache_config_flags_[hKernel]->emplace_back(flags);
    XDEBG("KernelSetCacheConfig(kernel: %p) deferred", hKernel);
    return ZE_RESULT_SUCCESS;
}

ze_result_t KernelArgsManager::AddGlobalOffsetExp(ze_kernel_handle_t hKernel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ)
{
    if (global_offset_.find(hKernel) == global_offset_.end()) {
        global_offset_[hKernel] = std::make_shared<std::list<ZeKernelGlobalOffset>>();
    }
    global_offset_[hKernel]->emplace_back(offsetX, offsetY, offsetZ);
    XDEBG("KernelSetGlobalOffsetExp(kernel: %p) deferred", hKernel);
    return ZE_RESULT_SUCCESS;
}

ze_result_t KernelArgsManager::Set(ze_kernel_handle_t hKernel)
{
    if (frozen_args_.find(hKernel) != frozen_args_.end()) {
        auto &args = frozen_args_[hKernel].front();
        if (args.group_size)
            for (auto &arg : *(args.group_size)) arg.Set(hKernel);
        if (args.argument_value)
            for (auto &arg : *(args.argument_value)) arg.Set(hKernel);
        if (args.indirect_access_flags)
            for (auto &arg : *(args.indirect_access_flags)) arg.Set(hKernel);
        if (args.cache_config_flags)
            for (auto &arg : *(args.cache_config_flags)) arg.Set(hKernel);
        if (args.global_offset)
            for (auto &arg : *(args.global_offset)) arg.Set(hKernel);
        frozen_args_[hKernel].pop_front();
        if (frozen_args_[hKernel].empty()) frozen_args_.erase(hKernel);
    }

    XDEBG("Set kernel %p arguments", hKernel);
    return ZE_RESULT_SUCCESS;
}

void KernelArgsManager::Freeze(ze_kernel_handle_t hKernel)
{
    FrozenArgs args;
    if (group_size_.find(hKernel) != group_size_.end()) {
        args.group_size = group_size_[hKernel];
        group_size_.erase(hKernel);
    }
    if (argument_value_.find(hKernel) != argument_value_.end()) {
        args.argument_value = argument_value_[hKernel];
        argument_value_.erase(hKernel);
    }
    if (indirect_access_flags_.find(hKernel) != indirect_access_flags_.end()) {
        args.indirect_access_flags = indirect_access_flags_[hKernel];
        indirect_access_flags_.erase(hKernel);
    }
    if (cache_config_flags_.find(hKernel) != cache_config_flags_.end()) {
        args.cache_config_flags = cache_config_flags_[hKernel];
        cache_config_flags_.erase(hKernel);
    }
    if (global_offset_.find(hKernel) != global_offset_.end()) {
        args.global_offset = global_offset_[hKernel];
        global_offset_.erase(hKernel);
    }
    frozen_args_[hKernel].emplace_back(std::move(args));

    XDEBG("Freeze kernel %p arguments", hKernel);
}
