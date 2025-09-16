#pragma once

#include <list>
#include <memory>

#include "xsched/levelzero/hal/driver.h"

namespace xsched::levelzero
{

enum ZeKernelArgType
{
    kGroupSize = 0,
    kArgumentValue = 1,
    kIndirectAccessFlags = 2,
    kCacheConfigFlags = 3,
    kGlobalOffset = 4,
};

class ZeKernelArg
{
public:
    ZeKernelArg(ZeKernelArgType type)
        : type_(type) { }
    virtual ~ZeKernelArg() = default;
    virtual void Set(ze_kernel_handle_t hKernel) = 0;

private:
    const ZeKernelArgType type_;
};

class ZeKernelGroupSize : public ZeKernelArg
{
public:
    ZeKernelGroupSize(uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ)
        : ZeKernelArg(kGroupSize), groupSizeX_(groupSizeX), groupSizeY_(groupSizeY), groupSizeZ_(groupSizeZ) { }
    ~ZeKernelGroupSize() { }
    void Set(ze_kernel_handle_t hKernel) override;

private:
    const uint32_t groupSizeX_;
    const uint32_t groupSizeY_;
    const uint32_t groupSizeZ_;
};

class ZeKernelArgumentValue : public ZeKernelArg
{
public:
    NO_COPY_CLASS(ZeKernelArgumentValue);
    NO_MOVE_CLASS(ZeKernelArgumentValue);

    ZeKernelArgumentValue(uint32_t argIndex, size_t argSize, const void *pArgValue);
    ~ZeKernelArgumentValue();
    void Set(ze_kernel_handle_t hKernel) override;

private:
    const uint32_t argIndex_;
    const size_t argSize_;
    void *pArgValue_ = nullptr;
};

class ZeKernelIndirectAccessFlags : public ZeKernelArg
{
public:
    ZeKernelIndirectAccessFlags(ze_kernel_indirect_access_flags_t flags)
        : ZeKernelArg(kIndirectAccessFlags), flags_(flags) { }
    ~ZeKernelIndirectAccessFlags() { }
    void Set(ze_kernel_handle_t hKernel) override;

private:
    const ze_kernel_indirect_access_flags_t flags_;
};

class ZeKernelCacheConfigFlags : public ZeKernelArg
{
public:
    ZeKernelCacheConfigFlags(ze_cache_config_flags_t flags)
        : ZeKernelArg(kCacheConfigFlags), flags_(flags) { }
    ~ZeKernelCacheConfigFlags() { }
    void Set(ze_kernel_handle_t hKernel) override;

private:
    const ze_cache_config_flags_t flags_;
};

class ZeKernelGlobalOffset : public ZeKernelArg
{
public:
    ZeKernelGlobalOffset(uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ)
        : ZeKernelArg(kGlobalOffset), offsetX_(offsetX), offsetY_(offsetY), offsetZ_(offsetZ) { }
    ~ZeKernelGlobalOffset() { }
    void Set(ze_kernel_handle_t hKernel) override;

private:
    const uint32_t offsetX_;
    const uint32_t offsetY_;
    const uint32_t offsetZ_;
};

class KernelArgsManager
{
public:
    static KernelArgsManager &Instance()
    {
        static KernelArgsManager args_manager;
        return args_manager;
    }

    ze_result_t AddGroupSize(ze_kernel_handle_t hKernel, uint32_t groupSizeX, uint32_t groupSizeY, uint32_t groupSizeZ);
    ze_result_t AddArgumentValue(ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize, const void *pArgValue);
    ze_result_t AddIndirectAccess(ze_kernel_handle_t hKernel, ze_kernel_indirect_access_flags_t flags);
    ze_result_t AddCacheConfig(ze_kernel_handle_t hKernel, ze_cache_config_flags_t flags);
    ze_result_t AddGlobalOffsetExp(ze_kernel_handle_t hKernel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ);
    ze_result_t Set(ze_kernel_handle_t hKernel);
    void Freeze(ze_kernel_handle_t hKernel);

private:
    KernelArgsManager() = default;
    ~KernelArgsManager() = default;

    std::unordered_map<ze_kernel_handle_t, std::shared_ptr<std::list<ZeKernelGroupSize>>> group_size_;
    std::unordered_map<ze_kernel_handle_t, std::shared_ptr<std::list<ZeKernelArgumentValue>>> argument_value_;
    std::unordered_map<ze_kernel_handle_t, std::shared_ptr<std::list<ZeKernelIndirectAccessFlags>>> indirect_access_flags_;
    std::unordered_map<ze_kernel_handle_t, std::shared_ptr<std::list<ZeKernelCacheConfigFlags>>> cache_config_flags_;
    std::unordered_map<ze_kernel_handle_t, std::shared_ptr<std::list<ZeKernelGlobalOffset>>> global_offset_;

    struct FrozenArgs
    {
        std::shared_ptr<std::list<ZeKernelGroupSize>> group_size;
        std::shared_ptr<std::list<ZeKernelArgumentValue>> argument_value;
        std::shared_ptr<std::list<ZeKernelIndirectAccessFlags>> indirect_access_flags;
        std::shared_ptr<std::list<ZeKernelCacheConfigFlags>> cache_config_flags;
        std::shared_ptr<std::list<ZeKernelGlobalOffset>> global_offset;
    };
    std::unordered_map<ze_kernel_handle_t, std::list<FrozenArgs>> frozen_args_;
};

} // namespace xsched::levelzero
