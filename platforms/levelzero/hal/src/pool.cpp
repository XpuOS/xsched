#include "xsched/levelzero/hal/pool.h"

using namespace xsched::levelzero;

#define POOL_SIZE 16384

EventPool::EventPool(ze_context_handle_t ctx): kContext(ctx)
{
    ze_event_pool_desc_t event_pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext = nullptr,
        .flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
        .count = POOL_SIZE,
    };
    ZE_ASSERT(Driver::EventPoolCreate(kContext, &event_pool_desc, 0, nullptr, &event_pool_));
}

EventPool::~EventPool()
{
    ZE_ASSERT(Driver::EventPoolDestroy(event_pool_));
}

std::shared_ptr<EventPool> EventPool::Instance(ze_context_handle_t ctx)
{
    static std::mutex mtx;
    static std::map<ze_context_handle_t, std::shared_ptr<EventPool>> pools;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = pools.find(ctx);
    if (it != pools.end()) return it->second;

    auto event_pool = std::make_shared<EventPool>(ctx);
    pools[ctx] = event_pool;
    return event_pool;
}

void *EventPool::Create()
{
    static uint32_t event_count = 0;
    if(event_count >= POOL_SIZE) XERRO("event count exceeds limit (%d)", POOL_SIZE);
    ze_event_handle_t event;
    static const ze_event_desc_t event_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext = nullptr,
        .index = event_count++,
        .signal = ZE_EVENT_SCOPE_FLAG_HOST,
        .wait = ZE_EVENT_SCOPE_FLAG_HOST,
    };
    ZE_ASSERT(Driver::EventCreate(event_pool_, &event_desc, &event));
    return event;
}

std::mutex FencePool::mtx_;
std::map<ze_command_queue_handle_t, std::shared_ptr<FencePool>> FencePool::pools_;

std::shared_ptr<FencePool> FencePool::Instance(ze_command_queue_handle_t cmdq)
{
    std::unique_lock<std::mutex> lock(mtx_);
    auto it = pools_.find(cmdq);
    if (it != pools_.end()) return it->second;

    auto fence_pool = std::make_shared<FencePool>(cmdq);
    pools_[cmdq] = fence_pool;
    return fence_pool;
}

void FencePool::Destroy(ze_command_queue_handle_t cmdq)
{
    std::lock_guard<std::mutex> lock(mtx_);
    pools_.erase(cmdq);
}

void *FencePool::Create()
{
    ze_fence_handle_t fence;
    static const ze_fence_desc_t fence_desc = {
        .stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
        .pNext = nullptr,
        .flags = !ZE_FENCE_FLAG_SIGNALED,
    };
    ZE_ASSERT(Driver::FenceCreate(kCmdq, &fence_desc, &fence));
    return fence;
}

void FencePool::Destroy(void *fence)
{
    ZE_ASSERT(Driver::FenceDestroy((ze_fence_handle_t)fence));
}

std::shared_ptr<CommandListPool> CommandListPool::Instance(ze_command_list_handle_t cmd_list)
{
    ze_context_handle_t ctx;
    ze_device_handle_t dev;
    ZE_ASSERT(Driver::CommandListGetContextHandle(cmd_list, &ctx));
    ZE_ASSERT(Driver::CommandListGetDeviceHandle(cmd_list, &dev));
    return Instance(ctx, dev);
}

std::shared_ptr<CommandListPool> CommandListPool::Instance(ze_context_handle_t ctx, ze_device_handle_t dev)
{
    static std::mutex mtx;
    static std::map<ze_context_handle_t,
           std::map<ze_device_handle_t, std::shared_ptr<CommandListPool>>> pools;

    std::lock_guard<std::mutex> lock(mtx);
    auto ctx_it = pools.find(ctx);
    if (ctx_it == pools.end()) {
        ctx_it = pools.emplace(ctx, std::map<ze_device_handle_t, std::shared_ptr<CommandListPool>>()).first;
    }

    auto dev_it = ctx_it->second.find(dev);
    if (dev_it != ctx_it->second.end()) return dev_it->second;

    auto cmd_list_pool = std::make_shared<CommandListPool>(ctx, dev);
    ctx_it->second[dev] = cmd_list_pool;
    return cmd_list_pool;
}

void *CommandListPool::Create()
{
    ze_command_list_handle_t cmd_list;
    ze_command_list_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = nullptr,
        .commandQueueGroupOrdinal = 0,
        .flags = 0,
    };
    ZE_ASSERT(Driver::CommandListCreate(kContext, kDevice, &desc, &cmd_list));
    return cmd_list;
}
