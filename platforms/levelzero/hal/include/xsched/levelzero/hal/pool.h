#pragma once

#include <map>
#include <memory>

#include "xsched/utils/pool.h"
#include "xsched/protocol/def.h"
#include "xsched/levelzero/hal/driver.h"
#include "xsched/levelzero/hal/ze_assert.h"

namespace xsched::levelzero
{

class EventPool : public xsched::utils::ObjectPool
{
public:
    EventPool(ze_context_handle_t ctx);
    virtual ~EventPool();
    static std::shared_ptr<EventPool> Instance(ze_context_handle_t ctx);

private:
    virtual void *Create() override;
    const ze_context_handle_t kContext;
    ze_event_pool_handle_t event_pool_ = nullptr;
};

class FencePool : public xsched::utils::ObjectPool
{
public:
    FencePool(ze_command_queue_handle_t cmdq)
        : ObjectPool(XSCHED_DEFAULT_COMMAND_THRESHOLD * 2), kCmdq(cmdq) {}
    virtual ~FencePool() = default;
    static std::shared_ptr<FencePool> Instance(ze_command_queue_handle_t cmdq);
    static void Destroy(ze_command_queue_handle_t cmdq);

private:
    virtual void *Create() override;
    virtual void Destroy(void *fence) override;
    const ze_command_queue_handle_t kCmdq;

    static std::mutex mtx_;
    static std::map<ze_command_queue_handle_t, std::shared_ptr<FencePool>> pools_;
};

class CommandListPool : public xsched::utils::ObjectPool
{
public:
    CommandListPool(ze_context_handle_t ctx, ze_device_handle_t dev)
        : ObjectPool(64), kContext(ctx), kDevice(dev) {}
    virtual ~CommandListPool() = default;
    static std::shared_ptr<CommandListPool> Instance(ze_command_list_handle_t cmd_list);
    static std::shared_ptr<CommandListPool> Instance(ze_context_handle_t ctx, ze_device_handle_t dev);

private:
    virtual void *Create() override;
    const ze_context_handle_t kContext;
    const ze_device_handle_t kDevice;
};

} // namespace xsched::levelzero
