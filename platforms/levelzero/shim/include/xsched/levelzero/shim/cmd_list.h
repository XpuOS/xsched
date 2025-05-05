#pragma once

#include <list>
#include <mutex>
#include <memory>
#include <functional>
#include <unordered_map>

#include "xsched/levelzero/hal/driver.h"

namespace xsched::levelzero
{

struct SlicedCommandList
{
    uint64_t cmd_cnt;
    const ze_context_handle_t ctx;
    const ze_device_handle_t dev;
    const ze_command_list_desc_t desc;
    std::list<ze_command_list_handle_t> cmd_lists;
};

class CommandListManager
{
public:
    static CommandListManager &Instance()
    {
        static CommandListManager slice_manager;
        return slice_manager;
    }

    std::shared_ptr<SlicedCommandList> Get(ze_command_list_handle_t cmd_list);
    ze_result_t Create(ze_context_handle_t ctx, ze_device_handle_t dev, const ze_command_list_desc_t *desc, ze_command_list_handle_t *cmd_list);
    ze_result_t Destroy(ze_command_list_handle_t cmd_list);
    ze_result_t Close(ze_command_list_handle_t cmd_list);
    ze_result_t Reset(ze_command_list_handle_t cmd_list);
    ze_result_t Append(ze_command_list_handle_t cmd_list, std::function<ze_result_t(ze_command_list_handle_t)> append_func);

private:
    CommandListManager() = default;
    ~CommandListManager() = default;
    static uint64_t GetSliceCmdCnt();

    std::mutex mtx_;
    std::unordered_map<ze_command_list_handle_t, std::shared_ptr<SlicedCommandList>> slices_;
};

} // namespace xsched::levelzero
