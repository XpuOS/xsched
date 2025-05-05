#include "xsched/protocol/def.h"
#include "xsched/levelzero/hal/pool.h"
#include "xsched/levelzero/shim/cmd_list.h"

using namespace xsched::levelzero;

std::shared_ptr<SlicedCommandList> CommandListManager::Get(ze_command_list_handle_t cmd_list)
{
    if (GetSliceCmdCnt() == 0) return nullptr;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(cmd_list);
    if (it == slices_.end()) return nullptr;
    return it->second;
}

ze_result_t CommandListManager::Create(ze_context_handle_t ctx, ze_device_handle_t dev,
                                       const ze_command_list_desc_t *desc,
                                       ze_command_list_handle_t *cmd_list)
{
    ze_result_t res = Driver::CommandListCreate(ctx, dev, desc, cmd_list);
    if (res != ZE_RESULT_SUCCESS) return res;
    if (GetSliceCmdCnt() == 0) return res;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(*cmd_list);
    XASSERT(it == slices_.end(), "slice for cmd_list %p already exists", *cmd_list);
    slices_[*cmd_list] = std::make_shared<SlicedCommandList>(SlicedCommandList{
        .cmd_cnt = 0, .ctx = ctx, .dev = dev, .desc = *desc, .cmd_lists = {}
    });
    return res;
}

ze_result_t CommandListManager::Destroy(ze_command_list_handle_t cmd_list)
{
    ze_result_t res = Driver::CommandListDestroy(cmd_list);
    if (res != ZE_RESULT_SUCCESS) return res;
    if (GetSliceCmdCnt() == 0) return res;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(cmd_list);
    XASSERT(it != slices_.end(), "slice for cmd_list %p not found", cmd_list);
    for (auto cl : it->second->cmd_lists) {
        ZE_ASSERT(Driver::CommandListReset(cl));
        CommandListPool::Instance(it->second->ctx, it->second->dev)->Push(cl);
    }
    slices_.erase(it);
    return res;
}

ze_result_t CommandListManager::Close(ze_command_list_handle_t cmd_list)
{
    ze_result_t res = Driver::CommandListClose(cmd_list);
    if (res != ZE_RESULT_SUCCESS) return res;
    if (GetSliceCmdCnt() == 0) return res;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(cmd_list);
    XASSERT(it != slices_.end(), "slice for cmd_list %p not found", cmd_list);
    for (auto cl : it->second->cmd_lists) {
        res = Driver::CommandListClose(cl);
        if (res != ZE_RESULT_SUCCESS) return res;
    }
    return res;
}

ze_result_t CommandListManager::Reset(ze_command_list_handle_t cmd_list)
{
    ze_result_t res = Driver::CommandListReset(cmd_list);
    if (res != ZE_RESULT_SUCCESS) return res;
    if (GetSliceCmdCnt() == 0) return res;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(cmd_list);
    XASSERT(it != slices_.end(), "slice for cmd_list %p not found", cmd_list);
    for (auto cl : it->second->cmd_lists) {
        res = Driver::CommandListReset(cl);
        if (res != ZE_RESULT_SUCCESS) return res;
        CommandListPool::Instance(cl)->Push(cl);
    }
    it->second->cmd_lists.clear();
    it->second->cmd_cnt = 0;
    return res;
}

ze_result_t CommandListManager::Append(ze_command_list_handle_t cmd_list, std::function<ze_result_t(ze_command_list_handle_t)> append_func)
{
    if (GetSliceCmdCnt() == 0) return append_func(cmd_list);

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = slices_.find(cmd_list);
    XASSERT(it != slices_.end(), "slice for cmd_list %p not found", cmd_list);
    if (it->second->cmd_cnt++ % GetSliceCmdCnt() == 0) {
        auto new_cmd_list = (ze_command_list_handle_t)CommandListPool::Instance(it->second->ctx, it->second->dev)->Pop();
        it->second->cmd_lists.push_back(new_cmd_list);
    }
    return append_func(it->second->cmd_lists.back());
}

uint64_t CommandListManager::GetSliceCmdCnt()
{
    static uint64_t slice_cmd_cnt = []()->uint64_t {
        uint64_t val = 0;
        char *str = std::getenv(XSCHED_LEVELZERO_SLICE_CNT_ENV_NAME);
        if (str == nullptr) return 0;
        try { val = std::stoll(str); } catch (...) { return 0; }
        return val;
    }();
    return slice_cmd_cnt;
}
