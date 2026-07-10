#pragma once

#include <list>
#include <mutex>
#include <memory>
#include <condition_variable>

#include "xsched/types.h"
#include "xsched/preempt/xqueue/xcommand.h"

namespace xsched::preempt
{

class CommandBuffer
{
public:
    CommandBuffer(XQueueHandle xq_h);
    ~CommandBuffer() = default;

    void WaitForNextIdle();
    XQueueState GetXQueueState();
    void ForEachCommand(std::function<bool (std::shared_ptr<XCommand>)> func);

    /// @brief Dequeue an XCommand from the CommandBuffer.
    /// @return Pointer to the dequeued XCommand.
    std::shared_ptr<XCommand> Dequeue();

    /// @brief Enqueue an XCommand to the CommandBuffer.
    /// @param xcmd Pointer to the enqueued XCommand.
    void Enqueue(std::shared_ptr<XCommand> xcmd);

    void DropAll();
    std::shared_ptr<XQueueWaitAllCommand> EnqueueXQueueWaitAllCommand();

private:
    const XQueueHandle kXQueueHandle;

    std::mutex mtx_;
    std::condition_variable cv_;

    XQueueState xq_state_ = kQueueStateIdle;
    std::shared_ptr<XCommand> last_enqueued_cmd_ = nullptr;
    std::shared_ptr<XCommand> last_dequeued_cmd_ = nullptr;
    std::list<std::shared_ptr<XCommand>> cmds_;

    uint64_t idle_cnt_ = 0;
    std::condition_variable idle_cv_;
};

} // namespace xsched::preempt
