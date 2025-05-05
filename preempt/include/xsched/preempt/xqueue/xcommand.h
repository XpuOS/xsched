#pragma once

#include <list>
#include <mutex>
#include <cstdint>
#include <functional>
#include <condition_variable>

#include "xsched/types.h"

namespace xsched::preempt
{

enum XCommandType
{
    kCommandTypeHardware         = 0,
    kCommandTypeHostFunction     = 1,
    kCommandTypeXQueueWaitAll    = 2,
    kCommandTypeBatchSynchronize = 3,
    kCommandTypeXQueueDestroy    = 4,
};

typedef int64_t XCommandProperties;

enum XCommandProperty
{
    /// @brief A none (normal) command is non-deactivatable and non-idempotent. XSched will wait
    /// until all previous deactivatable commands to complete before launching this command. If did
    /// not wait, when preemption happened, all previous commands will be deactivated, and this
    /// will not, then this command will write system memory, leading to an inconsist memory state.
    kCommandPropertyNone           = 0x0000,

    /// @brief A deactivatable command is deactivated in HwQueue->Deactivate(),
    /// i.e., will not be selected to execute by the XPU.
    /// The XAL should guarantee that the deactivation and the re-execution
    /// of the deactivated commands will not cause side effects or errors.
    kCommandPropertyDeactivatable  = 0x0001,

    /// @brief An idempotent command can be re-executed without side effects
    /// i.e., will not write device or host memory.
    /// Idempotent commands can be launched without waiting for previous
    /// deactivatable commands.
    kCommandPropertyIdempotent     = 0x0002,

    /// @brief A blocking-submit command will block XQueue->Submit() until
    /// it is actually launched to the XPU.
    kCommandPropertyBlockingSubmit = 0x0004,

    kCommandPropertyMaskAll        = -1,
};

enum XCommandState
{
    kCommandStateCreated    = 0,
    kCommandStatePending    = 1,
    kCommandStateInFlight   = 2,
    kCommandStateCompleted  = 3,
    kCommandStateMax        = 4,
};

class XCommand : public std::enable_shared_from_this<XCommand>
{
public:
    XCommand(XCommandType type, XCommandProperties props = kCommandPropertyNone)
        : kCommandType(type), props_(props) {}
    virtual ~XCommand() = default;

    virtual void Wait() { this->WaitUntil(kCommandStateCompleted); }

    XCommandState GetState();
    void SetState(XCommandState state);
    void WaitUntil(XCommandState state);
    void AddStateListener(std::function<void(XCommandState)> listener)
    { state_listeners_.push_back(listener); }

    /// @brief Get the lock of the command for some lightweight operations. Not recommended
    /// for heavy operations because XCommand needs the lock to protect the XCommandState.
    /// @return The lock of the command (locked).
    std::unique_lock<std::mutex> GetLock() { return std::unique_lock<std::mutex>(mtx_); }

    /// @brief Get the type of the command.
    /// @return The type of the command.
    XCommandType GetType() const { return kCommandType; }

    /// @brief Set command properties.
    /// @param props The combination of xsched::preempt::XCommandProperty.
    /// Multiple properties can be set by OR-ing them together.
    void SetProps(XCommandProperties props) {props_ |= props;}

    /// @brief Get command properties.
    /// @param props_mask The mask of the properties to be get.
    /// Multiple properties can be set by OR-ing them together.
    /// @return The properties in the prop_mask.
    XCommandProperties GetProps(XCommandProperties props_mask = kCommandPropertyMaskAll) const
    { return props_ & props_mask; }

private:
    const XCommandType kCommandType;
    XCommandState state_ = kCommandStateCreated;
    XCommandProperties props_;

    std::mutex mtx_;
    std::condition_variable state_cv_;
    std::list<std::function<void(XCommandState)>> state_listeners_;
};

class HostFunctionCommand final : public XCommand
{
public:
    HostFunctionCommand(std::function<void()> f): XCommand(kCommandTypeHostFunction), func_(f) {}
    virtual ~HostFunctionCommand() = default;
    void Execute() { return func_(); }

private:
    std::function<void()> func_;
};

class XQueueWaitAllCommand final : public XCommand
{
public:
    XQueueWaitAllCommand(): XCommand(kCommandTypeXQueueWaitAll) {}
    virtual ~XQueueWaitAllCommand() = default;
};

class BatchSynchronizeCommand final : public XCommand
{
public:
    BatchSynchronizeCommand(): XCommand(kCommandTypeBatchSynchronize) {}
    virtual ~BatchSynchronizeCommand() = default;
};

class XQueueDestroyCommand final : public XCommand
{
public:
    XQueueDestroyCommand(): XCommand(kCommandTypeXQueueDestroy) {}
    virtual ~XQueueDestroyCommand() = default;
};

} // namespace xsched::preempt
