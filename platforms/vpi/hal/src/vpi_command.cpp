#include "xsched/utils/xassert.h"
#include "xsched/vpi/hal/event_pool.h"
#include "xsched/vpi/hal/vpi_command.h"
#include "xsched/preempt/xqueue/xqueue.h"

using namespace xsched::vpi;

VpiCommand::~VpiCommand()
{
    if (following_event_ == nullptr) return;
    EventPool::Instance().Push(following_event_);
}

void VpiCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    VPI_ASSERT(Driver::EventSync(following_event_));
}

bool VpiCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool VpiCommand::EnableSynchronization()
{
    following_event_ = (VPIEvent)EventPool::Instance().Pop();
    return following_event_ != nullptr;
}

VPIStatus VpiCommand::LaunchWrapper(VPIStream stream)
{
    VPIStatus ret = Launch(stream);
    if (UNLIKELY(ret != VPI_SUCCESS)) return ret;
    if (following_event_ != nullptr) ret = Driver::EventRecord(following_event_, stream);
    return ret;
}

VpiEventRecordCommand::VpiEventRecordCommand(VPIEvent event)
    : VpiCommand(preempt::kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "vpi event should not be nullptr");
}
