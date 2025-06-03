#include <cstring>

#include "xsched/utils/xassert.h"
#include "xsched/sched/protocol/hint.h"

using namespace xsched::sched;

std::shared_ptr<const Hint> Hint::CopyConstructor(const void *data)
{
    auto meta = (const HintMeta *)data;
    switch (meta->type)
    {
    // NEW_POLICY: New cases handling new HintTypes should be added here
    // when creating a new policy with new hints.
    case kHintTypePriority:
        return std::make_shared<PriorityHint>(data);
    case kHintTypeUtilization:
        return std::make_shared<UtilizationHint>(data);
    case kHintTypeTimeslice:
        return std::make_shared<TimesliceHint>(data);
    case kHintTypeDeadline:
        return std::make_shared<DeadlineHint>(data);
    case kHintTypeKDeadline:
        return std::make_shared<KDeadlineHint>(data);
    case kHintTypeLaxity:
        return std::make_shared<LaxityHint>(data);
    // NEW_POLICY: New HintTypes handling goes here.
    default:
        XASSERT(false, "unknown hint type: %d", meta->type);
        return nullptr;
    }
}
