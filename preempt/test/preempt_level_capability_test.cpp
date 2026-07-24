#include <iostream>
#include <memory>
#include <string>

#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/xqueue.h"

namespace
{

using namespace xsched::preempt;

class CapabilityQueue final : public HwQueue
{
public:
    virtual void Launch(std::shared_ptr<HwCommand>) override { }
    virtual void Synchronize() override { }
    virtual void Deactivate() override { ++deactivate_count; }
    virtual void Reactivate(const CommandLog &) override
    {
        ++reactivate_count;
    }

    virtual XDevice GetDevice() override { return 0; }
    virtual HwQueueHandle GetHandle() override { return kHandle; }
    virtual bool SupportDynamicLevel() override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override
    {
        return kPreemptLevelDeactivate;
    }

    static constexpr HwQueueHandle kHandle = 0xc0ffee;
    int deactivate_count = 0;
    int reactivate_count = 0;
};

bool Expect(bool condition, const std::string &message)
{
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

} // namespace

int main()
{
    using namespace xsched::preempt;

    auto queue = std::make_shared<CapabilityQueue>();
    bool ok = Expect(
        HwQueueManager::Add(CapabilityQueue::kHandle,
            [queue]() { return queue; })
            == kXSchedSuccess,
        "register Level-2 HwQueue");

    XQueueHandle xqueue = 0;
    ok &= Expect(XQueueCreate(&xqueue, CapabilityQueue::kHandle,
                     kPreemptLevelInterrupt,
                     kQueueCreateFlagNone)
            == kXSchedErrorNotSupported,
        "reject Level-3 creation on a Level-2 HwQueue");
    ok &= Expect(XQueueCreate(&xqueue, CapabilityQueue::kHandle,
                     kPreemptLevelDeactivate,
                     kQueueCreateFlagNone)
            == kXSchedSuccess,
        "create Level-2 XQueue");
    if (xqueue != 0) {
        ok &= Expect(XQueueSetPreemptLevel(xqueue,
                         kPreemptLevelInterrupt)
                == kXSchedErrorNotSupported,
            "reject dynamic upgrade beyond Level-2");
        ok &= Expect(XQueueSuspend(xqueue, kQueueSuspendFlagNone) == kXSchedSuccess,
            "suspend Level-2 XQueue");
        ok &= Expect(queue->deactivate_count == 1,
            "suspend must call Deactivate");
        ok &= Expect(XQueueResume(xqueue, kQueueResumeFlagNone) == kXSchedSuccess,
            "resume Level-2 XQueue");
        ok &= Expect(queue->reactivate_count == 1,
            "resume must call Reactivate");
        ok &= Expect(XQueueDestroy(xqueue) == kXSchedSuccess,
            "destroy Level-2 XQueue");
    }
    ok &= Expect(HwQueueDestroy(CapabilityQueue::kHandle) == kXSchedSuccess,
        "destroy Level-2 HwQueue");

    if (!ok) return 1;
    std::cout << "HwQueue preemption capability tests passed.\n";
    return 0;
}
