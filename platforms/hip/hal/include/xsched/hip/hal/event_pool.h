#pragma once

#include <queue>
#include <mutex>

#include "xsched/utils/pool.h"
#include "xsched/utils/common.h"
#include "xsched/hip/hal/driver.h"
#include "xsched/hip/hal/hip_assert.h"

namespace xsched::hip
{

class EventPool : public xsched::utils::ObjectPool
{
public:
    EventPool() = default;
    virtual ~EventPool() = default;

    static EventPool &Instance()
    {
        static EventPool event_pool;
        return event_pool;
    }

private:
    virtual void *Create() override
    {
        hipEvent_t event;
        HIP_ASSERT(Driver::EventCreateWithFlags(&event, hipEventBlockingSync | hipEventDisableTiming));
        return event;
    }
};

} // namespace xsched::hip
