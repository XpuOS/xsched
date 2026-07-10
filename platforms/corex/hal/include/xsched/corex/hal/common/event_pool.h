#pragma once

#include "xsched/utils/pool.h"
#include "xsched/corex/hal/common/driver.h"
#include "xsched/corex/hal/common/corex_assert.h"

namespace xsched::corex
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
        cudaEvent_t event;
        COREXRT_ASSERT(RtDriver::EventCreateWithFlags(&event,
            cudaEventBlockingSync | cudaEventDisableTiming));
        return event;
    }
};

} // namespace xsched::corex
