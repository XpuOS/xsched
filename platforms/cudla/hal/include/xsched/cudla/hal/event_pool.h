#pragma once

#include "xsched/utils/pool.h"
#include "xsched/cudla/hal/driver.h"
#include "xsched/cudla/hal/cudla_assert.h"

namespace xsched::cudla
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
        CUDART_ASSERT(RtDriver::EventCreateWithFlags(&event,
            cudaEventBlockingSync | cudaEventDisableTiming));
        return event;
    }
};

extern EventPool g_event_pool;

} // namespace xsched::cudla
