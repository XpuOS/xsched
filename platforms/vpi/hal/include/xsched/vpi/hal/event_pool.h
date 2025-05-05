#pragma once

#include "xsched/utils/pool.h"
#include "xsched/vpi/hal/driver.h"
#include "xsched/vpi/hal/vpi_assert.h"

namespace xsched::vpi
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
        VPIEvent event;
        VPI_ASSERT(Driver::EventCreate(VPI_EVENT_DISABLE_TIMESTAMP, &event));
        return event;
    }
};

} // namespace xsched::vpi
