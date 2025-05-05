#pragma once

#include "xsched/utils/pool.h"
#include "xsched/ascend/hal/driver.h"
#include "xsched/ascend/hal/acl_assert.h"

namespace xsched::ascend
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
        aclrtEvent event;
        ACL_ASSERT(Driver::rtCreateEvent(&event));
        return event;
    }
};

} // namespace xsched::ascend
