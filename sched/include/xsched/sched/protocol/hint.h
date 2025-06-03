#pragma once

#include <memory>
#include <cstddef>

#include "xsched/types.h"
#include "xsched/utils/common.h"

namespace xsched::sched
{

// NEW_POLICY: New Hints and HintTypes should be added here
// when creating a new policy with new hints.

enum HintType
{
    kHintTypeUnknown     = 0,
    kHintTypePriority    = 1,
    kHintTypeUtilization = 2,
    kHintTypeTimeslice   = 3,
    kHintTypeDeadline    = 4,
    kHintTypeLaxity      = 5,
    kHintTypeKDeadline   = 6,
    // NEW_POLICY: New HintTypes go here.

    kHintTypeMax,
};

struct HintMeta
{
    HintType type;
};

class Hint
{
public:
    Hint() = default;
    virtual ~Hint() = default;

    /// @brief Get the data of the Hint. MUST start with EventType.
    virtual const void *Data() const = 0;
    virtual size_t      Size() const = 0;
    virtual HintType    Type() const = 0;

    static std::shared_ptr<const Hint> CopyConstructor(const void *data);
};

class PriorityHint : public Hint
{
public:
    PriorityHint(const void *data): data_(*(const HintData *)data) {}
    PriorityHint(XQueueHandle handle, Priority prio)
        : data_{
            .meta { .type = kHintTypePriority },
            .handle = handle,
            .prio = prio
        } {}
    virtual ~PriorityHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypePriority; }

    XQueueHandle Handle() const { return data_.handle; }
    Priority     Prio()   const { return data_.prio; }

private:
    struct HintData
    {
        HintMeta     meta;
        XQueueHandle handle;
        Priority     prio;
    };

    HintData data_;
};

class UtilizationHint : public Hint
{
public:
    UtilizationHint(const void *data): data_(*(const HintData *)data) {}
    /// @brief Create a UtilizationHint to set the utilization of a process or an XQueue.
    /// @param pid The process ID for which to set the utilization. 0 means not to set.
    /// @param handle The XQueueHandle for which to set the utilization. 0 means not to set.
    /// @param util The utilization to set.
    UtilizationHint(PID pid, XQueueHandle handle, Utilization util)
        : data_{
            .meta { .type = kHintTypeUtilization },
            .pid = pid,
            .handle = handle,
            .util = util
        } {}
    virtual ~UtilizationHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypeUtilization; }

    XQueueHandle Handle() const { return data_.handle; }
    Utilization  Util()   const { return data_.util; }
    PID          Pid()    const { return data_.pid; }

private:
    struct HintData
    {
        HintMeta     meta;
        PID          pid;
        XQueueHandle handle;
        Utilization  util;
    };

    HintData data_;
};


class TimesliceHint : public Hint
{
public:
    TimesliceHint(const void *data): data_(*(const HintData *)data) {}
    TimesliceHint(Timeslice ts_us)
        : data_{
            .meta { .type = kHintTypeTimeslice },
            .ts_us = ts_us
        } {}
    virtual ~TimesliceHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypeTimeslice; }

    Timeslice Ts() const { return data_.ts_us; }

private:
    struct HintData
    {
        HintMeta  meta;
        Timeslice ts_us;
    };

    HintData data_;
};

class DeadlineHint : public Hint
{
public:
    DeadlineHint(const void *data): data_(*(const HintData *)data) {}
    DeadlineHint(XQueueHandle handle, Deadline ddl)
        : data_{
            .meta { .type = kHintTypeDeadline },
            .handle = handle,
            .ddl = ddl
        } {}
    virtual ~DeadlineHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypeDeadline; }

    XQueueHandle Handle() const { return data_.handle; }
    Deadline     Ddl()    const { return data_.ddl; }

private:
    struct HintData
    {
        HintMeta     meta;
        XQueueHandle handle;
        Deadline     ddl;
    };
    HintData data_;
};

class KDeadlineHint : public Hint   
{
public:
    KDeadlineHint(const void *data): data_(*(const HintData *)data) {}
    KDeadlineHint(int k)
        : data_{
            .meta { .type = kHintTypeKDeadline },
            .k = k
        } {}
    virtual ~KDeadlineHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypeKDeadline; }

    int K() const { return data_.k; }

private:
    struct HintData
    {
        HintMeta meta;
        int      k;
    };
    HintData data_;
};

class LaxityHint : public Hint
{
public:
    LaxityHint(const void *data): data_(*(const HintData *)data) {}
    LaxityHint(XQueueHandle handle, Laxity lax, Priority lax_prio, Priority crit_prio)
        : data_{
            .meta { .type = kHintTypeLaxity },
            .handle = handle,
            .lax = lax,
            .lax_prio = lax_prio,
            .crit_prio = crit_prio
        } {}
    virtual ~LaxityHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintTypeLaxity; }

    XQueueHandle Handle()   const { return data_.handle; }
    Laxity       Lax()      const { return data_.lax; }
    Priority     LaxPrio()  const { return data_.lax_prio; }
    Priority     CritPrio() const { return data_.crit_prio; }

private:
    struct HintData
    {
        HintMeta     meta;
        XQueueHandle handle;
        Laxity       lax;
        Priority     lax_prio;
        Priority     crit_prio;
    };
    HintData data_;
};

// NEW_POLICY: New Hints go here.

} // namespace xsched::sched
