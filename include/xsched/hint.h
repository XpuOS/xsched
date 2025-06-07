#pragma once

#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Set the scheduler and policy.
/// @param scheduler [in] Scheduler type to set.
/// @param policy    [in] Policy type to set. Used only when scheduler is kSchedulerLocal.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintSetScheduler(XSchedulerType scheduler, XPolicyType policy);

/// @brief Set the priority of an XQueue.
/// @param xq    [in] Handle to the XQueue.
/// @param prio  [in] Priority to set, bigger value means higher priority.
///     PRIORITY_MIN: the lowest priority.
///     PRIORITY_MAX: the highest priority.
///     PRIORITY_NO_EXECUTE: The XQueue will not be executed.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintPriority(XQueueHandle xq, Priority prio);

/// @brief Set the utilization of an XQueue.
/// @param xq     [in] Handle to the XQueue.
/// @param util   [in] Utilization to set. A percentage integer,
/// with 0 indicating no utilization and 100 indicating full utilization.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintUtilization(XQueueHandle xq, Utilization util);

/// @brief Set the timeslice of the policy.
/// @param ts_us [in] Timeslice in microseconds to set.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintTimeslice(Timeslice ts_us);

/// @brief Set the laxity, the lax priority and critical priority of the XQueue.
/// @param xq         [in] Handle to the XQueue.
/// @param lax_us     [in] Laxity in microseconds to set.
/// Indicates how much time the XQueue can be delayed.
/// @param lax_prio   [in] Lax priority, the priority when laxity is not used up.
/// @param crit_prio  [in] Critical priority, the priority when laxity is used up.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintLaxity(XQueueHandle xq, Laxity lax_us, Priority lax_prio, Priority crit_prio);

/// @brief Set the deadline of the XQueue.
/// @param xq     [in] Handle to the XQueue.
/// @param ddl_us [in] Deadline in microseconds to set.
/// The deadline of the XQueue is the ready time of the XQueue plus ddl_us.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintDeadline(XQueueHandle xq, Deadline ddl_us);

/// @brief Set the concurrency (k) of the K-Earliest Deadline First (K-EDF) policy.
/// k XQueues with the earliest deadline will be executed concurrently.
/// @param k [in] Concurrency (k) to set.
/// @return kXSchedSuccess if successful, otherwise an error code.
XResult XHintKDeadline(size_t k);

#ifdef __cplusplus
}
#endif
