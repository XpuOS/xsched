#include "xsched/utils/lock.h"
#include "xsched/utils/common.h"

#if defined(__linux__)
    #if defined(ARCH_X86_64)
        #define memory_barrier() asm volatile("pause" ::: "memory")
    #elif defined(ARCH_AARCH64)
        #define memory_barrier() asm volatile("yield" ::: "memory")
    #endif
#elif defined(_WIN32)
    #include <intrin.h>
    #define memory_barrier() _mm_pause()
#endif

using namespace xsched::utils;

thread_local MCSLock::MCSNode MCSLock::me;

void MCSLock::lock()
{
    MCSNode *tail = nullptr;
    me.flag = kLockWaiting;
    me.next = nullptr;
    tail = tail_.exchange(&me);
    if (tail) {
        tail->next = &me;
        while (me.flag != kLockGranted) {
            memory_barrier();
        }
    }
}

void MCSLock::unlock()
{
    if (!me.next) {
        MCSNode *me_ptr = &me;
        if (tail_.compare_exchange_strong(me_ptr, nullptr)) {
            return;
        }
        while (!me.next) {
            memory_barrier();
        }
    }
    me.next->flag = kLockGranted;
}
