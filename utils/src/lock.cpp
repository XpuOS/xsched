#include "xsched/utils/lock.h"
#include "xsched/utils/common.h"

#if defined(_WIN32)
    #include <intrin.h>
#endif

using namespace xsched::utils;

namespace
{

inline void CpuRelax()
{
#if defined(_WIN32)
    _mm_pause();
#elif defined(ARCH_X86) || defined(ARCH_X86_64)
    asm volatile("pause" ::: "memory");
#elif defined(ARCH_ARM) || defined(ARCH_AARCH64)
    asm volatile("yield" ::: "memory");
#else
    // LoongArch has no userspace pause/yield instruction. The Linux kernel's
    // cpu_relax() is a compiler barrier on this architecture as well.
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

} // namespace

thread_local MCSLock::MCSNode MCSLock::me;

void MCSLock::lock()
{
    me.flag.store(kLockWaiting, std::memory_order_relaxed);
    me.next.store(nullptr, std::memory_order_relaxed);

    MCSNode *predecessor = tail_.exchange(&me, std::memory_order_acq_rel);
    if (predecessor != nullptr) {
        predecessor->next.store(&me, std::memory_order_release);
        while (me.flag.load(std::memory_order_acquire) != kLockGranted) {
            CpuRelax();
        }
    }
}

void MCSLock::unlock()
{
    MCSNode *successor = me.next.load(std::memory_order_acquire);
    if (successor == nullptr) {
        MCSNode *expected = &me;
        if (tail_.compare_exchange_strong(expected, nullptr,
                                          std::memory_order_release,
                                          std::memory_order_relaxed)) {
            return;
        }
        do {
            CpuRelax();
            successor = me.next.load(std::memory_order_acquire);
        } while (successor == nullptr);
    }

    successor->flag.store(kLockGranted, std::memory_order_release);
}
