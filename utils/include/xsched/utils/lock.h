#pragma once

#include <mutex>
#include <atomic>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace xsched::utils
{

class MutexLock
{
public:
    MutexLock() = default;
    virtual ~MutexLock() = default;

    virtual void lock() = 0;
    virtual void unlock() = 0;
};

class StdMutex : public MutexLock
{
private:
    std::mutex mutex_;

public:
    StdMutex() = default;
    virtual ~StdMutex() = default;

    virtual void lock() override { mutex_.lock(); }
    virtual void unlock() override { mutex_.unlock(); }
};

class SpinLock : public MutexLock
{
#if defined(_WIN32)
public:
    SpinLock() { InitializeSRWLock(&srwlock_); }
    virtual ~SpinLock() = default;
    virtual void lock()   override { AcquireSRWLockExclusive(&srwlock_); }
    virtual void unlock() override { ReleaseSRWLockExclusive(&srwlock_); }
    void tryLock()                 { TryAcquireSRWLockExclusive(&srwlock_); }

private:
    SRWLOCK srwlock_;
#else
public:
    SpinLock() { pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE); }
    virtual ~SpinLock()            { pthread_spin_destroy(&spinlock_); }
    virtual void lock()   override { pthread_spin_lock(&spinlock_); }
    virtual void unlock() override { pthread_spin_unlock(&spinlock_); }
    void tryLock()                 { pthread_spin_trylock(&spinlock_); }

private:
    pthread_spinlock_t spinlock_;
#endif
};

class MCSLock : public MutexLock
{
public:
    MCSLock() = default;
    virtual ~MCSLock() = default;

    virtual void lock() override;
    virtual void unlock() override;

private:
    enum LockStatus
    {
        kLockWaiting = 0,
        kLockGranted = 1
    };

    struct alignas(64) MCSNode
    {
        volatile LockStatus flag;
        volatile MCSNode *next;
    };

    static thread_local MCSNode me;
    std::atomic<MCSNode *> tail_{nullptr};
};

} // namespace xsched::utils
