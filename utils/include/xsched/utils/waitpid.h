#pragma once

#include <list>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include "xsched/utils/common.h"

namespace xsched::utils
{

enum EpollEventType
{
    kEpollEventTerminate = 0,
    kEpollEventPid       = 1,
};

typedef std::function<void (PID)> TerminateCallback;

class PidWaiter
{
public:
    PidWaiter() = default;
    ~PidWaiter() = default;

    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual void AddWait(PID pid) = 0;

    static std::unique_ptr<PidWaiter> Create(TerminateCallback callback);
};

class PidFdWaiter : public PidWaiter
{
public:
    PidFdWaiter(TerminateCallback callback): callback_(callback) {}
    ~PidFdWaiter();

    virtual void Start() override;
    virtual void Stop() override;
    virtual void AddWait(PID pid) override;
    static int OpenPidFd(PID pid, unsigned int flags);

private:
    void WaitWorker();
    PID GetEventPid(uint64_t data);
    EpollEventType GetEventType(uint64_t data);
    uint64_t PackEventData(EpollEventType type, PID pid);

    int event_fd_ = -1;
    int epoll_fd_ = -1;

    std::mutex mtx_;
    std::unordered_map<PID, int> pid_fds_;

    TerminateCallback callback_;
    std::unique_ptr<std::thread> thread_ = nullptr;
};

class ScanPidWaiter : public PidWaiter
{
public:
    ScanPidWaiter(TerminateCallback callback): running_(false), callback_(callback) {}
    ~ScanPidWaiter();
    
    virtual void Start() override;
    virtual void Stop() override;
    virtual void AddWait(PID pid) override;

private:
    void WaitWorker();

    std::mutex mtx_;
    std::atomic_bool running_;
    std::unordered_set<PID> pids_;
    TerminateCallback callback_;
    std::unique_ptr<std::thread> thread_ = nullptr;
};

// not working for /proc file system
class INotifyPidWaiter : public PidWaiter
{
public:
    INotifyPidWaiter(TerminateCallback callback): callback_(callback) {}
    ~INotifyPidWaiter();

    virtual void Start() override;
    virtual void Stop() override;
    virtual void AddWait(PID pid) override;

private:
    void WaitWorker();

    int inotify_fd_ = -1;
    std::mutex mtx_;
    std::unordered_map<int, PID> watch_pids_;

    TerminateCallback callback_;
    std::unique_ptr<std::thread> thread_ = nullptr;
};

} // namespace xsched::utils
