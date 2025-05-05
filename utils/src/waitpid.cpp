#include <string>
#include <unistd.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <sys/inotify.h>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"
#include "xsched/utils/waitpid.h"

#define SCAN_INTERVAL_US 1000

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif

using namespace xsched::utils;

std::unique_ptr<PidWaiter> PidWaiter::Create(TerminateCallback callback)
{
    if (callback == nullptr) {
        XERRO("callback is nullptr");
        return nullptr;
    }

    // check if the system supports pidfd_open
    int self_pid_fd = PidFdWaiter::OpenPidFd(GetProcessId(), 0);
    if (self_pid_fd == -1) {
        XWARN("pidfd_open is not supported, using scan method, which may consume more CPU");
        return std::make_unique<ScanPidWaiter>(callback);
    }

    XASSERT(!close(self_pid_fd), "fail to close self pid fd");
    XINFO("pidfd_open is supported, using pidfd_wait method");
    return std::make_unique<PidFdWaiter>(callback);
}

PidFdWaiter::~PidFdWaiter()
{
    this->Stop();
}

void PidFdWaiter::Start()
{
    event_fd_ = eventfd(0, EFD_CLOEXEC);
    XASSERT(event_fd_ >= 0, "fail to create event fd");
    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    XASSERT(epoll_fd_ >= 0, "fail to create epoll fd");

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = PackEventData(kEpollEventTerminate, 0);
    XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &ev),
            "fail to add event fd to epoll");

    thread_ = std::make_unique<std::thread>(&PidFdWaiter::WaitWorker, this);
}

void PidFdWaiter::Stop()
{
    if (thread_ != nullptr) {
        XASSERT(!eventfd_write(event_fd_, 1), "fail to write event fd");
        thread_->join();
        thread_ = nullptr;
    }

    if (event_fd_ >= 0) close(event_fd_);
    if (epoll_fd_ >= 0) close(epoll_fd_);
    for (auto& it : pid_fds_) { close(it.second); }
    
    event_fd_ = -1;
    epoll_fd_ = -1;
    pid_fds_.clear();
}

void PidFdWaiter::AddWait(PID pid)
{
    int pid_fd = OpenPidFd(pid, 0);
    if (pid_fd < 0) {
        // check if the process is already terminated
        if (errno == ESRCH) {
            XDEBG("process %d is already terminated", pid);
            callback_(pid);
            return;
        }
        XERRO("fail to open pid fd for pid %d", pid);
    }

    mtx_.lock();
    pid_fds_[pid] = pid_fd;
    mtx_.unlock();

    // According to the notes of linux man page of epoll_wait at
    // https://www.man7.org/linux/man-pages/man2/epoll_wait.2.html
    // "While one thread is blocked in a call to epoll_wait(), it is
    // possible for another thread to add a file descriptor to the
    // waited-upon epoll instance. If the new file descriptor becomes
    // ready, it will cause the epoll_wait() call to unblock."
    // So we can safely add the pid fd to epoll.
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = PackEventData(kEpollEventPid, pid);
    XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, pid_fd, &ev),
            "fail to add pid fd to epoll");
}

void PidFdWaiter::WaitWorker()
{
    struct epoll_event ev;
    while (true) {
        if (epoll_wait(epoll_fd_, &ev, 1, -1) == -1) {
            XASSERT(errno == EINTR, "fail during epoll wait");
			continue;
		}

        if (GetEventType(ev.data.u64) == kEpollEventTerminate) {
            eventfd_t v;
            XASSERT(!eventfd_read(event_fd_, &v), "fail to read event fd");
            return;
        }

        XASSERT(GetEventType(ev.data.u64) == kEpollEventPid,
                "invalid event type: %d", GetEventType(ev.data.u64));

        PID pid = GetEventPid(ev.data.u64);
        
        mtx_.lock();
        auto it = pid_fds_.find(pid);
        XASSERT(it != pid_fds_.end(), "pid fd not found");
        int pid_fd = it->second;
        pid_fds_.erase(it);
        mtx_.unlock();

        XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, pid_fd, nullptr),
                "fail to remove pid fd from epoll");
        XASSERT(!close(pid_fd), "fail to close pid fd");
        callback_(pid);
    }
}

PID PidFdWaiter::GetEventPid(uint64_t data)
{
    return PID(data & 0xFFFFFFFF);
}

EpollEventType PidFdWaiter::GetEventType(uint64_t data)
{
    return EpollEventType(data >> 32);
}

uint64_t PidFdWaiter::PackEventData(EpollEventType type, PID pid)
{
    return ((uint64_t)type << 32) | (uint64_t)pid;
}

int PidFdWaiter::OpenPidFd(PID pid, unsigned int flags)
{
	return syscall(SYS_pidfd_open, (pid_t)pid, flags);
}

ScanPidWaiter::~ScanPidWaiter()
{
    this->Stop();
}

void ScanPidWaiter::Start()
{
    running_.store(true);
    thread_ = std::make_unique<std::thread>(&ScanPidWaiter::WaitWorker, this);
}

void ScanPidWaiter::Stop()
{
    running_.store(false);
    if (thread_ != nullptr) thread_->join();
    thread_ = nullptr;
}

void ScanPidWaiter::AddWait(PID pid)
{
    std::lock_guard<std::mutex> lock(mtx_);
    pids_.insert(pid);
}

void ScanPidWaiter::WaitWorker()
{
    while (running_.load()) {
        mtx_.lock();
        std::unordered_set<PID> set = pids_;
        mtx_.unlock();

        std::list<PID> terminated;
        for (auto pid : set) {
            if (kill(pid, 0) == 0) continue;
            if (errno != ESRCH) {
                XWARN("fail to test process %d, errno: %d", pid, errno);
                continue;
            }
            terminated.emplace_back(pid);
            callback_(pid);
        }

        mtx_.lock();
        for (auto pid : terminated) pids_.erase(pid);
        mtx_.unlock();

        std::this_thread::sleep_for(std::chrono::microseconds(SCAN_INTERVAL_US));
    }
}

INotifyPidWaiter::~INotifyPidWaiter()
{
    this->Stop();
}

void INotifyPidWaiter::Start()
{
    inotify_fd_ = inotify_init1(0);
    XASSERT(inotify_fd_ >= 0, "fail to create inotify fd");
    thread_ = std::make_unique<std::thread>(&INotifyPidWaiter::WaitWorker, this);
}

void INotifyPidWaiter::Stop()
{
    if (inotify_fd_ >= 0) close(inotify_fd_);
    inotify_fd_ = -1;

    if (thread_ != nullptr) thread_->join();
    thread_ = nullptr;
}

void INotifyPidWaiter::AddWait(PID pid)
{
    std::string proc_path = "/proc/" + std::to_string(pid);
    std::lock_guard<std::mutex> lock(mtx_);
    int wd = inotify_add_watch(inotify_fd_, proc_path.c_str(), IN_DELETE_SELF);
    if (wd < 0) {
        // check if the process is already terminated
        if (errno == ENOENT) {
            XDEBG("process %d is already terminated", pid);
            callback_(pid);
            return;
        }
        XERRO("fail to add watch for pid %d", pid);
    }
    watch_pids_[wd] = pid;
}

void INotifyPidWaiter::WaitWorker()
{
    char buf[4096];
    while (inotify_fd_ >= 0) {
        ssize_t n = read(inotify_fd_, buf, sizeof(buf));
        if (n < 0) {
            if (errno == EBADF) return; // inotify fd is closed
            XWARN("read error during inotify wait");
            continue;
        }

        ssize_t i = 0;
        while (i < n) {
            struct inotify_event *event = (struct inotify_event *) &buf[i];
            if (event->mask & IN_DELETE_SELF) {
                mtx_.lock();
                auto it = watch_pids_.find(event->wd);
                XASSERT(it != watch_pids_.end(), "watch fd not found");
                PID pid = it->second;
                watch_pids_.erase(it);
                mtx_.unlock();
                callback_(pid);
                inotify_rm_watch(inotify_fd_, event->wd);
            }
            i += sizeof(struct inotify_event) + event->len;
        }
    }
}
