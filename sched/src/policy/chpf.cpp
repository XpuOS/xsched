#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "xsched/sched/policy/chpf.h"

using namespace xsched::sched;

void CPUHighestPriorityFirstPolicy::Sched(const Status &status)
{
    // get linux priority by procfs
    std::map<PID, Priority> pid_prio;
    for(auto &status : status.process_status) {
        PID pid = status.first;
        std::ifstream sched_file("/proc/" + std::to_string(pid) + "/sched");
        std::string line;
        while(std::getline(sched_file, line)) {
            if (line.find("prio") == std::string::npos) {
                continue;
            }
            std::string dummy;
            Priority prio;
            std::stringstream sin(line);
            sin >> dummy >> dummy >> prio;
            // for static priority, the more prio, the less priority level
            if (prio >= 100) {
                prio = - prio;
            }
            pid_prio[pid] = prio;
            // XINFO("pid -> prio : %10d -> %4d", pid, prio);
            break;
        }
    }

    // find the running highest priority task of each device
    std::map<XDevice, Priority> running_prio_max;
    for (auto &status : status.xqueue_status) {
        if (!status.second->ready) continue;
        Priority priority = pid_prio[status.second->pid];

        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it == running_prio_max.end()) {
            running_prio_max[status.second->device] = priority;
        } else if (priority > prio_it->second) {
            prio_it->second = priority;
        }
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Priority priority = pid_prio[status.second->pid];

        // get the running highest priority task of the device
        Priority prio_max = PRIORITY_MIN;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it != running_prio_max.end()) prio_max = prio_it->second;
        if (priority < prio_max) {
            this->Suspend(handle);
        } else {
            this->Resume(handle);
        }
    }
}

void CPUHighestPriorityFirstPolicy::RecvHint(std::shared_ptr<const Hint> hint)
{
    // CHPF omits any hint
    (void)hint;
    return;
}