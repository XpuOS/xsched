#pragma once

#include <string>

#include "task.h"
#include "xsched/xsched.h"

class VecAddTask : public OclTask
{
public:
    VecAddTask(cl_device_type dev_type, int elem_cnt, int add_cnt, bool use_hal_api = false);
    virtual ~VecAddTask();
    virtual void Run() override;
    void CheckResult();

private:
    static const std::string kCode;
    const int kElemCnt;
    const int kVecSize;
    const int kAddCnt;
    const bool kUseHalApi;
    XQueueHandle     xq_;
    HwQueueHandle    hwq_;

    int current_A = 0;
    int current_C = 1;
    int total_add_cnt = 0;
    int *host_vec[2] = {nullptr, nullptr};
    int *host_adder = nullptr;
    cl_mem dev_vec[2] = {nullptr, nullptr};
    cl_mem dev_adder = nullptr;
};
