#include "vecadd.h"
#include "xsched/utils.h"
#include "xsched/xsched.h"
#include "xsched/opencl/hal.h"

int main()
{
    XQueueHandle xq;
    HwQueueHandle hwq;

    VecAddTask task(CL_DEVICE_TYPE_GPU, 4096, 1024);
    cl_command_queue cmdq = task.GetCommandQueue();

    OclQueueCreate(&hwq, cmdq);
    XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone);

    task.Run();
    task.CheckResult();

    int64_t cnt_before = 0;
    int64_t cnt_after  = 0;

    XQueueProfileHwCommandCount(xq, &cnt_before);
    XINFO("hw cmd cnt BEFORE run: %ld", cnt_before);
    task.Run();
    XQueueProfileHwCommandCount(xq, &cnt_after);
    XINFO("hw cmd cnt AFTER  run: %ld", cnt_after);

    if (cnt_after != cnt_before) {
        XINFO("intercept success");
    } else {
        XWARN("intercept failed");
    }

    XQueueDestroy(xq);
    HwQueueDestroy(hwq);
}
