#include "vecadd.h"
#include "xsched/utils.h"
#include "xsched/xsched.h"
#include "xsched/opencl/hal.h"

int main()
{
    VecAddTask task(CL_DEVICE_TYPE_GPU, 4096, 1024, true);
    task.Run();
    task.CheckResult();
}
