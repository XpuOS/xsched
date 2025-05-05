#include "vecadd.h"
#include "ocl_utils.h"

const std::string VecAddTask::kCode = R"(
// VecAdd: for i < N, C[i] = A[i] + B[i]
__kernel void VecAdd(__global int *A, __global int *B, __global int *C, int N)
{
    int idx = get_global_id(0);
    if (idx < N) C[idx] = A[idx] + B[idx];
}
)";

VecAddTask::VecAddTask(cl_device_type dev_type, int elem_cnt, int add_cnt)
    : OclTask(dev_type, "VecAdd", kCode.c_str())
    , kElemCnt(elem_cnt), kVecSize(elem_cnt * sizeof(int)), kAddCnt(add_cnt)
{
    host_vec[0] = new int[elem_cnt];
    host_vec[1] = new int[elem_cnt];
    host_adder = new int[elem_cnt];

    for (int i = 0; i < kElemCnt; ++i) {
        host_vec[0][i] = 0;
        host_vec[1][i] = 0;
        host_adder[i] = 1;
    }

	dev_vec[0] = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, kVecSize, NULL, NULL);
	dev_vec[1] = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, kVecSize, NULL, NULL);
	dev_adder =  clCreateBuffer(ctx_, CL_MEM_READ_ONLY, kVecSize, NULL, NULL);

    OCL_ASSERT(clEnqueueWriteBuffer(cmdq_, dev_vec[0], CL_FALSE, 0, kVecSize, host_vec[0], 0, NULL, NULL));
    OCL_ASSERT(clEnqueueWriteBuffer(cmdq_, dev_vec[1], CL_FALSE, 0, kVecSize, host_vec[1], 0, NULL, NULL));
    OCL_ASSERT(clEnqueueWriteBuffer(cmdq_, dev_adder, CL_FALSE, 0, kVecSize, host_adder, 0, NULL, NULL));
    OCL_ASSERT(clFinish(cmdq_));

	OCL_ASSERT(clSetKernelArg(kern_, 1, sizeof(cl_mem), &dev_adder));
    OCL_ASSERT(clSetKernelArg(kern_, 3, sizeof(int), &elem_cnt));
}

VecAddTask::~VecAddTask()
{
    if (dev_vec[0]) clReleaseMemObject(dev_vec[0]);
    if (dev_vec[1]) clReleaseMemObject(dev_vec[1]);
    if (dev_adder)  clReleaseMemObject(dev_adder);
    if (host_vec[0]) delete[] host_vec[0];
    if (host_vec[1]) delete[] host_vec[1];
    if (host_adder)  delete[] host_adder;
}

void VecAddTask::Run()
{
    size_t global_work_size = kElemCnt;
    for (int i = 0; i < kAddCnt; ++i) {
        current_A = (current_A + 1) % 2;
        current_C = (current_C + 1) % 2;
        OCL_ASSERT(clSetKernelArg(kern_, 0, sizeof(cl_mem), &dev_vec[current_A]));
        OCL_ASSERT(clSetKernelArg(kern_, 2, sizeof(cl_mem), &dev_vec[current_C]));
        OCL_ASSERT(clEnqueueNDRangeKernel(cmdq_, kern_, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr));
        total_add_cnt++;
    }
    OCL_ASSERT(clFinish(cmdq_));
}

void VecAddTask::CheckResult()
{
    OCL_ASSERT(clEnqueueReadBuffer(cmdq_, dev_vec[current_C], CL_FALSE, 0, kVecSize, host_vec[current_C], 0, NULL, NULL));
    OCL_ASSERT(clFinish(cmdq_));
    for (int i = 0; i < kElemCnt; ++i) {
        if (host_vec[current_C][i] != total_add_cnt) {
            XWARN("check fail: C[%d] = %d != %d", i, host_vec[current_C][i], total_add_cnt);
            return;
        }
    }
    XINFO("check success: all results are correct!");
}
