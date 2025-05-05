#include "task.h"
#include "ocl_utils.h"

OclTask::OclTask(cl_device_type dev_type, const char *kern_name, const char *kern_code)
{
    // get platforms
    cl_uint plat_num;
	OCL_ASSERT(clGetPlatformIDs(0, nullptr, &plat_num));
    cl_platform_id *plat_ids = new cl_platform_id[plat_num];
    OCL_ASSERT(clGetPlatformIDs(plat_num, plat_ids, nullptr));
    // print platform info
    for (cl_uint i = 0; i < plat_num; ++i) {
        char plat_name[128];
        OCL_ASSERT(clGetPlatformInfo(plat_ids[i], CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr));
        XINFO("found platform %d: %s", i, plat_name);
    }
    // select platform
    plat_ = plat_ids[0];
    delete[] plat_ids;

    // get devices
    cl_uint dev_num;
    OCL_ASSERT(clGetDeviceIDs(plat_, dev_type, 0, nullptr, &dev_num));
    cl_device_id *dev_ids = new cl_device_id[dev_num];
    OCL_ASSERT(clGetDeviceIDs(plat_, dev_type, dev_num, dev_ids, nullptr));
    // print device info
    for (cl_uint i = 0; i < dev_num; ++i) {
        char dev_name[128];
        OCL_ASSERT(clGetDeviceInfo(dev_ids[i], CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr));
        XINFO("found device %d: %s", i, dev_name);
    }
    // select device
    dev_ = dev_ids[0];
    delete[] dev_ids;

    // create context
    cl_int errcode;
    ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &errcode);
    OCL_ASSERT(errcode);

    // create command queue
    cmdq_ = clCreateCommandQueueWithProperties(ctx_, dev_, nullptr, &errcode);
    OCL_ASSERT(errcode);

    // create program
    prog_ = clCreateProgramWithSource(ctx_, 1, &kern_code, nullptr, &errcode);
    OCL_ASSERT(errcode);
    OCL_ASSERT(clBuildProgram(prog_, 1, &dev_, nullptr, nullptr, nullptr));

    // create kernel
    kern_ = clCreateKernel(prog_, kern_name, &errcode);
    OCL_ASSERT(errcode);
}

OclTask::~OclTask()
{
    if (kern_) clReleaseKernel(kern_);
    if (prog_) clReleaseProgram(prog_);
    if (cmdq_) clReleaseCommandQueue(cmdq_);
    if (ctx_)  clReleaseContext(ctx_);
}
