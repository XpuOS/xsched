#pragma once

#include <CL/cl.h>

class OclTask
{
public:
    OclTask(cl_device_type dev_type, const char *kern_name, const char *kern_code);
    virtual ~OclTask();
    cl_command_queue GetCommandQueue() { return cmdq_; }

    virtual void Run() = 0;

protected:
    cl_platform_id   plat_;
    cl_device_id     dev_;
    cl_context       ctx_;
    cl_program       prog_;
    cl_kernel        kern_;
    cl_command_queue cmdq_;
};
