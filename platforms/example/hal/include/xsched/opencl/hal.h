#pragma once

#include <stddef.h>
#include "xsched/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// types definitions in cl.h
typedef int8_t          cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t         cl_short;
typedef uint16_t        cl_ushort;
typedef int32_t         cl_int;
typedef uint32_t        cl_uint;
typedef int64_t         cl_long;
typedef uint64_t        cl_ulong;
typedef uint16_t        cl_half;
typedef float           cl_float;
typedef double          cl_double;
typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;
typedef cl_uint             cl_bool;

struct KernelArgument
{
    cl_uint index;
    size_t  size;
    void *  value;
};

// create a HwQueue
XResult OpenclQueueCreate(HwQueueHandle *hwq, cl_command_queue cmdq);

// create HwCommands
XResult OpenclKernelCommandCreate(HwCommandHandle *hw_cmd,
                                  cl_kernel kernel, cl_uint work_dim,
                                  const size_t *global_work_offset,
                                  const size_t *global_work_size,
                                  const size_t *local_work_size,
                                  cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                  cl_event *event,
                                  size_t num_args, KernelArgument *args);
XResult OpenclReadCommandCreate(HwCommandHandle *hw_cmd,
                                cl_mem buffer, cl_bool blocking_read,
                                size_t offset, size_t size, void *ptr,
                                cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                cl_event *event);
XResult OpenclWriteCommandCreate(HwCommandHandle *hw_cmd,
                                 cl_mem buffer, cl_bool blocking_write,
                                 size_t offset, size_t size, const void *ptr,
                                 cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                 cl_event *event);

#ifdef __cplusplus
}
#endif
