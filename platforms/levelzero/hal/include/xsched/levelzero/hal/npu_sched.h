#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <drm/drm.h>
#include <sys/ioctl.h>

#include "xsched/utils/xassert.h"
#include "xsched/levelzero/hal/driver.h"

#define VPU_ENGINE_COMPUTE 0
#define VPU_ENGINE_COPY	   1
#define VPU_ENGINE_NB	   2

#define DRM_IVPU_JOB_PRIORITY_IDLE     1
#define DRM_IVPU_JOB_PRIORITY_NORMAL   2
#define DRM_IVPU_JOB_PRIORITY_FOCUS    3
#define DRM_IVPU_JOB_PRIORITY_REALTIME 4

#define DRM_IVPU_SCHED_OP_CMDQ_SUSPEND		0
#define DRM_IVPU_SCHED_OP_CMDQ_RESUME		1
#define DRM_IVPU_SCHED_OP_ENGINE_PREEMPT	2
#define DRM_IVPU_SCHED_OP_ENGINE_RESET		3
#define DRM_IVPU_SCHED_OP_ENGINE_RESUME		4

struct drm_ivpu_schedule {
	__u32 engine;
	__u32 priority;
	__u32 operation;
	__u32 request_id;
	__u64 ze_command_queue_id;
};

#define DRM_IVPU_SCHEDULE         0x10
#define DRM_IOCTL_IVPU_SCHEDULE                                                \
	DRM_IOW(DRM_COMMAND_BASE + DRM_IVPU_SCHEDULE, struct drm_ivpu_schedule)

#define FD_SCAN_MAX     1024
#define FD_LINK_LEN_MAX 128

inline int get_npu_fd()
{
    static int npu_fd = -1;
    static const char *npu_path = "/dev/accel/accel0";
    if (npu_fd != -1) return npu_fd;

    for (int fd = 0; fd < FD_SCAN_MAX; ++fd) {
        char path[FD_LINK_LEN_MAX];
        char link[FD_LINK_LEN_MAX];
    
        int link_len = sprintf(link, "/proc/self/fd/%d", fd);
        XASSERT(link_len < FD_LINK_LEN_MAX, "link overflow");
    
        ssize_t len = readlink(link, path, sizeof(path) - 1);
        if (len < 0) continue;
        XASSERT((size_t)len < sizeof(path), "path overflow");
        path[len] = '\0';

        if (strcmp(path, npu_path) == 0) {
            npu_fd = fd;
            printf("found npu fd (%s): %d\n", npu_path, npu_fd);
            return npu_fd;
        }
    }

    npu_fd = open(npu_path, O_RDWR);
    XASSERT(npu_fd != -1, "failed to open npu fd");
    printf("opened npu fd (%s): %d\n", npu_path, npu_fd);
    return npu_fd;
}

inline uint32_t get_driver_priority(ze_command_queue_priority_t p)
{
    switch (p) {
    case ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW:
        return DRM_IVPU_JOB_PRIORITY_IDLE;
    case ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH:
        return DRM_IVPU_JOB_PRIORITY_REALTIME;
    case ZE_COMMAND_QUEUE_PRIORITY_NORMAL:
    default:
        return DRM_IVPU_JOB_PRIORITY_NORMAL;
    }
}

inline int npu_sched_suspend_cmdq(ze_command_queue_priority_t priority)
{
    int fd = get_npu_fd();
    struct drm_ivpu_schedule params = {
        .engine = VPU_ENGINE_COMPUTE,
        .priority = get_driver_priority(priority),
        .operation = DRM_IVPU_SCHED_OP_CMDQ_SUSPEND,
        .request_id = 0,
        .ze_command_queue_id = 0,
    };

    return ioctl(fd, DRM_IOCTL_IVPU_SCHEDULE, &params);
}

inline int npu_sched_resume_cmdq(ze_command_queue_priority_t priority)
{
    int fd = get_npu_fd();
    struct drm_ivpu_schedule params = {
        .engine = VPU_ENGINE_COMPUTE,
        .priority = get_driver_priority(priority),
        .operation = DRM_IVPU_SCHED_OP_CMDQ_RESUME,
        .request_id = 0,
        .ze_command_queue_id = 0,
    };

    return ioctl(fd, DRM_IOCTL_IVPU_SCHEDULE, &params);
}

inline int npu_sched_preempt_engine(uint32_t request_id)
{
    int fd = get_npu_fd();
    struct drm_ivpu_schedule params = {
        .engine = VPU_ENGINE_COMPUTE,
        .priority = 0,
        .operation = DRM_IVPU_SCHED_OP_ENGINE_PREEMPT,
        .request_id = request_id,
        .ze_command_queue_id = 0,
    };

    return ioctl(fd, DRM_IOCTL_IVPU_SCHEDULE, &params);
}

inline int npu_sched_reset_engine()
{
    int fd = get_npu_fd();
    struct drm_ivpu_schedule params = {
        .engine = VPU_ENGINE_COMPUTE,
        .priority = 0,
        .operation = DRM_IVPU_SCHED_OP_ENGINE_RESET,
        .request_id = 0,
        .ze_command_queue_id = 0,
    };

    return ioctl(fd, DRM_IOCTL_IVPU_SCHEDULE, &params);
}

inline int npu_sched_resume_engine()
{
    int fd = get_npu_fd();
    struct drm_ivpu_schedule params = {
        .engine = VPU_ENGINE_COMPUTE,
        .priority = 0,
        .operation = DRM_IVPU_SCHED_OP_ENGINE_RESUME,
        .request_id = 0,
        .ze_command_queue_id = 0,
    };

    return ioctl(fd, DRM_IOCTL_IVPU_SCHEDULE, &params);
}
