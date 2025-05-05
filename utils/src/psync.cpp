#include <cstring>

#include "xsched/utils/log.h"
#include "xsched/utils/psync.h"
#include "xsched/utils/xassert.h"

using namespace xsched::utils;

#define barrier()            (__sync_synchronize())
#define AO_GET(ptr)          ({ __typeof__(*(ptr)) volatile *_val = (ptr); barrier(); (*_val); })
#define AO_ADD_F(ptr, value) ((__typeof__(*(ptr)))__sync_add_and_fetch((ptr), (value)))
#define AO_ADD(ptr, val)     ((void)AO_ADD_F((ptr), (val)))

ProcessSync::ProcessSync()
{
    bool is_master = false;
    shm_id_ = shmget(kShmKey, kSharedMemSize, 0666);
    if (shm_id_ == -1) {
        is_master = true;
        shm_id_ = shmget(kShmKey, kSharedMemSize, IPC_CREAT | 0666);
    }
    XASSERT(shm_id_ != -1, "fail to create shared memory\n");
    shm_ptr_ = (int *)shmat(shm_id_, nullptr, 0);
    XASSERT(shm_ptr_, "fail to get shared memory\n");
    if (is_master) {
        memset((void *)shm_ptr_, 0, kSharedMemSize);
    }
}

ProcessSync::~ProcessSync()
{
    shmdt((void *)shm_ptr_);
    shmctl(shm_id_, IPC_RMID, 0);
}

int ProcessSync::GetCnt()
{
    return AO_GET(shm_ptr_);
}

void ProcessSync::Sync(int expected_cnt)
{
    AO_ADD(shm_ptr_, 1);
    while (AO_GET(shm_ptr_) != expected_cnt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void ProcessSync::Sync(int expected_cnt, const char *client_name)
{
    XINFO("psync %s ready, waiting others", client_name);
    Sync(expected_cnt);
    XINFO("psync %s done", client_name);
}
