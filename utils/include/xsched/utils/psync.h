#pragma once

#include <thread>
#include <string>
#include <sys/ipc.h>
#include <sys/shm.h>

namespace xsched::utils
{

class ProcessSync
{
public:
    ProcessSync();
    ~ProcessSync();

    int GetCnt();
    void Sync(int expected_cnt);
    void Sync(int expected_cnt, const char *client_name);

private:
    int shm_id_;
    int *shm_ptr_;

    const key_t kShmKey = 0xbeef;
    const size_t kSharedMemSize = 16 * sizeof(int);
};
    
} // namespace xsched::utils
