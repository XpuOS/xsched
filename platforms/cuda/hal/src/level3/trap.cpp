#include <mutex>
#include <fstream>
#include <sstream>
#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/level3/trap.h"
#include "xsched/cuda/hal/level2/op_stream.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

#define TRAP_HANDLER_DEBUG    false

using namespace xsched::cuda;

TrapManager::TrapManager(CUcontext context)
    : kContext(context), kOpStream(OpStreamManager::GetOpStream(context))
{
    cuXtraGetTrapHandlerInfo(context, &trap_handler_dev_, &trap_handler_size_);
}

void TrapManager::SetTrapHandler()
{
    static std::mutex init_mtx;
    static bool initialized = false;

    std::lock_guard<std::mutex> lock(init_mtx);
    if (initialized) return;
    initialized = true;

#if TRAP_HANDLER_DEBUG
    DumpTrapHandler();
#endif

    static const size_t kExtraInstrsSize = 1024;

    char *trap_handler_host = (char *)malloc(trap_handler_size_);
    char *extra_instrs_host = (char *)malloc(kExtraInstrsSize);

    CUdeviceptr extra_instrs_device;
    CUDA_ASSERT(Driver::MemAllocAsync(&extra_instrs_device,
                                      kExtraInstrsSize,
                                      kOpStream));
    CUDA_ASSERT(Driver::StreamSynchronize(kOpStream));

    // copy trap handler instructions to host
    cuXtraMemcpyDtoH(trap_handler_host, trap_handler_dev_,
                     trap_handler_size_, kOpStream);

    InstrumentTrapHandler(trap_handler_host,
                          trap_handler_dev_,
                          trap_handler_size_,
                          extra_instrs_host,
                          extra_instrs_device,
                          kExtraInstrsSize);

    CUDA_ASSERT(Driver::MemcpyHtoDAsync_v2(extra_instrs_device,
                                          extra_instrs_host,
                                          kExtraInstrsSize,
                                          kOpStream));
    // copy trap handler instructions back to device
    cuXtraMemcpyHtoD(trap_handler_dev_, trap_handler_host,
                     trap_handler_size_, kOpStream);

    CUDA_ASSERT(Driver::StreamSynchronize(kOpStream));
    cuXtraInvalInstrCache(kContext);
    free(trap_handler_host);
    free(extra_instrs_host);
}

void TrapManager::InterruptContext()
{
    cuXtraTriggerTrap(kContext);
}

void TrapManager::DumpTrapHandler()
{
    printf("dumping trap handler...\n");

    char *trap_handler_host = (char *)malloc(trap_handler_size_);
    // copy trap handler instructions to host
    cuXtraMemcpyDtoH(trap_handler_host, trap_handler_dev_,
                     trap_handler_size_, kOpStream);

    std::stringstream filename;
    filename << "trap_handler_0x" << std::hex << trap_handler_dev_ << ".bin";
    std::ofstream out_file(filename.str(), std::ios::binary);
    out_file.write(trap_handler_host, trap_handler_size_);
    out_file.close();

    printf("dumped trap handler in %s\n", filename.str().c_str());
}

std::shared_ptr<TrapManager> TrapManager::GetTrapManager(CUcontext context)
{
    static std::mutex mgr_map_mtx;
    static std::map<CUcontext, std::shared_ptr<TrapManager>> mgr_map;

    std::lock_guard<std::mutex> lock(mgr_map_mtx);
    auto it = mgr_map.find(context);
    if (it != mgr_map.end()) return it->second;
    
    auto mgr = std::make_shared<TrapManager>(context);
    mgr_map[context] = mgr;
    return mgr;
}
