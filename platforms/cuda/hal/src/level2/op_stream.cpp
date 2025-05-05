#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/level2/op_stream.h"

using namespace xsched::cuda;

CUstream OpStreamManager::GetOpStream(CUcontext context)
{
    static std::mutex stream_map_mtx;
    static std::map<CUcontext, CUstream> stream_map;

    std::lock_guard<std::mutex> lock(stream_map_mtx);
    auto it = stream_map.find(context);
    if (it != stream_map.end()) return it->second;
    
    int lp, hp;
    CUstream op_stream;
    CUDA_ASSERT(Driver::CtxGetStreamPriorityRange(&lp, &hp));
    // create operation stream with highest priority
    CUDA_ASSERT(Driver::StreamCreateWithPriority(&op_stream, 0, hp));
    stream_map[context] = op_stream;
    return op_stream;
}
