#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/level2/op_stream.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

ResizableBuffer::ResizableBuffer()
{
    CUcontext context;
    CUDA_ASSERT(Driver::CtxGetCurrent(&context));
    op_stream_ = OpStreamManager::GetOpStream(context);

    CUdevice device;
    CUDA_ASSERT(Driver::CtxGetDevice(&device));
    prop_ = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .win32HandleMetaData = nullptr,
        .allocFlags = {
            .compressionType = 0,
            .gpuDirectRDMACapable = 0,
            .usage = 0,
            .reserved = {0},
        }
    };
    rw_desc_ = {
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    // alloc vm space
    CUDA_ASSERT(Driver::MemAddressReserve(&dev_ptr_, VM_DEFAULT_SIZE, 0, 0, 0));
    CUDA_ASSERT(Driver::MemGetAllocationGranularity(
        &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    size_ = ROUND_UP(BUFFER_DEFAULT_SIZE, granularity_);
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, size_, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size=size_, .handle=cu_handle});

    // map vm to pm
    CUDA_ASSERT(Driver::MemMap(dev_ptr_, size_, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(dev_ptr_, size_, &rw_desc_, 1));

    // clear buffer
    CUDA_ASSERT(Driver::MemsetD8Async(dev_ptr_, 0, size_, op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
}

ResizableBuffer::~ResizableBuffer()
{
    CUcontext context = nullptr; // check if cuda driver has deinitialized
    if (Driver::CtxGetCurrent(&context) == CUDA_ERROR_DEINITIALIZED) return;

    CUDA_ASSERT(Driver::MemUnmap(dev_ptr_, size_));
    for (auto h : handles_) CUDA_ASSERT(Driver::MemRelease(h.handle));
    CUDA_ASSERT(Driver::MemAddressFree(dev_ptr_, VM_DEFAULT_SIZE));
}

void ResizableBuffer::ExpandTo(size_t new_size)
{
    if (new_size <= size_) return;
    if (new_size > VM_DEFAULT_SIZE) {
        XERRO("resizable buffer %p cannot be expanded to %ldB: "
              "exceeds max size of %ldB",
              (void *)dev_ptr_, new_size, VM_DEFAULT_SIZE);
    }

    new_size = ROUND_UP(new_size, granularity_);
    XINFO("expanding buffer %p from %ldB to %ldB",
          (void *)dev_ptr_, size_, new_size);
    size_t handle_size = new_size - size_;

    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, handle_size, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size = handle_size,
                                           .handle = cu_handle});

    // map new vm to new pm
    CUdeviceptr new_vm = dev_ptr_ + size_;
    CUDA_ASSERT(Driver::MemMap(new_vm, handle_size, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(new_vm, handle_size, &rw_desc_, 1));

    // clear new buffer area
    CUDA_ASSERT(Driver::MemsetD8Async(new_vm, 0, handle_size, op_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));

    size_ = new_size;
}

InstrMemAllocator::InstrMemAllocator()
{
    CUDA_ASSERT(Driver::CtxGetDevice(&device_));
    CUDA_ASSERT(Driver::CtxGetCurrent(&context_));
    op_stream_ = OpStreamManager::GetOpStream(context_);

    CUmemAllocationProp prop {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device_,
        },
        .win32HandleMetaData = nullptr,
        .allocFlags = {
            .compressionType = 0,
            .gpuDirectRDMACapable = 0,
            .usage = 0,
            .reserved = {0},
        }
    };
    CUDA_ASSERT(Driver::MemGetAllocationGranularity(
        &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    total_size_ = ROUND_UP(BUFFER_DEFAULT_SIZE, granularity_);
    block_base_ = cuXtraInstrMemBlockAlloc(context_, total_size_);
}

InstrMemAllocator::~InstrMemAllocator()
{
    CUcontext context = nullptr; // check if cuda driver has deinitialized
    if (Driver::CtxGetCurrent(&context) == CUDA_ERROR_DEINITIALIZED) return;
    XASSERT(context == context_,
            "current context %p and instruction mem context %p mismatch",
            context, context_);
    cuXtraInstrMemBlockFree(context_, block_base_);
}

CUdeviceptr InstrMemAllocator::Alloc(size_t size)
{
    std::lock_guard<std::mutex> lock(mtx_);

    size_t offset = used_size_;
    used_size_ += size;
    if (used_size_ <= total_size_) return block_base_ + offset;

    XERRO("expand not supported yet");

    // the buffer needs to be expanded
    size_t old_size = total_size_;
    total_size_ = ROUND_UP(used_size_, granularity_);

    // alloc new instruction memory block
    CUdeviceptr old_base = block_base_;
    block_base_ = cuXtraInstrMemBlockAlloc(context_, total_size_);
    // copy old data to new block
    cuXtraInstrMemcpyDtoD(block_base_, old_base, old_size, op_stream_);
    cuXtraInstrMemBlockFree(context_, old_base);

    return block_base_ + offset;
}
