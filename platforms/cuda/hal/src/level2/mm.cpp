#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

ResizableBuffer::ResizableBuffer(CUdevice dev)
{
    CUdevice current_dev;
    CUDA_ASSERT(Driver::CtxGetDevice(&current_dev));
    XASSERT(current_dev == dev,
            "create resizable buffer failed: current device (%d) does not match device (%d)",
            current_dev, dev);

    prop_ = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = dev,
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
            .id = dev,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    // alloc vm space
    CUDA_ASSERT(Driver::MemAddressReserve(&ptr_, VM_DEFAULT_SIZE, 0, 0, 0));
    CUDA_ASSERT(Driver::MemGetAllocationGranularity(
        &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    size_ = ROUND_UP(BUFFER_DEFAULT_SIZE, granularity_);
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, size_, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size=size_, .handle=cu_handle});

    // map vm to pm
    CUDA_ASSERT(Driver::MemMap(ptr_, size_, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(ptr_, size_, &rw_desc_, 1));
}

ResizableBuffer::~ResizableBuffer()
{
    CUcontext ctx = nullptr; // check if cuda driver has deinitialized
    if (Driver::CtxGetCurrent(&ctx) == CUDA_ERROR_DEINITIALIZED) return;

    CUDA_ASSERT(Driver::MemUnmap(ptr_, size_));
    for (auto h : handles_) CUDA_ASSERT(Driver::MemRelease(h.handle));
    CUDA_ASSERT(Driver::MemAddressFree(ptr_, VM_DEFAULT_SIZE));
}

void ResizableBuffer::ExpandTo(size_t new_size, CUstream stream)
{
    if (new_size <= size_) return;
    if (new_size > VM_DEFAULT_SIZE) {
        XERRO("resizable buffer %p cannot be expanded to %zuB: exceeds max size of %ldB",
              (void *)ptr_, new_size, VM_DEFAULT_SIZE);
    }

    new_size = ROUND_UP(new_size, granularity_);
    XINFO("expanding buffer %p from %zuB to %zuB", (void *)ptr_, size_, new_size);
    size_t handle_size = new_size - size_;

    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, handle_size, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size = handle_size, .handle = cu_handle});

    // map new vm to new pm
    CUdeviceptr new_vm = ptr_ + size_;
    CUDA_ASSERT(Driver::MemMap(new_vm, handle_size, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(new_vm, handle_size, &rw_desc_, 1));

    // clear new buffer area
    CUDA_ASSERT(Driver::MemsetD8Async(new_vm, 0, handle_size, stream));
    CUDA_ASSERT(Driver::StreamSynchronize(stream));

    size_ = new_size;
}

InstrMemAllocator::InstrMemAllocator(CUcontext ctx, CUdevice dev): ctx_(ctx)
{
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx_,
            "create InstrMemAllocator failed: current context (%p) does not match context (%p)",
            current_ctx, ctx_);

    CUmemAllocationProp prop {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = dev,
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

    block_size_ = ROUND_UP(BUFFER_DEFAULT_SIZE, granularity_);
    blocks_.emplace_back(cuXtraInstrMemBlockAlloc(ctx_, block_size_));
}

InstrMemAllocator::~InstrMemAllocator()
{
    CUcontext ctx = nullptr; // check if cuda driver has deinitialized
    if (Driver::CtxGetCurrent(&ctx) == CUDA_ERROR_DEINITIALIZED) return;
    XASSERT(ctx == ctx_, "current context %p mismatch InstrMemAllocator context %p", ctx, ctx_);
    for (auto block : blocks_) cuXtraInstrMemBlockFree(ctx_, block);
}

CUdeviceptr InstrMemAllocator::Alloc(size_t size)
{
    std::lock_guard<std::mutex> lock(mtx_);
    size_t offset = used_size_;
    used_size_ += size;
    if (used_size_ <= block_size_) return blocks_.back() + offset;

    // alloc a new block
    CUdeviceptr block = cuXtraInstrMemBlockAlloc(ctx_, block_size_);
    XINFO("allocating new instruction memory block 0x%llx", block);
    blocks_.emplace_back(block);
    used_size_ = size;
    return block;
}
