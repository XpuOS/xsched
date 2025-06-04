#pragma once

#include <list>
#include <mutex>
#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/cuda.h"

#define VM_DEFAULT_SIZE     (1UL << 30)     // 1G
#define BUFFER_DEFAULT_SIZE (16UL << 20)    // 16M

namespace xsched::cuda
{

class ResizableBuffer
{
public:
    ResizableBuffer(CUdevice dev);
    virtual ~ResizableBuffer();
    void ExpandTo(size_t new_size, CUstream stream);
    size_t     Size() const { return size_; }
    CUdeviceptr Ptr() const { return ptr_; }

private:
    size_t size_;
    CUdeviceptr ptr_;
    size_t granularity_;
    CUmemAccessDesc rw_desc_;
    CUmemAllocationProp prop_;
    struct AllocationHandle
    {
        size_t size;
        CUmemGenericAllocationHandle handle;
    };
    std::list<AllocationHandle> handles_;
};

class InstrMemAllocator
{
public:
    InstrMemAllocator(CUcontext ctx, CUdevice dev);
    virtual ~InstrMemAllocator();

    /// @brief Allocate an instruction device memory block of input size.
    /// @param size the size of the instruction device memory block
    /// @return the virtual address of the instruction device memory block
    CUdeviceptr Alloc(size_t size);

private:
    size_t used_size_ = 0;
    std::list<CUdeviceptr> blocks_;

    std::mutex mtx_;
    CUcontext ctx_;
    size_t granularity_;
    size_t block_size_;
};

} // namespace xsched::cuda
