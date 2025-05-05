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
    ResizableBuffer();
    virtual ~ResizableBuffer();

    size_t      Size()   const { return size_; }
    CUdeviceptr DevPtr() const { return dev_ptr_; }

    void ExpandTo(size_t new_size);

private:
    size_t size_;
    CUdeviceptr dev_ptr_;
    CUstream op_stream_;

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
    InstrMemAllocator();
    virtual ~InstrMemAllocator();

    /// @brief Allocate an instruction device memory block of input size.
    /// @param size the size of the instruction device memory block
    /// @return the virtual address of the instruction device memory block
    CUdeviceptr Alloc(size_t size);

private:
    std::mutex mtx_;
    size_t used_size_ = 0;
    size_t total_size_;
    size_t granularity_;

    CUdevice device_;
    CUcontext context_;
    CUstream op_stream_;
    CUdeviceptr block_base_;
};

} // namespace xsched::cuda
