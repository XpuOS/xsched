#pragma once

#include <mutex>
#include <vector>

namespace xsched::utils
{

class ObjectPool
{
public:
    ObjectPool(size_t pool_size = 512): kPoolSize(pool_size) { pool_.reserve(kPoolSize); }
    virtual ~ObjectPool() { for (auto obj : pool_) Destroy(obj); }
    
    void *Pop()
    {
        void *obj;
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            for (size_t i = 0; i < kPoolSize; ++i) {
                obj = Create();
                pool_.emplace_back(obj);
            }
            obj = Create();
        } else {
            obj = pool_.back();
            pool_.pop_back();
        }
        return obj;
    }

    void Push(void *obj)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.emplace_back(obj);
    }

protected:
    virtual void *Create() = 0;
    virtual void Destroy(void *) {}

    std::mutex mutex_;
    std::vector<void *> pool_;
    const size_t kPoolSize = 0;
};

} // namespace xsched::utils
