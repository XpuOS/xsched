#pragma once

#include <mutex>
#include <functional>
#include <unordered_map>

namespace xsched::utils
{

template <typename Key, typename Value>
class ObjectMap
{
public:
    ObjectMap() = default;
    virtual ~ObjectMap() = default;

    void Add(Key key, Value value)
    {
        mutex_.lock();
        map_[key] = value;
        mutex_.unlock();
    }

    Value Del(Key key, Value not_found)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it == map_.end()) return not_found;
        Value val = it->second;
        map_.erase(it);
        return val;
    }

    Value DoThenDel(Key key, Value not_found, std::function<void(Value)> func)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it == map_.end()) return not_found;
        Value val = it->second;
        func(val);
        map_.erase(it);
        return val;
    }

    Value Get(Key key, Value not_found)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        return not_found;
    }

private:
    std::mutex mutex_;
    std::unordered_map<Key, Value> map_;
};

} // namespace xsched::utils
