#pragma once

#include <map>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <functional>
#include "xsched/utils/xassert.h"

namespace xsched::utils
{

class Region
{
public:
    Region() = default;
    ~Region() = default;

    void Merge(uint64_t begin, uint64_t end)
    {
        if (begin == end) return;
        if (begin > end) std::swap(begin, end);

        auto start_it = regions_.lower_bound(begin);
        if (start_it == regions_.end()) {
            // should insert at the end
            regions_[end] = begin;
            return;
        }

        // found merge start iterator:
        //        begin
        //          |-- to merge ---------
        //      |-- start_it --|
        // begin_start      end_start
        auto begin_start = start_it->second;
        begin = std::min(begin, begin_start);

        auto stop_it = start_it;
        for (; stop_it != regions_.end(); ++stop_it) {
            // find merge stop iterator
            auto end_stop = stop_it->first;
            auto begin_stop = stop_it->second;
            if (end < begin_stop) {
                // found merge stop iterator:
                //                  end
                //  ----- to merge --|
                //                      |-- stop_it --|
                //                 begin_stop
                break;
            }
            //                  end
            //  ----- to merge --|
            //          |-- stop_it --|
            //     begin_stop      end_stop
            end = std::max(end, end_stop);
        }

        //    begin                              end
        //      |        |----- to merge -----|   |
        //      |-- start_it --|    |---- xxx ----|    |-- stop_it --|
        // begin_start      end_start             begin_stop      end_stop
        stop_it = regions_.erase(start_it, stop_it);
        regions_.emplace_hint(stop_it, end, begin);
    }

    void Merge(const Region &other)
    {
        if (other.regions_.empty()) return;
        if (regions_.empty()) {
            regions_ = other.regions_;
            return;
        }

        bool have = false; // if we are having a region in merging
        uint64_t cur_begin = 0, cur_end = 0; // the current region in merging
        std::map<uint64_t, uint64_t> merged;
        auto push = [&](std::map<uint64_t, uint64_t>::const_iterator it) {
            // it is the region to merge
            uint64_t b = it->second;
            uint64_t e = it->first;
            if (!have) {
                have = true;
                cur_begin = b;
                cur_end = e;
                return;
            }
            if (b <= cur_end) {
                // the region to merge has intersection with the current region
                cur_end = std::max(cur_end, e);
            } else {
                // no intersection, push the current region to the result
                // and start a new region in merging
                merged[cur_end] = cur_begin;
                cur_begin = b;
                cur_end = e;
            }
        };

        // two-way scan
        auto it1 = regions_.begin();
        auto it2 = other.regions_.begin();
        while (it1 != regions_.end() && it2 != other.regions_.end()) {
            // merge the region with the smaller begin first
            if (it1->second < it2->second) push(it1++);
            else push(it2++);
        }
        while (it1 != regions_.end()) push(it1++);
        while (it2 != other.regions_.end()) push(it2++);
        if (have) merged[cur_end] = cur_begin;
        regions_ = std::move(merged);
    }

    void Intersect(const Region &other)
    {
        if (other.regions_.empty()) {
            regions_.clear();
            return;
        }

        // it is O(n*log(m)), so let the smaller be n
        bool this_smaller = regions_.size() < other.regions_.size();
        const auto &smaller = this_smaller ? this->regions_ : other.regions_;
        const auto &larger = this_smaller ? other.regions_ : this->regions_;

        std::map<uint64_t, uint64_t> result;
        // check every region in current regions_
        for (const auto [end, begin] : smaller) {
            // find the first region in the larger regions whose end > begin
            auto it = larger.upper_bound(begin);

            for (; it != larger.end(); ++it) {
                uint64_t cur_begin = it->second;
                uint64_t cur_end = it->first;

                // there is no intersection
                if (cur_begin >= end) break;

                // upper_bound ensures cur_end > begin, and cur_begin < end here,
                // so [begin, end) and [cur_begin, cur_end) must have intersection
                uint64_t intersect_begin = std::max(begin, cur_begin);
                uint64_t intersect_end = std::min(end, cur_end);
                XASSERT(intersect_begin < intersect_end,
                        "[%lu, %lu] must intersect with [%lu, %lu]",
                        begin, end, cur_begin, cur_end);
                result[intersect_end] = intersect_begin;

                // if the region is fully covered by the current region,
                // no need to move to the next node and check the next region
                if (cur_end >= end) break;
            }
        }
        regions_ = std::move(result);
    }

    void ForEach(std::function<void (uint64_t begin, uint64_t end)> func) const
    {
        for (const auto [end, begin] : regions_) {
            func(begin, end);
        }
    }

    size_t Size() const
    {
        size_t size = 0;
        for (const auto [end, begin] : regions_) {
            size += (end - begin);
        }
        return size;
    }

    size_t RegionCount() const
    {
        return regions_.size();
    }

    void Clear()
    {
        regions_.clear();
    }

private:
    // end => begin
    std::map<uint64_t, uint64_t> regions_;
};

} // namespace xsched::utils
