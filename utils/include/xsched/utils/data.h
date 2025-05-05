#pragma once

#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>

#include "xsched/utils/xassert.h"

namespace xsched::utils
{

template <typename DataType>
class DataProcessor
{
public:
    DataProcessor() { data_.reserve(kReserveSize); }
    ~DataProcessor() = default;

    void Add(const DataType& data)
    {
        sorted_ = false;
        sum_ += data;
        data_.push_back(data);
    }

    void Clear()
    {
        sorted_ = false;
        sum_ = 0;
        data_.clear();
    }

    void Organize()
    {
        if (sorted_) return;
        std::sort(data_.begin(), data_.end());
        sorted_ = true;
    }

    size_t Cnt() const
    {
        return data_.size();
    }

    DataType Sum() const
    {
        return sum_;
    }

    DataType Avg() const
    {
        if (data_.empty()) return 0;
        return sum_ / data_.size();
    }

    DataType Percentile(double p)
    {
        if (data_.empty()) return 0;

        this->Organize();
        size_t idx = p * data_.size();
        XASSERT(idx <= data_.size(), "Invalid percentile value: %f", p);
        return data_[idx];
    }

    void SaveCDF(const std::string &file)
    {
        this->Organize();
        std::ofstream ofs(file);
        for (size_t i = 0; i < data_.size(); ++i) {
            ofs << i / (double)data_.size() << " " << data_[i] << '\n';
        }
        ofs.close();
    }

    static const size_t kReserveSize = 1024;

private:
    bool sorted_ = false;
    DataType sum_ = 0;
    std::vector<DataType> data_;
};

} // namespace xsched::utils
