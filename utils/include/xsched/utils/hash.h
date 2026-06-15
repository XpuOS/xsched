#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>

inline uint64_t HashDjb2(const char *buf, size_t len)
{
    uint64_t hash = 5381;
    while (len--)
        hash = ((hash << 5) + hash) + (tolower(*buf++)); /* hash * 33 + c */
    return hash;
}

inline uint64_t HashDjb2(const char *str)
{
    uint64_t hash = 5381;
    while (true) {
        uint32_t c = tolower(*str++);
        if (c == 0) break;
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}
