#pragma once

#include <string>
#include <sstream>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <algorithm>

inline std::string ToHex(uint64_t x)
{
    std::stringstream ss;
    ss << "0x" << std::hex << x;
    return ss.str();
}

inline std::string ToLower(const std::string &str)
{
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower;
}

/// @brief Shrink a string to a given max size. Shrink from the middle.
/// The shrinked string will be in the form of "prefix...suffix".
/// E.g., ShrinkString("0123456789", 5) = "0...9"
/// @param str The string to shrink.
/// @param max_len The max length of the shrinked string.
/// @return The shrinked string.
inline std::string ShrinkString(const std::string &str, size_t max_len)
{
    if (max_len <= 0) return "";
    if (str.length() <= max_len) return str;
    if (max_len <= 3) return std::string("...").substr(0, max_len);
    const size_t keep_front = (max_len - 3 + 1) / 2;
    const size_t keep_back = max_len - keep_front - 3;
    return str.substr(0, keep_front) + "..." + str.substr(str.length() - keep_back);
}
