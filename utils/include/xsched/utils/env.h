#pragma once

#include <string>
#include <cstdint>
#include <cstdlib>

inline bool GetEnvInt64(const std::string &env_name, int64_t &val)
{
    char *value = std::getenv(env_name.c_str());
    if (value == nullptr) return false;
    try { val = std::stoll(value); } catch (...) { return false; }
    return true;
}
