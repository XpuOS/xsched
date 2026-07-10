#pragma once

#include <string>
#include <cstring>
#include <cstdint>
#include <cstdlib>

inline std::string GetEnv(const std::string &env_name)
{
#if defined(_WIN32)
    char *env = nullptr;
    size_t len = 0;
    _dupenv_s(&env, &len, env_name.c_str());
    if (env == nullptr) return "";
    std::string env_str(env);
    free(env);
    return env_str;
#else
    char *env = std::getenv(env_name.c_str());
    if (env == nullptr) return "";
    return std::string(env);
#endif
}

inline bool GetEnvInt64(const std::string &env_name, int64_t &val)
{
    std::string env = GetEnv(env_name);
    if (env.empty()) return false;
    try { val = std::stoll(env); } catch (...) { return false; }
    return true;
}

inline bool GetEnvOption(const std::string &env_name, bool default_val = false)
{
    std::string env = GetEnv(env_name);
    if (env == "0" || strcasecmp(env.c_str(), "off") == 0) return false;
    if (env == "1" || strcasecmp(env.c_str(), "on") == 0) return true;
    return default_val;
}
