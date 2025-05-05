#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cstring>

#include "xsched/utils/common.h"

#define XLOG_FD stderr

#define FLUSH_XLOG() do { fflush(XLOG_FD); } while (0);

#ifdef RELEASE_MODE
#define FLUSH_XLOG_IF_DEBG()
#else
#define FLUSH_XLOG_IF_DEBG() FLUSH_XLOG()
#endif

#define LOG_LEVEL_ERRO  0
#define LOG_LEVEL_WARN  1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_DEBG  3

inline int GetLogLevelFromEnv()
{
    char *level = std::getenv("XLOG_LEVEL");
    if (level == nullptr)           return LOG_LEVEL_DEBG;
    if (strcmp(level, "ERRO") == 0) return LOG_LEVEL_ERRO;
    if (strcmp(level, "WARN") == 0) return LOG_LEVEL_WARN;
    if (strcmp(level, "INFO") == 0) return LOG_LEVEL_INFO;
    if (strcmp(level, "DEBG") == 0) return LOG_LEVEL_DEBG;
    return LOG_LEVEL_INFO; // default log level is INFO
}

inline int GetLogLevel()
{
    static const int level = GetLogLevelFromEnv();
    return level;
}

#define XLOG_HELPER(level, level_str, format, ...) \
    do { \
        if (level > GetLogLevel()) break; \
        const auto now = std::chrono::system_clock::now();             \
        const auto now_tt = std::chrono::system_clock::to_time_t(now); \
        const auto now_lt = std::localtime(&now_tt);                   \
        const auto now_us = \
            std::chrono::duration_cast<std::chrono::microseconds>          \
                (now.time_since_epoch()).count() % 1000000;                \
        fprintf(XLOG_FD, "[%s @ T%d @ %02d:%02d:%02d.%06ld] " format "\n", \
                level_str, GetThreadId(),                                  \
                now_lt->tm_hour, now_lt->tm_min, now_lt->tm_sec, now_us,   \
                ##__VA_ARGS__); \
        FLUSH_XLOG_IF_DEBG();   \
    } while (0);

// first unfold the arguments, then unfold XLOG
#define XLOG(level, level_str, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD((level, level_str, format, ##__VA_ARGS__)))

#define XLOG_WITH_CODE(level, level_str, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD((level, level_str, format " @ %s:%d", \
           ##__VA_ARGS__, __FILE__, __LINE__)))

#ifdef RELEASE_MODE
#define XDEBG(format, ...)
#define XINFO(format, ...) XLOG(LOG_LEVEL_INFO, "INFO", format, ##__VA_ARGS__)
#else
#define XDEBG(format, ...) XLOG_WITH_CODE(LOG_LEVEL_DEBG, "DEBG", format, ##__VA_ARGS__)
#define XINFO(format, ...) XLOG_WITH_CODE(LOG_LEVEL_INFO, "INFO", format, ##__VA_ARGS__)
#endif

#define XWARN(format, ...) XLOG_WITH_CODE(LOG_LEVEL_WARN, "WARN", format, ##__VA_ARGS__)
#define XERRO(format, ...) \
    do { \
        XLOG_WITH_CODE(LOG_LEVEL_ERRO, "ERRO", format, ##__VA_ARGS__) \
        FLUSH_XLOG();       \
        exit(EXIT_FAILURE); \
    } while (0);
