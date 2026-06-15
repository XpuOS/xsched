#pragma once

#include <ctime>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cstring>

#if defined(_WIN32)
    #include <io.h>
#else
    #include <unistd.h>
#endif

#include "xsched/utils/env.h"
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

#define ANSI_COLOR_ERRO    "\x1b[31m"  // red
#define ANSI_COLOR_WARN    "\x1b[33m"  // yellow
#define ANSI_COLOR_INFO    "\x1b[32m"  // green
#define ANSI_COLOR_DEBG    "\x1b[36m"  // blue
#define ANSI_COLOR_RESET   "\x1b[0m"

inline int GetLogLevelFromEnv()
{
    const std::string level = GetEnv("XLOG_LEVEL");
    if (level.empty())   return LOG_LEVEL_DEBG;
    if (level == "ERRO") return LOG_LEVEL_ERRO;
    if (level == "WARN") return LOG_LEVEL_WARN;
    if (level == "INFO") return LOG_LEVEL_INFO;
    if (level == "DEBG") return LOG_LEVEL_DEBG;
    // default log level is INFO, log greater than INFO will be ignored
    return LOG_LEVEL_INFO;
}

inline int GetLogLevel()
{
    static const int level = GetLogLevelFromEnv();
    return level;
}

inline void GetLocalTime(const time_t tt, std::tm &lt)
{
#if defined(_WIN32)
    localtime_s(&lt, &tt);
#else
    localtime_r(&tt, &lt);
#endif
}

inline bool IsTerminal(FILE *fd)
{
#if defined(_WIN32)
    const static bool stdout_is_tty = _fileno(stdout) >= 0 && _isatty(_fileno(stdout));
    const static bool stderr_is_tty = _fileno(stderr) >= 0 && _isatty(_fileno(stderr));
#else
    const static bool stdout_is_tty = isatty(fileno(stdout)) != 0;
    const static bool stderr_is_tty = isatty(fileno(stderr)) != 0;
#endif
    return fd == stderr ? stderr_is_tty : (fd == stdout ? stdout_is_tty : false);
}

#define XLOG_HELPER(level, level_str, level_color, format, ...) \
    do { \
        if (level > GetLogLevel()) break;                              \
        const auto now = std::chrono::system_clock::now();             \
        const auto now_tt = std::chrono::system_clock::to_time_t(now); \
        std::tm now_lt{};                                              \
        GetLocalTime(now_tt, now_lt);                                  \
        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>   \
                            (now.time_since_epoch()).count() % 1000000;             \
        fprintf(XLOG_FD, "[XSCHED "                                                 \
                "%s" level_str "%s"                                                 \
                " @ T " FMT_TID " @ %02d:%02d:%02d.%06" PRId64 "] " format "\n",    \
                IsTerminal(XLOG_FD) ? level_color      : "",                        \
                IsTerminal(XLOG_FD) ? ANSI_COLOR_RESET : "",                        \
                GetThreadId(), now_lt.tm_hour, now_lt.tm_min, now_lt.tm_sec, now_us \
                __VA_OPT__(,) __VA_ARGS__);                                         \
        FLUSH_XLOG_IF_DEBG();                                                       \
    } while (0);

// first unfold the arguments, then unfold XLOG
#define XLOG(level, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD(( \
        CONCAT(LOG_LEVEL_, level), TOSTRING(level), CONCAT(ANSI_COLOR_, level), \
        format __VA_OPT__(,) __VA_ARGS__ \
    )))

#define XLOG_WITH_CODE(level, format, ...) \
    UNFOLD(XLOG_HELPER UNFOLD(( \
        CONCAT(LOG_LEVEL_, level), TOSTRING(level), CONCAT(ANSI_COLOR_, level), \
        format " @ %s:%d" __VA_OPT__(,) __VA_ARGS__, __FILE__, __LINE__ \
    )))

#ifdef RELEASE_MODE
#define XDEBG(format, ...)
#define XINFO(format, ...) XLOG(INFO, format __VA_OPT__(,) __VA_ARGS__)
#else
#define XDEBG(format, ...) XLOG_WITH_CODE(DEBG, format __VA_OPT__(,) __VA_ARGS__)
#define XINFO(format, ...) XLOG_WITH_CODE(INFO, format __VA_OPT__(,) __VA_ARGS__)
#endif

#define XWARN(format, ...)        XLOG_WITH_CODE(WARN, format __VA_OPT__(,) __VA_ARGS__)
#define XERRO_NOEXIT(format, ...) XLOG_WITH_CODE(ERRO, format __VA_OPT__(,) __VA_ARGS__)
#define XERRO(format, ...) \
    do { \
        XLOG_WITH_CODE(ERRO, format __VA_OPT__(,) __VA_ARGS__) \
        FLUSH_XLOG();       \
        exit(EXIT_FAILURE); \
    } while (0);

#define XERRO_UNSUPPORTED() XERRO("%s is not supported on %s", __func__, OS_STR);
