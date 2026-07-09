#pragma once

#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <type_traits>
#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define UNFOLD(...) __VA_ARGS__
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

#if defined(_WIN32)
#define OS_STR "Windows"
#elif defined(__linux__)
#define OS_STR "Linux"
#endif

#define FMT_32D "%" PRId32
#define FMT_32U "%" PRIu32
#define FMT_32X "%" PRIx32
#define FMT_64D "%" PRId64
#define FMT_64U "%" PRIu64
#define FMT_64X "%" PRIx64

#ifdef _WIN32
typedef int64_t TID;
typedef uint32_t PID;
typedef uint32_t ERRNO;
#define FMT_TID FMT_64D
#define FMT_PID FMT_32U
#define FMT_ERRNO FMT_32U
#else
typedef int32_t TID;
typedef pid_t PID;
typedef int32_t ERRNO;
#define FMT_TID FMT_32D
#define FMT_PID FMT_32D
#define FMT_ERRNO FMT_32D
#endif

inline TID GetThreadId()
{
#ifdef _WIN32
    static const thread_local TID tid = GetCurrentThreadId();
#else
    static const thread_local TID tid = syscall(SYS_gettid);
#endif
    return tid;
}