#pragma once

#include <fstream>
#include <cstdint>
#include <cinttypes>
#include <type_traits>

#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#include <shlwapi.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define NO_COPY_CLASS(TypeName) \
    TypeName(const TypeName &) = delete; \
    void operator=(const TypeName &) = delete;

#define NO_MOVE_CLASS(TypeName) \
    TypeName(TypeName &&) = delete; \
    void operator=(TypeName &&) = delete;

#define STATIC_CLASS(TypeName) \
    TypeName() = default; \
    ~TypeName() = default; \
    NO_COPY_CLASS(TypeName) \
    NO_MOVE_CLASS(TypeName)

#define UNFOLD(...) __VA_ARGS__
#define UNUSED(expr) do { (void)(expr); } while (0)

#define ROUND_UP(X, ALIGN) (((X) - 1) / (ALIGN) + 1) * (ALIGN)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Windows' dynamic link libraries (DLLs) use __declspec(dllexport) and __declspec(dllimport)
// while __attribute__(visibility("default")) is mainly used for ELF format (Linux/macOS)
#if defined(_WIN32)
    #ifdef __cplusplus
        #define EXPORT_C_FUNC   extern "C" __declspec(dllexport)
        #define EXPORT_CXX_FUNC __declspec(dllexport)
    #else
        #define EXPORT_C_FUNC   __declspec(dllexport)
    #endif
#elif defined(__linux__)
    #ifdef __cplusplus
        #define EXPORT_C_FUNC   extern "C" __attribute__((visibility("default")))
        #define EXPORT_CXX_FUNC __attribute__((visibility("default")))
    #else
        #define EXPORT_C_FUNC   __attribute__((visibility("default")))
    #endif
#endif

#if defined(__i386__)
    #define ARCH_X86
    #define ARCH_STR "x86"
#elif defined(__x86_64__) || defined(_M_X64)
    #define ARCH_X86_64
    #define ARCH_STR "x86_64"
#elif defined(__arm__)
    #define ARCH_ARM
    #define ARCH_STR "arm"
#elif defined(__aarch64__)
    #define ARCH_AARCH64
    #define ARCH_STR "aarch64"
#endif

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

#if defined(_WIN32)
    typedef int64_t       TID;
    typedef uint32_t      PID;
    #define FMT_TID       FMT_64D
    #define FMT_PID       FMT_32U
#elif defined(__linux__)
    typedef int32_t       TID;
    typedef pid_t         PID;
    #define FMT_TID       FMT_32D
    #define FMT_PID       FMT_32D
#endif

inline TID GetThreadId()
{
#if defined(_WIN32)
    static const thread_local TID tid = GetCurrentThreadId();
#elif defined(__linux__)
    static const thread_local TID tid = syscall(SYS_gettid);
#endif
    return tid;
}

inline PID GetProcessId()
{
#if defined(_WIN32)
    static const PID pid = GetCurrentProcessId();
#elif defined(__linux__)
    static const PID pid = getpid();
#endif
    return pid;
}

inline bool FileExists(const std::string &path)
{
#if defined(_WIN32)
    return PathFileExistsA(path.c_str());
#endif
    std::ifstream file(path);
    bool exists = file.good();
    file.close();
    return exists;
}
