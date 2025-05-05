#pragma once

#include <cstdint>
#include <fstream>
#include <unistd.h>
#include <type_traits>
#include <sys/syscall.h>

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

#define EXPORT_C_FUNC   extern "C" __attribute__((visibility("default")))
#define EXPORT_CXX_FUNC __attribute__((visibility("default")))

#if defined(__i386__)
#define ARCH_STR "x86"
#elif defined(__x86_64__) || defined(_M_X64)
#define ARCH_STR "x86_64"
#elif defined(__arm__)
#define ARCH_STR "arm"
#elif defined(__aarch64__)
#define ARCH_STR "aarch64"
#endif

typedef int32_t TID;
typedef pid_t   PID;

inline TID GetThreadId()
{
    static const thread_local TID tid = syscall(SYS_gettid);
    return tid;
}

inline PID GetProcessId()
{
    static const PID pid = getpid();
    return pid;
}

inline bool FileExists(const std::string &path)
{
    std::ifstream file(path);
    bool exists = file.good();
    file.close();
    return exists;
}
