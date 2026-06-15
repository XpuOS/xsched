#pragma once

#include <cerrno>
#include <string>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <filesystem>
#include <type_traits>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/syscall.h>
#endif

// branch prediction optimization
#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

// forbidden class operation macros
#define NO_COPY_CLASS(TypeName) \
    TypeName(const TypeName &) = delete; \
    TypeName& operator=(const TypeName &) = delete;

#define NO_MOVE_CLASS(TypeName) \
    TypeName(TypeName &&) = delete; \
    TypeName& operator=(TypeName &&) = delete;

#define STATIC_CLASS(TypeName) \
    TypeName() = delete; \
    ~TypeName() = delete; \
    NO_COPY_CLASS(TypeName) \
    NO_MOVE_CLASS(TypeName)

#define UNFOLD(...) __VA_ARGS__
#define UNUSED(expr) do { (void)(expr); } while (0)

#define ROUND_DOWN(X, ALIGN) ((X) / (ALIGN) * (ALIGN))
#define ROUND_UP(X, ALIGN)   ((((X) + (ALIGN) - 1) / (ALIGN)) * (ALIGN))

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

// DLL export/import definitions
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

// architecture detection
#if defined(__i386__) || defined(_M_IX86)
    #define ARCH_X86
    #define ARCH_STR "x86"
#elif defined(__x86_64__) || defined(_M_X64)
    #define ARCH_X86_64
    #define ARCH_STR "x86_64"
#elif defined(__arm__) || defined(_M_ARM)
    #define ARCH_ARM
    #define ARCH_STR "arm"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_AARCH64
    #define ARCH_STR "aarch64"
#endif

// operating system detection
#if defined(_WIN32)
    #define OS_STR "Windows"
#elif defined(__linux__)
    #define OS_STR "Linux"
#else
    #error unsupported OS
#endif

#define FMT_32D "%" PRId32
#define FMT_32U "%" PRIu32
#define FMT_32X "%" PRIx32
#define FMT_64D "%" PRId64
#define FMT_64U "%" PRIu64
#define FMT_64X "%" PRIx64

#if defined(_WIN32)
    typedef uint32_t      TID;
    typedef uint32_t      PID;
    typedef uint32_t      ERRNO;
    #define FMT_TID       FMT_32U
    #define FMT_PID       FMT_32U
    #define FMT_ERRNO     FMT_32U
#elif defined(__linux__)
    typedef pid_t         TID;
    typedef pid_t         PID;
    typedef int32_t       ERRNO;
    #define FMT_TID       FMT_32D
    #define FMT_PID       FMT_32D
    #define FMT_ERRNO     FMT_32D
#endif

// ----------------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------------

inline TID GetThreadId()
{
#if defined(_WIN32)
    static const thread_local TID tid = GetCurrentThreadId();
#elif defined(__linux__)
    /// FIXME: also buggy for forked processes
    static const thread_local TID tid = (TID)syscall(SYS_gettid);
#endif
    return tid;
}

inline PID GetProcessId()
{
#if defined(_WIN32)
    static const PID pid = GetCurrentProcessId();
#elif defined(__linux__)
    const PID pid = getpid(); // no static, for forked processes
#endif
    return pid;
}

inline bool FileExists(const std::string &path)
{
    // using std::error_code to avoid throwing exceptions on permission errors
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

inline ERRNO GetErrNo()
{
#if defined(_WIN32)
    return GetLastError();
#else
    return errno;
#endif
}

inline std::string GetErrStr(ERRNO err_no = 0)
{
#if defined(_WIN32)
    if (err_no == 0) err_no = GetLastError();
    LPSTR buffer = nullptr;
    DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err_no,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // use system default language
        (LPSTR)&buffer, 0, nullptr);

    if (length == 0) return "unknown error";
    std::string message(buffer, length);
    LocalFree(buffer);

    // remove trailing newline characters
    while (!message.empty() && (message.back() == '\r' || message.back() == '\n')) {
        message.pop_back();
    }
    return message;
#else
    if (err_no == 0) err_no = errno;
    char buf[512] = {0};
    auto res = strerror_r(err_no, buf, sizeof(buf));
    // strerror_r has two versions: GNU and XSI
    // GNU version returns char*, XSI version returns int
    if constexpr (std::is_same_v<decltype(res), char *>) {
        return std::string(res);
    } else {
        if (res != 0) return "unknown error";
        return std::string(buf);
    }
#endif
}
