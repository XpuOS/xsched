#pragma once

#include "log.h"

#ifdef _WIN32
#include <string>
#include <windows.h>

inline std::string win_strerror(DWORD errnum)
{
    LPSTR buffer = nullptr;
    DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        errnum,
        0, // Default language (English)
        (LPSTR)&buffer,
        0,
        nullptr);

    if (length == 0) {
        return "Unknown error";
    }

    std::string message(buffer, length);
    LocalFree(buffer); // Free the allocated buffer
    return message;
}
#endif

inline ERRNO get_errno()
{
#ifdef _WIN32
    return GetLastError();
#else
    return errno;
#endif
}

inline const char* get_errno_str()
{
#ifdef _WIN32
    return win_strerror(get_errno()).c_str();
#else
    return strerror(get_errno());
#endif
}

#define XASSERT(expr, format, ...)                                                 \
    do {                                                                           \
        if (static_cast<bool>(expr)) {                                             \
            XERRO(format ": (%d)%s", ##__VA_ARGS__, get_errno(), get_errno_str()); \
        }                                                                          \
    } while (0)

#define XASSERT_RETURN(expr, ret, format, ...)                                     \
    do {                                                                           \
        if (static_cast<bool>(expr)) {                                             \
            XERRO(format ": (%d)%s", ##__VA_ARGS__, get_errno(), get_errno_str()); \
            return ret;                                                            \
        }                                                                          \
    } while (0)

#define XASSERT_EXIT(expr, format, ...)                                            \
    do {                                                                           \
        if (static_cast<bool>(expr)) {                                             \
            XERRO(format ": (%d)%s", ##__VA_ARGS__, get_errno(), get_errno_str()); \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)
