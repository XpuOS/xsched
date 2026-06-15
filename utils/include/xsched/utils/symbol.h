#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "xsched/utils/lib.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"

#if defined(__linux__)

#include <dlfcn.h>

#ifndef RTLD_DEEPBIND
    #define XSCHED_RTLD_FLAGS (RTLD_NOW | RTLD_LOCAL)
#else
    #define XSCHED_RTLD_FLAGS (RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND)
#endif

inline void *GetRealDlSym()
{
    Dl_info info;
    XASSERT(dladdr((void *)dlvsym, &info) != 0, "dladdr() failed to get info of dlvsym");
    void *handle = dlopen(info.dli_fname, XSCHED_RTLD_FLAGS);
    if (handle == nullptr) {
        XERRO("fail to dlopen %s for dlsym", info.dli_fname);
        return nullptr;
    }

    // try GLIBC_2.y (y: 0-50)
    for (int y = 50; y >= 0; y--) {
        std::string version = "GLIBC_2." + std::to_string(y);
        void *sym = dlvsym(handle, "dlsym", version.c_str());
        if (sym != nullptr) {
            XDEBG("found dlsym@GLIBC_2.%d in %s", y, info.dli_fname);
            return sym;
        }
    }

    // try GLIBC_2.y.z (z: 0-10)
    for (int y = 50; y >= 0; y--) {
        for (int z = 10; z >= 0; z--) {
            std::string version = "GLIBC_2." + std::to_string(y) + "." + std::to_string(z);
            void *sym = dlvsym(handle, "dlsym", version.c_str());
            if (sym != nullptr) {
                XDEBG("found dlsym@GLIBC_2.%d.%d in %s", y, z, info.dli_fname);
                return sym;
            }
        }
    }

    XERRO("fail to get real dlsym");
    return nullptr;
}

inline void *RealDlSym(void *handle, const char *name)
{
    using DlSymFunc = void *(*)(void *, const char *);
    static const DlSymFunc real_dlsym = reinterpret_cast<DlSymFunc>(GetRealDlSym());
    return real_dlsym(handle, name);
}

#define DLSYM_INTERCEPT_ENTRY(symbol) {#symbol, (void *)symbol}
#define DLSYM_INTERCEPT_LIB(intercept_libs, ...) \
    const static std::vector<std::string> intercept_libs = {__VA_ARGS__};
#define DEFINE_DLSYM_INTERCEPT(intercept_symbol_map, intercept_libs) \
    EXPORT_C_FUNC void *dlsym(void *handle, const char *name) \
    { \
        Dl_info info; \
        void *caller = __builtin_return_address(0); \
        if (caller != nullptr && dladdr(caller, &info) != 0 && info.dli_fname != nullptr) { \
            const std::string path(info.dli_fname); \
            for (const std::string &lib : intercept_libs) { \
                if (path.find(lib) != std::string::npos) { \
                    XDEBG("dlsym symbol ignored: %s required by %s", name, info.dli_fname); \
                    return RealDlSym(handle, name); \
                } \
            } \
        } \
        auto it = intercept_symbol_map.find(name); \
        if (it != intercept_symbol_map.end()) { \
            XDEBG("dlsym symbol replaced: %s -> %p", name, it->second); \
            return it->second; \
        } \
        XDEBG("dlsym symbol ignored: %s", name); \
        return RealDlSym(handle, name); \
    }

#define DEFINE_GET_SYMBOL_FUNC(func, env_name, search_names, search_dirs)       \
    static void *func(const char *symbol_name)                                  \
    {                                                                           \
        static const std::vector<std::string> names = search_names;             \
        static const std::vector<std::string> dirs = search_dirs;               \
        static const std::string dll_path = FindLibrary(env_name, names, dirs); \
        static void *dll_handle = dlopen(dll_path.c_str(), XSCHED_RTLD_FLAGS);  \
        XASSERT(dll_handle != nullptr, "fail to dlopen %s", dll_path.c_str());  \
        void *symbol = RealDlSym(dll_handle, symbol_name);                      \
        XASSERT(symbol != nullptr, "fail to get symbol %s", symbol_name);       \
        return symbol;                                                          \
    }

#define DEFINE_CHECK_SYMBOL_FUNC(func, env_name, search_names, search_dirs)     \
    static bool func(const char *symbol_name)                                   \
    {                                                                           \
        static const std::vector<std::string> names = search_names;             \
        static const std::vector<std::string> dirs = search_dirs;               \
        static const std::string dll_path = FindLibrary(env_name, names, dirs); \
        static void *dll_handle = dlopen(dll_path.c_str(), XSCHED_RTLD_FLAGS);  \
        XASSERT(dll_handle != nullptr, "fail to dlopen %s", dll_path.c_str());  \
        void *symbol = RealDlSym(dll_handle, symbol_name);                      \
        return symbol != nullptr; \
    }

#elif defined(_WIN32)

#include <windows.h>

#define DLSYM_INTERCEPT_ENTRY(symbol) {#symbol, (void *)symbol}
#define DEFINE_DLSYM_INTERCEPT(intercept_symbol_map)                                \
    extern "C" __declspec(dllexport) FARPROC dlsym(HMODULE handle, LPCSTR name)     \
    {                                                                               \
        auto it = intercept_symbol_map.find(name);                                  \
        if (it != intercept_symbol_map.end()) {                                     \
            XDEBG("GetProcAddress symbol replaced: %s -> %p", name, it->second);    \
            return (FARPROC)it->second;                                             \
        }                                                                           \
        XDEBG("GetProcAddress symbol ignored: %s", name);                           \
        return GetProcAddress(handle, name);                                        \
    }

#define DEFINE_GET_SYMBOL_FUNC(func, env_name, search_names, search_dirs)           \
    static void *func(const char *symbol_name)                                      \
    {                                                                               \
        static const std::vector<std::string> names = search_names;                 \
        static const std::vector<std::string> dirs = search_dirs;                   \
        static const std::string dll_path = FindLibrary(env_name, names, dirs);     \
        static HMODULE dll_handle = LoadLibraryA(dll_path.c_str());                 \
        XASSERT(dll_handle != nullptr, "fail to LoadLibrary %s", dll_path.c_str()); \
        FARPROC symbol = ::GetProcAddress(dll_handle, symbol_name);                 \
        XASSERT(symbol != nullptr, "fail to get symbol %s", symbol_name);           \
        return (void*)symbol;                                                       \
    }

#define DEFINE_CHECK_SYMBOL_FUNC(func, env_name, search_names, search_dirs)         \
    static bool func(const char *symbol_name)                                       \
    {                                                                               \
        static const std::vector<std::string> names = search_names;                 \
        static const std::vector<std::string> dirs = search_dirs;                   \
        static const std::string dll_path = FindLibrary(env_name, names, dirs);     \
        static HMODULE dll_handle = LoadLibraryA(dll_path.c_str());                 \
        XASSERT(dll_handle != nullptr, "fail to LoadLibrary %s", dll_path.c_str()); \
        FARPROC symbol = GetProcAddress(dll_handle, symbol_name);                   \
        return symbol != nullptr;                                                   \
    }

#endif
