#pragma once

#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include "xsched/utils/lib.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"

inline void *GetRealDlSym()
{
    Dl_info info;
    XASSERT(dladdr((void *)dlvsym, &info) != 0, "dladdr() failed to get info of dlvsym");
    void *handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL);
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
#define DEFINE_DLSYM_INTERCEPT(intercept_symbol_map) \
    EXPORT_C_FUNC void *dlsym(void *handle, const char *name) \
    { \
        auto it = intercept_symbol_map.find(name); \
        if (it != intercept_symbol_map.end()) { \
            XDEBG("dlsym symbol replaced: %s -> %p", name, it->second); \
            return it->second; \
        } \
        XDEBG("dlsym symbol ignored: %s", name); \
        return RealDlSym(handle, name); \
    }

#define DEFINE_GET_SYMBOL_FUNC(func, env_name, search_names, search_dirs) \
    static void *func(const char *symbol_name) \
    { \
        static const std::vector<std::string> names = search_names;                \
        static const std::vector<std::string> dirs = search_dirs;                  \
        static const std::string dll_path = FindLibrary(env_name, names, dirs);    \
        static void *dll_handle = dlopen(dll_path.c_str(), RTLD_NOW | RTLD_LOCAL); \
        XASSERT(dll_handle != nullptr, "fail to dlopen %s", dll_path.c_str());     \
        void *symbol = RealDlSym(dll_handle, symbol_name);                         \
        XASSERT(symbol != nullptr, "fail to get symbol %s", symbol_name);          \
        return symbol; \
    }

#define DEFINE_CHECK_SYMBOL_FUNC(func, env_name, search_names, search_dirs) \
    static bool func(const char *symbol_name) \
    { \
        static const std::vector<std::string> names = search_names;                \
        static const std::vector<std::string> dirs = search_dirs;                  \
        static const std::string dll_path = FindLibrary(env_name, names, dirs);    \
        static void *dll_handle = dlopen(dll_path.c_str(), RTLD_NOW | RTLD_LOCAL); \
        XASSERT(dll_handle != nullptr, "fail to dlopen %s", dll_path.c_str());     \
        void *symbol = RealDlSym(dll_handle, symbol_name);                         \
        return symbol != nullptr; \
    }
