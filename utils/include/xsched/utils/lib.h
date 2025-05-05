#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"

inline std::string FindLibrary(const std::string &env_name,
                               const std::vector<std::string> &search_names,
                               const std::vector<std::string> &search_dirs)
{
    static const std::vector<std::string> default_dirs {
        "/lib/"          ARCH_STR "-linux-gnu",
        "/usr/lib/"      ARCH_STR "-linux-gnu",
        "/usr/local/lib" ARCH_STR "-linux-gnu",
        "/lib/",
        "/usr/lib/",
        "/usr/local/lib",
    };

    char *path = std::getenv(env_name.c_str());
    if (path != nullptr) {
        if (FileExists(path)) return std::string(path);
        XWARN("lib set by env %s = %s not found, fallback to dir search", env_name.c_str(), path);
    }

    for (const auto &dir : search_dirs) {
        for (const auto &name : search_names) {
            std::string path = dir + "/" + name;
            if (FileExists(path)) {
                XDEBG("lib %s found at %s", name.c_str(), path.c_str());
                return path;
            }
        }
    }

    for (const auto &dir : default_dirs) {
        for (const auto &name : search_names) {
            std::string path = dir + "/" + name;
            if (FileExists(path)) {
                XDEBG("lib %s found at %s", name.c_str(), path.c_str());
                return path;
            }
        }
    }

    for (const auto &name : search_names) XWARN("lib %s not found", name.c_str());
    return "";
}
