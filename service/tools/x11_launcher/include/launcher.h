#pragma once

#include <string>
#include <cstdint>

namespace xsched::service
{

class X11Launcher
{
public:
    X11Launcher(const std::string &addr, uint16_t port);
    ~X11Launcher() = default;

    void SetBind();

private:
    const std::string kAddr;
    const uint16_t kPort;
};

} // namespace xsched::service
