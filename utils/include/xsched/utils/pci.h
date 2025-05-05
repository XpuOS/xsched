#pragma once

#include <cstdint>

inline uint32_t MakePciId(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function)
{
    return (domain << 16) | (bus << 8) | (device << 3) | function;
}
