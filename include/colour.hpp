#pragma once
#include "config.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>


inline void hailo_prepare(uint8_t* data, std::size_t byte_count)
{
#if !NORTHSTAR_HAILO_BGR
    for (std::size_t i = 0; i < byte_count; i += 3)
        std::swap(data[i], data[i + 2]);
#else
    (void)data;
    (void)byte_count;
#endif
}

inline void hailo_prepare(std::vector<uint8_t>& data)
{
    hailo_prepare(data.data(), data.size());
}