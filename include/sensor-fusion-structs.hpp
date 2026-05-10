#pragma once
#include <cstdint>
#include <optional>

struct GridCell {
    int row;
    int col;
};

struct TemperatureStats {
    float mean;
    float peak;
};

struct ObjectReport {
    uint8_t object_id;
    GridCell direction;
    std::optional<float> distance;
    std::optional<TemperatureStats> temp;
};

enum class Visibility {
    None,
    Both,
    Cam0only,
    Cam1only
};

struct BoundingBox {
    float x0, y0;  
    float x1, y1;  
};