#pragma once
#include <cstdint>
#include <vector>


struct BoundingBox {
    float x_min = 0.f;   // left edge,   [0.0 – 1.0]
    float y_min = 0.f;   // top edge,    [0.0 – 1.0]
    float x_max = 0.f;   // right edge,  [0.0 – 1.0]
    float y_max = 0.f;   // bottom edge, [0.0 – 1.0]
};

/*
    Detection is one of the detected object in a frame.


    confidence is a probability in range [0.0, 1.0].

*/
struct Detection {
    uint8_t    object_id   = 0;     
    float      confidence  = 0.f;  
    BoundingBox box;
};


struct FilteredInferencePair {
    uint8_t              object_id   = 0;
    uint64_t             timestamp   = 0;   
    std::vector<Detection> cam0_detections;     
    std::vector<Detection> cam1_detections; 
};