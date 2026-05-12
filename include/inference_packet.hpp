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


struct InferencePacket {
    uint8_t              camera_id   = 0;
    uint64_t             timestamp   = 0;   // nanoseconds, matches FramePacket
    std::vector<Detection> detections;      // one entry per detected object
};