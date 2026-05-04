#pragma once
#include <cstdint>

/*
    FramePacket

    Lightweight handle passed through the camera queue.

    Memory ownership
    ----------------
    data points into a BufferPool.  The consumer MUST call pool.release(pkt.data)
    once it has finished with the buffer.  The pool keeps the backing allocation
    alive for the lifetime of the BufferPool object itself.
*/
struct FramePacket {
    uint8_t  camera_id   = 0;        // 0 = left / cam0, 1 = right / cam1
    uint64_t timestamp   = 0;        // CLOCK_MONOTONIC_RAW, nanoseconds
    uint8_t* data        = nullptr;  // 640 × 640 × 3 bytes, RGB, pool-owned
};
