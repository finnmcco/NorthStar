#pragma once

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

// Project headers (adjust if needed)
#include "frame_packet.hpp"
#include "util/queue.hpp" 
#include "util/buffer_pool.hpp"
#include "preprocess/downsampler.hpp"   

// libcamera 
#include <libcamera/libcamera.h> 

/*
    CameraCapture (libcamera)

    Purpose:
    --------
    Real camera producer that replaces FakeCapture.

    It:
      - configures the Raspberry Pi camera via libcamera
      - allocates a ring of frame buffers and Requests
      - blocks waiting for completed Requests (frame ready)
      - timestamps frames
      - converts/crops/downsamples to 640x640 RGB via DownsamplerRGB
      - pushes FramePacket{timestamp, dataPtr} into a BoundedQueue
      - requeues Requests

    Thread model:
    -------------
    - start() sets up libcamera and launches a worker thread
    - libcamera delivers frame completion via a callback (signal)
    - the callback enqueues completed requests into an internal queue
    - the worker thread blocks on a condition_variable until a request arrives

    Output:
    -------
    - Output buffers come from BufferPool
    - FramePacket holds only a pointer + timestamp (no ownership)
    - Consumer MUST release(pkt.data) back into BufferPool when done
*/

class CameraCapture
{
public:
    // Model expects 640x640 RGB
    static constexpr int kOutWidth  = DownsamplerRGB::kOutWidth;   // 640
    static constexpr int kOutHeight = DownsamplerRGB::kOutHeight;  // 640
    static constexpr int kChannels  = DownsamplerRGB::kChannels;   // 3

    /*
        Constructor

        outQueue: queue to publish FramePacket objects
        pool:     pool that provides output 640x640 RGB buffers
        fps:      desired frames per second (best-effort)
    */
    CameraCapture(queue<FramePacket>& outQueue,
                  BufferPool& pool,
                  int fps = 30);

    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;

    // Starts camera + worker thread. Returns false on failure.
    bool start();

    // Stops worker thread and camera. Safe to call multiple times.
    void stop();

private:
    // Worker loop: waits for completed Requests and processes them
    void run();

    // libcamera callback when a request completes (frame ready)
    void onRequestComplete(libcamera::Request* request);

    // Monotonic timestamp in nanoseconds (fallback if metadata not used)
    static std::uint64_t now_ns();

    // Map camera buffers into CPU address space (mmap), stored per FrameBuffer
    struct MappedBuffer
    {
        // Each plane can be mapped separately; on Pi RGB is usually one plane.
        struct PlaneMap {
            void*  addr = nullptr;
            size_t length = 0;
        };

        std::vector<PlaneMap> planes;

        // Unmap on destruction
        ~MappedBuffer();
    };

    // Helper: map a FrameBuffer’s planes (fd/length) for CPU reads
    std::shared_ptr<MappedBuffer> mapFrameBuffer(libcamera::FrameBuffer* fb);

private:
    queue<FramePacket>& outQueue_;
    BufferPool& pool_;
    int fps_ = 30;

    bool savedRaw_ = false;

    DownsamplerRGB downsampler_;

    std::atomic<bool> running_{false};
    std::thread worker_;

    // libcamera objects
    std::unique_ptr<libcamera::CameraManager> cm_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    libcamera::Stream* stream_ = nullptr;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;

    // Requests and buffer mappings
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    std::unordered_map<libcamera::FrameBuffer*, std::shared_ptr<MappedBuffer>> mapped_;

    // Internal completed-request queue (filled by callback, drained by worker)
    std::mutex completedMutex_;
    std::condition_variable completedCv_;
    std::queue<libcamera::Request*> completed_;

    // Used for a clean shutdown wake-up
    bool stopping_ = false;
};