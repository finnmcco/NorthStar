#include <iostream>
#include <cstdint>

// Your project headers (adjust paths if needed)
#include "capture/frame_packet.hpp"
#include "util/queue.hpp"
#include "util/buffer_pool.hpp"
#include "capture/camera_capture.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

int main()
{
    // Queue capacity: how many FramePackets to buffer.
    // With drop-oldest, this acts like “keep last N frames”.
    queue<FramePacket> frameQueue(4);

    // Pool size: how many output image buffers exist.
    // This should be >= queue capacity, and usually a bit larger.
    constexpr std::size_t kBuffers = 6;

    // 640x640 RGB = 640*640*3 bytes
    constexpr std::size_t kBytesPerFrame = 640 * 640 * 3;

    BufferPool pool(kBuffers, kBytesPerFrame);

    //DownsamplerRGB DownsamplerRGB();

    CameraCapture capture(frameQueue, pool, 30);
    capture.start();

    // Consumer loop: pop packets, do something, then release buffer back to pool.
    FramePacket pkt{};
    bool saved = false;
    std::uint64_t lastTs = 0;
    int received = 0;

    // Example stop condition: receive 120 frames (~4 seconds at 30 FPS)
    const int targetFrames = 500;

    while (received < targetFrames && frameQueue.pop(pkt))
    {
        // pkt.data points to a 640x640 RGB buffer
        // pkt.timestamp is in nanoseconds (monotonic)

        if (lastTs != 0) {
            std::uint64_t dt = pkt.timestamp - lastTs;
            std::cout << "Frame " << received
                      << " ts=" << pkt.timestamp
                      << " dt(ns)=" << dt
                      << " pool_avail=" << pool.available()
                      << "\n";
        } else {
            std::cout << "Frame " << received
                      << " ts=" << pkt.timestamp
                      << " pool_avail=" << pool.available()
                      << "\n";
        }

        lastTs = pkt.timestamp;
        received++;

        static int wait_cam = 0;
        wait_cam++; 
        static int frame_counter = 0;   // counts every frame
        static int save_index = 0;      // counts saved images

        frame_counter++;

        if (!saved && wait_cam > 30 && frame_counter % 5 == 0)
        {
            // Wrap raw 640x640 RGB buffer as cv::Mat (no copy)
            cv::Mat rgb(640, 640, CV_8UC3, pkt.data, 640 * 3);

            std::string filename = "frame" + std::to_string(save_index) + ".png";

            if (cv::imwrite(filename, rgb)) {
                std::cout << "Saved " << filename << "\n";
                save_index++;   // increment only when saved
            } else {
                std::cerr << "Failed to save " << filename << "\n";
            }
        }

        // IMPORTANT: return buffer to pool when done
        pool.release(pkt.data);
    }

    // Shut down in the correct order:
    // 1) Stop producer so it stops acquiring buffers
    capture.stop();

    // 2) Stop queue so any waiting consumer can exit
    frameQueue.stop();

    std::cout << "Done. Received " << received
              << " frames. Pool available buffers final: "
              << pool.available() << "\n";

    return 0;
}