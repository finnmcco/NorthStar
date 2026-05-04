#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include "ThermalUtils.hpp"
#include "MLX90640.hpp"
#include "ir_queue.hpp"
#include "frame_packet.hpp"

/*
    ThermalCapture

    Mirrors the CameraCapture producer pattern.
    Runs the MLX90640 sensor on a background thread,
    packages each completed dual-subpage frame as a
    ThermalPacket and pushes it into the supplied queue.

    Usage:
        queue<ThermalPacket> thermalQ(32);
        ThermalCapture capture(thermalQ, "/dev/i2c-1");
        capture.start();
        // ... consumer thread calls thermalQ.pop(pkt) ...
        capture.stop();
*/
class ThermalCapture
{
public:
    ThermalCapture(IRQueue& outQueue,
                   const std::string& device = "/dev/i2c-1");

    ThermalCapture(const ThermalCapture&) = delete;
    ThermalCapture& operator=(const ThermalCapture&) = delete;

    // Initialises sensor and launches worker thread. Returns false on failure.
    bool start();

    // Signals the worker to stop and joins the thread. Safe to call multiple times.
    void stop();

private:
    void run();

    static uint64_t now_ns();

    IRQueue& outQueue_;
    std::string           device_;
    std::atomic<bool>     running_{false};
    std::thread           worker_;
};