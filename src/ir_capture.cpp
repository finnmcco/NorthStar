#include "ir_capture.hpp"
#include <time.h>
#include <chrono>
#include <iostream>
#include <stdexcept>

IRCapture::IRCapture(IRQueue& outQueue,
                               const std::string& device)
    : outQueue_(outQueue), device_(device)
{}

bool IRCapture::start()
{
    if (running_.load()) return true;
    running_ = true;
    worker_  = std::thread(&IRCapture::run, this);
    return true;
}

void IRCapture::stop()
{
    running_ = false;
    outQueue_.stop();          // unblocks any waiting consumer
    if (worker_.joinable())
        worker_.join();
}

void IRCapture::run()
{
    try {
        MLX90640::Driver sensor(device_,
                                MLX90640::DEFAULT_I2C_ADDR,
                                MLX90640::RefreshRate::Hz4,
                                MLX90640::ReadPattern::Chess);

        float raw[MLX90640::PIXEL_COUNT] = {};
        bool  got[2] = {false, false};

        while (running_.load()) {
            int subpage = sensor.getFrame(raw);
            if (subpage < 0) continue;
            got[subpage] = true;
            if (!got[0] || !got[1]) continue;

            IRPacket pkt;
            pkt.timestamp_us   = now_us();
            pkt.ambientTemp = sensor.ambientTemperature();

            // Chess interpolation into packet's own array
            MLX90640::ChessInterpolator::interpolateBlend(
                raw, pkt.temps.data(), 0.4f);

            {
                std::lock_guard<std::mutex> lock(latest_mutex_);
                latest_ = pkt;   // copy
            }

            outQueue_.push(std::move(pkt));

            got[0] = false;
            got[1] = false;
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "[IRCapture] " << ex.what() << '\n';
    }

    running_ = false;
}

uint64_t IRCapture::now_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;
}

std::optional<IRPacket> IRCapture::get_latest() const {
    std::lock_guard<std::mutex> lock(latest_mutex_);
    return latest_;   //returns nullopt if no packets yet
}