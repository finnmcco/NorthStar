#include "ir_capture.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

ThermalCapture::ThermalCapture(IRQueue& outQueue,
                               const std::string& device)
    : outQueue_(outQueue), device_(device)
{}

bool ThermalCapture::start()
{
    if (running_.load()) return true;
    running_ = true;
    worker_  = std::thread(&ThermalCapture::run, this);
    return true;
}

void ThermalCapture::stop()
{
    running_ = false;
    outQueue_.stop();          // unblocks any waiting consumer
    if (worker_.joinable())
        worker_.join();
}

void ThermalCapture::run()
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
            pkt.timestamp   = now_ns();
            pkt.ambientTemp = sensor.ambientTemperature();

            // Chess interpolation into packet's own array
            MLX90640::ChessInterpolator::interpolateBlend(
                raw, pkt.temps.data(), 0.4f);

            outQueue_.push(std::move(pkt));
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "[ThermalCapture] " << ex.what() << '\n';
    }

    running_ = false;
}

uint64_t ThermalCapture::now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}