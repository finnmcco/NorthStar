#include "capture/camera_capture.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <pthread.h>

// mmap
#include <sys/mman.h>
#include <unistd.h>

// libcamera includes
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/formats.h>
#include <libcamera/request.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

CameraCapture::CameraCapture(queue<FramePacket>& outQueue,
                             BufferPool& pool,
                             int     fps,
                             uint8_t camera_id)
    : outQueue_(outQueue), pool_(pool), fps_(fps), camera_id_(camera_id)
{
}

bool CameraCapture::start()
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false; // already running
    }

    stopping_ = false;

    // 1) Start CameraManager
    cm_ = std::make_unique<libcamera::CameraManager>();
    if (cm_->start() != 0) {
        std::cerr << "CameraManager start() failed\n";
        running_ = false;
        return false;
    }

    // 2) Pick the first available camera
    if (cm_->cameras().empty()) {
        std::cerr << "No cameras found\n";
        cm_->stop();
        running_ = false;
        return false;
    }
    camera_ = cm_->cameras()[0];

    // 3) Acquire camera
    if (camera_->acquire() != 0) {
        std::cerr << "camera acquire() failed\n";
        cm_->stop();
        running_ = false;
        return false;
    }

    // 4) Generate a configuration (we want a video stream)
    config_ = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
    if (!config_) {
        std::cerr << "generateConfiguration() failed\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // Configure stream parameters.
    // We request an RGB format because your downsampler expects RGB input.
    libcamera::StreamConfiguration& sc = config_->at(0);

    // Choose a reasonable input size. You can change this later.
    // (bigger input -> more crop detail, more CPU work)
    sc.size.width  = 1640;
    sc.size.height = 1232;

    // Request RGB888 if supported.
    // On some systems this may be adjusted by validate().
    sc.pixelFormat = libcamera::formats::RGB888;

    // Set a frame rate (best-effort): via FrameDurationLimits
    // Frame duration in microseconds: min=max=1e6/fps
    /*
    if (fps_ > 0) {
        const int64_t frame_us = 1'000'000 / fps_;
        sc.controls.set(libcamera::controls::FrameDurationLimits,
                        libcamera::Span<const int64_t, 2>({ frame_us, frame_us }));
    }
    */
    // Validate and apply the configuration (libcamera may adjust settings)
    if (config_->validate() == libcamera::CameraConfiguration::Invalid) {
        std::cerr << "Camera configuration invalid\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    if (camera_->configure(config_.get()) != 0) {
        std::cerr << "camera configure() failed\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // Keep pointer to the configured stream
    stream_ = sc.stream();
    if (!stream_) {
        std::cerr << "Configured stream is null\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // IMPORTANT: check what format we actually got after validation
    // If it isn't RGB888, we can still proceed later by adding conversion,
    // but for now we fail fast with a clear message.
    if (sc.pixelFormat != libcamera::formats::RGB888) {
        std::cerr << "Requested RGB888 but got: " << sc.pixelFormat.toString()
                  << ". You likely need a YUV->RGB conversion step.\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // 5) Allocate buffers for the stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    if (allocator_->allocate(stream_) < 0) {
        std::cerr << "FrameBufferAllocator allocate() failed\n";
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    const auto& buffers = allocator_->buffers(stream_);
    if (buffers.empty()) {
        std::cerr << "No buffers allocated\n";
        allocator_.reset();
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // 6) Map all camera buffers once (avoid per-frame mmap)
    mapped_.clear();
    mapped_.reserve(buffers.size());
    for (auto& fb : buffers) {
        mapped_[fb.get()] = mapFrameBuffer(fb.get());
        if (!mapped_[fb.get()]) {
            std::cerr << "Failed to map framebuffer\n";
            allocator_.reset();
            camera_->release();
            cm_->stop();
            running_ = false;
            return false;
        }
    }

    // 7) Create Requests and attach buffers (request ring)
    requests_.clear();
    requests_.reserve(buffers.size());

    for (auto& fb : buffers)
    {
        std::unique_ptr<libcamera::Request> req = camera_->createRequest();
        if (!req) {
            std::cerr << "createRequest() failed\n";
            allocator_.reset();
            camera_->release();
            cm_->stop();
            running_ = false;
            return false;
        }

        if (req->addBuffer(stream_, fb.get()) < 0) {
            std::cerr << "Request addBuffer() failed\n";
            allocator_.reset();
            camera_->release();
            cm_->stop();
            running_ = false;
            return false;
        }

        requests_.push_back(std::move(req));
    }

    // 8) Connect completion callback
    camera_->requestCompleted.connect(this, &CameraCapture::onRequestComplete);

    // 9) Start camera
    if (camera_->start() != 0) {
        std::cerr << "camera start() failed\n";
        camera_->requestCompleted.disconnect(this, &CameraCapture::onRequestComplete);
        allocator_.reset();
        camera_->release();
        cm_->stop();
        running_ = false;
        return false;
    }

    // 10) Queue initial requests
    for (auto& req : requests_) {
        if (camera_->queueRequest(req.get()) < 0) {
            std::cerr << "queueRequest() failed\n";
            // stop camera cleanly
            camera_->stop();
            camera_->requestCompleted.disconnect(this, &CameraCapture::onRequestComplete);
            allocator_.reset();
            camera_->release();
            cm_->stop();
            running_ = false;
            return false;
        }
    }

    // 11) Start worker thread to drain completed requests and process frames
    worker_ = std::thread(&CameraCapture::run, this);
    return true;
}

void CameraCapture::stop()
{
    bool wasRunning = running_.exchange(false);
    if (!wasRunning) {
        return;
    }

    // Wake worker thread if blocked
    {
        std::lock_guard<std::mutex> lk(completedMutex_);
        stopping_ = true;
    }
    completedCv_.notify_all();

    if (worker_.joinable())
        worker_.join();

    // Stop camera and cleanup
    if (camera_) {
        camera_->stop();
        camera_->requestCompleted.disconnect(this, &CameraCapture::onRequestComplete);
    }

    // Free buffers
    allocator_.reset();

    // Release camera
    if (camera_) {
        camera_->release();
        camera_.reset();
    }

    // Stop manager
    if (cm_) {
        cm_->stop();
        cm_.reset();
    }

    // Clear internal state
    while (!completed_.empty()) completed_.pop();
    mapped_.clear();
    requests_.clear();
    config_.reset();
    stream_ = nullptr;
    stopping_ = false;
}

void CameraCapture::onRequestComplete(libcamera::Request* request)
{
    // This callback is invoked by libcamera thread context.
    // Keep it very light: just enqueue and notify the worker.

    if (!request)
        return;

    // Ignore cancelled requests during shutdown
    if (request->status() == libcamera::Request::RequestCancelled)
        return;

    {
        std::lock_guard<std::mutex> lk(completedMutex_);
        completed_.push(request);
    }
    completedCv_.notify_one();
}

void CameraCapture::run()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    auto requeue = [&](libcamera::Request* r) {
    r->reuse(libcamera::Request::ReuseBuffers);
    camera_->queueRequest(r);
};
    // Output stride is tightly packed RGB for 640 pixels
    const int outStrideBytes = kOutWidth * kChannels;

    while (true)
    {
        libcamera::Request* req = nullptr;

        // Block until a completed request arrives or we are stopping
        {
            std::unique_lock<std::mutex> lk(completedMutex_);
            completedCv_.wait(lk, [&] {
                return stopping_ || !completed_.empty();
            });

            if (stopping_ && completed_.empty()) {
                break; // shutdown
            }

            req = completed_.front();
            completed_.pop();
        }

        if (!req)
            continue;

        // Get buffer for our stream
        auto& bufs = req->buffers();
        auto it = bufs.find(stream_);
        if (it == bufs.end()) {
            // Requeue anyway to keep pipeline alive
            requeue(req);
            continue;
        }

        libcamera::FrameBuffer* fb = it->second;

        // Timestamp:
        // For now, use monotonic clock. We can upgrade to metadata timestamps later.
        const std::uint64_t ts = now_ns();

        // Acquire output buffer for downsampled 640x640 RGB
        uint8_t* out = pool_.try_acquire();
        if (!out) {
            // No output buffers free: drop frame (do NOT stall capture)
            requeue(req);
            continue;
        }

        // Access mapped camera buffer (RGB888 likely one plane)
        auto mit = mapped_.find(fb);
        if (mit == mapped_.end() || !mit->second || mit->second->planes.empty()) {
            pool_.release(out);
            requeue(req);
            continue;
        }

        const uint8_t* inData = static_cast<const uint8_t*>(mit->second->planes[0].addr);

        // ------------------------------------------------------
        // Save first raw (unrectified) camera frame for debug
        // ------------------------------------------------------
        static int warmup = 0;
        warmup++;
        if (!savedRaw_ && warmup > 30)
        {
            const libcamera::StreamConfiguration& sc = config_->at(0);
            const int inWidth  = sc.size.width;
            const int inHeight = sc.size.height;
            const int inStrideBytes = (sc.stride > 0) ? sc.stride : (inWidth * kChannels);
            std::cout << "inWidth=" << inWidth << " inHeight=" << inHeight 
                    << " inStride=" << inStrideBytes 
                    << " expected=" << (inWidth * 3) << "\n";

            // libcamera RGB888 = BGR byte order on Pi — already BGR, no conversion needed
            cv::Mat bgrRaw(inHeight, inWidth, CV_8UC3,
                    const_cast<uint8_t*>(inData),
                    inStrideBytes);

            if (cv::imwrite("raw_frame0.png", bgrRaw))
                std::cout << "Saved raw_frame0.png\n";
            else
                std::cerr << "Failed to save raw_frame0.png\n";

            savedRaw_ = true;
        }

        // Input dimensions and stride come from StreamConfiguration
        const libcamera::StreamConfiguration& sc = config_->at(0);
        const int inWidth = sc.size.width;
        const int inHeight = sc.size.height;

        // Stride: libcamera exposes it on the configuration (bytes per row)
        // Some libcamera versions use sc.stride; if your build errors here,
        // paste it and we’ll adjust.
        const int inStrideBytes = sc.stride;

        // Crop+downsample -> out
        const bool ok = downsampler_.process(
            inData, inWidth, inHeight, inStrideBytes,
            out, outStrideBytes
        );

        if (!ok) {
            pool_.release(out);
            requeue(req);
            continue;
        }

        // Publish packet
        FramePacket pkt;
        pkt.camera_id = camera_id_;
        pkt.timestamp = ts;
        pkt.data      = out;

        if (!outQueue_.push(pkt)) {
            // Downstream stopped: return buffer and exit soon
            pool_.release(out);
            // Still requeue request to allow orderly shutdown
            requeue(req);
            break;
        }

        // Requeue the request so camera can fill it again
        requeue(req);
    }
}

std::uint64_t CameraCapture::now_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    const std::uint64_t sec  = static_cast<std::uint64_t>(ts.tv_sec);
    const std::uint64_t nsec = static_cast<std::uint64_t>(ts.tv_nsec);
    return sec * 1'000'000'000ULL + nsec;
}

CameraCapture::MappedBuffer::~MappedBuffer()
{
    // Unmap all planes
    for (auto& p : planes) {
        if (p.addr && p.length) {
            munmap(p.addr, p.length);
        }
        p.addr = nullptr;
        p.length = 0;
    }
}

std::shared_ptr<CameraCapture::MappedBuffer>
CameraCapture::mapFrameBuffer(libcamera::FrameBuffer* fb)
{
    if (!fb)
        return nullptr;

    auto mapped = std::make_shared<MappedBuffer>();
    mapped->planes.reserve(fb->planes().size());

    // Map each plane using its dmabuf file descriptor
    for (const auto& plane : fb->planes())
    {
        const int fd = plane.fd.get();       // dmabuf fd
        const size_t length = plane.length;  // plane byte size

        void* addr = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            return nullptr;
        }

        MappedBuffer::PlaneMap pm;
        pm.addr = addr;
        pm.length = length;
        mapped->planes.push_back(pm);
    }

    return mapped;
}