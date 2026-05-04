#pragma once

#include <cstddef>      // std::size_t
#include <cstdint>      // uint8_t
#include <memory>       // std::unique_ptr
#include <mutex>        // std::mutex
#include <vector>

/*
    BufferPool

    Manages a fixed number of preallocated frame buffers.

    Why this exists:
    ----------------
    - We do NOT want to allocate memory every frame (slow, jittery).
    - FramePacket only holds a pointer, so we need memory that:
        * stays valid while processing happens
        * can be safely reused afterward
    - The pool guarantees deterministic memory usage.

    Thread model:
    -------------
    - Capture thread calls try_acquire()
    - Processing thread calls release()
    - Internal mutex protects the free list

    Policy:
    -------
    - try_acquire() is NON-BLOCKING.
    - If no buffer is available, it returns nullptr.
      (Capture thread can drop the frame.)
*/

class BufferPool
{
public:
    /*
        Constructor

        Parameters:
        - bufferCount: number of buffers to allocate
        - bufferSize: size (in bytes) of each buffer

        Example:
            640x640 grayscale  -> 640 * 640 * 1
            640x640 RGB        -> 640 * 640 * 3
    */
    BufferPool(std::size_t bufferCount, std::size_t bufferSize)
        : bufferSize_(bufferSize)
    {
        // Reserve storage for owning pointers
        buffers_.reserve(bufferCount);

        // Reserve storage for free list
        freeList_.reserve(bufferCount);

        // Allocate all buffers up front
        for (std::size_t i = 0; i < bufferCount; ++i)
        {
            // Allocate raw byte buffer
            auto buffer = std::make_unique<uint8_t[]>(bufferSize_);

            // Store raw pointer in free list
            freeList_.push_back(buffer.get());

            // Keep ownership so memory is freed automatically
            buffers_.push_back(std::move(buffer));
        }
    }

    // Disable copying (pool should not be copied)
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /*
        Try to acquire a free buffer.

        Returns:
        - pointer to buffer if available
        - nullptr if none are free

        Non-blocking: capture thread should not stall.
    */
    uint8_t* try_acquire()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (freeList_.empty())
        {
            return nullptr;
        }

        // Take last free buffer (LIFO is fine here)
        uint8_t* ptr = freeList_.back();
        freeList_.pop_back();

        return ptr;
    }

    /*
        Release a buffer back to the pool.

        Must only release buffers that originally came from this pool.
    */
    void release(uint8_t* buffer)
    {
        if (!buffer)
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        freeList_.push_back(buffer);
    }

    /*
        Returns size of each buffer (in bytes).
        Useful for debugging or validation.
    */
    std::size_t bufferSize() const
    {
        return bufferSize_;
    }

    /*
        Returns number of currently available buffers.
        (Mostly useful for debugging / monitoring.)
    */
    std::size_t available() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return freeList_.size();
    }

private:
    // Size of each buffer in bytes
    std::size_t bufferSize_;

    /*
        buffers_ owns the actual memory.

        Each element is a unique_ptr to a byte array.
        When BufferPool is destroyed, memory is automatically freed.
    */
    std::vector<std::unique_ptr<uint8_t[]>> buffers_;

    /*
        freeList_ holds raw pointers to currently available buffers.
        These pointers refer to memory owned by buffers_.
    */
    std::vector<uint8_t*> freeList_;

    // Protects freeList_
    mutable std::mutex mutex_;
};