#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * TTSEngine — persistent Piper TTS subprocess with zero-copy PCM output.
 *
 * The old design called std::system() per utterance, which forced Piper to:
 *   1. Fork a shell
 *   2. Exec piper
 *   3. Load the 65 MB ONNX model from disk
 *   4. Synthesise
 *   5. Exit
 * Steps 1-3 alone took ~2.4 s on aarch64, making RTF > 1 for short sentences.
 *
 * This design spawns Piper once at construction with --output-raw piped to
 * stdout and text on stdin.  The model stays resident in memory.  Subsequent
 * synthesise() calls take ~100-300 ms for a short sentence (RTF ~0.1-0.3×).
 *
 * Thread safety: synthesise() is serialised internally; calling it from
 * multiple threads is safe but requests are queued, not parallelised.
 *
 * Usage:
 *   TTSEngine tts("/usr/local/bin/piper",
 *                 "/opt/piper/en_US-lessac-medium.onnx",
 *                 22050);
 *   if (!tts.ready()) { ... handle error ... }
 *   auto pcm = tts.synthesise("Hello world");
 */
class TTSEngine {
public:
    /**
     * @param piper_bin  Full path to the piper executable.
     * @param model_path Full path to the .onnx voice model.
     * @param sample_rate Expected output sample rate (used for diagnostics).
     *                    Typical lessac-medium value is 22050.
     */
    TTSEngine(const std::string& piper_bin,
              const std::string& model_path,
              int sample_rate = 22050);

    ~TTSEngine();

    // Non-copyable, non-movable (owns file descriptors + thread).
    TTSEngine(const TTSEngine&)            = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;

    /**
     * Returns true if the subprocess launched successfully and is alive.
     * Check this before the first call to synthesise().
     */
    bool ready() const { return ready_; }

    /**
     * Returns a human-readable string describing why ready() is false,
     * or an empty string if ready() is true.
     */
    const std::string& error_message() const { return error_msg_; }

    /**
     * Synthesise @p text into raw 16-bit signed little-endian PCM samples.
     *
     * Blocks until synthesis is complete.  Returns an empty vector on error.
     * The sample rate is whatever the model was trained at (see sample_rate
     * passed to the constructor for reference).
     *
     * @param text   Plain text to synthesise.  Keep under ~500 chars; longer
     *               inputs may saturate the pipe and deadlock on some kernels.
     * @param timeout_ms  Maximum milliseconds to wait.  0 = no timeout.
     */
    std::vector<int16_t> synthesise(const std::string& text,
                                     unsigned timeout_ms = 10000);

    /**
     * Convenience: synthesise and write a WAV file to @p path.
     * Returns true on success.
     */
    bool synthesise_wav(const std::string& text,
                        const std::string& path,
                        unsigned timeout_ms = 10000);

    int sample_rate() const { return sample_rate_; }

private:
    void spawn(const std::string& piper_bin, const std::string& model_path);
    void reader_thread_fn();
    void teardown();

    int     stdin_fd_  = -1;   // write end of pipe → Piper stdin
    int     stdout_fd_ = -1;   // read  end of pipe ← Piper stdout
    pid_t   child_pid_ = -1;

    int     sample_rate_;
    bool    ready_     = false;
    std::string error_msg_;

    // Reader thread accumulates raw bytes from Piper stdout.
    std::thread          reader_thread_;
    std::mutex           buf_mutex_;
    std::condition_variable buf_cv_;
    std::vector<uint8_t> read_buf_;
    std::atomic<bool>    reading_done_{false};  // set when Piper closes stdout segment
    bool                 eof_seen_ = false;

    // Serialise concurrent synthesise() calls.
    std::mutex synth_mutex_;
};