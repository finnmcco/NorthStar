/**
 * tts.cpp — TTSEngine implementation
 *
 * Persistent-Piper design: one subprocess, model loaded once at startup.
 * See tts.hpp for full rationale and API docs.
 *
 * Protocol with Python pip Piper (--output-file /dev/stdout mode):
 *   This is the piper installed via pip into a venv.  It does NOT support
 *   --json-input or the C++ piper's framed raw output.  Instead:
 *
 *   stdin  ← one line of plain text per utterance, terminated with '\n'
 *   stdout → one complete WAV file per utterance (RIFF header + PCM data)
 *
 *   The WAV header gives us the exact data length at bytes 40-43, so we read
 *   the 44-byte header first, extract data_size, then read exactly that many
 *   bytes of PCM.  This gives clean per-utterance framing with no sentinel
 *   tricks and works reliably with the Python Piper venv version.
 *
 * Piper invocation:
 *   piper --model <model> --output-file /dev/stdout
 *
 * Framing:
 *   [44-byte RIFF/WAV header][data_size bytes of 16-bit PCM][next header...]
 */

#include "tts.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool set_cloexec(int fd) {
    int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0) return false;
    return ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

/** Read exactly n bytes from fd into buf.  Returns false on error/EOF. */
static bool read_exact(int fd, void* buf, size_t n, unsigned timeout_ms) {
    auto* p   = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < n) {
        if (timeout_ms > 0) {
            pollfd pfd{fd, POLLIN, 0};
            int r = ::poll(&pfd, 1, static_cast<int>(timeout_ms));
            if (r <= 0) return false;  // timeout or error
        }
        ssize_t r = ::read(fd, p + got, n - got);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

// ---------------------------------------------------------------------------
// WAV write helper (used by synthesise_wav())
// ---------------------------------------------------------------------------

static void write_le16(std::ostream& os, uint16_t v) {
    os.put(static_cast<char>(v & 0xff));
    os.put(static_cast<char>((v >> 8) & 0xff));
}
static void write_le32(std::ostream& os, uint32_t v) {
    os.put(static_cast<char>(v & 0xff));
    os.put(static_cast<char>((v >> 8) & 0xff));
    os.put(static_cast<char>((v >> 16) & 0xff));
    os.put(static_cast<char>((v >> 24) & 0xff));
}

static bool write_wav(const std::string& path,
                      const std::vector<int16_t>& pcm,
                      int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * 2);
    // RIFF header
    f.write("RIFF", 4);
    write_le32(f, 36 + data_bytes);
    f.write("WAVE", 4);
    // fmt chunk
    f.write("fmt ", 4);
    write_le32(f, 16);                              // chunk size
    write_le16(f, 1);                               // PCM
    write_le16(f, 1);                               // mono
    write_le32(f, static_cast<uint32_t>(sample_rate));
    write_le32(f, static_cast<uint32_t>(sample_rate * 2)); // byte rate
    write_le16(f, 2);                               // block align
    write_le16(f, 16);                              // bits/sample
    // data chunk
    f.write("data", 4);
    write_le32(f, data_bytes);
    f.write(reinterpret_cast<const char*>(pcm.data()), data_bytes);
    return f.good();
}

// ---------------------------------------------------------------------------
// TTSEngine — construction / teardown
// ---------------------------------------------------------------------------

TTSEngine::TTSEngine(const std::string& piper_bin,
                     const std::string& model_path,
                     int sample_rate)
    : sample_rate_(sample_rate)
{
    spawn(piper_bin, model_path);
}

TTSEngine::~TTSEngine() {
    teardown();
}

void TTSEngine::spawn(const std::string& piper_bin,
                      const std::string& model_path) {
    // Verify executables / model exist before forking.
    if (::access(piper_bin.c_str(), X_OK) != 0) {
        error_msg_ = "piper not found or not executable: " + piper_bin;
        return;
    }
    if (::access(model_path.c_str(), R_OK) != 0) {
        error_msg_ = "Piper model not readable: " + model_path;
        return;
    }

    // Create two pipe pairs: parent→child (stdin) and child→parent (stdout).
    int to_child[2], from_child[2];
    if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) {
        error_msg_ = std::string("pipe(): ") + ::strerror(errno);
        return;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        error_msg_ = std::string("fork(): ") + ::strerror(errno);
        ::close(to_child[0]);  ::close(to_child[1]);
        ::close(from_child[0]); ::close(from_child[1]);
        return;
    }

    if (pid == 0) {
        // ── Child ──
        // Wire up pipes.
        ::dup2(to_child[0],   STDIN_FILENO);
        ::dup2(from_child[1], STDOUT_FILENO);
        // Close all pipe ends we no longer need.
        ::close(to_child[0]);  ::close(to_child[1]);
        ::close(from_child[0]); ::close(from_child[1]);
        // Silence stderr so Piper's model-load chatter doesn't corrupt logs.
        int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) { ::dup2(devnull, STDERR_FILENO); ::close(devnull); }

        // Launch Python pip Piper: plain text on stdin, WAV on stdout.
        // --output-file /dev/stdout causes Piper to write one WAV per input line.
        const char* argv[] = {
            piper_bin.c_str(),
            "--model",        model_path.c_str(),
            "--output-file",  "/dev/stdout",
            nullptr
        };
        ::execv(piper_bin.c_str(), const_cast<char* const*>(argv));
        // execv only returns on failure.
        ::_exit(127);
    }

    // ── Parent ──
    ::close(to_child[0]);    // parent doesn't read from its own stdin pipe
    ::close(from_child[1]);  // parent doesn't write to its own stdout pipe

    set_cloexec(to_child[1]);
    set_cloexec(from_child[0]);

    stdin_fd_  = to_child[1];
    stdout_fd_ = from_child[0];
    child_pid_ = pid;
    ready_     = true;
}

void TTSEngine::teardown() {
    if (stdin_fd_ >= 0)  { ::close(stdin_fd_);  stdin_fd_  = -1; }
    if (stdout_fd_ >= 0) { ::close(stdout_fd_); stdout_fd_ = -1; }
    if (child_pid_ > 0) {
        ::kill(child_pid_, SIGTERM);
        // Give Piper 500 ms to exit cleanly, then force.
        for (int i = 0; i < 5; ++i) {
            int status;
            pid_t r = ::waitpid(child_pid_, &status, WNOHANG);
            if (r != 0) { child_pid_ = -1; return; }
            ::usleep(100'000);
        }
        ::kill(child_pid_, SIGKILL);
        ::waitpid(child_pid_, nullptr, 0);
        child_pid_ = -1;
    }
}

// ---------------------------------------------------------------------------
// WAV header parser — reads exactly 44 bytes and extracts data_size.
// Returns false if the header doesn't look like a valid PCM WAV.
// ---------------------------------------------------------------------------

static uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | static_cast<uint32_t>(p[1]) << 8
         | static_cast<uint32_t>(p[2]) << 16
         | static_cast<uint32_t>(p[3]) << 24;
}

static bool parse_wav_header(const uint8_t hdr[44], uint32_t& data_size, int& sample_rate) {
    // Bytes 0-3: "RIFF"
    if (hdr[0]!='R' || hdr[1]!='I' || hdr[2]!='F' || hdr[3]!='F') return false;
    // Bytes 8-11: "WAVE"
    if (hdr[8]!='W' || hdr[9]!='A' || hdr[10]!='V' || hdr[11]!='E') return false;
    // Bytes 12-15: "fmt "
    if (hdr[12]!='f' || hdr[13]!='m' || hdr[14]!='t' || hdr[15]!=' ') return false;
    // Bytes 36-39: "data"
    if (hdr[36]!='d' || hdr[37]!='a' || hdr[38]!='t' || hdr[39]!='a') return false;
    // Bytes 24-27: sample rate
    sample_rate = static_cast<int>(read_le32(hdr + 24));
    // Bytes 40-43: data chunk size
    data_size = read_le32(hdr + 40);
    return true;
}

// ---------------------------------------------------------------------------
// TTSEngine — synthesis
// ---------------------------------------------------------------------------

std::vector<int16_t> TTSEngine::synthesise(const std::string& text,
                                            unsigned timeout_ms) {
    if (!ready_) return {};

    // Serialise concurrent callers.
    std::lock_guard<std::mutex> lock(synth_mutex_);

    // Write plain text line to Piper stdin.
    std::string line = text + "\n";
    const char* p   = line.data();
    size_t      rem = line.size();
    while (rem > 0) {
        ssize_t w = ::write(stdin_fd_, p, rem);
        if (w <= 0) {
            error_msg_ = "write to piper stdin failed";
            ready_ = false;
            return {};
        }
        p   += w;
        rem -= static_cast<size_t>(w);
    }

    // Read the 44-byte WAV header Piper writes back.
    uint8_t hdr[44];
    if (!read_exact(stdout_fd_, hdr, 44, timeout_ms)) {
        error_msg_ = "timeout/error reading piper WAV header";
        ready_ = false;
        return {};
    }

    uint32_t data_size  = 0;
    int      wav_rate   = 0;
    if (!parse_wav_header(hdr, data_size, wav_rate)) {
        error_msg_ = "invalid WAV header from piper";
        ready_ = false;
        return {};
    }

    // Update sample_rate_ from the actual WAV header (self-calibrates).
    if (wav_rate > 0) sample_rate_ = wav_rate;

    if (data_size == 0) return {};
    if (data_size % 2 != 0) {
        error_msg_ = "odd data_size in piper WAV: " + std::to_string(data_size);
        ready_ = false;
        return {};
    }

    // Read exactly data_size bytes of 16-bit PCM.
    std::vector<uint8_t> raw(data_size);
    if (!read_exact(stdout_fd_, raw.data(), data_size, timeout_ms)) {
        error_msg_ = "timeout/error reading piper PCM data";
        ready_ = false;
        return {};
    }

    // Reinterpret as int16_t (little-endian — native on aarch64/x86).
    const size_t n_samples = data_size / 2;
    std::vector<int16_t> pcm(n_samples);
    std::memcpy(pcm.data(), raw.data(), data_size);
    return pcm;
}

bool TTSEngine::synthesise_wav(const std::string& text,
                                const std::string& path,
                                unsigned timeout_ms) {
    auto pcm = synthesise(text, timeout_ms);
    if (pcm.empty()) return false;
    return write_wav(path, pcm, sample_rate_);
}