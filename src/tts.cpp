// ─────────────────────────────────────────────────────────────────────────────
// tts.cpp  –  Piper TTS subprocess interface
// ─────────────────────────────────────────────────────────────────────────────
#include "tts.hpp"
#include "speaker.hpp"   // for parse_wav_header

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

// POSIX
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ── discovery ─────────────────────────────────────────────────────────────────

static const char* home_dir() {
    const char* h = std::getenv("HOME");
    return h ? h : "";
}

std::string TTSEngine::find_piper() {
    const std::string home = home_dir();

    const std::array<std::string, 3> candidates = {
        home + "/.local/piper/piper",
        "/usr/local/bin/piper",
        "/usr/bin/piper",
    };

    for (const auto& p : candidates)
        if (fs::exists(p) && fs::is_regular_file(p)) return p;

    // Fall back to PATH search
    FILE* f = popen("which piper 2>/dev/null", "r");
    if (f) {
        char buf[512] = {};
        if (fgets(buf, sizeof(buf), f)) {
            pclose(f);
            std::string s = buf;
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                s.pop_back();
            if (!s.empty() && fs::exists(s)) return s;
        } else {
            pclose(f);
        }
    }
    return "";
}

std::string TTSEngine::find_default_voice() {
    const std::string home = home_dir();

    // Prefer repo-local voices so the project is self-contained.
    // Typical runtime location is NorthStar/build, so ../model_inf is the
    // repository's model_inf directory. The other relative paths make this
    // robust when running from the repo root or from nested build folders.
    const std::array<std::string, 8> search_dirs = {
        "../model_inf",
        "./model_inf",
        "../../model_inf",
        "../model_inf/piper",
        "./model_inf/piper",
        home + "/.local/piper/voices",
        "/usr/share/piper/voices",
        "/usr/local/share/piper/voices",
    };

    std::string fallback;

    for (const auto& dir : search_dirs) {
        if (!fs::exists(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() != ".onnx") continue;
            const std::string name = entry.path().filename().string();
            if (fallback.empty()) fallback = entry.path().string();
            if (name.find("en_") != std::string::npos &&
                name.find("medium") != std::string::npos)
                return entry.path().string();
        }
    }
    return fallback;
}

// ── construction ──────────────────────────────────────────────────────────────

TTSEngine::TTSEngine(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.piper_path.empty()) cfg_.piper_path = find_piper();
    if (cfg_.voice_path.empty()) cfg_.voice_path = find_default_voice();

    if (cfg_.piper_path.empty())
        err_ = "Piper binary not found (run setup_tts.sh or set Config::piper_path)";
    else if (cfg_.voice_path.empty())
        err_ = "No .onnx voice found (run setup_tts.sh or set Config::voice_path)";
}

bool TTSEngine::ready() const noexcept {
    return !cfg_.piper_path.empty() && !cfg_.voice_path.empty() && err_.empty();
}

// ── synthesis ─────────────────────────────────────────────────────────────────

std::vector<uint8_t> TTSEngine::synthesise_wav(const std::string& text) {
    if (!ready()) return {};

    // ── Write input text to a temp file ──────────────────────────────────────
    char in_tmpl[] = "/tmp/tts_in_XXXXXX";
    int  in_fd     = mkstemp(in_tmpl);
    if (in_fd < 0) { err_ = "mkstemp (input) failed"; return {}; }

    {
        const ssize_t written = ::write(in_fd, text.data(), text.size());
        ::close(in_fd);
        if (written != static_cast<ssize_t>(text.size())) {
            err_ = "Failed to write input temp file";
            ::unlink(in_tmpl);
            return {};
        }
    }

    // ── Create output temp file (.wav suffix so Piper is happy) ──────────────
    char out_tmpl[] = "/tmp/tts_out_XXXXXX.wav";
    int  out_fd     = mkstemps(out_tmpl, 4);
    if (out_fd < 0) {
        err_ = "mkstemps (output) failed";
        ::unlink(in_tmpl);
        return {};
    }
    ::close(out_fd);

    // ── Build and run the Piper command ──────────────────────────────────────
    std::ostringstream cmd;
    cmd << "ESPEAK_DATA_PATH=" << cfg_.espeak_data_path << " "
        << cfg_.piper_path
        << " --model "            << cfg_.voice_path
        << " --output_file "      << out_tmpl
        << " --sentence_silence " << cfg_.sentence_silence
        << " --length_scale "     << cfg_.length_scale
        << " --noise_scale "      << cfg_.noise_scale
        << " --noise_w "          << cfg_.noise_w
        << " < "                  << in_tmpl
        << " 2>/dev/null";

    const int rc = std::system(cmd.str().c_str());

    ::unlink(in_tmpl);

    if (rc == -1) {
        err_ = "system() failed to launch shell";
        ::unlink(out_tmpl);
        return {};
    }
    if (WIFSIGNALED(rc)) {
        err_ = "Piper killed by signal " + std::to_string(WTERMSIG(rc));
        ::unlink(out_tmpl);
        return {};
    }
    if (WIFEXITED(rc) && WEXITSTATUS(rc) != 0) {
        err_ = "Piper exited with code " + std::to_string(WEXITSTATUS(rc));
        ::unlink(out_tmpl);
        return {};
    }

    // ── Read WAV back ─────────────────────────────────────────────────────────
    std::ifstream f(out_tmpl, std::ios::binary);
    const std::vector<uint8_t> wav_data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );
    ::unlink(out_tmpl);

    if (wav_data.empty()) {
        err_ = "Piper produced an empty output file";
        return {};
    }

    err_.clear();
    return wav_data;
}

std::vector<int16_t> TTSEngine::synthesise(const std::string& text) {
    const auto wav = synthesise_wav(text);
    if (wav.empty()) return {};

    WavInfo info{};
    if (!Speaker::parse_wav_header(wav.data(), wav.size(), info)) {
        err_ = "Could not parse WAV header from Piper output";
        return {};
    }

    const size_t avail     = wav.size() - info.data_offset;
    const size_t n_bytes   = std::min<size_t>(info.data_size, avail);
    const size_t n_samples = n_bytes / sizeof(int16_t);

    const auto* begin = reinterpret_cast<const int16_t*>(wav.data() + info.data_offset);
    return std::vector<int16_t>(begin, begin + n_samples);
}