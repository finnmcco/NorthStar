// ─────────────────────────────────────────────────────────────────────────────
// main.cpp  –  tts_speak: synthesise a sentence and play it on the I²S speakers
//
// Usage:
//   ./tts_speak "Your text here"
//   ./tts_speak --voice /path/to/voice.onnx "Your text here"
//   ./tts_speak --rate 1.2 "Your text here"
// ─────────────────────────────────────────────────────────────────────────────
#include "speaker.hpp"
#include "tts.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options] \"sentence\"\n\n"
        << "Options:\n"
        << "  --voice <path>   Path to .onnx Piper voice file\n"
        << "  --piper <path>   Path to piper binary\n"
        << "  --rate  <float>  Speaking rate multiplier (default 1.0)\n"
        << "  --sink  <name>   PipeWire/PA sink name (default: system default)\n"
        << "  --help\n\n"
        << "Examples:\n"
        << "  " << argv0 << " \"Hello from the Raspberry Pi!\"\n"
        << "  " << argv0 << " --rate 1.2 \"Hello, faster world!\"\n";
}

int main(int argc, char* argv[]) {
    TTSEngine::Config cfg;
    std::string       sink_name;
    std::string       sentence;

    // ── argument parsing ──────────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a value\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (arg == "--voice") {
            cfg.voice_path = next();
        } else if (arg == "--piper") {
            cfg.piper_path = next();
        } else if (arg == "--rate") {
            cfg.length_scale = 1.0f / std::stof(next());  // rate = 1/length_scale
        } else if (arg == "--sink") {
            sink_name = next();
        } else if (!arg.empty() && arg[0] != '-') {
            sentence = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (sentence.empty()) {
        std::cerr << "Error: no sentence provided\n";
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    // ── TTS engine ────────────────────────────────────────────────────────────
    TTSEngine tts(cfg);
    if (!tts.ready()) {
        std::cerr << "[tts] ERROR: " << tts.last_error() << "\n";
        return EXIT_FAILURE;
    }
    std::cerr << "[tts] piper : " << tts.piper_path() << "\n"
              << "[tts] voice : " << tts.voice_path() << "\n"
              << "[tts] text  : \"" << sentence << "\"\n";

    const auto wav = tts.synthesise_wav(sentence);
    if (wav.empty()) {
        std::cerr << "[tts] ERROR: synthesis failed – " << tts.last_error() << "\n";
        return EXIT_FAILURE;
    }
    std::cerr << "[tts] synthesised " << wav.size() << " bytes of WAV\n";

    // ── Speaker ───────────────────────────────────────────────────────────────
    Speaker spk(Speaker::DEFAULT_RATE, Speaker::DEFAULT_CHANNELS, sink_name);
    if (!spk.open()) {
        std::cerr << "[spk] ERROR: " << spk.last_error() << "\n";
        return EXIT_FAILURE;
    }
    std::cerr << "[spk] playing via PipeWire"
              << (sink_name.empty() ? " (default sink)" : " → " + sink_name)
              << "\n";

    if (!spk.play_wav(wav)) {
        std::cerr << "[spk] ERROR: " << spk.last_error() << "\n";
        return EXIT_FAILURE;
    }

    std::cerr << "[spk] done\n";
    return EXIT_SUCCESS;
}
