// infer.cpp  —  Stage 2: WAV → Vosk grammar-restricted intent detection
//
// Reads a 16kHz mono 16-bit PCM WAV file produced by record_prepare,
// feeds it to Vosk in fixed-size chunks, and prints detections.
//
// Build:
//   g++ -std=c++17 -O2 -o infer infer.cpp -lvosk
//   (vosk_api.h and libvosk.so must be on the include/lib paths)
//
// Usage:
//   ./infer output.wav model/vosk-model-small-en-us-0.15 cat dog person car

#include <vosk_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants — must match record_prepare output
// ---------------------------------------------------------------------------

static constexpr int    VOSK_RATE    = 16000;
static constexpr size_t CHUNK_FRAMES = 1600;   // 100ms at 16kHz

// ---------------------------------------------------------------------------
// Minimal WAV reader
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];
    uint32_t chunk_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_fmt;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_samp;
    char     data[4];
    uint32_t data_size;
};
#pragma pack(pop)

static void read_wav_header(std::ifstream& f, const std::string& path) {
    WavHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f) throw std::runtime_error("Cannot read WAV header: " + path);

    if (std::strncmp(hdr.riff, "RIFF", 4) != 0 ||
        std::strncmp(hdr.wave, "WAVE", 4) != 0)
        throw std::runtime_error("Not a RIFF/WAVE file: " + path);

    if (hdr.audio_fmt != 1)
        throw std::runtime_error("Only PCM WAV supported (audio_fmt must be 1)");

    if (hdr.num_channels != 1)
        throw std::runtime_error("Expected mono WAV, got " +
                                 std::to_string(hdr.num_channels) + " channels");

    if (hdr.sample_rate != VOSK_RATE)
        throw std::runtime_error("Expected " + std::to_string(VOSK_RATE) +
                                 " Hz WAV, got " + std::to_string(hdr.sample_rate));

    if (hdr.bits_per_samp != 16)
        throw std::runtime_error("Expected 16-bit WAV, got " +
                                 std::to_string(hdr.bits_per_samp) + "-bit");

    std::printf("WAV: %u Hz, %u ch, %u-bit PCM, %.2f s\n",
                hdr.sample_rate, hdr.num_channels, hdr.bits_per_samp,
                static_cast<double>(hdr.data_size) / (hdr.sample_rate * sizeof(int16_t)));
}

// ---------------------------------------------------------------------------
// Minimal JSON field extractor  (avoids pulling in a JSON library)
// Vosk returns small blobs: {"text":"cat"} or {"partial":"do"}
// ---------------------------------------------------------------------------

static std::string extract_field(const char* json, const char* field) {
    if (!json || !field) return {};
    for (const char* sep : {" : \"", ": \""}) {
        std::string key = std::string("\"") + field + "\"" + sep;
        const char* p = std::strstr(json, key.c_str());
        if (!p) continue;
        p += key.size();
        const char* e = std::strchr(p, '"');
        if (!e) continue;
        return std::string(p, e);
    }
    return {};
}

// ---------------------------------------------------------------------------
// Build Vosk grammar JSON from word list
// ---------------------------------------------------------------------------

static std::string build_grammar(const std::vector<std::string>& words) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) oss << ", ";
        oss << '"' << words[i] << '"';
    }
    oss << ", \"[unk]\"]";
    return oss.str();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <wav_file> <model_dir> <word1> [word2 ...]\n"
            "  e.g. %s output.wav model/vosk-model-small-en-us-0.15 cat dog person\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* wav_path   = argv[1];
    const char* model_path = argv[2];

    std::vector<std::string> object_list;
    for (int i = 3; i < argc; ++i)
        object_list.push_back(argv[i]);

    // -----------------------------------------------------------------------
    // Open WAV
    // -----------------------------------------------------------------------
    std::ifstream wav(wav_path, std::ios::binary);
    if (!wav) {
        std::fprintf(stderr, "Cannot open: %s\n", wav_path);
        return 1;
    }
    try {
        read_wav_header(wav, wav_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "WAV error: %s\n", e.what());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Load Vosk model and build grammar-restricted recogniser
    // -----------------------------------------------------------------------
    vosk_set_log_level(-1);   // suppress Vosk's stdout noise

    std::printf("Loading model: %s\n", model_path);
    VoskModel* model = vosk_model_new(model_path);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    std::string grammar = build_grammar(object_list);
    std::printf("Grammar: %s\n\n", grammar.c_str());

    VoskRecognizer* rec = vosk_recognizer_new_grm(model, VOSK_RATE, grammar.c_str());
    if (!rec) {
        std::fputs("Failed to create recogniser\n", stderr);
        vosk_model_free(model);
        return 1;
    }
    vosk_recognizer_set_words(rec, 0);

    // -----------------------------------------------------------------------
    // Feed WAV into Vosk chunk by chunk
    // -----------------------------------------------------------------------
    auto match = [&](const std::string& text) -> std::string {
        if (text.empty() || text == "[unk]") return {};
        for (const auto& obj : object_list)
            if (text.find(obj) != std::string::npos)
                return obj;
        return {};
    };

    std::vector<int16_t> chunk(CHUNK_FRAMES);
    size_t total_frames = 0;
    int    detections   = 0;

    std::puts("─────────────────────────────────────");
    std::puts("  time(s)   type      word");
    std::puts("─────────────────────────────────────");

    while (wav.read(reinterpret_cast<char*>(chunk.data()),
                    CHUNK_FRAMES * sizeof(int16_t))) {

        int final_flag = vosk_recognizer_accept_waveform_s(
            rec, chunk.data(), static_cast<int>(CHUNK_FRAMES));

        double t = static_cast<double>(total_frames) / VOSK_RATE;
        total_frames += CHUNK_FRAMES;

        if (final_flag) {
            std::string text = extract_field(
                vosk_recognizer_result(rec), "text");
            if (auto hit = match(text); !hit.empty()) {
                std::printf("  %7.2f    final     %s\n", t, hit.c_str());
                ++detections;
            }
        } else {
            std::string partial = extract_field(
                vosk_recognizer_partial_result(rec), "partial");
            if (auto hit = match(partial); !hit.empty()) {
                std::printf("  %7.2f    partial   %s\n", t, hit.c_str());
                ++detections;
            }
        }
    }

    // Flush any remaining audio
    {
        const char* json = vosk_recognizer_final_result(rec);
        std::string text = extract_field(json, "text");
        double t = static_cast<double>(total_frames) / VOSK_RATE;
        if (auto hit = match(text); !hit.empty()) {
            std::printf("  %7.2f    final     %s  [flushed]\n", t, hit.c_str());
            ++detections;
        }
    }

    std::puts("─────────────────────────────────────");
    std::printf("Total detections: %d\n", detections);

    vosk_recognizer_free(rec);
    vosk_model_free(model);
    return 0;
}