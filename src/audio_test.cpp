/*
    audio_test.cpp  --  batch test all WAVs in AudioTests/ against Vosk

    Usage:
        ./audio_test <gain> <model_dir> <word1> [word2 ...]

    Counts only FINAL detections in the summary.
    Partials are shown in the output and saved to the results file.
*/

#include "audio_config.hpp"
#include "intent_detector.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// -- WAV reader ---------------------------------------------------------------

struct WavInfo {
    uint32_t sample_rate     = 0;
    uint16_t channels        = 0;
    uint16_t bits_per_sample = 0;
    uint32_t num_samples     = 0;
};

static uint32_t read_u32_le(std::ifstream& f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b), 4);
    return uint32_t(b[0]) | (uint32_t(b[1])<<8) |
           (uint32_t(b[2])<<16) | (uint32_t(b[3])<<24);
}
static uint16_t read_u16_le(std::ifstream& f) {
    uint8_t b[2]; f.read(reinterpret_cast<char*>(b), 2);
    return uint16_t(b[0]) | (uint16_t(b[1])<<8);
}

static bool parse_wav(std::ifstream& f, WavInfo& info)
{
    char tag[5] = {};
    f.read(tag, 4);
    if (std::strncmp(tag, "RIFF", 4) != 0) {
        std::fprintf(stderr, "  [wav] not a RIFF file\n"); return false;
    }
    read_u32_le(f);
    f.read(tag, 4);
    if (std::strncmp(tag, "WAVE", 4) != 0) {
        std::fprintf(stderr, "  [wav] not WAVE\n"); return false;
    }

    uint32_t data_bytes = 0;
    bool got_fmt = false, got_data = false;

    while (!f.eof()) {
        f.read(tag, 4);
        if (f.gcount() < 4) break;
        const uint32_t chunk_size = read_u32_le(f);

        if (std::strncmp(tag, "fmt ", 4) == 0) {
            const uint16_t audio_fmt = read_u16_le(f);
            if (audio_fmt != 1) {
                std::fprintf(stderr, "  [wav] only PCM supported\n");
                return false;
            }
            info.channels        = read_u16_le(f);
            info.sample_rate     = read_u32_le(f);
            read_u32_le(f);
            read_u16_le(f);
            info.bits_per_sample = read_u16_le(f);
            if (chunk_size > 16) f.seekg(chunk_size - 16, std::ios::cur);
            got_fmt = true;
        } else if (std::strncmp(tag, "data", 4) == 0) {
            data_bytes = chunk_size;
            got_data   = true;
            break;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!got_fmt || !got_data) {
        std::fprintf(stderr, "  [wav] missing fmt or data chunk\n");
        return false;
    }

    info.num_samples = data_bytes / (info.bits_per_sample / 8 * info.channels);
    return true;
}

// -- Per-file result ----------------------------------------------------------

struct FileResult {
    std::string file;
    int         finals   = 0;
    int         partials = 0;
    std::string log;     // full detection log for writing to file
};

// -- Process one WAV ----------------------------------------------------------

static FileResult process_file(const std::string& path,
                               float gain,
                               IntentDetector& detector)
{
    FileResult result;
    result.file = path;

    std::ifstream wav(path, std::ios::binary);
    if (!wav) {
        std::printf("  ERROR: cannot open file\n");
        result.log = "  ERROR: cannot open file\n";
        return result;
    }

    WavInfo info{};
    if (!parse_wav(wav, info)) return result;

    const float duration_s =
        static_cast<float>(info.num_samples) / info.sample_rate;

    // Header for console and file
    std::ostringstream log;
    log << "  duration: " << duration_s << " s  |  "
        << info.sample_rate << " Hz  "
        << info.channels << " ch  "
        << info.bits_per_sample << "-bit\n";

    std::printf("  duration: %.2f s  |  %u Hz  %u ch  %u-bit\n",
                duration_s, info.sample_rate,
                info.channels, info.bits_per_sample);

    if (info.bits_per_sample != 16) {
        std::printf("  ERROR: only 16-bit WAV supported\n");
        result.log = log.str();
        return result;
    }

    detector.reset();

    const std::string header =
        "  time(s)     type        word\n"
        "  -------     -------     --------\n";
    std::printf("%s", header.c_str());
    log << header;

    const std::size_t chunk_size = Config::PERIOD_FRAMES / 3;
    std::vector<int16_t> chunk(chunk_size);
    uint32_t samples_read = 0;

    while (samples_read < info.num_samples) {
        const uint32_t remaining = info.num_samples - samples_read;
        const std::size_t to_read =
            std::min(static_cast<std::size_t>(remaining), chunk_size);

        wav.read(reinterpret_cast<char*>(chunk.data()),
                 to_read * sizeof(int16_t));
        const std::size_t got = wav.gcount() / sizeof(int16_t);
        if (got == 0) break;

        chunk.resize(got);

        for (auto& s : chunk) {
            const float g = s * gain;
            s = static_cast<int16_t>(
                std::clamp(g,
                           static_cast<float>(INT16_MIN),
                           static_cast<float>(INT16_MAX)));
        }

        auto det = detector.feed(chunk);
        if (det) {
            const float t =
                static_cast<float>(samples_read) / info.sample_rate;
            const bool is_final = det->is_final;
            const char* type    = is_final ? "final" : "partial";

            // Console
            std::printf("  %-10.2f  %-10s  %-12s\n",
                        t, type, det->word.c_str());
            std::fflush(stdout);

            // Log
            char line[128];
            std::snprintf(line, sizeof(line),
                          "  %-10.2f  %-10s  %-12s\n",
                          t, type, det->word.c_str());
            log << line;

            if (is_final) ++result.finals;
            else          ++result.partials;
        }

        samples_read += static_cast<uint32_t>(got);
        chunk.resize(chunk_size);
    }

    // Counts line
    char counts[128];
    std::snprintf(counts, sizeof(counts),
                  "  Finals: %d   Partials: %d\n",
                  result.finals, result.partials);
    std::printf("%s", counts);
    log << counts;

    result.log = log.str();
    return result;
}

// -- main ---------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <gain> <model_dir> <word1> [word2 ...]\n"
            "  e.g. %s 8.0 ../model_inf/vosk-model-small-en-us-0.15 person cup\n",
            argv[0], argv[0]);
        return 1;
    }

    const float       gain       = std::atof(argv[1]);
    const std::string model_path = argv[2];

    std::vector<std::string> words;
    for (int i = 3; i < argc; ++i) words.push_back(argv[i]);

    if (gain <= 0.f) {
        std::fprintf(stderr, "gain must be > 0\n"); return 1;
    }

    const std::string audio_dir = "AudioTests";
    std::vector<std::string> wav_files;

    if (!std::filesystem::exists(audio_dir)) {
        std::fprintf(stderr, "AudioTests/ directory not found\n");
        return 1;
    }

    for (const auto& entry :
         std::filesystem::directory_iterator(audio_dir)) {
        if (entry.path().extension() == ".wav")
            wav_files.push_back(entry.path().string());
    }

    if (wav_files.empty()) {
        std::printf("No .wav files found in %s/\n", audio_dir.c_str());
        return 0;
    }

    std::sort(wav_files.begin(), wav_files.end());

    const std::string model_name =
        std::filesystem::path(model_path).filename().string();

    std::printf("\ngain  = %.1f\n", gain);
    std::printf("model = %s\n", model_name.c_str());
    std::printf("words =");
    for (const auto& w : words) std::printf(" %s", w.c_str());
    std::printf("\nfiles = %zu\n\n", wav_files.size());

    std::printf("Loading model...\n");
    IntentDetector detector(model_path, words);
    std::printf("Model loaded.\n\n");

    std::vector<FileResult> results;

    for (const auto& path : wav_files) {
        std::printf("════════════════════════════════════════════════\n");
        std::printf("FILE: %s\n", path.c_str());
        std::printf("────────────────────────────────────────────────\n");

        results.push_back(process_file(path, gain, detector));
    }

    // -- Summary (finals only)
    std::printf("\n════════════════════════════════════════════════\n");
    std::printf("SUMMARY  gain=%.1f  model=%s\n",
                gain, model_name.c_str());
    std::printf("  (finals only)\n");
    std::printf("────────────────────────────────────────────────\n");

    int total_finals = 0, total_partials = 0;
    for (const auto& r : results) {
        std::printf("  %-40s  finals=%-3d  partials=%d\n",
                    std::filesystem::path(r.file).filename().string().c_str(),
                    r.finals, r.partials);
        total_finals   += r.finals;
        total_partials += r.partials;
    }

    std::printf("────────────────────────────────────────────────\n");
    std::printf("Files: %zu   Total finals: %d   Total partials: %d\n",
                results.size(), total_finals, total_partials);
    std::printf("════════════════════════════════════════════════\n\n");

    // -- Save results file
    std::string gain_str = std::to_string(gain);
    gain_str = gain_str.substr(0, gain_str.find_last_not_of('0') + 1);
    if (gain_str.back() == '.') gain_str.back() = '_';
    gain_str += (gain_str.back() == '_') ? "0" : "";
    std::replace(gain_str.begin(), gain_str.end(), '.', '_');

    const std::string results_path =
        audio_dir + "/results_" + model_name.substr(11) +
        "_gain" + gain_str + ".txt";

    std::ofstream out(results_path);
    if (out) {
        out << "gain  = " << gain << "\n";
        out << "model = " << model_name << "\n";
        out << "words =";
        for (const auto& w : words) out << " " << w;
        out << "\nfiles = " << results.size() << "\n\n";

        for (const auto& r : results) {
            out << "════════════════════════════════════════════════\n";
            out << "FILE: "
                << std::filesystem::path(r.file).filename().string() << "\n";
            out << "────────────────────────────────────────────────\n";
            out << r.log;
            out << "\n";
        }

        out << "════════════════════════════════════════════════\n";
        out << "SUMMARY (finals only)\n";
        out << "────────────────────────────────────────────────\n";
        for (const auto& r : results) {
            out << "  "
                << std::filesystem::path(r.file).filename().string()
                << "  finals=" << r.finals
                << "  partials=" << r.partials << "\n";
        }
        out << "────────────────────────────────────────────────\n";
        out << "Total finals: " << total_finals
            << "   Total partials: " << total_partials << "\n";

        out.close();
        std::printf("Results saved to %s\n", results_path.c_str());
    } else {
        std::fprintf(stderr, "WARNING: could not write %s\n",
                     results_path.c_str());
    }

    return 0;
}
