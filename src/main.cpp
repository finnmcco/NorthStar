#include "pipeline.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// g_running is referenced by ChunkQueue (chunk_queue.hpp) and pipeline.cpp.
std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <model_dir> <word1> [word2 ...]\n"
            "  e.g. %s ../model/vosk-model-small-en-us-0.15 cat dog person car\n",
            argv[0], argv[0]);
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    Pipeline::Config cfg;
    cfg.model_path = argv[1];
    for (int i = 2; i < argc; ++i)
        cfg.object_list.emplace_back(argv[i]);

    cfg.on_detection = [](const DetectionResult& r) {
        std::printf("  DETECTED: %-12s  (%s)\n",
                    r.word.c_str(),
                    r.is_final ? "final" : "partial");
        std::fflush(stdout);
    };

    try {
        Pipeline pipeline(std::move(cfg));
        pipeline.start();

        std::printf("\nListening — Ctrl+C to stop.\n\n");
        while (g_running)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        pipeline.stop();
        std::printf("\nQueue drops during run: %zu\n", pipeline.queue_drops());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }

    return 0;
}
