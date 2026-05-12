/*
    capture_frames.cpp  —  NorthStar training data capture

    Space  → capture 50 post-downsampled 640x640 frames from both cameras
    q      → quit

    Output: captured_frames/frame_cam<id>_<DDMMYYYY_HHMMSS>_<n>.png
*/

#include "capture_controller.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <string>
#include <termios.h>
#include <unistd.h>

static constexpr int         FRAMES_TO_CAPTURE = 50;
static constexpr const char* OUTPUT_DIR        = "captured_frames";

// Read one keypress without waiting for Enter
static char getch()
{
    struct termios old, raw;
    tcgetattr(STDIN_FILENO, &old);
    raw = old;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    char c;
    read(STDIN_FILENO, &c, 1);
    tcsetattr(STDIN_FILENO, TCSANOW, &old);
    return c;
}

static void capture_batch(CameraQueue& cam_q, int batch)
{
    std::time_t now = std::time(nullptr);
    std::tm*    tm  = std::localtime(&now);
    char        ts[32];
    std::strftime(ts, sizeof(ts), "%d%m%Y_%H%M%S", tm);

    int saved = 0;
    while (saved < FRAMES_TO_CAPTURE) {
        auto pkt = cam_q.pop();
        if (!pkt) break;

        // Frame data is 640x640 RGB — flip to BGR before imwrite
        cv::Mat frame(640, 640, CV_8UC3, pkt->data.data());
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        std::string path = std::string(OUTPUT_DIR) + "/frame_cam"
                         + std::to_string(pkt->camera_id) + "_"
                         + ts + "_b" + std::to_string(batch)
                         + "_" + std::to_string(saved) + ".png";

        if (!cv::imwrite(path, frame))
            std::cerr << "\nFailed to write: " << path << "\n";

        ++saved;
        std::printf("\r  [batch %d]  Saved %d/%d", batch, saved, FRAMES_TO_CAPTURE);
        std::fflush(stdout);
    }

    std::printf("\n  Done — %d frames saved.\n\n", saved);
}

int main()
{
    std::filesystem::create_directories(OUTPUT_DIR);

    CaptureController controller;
    controller.start_capture();
    auto& cam_q = controller.get_cam_queue();

    std::cout << "Ready. Space = capture " << FRAMES_TO_CAPTURE
              << " frames,  q = quit\n\n";

    int batch = 0;
    while (true) {
        char key = getch();
        if (key == 'q' || key == 'Q') break;
        if (key == ' ') capture_batch(cam_q, ++batch);
    }

    controller.stop_capture();
    std::printf("Quit — %d batch(es) captured to ./%s/\n", batch, OUTPUT_DIR);
    return 0;
}