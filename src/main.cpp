#include <iostream>
#include <thread>
#include <chrono>
#include "capture_controller.hpp"
#include "frame_saver.hpp"

int main() {
    std::cout << "Initialising capture controller..." << std::endl;
    CaptureController controller;

    std::cout << "Starting capture..." << std::endl;
    controller.start_capture();

    // Block main thread for slightly longer than capture duration
    std::this_thread::sleep_for(std::chrono::seconds(6));

    std::cout << "Capture complete." << std::endl;
    std::cout << "Frames in queue: " << controller.queue_size() << std::endl;

    // Save all frames to disk
    FrameSaver saver(controller.get_queue(), "./capture_output");
    saver.save_all();

    return 0;
}