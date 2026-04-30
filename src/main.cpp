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
    
    // Start saving frames concurrently on a separate thread
    std::thread saver_thread([&]() {
        FrameSaver saver(controller.get_queue(), "./capture_output");
        saver.save_all();
    });

    // Block main thread for slightly longer than capture duration
    std::this_thread::sleep_for(std::chrono::seconds(6));
    
    // Join the saver thread
    if (saver_thread.joinable()) {
        saver_thread.join();
    }

    std::cout << "Capture complete." << std::endl;

    return 0;
}
