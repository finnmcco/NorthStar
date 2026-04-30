#pragma once
#include <png.h>
#include <filesystem>
#include <iostream>
#include "frame_packet.hpp"
#include "camera_queue.hpp"

class FrameSaver {
public:
    FrameSaver(CameraQueue& queue, const std::string& output_dir)
        : queue_(queue), output_dir_(output_dir)
    {
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_dir_);
    }

    void save_all() {
        int saved = 0;
        while (true) {
            auto frame = queue_.pop();

            // nullopt means queue is stopped and drained
            if (!frame) {
                break;
            }

            save_frame(*frame);
            saved++;
        }
        std::cout << "Saved " << saved << " frames to " << output_dir_ << std::endl;
    }

private:
    CameraQueue& queue_;
    std::string output_dir_;

    void save_frame(const FramePacket& frame) {
        // Build filename: e.g. "frame_cam0_123456789.png"
        std::string filename = output_dir_ + "/frame_cam" +
                               std::to_string(frame.camera_id) + "_" +
                               std::to_string(frame.timestamp_us) + ".png";

        // Open file for writing
        FILE* fp = fopen(filename.c_str(), "wb");
        if (!fp) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Initialise libpng write struct
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                   nullptr, nullptr, nullptr);
        if (!png) {
            fclose(fp);
            return;
        }

        // Initialise libpng info struct
        png_infop info = png_create_info_struct(png);
        if (!info) {
            png_destroy_write_struct(&png, nullptr);
            fclose(fp);
            return;
        }

        // Set up error handling - if libpng hits an error it jumps here
        if (setjmp(png_jmpbuf(png))) {
            png_destroy_write_struct(&png, &info);
            fclose(fp);
            return;
        }

        // Tell libpng to write to our file
        png_init_io(png, fp);

        // Write image header
        // PNG_COLOR_TYPE_RGB = 3 channels (R, G, B)
        // 8 bit depth = 1 byte per channel
        png_set_IHDR(
            png, info,
            1920,           // image width
            1080,          // image height
            8,                     // bit depth
            PNG_COLOR_TYPE_RGB,    // colour type
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
        );
        png_write_info(png, info);

        // Write each row of pixel data
        // libpng expects an array of row pointers
        // each row is width * 3 bytes (R, G, B per pixel)
        const int row_stride = 1920 * 3;
        for (uint32_t y = 0; y < 1080; y++) {
            // pointer to the start of this row in the frame buffer
            png_bytep row = const_cast<png_bytep>(frame.data.data() + y * row_stride);
            png_write_row(png, row);
        }

        // Finalise the PNG file
        png_write_end(png, nullptr);

        // Clean up
        png_destroy_write_struct(&png, &info);
        fclose(fp);
    }
};
