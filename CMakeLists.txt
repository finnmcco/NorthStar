cmake_minimum_required(VERSION 3.16)
project(northstar_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
#  Model configuration
#
#  Copy your .hef file into model_inf/ then set NORTHSTAR_HEF_FILE to its name.
#  Override on the command line if needed:
#      cmake -DNORTHSTAR_HEF_FILE=yolov8n_h8.hef ..
#
#  DEFAULT_HEF_PATH is injected as a compile definition so every binary
#  resolves the model relative to the repo root at build time — no hardcoded
#  system paths anywhere in the source.
# ─────────────────────────────────────────────────────────────────────────────
set(NORTHSTAR_MODEL_DIR "${CMAKE_SOURCE_DIR}/model_inf"
    CACHE PATH   "Directory containing .hef model files")

# Only the directory is injected here.  The filename is set in src/config.hpp —
# edit that file and run make to switch models, no cmake re-run needed.
add_compile_definitions(
    NORTHSTAR_MODEL_DIR="${NORTHSTAR_MODEL_DIR}"
)


find_package(PkgConfig   REQUIRED)
find_package(OpenCV      REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads     REQUIRED)

pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# HailoRT is installed to a non-standard prefix on the Pi AI HAT+.
# Adjust HAILO_ROOT if your installation differs.
set(HAILO_ROOT "/usr/local/hailo" CACHE PATH "HailoRT installation root")
find_library(HAILO_LIB hailort HINTS "${HAILO_ROOT}/lib" REQUIRED)
find_path(HAILO_INCLUDE hailo/hailort.hpp HINTS "${HAILO_ROOT}/include" REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  Common include directories
# ─────────────────────────────────────────────────────────────────────────────
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HAILO_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

# ─────────────────────────────────────────────────────────────────────────────
#  Source groups
# ─────────────────────────────────────────────────────────────────────────────

# Inference — always needed
set(INFERENCE_SOURCES
    src/inference/hailo8_inference.cpp
    src/detection_utils.cpp
)

# Capture — real libcamera pipeline
set(CAMERA_SOURCES
    src/cam_downsampler.cpp
)

# Image folder capture — no libcamera, used by image_bench
set(IMAGE_CAPTURE_SOURCES
    src/image_capture.cpp
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper macro: apply common settings to a target
# ─────────────────────────────────────────────────────────────────────────────
macro(configure_target TARGET)
    target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDES})
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)
    target_link_libraries(${TARGET} PRIVATE
        ${HAILO_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
    )
endmacro()

macro(configure_camera_target TARGET)
    configure_target(${TARGET})
    target_include_directories(${TARGET} PRIVATE ${LIBCAMERA_INCLUDE_DIRS})
    target_compile_options(${TARGET} PRIVATE ${LIBCAMERA_CFLAGS_OTHER})
    target_link_libraries(${TARGET} PRIVATE ${LIBCAMERA_LIBRARIES})
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
#  perception_system  —  main pipeline (dual camera + Hailo)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(perception_system
    src/main.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(perception_system)

# ─────────────────────────────────────────────────────────────────────────────
#  live_detection  —  single-camera live demo
# ─────────────────────────────────────────────────────────────────────────────
add_executable(live_detection
    src/live_detection.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(live_detection)

# ─────────────────────────────────────────────────────────────────────────────
#  benchmark_detection  —  throughput benchmark with real camera
# ─────────────────────────────────────────────────────────────────────────────
add_executable(benchmark_detection
    src/benchmark_detection.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(benchmark_detection)

# ─────────────────────────────────────────────────────────────────────────────
#  image_bench  —  throughput benchmark from image folder (no libcamera)
#
#  Usage: ./image_bench <image_root> [fps]
#
#  image_root/
#      cam0/   <- 640x640 PNG/JPEG images for camera 0
#      cam1/   <- 640x640 PNG/JPEG images for camera 1 (optional)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Tests  (no libcamera dependency — run on the Pi without cameras attached)
# ─────────────────────────────────────────────────────────────────────────────

# test_single_frame
# Sends one synthetic frame through Hailo and checks the callback fires with
# a correctly-sized output buffer.
# Usage: ./test_single_frame [hef_path] [image_path]
add_executable(test_single_frame
    src/test_single_frame.cpp
    ${INFERENCE_SOURCES}
)
configure_target(test_single_frame)

# test_queue_ingest
# Pushes N synthetic frames into a queue, runs the consumer loop, and asserts
# that exactly N callbacks arrive with valid data.
# Usage: ./test_queue_ingest [hef_path] [n_frames]
add_executable(test_queue_ingest
    src/test_queue_ingest.cpp
    ${INFERENCE_SOURCES}
)
configure_target(test_queue_ingest)

# ─────────────────────────────────────────────────────────────────────────────
#  camera_throughput  —  raw camera frame rate test (no Hailo)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(camera_throughput
    src/camera_throughput.cpp
    ${CAMERA_SOURCES}
)
configure_camera_target(camera_throughput)

# test_detection_output
# Runs one real image through inference and prints a detection table + saves
# an annotated PNG.  For visual verification of model correctness.
# Usage: ./test_detection_output <image_path> [hef_path]
add_executable(test_detection_output
    src/test_detection_output.cpp
    ${INFERENCE_SOURCES}
)
configure_target(test_detection_output)

# ─────────────────────────────────────────────────────────────────────────────
#  test_camera_inference  —  capture for N seconds, print all InferencePackets
# ─────────────────────────────────────────────────────────────────────────────
add_executable(test_camera_inference
    src/test_camera_inference.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(test_camera_inference)