cmake_minimum_required(VERSION 3.16)
project(northstar_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────────
#  Model configuration
# ─────────────────────────────────────────────────────────────────────────────
set(NORTHSTAR_MODEL_DIR "${CMAKE_SOURCE_DIR}/model_inf"
    CACHE PATH "Directory containing .hef model files")

add_compile_definitions(
    NORTHSTAR_MODEL_DIR="${NORTHSTAR_MODEL_DIR}"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Dependencies
# ─────────────────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
find_package(OpenCV    REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads   REQUIRED)

# Camera
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Hailo — libs at /usr/lib, headers at /usr/include on this Pi
find_library(HAILO_LIB  hailort   HINTS /usr/lib               REQUIRED)
find_path   (HAILO_INCLUDE hailo/hailort.hpp HINTS /usr/include REQUIRED)

# Audio — ALSA, libsamplerate, Vosk
pkg_check_modules(ALSA       REQUIRED alsa)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

find_library(VOSK_LIB   vosk       HINTS /usr/local/lib /usr/lib REQUIRED)
find_path   (VOSK_INCLUDE vosk_api.h HINTS /usr/local/include /usr/include REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
#  Include directories
# ─────────────────────────────────────────────────────────────────────────────
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HAILO_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

set(AUDIO_INCLUDES
    ${VOSK_INCLUDE}
    ${ALSA_INCLUDE_DIRS}
    ${SAMPLERATE_INCLUDE_DIRS}
)

# ─────────────────────────────────────────────────────────────────────────────
#  Source groups
# ─────────────────────────────────────────────────────────────────────────────
set(INFERENCE_SOURCES
    src/inference/hailo8_inference.cpp
    src/detection_utils.cpp
)

set(CAMERA_SOURCES
    src/cam_downsampler.cpp
    src/ir_capture.cpp
    src/MLX90640.cpp
    src/ThermalUtils.cpp
)

set(AUDIO_SOURCES
    src/audio_capture.cpp
    src/resampler.cpp
    src/intent_detector.cpp
    src/pipeline.cpp
    src/detection_filter.cpp
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper macros
# ─────────────────────────────────────────────────────────────────────────────

# Hailo + OpenCV only
macro(configure_target TARGET)
    target_include_directories(${TARGET} PRIVATE ${COMMON_INCLUDES})
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)
    target_link_libraries(${TARGET} PRIVATE
        ${HAILO_LIB}
        ${OpenCV_LIBS}
        Threads::Threads
    )
endmacro()

# Hailo + OpenCV + libcamera
macro(configure_camera_target TARGET)
    configure_target(${TARGET})
    target_include_directories(${TARGET} PRIVATE ${LIBCAMERA_INCLUDE_DIRS})
    target_compile_options(${TARGET} PRIVATE ${LIBCAMERA_CFLAGS_OTHER})
    target_link_libraries(${TARGET} PRIVATE ${LIBCAMERA_LIBRARIES})
endmacro()

# Hailo + ALSA + samplerate + Vosk (no libcamera)
macro(configure_audio_target TARGET)
    target_include_directories(${TARGET} PRIVATE
        ${COMMON_INCLUDES}
        ${AUDIO_INCLUDES}
    )
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)
    target_link_libraries(${TARGET} PRIVATE
        ${HAILO_LIB}
        ${OpenCV_LIBS}
        ${VOSK_LIB}
        ${ALSA_LIBRARIES}
        ${SAMPLERATE_LIBRARIES}
        Threads::Threads
    )
endmacro()

# ─────────────────────────────────────────────────────────────────────────────
#  perception_system  —  full camera + Hailo pipeline
# ─────────────────────────────────────────────────────────────────────────────
add_executable(perception_system
    src/main.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
    ${AUDIO_SOURCES}
)
configure_camera_target(perception_system)
target_include_directories(perception_system PRIVATE ${AUDIO_INCLUDES})
target_link_libraries(perception_system PRIVATE
    ${VOSK_LIB}
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
)

# ─────────────────────────────────────────────────────────────────────────────
#  mic_pipeline  —  microphone + Vosk + DetectionFilter
#
#  Standalone test of the audio side: speak a COCO object name, verify
#  on_intent fires with the correct object_id.
#  Note: save the mic main as src/mic_pipeline_main.cpp
#
#  Usage: ./mic_pipeline ../model/vosk-model-small-en-us-0.15 cat dog person
# ─────────────────────────────────────────────────────────────────────────────
add_executable(mic_pipeline
    src/mic_pipeline_main.cpp
    ${AUDIO_SOURCES}
)
configure_audio_target(mic_pipeline)

# ─────────────────────────────────────────────────────────────────────────────
#  Other camera targets
# ─────────────────────────────────────────────────────────────────────────────
add_executable(live_detection
    src/live_detection.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(live_detection)

add_executable(benchmark_detection
    src/benchmark_detection.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(benchmark_detection)

add_executable(sim_session
    src/sim_session.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(sim_session)

add_executable(save_annotated
    src/save_annotated.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
)
configure_camera_target(save_annotated)


add_executable(test_detection_filter
    src/test_detection_filter.cpp
    src/detection_filter.cpp
)
configure_target(test_detection_filter)

add_executable(test_pipeline_logic
    src/test_pipeline_logic.cpp
)
configure_target(test_pipeline_logic)

add_executable(test_save_pairs
    src/test_save_pairs.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
    ${AUDIO_SOURCES}
)
configure_camera_target(test_save_pairs)
target_include_directories(test_save_pairs PRIVATE ${AUDIO_INCLUDES})
target_link_libraries(test_save_pairs PRIVATE
    ${VOSK_LIB}
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
)

add_executable(capture_frames
    src/capture_frames.cpp
    ${CAMERA_SOURCES}
)
configure_camera_target(capture_frames)
 
