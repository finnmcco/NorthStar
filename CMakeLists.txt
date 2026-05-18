cmake_minimum_required(VERSION 3.16)
project(northstar_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
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

# Speech output — PulseAudio simple API (PipeWire's PA compat layer fronts the
# I²S MAX98357A on the Pi 5). Required by integrated_main_audio.
pkg_check_modules(PULSE REQUIRED IMPORTED_TARGET libpulse-simple libpulse)

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
    src/hailo8_inference.cpp
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


#add_executable(test_save_pairs
    #src/test_save_pairs.cpp
    #${INFERENCE_SOURCES}
    #${CAMERA_SOURCES}
    #${AUDIO_SOURCES}
#)
#configure_camera_target(test_save_pairs)
#target_include_directories(test_save_pairs PRIVATE ${AUDIO_INCLUDES})
#target_link_libraries(test_save_pairs PRIVATE
    #${VOSK_LIB}
    #${ALSA_LIBRARIES}
    #${SAMPLERATE_LIBRARIES}
#)

add_executable(integrated_main
    src/integrated_main_2.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
    ${AUDIO_SOURCES}
    src/gpio.cpp
    src/button-driver.cpp
    src/stereo_distance.cpp
    src/ir_aligner.cpp
)
configure_camera_target(integrated_main)
target_include_directories(integrated_main PRIVATE ${AUDIO_INCLUDES})
target_link_libraries(integrated_main PRIVATE
    ${VOSK_LIB}
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    gpiodcxx
)

# ─────────────────────────────────────────────────────────────────────────────
#  integrated_main_audio  --  same pipeline as integrated_main, plus Piper TTS
#  output through the I²S speaker.  Source lives in src/integrated_main_audio.cpp
#  and pulls in the TTS engine (tts.cpp) + libpulse speaker wrapper (speaker.cpp)
#  from the NorthStar-tts_pipeline drop-in.
# ─────────────────────────────────────────────────────────────────────────────
add_executable(integrated_main_audio
    src/integrated_main_audio.cpp
    src/speaker.cpp
    src/tts.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
    ${AUDIO_SOURCES}
    src/gpio.cpp
    src/button-driver.cpp
    src/stereo_distance.cpp
    src/ir_aligner.cpp
)
configure_camera_target(integrated_main_audio)
target_include_directories(integrated_main_audio PRIVATE ${AUDIO_INCLUDES})
target_link_libraries(integrated_main_audio PRIVATE
    ${VOSK_LIB}
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    gpiodcxx
    PkgConfig::PULSE
)

add_executable(stereo_debug_main
    src/stereo_debug_main.cpp
    ${INFERENCE_SOURCES}
    ${CAMERA_SOURCES}
    ${AUDIO_SOURCES}
    src/gpio.cpp
    src/button-driver.cpp
    src/stereo_distance.cpp
)
configure_camera_target(stereo_debug_main)
target_include_directories(stereo_debug_main PRIVATE ${AUDIO_INCLUDES})
target_link_libraries(stereo_debug_main PRIVATE
    ${VOSK_LIB}
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    gpiodcxx
)

add_executable(test_stereo_depth
    src/test_stereo_depth.cpp
    src/stereo_distance.cpp
)
target_include_directories(test_stereo_depth PRIVATE include)
target_link_libraries(test_stereo_depth PRIVATE ${OpenCV_LIBS})
#configure_camera_target(capture_frames)
 
add_executable(ir_calib_capture
    src/ir_calib_capture.cpp
    ${CAMERA_SOURCES}
)
configure_camera_target(ir_calib_capture)
target_link_libraries(ir_calib_capture PRIVATE
    ${OpenCV_LIBS}
)

add_executable(capture_frames
    src/capture_frames.cpp
    ${CAMERA_SOURCES}
    src/gpio.cpp
    src/button-driver.cpp
)
target_link_libraries(capture_frames PRIVATE
    gpiodcxx
)
configure_camera_target(capture_frames)

# ─────────────────────────────────────────────────────────────────────────────
#  audio_record  --  capture raw mic audio (gain=1.0) to AudioTests/
#  Usage: ./audio_record [seconds]
# ─────────────────────────────────────────────────────────────────────────────
add_executable(audio_record
    src/audio_record.cpp
    src/resampler.cpp
)
target_include_directories(audio_record PRIVATE ${COMMON_INCLUDES} ${AUDIO_INCLUDES})
target_compile_options(audio_record PRIVATE -Wall -Wextra -O2)
target_link_libraries(audio_record PRIVATE
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    Threads::Threads
)
 
# ─────────────────────────────────────────────────────────────────────────────
#  audio_test  --  run a WAV file through Vosk with parameterised gain
#  Usage: ./audio_test <wav> <gain> <model_dir> <word1> [word2 ...]
# ─────────────────────────────────────────────────────────────────────────────
add_executable(audio_test
    src/audio_test.cpp
    src/intent_detector.cpp
)
target_include_directories(audio_test PRIVATE ${COMMON_INCLUDES} ${AUDIO_INCLUDES})
target_compile_options(audio_test PRIVATE -Wall -Wextra -O2)
target_link_libraries(audio_test PRIVATE
    ${VOSK_LIB}
    Threads::Threads
)

add_executable(stereo_debug_static
    src/stereo_debug_static.cpp
    src/stereo_distance.cpp
)
configure_camera_target(stereo_debug_static)

add_executable(stereo_infer_static
    src/stereo_infer_static.cpp
    ${INFERENCE_SOURCES}
    src/stereo_distance.cpp
)
target_include_directories(stereo_infer_static PRIVATE include)
target_link_libraries(stereo_infer_static PRIVATE
    ${OpenCV_LIBS}
    ${HAILO_LIB}
)