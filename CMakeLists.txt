cmake_minimum_required(VERSION 3.16)
project(pi_cam_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────
#  Build type
# ─────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────
#  Dependencies
# ─────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# hailort does not ship a .pc file -- paths confirmed on target system
set(HAILORT_INCLUDE_DIRS /usr/include)
set(HAILORT_LIBRARIES    /usr/lib/libhailort.so)

# ─────────────────────────────────────────────
#  Shared include path for all targets
# src/ is the root, plus each subdirectory so
# files can include siblings without a prefix
# ─────────────────────────────────────────────
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(COMMON_INCLUDES
    ${SRC}
    ${SRC}/capture
    ${SRC}/inference
    ${SRC}/preprocess
    ${SRC}/util
    ${OpenCV_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
    ${HAILORT_INCLUDE_DIRS}
)

set(COMMON_LIBS
    ${OpenCV_LIBS}
    ${LIBCAMERA_LIBRARIES}
    ${HAILORT_LIBRARIES}
    pthread
)

# ─────────────────────────────────────────────
#  Shared library: capture + inference + utils
# ─────────────────────────────────────────────
add_library(pi_cam_lib STATIC
    ${SRC}/capture/camera_capture.cpp
    ${SRC}/capture/fake_capture.cpp
    ${SRC}/preprocess/downsampler.cpp
    ${SRC}/inference/hailo8_inference.cpp
    ${SRC}/inference/halio8_helper.cpp
    ${SRC}/util/detection_utils.cpp
)

target_include_directories(pi_cam_lib PUBLIC ${COMMON_INCLUDES})
target_link_libraries(pi_cam_lib PUBLIC ${COMMON_LIBS})

# ─────────────────────────────────────────────
#  Executables
# ─────────────────────────────────────────────

# Main entry point
add_executable(main ${SRC}/main.cpp)
target_link_libraries(main PRIVATE pi_cam_lib)

# Live camera + Hailo detection
add_executable(live_detection ${SRC}/live_detection.cpp)
target_link_libraries(live_detection PRIVATE pi_cam_lib)

# Throughput benchmark (camera only, no inference)
add_executable(camera_throughput ${SRC}/camera_throughput.cpp)
target_link_libraries(camera_throughput PRIVATE pi_cam_lib)

# Hailo inference tests
add_executable(test_hailo          ${SRC}/test_hailo.cpp)
add_executable(test_hailo_debug    ${SRC}/test_hailo_debug.cpp)
add_executable(test_model          ${SRC}/test_model.cpp)
add_executable(benchmark_detection ${SRC}/benchmark_detection.cpp)

foreach(t test_hailo test_hailo_debug test_model benchmark_detection)
    target_link_libraries(${t} PRIVATE pi_cam_lib)
endforeach()

# ─────────────────────────────────────────────
#  Compiler warnings
# ─────────────────────────────────────────────
set(WARN_FLAGS -Wall -Wextra -Wpedantic -Wno-unused-parameter)

foreach(t pi_cam_lib main live_detection camera_throughput
          test_hailo test_hailo_debug test_model benchmark_detection)
    target_compile_options(${t} PRIVATE ${WARN_FLAGS})
endforeach()