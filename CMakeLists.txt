cmake_minimum_required(VERSION 3.16)
project(perception_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find libcamera
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Find libpng
find_package(PNG REQUIRED)

# Define the executable
add_executable(perception_system
    src/main.cpp
    src/cam_downsampler.cpp
)

# Include directories
target_include_directories(perception_system PRIVATE
    include/
    ${LIBCAMERA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(perception_system PRIVATE
    ${LIBCAMERA_LIBRARIES}
    PNG::PNG
)

# libcamera requires these compile flags
target_compile_options(perception_system PRIVATE
    ${LIBCAMERA_CFLAGS_OTHER}
)