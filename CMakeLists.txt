cmake_minimum_required(VERSION 3.16)
project(northstar_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)

add_executable(test_stereo_depth
    src/test_stereo_depth.cpp
    src/stereo_distance.cpp
)
target_include_directories(test_stereo_depth PRIVATE include)
target_link_libraries(test_stereo_depth PRIVATE ${OpenCV_LIBS})