cmake_minimum_required(VERSION 3.18)
project(tts_driver LANGUAGES CXX)

# ── Compiler settings ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-mcpu=cortex-a76>
    $<$<CONFIG:Release>:-march=armv8.2-a>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
)
add_link_options(
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fsanitize=undefined>
)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSE REQUIRED IMPORTED_TARGET libpulse-simple libpulse)

# ── Shared library (speaker + tts) ───────────────────────────────────────────
add_library(tts_lib STATIC
    src/speaker.cpp
    src/tts.cpp
)
target_include_directories(tts_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(tts_lib PUBLIC
    PkgConfig::PULSE
)

# ── Main executable: tts_speak ────────────────────────────────────────────────
add_executable(tts_speak src/main.cpp)
target_link_libraries(tts_speak PRIVATE tts_lib)

# ── Test: test_speaker ────────────────────────────────────────────────────────
add_executable(test_speaker tests/test_speaker.cpp)
target_link_libraries(test_speaker PRIVATE tts_lib)

# ── Test: test_tts ────────────────────────────────────────────────────────────
add_executable(test_tts tests/test_tts.cpp)
target_link_libraries(test_tts PRIVATE tts_lib)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS tts_speak DESTINATION bin)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "Targets     : tts_speak  test_speaker  test_tts")
message(STATUS "libpulse    : ${PULSE_LIBRARIES}")