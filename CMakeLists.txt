cmake_minimum_required(VERSION 3.16)
project(NorthStar CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA       REQUIRED alsa)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

find_path(VOSK_INCLUDE_DIR  vosk_api.h  HINTS /usr/local/include /usr/include)
find_library(VOSK_LIB       vosk        HINTS /usr/local/lib     /usr/lib)
if(NOT VOSK_INCLUDE_DIR OR NOT VOSK_LIB)
    message(FATAL_ERROR "Vosk not found — install libvosk.so and vosk_api.h")
endif()

# ── Shared compile options ────────────────────────────────────────────────────

set(COMMON_INCLUDES
    include
    ${ALSA_INCLUDE_DIRS}
    ${SAMPLERATE_INCLUDE_DIRS}
    ${VOSK_INCLUDE_DIR}
)

set(COMMON_LIBS
    ${ALSA_LIBRARIES}
    ${SAMPLERATE_LIBRARIES}
    ${VOSK_LIB}
    pthread
)

set(COMMON_FLAGS
    -Wall -Wextra
    -march=armv8-a   # Cortex-A76 (Pi 5) — enables NEON SIMD
    -O2
)

# ── Core library (shared by main driver and test programs) ────────────────────

add_library(northstar_core STATIC
    src/audio_capture.cpp
    src/resampler.cpp
    src/intent_detector.cpp
    src/pipeline.cpp
)
target_include_directories(northstar_core PUBLIC ${COMMON_INCLUDES})
target_link_libraries(northstar_core PUBLIC ${COMMON_LIBS})
target_compile_options(northstar_core PRIVATE ${COMMON_FLAGS})

# ── Main driver ───────────────────────────────────────────────────────────────

add_executable(intent_driver src/main.cpp)
target_link_libraries(intent_driver PRIVATE northstar_core)
target_compile_options(intent_driver PRIVATE ${COMMON_FLAGS})

# ── Test / diagnostic programs ────────────────────────────────────────────────
# These are standalone — they don't link northstar_core.

add_executable(test_alsa_open src/test_alsa_open.cpp)
target_include_directories(test_alsa_open PRIVATE include ${ALSA_INCLUDE_DIRS})
target_link_libraries(test_alsa_open PRIVATE ${ALSA_LIBRARIES})
target_compile_options(test_alsa_open PRIVATE ${COMMON_FLAGS})

add_executable(record_prepare src/record_prepare.cpp)
target_include_directories(record_prepare PRIVATE include ${ALSA_INCLUDE_DIRS} ${SAMPLERATE_INCLUDE_DIRS})
target_link_libraries(record_prepare PRIVATE ${ALSA_LIBRARIES} ${SAMPLERATE_LIBRARIES})
target_compile_options(record_prepare PRIVATE ${COMMON_FLAGS})

add_executable(record_playback src/record_playback.cpp)
target_include_directories(record_playback PRIVATE include ${ALSA_INCLUDE_DIRS})
target_link_libraries(record_playback PRIVATE ${ALSA_LIBRARIES})
target_compile_options(record_playback PRIVATE ${COMMON_FLAGS})

add_executable(infer src/infer.cpp)
target_include_directories(infer PRIVATE include ${VOSK_INCLUDE_DIR})
target_link_libraries(infer PRIVATE ${VOSK_LIB})
target_compile_options(infer PRIVATE ${COMMON_FLAGS})
