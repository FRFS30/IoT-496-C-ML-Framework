cmake_minimum_required(VERSION 3.13)

# ============================================================
# iotids-runtime — CMake build for Raspberry Pi Pico 2W
# ============================================================
# Prerequisites:
#   1. Pico SDK cloned and PICO_SDK_PATH set:
#        export PICO_SDK_PATH=/path/to/pico-sdk
#   2. pico-tflmicro cloned alongside this project:
#        git clone https://github.com/raspberrypi/pico-tflmicro
#        export PICO_TFLMICRO_PATH=/path/to/pico-tflmicro
#   3. arm-none-eabi-gcc toolchain on PATH
#
# Build:
#   mkdir build && cd build
#   cmake -DPICO_BOARD=pico2_w ..
#   make -j$(nproc)
#
# After a successful build, flash build/iotids_runtime.uf2
# by holding BOOTSEL and copying the .uf2 to the Pico mass storage.
# ============================================================

# Pull in Pico SDK (must come before project())
include(pico_sdk_import.cmake)

project(iotids_runtime C CXX ASM)

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

# Initialise the Pico SDK
pico_sdk_init()

# ---- Configurable options ------------------------------------------------

set(IOTIDS_NUM_FEATURES   "77"       CACHE STRING "Number of DNN input features")
set(IOTIDS_ARENA_SIZE     "204800"   CACHE STRING "TFLM tensor arena size in bytes (default 200 KB)")
set(IOTIDS_THRESHOLD      "0.5"      CACHE STRING "Binary classification threshold (recalibrate post-quantization)")
set(IOTIDS_RING_CAPACITY  "8"        CACHE STRING "Ring buffer slot count (must be power of two)")
set(WIFI_SSID             ""         CACHE STRING "Wi-Fi SSID")
set(WIFI_PASSWORD         ""         CACHE STRING "Wi-Fi WPA2 passphrase")
set(SERVER_HOST           "192.168.1.100" CACHE STRING "FedAvg server IPv4 address")
set(SERVER_PORT           "5555"     CACHE STRING "FedAvg server TCP port")
set(UART_BAUD             "115200"   CACHE STRING "Debug UART baud rate")

# ---- TFLM dependency ------------------------------------------------------
# pico-tflmicro provides a CMake target 'pico_tflmicro' that handles all
# TFLM source compilation, include paths, and linker flags.

if(NOT DEFINED ENV{PICO_TFLMICRO_PATH} AND NOT PICO_TFLMICRO_PATH)
    message(FATAL_ERROR
        "PICO_TFLMICRO_PATH is not set.\n"
        "Clone: git clone https://github.com/raspberrypi/pico-tflmicro\n"
        "Then:  cmake -DPICO_TFLMICRO_PATH=/path/to/pico-tflmicro ..")
endif()

if(NOT PICO_TFLMICRO_PATH)
    set(PICO_TFLMICRO_PATH $ENV{PICO_TFLMICRO_PATH})
endif()

add_subdirectory(${PICO_TFLMICRO_PATH} ${CMAKE_BINARY_DIR}/pico_tflmicro)

# ---- Source files ----------------------------------------------------------

set(IOTIDS_C_SOURCES
    src/main.c
    src/inference/arena.c
    src/inference/model_runner.c
    src/preprocessing/feature_scaler.c
    src/utils/ring_buffer.c
    src/utils/debug_uart.c
    src/network/wifi_init.c
    src/network/tcp_client.c
    src/federated/flash_storage.c
    src/federated/weight_receiver.c
)

set(IOTIDS_CXX_SOURCES
    src/inference/tflm_resolver.cc
)

# ---- Executable ------------------------------------------------------------

add_executable(iotids_runtime
    ${IOTIDS_C_SOURCES}
    ${IOTIDS_CXX_SOURCES}
)

# ---- Compile definitions ---------------------------------------------------

target_compile_definitions(iotids_runtime PRIVATE
    IOTIDS_NUM_FEATURES=${IOTIDS_NUM_FEATURES}
    IOTIDS_ARENA_SIZE=${IOTIDS_ARENA_SIZE}
    IOTIDS_THRESHOLD=${IOTIDS_THRESHOLD}f
    IOTIDS_RING_CAPACITY=${IOTIDS_RING_CAPACITY}
    WIFI_SSID="${WIFI_SSID}"
    WIFI_PASSWORD="${WIFI_PASSWORD}"
    SERVER_HOST="${SERVER_HOST}"
    SERVER_PORT=${SERVER_PORT}
    # Strip debug logs in release — comment out for development
    # NDEBUG
)

# ---- Include paths ---------------------------------------------------------

target_include_directories(iotids_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---- Compiler flags --------------------------------------------------------

target_compile_options(iotids_runtime PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter   # TFLM callbacks have many unused args
    -ffunction-sections     # enable --gc-sections dead-code elimination
    -fdata-sections
    # Float ABI — Cortex-M33 has FPU; use hardware float for scaler ops
    -mfloat-abi=hard
    -mfpu=fpv5-sp-d16
    # Size optimisation (balances speed vs flash usage)
    -Os
)

target_link_options(iotids_runtime PRIVATE
    -Wl,--gc-sections       # strip unused functions/data from final binary
)

# ---- Link against Pico SDK libraries ---------------------------------------

target_link_libraries(iotids_runtime
    pico_stdlib           # stdio, timing, GPIO
    pico_multicore        # multicore_launch_core1
    hardware_uart         # UART driver
    hardware_flash        # flash_range_erase / flash_range_program
    hardware_sync         # save_and_disable_interrupts
    hardware_watchdog     # watchdog_enable / watchdog_update
    pico_cyw43_arch_lwip_poll   # CYW43 Wi-Fi + lwIP cooperative polling
    pico_tflmicro         # TensorFlow Lite for Microcontrollers
)

# ---- Output formats --------------------------------------------------------

# Generate .uf2 for drag-and-drop flashing and .bin for direct programming
pico_add_extra_outputs(iotids_runtime)

# Enable USB stdio (useful for development without a UART adapter)
pico_enable_stdio_usb(iotids_runtime 1)
pico_enable_stdio_uart(iotids_runtime 1)

# ---- Size report -----------------------------------------------------------

add_custom_command(TARGET iotids_runtime POST_BUILD
    COMMAND ${CMAKE_SIZE} --format=berkeley $<TARGET_FILE:iotids_runtime>
    COMMENT "Binary size breakdown:"
)

# ---- Tests -----------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(tests)
endif()

# ---- Summary ---------------------------------------------------------------

message(STATUS "")
message(STATUS "iotids-runtime configuration:")
message(STATUS "  Board          : ${PICO_BOARD}")
message(STATUS "  Features       : ${IOTIDS_NUM_FEATURES}")
message(STATUS "  Arena size     : ${IOTIDS_ARENA_SIZE} bytes")
message(STATUS "  Threshold      : ${IOTIDS_THRESHOLD}")
message(STATUS "  Ring capacity  : ${IOTIDS_RING_CAPACITY}")
message(STATUS "  Wi-Fi SSID     : ${WIFI_SSID}")
message(STATUS "  Server         : ${SERVER_HOST}:${SERVER_PORT}")
message(STATUS "")
