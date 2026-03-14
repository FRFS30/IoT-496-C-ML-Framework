/*
 * main.c — iotids C runtime entry point.
 *
 * Boot sequence:
 *   1. UART logging
 *   2. Wi-Fi (CYW43 driver, WPA2)
 *   3. Flash: load persisted model (or fall back to compiled-in default)
 *   4. Scaler init (from model header constants)
 *   5. TFLM init
 *   6. Launch core 1: Wi-Fi polling + weight receiver
 *   7. Core 0 main loop: consume features -> scale -> infer -> send result
 *
 * Dual-core split (critical for RP2350 — CYW43 and TFLM cannot share a core):
 *   Core 0 : inference (model_runner), ring_buffer consumer, TCP send
 *   Core 1 : cyw43_arch_poll(), weight_rx_poll(), ring_buffer producer
 *
 * LED conventions:
 *   Solid on  : running normally
 *   Fast blink (100 ms) : error / waiting for Wi-Fi
 *   Slow blink (500 ms) : weight update in progress
 */

#include "src/inference/model_runner.h"
#include "src/inference/arena.h"
#include "src/preprocessing/feature_scaler.h"
#include "src/federated/flash_storage.h"
#include "src/federated/weight_receiver.h"
#include "src/network/wifi_init.h"
#include "src/network/tcp_client.h"
#include "src/utils/ring_buffer.h"
#include "src/utils/debug_uart.h"
#include "model/iotids_model.tflite.h"

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/watchdog.h"

#include <string.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Configuration — set via CMake definitions or edit here
 * -------------------------------------------------------------------------- */

#ifndef WIFI_SSID
#define WIFI_SSID     "YOUR_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_PASSWORD"
#endif

#ifndef SERVER_HOST
#define SERVER_HOST   "192.168.1.100"
#endif

#ifndef SERVER_PORT
#define SERVER_PORT   5555u
#endif

/* How many inferences between weight-update polls on core 1 */
#ifndef WEIGHT_POLL_INTERVAL
#define WEIGHT_POLL_INTERVAL 50u
#endif

/* Watchdog timeout — resets if main loop stalls for > 8 s */
#define WATCHDOG_MS 8000u

/* LED pin on Pico W is controlled via CYW43, not a GPIO pin */
#define LED_ON()   cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1)
#define LED_OFF()  cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0)

/* --------------------------------------------------------------------------
 * Shared state between cores
 * -------------------------------------------------------------------------- */

static RingBuf g_ring;   /* core 1 pushes features, core 0 pops them */

/* --------------------------------------------------------------------------
 * Result packet sent to Python server after each inference
 *
 *   [0..3]   magic    0x52534C54 ("RSLT")
 *   [4]      label    0 = benign, 1 = attack
 *   [5..8]   prob     float32 sigmoid output (little-endian)
 *   [9..12]  latency  uint32_t inference time in µs (little-endian)
 * -------------------------------------------------------------------------- */

#define RESULT_MAGIC   0x52534C54u
#define RESULT_PKT_LEN 13u

static void build_result_packet(uint8_t *buf, int label,
                                  float prob, uint32_t latency_us) {
    uint32_t magic = RESULT_MAGIC;
    memcpy(buf + 0, &magic,      4);
    buf[4] = (uint8_t)label;
    memcpy(buf + 5, &prob,       4);
    memcpy(buf + 9, &latency_us, 4);
}

/* --------------------------------------------------------------------------
 * Core 1 entry — Wi-Fi polling and weight receiver
 * -------------------------------------------------------------------------- */

static void core1_entry(void) {
    DLOG_INFO("Core 1: starting Wi-Fi + weight receiver loop");

    /* Connect to FedAvg server */
    TcpStatus ts = tcp_connect_with_retry(SERVER_HOST, SERVER_PORT, 0u);
    if (ts != TCP_OK) {
        DLOG_ERROR("Core 1: failed to connect to server — halting core 1");
        return;
    }

    weight_rx_init(SERVER_PORT, NULL /* callback handled in weight_rx_poll */);

    uint32_t poll_tick = 0u;

    while (true) {
        /* Must call cyw43_arch_poll() regularly to keep the Wi-Fi driver alive */
        cyw43_arch_poll();

        /* Reconnect if the link dropped */
        if (!wifi_is_connected()) {
            DLOG_WARN("Core 1: Wi-Fi dropped, reconnecting...");
            wifi_wait_connected(10000u);
            tcp_connect_with_retry(SERVER_HOST, SERVER_PORT, 5u);
        }

        /* Poll for incoming weight updates every WEIGHT_POLL_INTERVAL ticks */
        if (++poll_tick >= WEIGHT_POLL_INTERVAL) {
            poll_tick = 0u;
            weight_rx_poll();
        }

        sleep_ms(1u);
    }
}

/* --------------------------------------------------------------------------
 * Scaler + model init
 * -------------------------------------------------------------------------- */

static Scaler g_scaler;

static bool init_scaler(void) {
    return scaler_init(&g_scaler,
                       IOTIDS_SCALER_MEDIANS,
                       IOTIDS_SCALER_IQRS,
                       IOTIDS_SCALER_CLIP_LO,
                       IOTIDS_SCALER_CLIP_HI,
                       IOTIDS_NUM_FEATURES);
}

/*
 * Try to load a federated-updated model from flash first.
 * If none exists, fall back to the compiled-in default model.
 */
static const uint8_t *g_active_model     = NULL;
static size_t         g_active_model_len = 0u;

/* Buffer for flash-loaded model (FLASH_MODEL_MAX_BYTES) */
static uint8_t g_flash_model_buf[FLASH_MODEL_MAX_BYTES]
    __attribute__((aligned(16)));

static void load_model(void) {
    size_t flash_len = 0u;
    FlashStatus fs = flash_read_model(g_flash_model_buf,
                                      sizeof(g_flash_model_buf),
                                      &flash_len);
    if (fs == FLASH_OK && flash_len > 0u) {
        DLOG_INFO("Using flash model (%u bytes)", (unsigned)flash_len);
        g_active_model     = g_flash_model_buf;
        g_active_model_len = flash_len;
    } else {
        DLOG_INFO("Using compiled-in default model (%u bytes)",
                  (unsigned)iotids_model_tflite_len);
        g_active_model     = iotids_model_tflite;
        g_active_model_len = iotids_model_tflite_len;
    }
}

/* --------------------------------------------------------------------------
 * Error halt — blink fast, log, loop forever
 * -------------------------------------------------------------------------- */

static void fatal_error(const char *msg) {
    DLOG_ERROR("FATAL: %s", msg);
    while (true) {
        LED_ON();  sleep_ms(100u);
        LED_OFF(); sleep_ms(100u);
    }
}

/* --------------------------------------------------------------------------
 * main — core 0 entry point
 * -------------------------------------------------------------------------- */

int main(void) {
    /* --- 1. UART logging (first — captures all subsequent boot messages) --- */
    stdio_init_all();
    debug_init(115200u);
    DLOG_INFO("iotids runtime v0.1 — PSU CMPSC496");

    /* --- 2. Wi-Fi --- */
    if (!wifi_connect(WIFI_SSID, WIFI_PASSWORD)) {
        fatal_error("CYW43 driver init failed");
    }
    if (!wifi_wait_connected(10000u)) {
        fatal_error("Wi-Fi connect timeout");
    }

    char ip_buf[WIFI_IP_STR_LEN];
    DLOG_INFO("IP: %s", wifi_get_ip_str(ip_buf));

    /* --- 3. Load model (flash > compiled-in) --- */
    load_model();

    /* --- 4. Scaler init --- */
    if (!init_scaler()) {
        fatal_error("Scaler init failed — check IOTIDS_NUM_FEATURES");
    }
    DLOG_INFO("Scaler ready (%d features)", IOTIDS_NUM_FEATURES);

    /* --- 5. TFLM init --- */
    TflmStatus ts = tflm_init(g_active_model, g_active_model_len);
    if (ts != TFLM_OK) {
        DLOG_ERROR("tflm_init failed: %d", (int)ts);
        fatal_error("TFLM init failed — check model file");
    }
    DLOG_INFO("TFLM ready. Arena used: %u bytes", (unsigned)tflm_arena_used());

    if (tflm_arena_used() > IOTIDS_ARENA_SIZE) {
        fatal_error("Arena overflow — increase IOTIDS_ARENA_SIZE");
    }

    /* --- 6. Ring buffer + core 1 --- */
    ring_buf_init(&g_ring);
    multicore_launch_core1(core1_entry);
    DLOG_INFO("Core 1 launched");

    /* --- 7. Watchdog enable --- */
    watchdog_enable(WATCHDOG_MS, true /* pause_on_debug */);

    LED_ON();
    DLOG_INFO("Entering inference loop");

    /* --- 8. Main inference loop (core 0) --- */
    static float features[IOTIDS_NUM_FEATURES];
    uint8_t result_pkt[RESULT_PKT_LEN];
    uint32_t inference_count = 0u;

    while (true) {
        /* Kick the watchdog — if we stall > WATCHDOG_MS, system resets */
        watchdog_update();

        /* Wait for a feature vector from core 1 (blocks briefly) */
        if (!ring_buf_pop(&g_ring, features)) {
            sleep_ms(1u);
            continue;
        }

        /* Skip inference during a weight update (avoids partial-tensor reads) */
        if (weight_rx_is_updating()) {
            LED_OFF();
            sleep_ms(1u);
            LED_ON();
            continue;
        }

        /* Scale features in-place */
        if (!scaler_transform(&g_scaler, features, IOTIDS_NUM_FEATURES)) {
            DLOG_WARN("Scaler transform failed — skipping sample");
            continue;
        }

        /* Run INT8 inference */
        float    prob = 0.0f;
        TflmStatus infer_status = tflm_run(features, &prob);
        if (infer_status != TFLM_OK) {
            DLOG_ERROR("tflm_run error: %d", (int)infer_status);
            continue;
        }

        int label = (prob >= IOTIDS_THRESHOLD) ? 1 : 0;
        uint32_t latency = tflm_get_latency_us();

        DLOG_DEBUG("Inference #%u: label=%d prob=%.4f latency=%u us",
                   (unsigned)inference_count, label, (double)prob,
                   (unsigned)latency);

        inference_count++;

        /* Send result to FedAvg server */
        if (tcp_is_connected()) {
            build_result_packet(result_pkt, label, prob, latency);
            TcpStatus send_s = tcp_send(result_pkt, RESULT_PKT_LEN);
            if (send_s != TCP_OK) {
                DLOG_WARN("TCP send failed: %d", (int)send_s);
            }
        }

        /* Log arena and latency every 1000 inferences */
        if (inference_count % 1000u == 0u) {
            DLOG_INFO("Inferences: %u  Arena: %u B  Last latency: %u us",
                      (unsigned)inference_count,
                      (unsigned)tflm_arena_used(),
                      (unsigned)latency);
        }
    }

    /* Unreachable — watchdog will reset before we get here */
    return 0;
}
