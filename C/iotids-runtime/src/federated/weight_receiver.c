#include "weight_receiver.h"
#include "flash_storage.h"
#include "../network/tcp_client.h"
#include "../utils/debug_uart.h"

#include "pico/stdlib.h"
#include "pico/multicore.h"

#include <string.h>
#include <stdint.h>

/* --------------------------------------------------------------------------
 * Internal state
 * -------------------------------------------------------------------------- */

static WeightsUpdatedCb g_callback        = NULL;
static uint16_t         g_port            = 0u;
static volatile bool    g_updating        = false;
static bool             g_initialised     = false;

/* Packet receive buffer — header + weight payload */
static uint8_t g_recv_buf[WEIGHT_PACKET_HDR_SIZE + WEIGHT_MAX_PER_PACKET];

/* Weight staging area — reused across calls, avoids stack pressure */
static int8_t g_weight_staging[WEIGHT_MAX_PER_PACKET];

/* --------------------------------------------------------------------------
 * Packet parsing helpers
 * -------------------------------------------------------------------------- */

typedef struct {
    uint32_t magic;
    uint32_t total_len;
    uint32_t layer_id;
    uint32_t n_weights;
    uint32_t checksum;
} WeightPacketHdr;

static void parse_header(const uint8_t *buf, WeightPacketHdr *hdr) {
    /* Little-endian read — RP2350 is LE so memcpy is safe */
    memcpy(&hdr->magic,      buf +  0, 4);
    memcpy(&hdr->total_len,  buf +  4, 4);
    memcpy(&hdr->layer_id,   buf +  8, 4);
    memcpy(&hdr->n_weights,  buf + 12, 4);
    memcpy(&hdr->checksum,   buf + 16, 4);
}

static bool validate_packet(const WeightPacketHdr *hdr,
                             const uint8_t *weight_bytes,
                             size_t n) {
    if (hdr->magic != WEIGHT_PACKET_MAGIC) {
        DLOG_WARN("WeightRx: bad magic 0x%08X", (unsigned)hdr->magic);
        return false;
    }
    if (hdr->n_weights != (uint32_t)n) {
        DLOG_WARN("WeightRx: n_weights mismatch (hdr %u, actual %u)",
                  (unsigned)hdr->n_weights, (unsigned)n);
        return false;
    }
    if (hdr->n_weights > WEIGHT_MAX_PER_PACKET) {
        DLOG_WARN("WeightRx: n_weights %u exceeds max %u",
                  (unsigned)hdr->n_weights, (unsigned)WEIGHT_MAX_PER_PACKET);
        return false;
    }

    uint32_t actual_crc = flash_crc32(weight_bytes, hdr->n_weights);
    if (actual_crc != hdr->checksum) {
        DLOG_ERROR("WeightRx: CRC mismatch (expected 0x%08X, got 0x%08X)",
                   (unsigned)hdr->checksum, (unsigned)actual_crc);
        return false;
    }
    return true;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

bool weight_rx_init(uint16_t port, WeightsUpdatedCb callback) {
    g_port        = port;
    g_callback    = callback;
    g_updating    = false;
    g_initialised = true;
    DLOG_INFO("WeightRx: initialised on port %u", (unsigned)port);
    return true;
}

bool weight_rx_poll(void) {
    if (!g_initialised) return false;

    /* Re-connect if the TCP link is down */
    if (!tcp_is_connected()) {
        DLOG_INFO("WeightRx: TCP not connected, skipping poll");
        return false;
    }

    /* Non-blocking receive: try to get at least the header */
    size_t received = 0u;
    TcpStatus ts = tcp_recv(g_recv_buf, WEIGHT_PACKET_HDR_SIZE,
                             0u /* non-blocking */, &received);
    if (ts != TCP_OK || received < WEIGHT_PACKET_HDR_SIZE) {
        /* Nothing available — that's normal */
        return false;
    }

    /* Parse header */
    WeightPacketHdr hdr;
    parse_header(g_recv_buf, &hdr);

    if (hdr.magic != WEIGHT_PACKET_MAGIC) {
        DLOG_WARN("WeightRx: junk data in stream, dropping");
        return false;
    }

    /* How many weight bytes are expected? */
    uint32_t n = hdr.n_weights;
    if (n == 0u || n > WEIGHT_MAX_PER_PACKET) {
        DLOG_ERROR("WeightRx: invalid n_weights %u", (unsigned)n);
        return false;
    }

    /* Receive the weight payload (block up to 2 s) */
    size_t weight_received = 0u;
    ts = tcp_recv(g_recv_buf + WEIGHT_PACKET_HDR_SIZE, n,
                  2000u /* 2 s timeout */, &weight_received);
    if (ts != TCP_OK || weight_received < (size_t)n) {
        DLOG_ERROR("WeightRx: incomplete payload (%u/%u bytes)",
                   (unsigned)weight_received, (unsigned)n);
        return false;
    }

    const uint8_t *weight_bytes = g_recv_buf + WEIGHT_PACKET_HDR_SIZE;

    /* Validate */
    if (!validate_packet(&hdr, weight_bytes, n)) {
        return false;
    }

    /* Copy to staging area (keeps recv buffer free for next packet) */
    memcpy(g_weight_staging, weight_bytes, n);

    /* Signal core 0 to pause inference */
    g_updating = true;

    /* Full model update: persist to flash */
    if (hdr.layer_id == 0xFFFFFFFFu) {
        DLOG_INFO("WeightRx: full model update (%u bytes), writing to flash",
                  (unsigned)n);
        FlashStatus fs = flash_write_model((const uint8_t *)g_weight_staging, n);
        if (fs != FLASH_OK) {
            DLOG_ERROR("WeightRx: flash write failed (%d)", (int)fs);
        }
    } else {
        DLOG_INFO("WeightRx: layer %u weight update (%u weights)",
                  (unsigned)hdr.layer_id, (unsigned)n);
    }

    /* Fire application callback */
    if (g_callback != NULL) {
        g_callback(hdr.layer_id, g_weight_staging, n);
    }

    g_updating = false;

    DLOG_INFO("WeightRx: update applied successfully");
    return true;
}

bool weight_rx_is_updating(void) {
    return g_updating;
}
