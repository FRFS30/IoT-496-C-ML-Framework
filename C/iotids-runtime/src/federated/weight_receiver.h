#ifndef IOTIDS_WEIGHT_RECEIVER_H
#define IOTIDS_WEIGHT_RECEIVER_H

/*
 * weight_receiver.h — on-device federated weight update receiver.
 *
 * Listens for INT8 weight update payloads from the Python FedAvgServer via
 * TCP. Each payload follows a simple binary protocol:
 *
 *   Packet layout:
 *     [0..3]   magic         uint32_t  0x57475455 ("WGTU")
 *     [4..7]   total_len     uint32_t  total bytes in this packet (header incl.)
 *     [8..11]  layer_id      uint32_t  layer index (0 = first Dense layer)
 *     [12..15] n_weights     uint32_t  number of INT8 weights that follow
 *     [16..19] checksum      uint32_t  CRC32 of the weight bytes only
 *     [20 ..]  weights       int8_t[]  n_weights quantized weight values
 *
 * A layer_id of 0xFFFFFFFF indicates a full-model update (all layers packed
 * consecutively). In that case n_weights is the total weight count and the
 * receiver calls flash_write_model() to persist the update.
 *
 * After validation the on_weights_updated callback is fired. The inference
 * loop on core 0 should pause inference during weight application to avoid
 * reading partially-updated tensors. A simple flag (g_weights_updating) is
 * provided for this purpose.
 *
 * This module runs on core 1 (Wi-Fi core). tcp_recv is polled in a tight
 * loop; the inference loop on core 0 checks g_weights_updating before each
 * inference call.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WEIGHT_PACKET_MAGIC    0x57475455u   /* "WGTU" */
#define WEIGHT_PACKET_HDR_SIZE 20u

/* Maximum weights in a single update packet (one layer or full model) */
#ifndef WEIGHT_MAX_PER_PACKET
#define WEIGHT_MAX_PER_PACKET  4096u
#endif

/*
 * Callback type — fired after a weight update is validated and applied.
 *
 *   layer_id : layer index, or 0xFFFFFFFF for full-model update.
 *   weights  : pointer to INT8 weight array (valid only during callback).
 *   n        : number of weights.
 */
typedef void (*WeightsUpdatedCb)(uint32_t layer_id,
                                 const int8_t *weights,
                                 uint32_t n);

/*
 * weight_rx_init — prepare the receiver and open a TCP listen port.
 *
 *   port     : TCP port to listen on (should match Python server config).
 *   callback : called after each validated weight update. May be NULL.
 *
 * Call once from core 1 after wifi_wait_connected().
 * Returns true on success.
 */
bool weight_rx_init(uint16_t port, WeightsUpdatedCb callback);

/*
 * weight_rx_poll — non-blocking check for an incoming weight payload.
 *
 * Call in the core 1 main loop, interleaved with cyw43_arch_poll().
 * Returns true if a valid update was received and applied this call.
 * Returns false if nothing was received or the packet was invalid.
 *
 * On receipt:
 *   1. Validates header magic and CRC32.
 *   2. Applies weights to the TFLM input tensor buffer via the callback.
 *   3. Persists to flash if layer_id == 0xFFFFFFFF (full model).
 *   4. Sets g_weights_updating = false when done (inference may resume).
 */
bool weight_rx_poll(void);

/*
 * weight_rx_is_updating — returns true while a weight update is in progress.
 * Core 0 inference loop must check this before calling tflm_run().
 */
bool weight_rx_is_updating(void);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_WEIGHT_RECEIVER_H */
