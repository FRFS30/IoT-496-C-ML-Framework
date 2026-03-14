#ifndef IOTIDS_RING_BUFFER_H
#define IOTIDS_RING_BUFFER_H

/*
 * ring_buffer.h — fixed-size circular buffer for incoming feature vectors.
 *
 * Design: single-producer (network/TCP core 1) / single-consumer (inference
 * core 0). Lock-free on RP2350 because head and tail are written by different
 * cores and read atomically via the Pico SDK's sio registers.
 *
 * Each slot holds one complete float32 feature vector of IOTIDS_NUM_FEATURES
 * elements. Capacity is set at compile time via IOTIDS_RING_CAPACITY.
 *
 * Memory: capacity * n_features * sizeof(float) bytes, all static.
 * At 77 features, capacity=8: 77 * 8 * 4 = 2 464 bytes (~2.4 KB). Fine.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 77
#endif

/* Number of feature vector slots in the ring. Must be a power of two for
 * fast modulo via bitmask. Increase if network bursts outpace inference. */
#ifndef IOTIDS_RING_CAPACITY
#define IOTIDS_RING_CAPACITY 8u
#endif

/* Compile-time assert: capacity must be power-of-two */
typedef char ring_capacity_pow2_check[
    ((IOTIDS_RING_CAPACITY & (IOTIDS_RING_CAPACITY - 1u)) == 0u) ? 1 : -1
];

typedef struct {
    float    data[IOTIDS_RING_CAPACITY][IOTIDS_NUM_FEATURES];
    volatile uint32_t head;  /* write index (producer increments) */
    volatile uint32_t tail;  /* read index  (consumer increments) */
    uint32_t capacity;       /* always IOTIDS_RING_CAPACITY        */
    uint32_t n_features;     /* always IOTIDS_NUM_FEATURES         */
} RingBuf;

/*
 * ring_buf_init — zero-initialise a RingBuf and set metadata.
 * Call once before either producer or consumer touches the buffer.
 */
void ring_buf_init(RingBuf *rb);

/*
 * ring_buf_push — copy one feature vector into the next available slot.
 *
 *   features : float32 array of n_features elements (not modified).
 *
 * Returns true on success, false if the buffer is full (vector dropped).
 * The producer (network receiver on core 1) calls this.
 */
bool ring_buf_push(RingBuf *rb, const float *features);

/*
 * ring_buf_pop — copy the oldest feature vector out of the buffer.
 *
 *   out : float32 array of n_features elements to write into.
 *
 * Returns true on success, false if the buffer is empty.
 * The consumer (inference loop on core 0) calls this.
 */
bool ring_buf_pop(RingBuf *rb, float *out);

/*
 * ring_buf_full  — true if no more pushes can succeed.
 * ring_buf_empty — true if no more pops can succeed.
 *
 * These are instantaneous snapshots; race-safe to read from either core.
 */
bool ring_buf_full(const RingBuf *rb);
bool ring_buf_empty(const RingBuf *rb);

/*
 * ring_buf_count — number of items currently available to pop.
 */
uint32_t ring_buf_count(const RingBuf *rb);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_RING_BUFFER_H */
