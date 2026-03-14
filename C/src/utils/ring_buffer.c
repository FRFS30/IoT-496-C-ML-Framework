#include "ring_buffer.h"
#include <string.h>
#include <assert.h>

/*
 * Memory ordering note for RP2350 (Cortex-M33):
 *
 * The Cortex-M33 is a weakly ordered processor. To guarantee that the data
 * write (memcpy into the slot) is visible to the consumer before the head
 * increment is visible, we need a Data Memory Barrier (DMB) between the
 * store to the slot and the store to head.
 *
 * Similarly the consumer needs a DMB between reading tail and reading the
 * slot data to prevent the CPU from speculating the load of the slot before
 * it reads the tail.
 *
 * We use __DMB() (available in CMSIS / Pico SDK) for this.
 *
 * If built on a host (x86 for unit tests), __DMB() is a no-op macro.
 */
#ifndef __DMB
#define __DMB() __asm__ volatile ("" ::: "memory")
#endif

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static inline uint32_t next_idx(uint32_t idx) {
    return (idx + 1u) & (IOTIDS_RING_CAPACITY - 1u);
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

void ring_buf_init(RingBuf *rb) {
    assert(rb != NULL);
    memset(rb->data, 0, sizeof(rb->data));
    rb->head       = 0u;
    rb->tail       = 0u;
    rb->capacity   = IOTIDS_RING_CAPACITY;
    rb->n_features = IOTIDS_NUM_FEATURES;
}

bool ring_buf_push(RingBuf *rb, const float *features) {
    assert(rb != NULL && features != NULL);

    uint32_t h = rb->head;
    uint32_t t = rb->tail;  /* single load of volatile */

    if (next_idx(h) == t) {
        /* Buffer full — drop this sample. */
        return false;
    }

    /* Copy feature vector into the slot BEFORE advancing head. */
    memcpy(rb->data[h], features, IOTIDS_NUM_FEATURES * sizeof(float));

    /* DMB: ensure the data write is globally visible before head advances. */
    __DMB();

    rb->head = next_idx(h);
    return true;
}

bool ring_buf_pop(RingBuf *rb, float *out) {
    assert(rb != NULL && out != NULL);

    uint32_t t = rb->tail;
    uint32_t h = rb->head;  /* single load of volatile */

    if (t == h) {
        /* Buffer empty. */
        return false;
    }

    /* DMB: ensure we read the slot data AFTER reading tail. */
    __DMB();

    memcpy(out, rb->data[t], IOTIDS_NUM_FEATURES * sizeof(float));

    /* Advance tail only after the copy is complete. */
    __DMB();
    rb->tail = next_idx(t);
    return true;
}

bool ring_buf_full(const RingBuf *rb) {
    assert(rb != NULL);
    return next_idx(rb->head) == rb->tail;
}

bool ring_buf_empty(const RingBuf *rb) {
    assert(rb != NULL);
    return rb->head == rb->tail;
}

uint32_t ring_buf_count(const RingBuf *rb) {
    assert(rb != NULL);
    return (rb->head - rb->tail) & (IOTIDS_RING_CAPACITY - 1u);
}
