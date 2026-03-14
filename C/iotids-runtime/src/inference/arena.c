#include "arena.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static inline size_t align_up(size_t x, size_t align) {
    /* align must be power-of-two */
    return (x + (align - 1u)) & ~(align - 1u);
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

void arena_init(Arena *a, uint8_t *buf, size_t size) {
    assert(a    != NULL);
    assert(buf  != NULL);
    assert(size > 0u);
    /* Caller must supply a 16-byte-aligned buffer for TFLM SIMD safety. */
    assert(((uintptr_t)buf & 0xFu) == 0u);

    a->buf      = buf;
    a->capacity = size;
    a->used     = 0u;
    a->peak     = 0u;
}

void arena_reset(Arena *a) {
    assert(a != NULL);
    a->used = 0u;
    /* peak is intentionally NOT reset — it is a lifetime high-water mark. */
}

size_t arena_used_bytes(const Arena *a) {
    assert(a != NULL);
    return a->used;
}

size_t arena_peak_bytes(const Arena *a) {
    assert(a != NULL);
    return a->peak;
}

size_t arena_remaining_bytes(const Arena *a) {
    assert(a != NULL);
    return a->capacity - a->used;
}

void *arena_alloc(Arena *a, size_t size, size_t align) {
    assert(a     != NULL);
    assert(align >  0u);
    assert((align & (align - 1u)) == 0u); /* power-of-two */

    size_t aligned_start = align_up(a->used, align);
    size_t new_used      = aligned_start + size;

    if (new_used > a->capacity) {
        /* Out of arena space — caller must handle (assert/panic). */
        return NULL;
    }

    a->used = new_used;
    if (a->used > a->peak) {
        a->peak = a->used;
    }

    return (void *)(a->buf + aligned_start);
}
