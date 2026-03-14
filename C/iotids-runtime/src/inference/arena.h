#ifndef IOTIDS_ARENA_H
#define IOTIDS_ARENA_H

#include <stddef.h>
#include <stdint.h>

/*
 * arena.h — static memory arena for TFLM tensor scratch space.
 *
 * The RP2350 has 520 KB SRAM. TFLM model weights live in flash (read-only),
 * but activations and intermediate tensors must live in SRAM.
 * We carve out a single contiguous region and hand it to TFLM as its tensor
 * arena. No heap (malloc/free) is used anywhere in this runtime.
 *
 * Target arena size: < 200 KB (leaves ~320 KB for stack, Wi-Fi driver, etc.)
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum arena size in bytes. Tune via CMake: -DIOTIDS_ARENA_SIZE=204800 */
#ifndef IOTIDS_ARENA_SIZE
#define IOTIDS_ARENA_SIZE (200u * 1024u)   /* 200 KB default */
#endif

typedef struct {
    uint8_t *buf;       /* pointer to backing buffer                    */
    size_t   capacity;  /* total bytes in buffer                        */
    size_t   used;      /* high-water mark (bump-pointer style)         */
    size_t   peak;      /* maximum 'used' ever seen (profiling)         */
} Arena;

/*
 * arena_init — bind an Arena to a caller-supplied buffer.
 *
 *   buf      : pointer to a static or stack buffer of at least 'size' bytes.
 *              Must be 16-byte aligned for TFLM SIMD safety.
 *   size     : capacity in bytes.
 *
 * The buffer is NOT zeroed here; TFLM will initialise tensors itself.
 */
void arena_init(Arena *a, uint8_t *buf, size_t size);

/*
 * arena_reset — mark the arena as empty (does NOT touch memory).
 * Call between independent inference passes if you want to reuse scratch
 * space. Do NOT call while a TFLM interpreter is still live.
 */
void arena_reset(Arena *a);

/*
 * arena_used_bytes — bytes currently allocated (bump pointer position).
 * Useful after the first inference call to verify you fit within budget.
 */
size_t arena_used_bytes(const Arena *a);

/*
 * arena_peak_bytes — highest 'used' ever recorded.
 * More informative than arena_used_bytes when allocations are freed/reset.
 */
size_t arena_peak_bytes(const Arena *a);

/*
 * arena_remaining_bytes — bytes still available.
 */
size_t arena_remaining_bytes(const Arena *a);

/*
 * arena_alloc — bump-allocate 'size' bytes, aligned to 'align'.
 * Returns NULL if out of space (treat as fatal on Pico — use assert/panic).
 * align must be a power of two (typically 4 or 16).
 */
void *arena_alloc(Arena *a, size_t size, size_t align);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_ARENA_H */
