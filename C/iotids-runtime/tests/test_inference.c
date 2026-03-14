/*
 * test_inference.c — unit tests for arena.c and ring_buffer.c.
 *
 * Runs on the host without Pico SDK. Build with:
 *
 *   gcc -std=c11 -Wall -Wextra \
 *       -DIOTIDS_NUM_FEATURES=4 \
 *       -DIOTIDS_RING_CAPACITY=4 \
 *       test_inference.c \
 *       ../src/inference/arena.c \
 *       ../src/utils/ring_buffer.c \
 *       -o test_inference && ./test_inference
 *
 * model_runner.c is NOT tested here because it requires the Pico SDK's
 * time_us_32() and the TFLM headers. Test model_runner on hardware by
 * running the firmware with a known-good .tflite and observing UART output.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* Override for testing */
#undef  IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 4

#undef  IOTIDS_RING_CAPACITY
#define IOTIDS_RING_CAPACITY 4u

#include "../src/inference/arena.h"
#include "../src/utils/ring_buffer.h"

/* ---- Test helpers ---- */

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define CHECK(cond, msg) do {                                           \
    g_tests_run++;                                                      \
    if (cond) {                                                         \
        g_tests_passed++;                                               \
        printf("  PASS: %s\n", msg);                                   \
    } else {                                                            \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__);               \
    }                                                                   \
} while (0)

/* ============================================================
 * Arena tests
 * ============================================================ */

static void test_arena_init(void) {
    printf("\n[test_arena_init]\n");
    static uint8_t buf[256] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));
    CHECK(arena_used_bytes(&a) == 0u,             "used starts at 0");
    CHECK(arena_remaining_bytes(&a) == 256u,      "remaining == capacity");
    CHECK(arena_peak_bytes(&a) == 0u,             "peak starts at 0");
}

static void test_arena_alloc_basic(void) {
    printf("\n[test_arena_alloc_basic]\n");
    static uint8_t buf[256] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));

    void *p1 = arena_alloc(&a, 32u, 4u);
    CHECK(p1 != NULL,                             "alloc 32 bytes succeeds");
    CHECK(arena_used_bytes(&a) == 32u,            "used == 32 after first alloc");

    void *p2 = arena_alloc(&a, 64u, 4u);
    CHECK(p2 != NULL,                             "alloc 64 bytes succeeds");
    CHECK(arena_used_bytes(&a) == 96u,            "used == 96 after two allocs");

    /* Pointers must not overlap */
    CHECK((uint8_t *)p2 >= (uint8_t *)p1 + 32u,  "p2 does not overlap p1");
}

static void test_arena_alignment(void) {
    printf("\n[test_arena_alignment]\n");
    static uint8_t buf[256] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));

    /* Allocate 1 byte to misalign the bump pointer */
    arena_alloc(&a, 1u, 1u);  /* used = 1 */

    /* Next 16-byte-aligned alloc should skip bytes 1..15 */
    void *p = arena_alloc(&a, 16u, 16u);
    CHECK(p != NULL,                              "aligned alloc after 1-byte alloc");
    CHECK(((uintptr_t)p & 0xFu) == 0u,           "returned pointer is 16-byte aligned");
}

static void test_arena_overflow(void) {
    printf("\n[test_arena_overflow]\n");
    static uint8_t buf[64] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));

    void *p1 = arena_alloc(&a, 48u, 1u);
    CHECK(p1 != NULL,                             "first alloc fits");

    void *p2 = arena_alloc(&a, 32u, 1u);  /* 48 + 32 = 80 > 64 */
    CHECK(p2 == NULL,                             "overflow returns NULL");

    /* Used should not have advanced after failed alloc */
    CHECK(arena_used_bytes(&a) == 48u,            "used unchanged after overflow");
}

static void test_arena_reset(void) {
    printf("\n[test_arena_reset]\n");
    static uint8_t buf[64] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));

    arena_alloc(&a, 32u, 4u);
    arena_reset(&a);
    CHECK(arena_used_bytes(&a) == 0u,             "used == 0 after reset");
    CHECK(arena_peak_bytes(&a) == 32u,            "peak preserved after reset");

    /* Can allocate again from the beginning */
    void *p = arena_alloc(&a, 32u, 4u);
    CHECK(p != NULL,                              "can reallocate after reset");
}

static void test_arena_peak(void) {
    printf("\n[test_arena_peak]\n");
    static uint8_t buf[128] __attribute__((aligned(16)));
    Arena a;
    arena_init(&a, buf, sizeof(buf));

    arena_alloc(&a, 60u, 4u);
    CHECK(arena_peak_bytes(&a) == 60u,            "peak tracks first alloc");

    arena_reset(&a);
    arena_alloc(&a, 30u, 4u);
    CHECK(arena_peak_bytes(&a) == 60u,            "peak not reduced after reset");
}

/* ============================================================
 * Ring buffer tests
 * ============================================================ */

static void test_ring_init(void) {
    printf("\n[test_ring_init]\n");
    RingBuf rb;
    ring_buf_init(&rb);
    CHECK(ring_buf_empty(&rb),                    "empty after init");
    CHECK(!ring_buf_full(&rb),                    "not full after init");
    CHECK(ring_buf_count(&rb) == 0u,              "count == 0 after init");
}

static void test_ring_push_pop(void) {
    printf("\n[test_ring_push_pop]\n");
    RingBuf rb;
    ring_buf_init(&rb);

    float in_vec[4]  = { 1.0f, 2.0f, 3.0f, 4.0f };
    float out_vec[4] = { 0 };

    bool pushed = ring_buf_push(&rb, in_vec);
    CHECK(pushed,                                 "push succeeds on empty buffer");
    CHECK(ring_buf_count(&rb) == 1u,              "count == 1 after push");
    CHECK(!ring_buf_empty(&rb),                   "not empty after push");

    bool popped = ring_buf_pop(&rb, out_vec);
    CHECK(popped,                                 "pop succeeds on non-empty buffer");
    CHECK(ring_buf_empty(&rb),                    "empty after pop");

    /* Check data integrity */
    bool data_ok = (out_vec[0] == 1.0f && out_vec[1] == 2.0f &&
                    out_vec[2] == 3.0f && out_vec[3] == 4.0f);
    CHECK(data_ok,                                "popped data matches pushed data");
}

static void test_ring_fill_and_drain(void) {
    printf("\n[test_ring_fill_and_drain]\n");
    RingBuf rb;
    ring_buf_init(&rb);

    /* CAPACITY is 4; one slot is always reserved (full = head+1 == tail)
     * so we can push CAPACITY-1 = 3 items. */
    float vec[4] = { 0 };

    vec[0] = 1.0f; bool p1 = ring_buf_push(&rb, vec);
    vec[0] = 2.0f; bool p2 = ring_buf_push(&rb, vec);
    vec[0] = 3.0f; bool p3 = ring_buf_push(&rb, vec);
    bool full_now = ring_buf_full(&rb);
    vec[0] = 4.0f; bool p4 = ring_buf_push(&rb, vec); /* should fail */

    CHECK(p1 && p2 && p3,                         "three pushes succeed");
    CHECK(full_now,                               "buffer full after 3 pushes");
    CHECK(!p4,                                    "fourth push fails (full)");
    CHECK(ring_buf_count(&rb) == 3u,              "count == 3");

    /* Drain all */
    float out[4];
    bool d1 = ring_buf_pop(&rb, out); float v1 = out[0];
    bool d2 = ring_buf_pop(&rb, out); float v2 = out[0];
    bool d3 = ring_buf_pop(&rb, out); float v3 = out[0];
    bool d4 = ring_buf_pop(&rb, out); /* should fail */

    CHECK(d1 && d2 && d3,                         "three pops succeed");
    CHECK(!d4,                                    "fourth pop fails (empty)");
    CHECK(v1 == 1.0f && v2 == 2.0f && v3 == 3.0f, "FIFO order preserved");
    CHECK(ring_buf_empty(&rb),                    "empty after drain");
}

static void test_ring_wrap_around(void) {
    printf("\n[test_ring_wrap_around]\n");
    RingBuf rb;
    ring_buf_init(&rb);

    float vec[4] = { 0 };
    float out[4];

    /* Fill 2, drain 2, fill 2 more — this wraps the head/tail indices */
    vec[0] = 10.0f; ring_buf_push(&rb, vec);
    vec[0] = 20.0f; ring_buf_push(&rb, vec);
    ring_buf_pop(&rb, out); /* drains 10 */
    ring_buf_pop(&rb, out); /* drains 20 */

    vec[0] = 30.0f; ring_buf_push(&rb, vec);
    vec[0] = 40.0f; ring_buf_push(&rb, vec);

    ring_buf_pop(&rb, out); float v1 = out[0];
    ring_buf_pop(&rb, out); float v2 = out[0];

    CHECK(v1 == 30.0f && v2 == 40.0f,            "wrap-around: FIFO intact");
    CHECK(ring_buf_empty(&rb),                    "empty after wrap-around drain");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("=== iotids arena + ring_buffer unit tests ===\n");

    /* Arena */
    test_arena_init();
    test_arena_alloc_basic();
    test_arena_alignment();
    test_arena_overflow();
    test_arena_reset();
    test_arena_peak();

    /* Ring buffer */
    test_ring_init();
    test_ring_push_pop();
    test_ring_fill_and_drain();
    test_ring_wrap_around();

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
