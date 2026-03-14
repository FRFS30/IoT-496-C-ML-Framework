/*
 * test_scaler.c — unit tests for feature_scaler.c.
 *
 * Runs on the host (x86/Linux) without Pico SDK dependencies. Build with:
 *
 *   gcc -std=c11 -Wall -Wextra \
 *       -DIOTIDS_NUM_FEATURES=4 \
 *       test_scaler.c \
 *       ../src/preprocessing/feature_scaler.c \
 *       -o test_scaler && ./test_scaler
 *
 * Uses a 4-feature toy dataset so expected values can be verified by hand.
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

/* Override feature count for testing */
#undef  IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 4

#include "../src/preprocessing/feature_scaler.h"

/* ---- Test helpers ---- */

static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define CHECK(cond, msg) do {                                       \
    g_tests_run++;                                                  \
    if (cond) {                                                     \
        g_tests_passed++;                                           \
        printf("  PASS: %s\n", msg);                               \
    } else {                                                        \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__);           \
    }                                                               \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) \
    CHECK(fabsf((float)(a) - (float)(b)) < (float)(eps), msg)

/* ---- Test parameters (4 features) ---- */

/* medians   = [10, 20, 0,  5  ] */
/* iqrs      = [ 4,  8, 0,  2  ]  (feature 2 has IQR=0 — constant) */
/* clip_lo   = [ 0,  5, -1, 0  ] */
/* clip_hi   = [20, 40,  1, 10 ] */

static const float medians[4]  = { 10.0f, 20.0f, 0.0f,  5.0f };
static const float iqrs[4]     = {  4.0f,  8.0f, 0.0f,  2.0f };
static const float clip_lo[4]  = {  0.0f,  5.0f,-1.0f,  0.0f };
static const float clip_hi[4]  = { 20.0f, 40.0f, 1.0f, 10.0f };

/* ---- Individual tests ---- */

static void test_init_valid(void) {
    printf("\n[test_init_valid]\n");
    Scaler s;
    bool ok = scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 4);
    CHECK(ok,       "scaler_init returns true with valid args");
    CHECK(s.ready,  "scaler.ready is true after init");
}

static void test_init_null(void) {
    printf("\n[test_init_null]\n");
    Scaler s;
    CHECK(!scaler_init(NULL,  medians, iqrs, clip_lo, clip_hi, 4), "NULL scaler");
    CHECK(!scaler_init(&s, NULL, iqrs, clip_lo, clip_hi, 4),       "NULL medians");
    CHECK(!scaler_init(&s, medians, NULL, clip_lo, clip_hi, 4),    "NULL iqrs");
}

static void test_init_wrong_nfeatures(void) {
    printf("\n[test_init_wrong_nfeatures]\n");
    Scaler s;
    bool ok = scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 99);
    CHECK(!ok, "scaler_init rejects wrong n_features");
}

static void test_transform_normal(void) {
    printf("\n[test_transform_normal]\n");
    Scaler s;
    scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 4);

    /*
     * Input:  [14, 28, 0.5, 7]
     * Clip:   [14, 28, 0.5, 7]  (all within bounds)
     * Scale:
     *   f0 = (14 - 10) / 4  =  1.0
     *   f1 = (28 - 20) / 8  =  1.0
     *   f2 = IQR==0 -> 0.0
     *   f3 = (7  -  5) / 2  =  1.0
     */
    float features[4] = { 14.0f, 28.0f, 0.5f, 7.0f };
    bool ok = scaler_transform(&s, features, 4);
    CHECK(ok, "transform returns true");
    CHECK_NEAR(features[0],  1.0f, 1e-5f, "f0 scaled correctly");
    CHECK_NEAR(features[1],  1.0f, 1e-5f, "f1 scaled correctly");
    CHECK_NEAR(features[2],  0.0f, 1e-5f, "f2 constant feature -> 0");
    CHECK_NEAR(features[3],  1.0f, 1e-5f, "f3 scaled correctly");
}

static void test_transform_clip_high(void) {
    printf("\n[test_transform_clip_high]\n");
    Scaler s;
    scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 4);

    /*
     * Input:  [100, 100, 5, 100]
     * Clip:   [ 20,  40, 1,  10]  (all clamped to clip_hi)
     * Scale:
     *   f0 = (20 - 10) / 4 = 2.5
     *   f1 = (40 - 20) / 8 = 2.5
     *   f2 = 0.0  (constant)
     *   f3 = (10 -  5) / 2 = 2.5
     */
    float features[4] = { 100.0f, 100.0f, 5.0f, 100.0f };
    scaler_transform(&s, features, 4);
    CHECK_NEAR(features[0], 2.5f, 1e-5f, "f0 clipped + scaled");
    CHECK_NEAR(features[1], 2.5f, 1e-5f, "f1 clipped + scaled");
    CHECK_NEAR(features[3], 2.5f, 1e-5f, "f3 clipped + scaled");
}

static void test_transform_clip_low(void) {
    printf("\n[test_transform_clip_low]\n");
    Scaler s;
    scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 4);

    /*
     * Input:  [-999, -999, -5, -999]
     * Clip:   [   0,    5, -1,    0]  (all clamped to clip_lo)
     * Scale:
     *   f0 = (0 - 10) / 4 = -2.5
     *   f1 = (5 - 20) / 8 = -1.875
     *   f2 = 0.0  (constant)
     *   f3 = (0 -  5) / 2 = -2.5
     */
    float features[4] = { -999.0f, -999.0f, -5.0f, -999.0f };
    scaler_transform(&s, features, 4);
    CHECK_NEAR(features[0], -2.5f,   1e-5f, "f0 clipped low + scaled");
    CHECK_NEAR(features[1], -1.875f, 1e-5f, "f1 clipped low + scaled");
    CHECK_NEAR(features[2],  0.0f,   1e-5f, "f2 constant feature -> 0");
    CHECK_NEAR(features[3], -2.5f,   1e-5f, "f3 clipped low + scaled");
}

static void test_transform_not_ready(void) {
    printf("\n[test_transform_not_ready]\n");
    Scaler s;
    memset(&s, 0, sizeof(s));  /* ready = false */
    float features[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    bool ok = scaler_transform(&s, features, 4);
    CHECK(!ok, "transform returns false when scaler not ready");
}

static void test_transform_wrong_n(void) {
    printf("\n[test_transform_wrong_n]\n");
    Scaler s;
    scaler_init(&s, medians, iqrs, clip_lo, clip_hi, 4);
    float features[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    bool ok = scaler_transform(&s, features, 3);  /* wrong n */
    CHECK(!ok, "transform returns false on n mismatch");
}

static void test_clip_standalone(void) {
    printf("\n[test_clip_standalone]\n");
    float features[4] = { -100.0f, 50.0f, 0.0f, 5.5f };
    scaler_clip(features, clip_lo, clip_hi, 4);
    CHECK_NEAR(features[0], 0.0f,  1e-5f, "f0 clipped to lo");
    CHECK_NEAR(features[1], 40.0f, 1e-5f, "f1 clipped to hi");
    CHECK_NEAR(features[2], 0.0f,  1e-5f, "f2 within bounds unchanged");
    CHECK_NEAR(features[3], 5.5f,  1e-5f, "f3 within bounds unchanged");
}

/* ---- Main ---- */

int main(void) {
    printf("=== iotids feature_scaler unit tests ===\n");

    test_init_valid();
    test_init_null();
    test_init_wrong_nfeatures();
    test_transform_normal();
    test_transform_clip_high();
    test_transform_clip_low();
    test_transform_not_ready();
    test_transform_wrong_n();
    test_clip_standalone();

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
