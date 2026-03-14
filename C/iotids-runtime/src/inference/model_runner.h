#ifndef IOTIDS_MODEL_RUNNER_H
#define IOTIDS_MODEL_RUNNER_H

/*
 * model_runner.h — TFLM inference engine for the iotids IDS runtime.
 *
 * Provides a thin, pure-C API over TFLM. Internally it holds a
 * MicroInterpreter, the tensor arena, and timing state. All memory is static
 * (no malloc). Designed for a single model, single inference context.
 *
 * Lifecycle:
 *   1. tflm_init()       — call once at boot after Wi-Fi and scaler init.
 *   2. tflm_run()        — call for every network flow feature vector.
 *   3. tflm_get_latency_us() — optional, for profiling.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Result codes ---- */
typedef enum {
    TFLM_OK              =  0,
    TFLM_ERR_NULL        = -1,   /* NULL pointer argument                  */
    TFLM_ERR_ALLOC       = -2,   /* arena too small                        */
    TFLM_ERR_INVOKE      = -3,   /* interpreter Invoke() failed            */
    TFLM_ERR_INPUT_SHAPE = -4,   /* input tensor size mismatch             */
    TFLM_ERR_NOT_INIT    = -5,   /* tflm_run called before tflm_init       */
} TflmStatus;

/*
 * IOTIDS_NUM_FEATURES — number of numeric features per network flow sample.
 * Must match the .tflite model's input tensor shape (1 x N_FEATURES).
 * Exported by the Python pipeline into model/iotids_model.tflite.h as a
 * compile-time constant; override with -DIOTIDS_NUM_FEATURES=<n> in CMake.
 */
#ifndef IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 77
#endif

/*
 * Classification threshold for binary intrusion detection.
 * Values above this are classified as ATTACK (1), below as BENIGN (0).
 * Default 0.5; recalibrate with threshold_sweep() after quantization.
 * Override with -DIOTIDS_THRESHOLD=0.35f in CMake after calibration.
 */
#ifndef IOTIDS_THRESHOLD
#define IOTIDS_THRESHOLD 0.5f
#endif

/* ---- Public API ---- */

/*
 * tflm_init — load the .tflite FlatBuffer and prepare the interpreter.
 *
 *   model_data : pointer to the const uint8_t[] from iotids_model.tflite.h
 *   model_len  : sizeof(iotids_model_tflite) — used for bounds checking only
 *
 * Returns TFLM_OK on success. Any other value is fatal; loop + blink LED.
 *
 * MUST be called before tflm_run(). Safe to call exactly once.
 */
TflmStatus tflm_init(const uint8_t *model_data, size_t model_len);

/*
 * tflm_run — execute one inference pass.
 *
 *   input  : float32 array of IOTIDS_NUM_FEATURES pre-scaled features.
 *            The scaler must be applied BEFORE this call.
 *   output : pointer to a single float32 that receives the sigmoid
 *            probability (0.0 – 1.0). May be NULL if only the binary
 *            label is needed.
 *
 * Returns TFLM_OK on success, or an error code. On success, *output holds
 * the model probability and the binary classification is:
 *   (*output >= IOTIDS_THRESHOLD) ? 1 : 0   (1 = ATTACK, 0 = BENIGN)
 *
 * This function:
 *   1. Quantises the float32 input to INT8 using the model's input scale/zp.
 *   2. Calls interpreter Invoke().
 *   3. Dequantises the INT8 output back to float32.
 *   4. Records the latency for tflm_get_latency_us().
 */
TflmStatus tflm_run(const float *input, float *output);

/*
 * tflm_classify — convenience wrapper: calls tflm_run and applies threshold.
 * Returns 1 (ATTACK), 0 (BENIGN), or -1 on error.
 */
int tflm_classify(const float *input);

/*
 * tflm_get_latency_us — microseconds taken by the most recent tflm_run call.
 * Uses the Pico SDK's time_us_32(). Returns 0 before first inference.
 */
uint32_t tflm_get_latency_us(void);

/*
 * tflm_arena_used — bytes consumed in the tensor arena after the first
 * inference call. Call once after tflm_init + one tflm_run to verify fit.
 */
size_t tflm_arena_used(void);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_MODEL_RUNNER_H */
