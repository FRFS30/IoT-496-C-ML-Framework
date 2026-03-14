/*
 * model_runner.c — TFLM inference engine implementation.
 *
 * C99 with TFLM headers compiled as C++ (via the resolver translation unit).
 * All TFLM C++ types are hidden behind the opaque void* from tflm_resolver.h.
 *
 * Memory layout (all static, no heap):
 *   g_arena_buf   — 200 KB SRAM tensor arena (activations)
 *   g_interpreter — allocated inside arena by TFLM
 *   model weights — stay in flash (const uint8_t[] from tflite.h)
 *
 * INT8 quantization boundary:
 *   TFLM's full-integer quantized model expects int8_t inputs and produces
 *   int8_t outputs. We perform float32->int8 and int8->float32 conversion
 *   here using the scale and zero_point stored in the input/output tensors.
 *
 *   q = clamp(round(x / scale) + zero_point,  -128, 127)
 *   x = (q - zero_point) * scale
 */

#include "model_runner.h"
#include "arena.h"
#include "tflm_resolver.h"

/* TFLM C++ headers — compiled as C++ via the CMake INTERFACE target */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Pico SDK timing */
#include "pico/stdlib.h"     /* time_us_32() */

#include <math.h>
#include <string.h>
#include <assert.h>

/* --------------------------------------------------------------------------
 * Static storage
 * -------------------------------------------------------------------------- */

/* 200 KB tensor arena — must be 16-byte aligned for SIMD safety.
 * __attribute__((aligned(16))) works for both GCC and Clang on ARM Cortex-M.
 */
static uint8_t g_arena_buf[IOTIDS_ARENA_SIZE] __attribute__((aligned(16)));

static Arena   g_arena;

/* Opaque TFLM interpreter handle (C++ type hidden behind void*) */
static tflite::MicroInterpreter *g_interpreter = NULL;

static bool     g_initialised  = false;
static uint32_t g_last_latency = 0u;  /* µs of the most recent Invoke() */

/* --------------------------------------------------------------------------
 * INT8 quantization helpers
 * -------------------------------------------------------------------------- */

static inline int8_t float_to_int8(float x, float scale, int32_t zero_point) {
    float q = (x / scale) + (float)zero_point;
    /* Round-to-nearest, then saturate to [-128, 127] */
    int32_t qi = (int32_t)roundf(q);
    if (qi < -128) qi = -128;
    if (qi >  127) qi =  127;
    return (int8_t)qi;
}

static inline float int8_to_float(int8_t q, float scale, int32_t zero_point) {
    return ((float)(q - zero_point)) * scale;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

TflmStatus tflm_init(const uint8_t *model_data, size_t model_len) {
    if (model_data == NULL || model_len == 0u) {
        return TFLM_ERR_NULL;
    }

    /* Initialise arena */
    arena_init(&g_arena, g_arena_buf, IOTIDS_ARENA_SIZE);

    /* Parse the FlatBuffer model */
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        /* Schema mismatch — re-export the .tflite from Python pipeline */
        return TFLM_ERR_NULL;
    }

    /* Build the minimal op resolver (C++ object, C-linkage wrapper) */
    void *resolver_handle = resolver_create();
    if (resolver_handle == NULL) {
        return TFLM_ERR_NULL;
    }
    tflite::MicroMutableOpResolver<5> *resolver =
        static_cast<tflite::MicroMutableOpResolver<5> *>(resolver_handle);

    /*
     * Allocate the MicroInterpreter inside the arena.
     * We use placement-new into a static buffer so the interpreter struct
     * itself consumes arena bytes rather than needing heap.
     */
    static uint8_t interp_buf[sizeof(tflite::MicroInterpreter)]
        __attribute__((aligned(16)));

    g_interpreter = new (interp_buf) tflite::MicroInterpreter(
        model,
        *resolver,
        g_arena_buf,
        IOTIDS_ARENA_SIZE
    );

    /* Allocate tensors — this sets up the tensor arena layout. */
    TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        return TFLM_ERR_ALLOC;
    }

    /* Verify input tensor matches expected feature count */
    TfLiteTensor *input_tensor = g_interpreter->input(0);
    int n_elements = 1;
    for (int i = 0; i < input_tensor->dims->size; i++) {
        n_elements *= input_tensor->dims->data[i];
    }
    if (n_elements != IOTIDS_NUM_FEATURES) {
        return TFLM_ERR_INPUT_SHAPE;
    }

    g_initialised = true;
    return TFLM_OK;
}

TflmStatus tflm_run(const float *input, float *output) {
    if (!g_initialised)     return TFLM_ERR_NOT_INIT;
    if (input == NULL)      return TFLM_ERR_NULL;

    TfLiteTensor *input_tensor  = g_interpreter->input(0);
    TfLiteTensor *output_tensor = g_interpreter->output(0);

    /* ---- Quantise float32 input -> INT8 ---- */
    float   in_scale  = input_tensor->params.scale;
    int32_t in_zp     = input_tensor->params.zero_point;
    int8_t *in_data   = input_tensor->data.int8;

    for (int i = 0; i < IOTIDS_NUM_FEATURES; i++) {
        in_data[i] = float_to_int8(input[i], in_scale, in_zp);
    }

    /* ---- Run inference ---- */
    uint32_t t0 = time_us_32();
    TfLiteStatus status = g_interpreter->Invoke();
    g_last_latency = time_us_32() - t0;

    if (status != kTfLiteOk) {
        return TFLM_ERR_INVOKE;
    }

    /* ---- Dequantise INT8 output -> float32 ---- */
    if (output != NULL) {
        float   out_scale = output_tensor->params.scale;
        int32_t out_zp    = output_tensor->params.zero_point;
        int8_t  out_q     = output_tensor->data.int8[0];
        *output = int8_to_float(out_q, out_scale, out_zp);

        /* Clamp to [0,1] — sigmoid output should already be in range, but
         * dequantization rounding can produce tiny out-of-range values. */
        if (*output < 0.0f) *output = 0.0f;
        if (*output > 1.0f) *output = 1.0f;
    }

    return TFLM_OK;
}

int tflm_classify(const float *input) {
    float prob = 0.0f;
    TflmStatus s = tflm_run(input, &prob);
    if (s != TFLM_OK) return -1;
    return (prob >= IOTIDS_THRESHOLD) ? 1 : 0;
}

uint32_t tflm_get_latency_us(void) {
    return g_last_latency;
}

size_t tflm_arena_used(void) {
    if (!g_initialised) return 0u;
    return arena_used_bytes(&g_arena);
}
