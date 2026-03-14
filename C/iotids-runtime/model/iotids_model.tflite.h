/*
 * iotids_model.tflite.h — INT8 quantized DNN model weights.
 *
 * AUTO-GENERATED — do not edit by hand.
 *
 * Regenerate after each Python training run:
 *   xxd -i iotids_model.tflite > model/iotids_model.tflite.h
 *
 * The xxd output will replace this placeholder with:
 *   const unsigned char iotids_model_tflite[] = { 0x18, 0x00, ... };
 *   const unsigned int  iotids_model_tflite_len = NNNN;
 *
 * This array is placed in flash (XIP) by the linker because it is const.
 * The TFLM interpreter reads model weights directly from this flash address
 * via the XIP cache — no RAM copy needed.
 *
 * Size constraint: must be <= 30 KB after quantization.
 *   Current size: UNKNOWN (placeholder — run Python export pipeline first)
 *
 * Model metadata (filled in by Python tflm_export.py):
 *   Input  : int8[1 x IOTIDS_NUM_FEATURES]  scale=<s>  zero_point=<zp>
 *   Output : int8[1 x 1]                    scale=<s>  zero_point=<zp>
 *   Ops    : FullyConnected, Quantize, Dequantize, Relu, Logistic
 */

#ifndef IOTIDS_MODEL_TFLITE_H
#define IOTIDS_MODEL_TFLITE_H

#include <stdint.h>

/*
 * Placeholder model — a minimal valid TFLite FlatBuffer that will cause
 * tflm_init() to return TFLM_ERR_ALLOC or TFLM_ERR_INPUT_SHAPE (expected
 * during development before the real model is exported). Replace with the
 * xxd -i output from the Python pipeline.
 *
 * The array below is intentionally left as a single-byte stub so the firmware
 * compiles cleanly without a real model file present in the repo.
 */
static const uint8_t iotids_model_tflite[]     = { 0x00 };
static const size_t  iotids_model_tflite_len   = 0u;

/*
 * Scaler parameters — exported by Python iotids quantize/tflm_export.py.
 * Replace the zero-initialised arrays below with the actual values.
 *
 * Python export snippet (add to tflm_export.py):
 *
 *   import numpy as np
 *   medians = scaler.medians_          # shape (n_features,)
 *   iqrs    = scaler.iqrs_             # shape (n_features,)
 *   clip_lo = scaler.clip_lo_          # shape (n_features,)
 *   clip_hi = scaler.clip_hi_          # shape (n_features,)
 *
 *   with open("model/iotids_model.tflite.h", "a") as f:
 *       def arr(name, vals):
 *           vals_str = ", ".join(f"{v:.8f}f" for v in vals)
 *           f.write(f"static const float {name}[] = {{{vals_str}}};\n")
 *       arr("IOTIDS_SCALER_MEDIANS", medians)
 *       arr("IOTIDS_SCALER_IQRS",    iqrs)
 *       arr("IOTIDS_SCALER_CLIP_LO", clip_lo)
 *       arr("IOTIDS_SCALER_CLIP_HI", clip_hi)
 */

#ifndef IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 77
#endif

/* Stub arrays — replaced by Python export */
static const float IOTIDS_SCALER_MEDIANS[IOTIDS_NUM_FEATURES] = { 0 };
static const float IOTIDS_SCALER_IQRS   [IOTIDS_NUM_FEATURES] = { 0 };
static const float IOTIDS_SCALER_CLIP_LO[IOTIDS_NUM_FEATURES] = { 0 };
static const float IOTIDS_SCALER_CLIP_HI[IOTIDS_NUM_FEATURES] = { 0 };

#endif /* IOTIDS_MODEL_TFLITE_H */
