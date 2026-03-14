#ifndef IOTIDS_FEATURE_SCALER_H
#define IOTIDS_FEATURE_SCALER_H

/*
 * feature_scaler.h — on-device RobustScaler for network flow features.
 *
 * Mirrors the Python iotids preprocessing.RobustScaler exactly:
 *
 *   scaled = (x - median) / IQR
 *
 * The median and IQR values are exported from the Python training pipeline
 * into model/scaler_params.h as two const float[] arrays (one per feature).
 * This file loads them at boot and applies the transform each inference pass.
 *
 * All arithmetic is float32 — no dynamic allocation, no libm beyond basic ops.
 * Outlier clipping (1st–99th percentile) is applied before scaling, matching
 * the Python clip_outliers() call.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of features — must match IOTIDS_NUM_FEATURES in model_runner.h */
#ifndef IOTIDS_NUM_FEATURES
#define IOTIDS_NUM_FEATURES 77
#endif

/*
 * Scaler parameter struct.
 * Populated by scaler_init() from the exported C header arrays.
 */
typedef struct {
    const float *medians;       /* per-feature medians  [n_features]      */
    const float *iqrs;          /* per-feature IQRs     [n_features]       */
    const float *clip_lo;       /* 1st-percentile clip bounds [n_features] */
    const float *clip_hi;       /* 99th-percentile clip bounds[n_features] */
    int          n_features;    /* must equal IOTIDS_NUM_FEATURES           */
    bool         ready;         /* set true by scaler_init on success       */
} Scaler;

/*
 * scaler_init — bind the Scaler to exported parameter arrays.
 *
 *   medians, iqrs, clip_lo, clip_hi : const float[] from scaler_params.h
 *   n_features                      : length of each array
 *
 * Returns true on success, false if any pointer is NULL or n_features
 * doesn't match IOTIDS_NUM_FEATURES.
 *
 * Call once at boot after flash_read_model().
 */
bool scaler_init(Scaler *s,
                 const float *medians,
                 const float *iqrs,
                 const float *clip_lo,
                 const float *clip_hi,
                 int n_features);

/*
 * scaler_transform — apply RobustScaler to one feature vector in-place.
 *
 *   raw_features : float32 array of n raw network-flow feature values.
 *                  Modified in place: clipped, then scaled.
 *   n            : must equal s->n_features.
 *
 * Pipeline per feature i:
 *   1. Clip:  x = clamp(x, clip_lo[i], clip_hi[i])
 *   2. Scale: x = (x - median[i]) / iqr[i]
 *             (if iqr[i] == 0, x is set to 0 — constant feature guard)
 *
 * Returns true on success, false if scaler not initialised or n mismatch.
 */
bool scaler_transform(const Scaler *s, float *raw_features, int n);

/*
 * scaler_clip — apply only the outlier clipping step, without scaling.
 * Useful for a pre-validation pass before calling scaler_transform.
 *
 *   features : float32 array, modified in place.
 *   lo, hi   : per-feature bounds arrays (length n).
 *   n        : number of features.
 */
void scaler_clip(float *features, const float *lo, const float *hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_FEATURE_SCALER_H */
