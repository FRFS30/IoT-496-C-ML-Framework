#include "feature_scaler.h"
#include <assert.h>
#include <stddef.h>

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

bool scaler_init(Scaler *s,
                 const float *medians,
                 const float *iqrs,
                 const float *clip_lo,
                 const float *clip_hi,
                 int n_features) {
    if (s == NULL || medians == NULL || iqrs == NULL ||
        clip_lo == NULL || clip_hi == NULL) {
        return false;
    }
    if (n_features != IOTIDS_NUM_FEATURES) {
        return false;
    }

    s->medians    = medians;
    s->iqrs       = iqrs;
    s->clip_lo    = clip_lo;
    s->clip_hi    = clip_hi;
    s->n_features = n_features;
    s->ready      = true;
    return true;
}

void scaler_clip(float *features, const float *lo, const float *hi, int n) {
    assert(features != NULL);
    assert(lo != NULL && hi != NULL);
    for (int i = 0; i < n; i++) {
        features[i] = clampf(features[i], lo[i], hi[i]);
    }
}

bool scaler_transform(const Scaler *s, float *raw_features, int n) {
    if (s == NULL || !s->ready) return false;
    if (raw_features == NULL)   return false;
    if (n != s->n_features)     return false;

    /* Step 1: clip to 1st–99th percentile bounds */
    scaler_clip(raw_features, s->clip_lo, s->clip_hi, n);

    /* Step 2: robust scale — (x - median) / IQR */
    for (int i = 0; i < n; i++) {
        float iqr = s->iqrs[i];
        if (iqr == 0.0f) {
            /*
             * Constant feature — IQR is zero, scaling is undefined.
             * Set to 0 to match Python behaviour (divide-by-zero guard).
             * These features contribute nothing to the model; a later
             * pruning/feature-selection pass should remove them.
             */
            raw_features[i] = 0.0f;
        } else {
            raw_features[i] = (raw_features[i] - s->medians[i]) / iqr;
        }
    }

    return true;
}
