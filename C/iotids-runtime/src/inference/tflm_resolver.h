#ifndef IOTIDS_TFLM_RESOLVER_H
#define IOTIDS_TFLM_RESOLVER_H

/*
 * tflm_resolver.h — C-linkage wrapper around the TFLM MicroMutableOpResolver.
 *
 * TFLM's AllOpsResolver pulls in ~300 KB of op code. The iotids DNN uses only:
 *
 *   FullyConnected  — every Dense layer
 *   Quantize        — INT8 input quantization boundary
 *   Dequantize      — INT8 output dequantization boundary
 *   Relu            — activation after hidden Dense layers
 *   Logistic        — sigmoid activation on the output layer
 *
 * Registering only these five cuts the TFLM footprint by ~60%.
 *
 * This file exposes a C interface so model_runner.c (pure C) can call it.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * resolver_create — allocate and return an opaque resolver handle.
 * Returns NULL on failure (should never happen on a correctly built firmware).
 */
void *resolver_create(void);

/*
 * resolver_destroy — free the resolver. Call after the interpreter is torn
 * down (typically never on Pico — firmware runs forever, but good practice).
 */
void resolver_destroy(void *resolver);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_TFLM_RESOLVER_H */
