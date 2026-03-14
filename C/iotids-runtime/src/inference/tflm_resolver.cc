/*
 * tflm_resolver.cc — minimal TFLM op resolver, C++ implementation.
 *
 * Only ops actually used by the iotids DNN are registered. This is the
 * single most impactful binary-size optimisation available in TFLM.
 *
 * Compile with the Pico SDK C++ toolchain; the C-linkage wrappers let
 * model_runner.c call through without knowing about C++ types.
 *
 * TFLM version assumed: tensorflow-lite-micro (tip-of-tree or 2.x stable).
 * Adjust the include paths in CMakeLists.txt as needed.
 */

#include "tflm_resolver.h"

/* TFLM headers — provided by the pico-tflmicro CMake target */
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

/* We only need 5 ops; reserve exactly 5 slots to avoid wasted static RAM. */
static constexpr int kNumOps = 5;

using OpsResolver = tflite::MicroMutableOpResolver<kNumOps>;

extern "C" {

void *resolver_create(void) {
    /*
     * Allocate in static storage so it outlives this function. On a bare-metal
     * Pico there is no heap, and the resolver must persist for the lifetime of
     * the interpreter. Using a function-static is safe for a single-interpreter
     * design (one model, one resolver, forever).
     */
    static OpsResolver resolver;

    /* FullyConnected covers every Dense layer in the DNN. */
    resolver.AddFullyConnected();

    /*
     * Quantize / Dequantize handle the INT8 <-> float32 boundary ops that
     * TFLM inserts at the model's input and output tensors when the model was
     * exported with full-integer quantization.
     */
    resolver.AddQuantize();
    resolver.AddDequantize();

    /*
     * Relu is the hidden-layer activation. If LeakyRelu or another activation
     * was used during training, add it here and adjust kNumOps above.
     */
    resolver.AddRelu();

    /*
     * Logistic = sigmoid. Used on the binary-classification output neuron.
     * Note: if the model was exported with from_logits=True and the sigmoid
     * was folded into the loss, this op may not appear in the .tflite graph.
     * Keep it registered regardless — unused registrations waste ~4 bytes of
     * pointer storage, not code size.
     */
    resolver.AddLogistic();

    return static_cast<void *>(&resolver);
}

void resolver_destroy(void * /* resolver */) {
    /*
     * Nothing to do — resolver lives in static storage.
     * This function exists for symmetry and future-proofing if the design
     * ever moves to dynamic allocation.
     */
}

} /* extern "C" */
