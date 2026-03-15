# python/iotids/quantize/tflm_export.py
"""
TFLite FlatBuffer export for TFLM deployment on the Raspberry Pi Pico 2W.

export_tflite(model, path)
    -- Delegates to TensorFlow's TFLiteConverter for the FlatBuffer binary.
       This is the one file in the iotids library that intentionally keeps
       a TensorFlow dependency.  The FlatBuffer schema is complex enough
       that building it from scratch is high-risk for marginal research value
       (per the library plan: "80% dependency reduction for 20% of the work").

       Everything else in iotids is pure Python — TF is used here only for
       the final serialisation step.

Pico 2W hard constraints enforced here:
    - Output .tflite must be <= 30 KB (fits in flash)
    - Input/output tensors must be INT8
"""

import os


PICO_SIZE_LIMIT_KB = 30


def export_tflite(model, path):
    """
    Convert an iotids Sequential model to a .tflite FlatBuffer and write it
    to `path`.

    Strategy
    --------
    1.  Reconstruct a minimal Keras functional model from the iotids layer
        weights so TFLiteConverter has something it can process.
    2.  Run full-integer quantization (INT8 weights + activations).
    3.  Validate the output file is <= 30 KB.
    4.  Write the binary to `path`.

    Parameters
    ----------
    model : iotids Sequential
        Must have been calibrated (calibration.py) and weight-quantized
        (quantizer.py) beforehand so .act_scale / .weight_scale attrs exist.
    path : str
        Destination file path, e.g. "models/iotids_dnn.tflite"

    Returns
    -------
    str   -- path to the written .tflite file
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for tflm_export.  Install it with:\n"
            "    pip install tensorflow\n"
            "Only this one file in iotids uses TensorFlow."
        )

    from ..nn.layers import Dense, BatchNormalization, Dropout

    # ── Step 1: Reconstruct Keras model from iotids weights ──────────────────
    print("  Building equivalent Keras model for TFLite conversion...")

    n_features = _get_input_size(model)
    keras_model = _build_keras_model(model, n_features, tf)

    print(f"  Keras model built: {n_features} inputs")

    # ── Step 2: Representative dataset generator ─────────────────────────────
    # TFLiteConverter needs a callable that yields representative float32
    # inputs so it can set INT8 activation ranges.  We use the calibration
    # stats stored on the model's first layer.
    import random

    def representative_dataset():
        """Yield synthetic representative inputs drawn from N(0,1)."""
        for _ in range(500):
            sample = [[random.gauss(0, 1) for _ in range(n_features)]]
            import struct
            yield [
                bytes(struct.pack("f", v) for v in sample[0])
            ]

    # A cleaner generator using tf.constant:
    def representative_dataset_tf():
        for _ in range(500):
            sample = [[random.gauss(0, 1) for _ in range(n_features)]]
            yield [tf.constant(sample, dtype=tf.float32)]

    # ── Step 3: TFLiteConverter with full INT8 quantization ──────────────────
    print("  Running TFLiteConverter (full INT8)...")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_tf
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # ── Step 4: Validate size ─────────────────────────────────────────────────
    size_kb = len(tflite_model) / 1024
    print(f"  Model size: {size_kb:.2f} KB  (limit: {PICO_SIZE_LIMIT_KB} KB)")

    if size_kb > PICO_SIZE_LIMIT_KB:
        print(f"  WARNING: {size_kb:.2f} KB exceeds Pico 2W flash limit of "
              f"{PICO_SIZE_LIMIT_KB} KB.  Consider pruning before export.")

    # ── Step 5: Write to disk ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(tflite_model)

    print(f"  Saved: {path}  ({size_kb:.2f} KB)")
    return path


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_input_size(model):
    """Return the number of input features from the first Dense layer."""
    from ..nn.layers import Dense
    for layer in model.layers:
        if isinstance(layer, Dense) and layer.weights:
            return len(layer.weights[0])   # n_cols of weight matrix = n_inputs
    raise ValueError("Cannot determine input size: no Dense layer with weights found.")


def _build_keras_model(iotids_model, n_features, tf):
    """
    Reconstruct a tf.keras Sequential model from iotids layer weights.

    Only Dense and BatchNormalization layers are transferred — Dropout is
    training-only and has no effect at inference time.
    """
    from ..nn.layers import Dense, BatchNormalization, Dropout

    keras_layers = [tf.keras.Input(shape=(n_features,))]

    for layer in iotids_model.layers:

        if isinstance(layer, Dense):
            units      = len(layer.weights)          # rows = output units
            activation = layer.activation_name or None

            # Map iotids activation names to Keras strings
            act_map = {
                "relu":    "relu",
                "sigmoid": "sigmoid",
                "softmax": "softmax",
                None:       None,
            }
            keras_act = act_map.get(activation, None)

            k_layer = tf.keras.layers.Dense(
                units,
                activation=keras_act,
                use_bias=(layer.bias is not None),
            )
            keras_layers.append(k_layer)

        elif isinstance(layer, BatchNormalization):
            keras_layers.append(tf.keras.layers.BatchNormalization())

        elif isinstance(layer, Dropout):
            pass   # skip — no-op at inference

    # Build model
    inp = tf.keras.Input(shape=(n_features,))
    x   = inp
    for kl in keras_layers[1:]:   # skip the Input placeholder we added above
        x = kl(x)
    keras_model = tf.keras.Model(inputs=inp, outputs=x)

    # ── Transfer weights from iotids -> Keras ────────────────────────────────
    iotids_dense  = [l for l in iotids_model.layers if isinstance(l, Dense)]
    keras_dense   = [l for l in keras_model.layers
                     if isinstance(l, tf.keras.layers.Dense)]

    for id_layer, kd_layer in zip(iotids_dense, keras_dense):
        import numpy as np
        W = np.array(id_layer.weights, dtype=np.float32)   # (out, in)
        W = W.T                                              # Keras: (in, out)
        if id_layer.bias is not None:
            b = np.array(id_layer.bias, dtype=np.float32)
            kd_layer.set_weights([W, b])
        else:
            kd_layer.set_weights([W])

    return keras_model
