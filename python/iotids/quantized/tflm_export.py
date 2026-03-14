"""
tflm_export.py — Export iotids model to TFLite FlatBuffer for Pico 2W deployment.

DESIGN NOTE (from implementation plan):
  The TFLite FlatBuffer schema is complex to implement from scratch.
  This module delegates the serialisation step to TensorFlow, which is
  academically defensible — it represents ~20% of the effort for ~80% of
  the dependency reduction. Everything else in iotids is dependency-free.

Pipeline:
  prune -> fine-tune -> calibrate -> quantize_model_weights -> export_tflite

The exported .tflite file is then embedded into C via:
  xxd -i iotids_model.tflite > model/iotids_model.tflite.h
"""

import os
import struct


SIZE_LIMIT_BYTES = 30 * 1024   # 30 KB — Pico 2W hard constraint


def export_tflite(model, path, use_tensorflow=True):
    """
    Export a trained (and optionally pruned/quantized) iotids Sequential
    model to a TFLite FlatBuffer.

    model          : iotids Sequential
    path           : output .tflite file path
    use_tensorflow : if True (recommended), rebuild in TF and convert;
                     if False, attempt raw FlatBuffer write (experimental).

    Validates output file size <= 30 KB after export.
    """
    if use_tensorflow:
        _export_via_tensorflow(model, path)
    else:
        _export_flatbuffer(model, path)

    size = os.path.getsize(path)
    if size > SIZE_LIMIT_BYTES:
        raise RuntimeError(
            f"Exported model is {size / 1024:.1f} KB — exceeds 30 KB Pico 2W limit. "
            "Consider pruning further or reducing layer width."
        )
    print(f"Exported: {path} ({size / 1024:.1f} KB) — OK")
    return size


# ------------------------------------------------------------------ #
# TensorFlow-delegated export (recommended)
# ------------------------------------------------------------------ #
def _export_via_tensorflow(model, path):
    """
    Reconstruct the iotids weight values inside a TF/Keras model,
    then use tf.lite.TFLiteConverter for INT8 quantization and export.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for tflm_export. "
            "Install it on the training server, not the Pico."
        )

    from ..nn.layers import Dense as IotDense

    # Build equivalent Keras Sequential
    keras_layers = []
    input_set = False
    for layer in model.layers:
        if isinstance(layer, IotDense):
            act = layer._act_name or "linear"
            if not input_set:
                keras_layers.append(
                    tf.keras.layers.Dense(
                        layer.units,
                        activation=act,
                        input_shape=(layer.in_features,),
                        use_bias=layer.use_bias,
                    )
                )
                input_set = True
            else:
                keras_layers.append(
                    tf.keras.layers.Dense(
                        layer.units, activation=act, use_bias=layer.use_bias
                    )
                )

    keras_model = tf.keras.Sequential(keras_layers)
    keras_model.build()

    # Copy weights from iotids into Keras
    for iot_layer, keras_layer in zip(
        [l for l in model.layers if isinstance(l, IotDense)],
        [l for l in keras_model.layers if isinstance(l, tf.keras.layers.Dense)],
    ):
        import array as _array
        W = []
        for i in range(iot_layer.in_features):
            W.append(list(iot_layer.W[i]))
        b = list(iot_layer.b) if iot_layer.use_bias else None
        keras_layer.set_weights([W, b] if b is not None else [W])

    # INT8 post-training quantization via TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)


# ------------------------------------------------------------------ #
# Minimal raw FlatBuffer writer (experimental — use TF path instead)
# ------------------------------------------------------------------ #
def _export_flatbuffer(model, path):
    """
    Writes a minimal TFLite FlatBuffer for a fully-connected INT8 model.
    This is experimental and covers simple Sequential(Dense) topologies only.
    For production use, prefer _export_via_tensorflow.
    """
    from ..nn.layers import Dense as IotDense
    from .quantizer import compute_scale_zeropoint, quantize_tensor

    # Collect quantized weight arrays and metadata
    layers = [l for l in model.layers if isinstance(l, IotDense)]
    if not layers:
        raise ValueError("No Dense layers found in model")

    # Minimal FlatBuffer layout:
    # [4-byte magic][version][n_layers][per-layer: in, out, W_scale, W_zp, W_int8[], b_int8[]...]
    # This is a simplified binary — readable by a custom C loader, not standard TFLM.
    # For TFLM compatibility, use the TF path.
    with open(path, "wb") as f:
        f.write(b"IOTLM")                           # magic
        f.write(struct.pack(">B", 1))               # version
        f.write(struct.pack(">B", len(layers)))     # n_layers

        for layer in layers:
            flat_W = [v for row in layer.W for v in row]
            scale, zp = compute_scale_zeropoint(flat_W)
            q_W = quantize_tensor(flat_W, scale, zp)

            f.write(struct.pack(">HH", layer.in_features, layer.units))
            f.write(struct.pack(">f", scale))
            f.write(struct.pack(">b", zp))
            f.write(bytes([b & 0xFF for b in q_W]))

            if layer.use_bias:
                scale_b, zp_b = compute_scale_zeropoint(layer.b)
                q_b = quantize_tensor(layer.b, scale_b, zp_b)
                f.write(struct.pack(">?", True))
                f.write(struct.pack(">f", scale_b))
                f.write(struct.pack(">b", zp_b))
                f.write(bytes([b & 0xFF for b in q_b]))
            else:
                f.write(struct.pack(">?", False))
