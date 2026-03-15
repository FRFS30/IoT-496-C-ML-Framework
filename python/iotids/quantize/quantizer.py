# python/iotids/quantize/quantizer.py
"""
INT8 per-tensor quantization primitives.

compute_scale_zeropoint  -- derive scale + zero-point from a float tensor
quantize_tensor          -- float32 -> int8
dequantize_tensor        -- int8   -> float32
quantize_model_weights   -- walk every Dense layer and quantize in-place
"""


# ── constants ────────────────────────────────────────────────────────────────
INT8_MIN  = -128
INT8_MAX  =  127


# ── core math ────────────────────────────────────────────────────────────────

def compute_scale_zeropoint(tensor, dtype="int8"):
    """
    Compute per-tensor affine quantization parameters.

    q = clamp(round(x / scale) + zero_point,  qmin, qmax)
    x ≈ scale * (q - zero_point)

    Returns (scale: float, zero_point: int)
    """
    if dtype != "int8":
        raise ValueError(f"Only int8 supported, got {dtype!r}")

    flat = _flatten(tensor)
    if not flat:
        return 1.0, 0

    t_min = min(flat)
    t_max = max(flat)

    # Symmetric around zero is simpler for weights; asymmetric for activations.
    # We implement asymmetric (covers both cases).
    t_min = min(t_min, 0.0)   # always include 0 in range
    t_max = max(t_max, 0.0)

    scale = (t_max - t_min) / (INT8_MAX - INT8_MIN)
    if scale == 0.0:
        scale = 1e-8            # avoid division by zero for constant tensors

    zero_point = INT8_MIN - round(t_min / scale)
    zero_point = max(INT8_MIN, min(INT8_MAX, zero_point))

    return scale, int(zero_point)


def quantize_tensor(tensor, scale, zero_point):
    """
    Quantize a float32 tensor to int8.

    Accepts list, list-of-lists, or flat iterable.
    Returns same shape as input but with int8 values (stored as Python ints).
    """
    if isinstance(tensor[0], (list, tuple)):
        return [quantize_tensor(row, scale, zero_point) for row in tensor]

    result = []
    for x in tensor:
        q = round(x / scale) + zero_point
        q = max(INT8_MIN, min(INT8_MAX, q))
        result.append(q)
    return result


def dequantize_tensor(qtensor, scale, zero_point):
    """
    Dequantize an int8 tensor back to float32.

    x ≈ scale * (q - zero_point)
    """
    if isinstance(qtensor[0], (list, tuple)):
        return [dequantize_tensor(row, scale, zero_point) for row in qtensor]

    return [scale * (q - zero_point) for q in qtensor]


# ── model-level quantization ─────────────────────────────────────────────────

def quantize_model_weights(model):
    """
    Walk every layer in a Sequential model and quantize Dense weight matrices
    and bias vectors in-place (float32 -> int8 -> dequantized float32).

    The weights are stored back as dequantized float32 so the existing
    forward-pass arithmetic still works — this is post-training quantization
    (PTQ), not true fixed-point inference.  For real INT8 inference on the
    Pico 2W the .tflite export (tflm_export.py) handles the final conversion.

    Attaches .weight_scale, .weight_zp, .bias_scale, .bias_zp attributes
    to each layer so calibration.py and tflm_export.py can read the params.
    """
    from ..nn.layers import Dense  # local import avoids circular dependency

    for layer in model.layers:
        if not isinstance(layer, Dense):
            continue

        # ── Quantize weight matrix ────────────────────────────────────────
        W = layer.weights                         # list of rows
        scale_w, zp_w = compute_scale_zeropoint(W)
        W_q  = quantize_tensor(W, scale_w, zp_w)
        W_dq = dequantize_tensor(W_q, scale_w, zp_w)
        layer.weights    = W_dq
        layer.weight_scale = scale_w
        layer.weight_zp    = zp_w

        # ── Quantize bias vector ──────────────────────────────────────────
        if layer.bias is not None:
            scale_b, zp_b = compute_scale_zeropoint(layer.bias)
            b_q  = quantize_tensor(layer.bias, scale_b, zp_b)
            b_dq = dequantize_tensor(b_q, scale_b, zp_b)
            layer.bias      = b_dq
            layer.bias_scale = scale_b
            layer.bias_zp    = zp_b

    return model


# ── internal helpers ─────────────────────────────────────────────────────────

def _flatten(tensor):
    """Recursively flatten a nested list to a flat list of floats."""
    out = []
    for v in tensor:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten(v))
        else:
            out.append(float(v))
    return out
