import math
from ..core.dtypes import INT8_MIN, INT8_MAX


def compute_scale_zeropoint(tensor_vals, dtype="int8"):
    """
    Per-tensor symmetric calibration.
    Returns (scale, zero_point) for INT8 mapping.

    scale      = max_abs / 127
    zero_point = 0  (symmetric)
    """
    if not tensor_vals:
        return 1.0, 0

    max_abs = max(abs(v) for v in tensor_vals)
    if max_abs == 0.0:
        return 1.0, 0

    if dtype == "int8":
        scale = max_abs / 127.0
        zero_point = 0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return scale, zero_point


def quantize_tensor(tensor_vals, scale, zero_point):
    """
    Map float32 -> INT8.
    Returns list of ints clamped to [-128, 127].
    """
    out = []
    for v in tensor_vals:
        q = int(round(v / scale)) + zero_point
        out.append(max(INT8_MIN, min(INT8_MAX, q)))
    return out


def dequantize_tensor(qtensor, scale, zero_point):
    """Map INT8 -> float32 approximation."""
    return [(q - zero_point) * scale for q in qtensor]


def quantize_model_weights(model):
    """
    Walk all Dense layers and quantize W + b in-place.
    Also stores (scale, zero_point) on each layer for export.
    Returns dict of per-layer quantization params.
    """
    from ..nn.layers import Dense
    qparams = {}

    for i, layer in enumerate(model.layers):
        if not isinstance(layer, Dense):
            continue

        flat_W = [v for row in layer.W for v in row]
        scale_W, zp_W = compute_scale_zeropoint(flat_W)
        q_W = quantize_tensor(flat_W, scale_W, zp_W)

        # Write back as float (simulated quantization)
        dq_W = dequantize_tensor(q_W, scale_W, zp_W)
        idx = 0
        for r in range(layer.in_features):
            for c in range(layer.units):
                layer.W[r][c] = dq_W[idx]
                idx += 1

        layer_params = {"W_scale": scale_W, "W_zp": zp_W}

        if layer.use_bias:
            scale_b, zp_b = compute_scale_zeropoint(layer.b)
            q_b = quantize_tensor(layer.b, scale_b, zp_b)
            layer.b = dequantize_tensor(q_b, scale_b, zp_b)
            layer_params["b_scale"] = scale_b
            layer_params["b_zp"]    = zp_b

        # Store on layer for TFLM export
        layer.quant_params = layer_params
        qparams[i] = layer_params

    return qparams
