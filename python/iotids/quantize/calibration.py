# python/iotids/quantize/calibration.py
"""
Post-training calibration for INT8 quantization.

calibrate(model, representative_dataset)
    -- runs a forward pass over the calibration set and records the
       per-layer activation min/max ranges needed to set scale/zero-point
       for activations (not just weights).

The calibration dataset should be a balanced sample of real inputs
(benign + attack) so activation ranges are representative of both classes.
"""

from .quantizer import compute_scale_zeropoint


def calibrate(model, representative_dataset):
    """
    Collect per-layer activation statistics on a representative dataset.

    Parameters
    ----------
    model : Sequential
        The iotids DNN model (must have a .layers attribute).
    representative_dataset : list[list[float]]
        A flat list of input feature vectors.  Aim for ~500–2000 samples,
        stratified across benign and attack classes.

    Side-effects
    ------------
    Attaches to each Dense / BatchNorm layer:
        .act_min   float  -- minimum observed activation value
        .act_max   float  -- maximum observed activation value
        .act_scale float  -- derived quantization scale
        .act_zp    int    -- derived zero-point
    """
    from ..nn.layers import Dense, BatchNormalization  # avoid circular import

    # ── Initialise per-layer tracking ────────────────────────────────────────
    for layer in model.layers:
        if isinstance(layer, (Dense, BatchNormalization)):
            layer.act_min =  float("inf")
            layer.act_max = -float("inf")

    # ── Forward pass — collect activations layer by layer ───────────────────
    print(f"  Calibrating on {len(representative_dataset)} samples...")
    n_batches = max(1, len(representative_dataset) // 64)

    for batch_idx in range(n_batches):
        start = batch_idx * 64
        end   = min(start + 64, len(representative_dataset))
        batch = representative_dataset[start:end]

        if not batch:
            break

        # Run the forward pass, capturing intermediate outputs
        x = batch
        for layer in model.layers:
            x = layer.forward(x, training=False)

            if isinstance(layer, (Dense, BatchNormalization)):
                flat = _flatten_activations(x)
                if flat:
                    layer.act_min = min(layer.act_min, min(flat))
                    layer.act_max = max(layer.act_max, max(flat))

    # ── Derive scale / zero-point from observed ranges ───────────────────────
    calibrated = 0
    for layer in model.layers:
        if isinstance(layer, (Dense, BatchNormalization)):
            if layer.act_min == float("inf"):
                continue   # layer was never reached (e.g. after early exit)

            scale, zp = compute_scale_zeropoint(
                [layer.act_min, layer.act_max]
            )
            layer.act_scale = scale
            layer.act_zp    = zp
            calibrated += 1

    print(f"  Calibration complete — {calibrated} layer(s) profiled.")
    return model


# ── helpers ──────────────────────────────────────────────────────────────────

def _flatten_activations(x):
    """
    Flatten a batch of activations (list of rows) to a single flat list
    so we can compute global min/max efficiently.
    """
    out = []
    if not x:
        return out
    if isinstance(x[0], (list, tuple)):
        for row in x:
            out.extend(float(v) for v in row)
    else:
        out.extend(float(v) for v in x)
    return out
