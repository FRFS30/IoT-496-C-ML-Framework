from .quantizer import compute_scale_zeropoint
from ..nn.layers import Dense, BatchNormalization


class CalibrationRecord:
    """Tracks min/max of activations seen during forward passes."""

    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.samples = 0

    def update(self, activations):
        for v in activations:
            if v < self.min_val:
                self.min_val = v
            if v > self.max_val:
                self.max_val = v
        self.samples += len(activations)

    def scale_zp(self):
        if self.min_val == float("inf"):
            return 1.0, 0
        vals = [self.min_val, self.max_val]
        return compute_scale_zeropoint(vals)


def calibrate(model, representative_dataset):
    """
    Run calibration forward passes to collect per-layer activation ranges.

    model                : Sequential
    representative_dataset: Dataset — should be a stratified sample
                           (benign + attack balanced) via dataset.sample()

    Returns dict {layer_index: CalibrationRecord}.
    """
    records = {i: CalibrationRecord() for i in range(len(model.layers))}

    model._set_training(False)

    for X_batch, _ in representative_dataset.batch(128):
        # Convert tensor to list-of-rows
        r, c = X_batch.shape
        rows = [[X_batch.data[i * c + j] for j in range(c)] for i in range(r)]

        activations = rows
        for idx, layer in enumerate(model.layers):
            activations = layer.forward(activations)
            flat = [v for row in activations for v in row]
            records[idx].update(flat)

    return records


def apply_calibration(model, records):
    """
    Store calibration-derived scale/zero_point on each layer.
    Used downstream by tflm_export.
    """
    for idx, layer in enumerate(model.layers):
        if idx in records:
            scale, zp = records[idx].scale_zp()
            layer.activation_scale = scale
            layer.activation_zp    = zp
