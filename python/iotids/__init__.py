"""
iotids — Custom ML library for IoT intrusion detection.
Target: Raspberry Pi Pico 2W (4 MB Flash, 520 KB SRAM)
Dataset: CIC-IDS-2017

Python Training Library — runs on university servers.
Produces a quantized .tflite model for on-device deployment.

Build order (per architecture plan):
  1. core/       — Tensor primitive and math ops
  2. utils/      — Low-level helpers
  3. data/       — CSV reader, preprocessing, dataset
  4. metrics/    — Classification evaluation
  5. nn/         — DNN stack
  6. prune/      — Weight and neuron pruning
  7. forest/     — Random Forest baseline
  8. federated/  — FedAvg simulation
  9. quantize/   — INT8 quantization and .tflite export
"""

from . import core, utils, data, metrics, nn, forest, federated, prune, quantize
