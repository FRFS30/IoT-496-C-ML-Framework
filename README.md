# IoT-496 C ML Framework

A federated, privacy-preserving, and adversarially robust intrusion detection system (IDS) for IoT devices, built as part of an independent research capstone. The project spans the full ML pipeline — from training on a university server down to bare-metal inference on a Raspberry Pi Pico 2W.

---

## Overview

This repository contains `iotids`, a custom two-part machine learning library purpose-built for IoT network intrusion detection. It replaces standard Python ML dependencies (TensorFlow, NumPy, scikit-learn) with a lightweight, self-contained stack designed for constrained hardware deployment.

**Dataset:** [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) — ~2M network flow samples, 78 numeric features

**Target hardware:** Raspberry Pi Pico 2W (RP2350 · 4 MB Flash · 520 KB SRAM)

---

## Architecture

The library is split into two halves that work together:

```
CIC-IDS-2017  →  iotids Python Library  →  .tflite (<30 KB)  →  iotids C Runtime  →  Pico 2W Inference
```

### Part I — Python Training Library
Runs on university servers. Handles the full training, compression, and export pipeline.

| Module | Responsibility |
|--------|---------------|
| `core/` | Tensor primitives, math ops, dtype casting |
| `data/` | CSV ingestion, RobustScaler, outlier clipping, batching |
| `nn/` | DNN stack — Dense, BatchNorm, Dropout, Adam, FocalLoss |
| `forest/` | Random Forest with FedAvg-compatible weight API |
| `federated/` | FedAvg client/server simulation, IID & non-IID partitioning |
| `prune/` | Magnitude and structured pruning with gradual decay scheduler |
| `quantize/` | INT8 quantization, calibration, and `.tflite` export |
| `metrics/` | Accuracy, precision, recall, F1, AUC, threshold sweep |

### Part II — C Inference Runtime
Bare-metal C/C++ on the Pico 2W. Loads the `.tflite` model and runs inference in real time.

| Module | Responsibility |
|--------|---------------|
| `inference/` | TFLM model runner, static memory arena, minimal op resolver |
| `preprocessing/` | On-device RobustScaler using params exported from Python |
| `federated/` | TCP weight receiver, flash persistence with wear-leveling |
| `network/` | CYW43 Wi-Fi init, TCP client with retry/backoff |
| `utils/` | Ring buffer for feature vectors, UART debug logging |

---

## Results

### Centralized Baselines

| Model | Accuracy | Precision | Recall | F1 | AUC | FPR | FNR | Throughput |
|-------|----------|-----------|--------|----|-----|-----|-----|------------|
| DNN (float32, 77 features) | 99.42% | 99.14% | 96.93% | — | 0.9994 | — | — | — |
| DNN (float32, 24 features, threshold=0.35) | 96.0% | — | 96.22% | — | — | 3.95% | 3.78% | — |
| DNN (TFLite quantized) | ~95.7% | — | — | — | — | — | — | — |
| Random Forest | 99.83% | 99.35% | 99.63% | 0.9949 | 0.9999 | 0.13% | 0.37% | 147,373 samples/sec |

**DNN architecture:** Dense 128→64→32→1 · BatchNorm + Dropout (0.4) · Focal Loss (α=0.70, γ=1.8) · Adam (lr=2e-4) · RobustScaler · 70/15/15 split · oversampling on train only

**DNN confusion matrix (test set, threshold=0.35):** 215,538 TN · 8,867 FP · 1,696 FN · 43,163 TP

**Random Forest confusion matrix (test set):** 298,818 TN · 388 FP · 219 FN · 59,593 TP

> The 24-feature DNN uses the currently available `clean_dataset.csv`. The 77-feature DNN was trained on the full preprocessed feature set. The optimal decision threshold shifts from the default 0.5 to **0.35** post-compression — threshold must be recalibrated after each compression stage.
>
> TFLite conversion uses float16 quantization (safer than strict INT8 on TF 2.16 with BatchNorm layers). Full INT8 QAT pipeline is in progress to recover float32-level performance.

### Federated Learning — IID

20 clients · 20 communication rounds · FedAvg aggregation

| Metric | Federated | Centralized | Gap |
|--------|-----------|-------------|-----|
| Accuracy | 99.79% | 99.83% | **0.04%** |
| F1-Score | 0.9937 | 0.9948 | 0.0011 |
| Precision | 99.23% | — | — |
| Recall | 99.51% | — | — |

Status: **CONVERGED** — federated accuracy tracks centralized baseline within 0.04% across all 20 rounds.

### Federated Learning — Non-IID

> In progress — Dirichlet partitioning (α = 0.1 → 0.5) experiments pending.

### On-Device Inference (Pico 2W)

> Latency benchmarking in progress.

The quantized model fits comfortably at ~30 KB — well within the Pico 2W flash budget.

---

## Novel Contributions

- **Federated learning framework** — FedAvg simulation across IID and non-IID client partitions
- **Differential privacy** — privacy-preserving training integrated into the federated stack
- **Adversarial robustness** — hardening against adversarial network flow inputs
- **End-to-end compression pipeline** — prune → quantize → export → deploy, targeting <30 KB on bare metal

This builds on prior baselines: Random Forest (~99.2%) and 1D CNN on Raspberry Pi.

---

## Repository Structure

```
IoT-496-C-ML-Framework/
├── iotids/                  # Python training library
│   ├── core/
│   ├── data/
│   ├── nn/
│   ├── forest/
│   ├── federated/
│   ├── prune/
│   ├── quantize/
│   ├── metrics/
│   └── utils/
├── iotids-runtime/          # C inference runtime
│   ├── src/
│   │   ├── inference/
│   │   ├── federated/
│   │   ├── network/
│   │   ├── preprocessing/
│   │   └── utils/
│   ├── model/               # Auto-generated .tflite header
│   └── tests/
└── README.md
```

---

## Deployment Notes

- The Pico 2W never runs TensorFlow. It receives only a `.tflite` file embedded as a C header via `xxd -i`.
- Model weights live in flash; activations use a static SRAM arena (target < 200 KB).
- Wi-Fi (CYW43 driver) and inference run on separate RP2350 cores to avoid contention.
- The TFLM op resolver registers only `FullyConnected` + quantize/dequantize ops, cutting binary footprint by more than half vs. `AllOpsResolver`.
- Optimal classification threshold must be recalibrated after each compression stage (float32 → pruned → INT8).

---

## Research Context

Presented at the **Penn State Undergraduate Research Conference — April 6, 2026**.
