"""
iotids.boosting.serializer
==========================
Compact binary serializer for XGBoostClassifier and GradientBooster models.

Design goals
------------
* float64 -> float32 downcast on save — leaf values and thresholds are stored
  as 4-byte floats, matching forest/serializer.py and avoiding the silent
  float64 bloat of pickle.
* Checksum validation on load — detects truncated files from interrupted
  training saves (common on shared university servers with quota limits).
* Unified binary layout compatible with forest/serializer.py so a single
  model store can hold Random Forest, XGBoost, and DNN checkpoints.
* Zero third-party dependencies — uses only Python stdlib (struct, json, zlib).

File format
-----------
All values are big-endian (network byte order) for portability.

    HEADER (32 bytes)
    ─────────────────
    magic       : 8 bytes  = b"IOTIDSXG"
    version     : 2 bytes  = uint16 (current: 1)
    model_type  : 2 bytes  = uint16 (1=GradientBooster, 2=XGBoostClassifier)
    json_len    : 4 bytes  = uint32 — byte length of JSON metadata block
    data_len    : 4 bytes  = uint32 — byte length of binary leaf-value block
    reserved    : 8 bytes  = 0x00 padding
    crc32       : 4 bytes  = zlib.crc32 of everything AFTER the header

    JSON METADATA (json_len bytes)
    ──────────────────────────────
    UTF-8 JSON blob containing all hyperparameters and tree structure
    (feature indices, thresholds, node counts).  Leaf values are stored
    in the binary block instead of JSON to allow fast float32 I/O.

    BINARY LEAF BLOCK (data_len bytes)
    ────────────────────────────────────
    Flat sequence of float32 values (big-endian) storing all leaf values
    from all trees in node-index order.  A per-tree leaf count table in
    the JSON metadata maps ranges of this flat array back to individual trees.

    [tree_0_leaf_0, tree_0_leaf_1, ..., tree_1_leaf_0, ...]

Usage
-----
    from iotids.boosting.serializer import save_xgb, load_xgb

    save_xgb(model, "checkpoints/xgb_round_5.iotids")
    model = load_xgb("checkpoints/xgb_round_5.iotids")

    # Convenience: load leaf values only (fast, no full model rebuild)
    weights = load_weights("checkpoints/xgb_round_5.iotids")
"""

from __future__ import annotations

import json
import math
import os
import struct
import zlib
from typing import List, Union

from .gradient_booster import GradientBooster
from .node import Node
from .tree import BoostingTree
from .xgboost_classifier import XGBoostClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC           = b"IOTIDSXG"
_VERSION         = 1
_MODEL_BOOSTER   = 1
_MODEL_XGB       = 2

_HEADER_FMT      = "!8sHHII8sI"   # magic, version, model_type, json_len, data_len, reserved, crc32
_HEADER_SIZE     = struct.calcsize(_HEADER_FMT)   # 32 bytes

_F32_FMT         = "!f"
_F32_SIZE        = struct.calcsize(_F32_FMT)      # 4 bytes


# ---------------------------------------------------------------------------
# Float32 helpers
# ---------------------------------------------------------------------------

def _to_f32(v: float) -> float:
    """Downcast a float64 to float32 precision (round-trip through struct)."""
    return struct.unpack("!f", struct.pack("!f", v))[0]


def _pack_f32_list(values: List[float]) -> bytes:
    """Pack a list of floats as big-endian float32."""
    return struct.pack(f"!{len(values)}f", *values)


def _unpack_f32_list(data: bytes, count: int, offset: int = 0) -> List[float]:
    """Unpack count float32 values from data starting at offset."""
    return list(struct.unpack_from(f"!{count}f", data, offset))


# ---------------------------------------------------------------------------
# Internal: extract / restore leaf values
# ---------------------------------------------------------------------------

def _extract_leaves(booster: GradientBooster) -> tuple[dict, bytes]:
    """
    Separate the booster's leaf values from its structural metadata.

    Returns
    -------
    meta  : dict  – JSON-serializable dict (no leaf values)
    data  : bytes – flat float32 binary blob of all leaf values
    """
    # Build a JSON-safe meta dict from booster.to_dict(), but strip leaf values
    full_dict = booster.to_dict()

    # Collect leaf counts per tree and the flat leaf array
    leaf_counts: List[int] = []
    flat_leaves: List[float] = []

    for tree_dict in full_dict["trees"]:
        leaves_in_tree = []
        for node_dict in tree_dict["nodes"]:
            if node_dict["is_leaf"]:
                leaves_in_tree.append(_to_f32(node_dict["leaf_value"]))
                # Zero out in the meta dict; we reconstruct from binary block
                node_dict["leaf_value"] = 0.0
            else:
                # Downcast threshold to f32 for consistency
                node_dict["threshold"] = _to_f32(node_dict["threshold"])
        leaf_counts.append(len(leaves_in_tree))
        flat_leaves.extend(leaves_in_tree)

    full_dict["_leaf_counts"] = leaf_counts
    meta = full_dict
    data = _pack_f32_list(flat_leaves)
    return meta, data


def _restore_leaves(booster: GradientBooster, meta: dict, data: bytes) -> None:
    """
    Restore leaf values from the binary block into a booster built from meta.
    Called by load functions after reconstructing the tree structure.
    """
    leaf_counts = meta.get("_leaf_counts", [])
    offset = 0
    for tree, count in zip(booster._trees, leaf_counts):
        leaves = _unpack_f32_list(data, count, offset * _F32_SIZE)
        offset += count
        # Write leaves back into the tree nodes in order
        vi = 0
        for node in tree._nodes:
            if node.is_leaf:
                if vi < len(leaves):
                    node.leaf_value = float(leaves[vi])
                    vi += 1


# ---------------------------------------------------------------------------
# Internal: header pack / unpack
# ---------------------------------------------------------------------------

def _write_header(
    model_type: int,
    json_bytes: bytes,
    data_bytes: bytes,
    payload_crc: int,
) -> bytes:
    return struct.pack(
        _HEADER_FMT,
        _MAGIC,
        _VERSION,
        model_type,
        len(json_bytes),
        len(data_bytes),
        b"\x00" * 8,
        payload_crc & 0xFFFFFFFF,
    )


def _read_header(data: bytes) -> tuple[int, int, int, int, int]:
    """
    Parse the 32-byte file header.

    Returns
    -------
    version, model_type, json_len, data_len, stored_crc
    """
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"File too short: {len(data)} bytes (header requires {_HEADER_SIZE})")

    magic, version, model_type, json_len, data_len, _, stored_crc = struct.unpack(
        _HEADER_FMT, data[:_HEADER_SIZE]
    )

    if magic != _MAGIC:
        raise ValueError(
            f"Invalid magic bytes: expected {_MAGIC!r}, got {magic!r}. "
            "Is this an iotids boosting model file?"
        )
    if version != _VERSION:
        raise ValueError(f"Unsupported file version {version} (this library supports {_VERSION})")

    return version, model_type, json_len, data_len, stored_crc


# ---------------------------------------------------------------------------
# Public: save
# ---------------------------------------------------------------------------

def save_booster(model: GradientBooster, path: str) -> None:
    """
    Save a GradientBooster to a binary .iotids file.

    Parameters
    ----------
    model : GradientBooster – must be fit
    path  : str             – destination file path
    """
    if not model._is_fit:
        raise RuntimeError("Cannot save an unfit GradientBooster")

    meta, data_bytes = _extract_leaves(model)
    json_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    payload = json_bytes + data_bytes
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    header = _write_header(_MODEL_BOOSTER, json_bytes, data_bytes, crc)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(json_bytes)
        f.write(data_bytes)


def save_xgb(model: XGBoostClassifier, path: str) -> None:
    """
    Save an XGBoostClassifier to a binary .iotids file.

    Parameters
    ----------
    model : XGBoostClassifier – must be fit
    path  : str               – destination file path
    """
    if not model._is_fit or model._booster is None:
        raise RuntimeError("Cannot save an unfit XGBoostClassifier")

    meta, data_bytes = _extract_leaves(model._booster)

    # Attach XGBoostClassifier-level fields
    meta["_xgb_wrapper"] = {
        "params":             model.get_params(),
        "optimal_threshold":  _to_f32(model._optimal_threshold),
        "best_n_trees":       model._best_n_trees,
    }

    json_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    payload = json_bytes + data_bytes
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    header = _write_header(_MODEL_XGB, json_bytes, data_bytes, crc)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(json_bytes)
        f.write(data_bytes)


# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def load_booster(path: str) -> GradientBooster:
    """
    Load a GradientBooster from a binary .iotids file.

    Raises
    ------
    ValueError  : magic bytes wrong, version mismatch, or CRC32 failure
    FileNotFoundError : path does not exist
    """
    with open(path, "rb") as f:
        raw = f.read()

    _, model_type, json_len, data_len, stored_crc = _read_header(raw)

    if model_type not in (_MODEL_BOOSTER, _MODEL_XGB):
        raise ValueError(f"Expected model_type 1 or 2, got {model_type}")

    payload_start = _HEADER_SIZE
    json_bytes = raw[payload_start: payload_start + json_len]
    data_bytes = raw[payload_start + json_len: payload_start + json_len + data_len]

    # CRC check
    actual_crc = zlib.crc32(json_bytes + data_bytes) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        raise ValueError(
            f"CRC32 mismatch: stored={stored_crc:#010x}, computed={actual_crc:#010x}. "
            "File may be truncated or corrupted."
        )

    meta = json.loads(json_bytes.decode("utf-8"))

    # Strip internal keys before rebuilding
    meta.pop("_xgb_wrapper", None)
    leaf_counts = meta.pop("_leaf_counts", [])
    meta["_leaf_counts"] = leaf_counts   # put back for _restore_leaves

    booster = GradientBooster.from_dict(meta)
    _restore_leaves(booster, {"_leaf_counts": leaf_counts}, data_bytes)
    return booster


def load_xgb(path: str) -> XGBoostClassifier:
    """
    Load an XGBoostClassifier from a binary .iotids file.

    Raises
    ------
    ValueError  : magic bytes wrong, version mismatch, or CRC32 failure
    FileNotFoundError : path does not exist
    """
    with open(path, "rb") as f:
        raw = f.read()

    _, model_type, json_len, data_len, stored_crc = _read_header(raw)

    payload_start = _HEADER_SIZE
    json_bytes = raw[payload_start: payload_start + json_len]
    data_bytes = raw[payload_start + json_len: payload_start + json_len + data_len]

    actual_crc = zlib.crc32(json_bytes + data_bytes) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        raise ValueError(
            f"CRC32 mismatch: stored={stored_crc:#010x}, computed={actual_crc:#010x}. "
            "File may be truncated or corrupted."
        )

    meta = json.loads(json_bytes.decode("utf-8"))

    xgb_wrapper = meta.pop("_xgb_wrapper", {})
    leaf_counts  = meta.pop("_leaf_counts", [])
    meta["_leaf_counts"] = leaf_counts

    # Reconstruct GradientBooster
    booster = GradientBooster.from_dict(meta)
    _restore_leaves(booster, {"_leaf_counts": leaf_counts}, data_bytes)

    # Reconstruct XGBoostClassifier shell
    params = xgb_wrapper.get("params", {})
    clf = XGBoostClassifier(**params)
    clf._booster           = booster
    clf._is_fit            = booster._is_fit
    clf._optimal_threshold = float(xgb_wrapper.get("optimal_threshold", 0.5))
    clf._best_n_trees      = int(xgb_wrapper.get("best_n_trees", len(booster._trees)))

    return clf


def load_weights(path: str) -> List[List[float]]:
    """
    Load only the leaf values from a saved model.

    Significantly faster than load_xgb() when you only need to inspect or
    broadcast weights for federated averaging — no tree structure is rebuilt.

    Returns
    -------
    List[List[float]] — one leaf list per tree, same structure as
    XGBoostClassifier.get_weights()
    """
    with open(path, "rb") as f:
        raw = f.read()

    _, model_type, json_len, data_len, stored_crc = _read_header(raw)

    payload_start = _HEADER_SIZE
    json_bytes = raw[payload_start: payload_start + json_len]
    data_bytes = raw[payload_start + json_len: payload_start + json_len + data_len]

    actual_crc = zlib.crc32(json_bytes + data_bytes) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        raise ValueError(
            f"CRC32 mismatch: stored={stored_crc:#010x}, computed={actual_crc:#010x}."
        )

    meta         = json.loads(json_bytes.decode("utf-8"))
    leaf_counts  = meta.get("_leaf_counts", [])

    result: List[List[float]] = []
    offset = 0
    for count in leaf_counts:
        leaves = _unpack_f32_list(data_bytes, count, offset * _F32_SIZE)
        result.append(list(leaves))
        offset += count

    return result


# ---------------------------------------------------------------------------
# Public: file info
# ---------------------------------------------------------------------------

def model_info(path: str) -> dict:
    """
    Return a summary dict about a saved model file without fully loading it.

    Useful for quick inspection of checkpoints in the model store.

    Returns keys: path, size_bytes, version, model_type_str, n_trees,
                  n_total_leaves, n_features, is_fit
    """
    with open(path, "rb") as f:
        raw = f.read()

    _, model_type, json_len, data_len, _ = _read_header(raw)

    payload_start = _HEADER_SIZE
    json_bytes    = raw[payload_start: payload_start + json_len]
    meta          = json.loads(json_bytes.decode("utf-8"))

    leaf_counts    = meta.get("_leaf_counts", [])
    model_type_str = {_MODEL_BOOSTER: "GradientBooster",
                      _MODEL_XGB:     "XGBoostClassifier"}.get(model_type, "unknown")

    return {
        "path":            path,
        "size_bytes":      len(raw),
        "version":         _VERSION,
        "model_type":      model_type_str,
        "n_trees":         len(leaf_counts),
        "n_total_leaves":  sum(leaf_counts),
        "n_features":      meta.get("n_features", -1),
        "is_fit":          meta.get("is_fit", False),
    }
