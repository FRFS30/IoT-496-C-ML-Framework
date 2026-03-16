"""
iotids.boosting.node
====================
Compact, serializable primitives for gradient-boosted tree nodes.

Design goals
------------
* No recursive object graphs — every tree is a flat array of Node objects
  indexed by position, making serialization trivial and cache access linear.
* All numeric values stored as float32 to match the rest of the iotids
  library and avoid the silent float64 bloat that pickle produces.
* Dataclass-style API with explicit to_dict / from_dict for the binary
  serializer, keeping the boosting/ serializer consistent with forest/.

Node layout
-----------
Each node occupies one slot in a pre-allocated list.  For a node at index i:
  left  child  -> 2*i + 1
  right child  -> 2*i + 2
  parent       -> (i - 1) // 2   (root is index 0)

A node is a leaf when left_child == -1 (right_child is then also -1).
Split nodes store the split feature and threshold; leaf nodes store the
raw prediction delta (before sigmoid) that gets accumulated across trees.
"""

from __future__ import annotations

import struct
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_STRUCT_FMT = "!iff?"   # network byte-order: int32, float32, float32, bool
_NODE_STRUCT_SIZE = struct.calcsize(_NODE_STRUCT_FMT)  # 10 bytes


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A single node in a BoostingTree.

    For split nodes
    ---------------
    feature_idx  : int   – column index of the splitting feature
    threshold    : float – samples with feature <= threshold go left
    gain         : float – XGBoost gain achieved by this split (diagnostic)
    is_leaf      : False

    For leaf nodes
    --------------
    feature_idx  : -1    (sentinel)
    threshold    : 0.0   (unused)
    leaf_value   : float – raw score delta accumulated across boosting rounds
    is_leaf      : True
    """

    __slots__ = (
        "feature_idx",
        "threshold",
        "leaf_value",
        "gain",
        "is_leaf",
        # coverage counters kept for feature importance computation
        "_sample_count",
        "_sum_hessian",
    )

    def __init__(
        self,
        *,
        is_leaf: bool,
        feature_idx: int = -1,
        threshold: float = 0.0,
        leaf_value: float = 0.0,
        gain: float = 0.0,
        sample_count: int = 0,
        sum_hessian: float = 0.0,
    ) -> None:
        self.is_leaf: bool = is_leaf
        self.feature_idx: int = int(feature_idx)
        self.threshold: float = float(threshold)
        self.leaf_value: float = float(leaf_value)
        self.gain: float = float(gain)
        self._sample_count: int = int(sample_count)
        self._sum_hessian: float = float(sum_hessian)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def make_split(
        cls,
        feature_idx: int,
        threshold: float,
        gain: float,
        sample_count: int = 0,
        sum_hessian: float = 0.0,
    ) -> "Node":
        return cls(
            is_leaf=False,
            feature_idx=feature_idx,
            threshold=threshold,
            gain=gain,
            sample_count=sample_count,
            sum_hessian=sum_hessian,
        )

    @classmethod
    def make_leaf(
        cls,
        leaf_value: float,
        sample_count: int = 0,
        sum_hessian: float = 0.0,
    ) -> "Node":
        return cls(
            is_leaf=True,
            feature_idx=-1,
            threshold=0.0,
            leaf_value=leaf_value,
            sample_count=sample_count,
            sum_hessian=sum_hessian,
        )

    @classmethod
    def make_empty(cls) -> "Node":
        """Placeholder for nodes in a full binary tree that are never reached."""
        return cls(is_leaf=True, leaf_value=0.0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "is_leaf":       self.is_leaf,
            "feature_idx":   self.feature_idx,
            "threshold":     float(self.threshold),
            "leaf_value":    float(self.leaf_value),
            "gain":          float(self.gain),
            "sample_count":  self._sample_count,
            "sum_hessian":   float(self._sum_hessian),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            is_leaf=bool(d["is_leaf"]),
            feature_idx=int(d["feature_idx"]),
            threshold=float(d["threshold"]),
            leaf_value=float(d["leaf_value"]),
            gain=float(d.get("gain", 0.0)),
            sample_count=int(d.get("sample_count", 0)),
            sum_hessian=float(d.get("sum_hessian", 0.0)),
        )

    def to_bytes(self) -> bytes:
        """
        Compact 10-byte binary representation for the serializer.
        Layout: feature_idx (int32) | threshold (float32) | leaf_value (float32) | is_leaf (bool->uint8)
        The bool packs into the '?' format character (1 byte, padded to align).
        """
        return struct.pack(
            _NODE_STRUCT_FMT,
            self.feature_idx,
            self.threshold,
            self.leaf_value,
            self.is_leaf,
        )

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> "Node":
        feature_idx, threshold, leaf_value, is_leaf = struct.unpack_from(
            _NODE_STRUCT_FMT, data, offset
        )
        return cls(
            is_leaf=bool(is_leaf),
            feature_idx=int(feature_idx),
            threshold=float(threshold),
            leaf_value=float(leaf_value),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_split(self) -> bool:
        return not self.is_leaf

    def route(self, feature_value: float) -> str:
        """
        Given a single feature value, return 'left' or 'right'.
        Only valid on split nodes.
        """
        if self.is_leaf:
            raise ValueError("route() called on a leaf node")
        return "left" if feature_value <= self.threshold else "right"

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"LeafNode(value={self.leaf_value:.6f}, n={self._sample_count})"
        return (
            f"SplitNode(feat={self.feature_idx}, thresh={self.threshold:.6f}, "
            f"gain={self.gain:.6f}, n={self._sample_count})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience exports used by tree.py
# ---------------------------------------------------------------------------

NODE_STRUCT_SIZE: int = _NODE_STRUCT_SIZE
