"""
iotids.boosting.node
====================
Compact, serializable primitives for gradient-boosted tree nodes.

v2 change: added left_child / right_child fields so trees can use
compact sequential storage instead of heap indexing. Heap indexing
wastes ~60% of node slots when branches terminate early (a depth-5
tree with early stops occupies heap index 89 but has only 37 real nodes).

Backward compatible: left_child/right_child default to -1, and
from_dict() reads them if present, ignores them if absent (v1 files).
"""

from __future__ import annotations

import struct
from typing import Optional


_NODE_STRUCT_FMT  = "!iff?"
_NODE_STRUCT_SIZE = struct.calcsize(_NODE_STRUCT_FMT)


class Node:
    """
    A single node in a BoostingTree.

    Split nodes : feature_idx >= 0, is_leaf=False
                  left_child / right_child = indices into _nodes list
    Leaf nodes  : feature_idx = -1, is_leaf=True
                  left_child = right_child = -1
    """

    __slots__ = (
        "feature_idx",
        "threshold",
        "leaf_value",
        "gain",
        "is_leaf",
        "left_child",
        "right_child",
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
        left_child: int = -1,
        right_child: int = -1,
    ) -> None:
        self.is_leaf       = is_leaf
        self.feature_idx   = int(feature_idx)
        self.threshold     = float(threshold)
        self.leaf_value    = float(leaf_value)
        self.gain          = float(gain)
        self._sample_count = int(sample_count)
        self._sum_hessian  = float(sum_hessian)
        self.left_child    = int(left_child)
        self.right_child   = int(right_child)

    @classmethod
    def make_split(cls, feature_idx, threshold, gain,
                   sample_count=0, sum_hessian=0.0) -> "Node":
        return cls(is_leaf=False, feature_idx=feature_idx,
                   threshold=threshold, gain=gain,
                   sample_count=sample_count, sum_hessian=sum_hessian)

    @classmethod
    def make_leaf(cls, leaf_value, sample_count=0, sum_hessian=0.0) -> "Node":
        return cls(is_leaf=True, feature_idx=-1, threshold=0.0,
                   leaf_value=leaf_value,
                   sample_count=sample_count, sum_hessian=sum_hessian)

    @classmethod
    def make_empty(cls) -> "Node":
        return cls(is_leaf=True, leaf_value=0.0)

    def to_dict(self) -> dict:
        return {
            "is_leaf":      self.is_leaf,
            "feature_idx":  self.feature_idx,
            "threshold":    float(self.threshold),
            "leaf_value":   float(self.leaf_value),
            "gain":         float(self.gain),
            "sample_count": self._sample_count,
            "sum_hessian":  float(self._sum_hessian),
            "left_child":   self.left_child,
            "right_child":  self.right_child,
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
            left_child=int(d.get("left_child", -1)),
            right_child=int(d.get("right_child", -1)),
        )

    def to_bytes(self) -> bytes:
        return struct.pack(_NODE_STRUCT_FMT,
            self.feature_idx, self.threshold,
            self.leaf_value, self.is_leaf)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> "Node":
        feature_idx, threshold, leaf_value, is_leaf = struct.unpack_from(
            _NODE_STRUCT_FMT, data, offset)
        return cls(is_leaf=bool(is_leaf), feature_idx=int(feature_idx),
                   threshold=float(threshold), leaf_value=float(leaf_value))

    def is_split(self) -> bool:
        return not self.is_leaf

    def route(self, feature_value: float) -> str:
        if self.is_leaf:
            raise ValueError("route() called on a leaf node")
        return "left" if feature_value <= self.threshold else "right"

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"LeafNode(value={self.leaf_value:.6f}, n={self._sample_count})"
        return (f"SplitNode(feat={self.feature_idx}, thresh={self.threshold:.6f}, "
                f"gain={self.gain:.6f}, n={self._sample_count})")


NODE_STRUCT_SIZE: int = _NODE_STRUCT_SIZE