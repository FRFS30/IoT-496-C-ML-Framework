"""
iotids.boosting
===============
XGBoost gradient-boosted trees baseline for the iotids federated IDS framework.

Exports the three entry points used by the rest of the library:

    XGBoostClassifier  -- full federated-ready classifier (primary interface)
    GradientBooster    -- core boosting engine (used internally and for testing)
    save_xgb           -- save a trained classifier to binary .iotids format
    load_xgb           -- load a classifier checkpoint
    load_weights       -- fast leaf-only load for FedAvg weight inspection
    model_info         -- inspect a checkpoint file without full loading

Typical usage
-------------
    from iotids.boosting import XGBoostClassifier, save_xgb, load_xgb

    model = XGBoostClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    metrics = model.evaluate(X_test, y_test)
    print(metrics)

    save_xgb(model, "checkpoints/xgb_baseline.iotids")

Federated usage
---------------
    # Server broadcasts round_seed; each client calls:
    client_model.local_train(X_local, y_local, seed=round_seed)

    # Server aggregates leaf values:
    all_weights = [c.model.get_weights() for c in clients]
    avg_weights = fedavg_aggregate(all_weights, client_sizes)
    for c in clients:
        c.model.set_weights(avg_weights)
"""

from .gradient_booster import GradientBooster
from .node import Node
from .serializer import load_weights, load_xgb, model_info, save_xgb
from .tree import BoostingTree
from .xgboost_classifier import XGBoostClassifier

__all__ = [
    "Node",
    "BoostingTree",
    "GradientBooster",
    "XGBoostClassifier",
    "save_xgb",
    "load_xgb",
    "load_weights",
    "model_info",
]
