from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from finetune.surv_pred_survboard.runner import BreslowEstimator


@dataclass
class SurvivalRegressionOutput:
    train_risk: np.ndarray
    test_risk: np.ndarray
    test_survival: np.ndarray
    time_points: np.ndarray
    pca_components: int
    embedding_dim: int


def _event_ipcw_weights(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Return inverse-censoring weights for uncensored log-time regression."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    order = np.argsort(time, kind="stable")
    ordered_time = time[order]
    ordered_event = event[order]

    censor_survival_before = np.ones(len(time), dtype=float)
    current = 1.0
    for observed_time in np.unique(ordered_time):
        at_time = ordered_time == observed_time
        at_risk = int(np.sum(ordered_time >= observed_time))
        censor_survival_before[order[at_time]] = current
        censored = int(np.sum(~ordered_event[at_time]))
        if at_risk > 0:
            current *= 1.0 - censored / at_risk

    weights = np.zeros(len(time), dtype=float)
    weights[event] = 1.0 / np.clip(censor_survival_before[event], 1e-6, None)
    if not np.any(weights > 0):
        raise ValueError("Random-forest survival regression requires an observed event.")
    return weights


def fit_pca_random_forest_regressor(
    task_cfg,
    train_x: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_x: np.ndarray,
    test_time: np.ndarray,
    *,
    prefix: str,
) -> SurvivalRegressionOutput:
    """Fit train-only scaling/PCA and an IPCW random-forest regressor."""
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    if bool(getattr(task_cfg, f"{prefix}_standardize", True)):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

    requested = int(getattr(task_cfg, f"{prefix}_components", 256))
    n_components = min(requested, train_x.shape[0] - 1, train_x.shape[1])
    if n_components < 1:
        raise ValueError("Cannot fit PCA with fewer than one component.")
    pca = PCA(
        n_components=n_components,
        whiten=bool(getattr(task_cfg, f"{prefix}_whiten", False)),
        random_state=int(getattr(task_cfg, "random_seed", 42)),
    )
    train_z = pca.fit_transform(train_x)
    test_z = pca.transform(test_x)

    forest_prefix = {
        "pca_rf": "pca_rf",
        "raw_pca_rf": "raw_pca_rf",
        "scgpt_pca": "scgpt_rf",
        "bulkformer_pca": "bulkformer_rf",
    }.get(prefix, prefix)
    model = RandomForestRegressor(
        n_estimators=int(getattr(task_cfg, f"{forest_prefix}_n_estimators", 500)),
        max_depth=getattr(task_cfg, f"{forest_prefix}_max_depth", None),
        min_samples_leaf=int(
            getattr(task_cfg, f"{forest_prefix}_min_samples_leaf", 1)
        ),
        min_samples_split=int(
            getattr(task_cfg, f"{forest_prefix}_min_samples_split", 2)
        ),
        max_features=getattr(task_cfg, f"{forest_prefix}_max_features", "sqrt"),
        bootstrap=bool(getattr(task_cfg, f"{forest_prefix}_bootstrap", True)),
        n_jobs=int(getattr(task_cfg, f"{forest_prefix}_n_jobs", -1)),
        random_state=int(getattr(task_cfg, "random_seed", 42)),
    )
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=bool)
    model.fit(
        train_z,
        np.log1p(train_time),
        sample_weight=_event_ipcw_weights(train_time, train_event),
    )
    train_risk = -np.asarray(model.predict(train_z), dtype=float)
    test_risk = -np.asarray(model.predict(test_z), dtype=float)

    time_points = np.unique(train_time[train_event])
    max_test_time = float(np.max(test_time))
    if max_test_time > time_points[-1]:
        time_points = np.append(time_points, max_test_time)
    min_test_time = float(np.min(test_time))
    if min_test_time < time_points[0]:
        time_points = np.insert(time_points, 0, min_test_time)
    breslow = BreslowEstimator()
    breslow.fit(train_risk, train_time, train_event)
    test_survival = breslow.predict_survival(test_risk, time_points)

    return SurvivalRegressionOutput(
        train_risk=train_risk,
        test_risk=test_risk,
        test_survival=test_survival,
        time_points=time_points,
        pca_components=int(n_components),
        embedding_dim=int(train_x.shape[1]),
    )
