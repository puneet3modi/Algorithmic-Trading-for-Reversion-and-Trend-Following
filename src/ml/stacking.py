from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class StackingParams:
    rf_n_estimators: int = 300
    rf_max_depth: int | None = 6
    gb_n_estimators: int = 250
    gb_learning_rate: float = 0.05
    meta_C: float = 1.0
    seed: int = 42


def fit_predict_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: StackingParams = StackingParams(),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns:
    p_test: predicted probability of class 1 for X_test
    info: diagnostics on train fit (AUCs etc.)
    """
    # Base learners
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=params.seed)
    rf = RandomForestClassifier(
        n_estimators=params.rf_n_estimators,
        max_depth=params.rf_max_depth,
        random_state=params.seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=params.gb_n_estimators,
        learning_rate=params.gb_learning_rate,
        random_state=params.seed,
    )

    # Fit bases
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    p_lr_tr = lr.predict_proba(X_train)[:, 1]
    p_rf_tr = rf.predict_proba(X_train)[:, 1]
    p_gb_tr = gb.predict_proba(X_train)[:, 1]

    Z_train = np.column_stack([p_lr_tr, p_rf_tr, p_gb_tr])

    meta = LogisticRegression(C=params.meta_C, max_iter=2000, solver="lbfgs", random_state=params.seed)
    meta.fit(Z_train, y_train)

    # Predict on test
    p_lr_te = lr.predict_proba(X_test)[:, 1]
    p_rf_te = rf.predict_proba(X_test)[:, 1]
    p_gb_te = gb.predict_proba(X_test)[:, 1]
    Z_test = np.column_stack([p_lr_te, p_rf_te, p_gb_te])

    p_test = meta.predict_proba(Z_test)[:, 1]

    info = {
        "auc_lr_train": roc_auc_score(y_train, p_lr_tr),
        "auc_rf_train": roc_auc_score(y_train, p_rf_tr),
        "auc_gb_train": roc_auc_score(y_train, p_gb_tr),
        "auc_meta_train": roc_auc_score(y_train, meta.predict_proba(Z_train)[:, 1]),
    }
    return p_test, info
