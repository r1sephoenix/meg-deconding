from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class EvalMetrics:
    mse: float
    r2: float


@dataclass(slots=True)
class TrainedModel:
    pipeline: Pipeline
    metrics: EvalMetrics


def fit_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> TrainedModel:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(Ridge(alpha=alpha))),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = EvalMetrics(
        mse=float(mean_squared_error(y_test, preds)),
        r2=float(r2_score(y_test, preds, multioutput="uniform_average")),
    )
    return TrainedModel(pipeline=model, metrics=metrics)
