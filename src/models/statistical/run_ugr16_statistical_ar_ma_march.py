#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
Cribado March de las familias AR y MA.

Protocolo autoritativo:
TFM-STAT-PROTOCOL-001 / v1.0 / FROZEN
SHA-256:
031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699

Implementa:
- March 5 min, split cronológico 512/220.
- AR(p), p={1,2,3,6,12,24}, AutoReg, hold_back=24, trend="c".
- MA(q), q={1,2,3,6,12}, ARIMA(0,0,q), trend="c".
- Política R3: un fit inicial, parámetros congelados, rolling-origin,
  actualización con observaciones reales ya disponibles, sin refit.
- Forecasts H1/H3/H6/H12 = 5/15/30/60 min desde el mismo origen.
- Métricas y Score normalizado frente a persistencia.
- Cobertura obligatoria del 100 % y mismos timestamps objetivo.
- Regla de empate práctico del 1 %.
- Shortlist March: máximo 3 candidatos por familia.
- Seeds ARMA: exactamente top-2 p x top-2 q.
- Límite de recursos: 30 min por configuración.
- Un único retry para MA si no converge, sin cambiar orden/datos/modelo.

NO usa June.
NO modifica el protocolo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


CAMPAIGN_ID = "UGR16-STATISTICAL-MODELS-001"
RUN_ID = "UGR16-STATISTICAL-AR-MA-MARCH-001"

EXPECTED_PROTOCOL_SHA256 = (
    "031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699"
)
EXPECTED_MARCH_5MIN_SHA256 = (
    "fd6daea6007f1b411617a5c67a489fa33c2efda35fdd13616cc66a40134fa1b1"
)

EXPECTED_DIAGNOSTIC_ID = "UGR16-STATISTICAL-MARCH-DIAGNOSTICS-001"
EXPECTED_DIAGNOSTIC_STATUS = "PASS"

TARGET = "bitrate_bps"
TRAIN_FRACTION = 0.70
EXPECTED_TOTAL = 732
EXPECTED_TRAIN = 512
EXPECTED_VALIDATION = 220
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {h: EXPECTED_VALIDATION - (h - 1) for h in HORIZONS}

AR_ORDERS = (1, 2, 3, 6, 12, 24)
MA_ORDERS = (1, 2, 3, 6, 12)
AR_HOLD_BACK = 24

PRACTICAL_TIE_REL = 0.01
SHORTLIST_SIZE = 3
ARMA_SEED_SIZE = 2

RESOURCE_LIMIT_SECONDS = 30 * 60
MA_RETRY_MAXITER = 1000

STATUS_PASS = "PASS"
STATUS_INVALID_NUMERICAL = "INVALID_NUMERICAL"
STATUS_INVALID_COVERAGE = "INVALID_COVERAGE"
STATUS_INVALID_RESOURCE_LIMIT = "INVALID_RESOURCE_LIMIT"


class CandidateTimeoutError(TimeoutError):
    pass


@contextmanager
def candidate_timeout(seconds: int):
    """Linux/WSL hard guardrail for one candidate."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def handler(signum, frame):
        raise CandidateTimeoutError(
            f"Candidate exceeded resource limit of {seconds} seconds."
        )

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cribado March AR+MA para UGR16-STATISTICAL-MODELS-001."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ugr16/march_week3_prepared_5min.parquet"),
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=Path(
            "docs/project_governance/"
            "003_statistical_models_protocol_2026-08-14.md"
        ),
    )
    parser.add_argument(
        "--diagnostic-manifest",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_march_diagnostics_manifest.json"
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/predictions"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(
            "results/figures/ugr16_statistical_ar_ma_march"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="ugr16_statistical_ar_ma_march",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists and --overwrite is not active: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_protocol(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FROZEN protocol: {path}")
    observed = sha256_file(path)
    if observed != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Protocol SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Observed: {observed}"
        )
    return observed


def validate_diagnostic_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic manifest: {path}")

    observed_hash = sha256_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    checks = {
        "campaign_id": data.get("campaign_id") == CAMPAIGN_ID,
        "diagnostic_id": data.get("diagnostic_id") == EXPECTED_DIAGNOSTIC_ID,
        "status": data.get("status") == EXPECTED_DIAGNOSTIC_STATUS,
        "protocol_sha": (
            data.get("protocol", {}).get("sha256")
            == EXPECTED_PROTOCOL_SHA256
        ),
        "input_sha": (
            data.get("input", {}).get("sha256")
            == EXPECTED_MARCH_5MIN_SHA256
        ),
        "n_train": data.get("split", {}).get("n_train") == EXPECTED_TRAIN,
        "n_validation": (
            data.get("split", {}).get("n_validation")
            == EXPECTED_VALIDATION
        ),
        "diagnostics_no_validation": (
            data.get("split", {}).get("diagnostics_use_validation") is False
        ),
        "june_unused": (
            data.get("blindness", {}).get("june_used") is False
        ),
        "models_not_fitted": (
            data.get("blindness", {}).get("models_fitted") is False
        ),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(
            "Diagnostic manifest precheck failed: " + ", ".join(failed)
        )

    return data, observed_hash


def load_series(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing March input: {path}")

    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_MARCH_5MIN_SHA256:
        raise RuntimeError(
            "March input SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MARCH_5MIN_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        if frame.index.name == "timestamp":
            frame = frame.reset_index()
        else:
            raise ValueError("Input has no timestamp column/index.")

    if TARGET not in frame.columns:
        raise ValueError(f"Input has no target column: {TARGET}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(
            f"Unexpected March row count: {len(frame)} != {EXPECTED_TOTAL}"
        )
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")
    if frame[TARGET].isna().any():
        raise ValueError("NaN target values found.")
    y = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite target values found.")
    if (y < 0).any():
        raise ValueError("Negative bitrate values found.")

    deltas = frame["timestamp"].diff().dropna()
    if not (deltas == EXPECTED_FREQUENCY).all():
        raise ValueError("March series is not continuous at 5-minute cadence.")

    return frame, observed_hash


def split_series(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(frame) * TRAIN_FRACTION)
    if n_train != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Unexpected train size: {n_train} != {EXPECTED_TRAIN}"
        )

    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:].copy()

    if len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError(
            "Unexpected validation size: "
            f"{len(validation)} != {EXPECTED_VALIDATION}"
        )

    if train["timestamp"].max() >= validation["timestamp"].min():
        raise RuntimeError("Train/validation chronology violation.")

    return train, validation


def mase_scale_from_train(train_y: np.ndarray) -> float:
    scale = float(np.mean(np.abs(np.diff(train_y))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid MASE scale: {scale}")
    return scale


def expected_target_timestamp(
    validation: pd.DataFrame,
    origin_k: int,
    horizon: int,
) -> pd.Timestamp | None:
    target_pos = origin_k + horizon - 1
    if target_pos >= len(validation):
        return None
    return pd.Timestamp(validation.iloc[target_pos]["timestamp"])


def build_prediction_row(
    family: str,
    config: str,
    p: int | None,
    q: int | None,
    origin_timestamp: pd.Timestamp,
    target_timestamp: pd.Timestamp,
    horizon: int,
    y_true: float,
    y_pred: float,
    persistence_pred: float,
) -> dict[str, Any]:
    residual = float(y_true - y_pred)
    return {
        "family": family,
        "config": config,
        "p": p,
        "q": q,
        "origin_timestamp": origin_timestamp,
        "target_timestamp": target_timestamp,
        "horizon_steps": horizon,
        "horizon_minutes": HORIZON_MINUTES[horizon],
        "y_true": float(y_true),
        "y_pred": float(y_pred),
        "persistence_pred": float(persistence_pred),
        "residual": residual,
        "abs_error": abs(residual),
    }


def forecast_ar_fixed(
    params: pd.Series,
    p: int,
    actual_history: list[float],
    steps: int,
) -> np.ndarray:
    # AutoReg names lag coefficients after the endogenous variable
    # (e.g. "bitrate_bps.L1"), so positional access is deliberately used.
    # With trend="c", no exogenous terms and contiguous lags 1..p:
    # params[0] = const, params[1:1+p] = AR coefficients.
    param_values = np.asarray(params, dtype=float)
    if len(param_values) != p + 1:
        raise RuntimeError(
            f"Unexpected AutoReg parameter count: {len(param_values)} != {p + 1}"
        )

    const = float(param_values[0])
    coefficients = param_values[1 : p + 1]

    working = list(actual_history)
    forecasts: list[float] = []

    for _ in range(steps):
        value = const
        for lag in range(1, p + 1):
            value += float(coefficients[lag - 1]) * working[-lag]
        forecasts.append(float(value))
        working.append(float(value))

    return np.asarray(forecasts, dtype=float)


def evaluate_ar_candidate(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    p: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    family = "AR"
    config = f"AR({p})"
    total_start = time.perf_counter()
    warning_messages: list[str] = []

    try:
        with candidate_timeout(RESOURCE_LIMIT_SECONDS):
            fit_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = AutoReg(
                    train[TARGET].astype(float),
                    lags=p,
                    trend="c",
                    hold_back=AR_HOLD_BACK,
                    old_names=False,
                ).fit()
                warning_messages.extend(
                    f"{item.category.__name__}: {item.message}"
                    for item in caught
                )
            fit_time = time.perf_counter() - fit_start

            params = result.params
            if not np.isfinite(np.asarray(params, dtype=float)).all():
                raise FloatingPointError("Non-finite AR parameters.")

            rows: list[dict[str, Any]] = []
            inference_start = time.perf_counter()

            train_end_index = EXPECTED_TRAIN - 1
            full_y = frame[TARGET].to_numpy(dtype=float)

            for k in range(len(validation)):
                origin_index = train_end_index + k
                if origin_index >= len(frame) - 1:
                    break

                origin_timestamp = pd.Timestamp(
                    frame.iloc[origin_index]["timestamp"]
                )
                origin_value = float(full_y[origin_index])
                remaining = len(frame) - 1 - origin_index
                steps = min(max(HORIZONS), remaining)

                history = full_y[: origin_index + 1].tolist()
                forecasts = forecast_ar_fixed(
                    params=params,
                    p=p,
                    actual_history=history,
                    steps=steps,
                )

                for horizon in HORIZONS:
                    if horizon > steps:
                        continue
                    target_index = origin_index + horizon
                    y_true = float(full_y[target_index])
                    target_timestamp = pd.Timestamp(
                        frame.iloc[target_index]["timestamp"]
                    )
                    rows.append(
                        build_prediction_row(
                            family=family,
                            config=config,
                            p=p,
                            q=None,
                            origin_timestamp=origin_timestamp,
                            target_timestamp=target_timestamp,
                            horizon=horizon,
                            y_true=y_true,
                            y_pred=float(forecasts[horizon - 1]),
                            persistence_pred=origin_value,
                        )
                    )

            inference_time = time.perf_counter() - inference_start
            elapsed = time.perf_counter() - total_start

            predictions = pd.DataFrame(rows)
            return (
                {
                    "family": family,
                    "config": config,
                    "p": p,
                    "q": None,
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "n_params": int(len(result.params)),
                    "converged": True,
                    "fit_attempts": 1,
                    "warnings": " | ".join(warning_messages),
                    "training_time_sec": float(fit_time),
                    "inference_time_sec": float(inference_time),
                    "total_time_sec": float(elapsed),
                    "status": STATUS_PASS,
                },
                predictions,
            )

    except CandidateTimeoutError as exc:
        return (
            {
                "family": family,
                "config": config,
                "p": p,
                "q": None,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": 1,
                "warnings": str(exc),
                "training_time_sec": np.nan,
                "inference_time_sec": np.nan,
                "total_time_sec": float(time.perf_counter() - total_start),
                "status": STATUS_INVALID_RESOURCE_LIMIT,
            },
            pd.DataFrame(),
        )
    except Exception as exc:
        return (
            {
                "family": family,
                "config": config,
                "p": p,
                "q": None,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": 1,
                "warnings": f"{type(exc).__name__}: {exc}",
                "training_time_sec": np.nan,
                "inference_time_sec": np.nan,
                "total_time_sec": float(time.perf_counter() - total_start),
                "status": STATUS_INVALID_NUMERICAL,
            },
            pd.DataFrame(),
        )


def fit_ma_once(
    train_series: pd.Series,
    q: int,
    retry: bool,
):
    model = ARIMA(
        train_series,
        order=(0, 0, q),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    if retry:
        return model.fit(method_kwargs={"maxiter": MA_RETRY_MAXITER})
    return model.fit()


def evaluate_ma_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    q: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    family = "MA"
    config = f"MA({q})"
    total_start = time.perf_counter()
    warning_messages: list[str] = []
    fit_attempts = 0

    train_series = pd.Series(
        train[TARGET].to_numpy(dtype=float),
        index=pd.DatetimeIndex(train["timestamp"]),
        name=TARGET,
    )
    validation_series = pd.Series(
        validation[TARGET].to_numpy(dtype=float),
        index=pd.DatetimeIndex(validation["timestamp"]),
        name=TARGET,
    )

    try:
        with candidate_timeout(RESOURCE_LIMIT_SECONDS):
            fit_start = time.perf_counter()
            result = None
            converged = False

            for retry in (False, True):
                fit_attempts += 1
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = fit_ma_once(
                        train_series=train_series,
                        q=q,
                        retry=retry,
                    )
                    warning_messages.extend(
                        f"{item.category.__name__}: {item.message}"
                        for item in caught
                    )

                converged = bool(
                    getattr(result, "mle_retvals", {}).get(
                        "converged",
                        False,
                    )
                )

                params_array = np.asarray(result.params, dtype=float)
                finite = np.isfinite(params_array).all()
                if converged and finite:
                    break

                if retry:
                    raise FloatingPointError(
                        "MA failed convergence/finite-parameter check "
                        "after the single permitted retry."
                    )

            fit_time = time.perf_counter() - fit_start
            assert result is not None

            rows: list[dict[str, Any]] = []
            current_result = result
            inference_start = time.perf_counter()

            for k in range(len(validation)):
                remaining = len(validation) - k
                steps = min(max(HORIZONS), remaining)

                origin_timestamp = (
                    pd.Timestamp(train_series.index[-1])
                    if k == 0
                    else pd.Timestamp(validation_series.index[k - 1])
                )
                origin_value = (
                    float(train_series.iloc[-1])
                    if k == 0
                    else float(validation_series.iloc[k - 1])
                )

                forecast_values = np.asarray(
                    current_result.forecast(steps=steps),
                    dtype=float,
                )
                if len(forecast_values) != steps:
                    raise RuntimeError("Unexpected MA forecast length.")
                if not np.isfinite(forecast_values).all():
                    raise FloatingPointError("Non-finite MA forecast.")

                for horizon in HORIZONS:
                    if horizon > steps:
                        continue
                    target_pos = k + horizon - 1
                    target_timestamp = pd.Timestamp(
                        validation_series.index[target_pos]
                    )
                    y_true = float(validation_series.iloc[target_pos])

                    rows.append(
                        build_prediction_row(
                            family=family,
                            config=config,
                            p=None,
                            q=q,
                            origin_timestamp=origin_timestamp,
                            target_timestamp=target_timestamp,
                            horizon=horizon,
                            y_true=y_true,
                            y_pred=float(forecast_values[horizon - 1]),
                            persistence_pred=origin_value,
                        )
                    )

                # R3: incorporate the newly observed real point for the next
                # origin, with refit=False. Do not append after the final
                # origin because no further forecast is required.
                if k < len(validation) - 1:
                    new_observation = validation_series.iloc[k : k + 1]
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        current_result = current_result.append(
                            new_observation,
                            refit=False,
                        )
                        warning_messages.extend(
                            f"{item.category.__name__}: {item.message}"
                            for item in caught
                        )

            inference_time = time.perf_counter() - inference_start
            elapsed = time.perf_counter() - total_start

            predictions = pd.DataFrame(rows)
            return (
                {
                    "family": family,
                    "config": config,
                    "p": None,
                    "q": q,
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "n_params": int(len(result.params)),
                    "converged": bool(converged),
                    "fit_attempts": int(fit_attempts),
                    "warnings": " | ".join(warning_messages),
                    "training_time_sec": float(fit_time),
                    "inference_time_sec": float(inference_time),
                    "total_time_sec": float(elapsed),
                    "status": STATUS_PASS,
                },
                predictions,
            )

    except CandidateTimeoutError as exc:
        return (
            {
                "family": family,
                "config": config,
                "p": None,
                "q": q,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": int(fit_attempts),
                "warnings": " | ".join(
                    warning_messages + [str(exc)]
                ),
                "training_time_sec": np.nan,
                "inference_time_sec": np.nan,
                "total_time_sec": float(time.perf_counter() - total_start),
                "status": STATUS_INVALID_RESOURCE_LIMIT,
            },
            pd.DataFrame(),
        )
    except Exception as exc:
        return (
            {
                "family": family,
                "config": config,
                "p": None,
                "q": q,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": int(fit_attempts),
                "warnings": " | ".join(
                    warning_messages
                    + [f"{type(exc).__name__}: {exc}"]
                ),
                "training_time_sec": np.nan,
                "inference_time_sec": np.nan,
                "total_time_sec": float(time.perf_counter() - total_start),
                "status": STATUS_INVALID_NUMERICAL,
            },
            pd.DataFrame(),
        )


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    contribution = np.zeros_like(denominator, dtype=float)
    mask = denominator != 0
    contribution[mask] = (
        2.0 * np.abs(y_true[mask] - y_pred[mask])
        / denominator[mask]
    )
    return float(100.0 * np.mean(contribution))


def compute_metrics(
    predictions: pd.DataFrame,
    mase_scale: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (family, config, horizon), group in predictions.groupby(
        ["family", "config", "horizon_steps"],
        sort=False,
    ):
        group = group.sort_values("target_timestamp").reset_index(drop=True)

        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        persistence = group["persistence_pred"].to_numpy(dtype=float)

        residual = y_true - y_pred
        abs_error = np.abs(residual)
        mae = float(np.mean(abs_error))
        persistence_mae = float(np.mean(np.abs(y_true - persistence)))

        rows.append(
            {
                "family": family,
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "n_predictions": int(len(group)),
                "expected_predictions": EXPECTED_COUNTS[int(horizon)],
                "coverage": float(
                    len(group) / EXPECTED_COUNTS[int(horizon)]
                ),
                "mae_bps": mae,
                "rmse_bps": float(np.sqrt(np.mean(residual ** 2))),
                "smape_pct": smape_percent(y_true, y_pred),
                "mase": float(mae / mase_scale),
                "persistence_mae_bps": persistence_mae,
                "skill_vs_persistence": float(
                    1.0 - mae / persistence_mae
                ),
                "mean_bias_bps": float(np.mean(residual)),
                "underprediction_pct": float(
                    100.0 * np.mean(y_true > y_pred)
                ),
                "p95_abs_error_bps": float(
                    np.percentile(abs_error, 95)
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_coverage(
    candidate_summary: dict[str, Any],
    candidate_predictions: pd.DataFrame,
) -> tuple[str, str]:
    if candidate_summary["status"] != STATUS_PASS:
        return candidate_summary["status"], ""

    if candidate_predictions.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."

    required_columns = [
        "y_true",
        "y_pred",
        "persistence_pred",
    ]
    if not np.isfinite(
        candidate_predictions[required_columns].to_numpy(dtype=float)
    ).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf in predictions."

    problems: list[str] = []

    for horizon in HORIZONS:
        subset = candidate_predictions[
            candidate_predictions["horizon_steps"] == horizon
        ].sort_values("target_timestamp")

        expected_count = EXPECTED_COUNTS[horizon]
        if len(subset) != expected_count:
            problems.append(
                f"H{horizon}: {len(subset)} != {expected_count}"
            )

        if subset["target_timestamp"].duplicated().any():
            problems.append(f"H{horizon}: duplicate target timestamps")

    if problems:
        return STATUS_INVALID_COVERAGE, " | ".join(problems)

    return STATUS_PASS, ""


def attach_scores(
    summaries: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = summaries.copy()
    output["score"] = np.nan

    for idx, row in output.iterrows():
        if row["status"] != STATUS_PASS:
            continue

        subset = metrics[
            (metrics["family"] == row["family"])
            & (metrics["config"] == row["config"])
        ]

        if set(subset["horizon_steps"].astype(int)) != set(HORIZONS):
            output.at[idx, "status"] = STATUS_INVALID_COVERAGE
            output.at[idx, "coverage_note"] = (
                "Missing one or more required horizons."
            )
            continue

        ratios = (
            subset["mae_bps"].to_numpy(dtype=float)
            / subset["persistence_mae_bps"].to_numpy(dtype=float)
        )
        output.at[idx, "score"] = float(np.mean(ratios))

    return output


def deterministic_ranking(
    family_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sequential extension of the frozen practical-tie rule:
    - identify the best remaining Score;
    - form the <=1% relative practical-tie group around that best Score;
    - inside that group order by BIC, n_params, training time;
    - assign ranks, remove the group, repeat.

    This preserves Score as the primary criterion while making the complete
    shortlist ordering reproducible.
    """
    eligible = family_table[
        family_table["status"] == STATUS_PASS
    ].copy()

    if eligible.empty:
        return eligible.assign(rank=pd.Series(dtype="Int64"))

    remaining = eligible.copy()
    ranked_parts: list[pd.DataFrame] = []
    next_rank = 1

    while not remaining.empty:
        best_score = float(remaining["score"].min())
        relative = (remaining["score"] - best_score) / best_score
        tie_mask = relative <= PRACTICAL_TIE_REL + 1e-15

        tie_group = remaining.loc[tie_mask].copy()
        tie_group = tie_group.sort_values(
            ["bic", "n_params", "training_time_sec", "config"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        tie_group["practical_tie_group_best_score"] = best_score
        tie_group["relative_to_group_best"] = (
            tie_group["score"] - best_score
        ) / best_score
        tie_group["rank"] = range(
            next_rank,
            next_rank + len(tie_group),
        )

        ranked_parts.append(tie_group)
        next_rank += len(tie_group)
        remaining = remaining.drop(index=tie_group.index)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    return ranked


def make_rankings(
    summaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranking_parts: list[pd.DataFrame] = []
    shortlist_parts: list[pd.DataFrame] = []
    arma_seed: dict[str, Any] = {}

    for family in ("AR", "MA"):
        family_table = summaries[summaries["family"] == family].copy()
        ranked = deterministic_ranking(family_table)
        ranking_parts.append(ranked)

        shortlist = ranked.head(SHORTLIST_SIZE).copy()
        shortlist["shortlisted_for_april"] = True
        shortlist_parts.append(shortlist)

        seed = ranked.head(ARMA_SEED_SIZE).copy()
        if family == "AR":
            arma_seed["top_2_p"] = [
                int(value) for value in seed["p"].tolist()
            ]
        else:
            arma_seed["top_2_q"] = [
                int(value) for value in seed["q"].tolist()
            ]

    ranking = (
        pd.concat(ranking_parts, ignore_index=True)
        if ranking_parts
        else pd.DataFrame()
    )
    shortlist = (
        pd.concat(shortlist_parts, ignore_index=True)
        if shortlist_parts
        else pd.DataFrame()
    )

    top_p = arma_seed.get("top_2_p", [])
    top_q = arma_seed.get("top_2_q", [])
    arma_seed["arma_candidates"] = [
        {"p": p, "q": q, "config": f"ARMA({p},{q})"}
        for p in top_p
        for q in top_q
    ]

    return ranking, shortlist, arma_seed


def validate_common_timestamps(
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    eligible = summaries[summaries["status"] == STATUS_PASS]

    for horizon in HORIZONS:
        reference: tuple[pd.Timestamp, ...] | None = None
        reference_name: str | None = None

        for _, row in eligible.iterrows():
            subset = predictions[
                (predictions["family"] == row["family"])
                & (predictions["config"] == row["config"])
                & (predictions["horizon_steps"] == horizon)
            ].sort_values("target_timestamp")

            timestamps = tuple(
                pd.to_datetime(subset["target_timestamp"]).tolist()
            )
            if reference is None:
                reference = timestamps
                reference_name = str(row["config"])
            elif timestamps != reference:
                raise RuntimeError(
                    "Target timestamp mismatch at "
                    f"H{horizon}: {row['config']} vs {reference_name}"
                )


def save_figure(base_path: Path, dpi: int) -> list[Path]:
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def plot_score_ranking(
    ranking: pd.DataFrame,
    family: str,
    base_path: Path,
    dpi: int,
) -> list[Path]:
    subset = ranking[ranking["family"] == family].sort_values("rank")
    plt.figure(figsize=(9, 5.5))
    plt.bar(subset["config"], subset["score"])
    plt.axhline(1.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Configuración")
    plt.ylabel("Score medio MAE / persistencia")
    plt.title(f"UGR'16 March — {family} — ranking por Score")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return save_figure(base_path, dpi)


def plot_mae_horizons(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    family: str,
    base_path: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(figsize=(10, 5.8))

    configs = (
        ranking[ranking["family"] == family]
        .sort_values("rank")["config"]
        .tolist()
    )
    for config in configs:
        subset = metrics[
            (metrics["family"] == family)
            & (metrics["config"] == config)
        ].sort_values("horizon_minutes")
        plt.plot(
            subset["horizon_minutes"],
            subset["mae_bps"] / 1e6,
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.xlabel("Horizonte (min)")
    plt.ylabel("MAE (Mbit/s)")
    plt.title(f"UGR'16 March — {family} — MAE frente a horizonte")
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base_path, dpi)


def hash_artifacts(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def main() -> int:
    args = parse_args()

    print("=" * 100)
    print("UGR'16 MARCH — CRIBADO AR + MA")
    print("=" * 100)
    print(f"Campaña:  {CAMPAIGN_ID}")
    print(f"Run:      {RUN_ID}")
    print("June:     NO")
    print("Refit:    NO — política R3")
    print()

    protocol_hash = validate_protocol(args.protocol_file)
    diagnostic, diagnostic_hash = validate_diagnostic_manifest(
        args.diagnostic_manifest
    )
    frame, input_hash = load_series(args.input)
    train, validation = split_series(frame)
    mase_scale = mase_scale_from_train(
        train[TARGET].to_numpy(dtype=float)
    )

    print("PRECHECK")
    print("-" * 100)
    print(f"Protocol SHA-256:    {protocol_hash}")
    print(f"Diagnostic manifest: {diagnostic_hash}")
    print(f"Input SHA-256:       {input_hash}")
    print(
        f"Rows: total={len(frame)} "
        f"train={len(train)} validation={len(validation)}"
    )
    print(f"MASE scale:          {mase_scale:.6f} bps")
    print()

    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    print("AR GRID")
    print("-" * 100)
    for p in AR_ORDERS:
        summary, predictions = evaluate_ar_candidate(
            frame=frame,
            train=train,
            validation=validation,
            p=p,
        )
        status, coverage_note = validate_coverage(
            summary,
            predictions,
        )
        summary["status"] = status
        summary["coverage_note"] = coverage_note
        summaries.append(summary)
        if not predictions.empty:
            prediction_frames.append(predictions)
        print(
            f"{summary['config']:10s} "
            f"status={summary['status']:24s} "
            f"fit={summary['training_time_sec']!s:>10} "
            f"infer={summary['inference_time_sec']!s:>10}"
        )

    print()
    print("MA GRID")
    print("-" * 100)
    for q in MA_ORDERS:
        summary, predictions = evaluate_ma_candidate(
            train=train,
            validation=validation,
            q=q,
        )
        status, coverage_note = validate_coverage(
            summary,
            predictions,
        )
        summary["status"] = status
        summary["coverage_note"] = coverage_note
        summaries.append(summary)
        if not predictions.empty:
            prediction_frames.append(predictions)
        print(
            f"{summary['config']:10s} "
            f"status={summary['status']:24s} "
            f"attempts={summary['fit_attempts']} "
            f"fit={summary['training_time_sec']!s:>10} "
            f"infer={summary['inference_time_sec']!s:>10}"
        )

    summary_table = pd.DataFrame(summaries)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )

    if predictions.empty:
        raise RuntimeError("No candidate produced predictions.")

    metrics = compute_metrics(predictions, mase_scale=mase_scale)
    summary_table = attach_scores(summary_table, metrics)

    # Re-check status after score attachment.
    validate_common_timestamps(predictions, summary_table)

    ranking, shortlist, arma_seed = make_rankings(summary_table)

    # Each family needs at least two valid candidates to construct ARMA seeds.
    if len(arma_seed.get("top_2_p", [])) != 2:
        raise RuntimeError("AR produced fewer than 2 valid ranked candidates.")
    if len(arma_seed.get("top_2_q", [])) != 2:
        raise RuntimeError("MA produced fewer than 2 valid ranked candidates.")
    if len(arma_seed.get("arma_candidates", [])) != 4:
        raise RuntimeError("ARMA seed must contain exactly 4 candidates.")

    outputs = {
        "candidate_summary": (
            args.metrics_dir / f"{args.prefix}_candidate_summary.csv"
        ),
        "metrics": args.metrics_dir / f"{args.prefix}_metrics.csv",
        "ranking": args.metrics_dir / f"{args.prefix}_ranking.csv",
        "shortlist": args.metrics_dir / f"{args.prefix}_shortlist.csv",
        "arma_seed": args.metrics_dir / f"{args.prefix}_arma_seed.json",
        "report": args.metrics_dir / f"{args.prefix}_report.txt",
        "manifest": args.metrics_dir / f"{args.prefix}_manifest.json",
        "predictions_parquet": (
            args.predictions_dir / f"{args.prefix}_predictions.parquet"
        ),
        "predictions_csv": (
            args.predictions_dir / f"{args.prefix}_predictions.csv"
        ),
    }
    for path in outputs.values():
        ensure_output(path, args.overwrite)

    figure_bases = {
        "ar_score": args.figures_dir / f"{args.prefix}_ar_score",
        "ma_score": args.figures_dir / f"{args.prefix}_ma_score",
        "ar_mae": args.figures_dir / f"{args.prefix}_ar_mae_horizon",
        "ma_mae": args.figures_dir / f"{args.prefix}_ma_mae_horizon",
    }
    for base in figure_bases.values():
        for suffix in (".png", ".pdf"):
            ensure_output(base.with_suffix(suffix), args.overwrite)

    summary_table.to_csv(outputs["candidate_summary"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    ranking.to_csv(outputs["ranking"], index=False)
    shortlist.to_csv(outputs["shortlist"], index=False)
    outputs["arma_seed"].write_text(
        json.dumps(arma_seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_parquet(outputs["predictions_parquet"], index=False)
    predictions.to_csv(outputs["predictions_csv"], index=False)

    figure_paths: list[Path] = []
    figure_paths += plot_score_ranking(
        ranking, "AR", figure_bases["ar_score"], args.dpi
    )
    figure_paths += plot_score_ranking(
        ranking, "MA", figure_bases["ma_score"], args.dpi
    )
    figure_paths += plot_mae_horizons(
        metrics, ranking, "AR", figure_bases["ar_mae"], args.dpi
    )
    figure_paths += plot_mae_horizons(
        metrics, ranking, "MA", figure_bases["ma_mae"], args.dpi
    )

    ar_ranked = ranking[ranking["family"] == "AR"].sort_values("rank")
    ma_ranked = ranking[ranking["family"] == "MA"].sort_values("rank")
    invalid = summary_table[summary_table["status"] != STATUS_PASS]

    report_lines = [
        "=" * 100,
        "UGR'16 MARCH — CRIBADO AR + MA",
        "=" * 100,
        f"Campaign:                  {CAMPAIGN_ID}",
        f"Run:                       {RUN_ID}",
        f"Protocol SHA-256:          {protocol_hash}",
        f"Diagnostic manifest SHA:   {diagnostic_hash}",
        f"Input SHA-256:             {input_hash}",
        "",
        "SPLIT / R3",
        "-" * 100,
        f"Total:                     {len(frame)}",
        f"Train:                     {len(train)}",
        f"Validation:                {len(validation)}",
        f"Origin_0:                  {train['timestamp'].max().isoformat()}",
        "Refit validation:          NO",
        "Observed-state update:     YES",
        "June used:                NO",
        "",
        "EXPECTED COVERAGE",
        "-" * 100,
        *[
            f"H{h}: {EXPECTED_COUNTS[h]} targets"
            for h in HORIZONS
        ],
        "",
        "AR RANKING",
        "-" * 100,
    ]
    for _, row in ar_ranked.iterrows():
        report_lines.append(
            f"rank={int(row['rank'])} "
            f"{row['config']} "
            f"Score={row['score']:.8f} "
            f"BIC={row['bic']:.6f} "
            f"status={row['status']}"
        )

    report_lines += ["", "MA RANKING", "-" * 100]
    for _, row in ma_ranked.iterrows():
        report_lines.append(
            f"rank={int(row['rank'])} "
            f"{row['config']} "
            f"Score={row['score']:.8f} "
            f"BIC={row['bic']:.6f} "
            f"status={row['status']}"
        )

    report_lines += [
        "",
        "SHORTLIST APRIL",
        "-" * 100,
    ]
    for family in ("AR", "MA"):
        configs = shortlist[
            shortlist["family"] == family
        ].sort_values("rank")["config"].tolist()
        report_lines.append(f"{family}: {configs}")

    report_lines += [
        "",
        "ARMA SEED — DETERMINISTIC 2 x 2",
        "-" * 100,
        f"top_2_p: {arma_seed['top_2_p']}",
        f"top_2_q: {arma_seed['top_2_q']}",
        "candidates:",
        *[
            f"  {item['config']}"
            for item in arma_seed["arma_candidates"]
        ],
        "",
        "INVALID CANDIDATES",
        "-" * 100,
    ]
    if invalid.empty:
        report_lines.append("NONE")
    else:
        for _, row in invalid.iterrows():
            report_lines.append(
                f"{row['config']}: {row['status']} "
                f"{row.get('coverage_note', '')}"
            )

    report_lines += [
        "",
        "GLOBAL RESULT",
        "-" * 100,
        "VALIDACIÓN GLOBAL: PASS",
        "",
        "Next authorized step:",
        "implement/evaluate the four deterministic ARMA candidates on March.",
        "",
    ]
    outputs["report"].write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    artifact_paths = [
        outputs["candidate_summary"],
        outputs["metrics"],
        outputs["ranking"],
        outputs["shortlist"],
        outputs["arma_seed"],
        outputs["report"],
        outputs["predictions_parquet"],
        outputs["predictions_csv"],
        *figure_paths,
    ]

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "name": "TFM-STAT-PROTOCOL-001",
            "version": "1.0",
            "state": "FROZEN",
            "path": str(args.protocol_file),
            "sha256": protocol_hash,
        },
        "diagnostic_prerequisite": {
            "path": str(args.diagnostic_manifest),
            "sha256": diagnostic_hash,
            "diagnostic_id": diagnostic["diagnostic_id"],
            "status": diagnostic["status"],
        },
        "input": {
            "path": str(args.input),
            "sha256": input_hash,
            "rows": len(frame),
        },
        "split": {
            "n_train": len(train),
            "n_validation": len(validation),
            "origin_0": train["timestamp"].max().isoformat(),
            "refit_validation": False,
            "observed_state_update": True,
        },
        "grids": {
            "AR": list(AR_ORDERS),
            "MA": list(MA_ORDERS),
            "AR_hold_back": AR_HOLD_BACK,
        },
        "horizons": {
            str(h): {
                "minutes": HORIZON_MINUTES[h],
                "expected_predictions": EXPECTED_COUNTS[h],
            }
            for h in HORIZONS
        },
        "selection": {
            "primary_metric": "MAE",
            "score": (
                "mean_h(MAE_model_h / MAE_persistence_h)"
            ),
            "practical_tie_relative": PRACTICAL_TIE_REL,
            "tiebreakers": [
                "BIC",
                "n_params",
                "training_time_sec",
            ],
            "shortlist_size_per_family": SHORTLIST_SIZE,
            "arma_seed_top_p": arma_seed["top_2_p"],
            "arma_seed_top_q": arma_seed["top_2_q"],
            "arma_candidates": arma_seed["arma_candidates"],
        },
        "resource_policy": {
            "limit_seconds_per_candidate": RESOURCE_LIMIT_SECONDS,
            "ma_single_retry_maxiter": MA_RETRY_MAXITER,
        },
        "mase_scale_bps": mase_scale,
        "blindness": {
            "june_used": False,
            "diagnostic_validation_used_for_tuning": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": hash_artifacts(artifact_paths),
    }

    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("RESULT")
    print("-" * 100)
    print("AR shortlist:", shortlist[
        shortlist["family"] == "AR"
    ].sort_values("rank")["config"].tolist())
    print("MA shortlist:", shortlist[
        shortlist["family"] == "MA"
    ].sort_values("rank")["config"].tolist())
    print("ARMA top-2 p:", arma_seed["top_2_p"])
    print("ARMA top-2 q:", arma_seed["top_2_q"])
    print(
        "ARMA candidates:",
        [item["config"] for item in arma_seed["arma_candidates"]],
    )
    print("VALIDACIÓN GLOBAL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
