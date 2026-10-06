#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
June blind external evaluation — FINAL FROZEN statistical representatives.

THIS RUNNER PERFORMS EVALUATION, NOT MODEL SELECTION.

Hard authorities:
- 008_statistical_final_representatives_freeze_2026-08-16.md
- April final representatives artifact
- June frozen split manifest
- June 5-minute prepared series

Frozen representatives:
- AR(24)
- MA(12)
- ARMA(24,3)
- ARIMA(6,1,12)
- VAR(5)

SARIMA:
- NOT_EVALUABLE_WITHIN_FROZEN_RESOURCE_BUDGET
- no June execution

June:
- N = 2004
- initial train = 1402
- blind test = 602
- boundary = 2016-06-17T21:50:00
- train: timestamp < boundary
- test:  timestamp >= boundary

R3:
- fit once on initial June 70 %
- parameter refit = NO
- observed history/state update = YES
- same frozen configuration at H1/H3/H6/H12

If a frozen representative fails numerically or by resource limit in June:
- DO NOT substitute it.
- DO NOT reopen April.
- Preserve the failure as external-evaluation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

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
from statsmodels.tsa.api import VAR


CAMPAIGN_ID = "UGR16-STATISTICAL-MODELS-001"
RUN_ID = "UGR16-STATISTICAL-JUNE-BLIND-001"
RUNNER_VERSION = "1.0"

EXPECTED_FREEZE_008_SHA256 = (
    "cc98c4f7b74e2d0ce6515ac1a989bef2b20de0a921f63a5d0346591f2456ad1a"
)
EXPECTED_APRIL_FINAL_REPS_SHA256 = (
    "39e0d0887c51df151176aa467f8b3c8ea53cc305405554b2006e327d3839e567"
)
EXPECTED_APRIL_MANIFEST_SHA256 = (
    "a375004739240860d21448c3cb62309f72d63e7e9acddf9e7b59cbee280faf39"
)
EXPECTED_JUNE_SPLIT_MANIFEST_SHA256 = (
    "a641f0bf011c705cfb8fb25c9042a198461918b01dddb284644a309b783adb39"
)
EXPECTED_JUNE_5MIN_SHA256 = (
    "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3"
)

TARGET = "bitrate_bps"
VAR_COLUMNS = ("bitrate_bps", "packet_rate_pps", "flow_rate_fps")
TARGET_INDEX = 0

EXPECTED_TOTAL = 2004
EXPECTED_TRAIN = 1402
EXPECTED_TEST = 602
BOUNDARY = pd.Timestamp("2016-06-17T21:50:00")
EXPECTED_START = pd.Timestamp("2016-06-13T01:00:00")
EXPECTED_END = pd.Timestamp("2016-06-19T23:55:00")
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

RESOURCE_LIMIT_SECONDS = 30 * 60
TERMINATE_GRACE_SECONDS = 10
RETRY_MAXITER = 1000
AR_HOLD_BACK = 24

STATUS_PASS = "PASS"
STATUS_INVALID_NUMERICAL = "INVALID_NUMERICAL"
STATUS_INVALID_STABILITY = "INVALID_STABILITY"
STATUS_INVALID_COVERAGE = "INVALID_COVERAGE"
STATUS_INVALID_RESOURCE_LIMIT = "INVALID_RESOURCE_LIMIT"
STATUS_TECHNICAL_ERROR = "TECHNICAL_ERROR"
STATUS_BLOCKED = "BLOCKED_FROZEN_REPRESENTATIVE_FAILURE"

FROZEN_CANDIDATES = [
    {"family": "AR", "config": "AR(24)", "p": 24},
    {"family": "MA", "config": "MA(12)", "p": 0, "d": 0, "q": 12},
    {"family": "ARMA", "config": "ARMA(24,3)", "p": 24, "d": 0, "q": 3},
    {"family": "ARIMA", "config": "ARIMA(6,1,12)", "p": 6, "d": 1, "q": 12},
    {"family": "VAR", "config": "VAR(5)", "lag": 5},
]
FROZEN_CONFIGS = [c["config"] for c in FROZEN_CANDIDATES]
FAMILIES = [c["family"] for c in FROZEN_CANDIDATES]

SARIMA_STATUS = "NOT_EVALUABLE_WITHIN_FROZEN_RESOURCE_BUDGET"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def log(message: str = "") -> None:
    print(message, flush=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="June blind external evaluation for frozen statistical representatives."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ugr16/june_week3_prepared_5min.parquet"),
    )
    parser.add_argument(
        "--freeze-008",
        type=Path,
        default=Path(
            "docs/project_governance/"
            "008_statistical_final_representatives_freeze_2026-08-16.md"
        ),
    )
    parser.add_argument(
        "--april-final-reps",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_april_selection_final_representatives.csv"
        ),
    )
    parser.add_argument(
        "--april-manifest",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_april_selection_manifest.json"
        ),
    )
    parser.add_argument(
        "--june-split-manifest",
        type=Path,
        default=Path("results/metrics/ugr16_final_split_manifest.json"),
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
        default=Path("results/figures/ugr16_statistical_june_blind"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/ugr16_statistical_june_blind"),
    )
    parser.add_argument(
        "--prefix",
        default="ugr16_statistical_june_blind",
    )
    parser.add_argument("--dpi", type=int, default=180)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")

    parser.add_argument("--preflight-only", action="store_true")

    # Internal worker mode.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--runner-sha", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input-sha", default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")

    observed = sha256_file(path)

    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )

    return observed


def current_runner_sha() -> str:
    return sha256_file(Path(__file__).resolve())


def validate_frozen_representatives(path: Path) -> str:
    observed_hash = validate_hash(
        path,
        EXPECTED_APRIL_FINAL_REPS_SHA256,
        "April final representatives",
    )

    frame = pd.read_csv(path)

    if not {"family", "config"}.issubset(frame.columns):
        raise RuntimeError(
            "April final representatives lacks family/config columns."
        )

    observed_pairs = list(
        zip(
            frame["family"].astype(str).tolist(),
            frame["config"].astype(str).tolist(),
        )
    )

    expected_pairs = [
        (candidate["family"], candidate["config"])
        for candidate in FROZEN_CANDIDATES
    ]

    if set(observed_pairs) != set(expected_pairs):
        raise RuntimeError(
            "Frozen representative mismatch.\n"
            f"Expected: {expected_pairs}\n"
            f"Observed: {observed_pairs}"
        )

    if len(observed_pairs) != len(expected_pairs):
        raise RuntimeError(
            "Unexpected number of April final representatives."
        )

    return observed_hash


def validate_april_manifest(path: Path) -> str:
    observed_hash = validate_hash(
        path,
        EXPECTED_APRIL_MANIFEST_SHA256,
        "April selection manifest",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("April manifest campaign mismatch.")

    if payload.get("status") != "PASS":
        raise RuntimeError("April selection is not PASS.")

    if payload.get("blindness", {}).get("june_used") is not False:
        raise RuntimeError(
            "April manifest indicates June was already used."
        )

    return observed_hash


def validate_june_split_manifest(path: Path) -> str:
    # The exact frozen artifact is the authority; the runner additionally
    # validates the split directly against the June input.
    return validate_hash(
        path,
        EXPECTED_JUNE_SPLIT_MANIFEST_SHA256,
        "June split manifest",
    )


def load_june_input(path: Path) -> tuple[pd.DataFrame, str]:
    observed_hash = validate_hash(
        path,
        EXPECTED_JUNE_5MIN_SHA256,
        "June 5-minute prepared input",
    )

    frame = pd.read_parquet(path).copy()

    if "timestamp" not in frame.columns:
        if frame.index.name == "timestamp":
            frame = frame.reset_index()
        else:
            raise ValueError("June input has no timestamp column/index.")

    required = {"timestamp", *VAR_COLUMNS}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"June input missing required columns: {sorted(missing)}"
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="raise",
    )

    for column in VAR_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(
            f"June rows {len(frame)} != {EXPECTED_TOTAL}"
        )

    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate June timestamps.")

    if frame["timestamp"].iloc[0] != EXPECTED_START:
        raise ValueError(
            f"Unexpected June start: {frame['timestamp'].iloc[0]}"
        )

    if frame["timestamp"].iloc[-1] != EXPECTED_END:
        raise ValueError(
            f"Unexpected June end: {frame['timestamp'].iloc[-1]}"
        )

    deltas = frame["timestamp"].diff().dropna()

    if not (deltas == EXPECTED_FREQUENCY).all():
        raise ValueError("June series is not exactly 5-minute continuous.")

    values = frame[list(VAR_COLUMNS)].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite June values.")

    if (values < 0).any():
        raise ValueError("Negative June traffic values.")

    return frame, observed_hash


def split_june(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[
        frame["timestamp"] < BOUNDARY
    ].copy()

    test = frame.loc[
        frame["timestamp"] >= BOUNDARY
    ].copy()

    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"June train {len(train)} != {EXPECTED_TRAIN}"
        )

    if len(test) != EXPECTED_TEST:
        raise RuntimeError(
            f"June blind test {len(test)} != {EXPECTED_TEST}"
        )

    if train["timestamp"].iloc[-1] != pd.Timestamp("2016-06-17T21:45:00"):
        raise RuntimeError(
            "Unexpected last June training timestamp."
        )

    if test["timestamp"].iloc[0] != BOUNDARY:
        raise RuntimeError(
            "Unexpected first June blind-test timestamp."
        )

    if train["timestamp"].max() >= test["timestamp"].min():
        raise RuntimeError("June chronology violation.")

    return train.reset_index(drop=True), test.reset_index(drop=True)


def mase_scale(train_y: np.ndarray) -> float:
    scale = float(
        np.mean(
            np.abs(
                np.diff(train_y)
            )
        )
    )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Invalid June MASE scale: {scale}"
        )

    return scale


def prediction_row(
    candidate: dict[str, Any],
    origin_ts: pd.Timestamp,
    target_ts: pd.Timestamp,
    h: int,
    y_true: float,
    y_pred: float,
    persistence: float,
) -> dict[str, Any]:
    residual = y_true - y_pred

    return {
        "family": candidate["family"],
        "config": candidate["config"],
        "origin_timestamp": origin_ts,
        "target_timestamp": target_ts,
        "horizon_steps": h,
        "horizon_minutes": HORIZON_MINUTES[h],
        "y_true": float(y_true),
        "y_pred": float(y_pred),
        "persistence_pred": float(persistence),
        "residual": float(residual),
        "abs_error": float(abs(residual)),
    }


def validate_coverage(
    predictions: pd.DataFrame,
) -> tuple[str, str]:
    if predictions.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."

    numeric = predictions[
        ["y_true", "y_pred", "persistence_pred"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf predictions."

    problems = []

    for horizon in HORIZONS:
        subset = predictions[
            predictions["horizon_steps"] == horizon
        ].sort_values("target_timestamp")

        expected = EXPECTED_COUNTS[horizon]

        if len(subset) != expected:
            problems.append(
                f"H{horizon}: {len(subset)} != {expected}"
            )

        if subset["target_timestamp"].duplicated().any():
            problems.append(
                f"H{horizon}: duplicate target timestamps"
            )

    if problems:
        return STATUS_INVALID_COVERAGE, " | ".join(problems)

    return STATUS_PASS, ""


def fit_ar(
    train_y: pd.Series,
    p: int,
) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        result = AutoReg(
            train_y,
            lags=p,
            trend="c",
            old_names=False,
            hold_back=AR_HOLD_BACK,
        ).fit()

    params = np.asarray(
        result.params,
        dtype=float,
    )

    detail = {
        "attempts": 1,
        "warnings": [
            f"{item.category.__name__}: {item.message}"
            for item in caught
        ],
        "converged": True,
        "finite_params": bool(
            np.isfinite(params).all()
        ),
        "training_time_sec": float(
            time.perf_counter() - start
        ),
    }

    return result, detail


def evaluate_ar(
    result: Any,
    candidate: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    p = int(candidate["p"])
    params = np.asarray(
        result.params,
        dtype=float,
    )

    if len(params) != p + 1:
        raise RuntimeError(
            f"Unexpected AutoReg parameter count for p={p}."
        )

    intercept = float(params[0])
    phi = params[1:].copy()

    history = train[TARGET].to_numpy(dtype=float).tolist()
    test_y = test[TARGET].to_numpy(dtype=float)

    rows = []
    start = time.perf_counter()

    for k in range(len(test)):
        origin_ts = (
            pd.Timestamp(train["timestamp"].iloc[-1])
            if k == 0
            else pd.Timestamp(test["timestamp"].iloc[k - 1])
        )

        persistence = float(history[-1])
        local = history.copy()
        forecast = []

        for _ in range(max(HORIZONS)):
            lag_values = np.array(
                local[-p:][::-1],
                dtype=float,
            )

            y_hat = intercept + float(
                np.dot(
                    phi,
                    lag_values,
                )
            )

            forecast.append(y_hat)
            local.append(y_hat)

        for horizon in HORIZONS:
            target_pos = k + horizon - 1

            if target_pos >= len(test):
                continue

            rows.append(
                prediction_row(
                    candidate,
                    origin_ts,
                    pd.Timestamp(
                        test["timestamp"].iloc[target_pos]
                    ),
                    horizon,
                    float(test_y[target_pos]),
                    float(forecast[horizon - 1]),
                    persistence,
                )
            )

        # R3: only already revealed actual value is appended.
        if k < len(test) - 1:
            history.append(
                float(test_y[k])
            )

    return pd.DataFrame(rows), float(
        time.perf_counter() - start
    )


def fit_arima_family(
    train_series: pd.Series,
    candidate: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    order = (
        int(candidate["p"]),
        int(candidate["d"]),
        int(candidate["q"]),
    )

    trend = "c" if order[1] == 0 else "n"

    attempt_details = []
    total_time = 0.0
    final_result = None

    for maxiter in (None, RETRY_MAXITER):
        start = time.perf_counter()

        detail = {
            "maxiter": maxiter,
            "warnings": [],
            "exception": None,
            "converged": False,
            "finite_params": False,
        }

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model = ARIMA(
                    train_series,
                    order=order,
                    trend=trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )

                result = (
                    model.fit()
                    if maxiter is None
                    else model.fit(
                        method_kwargs={
                            "maxiter": maxiter,
                        }
                    )
                )

                detail["warnings"] = [
                    f"{item.category.__name__}: {item.message}"
                    for item in caught
                ]

            mle_retvals = getattr(
                result,
                "mle_retvals",
                {},
            )

            detail["converged"] = bool(
                mle_retvals.get(
                    "converged",
                    True,
                )
            )

            detail["finite_params"] = bool(
                np.isfinite(
                    np.asarray(
                        result.params,
                        dtype=float,
                    )
                ).all()
            )

            if (
                detail["converged"]
                and detail["finite_params"]
            ):
                final_result = result

        except Exception as exc:
            detail["exception"] = (
                f"{type(exc).__name__}: {exc}"
            )

        elapsed = float(
            time.perf_counter() - start
        )

        detail["elapsed_sec"] = elapsed
        total_time += elapsed
        attempt_details.append(detail)

        if final_result is not None:
            break

    return final_result, {
        "attempts": len(attempt_details),
        "attempt_details": attempt_details,
        "warnings": [
            warning
            for detail in attempt_details
            for warning in detail["warnings"]
        ],
        "converged": final_result is not None,
        "finite_params": final_result is not None,
        "training_time_sec": total_time,
        "trend": trend,
    }


def evaluate_state_space(
    result: Any,
    candidate: dict[str, Any],
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[pd.DataFrame, float]:
    current = result
    rows = []
    start = time.perf_counter()

    for k in range(len(test_series)):
        remaining = len(test_series) - k
        steps = min(
            max(HORIZONS),
            remaining,
        )

        origin_ts = (
            pd.Timestamp(train_series.index[-1])
            if k == 0
            else pd.Timestamp(test_series.index[k - 1])
        )

        persistence = (
            float(train_series.iloc[-1])
            if k == 0
            else float(test_series.iloc[k - 1])
        )

        forecast = np.asarray(
            current.forecast(
                steps=steps
            ),
            dtype=float,
        )

        if (
            len(forecast) != steps
            or not np.isfinite(forecast).all()
        ):
            raise FloatingPointError(
                "Invalid state-space forecast."
            )

        for horizon in HORIZONS:
            if horizon > steps:
                continue

            target_pos = k + horizon - 1

            rows.append(
                prediction_row(
                    candidate,
                    origin_ts,
                    pd.Timestamp(
                        test_series.index[target_pos]
                    ),
                    horizon,
                    float(
                        test_series.iloc[target_pos]
                    ),
                    float(
                        forecast[horizon - 1]
                    ),
                    persistence,
                )
            )

        # R3 state update, NO parameter refit.
        if k < len(test_series) - 1:
            current = current.append(
                test_series.iloc[k:k + 1],
                refit=False,
            )

    return pd.DataFrame(rows), float(
        time.perf_counter() - start
    )


def build_var_training(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    levels = train.set_index(
        "timestamp"
    )[list(VAR_COLUMNS)].astype(float)

    diff = levels.diff().dropna()

    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=0)

    if (
        not np.isfinite(mean.to_numpy(dtype=float)).all()
        or not np.isfinite(std.to_numpy(dtype=float)).all()
        or (std <= 0).any()
    ):
        raise ValueError(
            "Invalid June VAR scaler."
        )

    standardized = (
        diff - mean
    ) / std

    if not np.isfinite(
        standardized.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite standardized June VAR training."
        )

    scaler = {
        "representation": "first_differences",
        "fit_scope": "june_training_only",
        "ddof": 0,
        "columns": list(VAR_COLUMNS),
        "mean": {
            column: float(mean[column])
            for column in VAR_COLUMNS
        },
        "std": {
            column: float(std[column])
            for column in VAR_COLUMNS
        },
        "n_transformed_train": int(len(diff)),
    }

    return standardized, scaler


def fit_var(
    train: pd.DataFrame,
    lag: int,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    train_z, scaler = build_var_training(train)

    start = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        result = VAR(
            train_z
        ).fit(
            maxlags=lag,
            ic=None,
            trend="c",
        )

    elapsed = float(
        time.perf_counter() - start
    )

    params = np.asarray(
        result.params,
        dtype=float,
    )

    roots = np.asarray(
        result.roots,
        dtype=complex,
    )

    detail = {
        "attempts": 1,
        "warnings": [
            f"{item.category.__name__}: {item.message}"
            for item in caught
        ],
        "converged": True,
        "finite_params": bool(
            np.isfinite(params).all()
        ),
        "stable": bool(
            result.is_stable(
                verbose=False
            )
        ),
        "min_abs_root": (
            float(
                np.min(
                    np.abs(roots)
                )
            )
            if roots.size
            else np.nan
        ),
        "training_time_sec": elapsed,
    }

    return result, detail, train_z, scaler


def evaluate_var(
    result: Any,
    candidate: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_z: pd.DataFrame,
    scaler: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    lag = int(candidate["lag"])

    history_z = train_z.to_numpy(
        dtype=float
    ).copy()

    train_levels = train[
        list(VAR_COLUMNS)
    ].to_numpy(dtype=float)

    test_levels = test[
        list(VAR_COLUMNS)
    ].to_numpy(dtype=float)

    mean = np.array(
        [
            scaler["mean"][column]
            for column in VAR_COLUMNS
        ],
        dtype=float,
    )

    std = np.array(
        [
            scaler["std"][column]
            for column in VAR_COLUMNS
        ],
        dtype=float,
    )

    rows = []
    start = time.perf_counter()

    for k in range(len(test)):
        remaining = len(test) - k
        steps = min(
            max(HORIZONS),
            remaining,
        )

        origin_ts = (
            pd.Timestamp(train["timestamp"].iloc[-1])
            if k == 0
            else pd.Timestamp(test["timestamp"].iloc[k - 1])
        )

        origin_level = (
            train_levels[-1]
            if k == 0
            else test_levels[k - 1]
        )

        forecast_z = np.asarray(
            result.forecast(
                history_z[-lag:],
                steps=steps,
            ),
            dtype=float,
        )

        if (
            forecast_z.shape
            != (steps, len(VAR_COLUMNS))
            or not np.isfinite(forecast_z).all()
        ):
            raise FloatingPointError(
                "Invalid VAR forecast."
            )

        forecast_diff = (
            forecast_z * std + mean
        )

        bitrate_levels = (
            origin_level[TARGET_INDEX]
            + np.cumsum(
                forecast_diff[
                    :,
                    TARGET_INDEX,
                ]
            )
        )

        for horizon in HORIZONS:
            if horizon > steps:
                continue

            target_pos = k + horizon - 1

            rows.append(
                prediction_row(
                    candidate,
                    origin_ts,
                    pd.Timestamp(
                        test["timestamp"].iloc[target_pos]
                    ),
                    horizon,
                    float(
                        test_levels[
                            target_pos,
                            TARGET_INDEX,
                        ]
                    ),
                    float(
                        bitrate_levels[
                            horizon - 1
                        ]
                    ),
                    float(
                        origin_level[
                            TARGET_INDEX
                        ]
                    ),
                )
            )

        # R3: reveal next real vector and append its first difference.
        if k < len(test) - 1:
            previous_real = (
                train_levels[-1]
                if k == 0
                else test_levels[k - 1]
            )

            current_real = test_levels[k]
            observed_diff = (
                current_real
                - previous_real
            )

            observed_z = (
                observed_diff
                - mean
            ) / std

            if not np.isfinite(
                observed_z
            ).all():
                raise FloatingPointError(
                    "Invalid observed VAR R3 update."
                )

            history_z = np.vstack(
                [
                    history_z,
                    observed_z,
                ]
            )

    return pd.DataFrame(rows), float(
        time.perf_counter() - start
    )


def worker_evaluate(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, test = split_june(frame)

    family = candidate["family"]
    status = STATUS_PASS
    coverage_note = ""
    result = None
    fit_detail: dict[str, Any] = {}
    predictions = pd.DataFrame()
    inference_time = np.nan
    extra: dict[str, Any] = {}

    total_start = time.perf_counter()

    try:
        if family == "AR":
            train_y = pd.Series(
                train[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(
                    train["timestamp"].to_numpy(),
                    freq="5min",
                ),
                name=TARGET,
            )

            result, fit_detail = fit_ar(
                train_y,
                int(candidate["p"]),
            )

            if not fit_detail["finite_params"]:
                status = STATUS_INVALID_NUMERICAL
            else:
                predictions, inference_time = evaluate_ar(
                    result,
                    candidate,
                    train,
                    test,
                )

        elif family in ("MA", "ARMA", "ARIMA"):
            train_series = pd.Series(
                train[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(
                    train["timestamp"].to_numpy(),
                    freq="5min",
                ),
                name=TARGET,
            )

            test_series = pd.Series(
                test[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(
                    test["timestamp"].to_numpy(),
                    freq="5min",
                ),
                name=TARGET,
            )

            result, fit_detail = fit_arima_family(
                train_series,
                candidate,
            )

            if result is None:
                status = STATUS_INVALID_NUMERICAL
            else:
                predictions, inference_time = evaluate_state_space(
                    result,
                    candidate,
                    train_series,
                    test_series,
                )

        elif family == "VAR":
            result, fit_detail, train_z, scaler = fit_var(
                train,
                int(candidate["lag"]),
            )

            extra["var_scaler"] = scaler

            if not fit_detail["finite_params"]:
                status = STATUS_INVALID_NUMERICAL

            elif not fit_detail["stable"]:
                status = STATUS_INVALID_STABILITY

            else:
                predictions, inference_time = evaluate_var(
                    result,
                    candidate,
                    train,
                    test,
                    train_z,
                    scaler,
                )

        else:
            raise ValueError(
                f"Unknown frozen family: {family}"
            )

        if status == STATUS_PASS:
            status, coverage_note = validate_coverage(
                predictions
            )

    except MemoryError as exc:
        status = STATUS_INVALID_RESOURCE_LIMIT
        coverage_note = (
            f"MemoryError: {exc}"
        )

    except Exception as exc:
        status = STATUS_INVALID_NUMERICAL
        coverage_note = (
            f"{type(exc).__name__}: {exc}"
        )

    aic = np.nan
    bic = np.nan
    n_params = np.nan

    if result is not None:
        try:
            aic = float(result.aic)
        except Exception:
            pass

        try:
            bic = float(result.bic)
        except Exception:
            pass

        try:
            n_params = int(
                np.asarray(
                    result.params
                ).size
            )
        except Exception:
            pass

    summary = {
        "family": family,
        "config": candidate["config"],
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
        "converged": bool(
            fit_detail.get(
                "converged",
                False,
            )
        ),
        "finite_params": bool(
            fit_detail.get(
                "finite_params",
                False,
            )
        ),
        "stable": (
            bool(
                fit_detail.get(
                    "stable"
                )
            )
            if family == "VAR"
            else True
        ),
        "min_abs_root": (
            fit_detail.get(
                "min_abs_root",
                np.nan,
            )
            if family == "VAR"
            else np.nan
        ),
        "attempts": int(
            fit_detail.get(
                "attempts",
                0,
            )
        ),
        "warnings": " | ".join(
            fit_detail.get(
                "warnings",
                [],
            )
        ),
        "attempt_details": fit_detail.get(
            "attempt_details",
            [],
        ),
        "training_time_sec": fit_detail.get(
            "training_time_sec",
            np.nan,
        ),
        "inference_time_sec": inference_time,
        "total_time_sec": float(
            time.perf_counter()
            - total_start
        ),
        "status": status,
        "coverage_note": coverage_note,
        **extra,
    }

    return summary, predictions


def worker_main(
    args: argparse.Namespace,
) -> int:
    if (
        not args.candidate_json
        or args.worker_dir is None
        or not args.runner_sha
        or not args.input_sha
    ):
        raise ValueError(
            "Internal worker arguments missing."
        )

    if current_runner_sha() != args.runner_sha:
        raise RuntimeError(
            "Worker runner SHA mismatch."
        )

    frame, input_sha = load_june_input(
        args.input
    )

    if input_sha != args.input_sha:
        raise RuntimeError(
            "Worker June input SHA mismatch."
        )

    candidate = json.loads(
        args.candidate_json
    )

    if candidate not in FROZEN_CANDIDATES:
        raise RuntimeError(
            f"Worker candidate is not frozen: {candidate}"
        )

    args.worker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary, predictions = worker_evaluate(
        frame,
        candidate,
    )

    prediction_meta = None

    if summary["status"] == STATUS_PASS:
        prediction_path = (
            args.worker_dir
            / "predictions.parquet"
        )

        tmp = prediction_path.with_name(
            prediction_path.name
            + f".tmp.{os.getpid()}"
        )

        predictions.to_parquet(
            tmp,
            index=False,
        )

        os.replace(
            tmp,
            prediction_path,
        )

        prediction_meta = {
            "path": str(prediction_path),
            "sha256": sha256_file(
                prediction_path
            ),
            "rows": int(
                len(predictions)
            ),
        }

    payload = {
        "completed": True,
        "run_id": RUN_ID,
        "runner_sha256": args.runner_sha,
        "input_sha256": input_sha,
        "candidate": candidate,
        "summary": summary,
        "predictions": prediction_meta,
        "finished_at": now_iso(),
    }

    atomic_write_json(
        args.worker_dir / "result.json",
        payload,
    )

    log(
        f"WORKER DONE {candidate['config']} "
        f"status={summary['status']} "
        f"fit={summary['training_time_sec']} "
        f"infer={summary['inference_time_sec']}"
    )

    return 0


def safe_slug(
    candidate: dict[str, Any],
) -> str:
    return (
        candidate["config"]
        .lower()
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
    )


def terminate_group(
    process: subprocess.Popen[Any],
) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGTERM,
        )
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass

    try:
        process.wait(
            timeout=TERMINATE_GRACE_SECONDS
        )
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def timeout_payload(
    candidate: dict[str, Any],
    runner_sha: str,
    input_sha: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "completed": True,
        "run_id": RUN_ID,
        "runner_sha256": runner_sha,
        "input_sha256": input_sha,
        "candidate": candidate,
        "summary": {
            "family": candidate["family"],
            "config": candidate["config"],
            "aic": np.nan,
            "bic": np.nan,
            "n_params": np.nan,
            "converged": False,
            "finite_params": False,
            "stable": (
                False
                if candidate["family"] == "VAR"
                else True
            ),
            "min_abs_root": np.nan,
            "attempts": 0,
            "warnings": (
                "Parent hard timeout at "
                f"{RESOURCE_LIMIT_SECONDS}s."
            ),
            "attempt_details": [],
            "training_time_sec": np.nan,
            "inference_time_sec": np.nan,
            "total_time_sec": elapsed,
            "status": STATUS_INVALID_RESOURCE_LIMIT,
            "coverage_note": "",
        },
        "predictions": None,
        "finished_at": now_iso(),
    }


def validate_checkpoint(
    payload: dict[str, Any],
    candidate: dict[str, Any],
    runner_sha: str,
    input_sha: str,
    worker_dir: Path,
) -> dict[str, Any]:
    if payload.get("completed") is not True:
        raise RuntimeError(
            "Incomplete June checkpoint."
        )

    if payload.get("run_id") != RUN_ID:
        raise RuntimeError(
            "June checkpoint run mismatch."
        )

    if payload.get("runner_sha256") != runner_sha:
        raise RuntimeError(
            "June checkpoint runner mismatch."
        )

    if payload.get("input_sha256") != input_sha:
        raise RuntimeError(
            "June checkpoint input mismatch."
        )

    if payload.get("candidate") != candidate:
        raise RuntimeError(
            "June checkpoint frozen candidate mismatch."
        )

    if payload["summary"]["status"] == STATUS_PASS:
        meta = payload.get(
            "predictions"
        )

        if not isinstance(
            meta,
            dict,
        ):
            raise RuntimeError(
                "PASS checkpoint lacks predictions."
            )

        prediction_path = Path(
            meta["path"]
        )

        if not prediction_path.is_file():
            prediction_path = (
                worker_dir
                / "predictions.parquet"
            )

        if not prediction_path.is_file():
            raise RuntimeError(
                "Missing checkpoint predictions."
            )

        if sha256_file(
            prediction_path
        ) != meta["sha256"]:
            raise RuntimeError(
                "June checkpoint prediction SHA mismatch."
            )

        payload["predictions"]["path"] = str(
            prediction_path
        )

    return payload


def run_candidate(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    runner_sha: str,
    input_sha: str,
) -> tuple[str, dict[str, Any]]:
    worker_dir = (
        args.checkpoint_dir
        / safe_slug(candidate)
    )

    worker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in (
        "result.json",
        "predictions.parquet",
    ):
        path = worker_dir / name

        if path.exists():
            path.unlink()

    log_path = worker_dir / "worker.log"

    if log_path.exists():
        rotated = worker_dir / (
            "worker.previous."
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".log"
        )

        os.replace(
            log_path,
            rotated,
        )

    command = [
        sys.executable,
        "-u",
        str(
            Path(__file__).resolve()
        ),
        "--worker",
        "--input",
        str(args.input),
        "--candidate-json",
        json.dumps(
            candidate,
            separators=(",", ":"),
        ),
        "--worker-dir",
        str(worker_dir),
        "--runner-sha",
        runner_sha,
        "--input-sha",
        input_sha,
    ]

    start = time.perf_counter()

    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        handle.write(
            f"[PARENT] {now_iso()} "
            f"START {candidate['config']}\n"
        )

        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        try:
            return_code = process.wait(
                timeout=RESOURCE_LIMIT_SECONDS
            )

        except subprocess.TimeoutExpired:
            terminate_group(
                process
            )

            elapsed = float(
                time.perf_counter()
                - start
            )

            payload = timeout_payload(
                candidate,
                runner_sha,
                input_sha,
                elapsed,
            )

            atomic_write_json(
                worker_dir / "result.json",
                payload,
            )

            handle.write(
                f"[PARENT] HARD TIMEOUT "
                f"{elapsed:.3f}s\n"
            )

            handle.flush()

            return "COMPLETED", payload

    result_path = (
        worker_dir
        / "result.json"
    )

    if (
        return_code != 0
        or not result_path.is_file()
    ):
        return "TECHNICAL_ERROR", {
            "status": STATUS_TECHNICAL_ERROR,
            "candidate": candidate,
            "return_code": return_code,
            "worker_log": str(log_path),
        }

    payload = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    payload = validate_checkpoint(
        payload,
        candidate,
        runner_sha,
        input_sha,
        worker_dir,
    )

    return "COMPLETED", payload


def smape_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    contribution = np.zeros_like(
        denominator,
        dtype=float,
    )

    mask = denominator != 0

    contribution[mask] = (
        2.0
        * np.abs(
            y_true[mask]
            - y_pred[mask]
        )
        / denominator[mask]
    )

    return float(
        100.0
        * np.mean(
            contribution
        )
    )


def compute_metrics(
    predictions: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    rows = []

    for (
        family,
        config,
        horizon,
    ), group in predictions.groupby(
        [
            "family",
            "config",
            "horizon_steps",
        ],
        sort=False,
    ):
        group = group.sort_values(
            "target_timestamp"
        )

        y_true = group[
            "y_true"
        ].to_numpy(dtype=float)

        y_pred = group[
            "y_pred"
        ].to_numpy(dtype=float)

        persistence = group[
            "persistence_pred"
        ].to_numpy(dtype=float)

        residual = (
            y_true
            - y_pred
        )

        abs_error = np.abs(
            residual
        )

        mae = float(
            np.mean(
                abs_error
            )
        )

        persistence_mae = float(
            np.mean(
                np.abs(
                    y_true
                    - persistence
                )
            )
        )

        rows.append(
            {
                "family": family,
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[
                    int(horizon)
                ],
                "n_predictions": int(
                    len(group)
                ),
                "expected_predictions": EXPECTED_COUNTS[
                    int(horizon)
                ],
                "coverage": float(
                    len(group)
                    / EXPECTED_COUNTS[
                        int(horizon)
                    ]
                ),
                "mae_bps": mae,
                "rmse_bps": float(
                    np.sqrt(
                        np.mean(
                            residual ** 2
                        )
                    )
                ),
                "smape_pct": smape_percent(
                    y_true,
                    y_pred,
                ),
                "mase": float(
                    mae / scale
                ),
                "persistence_mae_bps": persistence_mae,
                "skill_vs_persistence": float(
                    1.0
                    - mae
                    / persistence_mae
                ),
                "mean_bias_bps": float(
                    np.mean(
                        residual
                    )
                ),
                "underprediction_pct": float(
                    100.0
                    * np.mean(
                        y_true
                        > y_pred
                    )
                ),
                "p95_abs_error_bps": float(
                    np.percentile(
                        abs_error,
                        95,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def attach_descriptive_score(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()
    output["score"] = np.nan

    for index, row in output.iterrows():
        if row["status"] != STATUS_PASS:
            continue

        subset = metrics[
            metrics["config"]
            == row["config"]
        ]

        if set(
            subset[
                "horizon_steps"
            ].astype(int)
        ) != set(HORIZONS):
            output.at[
                index,
                "status",
            ] = STATUS_INVALID_COVERAGE

            output.at[
                index,
                "coverage_note",
            ] = (
                "Missing required June horizon."
            )

            continue

        ratios = (
            subset[
                "mae_bps"
            ].to_numpy(dtype=float)
            / subset[
                "persistence_mae_bps"
            ].to_numpy(dtype=float)
        )

        output.at[
            index,
            "score",
        ] = float(
            np.mean(
                ratios
            )
        )

    return output


def validate_common_timestamps(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    eligible = summary[
        summary["status"]
        == STATUS_PASS
    ]

    for horizon in HORIZONS:
        reference = None
        reference_config = None

        for _, row in eligible.iterrows():
            subset = predictions[
                (
                    predictions["config"]
                    == row["config"]
                )
                & (
                    predictions["horizon_steps"]
                    == horizon
                )
            ].sort_values(
                "target_timestamp"
            )

            timestamps = tuple(
                pd.to_datetime(
                    subset[
                        "target_timestamp"
                    ]
                ).tolist()
            )

            if reference is None:
                reference = timestamps
                reference_config = row["config"]

            elif timestamps != reference:
                raise RuntimeError(
                    f"June target timestamp mismatch H{horizon}: "
                    f"{row['config']} vs {reference_config}"
                )


def build_worst_intervals(
    predictions: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    return (
        predictions
        .sort_values(
            "abs_error",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def save_figure(
    base: Path,
    dpi: int,
) -> list[Path]:
    base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = base.with_suffix(
        ".png"
    )

    pdf = base.with_suffix(
        ".pdf"
    )

    plt.tight_layout()

    plt.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close()

    return [png, pdf]


def plot_mae(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(
        figsize=(10, 6)
    )

    for config, subset in metrics.groupby(
        "config",
        sort=False,
    ):
        subset = subset.sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            subset["mae_bps"] / 1e6,
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "MAE (Mbit/s)"
    )
    plt.title(
        "UGR'16 June blind — representantes estadísticos — MAE"
    )
    plt.xticks(
        [5, 15, 30, 60]
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_skill(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(
        figsize=(10, 6)
    )

    for config, subset in metrics.groupby(
        "config",
        sort=False,
    ):
        subset = subset.sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            100.0
            * subset[
                "skill_vs_persistence"
            ],
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "Skill frente a persistencia (%)"
    )
    plt.title(
        "UGR'16 June blind — skill frente a persistencia"
    )
    plt.xticks(
        [5, 15, 30, 60]
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_heatmap(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    pivot = (
        metrics.pivot(
            index="config",
            columns="horizon_minutes",
            values="mae_bps",
        )
        / 1e6
    )

    pivot = pivot.reindex(
        columns=[5, 15, 30, 60]
    )

    # Preserve frozen-model order.
    pivot = pivot.reindex(
        index=FROZEN_CONFIGS
    )

    plt.figure(
        figsize=(8.7, 5.4)
    )

    image = plt.imshow(
        pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="MAE (Mbit/s)",
    )

    plt.xticks(
        range(
            len(pivot.columns)
        ),
        [
            str(value)
            for value
            in pivot.columns
        ],
    )

    plt.yticks(
        range(
            len(pivot.index)
        ),
        pivot.index,
    )

    for i in range(
        len(pivot.index)
    ):
        for j in range(
            len(pivot.columns)
        ):
            value = pivot.iloc[
                i,
                j,
            ]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "Representante congelado"
    )
    plt.title(
        "UGR'16 June blind — MAE: modelo × horizonte"
    )

    return save_figure(
        base,
        dpi,
    )


def plot_score(
    summary: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    eligible = (
        summary[
            summary["status"]
            == STATUS_PASS
        ]
        .sort_values(
            "score"
        )
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.bar(
        eligible["config"],
        eligible["score"],
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.ylabel(
        "Score medio MAE / persistencia"
    )
    plt.xlabel(
        "Representante congelado"
    )
    plt.title(
        "UGR'16 June blind — Score descriptivo"
    )
    plt.xticks(
        rotation=25,
        ha="right",
    )
    plt.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_bias(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(
        figsize=(10, 6)
    )

    for config, subset in metrics.groupby(
        "config",
        sort=False,
    ):
        subset = subset.sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            subset["mean_bias_bps"] / 1e6,
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "Sesgo medio observado − predicho (Mbit/s)"
    )
    plt.title(
        "UGR'16 June blind — sesgo por horizonte"
    )
    plt.xticks(
        [5, 15, 30, 60]
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_p95(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(
        figsize=(10, 6)
    )

    for config, subset in metrics.groupby(
        "config",
        sort=False,
    ):
        subset = subset.sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            subset["p95_abs_error_bps"] / 1e6,
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "P95 error absoluto (Mbit/s)"
    )
    plt.title(
        "UGR'16 June blind — P95 del error absoluto"
    )
    plt.xticks(
        [5, 15, 30, 60]
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_underprediction(
    metrics: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(
        figsize=(10, 6)
    )

    for config, subset in metrics.groupby(
        "config",
        sort=False,
    ):
        subset = subset.sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            subset["underprediction_pct"],
            marker="o",
            linewidth=1.2,
            label=config,
        )

    plt.axhline(
        50.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Horizonte (min)"
    )
    plt.ylabel(
        "Infrapredicción (%)"
    )
    plt.title(
        "UGR'16 June blind — frecuencia de infrapredicción"
    )
    plt.xticks(
        [5, 15, 30, 60]
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def plot_observed_vs_predicted_h1(
    predictions: pd.DataFrame,
    config: str,
    base: Path,
    dpi: int,
) -> list[Path]:
    subset = predictions[
        (
            predictions["config"]
            == config
        )
        & (
            predictions["horizon_steps"]
            == 1
        )
    ].sort_values(
        "target_timestamp"
    )

    plt.figure(
        figsize=(13, 5.8)
    )

    plt.plot(
        pd.to_datetime(
            subset["target_timestamp"]
        ),
        subset["y_true"] / 1e6,
        label="Observado",
    )

    plt.plot(
        pd.to_datetime(
            subset["target_timestamp"]
        ),
        subset["y_pred"] / 1e6,
        label="Predicho",
    )

    plt.xlabel(
        "Tiempo"
    )
    plt.ylabel(
        "Bitrate (Mbit/s)"
    )
    plt.title(
        f"UGR'16 June blind — {config} — observado vs predicho — H1"
    )
    plt.legend()
    plt.grid(
        True,
        alpha=0.25,
    )

    return save_figure(
        base,
        dpi,
    )


def checkpoint_manifest_path(
    args: argparse.Namespace,
) -> Path:
    return (
        args.checkpoint_dir
        / "checkpoint_manifest.json"
    )


def write_checkpoint_manifest(
    args: argparse.Namespace,
    context: dict[str, Any],
    completed: dict[str, dict[str, Any]],
) -> None:
    rows = []

    for config, payload in sorted(
        completed.items()
    ):
        worker_dir = (
            args.checkpoint_dir
            / safe_slug(
                payload["candidate"]
            )
        )

        result_path = (
            worker_dir
            / "result.json"
        )

        rows.append(
            {
                "config": config,
                "status": payload[
                    "summary"
                ]["status"],
                "result_path": str(
                    result_path
                ),
                "result_sha256": (
                    sha256_file(
                        result_path
                    )
                    if result_path.is_file()
                    else None
                ),
                "predictions": payload.get(
                    "predictions"
                ),
            }
        )

    atomic_write_json(
        checkpoint_manifest_path(
            args
        ),
        {
            **context,
            "updated_at": now_iso(),
            "completed_count": len(
                rows
            ),
            "completed_candidates": rows,
        },
    )


def load_completed(
    args: argparse.Namespace,
    context: dict[str, Any],
    runner_sha: str,
    input_sha: str,
) -> dict[str, dict[str, Any]]:
    manifest_path = checkpoint_manifest_path(
        args
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "--resume requested but June checkpoint manifest is missing."
        )

    saved = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    for key in (
        "run_id",
        "runner_sha256",
        "input_sha256",
        "freeze_008_sha256",
        "april_final_reps_sha256",
        "june_split_manifest_sha256",
    ):
        if saved.get(key) != context.get(key):
            raise RuntimeError(
                f"June resume context mismatch: {key}"
            )

    completed = {}

    for candidate in FROZEN_CANDIDATES:
        worker_dir = (
            args.checkpoint_dir
            / safe_slug(candidate)
        )

        result_path = (
            worker_dir
            / "result.json"
        )

        if not result_path.is_file():
            continue

        payload = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        completed[
            candidate["config"]
        ] = validate_checkpoint(
            payload,
            candidate,
            runner_sha,
            input_sha,
            worker_dir,
        )

    return completed


def main_parent(
    args: argparse.Namespace,
) -> int:
    runner_sha = current_runner_sha()

    log("=" * 112)
    log("UGR'16 JUNE BLIND — FINAL STATISTICAL EXTERNAL EVALUATION")
    log("=" * 112)
    log(f"Campaign: {CAMPAIGN_ID}")
    log(f"Run:      {RUN_ID}")
    log(f"Runner SHA-256: {runner_sha}")
    log("Selection/tuning in June: FORBIDDEN")
    log("SARIMA execution: NO")
    log("R3 parameter refit: NO")
    log()

    freeze_hash = validate_hash(
        args.freeze_008,
        EXPECTED_FREEZE_008_SHA256,
        "Freeze 008",
    )

    final_reps_hash = validate_frozen_representatives(
        args.april_final_reps
    )

    april_manifest_hash = validate_april_manifest(
        args.april_manifest
    )

    split_manifest_hash = validate_june_split_manifest(
        args.june_split_manifest
    )

    frame, input_hash = load_june_input(
        args.input
    )

    train, test = split_june(
        frame
    )

    scale = mase_scale(
        train[TARGET].to_numpy(
            dtype=float
        )
    )

    log("PRECHECK")
    log("-" * 112)
    log(f"Freeze 008 SHA-256:       {freeze_hash}")
    log(f"April final reps SHA-256: {final_reps_hash}")
    log(f"April manifest SHA-256:   {april_manifest_hash}")
    log(f"June split SHA-256:       {split_manifest_hash}")
    log(f"June input SHA-256:       {input_hash}")
    log(
        f"Rows: total={len(frame)} "
        f"train={len(train)} "
        f"blind_test={len(test)}"
    )
    log(
        f"Boundary: {BOUNDARY.isoformat()}"
    )
    log(
        f"Train end: {train['timestamp'].iloc[-1].isoformat()}"
    )
    log(
        f"Blind start: {test['timestamp'].iloc[0].isoformat()}"
    )
    log(
        "Expected coverage: "
        "H1=602 H3=600 H6=597 H12=591"
    )
    log()
    log("FROZEN MODEL PLAN")
    log("-" * 112)

    for candidate in FROZEN_CANDIDATES:
        log(
            f"{candidate['family']:5s}: "
            f"{candidate['config']}"
        )

    log(
        f"SARIMA: {SARIMA_STATUS}"
    )

    if args.preflight_only:
        log()
        log("PRECHECK GLOBAL: PASS")
        log("Blind test structurally unlocked by Freeze 008.")
        log("No model fitted (--preflight-only).")
        return 0

    context = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "runner_sha256": runner_sha,
        "input_sha256": input_hash,
        "freeze_008_sha256": freeze_hash,
        "april_final_reps_sha256": final_reps_hash,
        "april_manifest_sha256": april_manifest_hash,
        "june_split_manifest_sha256": split_manifest_hash,
        "selection_performed": False,
        "tuning_performed": False,
    }

    if args.resume:
        completed = load_completed(
            args,
            context,
            runner_sha,
            input_hash,
        )

        log(
            f"\nRESUME: recovered "
            f"{len(completed)}/5 completed frozen representatives."
        )

    else:
        if args.checkpoint_dir.exists():
            if args.overwrite:
                shutil.rmtree(
                    args.checkpoint_dir
                )
            else:
                raise FileExistsError(
                    "June checkpoint directory exists. "
                    "Use --resume or --overwrite."
                )

        args.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        completed: dict[
            str,
            dict[str, Any],
        ] = {}

        write_checkpoint_manifest(
            args,
            context,
            completed,
        )

    log()
    log("JUNE BLIND GRID")
    log("-" * 112)

    for candidate in FROZEN_CANDIDATES:
        config = candidate["config"]

        if config in completed:
            log(
                f"SKIP {config:18s} "
                f"checkpoint status="
                f"{completed[config]['summary']['status']}"
            )
            continue

        log(
            f"START {config:18s} "
            f"family={candidate['family']:5s} "
            f"hard_timeout={RESOURCE_LIMIT_SECONDS}s"
        )

        outcome, payload = run_candidate(
            args,
            candidate,
            runner_sha,
            input_hash,
        )

        if outcome == "TECHNICAL_ERROR":
            log(
                f"TECHNICAL_ERROR {config}: "
                f"{payload}"
            )

            write_checkpoint_manifest(
                args,
                context,
                completed,
            )

            return 3

        worker_dir = (
            args.checkpoint_dir
            / safe_slug(candidate)
        )

        payload = validate_checkpoint(
            payload,
            candidate,
            runner_sha,
            input_hash,
            worker_dir,
        )

        completed[
            config
        ] = payload

        write_checkpoint_manifest(
            args,
            context,
            completed,
        )

        summary = payload[
            "summary"
        ]

        log(
            f"DONE  {config:18s} "
            f"status={summary['status']:24s} "
            f"attempts={summary['attempts']} "
            f"fit={summary['training_time_sec']} "
            f"infer={summary['inference_time_sec']}"
        )

    summaries = []
    prediction_frames = []

    for candidate in FROZEN_CANDIDATES:
        payload = completed[
            candidate["config"]
        ]

        summaries.append(
            payload["summary"]
        )

        if (
            payload[
                "summary"
            ]["status"]
            == STATUS_PASS
        ):
            prediction_frames.append(
                pd.read_parquet(
                    payload[
                        "predictions"
                    ]["path"]
                )
            )

    summary_table = pd.DataFrame(
        summaries
    )

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        if prediction_frames
        else pd.DataFrame()
    )

    metrics = (
        compute_metrics(
            predictions,
            scale,
        )
        if not predictions.empty
        else pd.DataFrame()
    )

    summary_table = attach_descriptive_score(
        summary_table,
        metrics,
    )

    if not predictions.empty:
        validate_common_timestamps(
            predictions,
            summary_table,
        )

    failed = summary_table[
        summary_table["status"]
        != STATUS_PASS
    ]["config"].tolist()

    overall_status = (
        STATUS_PASS
        if not failed
        else STATUS_BLOCKED
    )

    descriptive_ranking = (
        summary_table[
            summary_table["status"]
            == STATUS_PASS
        ]
        .sort_values(
            "score"
        )
        .reset_index(drop=True)
    )

    if not descriptive_ranking.empty:
        descriptive_ranking[
            "descriptive_rank"
        ] = range(
            1,
            len(descriptive_ranking)
            + 1,
        )

        descriptive_ranking[
            "used_for_selection"
        ] = False

    worst_intervals = build_worst_intervals(
        predictions,
        top_n=20,
    )

    outputs = {
        "candidate_summary": (
            args.metrics_dir
            / f"{args.prefix}_candidate_summary.csv"
        ),
        "metrics": (
            args.metrics_dir
            / f"{args.prefix}_metrics.csv"
        ),
        "descriptive_ranking": (
            args.metrics_dir
            / f"{args.prefix}_descriptive_ranking.csv"
        ),
        "worst_intervals": (
            args.metrics_dir
            / f"{args.prefix}_worst_intervals.csv"
        ),
        "report": (
            args.metrics_dir
            / f"{args.prefix}_report.txt"
        ),
        "manifest": (
            args.metrics_dir
            / f"{args.prefix}_manifest.json"
        ),
        "predictions_parquet": (
            args.predictions_dir
            / f"{args.prefix}_predictions.parquet"
        ),
        "predictions_csv": (
            args.predictions_dir
            / f"{args.prefix}_predictions.csv"
        ),
    }

    for path in outputs.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            path.exists()
            and not (
                args.overwrite
                or args.resume
            )
        ):
            raise FileExistsError(
                path
            )

    summary_table.to_csv(
        outputs[
            "candidate_summary"
        ],
        index=False,
    )

    metrics.to_csv(
        outputs["metrics"],
        index=False,
    )

    descriptive_ranking.to_csv(
        outputs[
            "descriptive_ranking"
        ],
        index=False,
    )

    worst_intervals.to_csv(
        outputs[
            "worst_intervals"
        ],
        index=False,
    )

    predictions.to_parquet(
        outputs[
            "predictions_parquet"
        ],
        index=False,
    )

    predictions.to_csv(
        outputs[
            "predictions_csv"
        ],
        index=False,
    )

    args.figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_paths = []

    if not metrics.empty:
        figure_paths += plot_mae(
            metrics,
            args.figures_dir
            / f"{args.prefix}_mae_horizon",
            args.dpi,
        )

        figure_paths += plot_skill(
            metrics,
            args.figures_dir
            / f"{args.prefix}_skill_horizon",
            args.dpi,
        )

        figure_paths += plot_heatmap(
            metrics,
            args.figures_dir
            / f"{args.prefix}_mae_heatmap",
            args.dpi,
        )

        figure_paths += plot_bias(
            metrics,
            args.figures_dir
            / f"{args.prefix}_bias_horizon",
            args.dpi,
        )

        figure_paths += plot_p95(
            metrics,
            args.figures_dir
            / f"{args.prefix}_p95_abs_error_horizon",
            args.dpi,
        )

        figure_paths += plot_underprediction(
            metrics,
            args.figures_dir
            / f"{args.prefix}_underprediction_horizon",
            args.dpi,
        )

    if not descriptive_ranking.empty:
        figure_paths += plot_score(
            summary_table,
            args.figures_dir
            / f"{args.prefix}_score_descriptive",
            args.dpi,
        )

        best_config = (
            descriptive_ranking.iloc[
                0
            ]["config"]
        )

        figure_paths += plot_observed_vs_predicted_h1(
            predictions,
            best_config,
            args.figures_dir
            / f"{args.prefix}_best_score_observed_h1",
            args.dpi,
        )

    report_lines = [
        "=" * 112,
        "UGR'16 JUNE BLIND — FINAL STATISTICAL EXTERNAL EVALUATION",
        "=" * 112,
        f"Campaign: {CAMPAIGN_ID}",
        f"Run: {RUN_ID}",
        f"Runner SHA-256: {runner_sha}",
        f"Freeze 008 SHA-256: {freeze_hash}",
        f"June input SHA-256: {input_hash}",
        f"June split manifest SHA-256: {split_manifest_hash}",
        "",
        "GOVERNANCE",
        "-" * 112,
        "Selection performed in June: NO",
        "Tuning performed in June: NO",
        f"SARIMA_STATUS: {SARIMA_STATUS}",
        "",
        "SPLIT",
        "-" * 112,
        f"Total: {len(frame)}",
        f"Train: {len(train)}",
        f"Blind test: {len(test)}",
        f"Boundary: {BOUNDARY.isoformat()}",
        f"Train end: {train['timestamp'].iloc[-1].isoformat()}",
        f"Blind start: {test['timestamp'].iloc[0].isoformat()}",
        "Parameter refit: NO",
        "Observed history/state update: YES",
        "",
        "FROZEN REPRESENTATIVES",
        "-" * 112,
    ]

    for candidate in FROZEN_CANDIDATES:
        report_lines.append(
            f"{candidate['family']}: "
            f"{candidate['config']}"
        )

    report_lines += [
        "",
        "EXTERNAL RESULTS",
        "-" * 112,
    ]

    if descriptive_ranking.empty:
        report_lines.append(
            "No valid frozen representative."
        )
    else:
        for _, row in descriptive_ranking.iterrows():
            report_lines.append(
                f"rank={int(row['descriptive_rank'])} "
                f"{row['family']} {row['config']} "
                f"Score={row['score']:.8f} "
                f"(descriptive only)"
            )

    report_lines += [
        "",
        "GLOBAL RESULT",
        "-" * 112,
        f"STATUS: {overall_status}",
        f"Failed frozen representatives: {failed}",
        "Frozen-model substitution: FORBIDDEN",
        "June used for selection: NO",
        "",
    ]

    outputs["report"].write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    artifact_paths = [
        outputs[
            "candidate_summary"
        ],
        outputs["metrics"],
        outputs[
            "descriptive_ranking"
        ],
        outputs[
            "worst_intervals"
        ],
        outputs["report"],
        outputs[
            "predictions_parquet"
        ],
        outputs[
            "predictions_csv"
        ],
        *figure_paths,
    ]

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": overall_status,
        "created_at": now_iso(),
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(
                Path(__file__).resolve()
            ),
            "sha256": runner_sha,
            "worker_isolation": True,
            "checkpoint_resume": True,
            "hard_timeout_seconds_per_candidate": RESOURCE_LIMIT_SECONDS,
        },
        "authorities": {
            "freeze_008": {
                "path": str(
                    args.freeze_008
                ),
                "sha256": freeze_hash,
            },
            "april_final_representatives": {
                "path": str(
                    args.april_final_reps
                ),
                "sha256": final_reps_hash,
            },
            "april_manifest": {
                "path": str(
                    args.april_manifest
                ),
                "sha256": april_manifest_hash,
            },
            "june_split_manifest": {
                "path": str(
                    args.june_split_manifest
                ),
                "sha256": split_manifest_hash,
            },
        },
        "input": {
            "path": str(
                args.input
            ),
            "sha256": input_hash,
            "rows": len(frame),
        },
        "split": {
            "n_train": len(train),
            "n_blind_test": len(test),
            "boundary": BOUNDARY.isoformat(),
            "train_end": (
                train[
                    "timestamp"
                ].iloc[-1].isoformat()
            ),
            "blind_start": (
                test[
                    "timestamp"
                ].iloc[0].isoformat()
            ),
            "refit_test": False,
            "observed_update": True,
        },
        "frozen_representatives": FROZEN_CANDIDATES,
        "sarima_status": SARIMA_STATUS,
        "evaluation": {
            "selection_performed": False,
            "tuning_performed": False,
            "substitution_allowed": False,
            "score_role": "descriptive_only",
            "primary_metric": "MAE",
            "same_target_timestamps_required": True,
            "coverage_required": 1.0,
        },
        "horizons": {
            str(horizon): {
                "minutes": HORIZON_MINUTES[
                    horizon
                ],
                "expected_predictions": EXPECTED_COUNTS[
                    horizon
                ],
            }
            for horizon in HORIZONS
        },
        "mase_scale_bps": scale,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": {
            str(path): sha256_file(
                path
            )
            for path in artifact_paths
            if path.is_file()
        },
    }

    outputs["manifest"].write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("RESULT")
    log("-" * 112)
    log(
        f"STATUS: {overall_status}"
    )
    log(
        f"Failed frozen representatives: {failed}"
    )

    if not descriptive_ranking.empty:
        log(
            "Descriptive ranking: "
            + str(
                descriptive_ranking[
                    [
                        "family",
                        "config",
                        "score",
                    ]
                ].to_dict(
                    "records"
                )
            )
        )

    log(
        f"SARIMA_STATUS={SARIMA_STATUS}"
    )
    log(
        "Selection/tuning in June: NO"
    )

    return (
        0
        if overall_status
        == STATUS_PASS
        else 2
    )


def main() -> int:
    args = parse_args()

    if args.worker:
        return worker_main(
            args
        )

    return main_parent(
        args
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
