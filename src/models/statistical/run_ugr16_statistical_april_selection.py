#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
April final selection — AR / MA / ARMA / ARIMA / VAR.

Authorities:
- 003 TFM-STAT-PROTOCOL-001 — FROZEN
- 004 ARMA numerical fallback — FROZEN
- 005 SARIMA structural spec — FROZEN
- 006 SARIMA resource feasibility record — FROZEN
- 007 SARIMA scope resolution — FROZEN

April role:
- final known-data selection, before June;
- 2004 intervals @ 5 min;
- initial 70 % = 1402 fit;
- final 30 % = 602 R3 evaluation;
- no June use.

Exactly 15 candidates:
AR:    24, 12, 6
MA:    12, 6, 3
ARMA:  (24,3), (6,12), (12,6)
ARIMA: (6,1,12), (24,1,3), (12,1,6)
VAR:   5, 6, 2

One representative per family is selected by:
Score -> practical tie <=1% -> BIC -> n_params -> training_time.

The output representatives are selection results, NOT yet the documentary
freeze that authorizes June. They must be reviewed and then frozen separately.
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
RUN_ID = "UGR16-STATISTICAL-APRIL-SELECTION-001"
RUNNER_VERSION = "1.0"

EXPECTED_PROTOCOL_SHA256 = (
    "031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699"
)
EXPECTED_AMENDMENT_004_SHA256 = (
    "03704a5a2e072e8f13a7296ef1623a8131755c809ccfb36e897dab195c5bea21"
)
EXPECTED_AMENDMENT_005_SHA256 = (
    "e049defd946cc5e6f111c488800ce280b96027e602f577bc8cefc6a9a912649a"
)
EXPECTED_RECORD_006_SHA256 = (
    "a30f10fc0dc29e103bc8f8adb629a588d2597a23fc418258a1e5614888883f5d"
)
EXPECTED_AMENDMENT_007_SHA256 = "e7e2f2ba38e5dbef6eca7834826e5f89ce125db659eeb53661a2f9909f03806b"

EXPECTED_AR_MA_MANIFEST_SHA256 = (
    "cabc7d6166d3ceb5874f6d2b00eb8974aa4ee22acf4ccd1eb4639166ea543f31"
)
EXPECTED_ARMA_MANIFEST_SHA256 = (
    "e9849350e7abf551f75a15943ff26df31cbc2aed3009fa9c5b316a672660363f"
)
EXPECTED_ARIMA_MANIFEST_SHA256 = (
    "5fdc6638b48d785a417d60d41545a6206421978a7669ba40aa63376d33e4a80d"
)
EXPECTED_VAR_MANIFEST_SHA256 = (
    "8cfc4c7e29c5f93ca1e3296b8a402a48f5ea1cf5eab394a1522092f3989a0b7f"
)

EXPECTED_APRIL_5MIN_SHA256 = (
    "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0"
)

TARGET = "bitrate_bps"
VAR_COLUMNS = ("bitrate_bps", "packet_rate_pps", "flow_rate_fps")
TARGET_INDEX = 0

EXPECTED_TOTAL = 2004
EXPECTED_TRAIN = 1402
EXPECTED_VALIDATION = 602
TRAIN_FRACTION = 0.70
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

PRACTICAL_TIE_REL = 0.01
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
STATUS_BLOCKED = "BLOCKED_FAMILY_WITHOUT_VALID_CANDIDATE"

CANDIDATES = [
    {"family": "AR", "config": "AR(24)", "p": 24},
    {"family": "AR", "config": "AR(12)", "p": 12},
    {"family": "AR", "config": "AR(6)", "p": 6},

    {"family": "MA", "config": "MA(12)", "p": 0, "d": 0, "q": 12},
    {"family": "MA", "config": "MA(6)", "p": 0, "d": 0, "q": 6},
    {"family": "MA", "config": "MA(3)", "p": 0, "d": 0, "q": 3},

    {"family": "ARMA", "config": "ARMA(24,3)", "p": 24, "d": 0, "q": 3},
    {"family": "ARMA", "config": "ARMA(6,12)", "p": 6, "d": 0, "q": 12},
    {"family": "ARMA", "config": "ARMA(12,6)", "p": 12, "d": 0, "q": 6},

    {"family": "ARIMA", "config": "ARIMA(6,1,12)", "p": 6, "d": 1, "q": 12},
    {"family": "ARIMA", "config": "ARIMA(24,1,3)", "p": 24, "d": 1, "q": 3},
    {"family": "ARIMA", "config": "ARIMA(12,1,6)", "p": 12, "d": 1, "q": 6},

    {"family": "VAR", "config": "VAR(5)", "lag": 5},
    {"family": "VAR", "config": "VAR(6)", "lag": 6},
    {"family": "VAR", "config": "VAR(2)", "lag": 2},
]

EXPECTED_BY_FAMILY = {
    "AR": ["AR(24)", "AR(12)", "AR(6)"],
    "MA": ["MA(12)", "MA(6)", "MA(3)"],
    "ARMA": ["ARMA(24,3)", "ARMA(6,12)", "ARMA(12,6)"],
    "ARIMA": ["ARIMA(6,1,12)", "ARIMA(24,1,3)", "ARIMA(12,1,6)"],
    "VAR": ["VAR(5)", "VAR(6)", "VAR(2)"],
}
FAMILIES = ("AR", "MA", "ARMA", "ARIMA", "VAR")


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


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified April statistical selection for UGR16."
    )
    p.add_argument(
        "--input", type=Path,
        default=Path("data/processed/ugr16/april_week3_prepared_5min.parquet")
    )
    p.add_argument(
        "--protocol-file", type=Path,
        default=Path(
            "docs/project_governance/003_statistical_models_protocol_2026-08-14.md"
        )
    )
    p.add_argument(
        "--amendment-004-file", type=Path,
        default=Path(
            "docs/project_governance/004_arma_numerical_fallback_2026-08-15.md"
        )
    )
    p.add_argument(
        "--amendment-005-file", type=Path,
        default=Path(
            "docs/project_governance/005_sarima_structural_spec_2026-08-15.md"
        )
    )
    p.add_argument(
        "--record-006-file", type=Path,
        default=Path(
            "docs/project_governance/006_sarima_resource_feasibility_2026-08-16.md"
        )
    )
    p.add_argument(
        "--amendment-007-file", type=Path,
        default=Path(
            "docs/project_governance/007_sarima_scope_resolution_2026-08-16.md"
        )
    )
    p.add_argument(
        "--ar-ma-manifest", type=Path,
        default=Path("results/metrics/ugr16_statistical_ar_ma_march_manifest.json")
    )
    p.add_argument(
        "--arma-manifest", type=Path,
        default=Path("results/metrics/ugr16_statistical_arma_march_v2_manifest.json")
    )
    p.add_argument(
        "--arima-manifest", type=Path,
        default=Path("results/metrics/ugr16_statistical_arima_march_manifest.json")
    )
    p.add_argument(
        "--var-manifest", type=Path,
        default=Path("results/metrics/ugr16_statistical_var_march_manifest.json")
    )
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    p.add_argument(
        "--figures-dir", type=Path,
        default=Path("results/figures/ugr16_statistical_april_selection")
    )
    p.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("results/checkpoints/ugr16_statistical_april_selection")
    )
    p.add_argument("--prefix", default="ugr16_statistical_april_selection")
    p.add_argument("--dpi", type=int, default=180)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")

    # internal worker
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--candidate-json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--runner-sha", default=None, help=argparse.SUPPRESS)
    p.add_argument("--input-sha", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def validate_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA mismatch.\nExpected: {expected}\nObserved: {observed}"
        )
    return observed


def current_runner_sha() -> str:
    return sha256_file(Path(__file__).resolve())


def load_input(path: Path) -> tuple[pd.DataFrame, str]:
    observed = validate_hash(path, EXPECTED_APRIL_5MIN_SHA256, "April input")
    frame = pd.read_parquet(path).copy()
    if "timestamp" not in frame.columns:
        if frame.index.name == "timestamp":
            frame = frame.reset_index()
        else:
            raise ValueError("Missing timestamp.")
    required = {"timestamp", *VAR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for col in VAR_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(f"Rows {len(frame)} != {EXPECTED_TOTAL}")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")
    if not np.isfinite(frame[list(VAR_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite input.")
    if not (frame["timestamp"].diff().dropna() == EXPECTED_FREQUENCY).all():
        raise ValueError("Input is not exactly 5-minute continuous.")
    return frame, observed


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(frame) * TRAIN_FRACTION)
    if n_train != EXPECTED_TRAIN:
        raise RuntimeError(f"Train {n_train} != {EXPECTED_TRAIN}")
    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:].copy()
    if len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError("Unexpected validation length.")
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise RuntimeError("Chronology violation.")
    return train, validation


def validate_source_manifests(args: argparse.Namespace) -> dict[str, str]:
    sources = {
        "AR_MA": (args.ar_ma_manifest, EXPECTED_AR_MA_MANIFEST_SHA256),
        "ARMA": (args.arma_manifest, EXPECTED_ARMA_MANIFEST_SHA256),
        "ARIMA": (args.arima_manifest, EXPECTED_ARIMA_MANIFEST_SHA256),
        "VAR": (args.var_manifest, EXPECTED_VAR_MANIFEST_SHA256),
    }
    observed = {}
    for label, (path, expected) in sources.items():
        observed[label] = validate_hash(path, expected, f"{label} March manifest")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("campaign_id") != CAMPAIGN_ID:
            raise RuntimeError(f"{label} campaign mismatch.")
        if payload.get("status") != "PASS":
            raise RuntimeError(f"{label} March manifest is not PASS.")
        if payload.get("blindness", {}).get("june_used") is not False:
            raise RuntimeError(f"{label} March manifest indicates June use.")
    return observed


def validate_candidate_plan() -> None:
    if len(CANDIDATES) != 15:
        raise RuntimeError("Expected exactly 15 April candidates.")
    for family in FAMILIES:
        observed = [c["config"] for c in CANDIDATES if c["family"] == family]
        if observed != EXPECTED_BY_FAMILY[family]:
            raise RuntimeError(
                f"{family} shortlist mismatch. "
                f"Expected={EXPECTED_BY_FAMILY[family]} observed={observed}"
            )


def mase_scale(train_y: np.ndarray) -> float:
    scale = float(np.mean(np.abs(np.diff(train_y))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid MASE scale.")
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


def validate_coverage(pred: pd.DataFrame) -> tuple[str, str]:
    if pred.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."
    if not np.isfinite(
        pred[["y_true", "y_pred", "persistence_pred"]].to_numpy(dtype=float)
    ).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf predictions."
    problems = []
    for h in HORIZONS:
        sub = pred[pred["horizon_steps"] == h].sort_values("target_timestamp")
        if len(sub) != EXPECTED_COUNTS[h]:
            problems.append(f"H{h}: {len(sub)} != {EXPECTED_COUNTS[h]}")
        if sub["target_timestamp"].duplicated().any():
            problems.append(f"H{h}: duplicate target timestamps")
    return (
        (STATUS_PASS, "")
        if not problems
        else (STATUS_INVALID_COVERAGE, " | ".join(problems))
    )


def fit_ar(train_y: pd.Series, p: int) -> tuple[Any, dict[str, Any]]:
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
    params = np.asarray(result.params, dtype=float)
    detail = {
        "attempts": 1,
        "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
        "converged": True,
        "finite_params": bool(np.isfinite(params).all()),
        "training_time_sec": float(time.perf_counter() - start),
    }
    return result, detail


def eval_ar(
    result: Any,
    candidate: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    p = int(candidate["p"])
    params = np.asarray(result.params, dtype=float)
    if len(params) != p + 1:
        raise RuntimeError(f"Unexpected AutoReg params for p={p}: {len(params)}")
    intercept = float(params[0])
    phi = params[1:].copy()
    history = train[TARGET].to_numpy(dtype=float).tolist()
    valid_y = validation[TARGET].to_numpy(dtype=float)
    rows = []
    start = time.perf_counter()

    for k in range(len(validation)):
        origin_ts = (
            pd.Timestamp(train["timestamp"].iloc[-1])
            if k == 0 else pd.Timestamp(validation["timestamp"].iloc[k - 1])
        )
        persistence = float(history[-1])
        local = history.copy()
        forecast = []
        for _ in range(max(HORIZONS)):
            lag_values = np.array(local[-p:][::-1], dtype=float)
            yhat = intercept + float(np.dot(phi, lag_values))
            forecast.append(yhat)
            local.append(yhat)

        for h in HORIZONS:
            target_pos = k + h - 1
            if target_pos >= len(validation):
                continue
            rows.append(
                prediction_row(
                    candidate, origin_ts,
                    pd.Timestamp(validation["timestamp"].iloc[target_pos]),
                    h, float(valid_y[target_pos]),
                    float(forecast[h - 1]), persistence
                )
            )
        if k < len(validation) - 1:
            history.append(float(valid_y[k]))

    return pd.DataFrame(rows), float(time.perf_counter() - start)


def fit_arima_family(
    train_series: pd.Series,
    candidate: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    order = (int(candidate["p"]), int(candidate["d"]), int(candidate["q"]))
    trend = "c" if order[1] == 0 else "n"
    attempts = []
    total = 0.0
    final = None
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
                    else model.fit(method_kwargs={"maxiter": maxiter})
                )
                detail["warnings"] = [
                    f"{w.category.__name__}: {w.message}" for w in caught
                ]
            ret = getattr(result, "mle_retvals", {})
            detail["converged"] = bool(ret.get("converged", True))
            detail["finite_params"] = bool(
                np.isfinite(np.asarray(result.params, dtype=float)).all()
            )
            if detail["converged"] and detail["finite_params"]:
                final = result
        except Exception as exc:
            detail["exception"] = f"{type(exc).__name__}: {exc}"
        elapsed = float(time.perf_counter() - start)
        detail["elapsed_sec"] = elapsed
        total += elapsed
        attempts.append(detail)
        if final is not None:
            break

    return final, {
        "attempts": len(attempts),
        "attempt_details": attempts,
        "warnings": [
            warning
            for detail in attempts
            for warning in detail.get("warnings", [])
        ],
        "converged": final is not None,
        "finite_params": final is not None,
        "training_time_sec": total,
        "trend": trend,
    }


def eval_state_space(
    result: Any,
    candidate: dict[str, Any],
    train_series: pd.Series,
    validation_series: pd.Series,
) -> tuple[pd.DataFrame, float]:
    current = result
    rows = []
    start = time.perf_counter()

    for k in range(len(validation_series)):
        remaining = len(validation_series) - k
        steps = min(max(HORIZONS), remaining)
        origin_ts = (
            pd.Timestamp(train_series.index[-1])
            if k == 0 else pd.Timestamp(validation_series.index[k - 1])
        )
        persistence = (
            float(train_series.iloc[-1])
            if k == 0 else float(validation_series.iloc[k - 1])
        )
        forecast = np.asarray(current.forecast(steps=steps), dtype=float)
        if len(forecast) != steps or not np.isfinite(forecast).all():
            raise FloatingPointError("Invalid ARIMA-family forecast.")

        for h in HORIZONS:
            if h > steps:
                continue
            target_pos = k + h - 1
            rows.append(
                prediction_row(
                    candidate, origin_ts,
                    pd.Timestamp(validation_series.index[target_pos]),
                    h, float(validation_series.iloc[target_pos]),
                    float(forecast[h - 1]), persistence
                )
            )

        if k < len(validation_series) - 1:
            current = current.append(validation_series.iloc[k:k+1], refit=False)

    return pd.DataFrame(rows), float(time.perf_counter() - start)


def build_var_training(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    levels = train.set_index("timestamp")[list(VAR_COLUMNS)].astype(float)
    diff = levels.diff().dropna()
    mu = diff.mean(axis=0)
    sigma = diff.std(axis=0, ddof=0)
    if (sigma <= 0).any():
        raise ValueError("Invalid VAR scaler.")
    z = (diff - mu) / sigma
    scaler = {
        "ddof": 0,
        "mean": {c: float(mu[c]) for c in VAR_COLUMNS},
        "std": {c: float(sigma[c]) for c in VAR_COLUMNS},
    }
    return z, scaler


def fit_var(
    train: pd.DataFrame,
    lag: int,
) -> tuple[Any | None, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    train_z, scaler = build_var_training(train)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = VAR(train_z).fit(maxlags=lag, ic=None, trend="c")
    elapsed = float(time.perf_counter() - start)
    params = np.asarray(result.params, dtype=float)
    roots = np.asarray(result.roots, dtype=complex)
    detail = {
        "attempts": 1,
        "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
        "converged": True,
        "finite_params": bool(np.isfinite(params).all()),
        "stable": bool(result.is_stable(verbose=False)),
        "min_abs_root": float(np.min(np.abs(roots))) if roots.size else np.nan,
        "training_time_sec": elapsed,
    }
    return result, detail, train_z, scaler


def eval_var(
    result: Any,
    candidate: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_z: pd.DataFrame,
    scaler: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    lag = int(candidate["lag"])
    history_z = train_z.to_numpy(dtype=float).copy()
    train_levels = train[list(VAR_COLUMNS)].to_numpy(dtype=float)
    valid_levels = validation[list(VAR_COLUMNS)].to_numpy(dtype=float)
    mu = np.array([scaler["mean"][c] for c in VAR_COLUMNS], dtype=float)
    sigma = np.array([scaler["std"][c] for c in VAR_COLUMNS], dtype=float)
    rows = []
    start = time.perf_counter()

    for k in range(len(validation)):
        remaining = len(validation) - k
        steps = min(max(HORIZONS), remaining)
        origin_ts = (
            pd.Timestamp(train["timestamp"].iloc[-1])
            if k == 0 else pd.Timestamp(validation["timestamp"].iloc[k - 1])
        )
        origin_level = train_levels[-1] if k == 0 else valid_levels[k - 1]
        forecast_z = np.asarray(
            result.forecast(history_z[-lag:], steps=steps), dtype=float
        )
        if not np.isfinite(forecast_z).all():
            raise FloatingPointError("Invalid VAR forecast.")
        forecast_diff = forecast_z * sigma + mu
        bitrate_levels = origin_level[TARGET_INDEX] + np.cumsum(
            forecast_diff[:, TARGET_INDEX]
        )

        for h in HORIZONS:
            if h > steps:
                continue
            target_pos = k + h - 1
            rows.append(
                prediction_row(
                    candidate, origin_ts,
                    pd.Timestamp(validation["timestamp"].iloc[target_pos]),
                    h, float(valid_levels[target_pos, TARGET_INDEX]),
                    float(bitrate_levels[h - 1]),
                    float(origin_level[TARGET_INDEX])
                )
            )

        if k < len(validation) - 1:
            previous = train_levels[-1] if k == 0 else valid_levels[k - 1]
            observed_diff = valid_levels[k] - previous
            observed_z = (observed_diff - mu) / sigma
            history_z = np.vstack([history_z, observed_z])

    return pd.DataFrame(rows), float(time.perf_counter() - start)


def worker_evaluate(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, validation = split_frame(frame)
    family = candidate["family"]
    total_start = time.perf_counter()
    status = STATUS_PASS
    coverage_note = ""
    result = None
    fit_detail: dict[str, Any] = {}
    pred = pd.DataFrame()
    inference_time = np.nan
    extra: dict[str, Any] = {}

    try:
        if family == "AR":
            train_y = pd.Series(
                train[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(train["timestamp"].to_numpy(), freq="5min"),
                name=TARGET,
            )
            result, fit_detail = fit_ar(train_y, int(candidate["p"]))
            if not fit_detail["finite_params"]:
                status = STATUS_INVALID_NUMERICAL
            else:
                pred, inference_time = eval_ar(result, candidate, train, validation)

        elif family in ("MA", "ARMA", "ARIMA"):
            train_series = pd.Series(
                train[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(train["timestamp"].to_numpy(), freq="5min"),
                name=TARGET,
            )
            validation_series = pd.Series(
                validation[TARGET].to_numpy(dtype=float),
                index=pd.DatetimeIndex(validation["timestamp"].to_numpy(), freq="5min"),
                name=TARGET,
            )
            result, fit_detail = fit_arima_family(train_series, candidate)
            if result is None:
                status = STATUS_INVALID_NUMERICAL
            else:
                pred, inference_time = eval_state_space(
                    result, candidate, train_series, validation_series
                )

        elif family == "VAR":
            result, fit_detail, train_z, scaler = fit_var(
                train, int(candidate["lag"])
            )
            extra["var_scaler"] = scaler
            if not fit_detail["finite_params"]:
                status = STATUS_INVALID_NUMERICAL
            elif not fit_detail["stable"]:
                status = STATUS_INVALID_STABILITY
            else:
                pred, inference_time = eval_var(
                    result, candidate, train, validation, train_z, scaler
                )
        else:
            raise ValueError(f"Unknown family: {family}")

        if status == STATUS_PASS:
            status, coverage_note = validate_coverage(pred)

    except MemoryError as exc:
        status = STATUS_INVALID_RESOURCE_LIMIT
        coverage_note = f"MemoryError: {exc}"
    except Exception as exc:
        status = STATUS_INVALID_NUMERICAL
        coverage_note = f"{type(exc).__name__}: {exc}"

    aic = bic = np.nan
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
            n_params = int(np.asarray(result.params).size)
        except Exception:
            pass

    summary = {
        "family": family,
        "config": candidate["config"],
        "aic": aic,
        "bic": bic,
        "n_params": n_params,
        "converged": bool(fit_detail.get("converged", False)),
        "finite_params": bool(fit_detail.get("finite_params", False)),
        "stable": (
            bool(fit_detail.get("stable"))
            if family == "VAR" else True
        ),
        "min_abs_root": (
            fit_detail.get("min_abs_root", np.nan)
            if family == "VAR" else np.nan
        ),
        "attempts": int(fit_detail.get("attempts", 0)),
        "warnings": " | ".join(fit_detail.get("warnings", [])),
        "attempt_details": fit_detail.get("attempt_details", []),
        "training_time_sec": fit_detail.get("training_time_sec", np.nan),
        "inference_time_sec": inference_time,
        "total_time_sec": float(time.perf_counter() - total_start),
        "status": status,
        "coverage_note": coverage_note,
        **extra,
    }
    return summary, pred


def worker_main(args: argparse.Namespace) -> int:
    if not args.candidate_json or args.worker_dir is None:
        raise ValueError("Worker arguments missing.")
    if current_runner_sha() != args.runner_sha:
        raise RuntimeError("Worker runner SHA mismatch.")
    frame, input_sha = load_input(args.input)
    if input_sha != args.input_sha:
        raise RuntimeError("Worker input SHA mismatch.")
    candidate = json.loads(args.candidate_json)
    args.worker_dir.mkdir(parents=True, exist_ok=True)
    summary, pred = worker_evaluate(frame, candidate)

    pred_meta = None
    if summary["status"] == STATUS_PASS:
        pred_path = args.worker_dir / "predictions.parquet"
        tmp = pred_path.with_name(pred_path.name + f".tmp.{os.getpid()}")
        pred.to_parquet(tmp, index=False)
        os.replace(tmp, pred_path)
        pred_meta = {
            "path": str(pred_path),
            "sha256": sha256_file(pred_path),
            "rows": len(pred),
        }

    payload = {
        "completed": True,
        "run_id": RUN_ID,
        "runner_sha256": args.runner_sha,
        "input_sha256": input_sha,
        "candidate": candidate,
        "summary": summary,
        "predictions": pred_meta,
        "finished_at": now_iso(),
    }
    atomic_write_json(args.worker_dir / "result.json", payload)
    log(
        f"WORKER DONE {candidate['config']} "
        f"status={summary['status']} "
        f"fit={summary['training_time_sec']} "
        f"infer={summary['inference_time_sec']}"
    )
    return 0


def safe_slug(candidate: dict[str, Any]) -> str:
    family = candidate["family"].lower()
    config = (
        candidate["config"]
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
    )
    return f"{family}_{config}"


def terminate_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
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
            "stable": False if candidate["family"] == "VAR" else True,
            "min_abs_root": np.nan,
            "attempts": 0,
            "warnings": f"Parent hard timeout at {RESOURCE_LIMIT_SECONDS}s.",
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
        raise RuntimeError("Incomplete checkpoint.")
    if payload.get("run_id") != RUN_ID:
        raise RuntimeError("Checkpoint run mismatch.")
    if payload.get("runner_sha256") != runner_sha:
        raise RuntimeError("Checkpoint runner SHA mismatch.")
    if payload.get("input_sha256") != input_sha:
        raise RuntimeError("Checkpoint input SHA mismatch.")
    if payload.get("candidate") != candidate:
        raise RuntimeError("Checkpoint candidate mismatch.")
    if payload["summary"]["status"] == STATUS_PASS:
        meta = payload.get("predictions")
        if not isinstance(meta, dict):
            raise RuntimeError("PASS checkpoint missing predictions.")
        pred_path = Path(meta["path"])
        if not pred_path.is_file():
            pred_path = worker_dir / "predictions.parquet"
        if not pred_path.is_file():
            raise RuntimeError("Missing checkpoint predictions.")
        if sha256_file(pred_path) != meta["sha256"]:
            raise RuntimeError("Checkpoint prediction SHA mismatch.")
        payload["predictions"]["path"] = str(pred_path)
    return payload


def run_candidate(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    runner_sha: str,
    input_sha: str,
) -> tuple[str, dict[str, Any]]:
    worker_dir = args.checkpoint_dir / safe_slug(candidate)
    worker_dir.mkdir(parents=True, exist_ok=True)

    for name in ("result.json", "predictions.parquet"):
        p = worker_dir / name
        if p.exists():
            p.unlink()

    log_path = worker_dir / "worker.log"
    if log_path.exists():
        rotated = worker_dir / (
            "worker.previous." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        )
        os.replace(log_path, rotated)

    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker",
        "--input", str(args.input),
        "--candidate-json", json.dumps(candidate, separators=(",", ":")),
        "--worker-dir", str(worker_dir),
        "--runner-sha", runner_sha,
        "--input-sha", input_sha,
    ]

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        handle.write(f"[PARENT] {now_iso()} START {candidate['config']}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=RESOURCE_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            terminate_group(proc)
            elapsed = float(time.perf_counter() - start)
            payload = timeout_payload(candidate, runner_sha, input_sha, elapsed)
            atomic_write_json(worker_dir / "result.json", payload)
            handle.write(f"[PARENT] HARD TIMEOUT {elapsed:.3f}s\n")
            return "COMPLETED", payload

    result_path = worker_dir / "result.json"
    if code != 0 or not result_path.is_file():
        return "TECHNICAL_ERROR", {
            "status": STATUS_TECHNICAL_ERROR,
            "candidate": candidate,
            "return_code": code,
            "worker_log": str(log_path),
        }
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload = validate_checkpoint(
        payload, candidate, runner_sha, input_sha, worker_dir
    )
    return "COMPLETED", payload


def smape(y: np.ndarray, p: np.ndarray) -> float:
    den = np.abs(y) + np.abs(p)
    part = np.zeros_like(den, dtype=float)
    mask = den != 0
    part[mask] = 2.0 * np.abs(y[mask] - p[mask]) / den[mask]
    return float(100.0 * np.mean(part))


def compute_metrics(pred: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for (family, config, h), group in pred.groupby(
        ["family", "config", "horizon_steps"], sort=False
    ):
        group = group.sort_values("target_timestamp")
        y = group["y_true"].to_numpy(dtype=float)
        p = group["y_pred"].to_numpy(dtype=float)
        pers = group["persistence_pred"].to_numpy(dtype=float)
        residual = y - p
        ae = np.abs(residual)
        mae = float(np.mean(ae))
        pmae = float(np.mean(np.abs(y - pers)))
        rows.append({
            "family": family,
            "config": config,
            "horizon_steps": int(h),
            "horizon_minutes": HORIZON_MINUTES[int(h)],
            "n_predictions": len(group),
            "expected_predictions": EXPECTED_COUNTS[int(h)],
            "coverage": len(group) / EXPECTED_COUNTS[int(h)],
            "mae_bps": mae,
            "rmse_bps": float(np.sqrt(np.mean(residual ** 2))),
            "smape_pct": smape(y, p),
            "mase": mae / scale,
            "persistence_mae_bps": pmae,
            "skill_vs_persistence": 1.0 - mae / pmae,
            "mean_bias_bps": float(np.mean(residual)),
            "underprediction_pct": float(100.0 * np.mean(y > p)),
            "p95_abs_error_bps": float(np.percentile(ae, 95)),
        })
    return pd.DataFrame(rows)


def attach_score(summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["score"] = np.nan
    for idx, row in out.iterrows():
        if row["status"] != STATUS_PASS:
            continue
        sub = metrics[metrics["config"] == row["config"]]
        if set(sub["horizon_steps"].astype(int)) != set(HORIZONS):
            out.at[idx, "status"] = STATUS_INVALID_COVERAGE
            continue
        out.at[idx, "score"] = float(np.mean(
            sub["mae_bps"].to_numpy(dtype=float)
            / sub["persistence_mae_bps"].to_numpy(dtype=float)
        ))
    return out


def rank_family(group: pd.DataFrame) -> pd.DataFrame:
    remaining = group[group["status"] == STATUS_PASS].copy()
    ranked = []
    next_rank = 1
    while not remaining.empty:
        best = float(remaining["score"].min())
        rel = (remaining["score"] - best) / best
        tie = remaining.loc[rel <= PRACTICAL_TIE_REL + 1e-15].copy()
        tie = tie.sort_values(
            ["bic", "n_params", "training_time_sec", "config"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        tie["practical_tie_group_best_score"] = best
        tie["relative_to_group_best"] = (tie["score"] - best) / best
        tie["rank"] = range(next_rank, next_rank + len(tie))
        ranked.append(tie)
        next_rank += len(tie)
        remaining = remaining.drop(index=tie.index)
    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def validate_common_timestamps(pred: pd.DataFrame, summary: pd.DataFrame) -> None:
    eligible = summary[summary["status"] == STATUS_PASS]
    for h in HORIZONS:
        reference = None
        ref_config = None
        for _, row in eligible.iterrows():
            sub = pred[
                (pred["config"] == row["config"])
                & (pred["horizon_steps"] == h)
            ].sort_values("target_timestamp")
            timestamps = tuple(pd.to_datetime(sub["target_timestamp"]).tolist())
            if reference is None:
                reference = timestamps
                ref_config = row["config"]
            elif timestamps != reference:
                raise RuntimeError(
                    f"Timestamp mismatch H{h}: {row['config']} vs {ref_config}"
                )


def savefig(base: Path, dpi: int) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def plot_candidate_scores(ranking: pd.DataFrame, base: Path, dpi: int) -> list[Path]:
    if ranking.empty:
        return []
    ordered = ranking.sort_values(["family", "rank"])
    plt.figure(figsize=(13, 6))
    plt.bar(ordered["config"], ordered["score"])
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.ylabel("Score medio MAE / persistencia")
    plt.xlabel("Configuración")
    plt.title("UGR'16 April — selección estadística — Score por candidato")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return savefig(base, dpi)


def plot_final_mae(final_metrics: pd.DataFrame, base: Path, dpi: int) -> list[Path]:
    plt.figure(figsize=(10, 6))
    for config, sub in final_metrics.groupby("config", sort=False):
        sub = sub.sort_values("horizon_minutes")
        plt.plot(
            sub["horizon_minutes"], sub["mae_bps"] / 1e6,
            marker="o", linewidth=1.2, label=config
        )
    plt.xlabel("Horizonte (min)")
    plt.ylabel("MAE (Mbit/s)")
    plt.title("UGR'16 April — representantes estadísticos — MAE frente a horizonte")
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)
    return savefig(base, dpi)


def plot_final_skill(final_metrics: pd.DataFrame, base: Path, dpi: int) -> list[Path]:
    plt.figure(figsize=(10, 6))
    for config, sub in final_metrics.groupby("config", sort=False):
        sub = sub.sort_values("horizon_minutes")
        plt.plot(
            sub["horizon_minutes"], 100.0 * sub["skill_vs_persistence"],
            marker="o", linewidth=1.2, label=config
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Horizonte (min)")
    plt.ylabel("Skill frente a persistencia (%)")
    plt.title("UGR'16 April — representantes estadísticos — skill por horizonte")
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)
    return savefig(base, dpi)


def plot_heatmap(final_metrics: pd.DataFrame, base: Path, dpi: int) -> list[Path]:
    pivot = final_metrics.pivot(
        index="config", columns="horizon_minutes", values="mae_bps"
    ) / 1e6
    pivot = pivot.reindex(columns=[5, 15, 30, 60])
    plt.figure(figsize=(8.5, 5.2))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(image, label="MAE (Mbit/s)")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center")
    plt.xlabel("Horizonte (min)")
    plt.ylabel("Representante")
    plt.title("UGR'16 April — MAE: modelo × horizonte")
    return savefig(base, dpi)


def plot_final_scores(final_reps: pd.DataFrame, base: Path, dpi: int) -> list[Path]:
    ordered = final_reps.sort_values("score")
    plt.figure(figsize=(9, 5.4))
    plt.bar(ordered["config"], ordered["score"])
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.ylabel("Score medio MAE / persistencia")
    plt.xlabel("Representante")
    plt.title("UGR'16 April — representantes finales — Score")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return savefig(base, dpi)


def plot_winner_observed(
    pred: pd.DataFrame,
    winner_config: str,
    base: Path,
    dpi: int,
) -> list[Path]:
    sub = pred[
        (pred["config"] == winner_config)
        & (pred["horizon_steps"] == 1)
    ].sort_values("target_timestamp")
    plt.figure(figsize=(13, 5.8))
    plt.plot(pd.to_datetime(sub["target_timestamp"]), sub["y_true"] / 1e6, label="Observado")
    plt.plot(pd.to_datetime(sub["target_timestamp"]), sub["y_pred"] / 1e6, label="Predicho")
    plt.xlabel("Tiempo")
    plt.ylabel("Bitrate (Mbit/s)")
    plt.title(
        f"UGR'16 April — {winner_config} — observado vs predicho — H1 (5 min)"
    )
    plt.legend()
    plt.grid(True, alpha=0.25)
    return savefig(base, dpi)


def checkpoint_manifest_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_dir / "checkpoint_manifest.json"


def write_checkpoint_manifest(
    args: argparse.Namespace,
    context: dict[str, Any],
    completed: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for config, payload in sorted(completed.items()):
        worker_dir = args.checkpoint_dir / safe_slug(payload["candidate"])
        result_path = worker_dir / "result.json"
        rows.append({
            "config": config,
            "status": payload["summary"]["status"],
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
            "predictions": payload.get("predictions"),
        })
    atomic_write_json(
        checkpoint_manifest_path(args),
        {
            **context,
            "updated_at": now_iso(),
            "completed_count": len(rows),
            "completed_candidates": rows,
        },
    )


def load_completed(
    args: argparse.Namespace,
    context: dict[str, Any],
    runner_sha: str,
    input_sha: str,
) -> dict[str, dict[str, Any]]:
    path = checkpoint_manifest_path(args)
    if not path.is_file():
        raise FileNotFoundError("--resume requested but checkpoint manifest is missing.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "run_id", "runner_sha256", "input_sha256", "protocol_sha256",
        "amendment_007_sha256", "source_manifest_sha256"
    ):
        if saved.get(key) != context.get(key):
            raise RuntimeError(f"Resume context mismatch: {key}")
    completed = {}
    for candidate in CANDIDATES:
        worker_dir = args.checkpoint_dir / safe_slug(candidate)
        result_path = worker_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        completed[candidate["config"]] = validate_checkpoint(
            payload, candidate, runner_sha, input_sha, worker_dir
        )
    return completed


def main_parent(args: argparse.Namespace) -> int:
    runner_sha = current_runner_sha()
    log("=" * 112)
    log("UGR'16 APRIL — FINAL STATISTICAL SELECTION")
    log("=" * 112)
    log(f"Campaign: {CAMPAIGN_ID}")
    log(f"Run:      {RUN_ID}")
    log(f"Runner SHA-256: {runner_sha}")
    log("Families: AR / MA / ARMA / ARIMA / VAR")
    log("SARIMA: scope-resolved; NO April execution")
    log("June: NO")
    log("R3: NO refit")
    log()

    authorities = {
        "protocol": validate_hash(
            args.protocol_file, EXPECTED_PROTOCOL_SHA256, "Protocol 003"
        ),
        "amendment_004": validate_hash(
            args.amendment_004_file, EXPECTED_AMENDMENT_004_SHA256, "Amendment 004"
        ),
        "amendment_005": validate_hash(
            args.amendment_005_file, EXPECTED_AMENDMENT_005_SHA256, "Amendment 005"
        ),
        "record_006": validate_hash(
            args.record_006_file, EXPECTED_RECORD_006_SHA256, "Record 006"
        ),
        "amendment_007": validate_hash(
            args.amendment_007_file, EXPECTED_AMENDMENT_007_SHA256, "Amendment 007"
        ),
    }
    source_hashes = validate_source_manifests(args)
    validate_candidate_plan()
    frame, input_sha = load_input(args.input)
    train, validation = split_frame(frame)
    scale = mase_scale(train[TARGET].to_numpy(dtype=float))

    log("PRECHECK")
    log("-" * 112)
    for key, value in authorities.items():
        log(f"{key:16s}: {value}")
    for key, value in source_hashes.items():
        log(f"March {key:10s}: {value}")
    log(f"April input     : {input_sha}")
    log(f"Rows: total={len(frame)} train={len(train)} validation={len(validation)}")
    log(f"Origin_0: {train['timestamp'].max().isoformat()}")
    log("Expected coverage: H1=602 H3=600 H6=597 H12=591")
    log()
    log("CANDIDATE PLAN")
    log("-" * 112)
    for family in FAMILIES:
        log(f"{family:5s}: {EXPECTED_BY_FAMILY[family]}")

    if args.preflight_only:
        log()
        log("PRECHECK GLOBAL: PASS")
        log("No April candidate fitted (--preflight-only).")
        return 0

    context = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "runner_sha256": runner_sha,
        "input_sha256": input_sha,
        "protocol_sha256": authorities["protocol"],
        "amendment_007_sha256": authorities["amendment_007"],
        "source_manifest_sha256": source_hashes,
        "june_used": False,
    }

    if args.resume:
        completed = load_completed(args, context, runner_sha, input_sha)
        log(f"\nRESUME: recovered {len(completed)}/15 completed candidates.")
    else:
        if args.checkpoint_dir.exists():
            if args.overwrite:
                shutil.rmtree(args.checkpoint_dir)
            else:
                raise FileExistsError(
                    "Checkpoint exists. Use --resume or --overwrite."
                )
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, dict[str, Any]] = {}
        write_checkpoint_manifest(args, context, completed)

    log()
    log("APRIL GRID")
    log("-" * 112)

    for candidate in CANDIDATES:
        config = candidate["config"]
        if config in completed:
            log(
                f"SKIP {config:18s} checkpoint "
                f"status={completed[config]['summary']['status']}"
            )
            continue

        log(
            f"START {config:18s} family={candidate['family']:5s} "
            f"hard_timeout={RESOURCE_LIMIT_SECONDS}s"
        )
        outcome, payload = run_candidate(
            args, candidate, runner_sha, input_sha
        )
        if outcome == "TECHNICAL_ERROR":
            log(f"TECHNICAL_ERROR {config}: {payload}")
            write_checkpoint_manifest(args, context, completed)
            return 3

        worker_dir = args.checkpoint_dir / safe_slug(candidate)
        payload = validate_checkpoint(
            payload, candidate, runner_sha, input_sha, worker_dir
        )
        completed[config] = payload
        write_checkpoint_manifest(args, context, completed)
        s = payload["summary"]
        log(
            f"DONE  {config:18s} status={s['status']:24s} "
            f"attempts={s['attempts']} "
            f"fit={s['training_time_sec']} infer={s['inference_time_sec']}"
        )

    summaries = []
    pred_frames = []
    for candidate in CANDIDATES:
        payload = completed[candidate["config"]]
        summaries.append(payload["summary"])
        if payload["summary"]["status"] == STATUS_PASS:
            pred_frames.append(pd.read_parquet(payload["predictions"]["path"]))

    summary = pd.DataFrame(summaries)
    pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = compute_metrics(pred, scale) if not pred.empty else pd.DataFrame()
    summary = attach_score(summary, metrics)
    validate_common_timestamps(pred, summary)

    rankings = []
    for family in FAMILIES:
        r = rank_family(summary[summary["family"] == family])
        if not r.empty:
            rankings.append(r)
    ranking = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()

    winners = []
    missing_families = []
    for family in FAMILIES:
        r = ranking[ranking["family"] == family].sort_values("rank")
        if r.empty:
            missing_families.append(family)
        else:
            winners.append(r.iloc[0].to_dict())
    final_reps = pd.DataFrame(winners)

    overall_status = STATUS_PASS if not missing_families else STATUS_BLOCKED

    final_metrics = (
        metrics[metrics["config"].isin(final_reps["config"])].copy()
        if not final_reps.empty else pd.DataFrame()
    )
    cross_family = (
        final_reps[
            ["family", "config", "score", "training_time_sec", "inference_time_sec"]
        ].sort_values("score").reset_index(drop=True)
        if not final_reps.empty else pd.DataFrame()
    )
    if not cross_family.empty:
        cross_family["predictive_rank_by_score_only"] = range(
            1, len(cross_family) + 1
        )

    outputs = {
        "candidate_summary": args.metrics_dir / f"{args.prefix}_candidate_summary.csv",
        "metrics": args.metrics_dir / f"{args.prefix}_metrics.csv",
        "family_ranking": args.metrics_dir / f"{args.prefix}_family_ranking.csv",
        "final_representatives": args.metrics_dir / f"{args.prefix}_final_representatives.csv",
        "cross_family": args.metrics_dir / f"{args.prefix}_cross_family_summary.csv",
        "report": args.metrics_dir / f"{args.prefix}_report.txt",
        "manifest": args.metrics_dir / f"{args.prefix}_manifest.json",
        "predictions_parquet": args.predictions_dir / f"{args.prefix}_predictions.parquet",
        "predictions_csv": args.predictions_dir / f"{args.prefix}_predictions.csv",
    }
    for path in outputs.values():
        if path.exists() and not (args.overwrite or args.resume):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(outputs["candidate_summary"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    ranking.to_csv(outputs["family_ranking"], index=False)
    final_reps.to_csv(outputs["final_representatives"], index=False)
    cross_family.to_csv(outputs["cross_family"], index=False)
    pred.to_parquet(outputs["predictions_parquet"], index=False)
    pred.to_csv(outputs["predictions_csv"], index=False)

    figure_bases = {
        "candidate_scores": args.figures_dir / f"{args.prefix}_candidate_scores",
        "final_mae": args.figures_dir / f"{args.prefix}_final_mae_horizon",
        "final_skill": args.figures_dir / f"{args.prefix}_final_skill_horizon",
        "heatmap": args.figures_dir / f"{args.prefix}_final_mae_heatmap",
        "final_scores": args.figures_dir / f"{args.prefix}_final_scores",
        "winner_observed": args.figures_dir / f"{args.prefix}_winner_observed_h1",
    }
    figure_paths = []
    if not ranking.empty:
        figure_paths += plot_candidate_scores(
            ranking, figure_bases["candidate_scores"], args.dpi
        )
    if not final_metrics.empty:
        figure_paths += plot_final_mae(
            final_metrics, figure_bases["final_mae"], args.dpi
        )
        figure_paths += plot_final_skill(
            final_metrics, figure_bases["final_skill"], args.dpi
        )
        figure_paths += plot_heatmap(
            final_metrics, figure_bases["heatmap"], args.dpi
        )
    if not final_reps.empty:
        figure_paths += plot_final_scores(
            final_reps, figure_bases["final_scores"], args.dpi
        )
        best_config = cross_family.iloc[0]["config"]
        figure_paths += plot_winner_observed(
            pred, best_config, figure_bases["winner_observed"], args.dpi
        )

    report = [
        "=" * 112,
        "UGR'16 APRIL — FINAL STATISTICAL SELECTION",
        "=" * 112,
        f"Campaign: {CAMPAIGN_ID}",
        f"Run: {RUN_ID}",
        f"Runner SHA-256: {runner_sha}",
        f"Protocol SHA-256: {authorities['protocol']}",
        f"Amendment 007 SHA-256: {authorities['amendment_007']}",
        f"Input SHA-256: {input_sha}",
        "",
        "SCOPE",
        "-" * 112,
        "Predictive families: AR / MA / ARMA / ARIMA / VAR",
        "SARIMA_STATUS: NOT_EVALUABLE_WITHIN_FROZEN_RESOURCE_BUDGET",
        "SARIMA April execution: NO",
        "June used: NO",
        "",
        "SPLIT / R3",
        "-" * 112,
        f"Train: {len(train)}",
        f"Validation: {len(validation)}",
        f"Origin_0: {train['timestamp'].max().isoformat()}",
        "Parameter refit in validation: NO",
        "Observed history/state update: YES",
        "",
        "FAMILY WINNERS",
        "-" * 112,
    ]
    for _, row in final_reps.sort_values("family").iterrows():
        report.append(
            f"{row['family']}: {row['config']} "
            f"Score={row['score']:.8f} BIC={row['bic']}"
        )
    report += [
        "",
        "CROSS-FAMILY SCORE (DESCRIPTIVE; NO AIC/BIC ACROSS FAMILIES)",
        "-" * 112,
    ]
    for _, row in cross_family.iterrows():
        report.append(
            f"rank={int(row['predictive_rank_by_score_only'])} "
            f"{row['family']} {row['config']} Score={row['score']:.8f}"
        )
    report += [
        "",
        "GLOBAL RESULT",
        "-" * 112,
        f"STATUS: {overall_status}",
        f"Missing families: {missing_families}",
        "Representatives are NOT YET documentary-FROZEN for June.",
        "Next step after review: freeze five representatives + SARIMA status.",
        "June used: NO",
        "",
    ]
    outputs["report"].write_text("\n".join(report), encoding="utf-8")

    artifact_paths = [
        outputs["candidate_summary"], outputs["metrics"],
        outputs["family_ranking"], outputs["final_representatives"],
        outputs["cross_family"], outputs["report"],
        outputs["predictions_parquet"], outputs["predictions_csv"],
        *figure_paths,
    ]

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": overall_status,
        "created_at": now_iso(),
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(Path(__file__).resolve()),
            "sha256": runner_sha,
            "worker_isolation": True,
            "hard_timeout_seconds_per_candidate": RESOURCE_LIMIT_SECONDS,
            "checkpoint_resume": True,
        },
        "authorities": authorities,
        "source_manifest_sha256": source_hashes,
        "input": {
            "path": str(args.input),
            "sha256": input_sha,
            "rows": len(frame),
        },
        "split": {
            "n_train": len(train),
            "n_validation": len(validation),
            "origin_0": train["timestamp"].max().isoformat(),
            "refit_validation": False,
            "observed_update": True,
        },
        "scope": {
            "predictive_families": list(FAMILIES),
            "sarima_status": "NOT_EVALUABLE_WITHIN_FROZEN_RESOURCE_BUDGET",
            "sarima_april_execution": False,
        },
        "candidate_plan": CANDIDATES,
        "selection": {
            "score": "mean_h(MAE_model_h / MAE_persistence_h)",
            "practical_tie_relative": PRACTICAL_TIE_REL,
            "tiebreakers_within_family": ["BIC", "n_params", "training_time_sec"],
            "cross_family_bic_used": False,
            "representatives_ready_for_review": (
                final_reps[["family", "config"]].to_dict("records")
                if not final_reps.empty else []
            ),
            "documentary_freeze_completed": False,
        },
        "horizons": {
            str(h): {
                "minutes": HORIZON_MINUTES[h],
                "expected_predictions": EXPECTED_COUNTS[h],
            }
            for h in HORIZONS
        },
        "mase_scale_bps": scale,
        "blindness": {
            "june_used": False,
            "june_used_for_selection": False,
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
        "artifacts": {
            str(path): sha256_file(path)
            for path in artifact_paths if path.is_file()
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log()
    log("RESULT")
    log("-" * 112)
    log(f"STATUS: {overall_status}")
    log(
        "Representatives ready for review: "
        + str(
            final_reps[["family", "config"]].to_dict("records")
            if not final_reps.empty else []
        )
    )
    log("SARIMA_STATUS=NOT_EVALUABLE_WITHIN_FROZEN_RESOURCE_BUDGET")
    log("Documentary freeze for June: NO — review required")
    log("June used: NO")
    return 0 if overall_status == STATUS_PASS else 2


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)
    return main_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
