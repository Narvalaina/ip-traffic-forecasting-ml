#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
Cribado March ARMA con ARMA-NUMERICAL-FALLBACK.

Autoridades:
- TFM-STAT-PROTOCOL-001 / v1.0 / FROZEN
- TFM-STAT-AMENDMENT-ARMA-NUMERICAL-FALLBACK-001 / v1.0 / FROZEN

El runner:
1) evalúa siempre los 4 ARMA originales top-2 x top-2;
2) si hay menos de 3 válidos, activa fallback top-3 x top-3;
3) prueba candidatos adicionales en orden determinista;
4) se detiene al obtener 3 válidos o agotar el pool;
5) guarda artefactos incluso si queda BLOCKED;
6) no usa June.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
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
from statsmodels.tsa.arima.model import ARIMA


CAMPAIGN_ID = "UGR16-STATISTICAL-MODELS-001"
RUN_ID = "UGR16-STATISTICAL-ARMA-MARCH-002"
PREREQ_RUN_ID = "UGR16-STATISTICAL-AR-MA-MARCH-001"
DIAGNOSTIC_ID = "UGR16-STATISTICAL-MARCH-DIAGNOSTICS-001"

EXPECTED_PROTOCOL_SHA256 = (
    "031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699"
)
EXPECTED_AMENDMENT_SHA256 = "03704a5a2e072e8f13a7296ef1623a8131755c809ccfb36e897dab195c5bea21"
EXPECTED_MARCH_5MIN_SHA256 = (
    "fd6daea6007f1b411617a5c67a489fa33c2efda35fdd13616cc66a40134fa1b1"
)

TARGET = "bitrate_bps"
TRAIN_FRACTION = 0.70
EXPECTED_TOTAL = 732
EXPECTED_TRAIN = 512
EXPECTED_VALIDATION = 220
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {h: EXPECTED_VALIDATION - (h - 1) for h in HORIZONS}

PRACTICAL_TIE_REL = 0.01
REQUIRED_VALID_ARMA = 3
RESOURCE_LIMIT_SECONDS = 30 * 60
RETRY_MAXITER = 1000

STATUS_PASS = "PASS"
STATUS_INVALID_NUMERICAL = "INVALID_NUMERICAL"
STATUS_INVALID_COVERAGE = "INVALID_COVERAGE"
STATUS_INVALID_RESOURCE_LIMIT = "INVALID_RESOURCE_LIMIT"
STATUS_BLOCKED = "BLOCKED_INSUFFICIENT_VALID_ARMA"


class CandidateTimeoutError(TimeoutError):
    pass


@contextmanager
def candidate_timeout(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)

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
        signal.signal(signal.SIGALRM, old_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="March ARMA con fallback numérico determinista."
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
        "--amendment-file",
        type=Path,
        default=Path(
            "docs/project_governance/"
            "004_arma_numerical_fallback_2026-08-15.md"
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
        "--ar-ma-manifest",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_ar_ma_march_manifest.json"
        ),
    )
    parser.add_argument(
        "--ar-ma-shortlist",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_ar_ma_march_shortlist.csv"
        ),
    )
    parser.add_argument(
        "--arma-seed",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_ar_ma_march_arma_seed.json"
        ),
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("results/metrics")
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("results/predictions")
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/ugr16_statistical_arma_march"),
    )
    parser.add_argument(
        "--prefix",
        default="ugr16_statistical_arma_march_v2",
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


def validate_frozen_file(path: Path, expected_sha: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(
            f"{label} SHA-256 mismatch.\n"
            f"Expected: {expected_sha}\n"
            f"Observed: {observed}"
        )
    return observed


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def validate_prerequisites(args: argparse.Namespace):
    diagnostic, diagnostic_hash = load_json(args.diagnostic_manifest)
    ar_ma, ar_ma_hash = load_json(args.ar_ma_manifest)
    seed, seed_hash = load_json(args.arma_seed)

    if not args.ar_ma_shortlist.is_file():
        raise FileNotFoundError(f"Missing shortlist: {args.ar_ma_shortlist}")
    shortlist_hash = sha256_file(args.ar_ma_shortlist)

    registered_shortlist_hash = None
    registered_seed_hash = None
    for artifact_path, artifact_hash in ar_ma.get("artifacts", {}).items():
        name = Path(artifact_path).name
        if name == args.ar_ma_shortlist.name:
            registered_shortlist_hash = artifact_hash
        if name == args.arma_seed.name:
            registered_seed_hash = artifact_hash

    checks = {
        "diagnostic_campaign": diagnostic.get("campaign_id") == CAMPAIGN_ID,
        "diagnostic_id": diagnostic.get("diagnostic_id") == DIAGNOSTIC_ID,
        "diagnostic_pass": diagnostic.get("status") == "PASS",
        "diagnostic_protocol": (
            diagnostic.get("protocol", {}).get("sha256")
            == EXPECTED_PROTOCOL_SHA256
        ),
        "diagnostic_june_unused": (
            diagnostic.get("blindness", {}).get("june_used") is False
        ),
        "diagnostic_d_is_1": (
            diagnostic.get("protocol_decisions_from_diagnostics", {})
            .get("arima_d_candidates_from_diagnostic") == [1]
        ),
        "ar_ma_run": ar_ma.get("run_id") == PREREQ_RUN_ID,
        "ar_ma_pass": ar_ma.get("status") == "PASS",
        "ar_ma_protocol": (
            ar_ma.get("protocol", {}).get("sha256")
            == EXPECTED_PROTOCOL_SHA256
        ),
        "ar_ma_june_unused": (
            ar_ma.get("blindness", {}).get("june_used") is False
        ),
        "shortlist_registered": registered_shortlist_hash is not None,
        "shortlist_hash": shortlist_hash == registered_shortlist_hash,
        "seed_registered": registered_seed_hash is not None,
        "seed_hash": seed_hash == registered_seed_hash,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(
            "Prerequisite validation failed: " + ", ".join(failed)
        )

    shortlist = pd.read_csv(args.ar_ma_shortlist)
    required = {"family", "config", "p", "q", "rank"}
    missing = required.difference(shortlist.columns)
    if missing:
        raise ValueError(f"Shortlist missing columns: {sorted(missing)}")

    return (
        diagnostic, diagnostic_hash,
        ar_ma, ar_ma_hash,
        shortlist, shortlist_hash,
        seed, seed_hash,
    )


def derive_candidate_plan(
    shortlist: pd.DataFrame,
    seed: dict[str, Any],
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    ar = (
        shortlist[shortlist["family"] == "AR"]
        .sort_values("rank")
        .head(3)
        .copy()
    )
    ma = (
        shortlist[shortlist["family"] == "MA"]
        .sort_values("rank")
        .head(3)
        .copy()
    )

    if len(ar) != 3 or len(ma) != 3:
        raise RuntimeError("Need top-3 AR and top-3 MA shortlists.")

    ar_rank = {int(row["p"]): int(row["rank"]) for _, row in ar.iterrows()}
    ma_rank = {int(row["q"]): int(row["rank"]) for _, row in ma.iterrows()}

    initial = [
        {
            "p": int(item["p"]),
            "q": int(item["q"]),
            "config": str(item["config"]),
            "ar_rank": ar_rank[int(item["p"])],
            "ma_rank": ma_rank[int(item["q"])],
        }
        for item in seed["arma_candidates"]
    ]
    if len(initial) != 4:
        raise RuntimeError("Initial ARMA seed must contain exactly 4 candidates.")

    initial_pairs = {(int(x["p"]), int(x["q"])) for x in initial}

    fallback = []
    for p, rp in ar_rank.items():
        for q, rq in ma_rank.items():
            if (p, q) in initial_pairs:
                continue
            fallback.append(
                {
                    "p": p,
                    "q": q,
                    "config": f"ARMA({p},{q})",
                    "ar_rank": rp,
                    "ma_rank": rq,
                    "rank_sum": rp + rq,
                    "rank_max": max(rp, rq),
                }
            )

    fallback.sort(
        key=lambda x: (
            int(x["rank_sum"]),
            int(x["rank_max"]),
            int(x["ar_rank"]),
            int(x["ma_rank"]),
        )
    )

    expected = [
        "ARMA(24,3)",
        "ARMA(6,12)",
        "ARMA(12,3)",
        "ARMA(6,6)",
        "ARMA(6,3)",
    ]
    observed = [str(x["config"]) for x in fallback]
    if observed != expected:
        raise RuntimeError(
            "Fallback ordering differs from frozen amendment.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )

    return initial, fallback


def load_series(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing March input: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_MARCH_5MIN_SHA256:
        raise RuntimeError(
            "March input SHA mismatch.\n"
            f"Expected: {EXPECTED_MARCH_5MIN_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        if frame.index.name == "timestamp":
            frame = frame.reset_index()
        else:
            raise ValueError("No timestamp column/index.")
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target: {TARGET}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(f"Unexpected March rows: {len(frame)}")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")
    y = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid target values.")
    if not (
        frame["timestamp"].diff().dropna() == EXPECTED_FREQUENCY
    ).all():
        raise ValueError("5-minute continuity failed.")

    return frame, observed_hash


def split_series(frame: pd.DataFrame):
    n_train = int(len(frame) * TRAIN_FRACTION)
    if n_train != EXPECTED_TRAIN:
        raise RuntimeError(f"Unexpected train size: {n_train}")
    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:].copy()
    if len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError("Unexpected validation size.")
    return train, validation


def mase_scale_from_train(train_y: np.ndarray) -> float:
    scale = float(np.mean(np.abs(np.diff(train_y))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid MASE scale: {scale}")
    return scale


def build_prediction_row(
    config: str,
    p: int,
    q: int,
    origin_timestamp: pd.Timestamp,
    target_timestamp: pd.Timestamp,
    horizon: int,
    y_true: float,
    y_pred: float,
    persistence_pred: float,
):
    residual = float(y_true - y_pred)
    return {
        "family": "ARMA",
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


def attempt_fit(
    train_series: pd.Series,
    p: int,
    q: int,
    maxiter: int | None,
) -> tuple[Any | None, dict[str, Any]]:
    detail: dict[str, Any] = {
        "maxiter": maxiter,
        "exception": None,
        "warnings": [],
        "converged": False,
        "finite_params": False,
        "mle_retvals": None,
        "elapsed_sec": None,
    }
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ARIMA(
                train_series,
                order=(p, 0, q),
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            if maxiter is None:
                result = model.fit()
            else:
                result = model.fit(method_kwargs={"maxiter": maxiter})
            detail["warnings"] = [
                f"{item.category.__name__}: {item.message}"
                for item in caught
            ]

        detail["mle_retvals"] = {
            str(k): (
                v.item() if isinstance(v, np.generic)
                else v.tolist() if isinstance(v, np.ndarray)
                else v
            )
            for k, v in getattr(result, "mle_retvals", {}).items()
        }
        detail["converged"] = bool(
            getattr(result, "mle_retvals", {}).get("converged", False)
        )
        detail["finite_params"] = bool(
            np.isfinite(np.asarray(result.params, dtype=float)).all()
        )
        detail["elapsed_sec"] = float(time.perf_counter() - start)
        return result, detail

    except Exception as exc:
        detail["exception"] = f"{type(exc).__name__}: {exc}"
        detail["elapsed_sec"] = float(time.perf_counter() - start)
        return None, detail


def evaluate_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    p: int,
    q: int,
    source_stage: str,
):
    config = f"ARMA({p},{q})"
    total_start = time.perf_counter()

    train_index = pd.DatetimeIndex(
        train["timestamp"].to_numpy(), freq="5min"
    )
    val_index = pd.DatetimeIndex(
        validation["timestamp"].to_numpy(), freq="5min"
    )
    train_series = pd.Series(
        train[TARGET].to_numpy(dtype=float),
        index=train_index,
        name=TARGET,
    )
    validation_series = pd.Series(
        validation[TARGET].to_numpy(dtype=float),
        index=val_index,
        name=TARGET,
    )

    attempts: list[dict[str, Any]] = []
    final_result = None
    fit_time = 0.0

    try:
        with candidate_timeout(RESOURCE_LIMIT_SECONDS):
            for maxiter in (None, RETRY_MAXITER):
                result, detail = attempt_fit(
                    train_series, p, q, maxiter=maxiter
                )
                attempts.append(detail)
                fit_time += float(detail["elapsed_sec"] or 0.0)

                if (
                    result is not None
                    and detail["converged"]
                    and detail["finite_params"]
                ):
                    final_result = result
                    break

            if final_result is None:
                return (
                    {
                        "family": "ARMA",
                        "config": config,
                        "p": p,
                        "q": q,
                        "d": 0,
                        "source_stage": source_stage,
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_params": np.nan,
                        "converged": False,
                        "fit_attempts": len(attempts),
                        "warnings": " | ".join(
                            w
                            for detail in attempts
                            for w in detail.get("warnings", [])
                        ),
                        "attempt_details": json.dumps(
                            attempts, ensure_ascii=False, default=str
                        ),
                        "training_time_sec": float(fit_time),
                        "inference_time_sec": np.nan,
                        "total_time_sec": float(
                            time.perf_counter() - total_start
                        ),
                        "status": STATUS_INVALID_NUMERICAL,
                        "coverage_note": "",
                    },
                    pd.DataFrame(),
                )

            rows = []
            current_result = final_result
            inference_start = time.perf_counter()

            for k in range(len(validation_series)):
                remaining = len(validation_series) - k
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
                    current_result.forecast(steps=steps), dtype=float
                )
                if (
                    len(forecast_values) != steps
                    or not np.isfinite(forecast_values).all()
                ):
                    raise FloatingPointError(
                        "Invalid forecast length or non-finite forecast."
                    )

                for h in HORIZONS:
                    if h > steps:
                        continue
                    target_pos = k + h - 1
                    rows.append(
                        build_prediction_row(
                            config=config,
                            p=p,
                            q=q,
                            origin_timestamp=origin_timestamp,
                            target_timestamp=pd.Timestamp(
                                validation_series.index[target_pos]
                            ),
                            horizon=h,
                            y_true=float(validation_series.iloc[target_pos]),
                            y_pred=float(forecast_values[h - 1]),
                            persistence_pred=origin_value,
                        )
                    )

                if k < len(validation_series) - 1:
                    new_obs = validation_series.iloc[k : k + 1]
                    current_result = current_result.append(
                        new_obs, refit=False
                    )

            inference_time = float(time.perf_counter() - inference_start)
            predictions = pd.DataFrame(rows)

            return (
                {
                    "family": "ARMA",
                    "config": config,
                    "p": p,
                    "q": q,
                    "d": 0,
                    "source_stage": source_stage,
                    "aic": float(final_result.aic),
                    "bic": float(final_result.bic),
                    "n_params": int(len(final_result.params)),
                    "converged": True,
                    "fit_attempts": len(attempts),
                    "warnings": " | ".join(
                        w
                        for detail in attempts
                        for w in detail.get("warnings", [])
                    ),
                    "attempt_details": json.dumps(
                        attempts, ensure_ascii=False, default=str
                    ),
                    "training_time_sec": float(fit_time),
                    "inference_time_sec": inference_time,
                    "total_time_sec": float(
                        time.perf_counter() - total_start
                    ),
                    "status": STATUS_PASS,
                    "coverage_note": "",
                },
                predictions,
            )

    except CandidateTimeoutError as exc:
        return (
            {
                "family": "ARMA",
                "config": config,
                "p": p,
                "q": q,
                "d": 0,
                "source_stage": source_stage,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": len(attempts),
                "warnings": str(exc),
                "attempt_details": json.dumps(
                    attempts, ensure_ascii=False, default=str
                ),
                "training_time_sec": float(fit_time),
                "inference_time_sec": np.nan,
                "total_time_sec": float(
                    time.perf_counter() - total_start
                ),
                "status": STATUS_INVALID_RESOURCE_LIMIT,
                "coverage_note": "",
            },
            pd.DataFrame(),
        )
    except Exception as exc:
        return (
            {
                "family": "ARMA",
                "config": config,
                "p": p,
                "q": q,
                "d": 0,
                "source_stage": source_stage,
                "aic": np.nan,
                "bic": np.nan,
                "n_params": np.nan,
                "converged": False,
                "fit_attempts": len(attempts),
                "warnings": f"{type(exc).__name__}: {exc}",
                "attempt_details": json.dumps(
                    attempts, ensure_ascii=False, default=str
                ),
                "training_time_sec": float(fit_time),
                "inference_time_sec": np.nan,
                "total_time_sec": float(
                    time.perf_counter() - total_start
                ),
                "status": STATUS_INVALID_NUMERICAL,
                "coverage_note": "",
            },
            pd.DataFrame(),
        )


def validate_coverage(summary: dict[str, Any], predictions: pd.DataFrame):
    if summary["status"] != STATUS_PASS:
        return summary["status"], summary.get("coverage_note", "")
    if predictions.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."

    if not np.isfinite(
        predictions[["y_true", "y_pred", "persistence_pred"]]
        .to_numpy(dtype=float)
    ).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf."

    problems = []
    for h in HORIZONS:
        subset = predictions[
            predictions["horizon_steps"] == h
        ].sort_values("target_timestamp")
        if len(subset) != EXPECTED_COUNTS[h]:
            problems.append(
                f"H{h}: {len(subset)} != {EXPECTED_COUNTS[h]}"
            )
        if subset["target_timestamp"].duplicated().any():
            problems.append(f"H{h} duplicate timestamps")
    if problems:
        return STATUS_INVALID_COVERAGE, " | ".join(problems)
    return STATUS_PASS, ""


def smape_percent(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    contrib = np.zeros_like(denominator, dtype=float)
    mask = denominator != 0
    contrib[mask] = (
        2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    )
    return float(100.0 * np.mean(contrib))


def compute_metrics(predictions: pd.DataFrame, mase_scale: float):
    rows = []
    for (config, h), group in predictions.groupby(
        ["config", "horizon_steps"], sort=False
    ):
        group = group.sort_values("target_timestamp")
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        persistence = group["persistence_pred"].to_numpy(dtype=float)

        residual = y_true - y_pred
        abs_error = np.abs(residual)
        mae = float(np.mean(abs_error))
        p_mae = float(np.mean(np.abs(y_true - persistence)))

        rows.append(
            {
                "family": "ARMA",
                "config": config,
                "horizon_steps": int(h),
                "horizon_minutes": HORIZON_MINUTES[int(h)],
                "n_predictions": len(group),
                "expected_predictions": EXPECTED_COUNTS[int(h)],
                "coverage": len(group) / EXPECTED_COUNTS[int(h)],
                "mae_bps": mae,
                "rmse_bps": float(np.sqrt(np.mean(residual ** 2))),
                "smape_pct": smape_percent(y_true, y_pred),
                "mase": float(mae / mase_scale),
                "persistence_mae_bps": p_mae,
                "skill_vs_persistence": float(1.0 - mae / p_mae),
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


def attach_scores(summary: pd.DataFrame, metrics: pd.DataFrame):
    out = summary.copy()
    out["score"] = np.nan
    for idx, row in out.iterrows():
        if row["status"] != STATUS_PASS:
            continue
        subset = metrics[metrics["config"] == row["config"]]
        if set(subset["horizon_steps"].astype(int)) != set(HORIZONS):
            out.at[idx, "status"] = STATUS_INVALID_COVERAGE
            out.at[idx, "coverage_note"] = "Missing horizon."
            continue
        ratios = (
            subset["mae_bps"].to_numpy(dtype=float)
            / subset["persistence_mae_bps"].to_numpy(dtype=float)
        )
        out.at[idx, "score"] = float(np.mean(ratios))
    return out


def deterministic_ranking(table: pd.DataFrame):
    eligible = table[table["status"] == STATUS_PASS].copy()
    if eligible.empty:
        return eligible.assign(rank=pd.Series(dtype="Int64"))

    remaining = eligible.copy()
    parts = []
    next_rank = 1

    while not remaining.empty:
        best_score = float(remaining["score"].min())
        relative = (remaining["score"] - best_score) / best_score
        tie = remaining.loc[
            relative <= PRACTICAL_TIE_REL + 1e-15
        ].copy()
        tie = tie.sort_values(
            ["bic", "n_params", "training_time_sec", "config"],
            kind="stable",
        )
        tie["practical_tie_group_best_score"] = best_score
        tie["relative_to_group_best"] = (
            tie["score"] - best_score
        ) / best_score
        tie["rank"] = range(next_rank, next_rank + len(tie))
        parts.append(tie)
        next_rank += len(tie)
        remaining = remaining.drop(index=tie.index)

    return pd.concat(parts, ignore_index=True)


def validate_common_timestamps(predictions, summary):
    eligible = summary[summary["status"] == STATUS_PASS]
    for h in HORIZONS:
        ref = None
        ref_name = None
        for _, row in eligible.iterrows():
            subset = predictions[
                (predictions["config"] == row["config"])
                & (predictions["horizon_steps"] == h)
            ].sort_values("target_timestamp")
            stamps = tuple(
                pd.to_datetime(subset["target_timestamp"]).tolist()
            )
            if ref is None:
                ref = stamps
                ref_name = row["config"]
            elif stamps != ref:
                raise RuntimeError(
                    f"Timestamp mismatch H{h}: "
                    f"{row['config']} vs {ref_name}"
                )


def save_figure(base: Path, dpi: int):
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def plot_score(ranking, base, dpi):
    if ranking.empty:
        return []
    ordered = ranking.sort_values("rank")
    plt.figure(figsize=(10, 5.5))
    plt.bar(ordered["config"], ordered["score"])
    plt.axhline(1.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Configuración")
    plt.ylabel("Score medio MAE / persistencia")
    plt.title("UGR'16 March — ARMA — ranking por Score")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return save_figure(base, dpi)


def plot_mae(metrics, ranking, base, dpi):
    if ranking.empty:
        return []
    plt.figure(figsize=(10, 5.8))
    for config in ranking.sort_values("rank")["config"].tolist():
        subset = metrics[
            metrics["config"] == config
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
    plt.title("UGR'16 March — ARMA — MAE frente a horizonte")
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def hash_artifacts(paths):
    return {
        str(path): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def main() -> int:
    args = parse_args()

    print("=" * 104)
    print("UGR'16 MARCH — CRIBADO ARMA v2 + NUMERICAL FALLBACK")
    print("=" * 104)
    print(f"Campaña:  {CAMPAIGN_ID}")
    print(f"Run:      {RUN_ID}")
    print("June:     NO")
    print("Refit:    NO — política R3")
    print()

    protocol_hash = validate_frozen_file(
        args.protocol_file, EXPECTED_PROTOCOL_SHA256, "Protocol"
    )
    amendment_hash = validate_frozen_file(
        args.amendment_file, EXPECTED_AMENDMENT_SHA256, "Amendment"
    )
    (
        diagnostic, diagnostic_hash,
        ar_ma_manifest, ar_ma_manifest_hash,
        shortlist, shortlist_hash,
        seed, seed_hash,
    ) = validate_prerequisites(args)

    initial_plan, fallback_plan = derive_candidate_plan(shortlist, seed)

    frame, input_hash = load_series(args.input)
    train, validation = split_series(frame)
    mase_scale = mase_scale_from_train(
        train[TARGET].to_numpy(dtype=float)
    )

    print("PRECHECK")
    print("-" * 104)
    print(f"Protocol SHA-256:    {protocol_hash}")
    print(f"Amendment SHA-256:   {amendment_hash}")
    print(f"Diagnostic SHA-256:  {diagnostic_hash}")
    print(f"AR+MA manifest SHA:  {ar_ma_manifest_hash}")
    print(f"Shortlist SHA-256:   {shortlist_hash}")
    print(f"ARMA seed SHA-256:   {seed_hash}")
    print(f"Input SHA-256:       {input_hash}")
    print(
        f"Rows: total={len(frame)} "
        f"train={len(train)} validation={len(validation)}"
    )
    print("Initial:", [x["config"] for x in initial_plan])
    print("Fallback:", [x["config"] for x in fallback_plan])
    print()

    summaries = []
    prediction_frames = []
    fallback_trace = []

    def run_one(item, stage):
        p, q = int(item["p"]), int(item["q"])
        summary, preds = evaluate_candidate(
            train, validation, p, q, source_stage=stage
        )
        status, note = validate_coverage(summary, preds)
        summary["status"] = status
        summary["coverage_note"] = note
        summaries.append(summary)
        if not preds.empty:
            prediction_frames.append(preds)

        print(
            f"{summary['config']:14s} "
            f"stage={stage:8s} "
            f"status={summary['status']:24s} "
            f"attempts={summary['fit_attempts']} "
            f"fit={summary['training_time_sec']:.3f} "
            f"infer={summary['inference_time_sec'] if np.isfinite(summary['inference_time_sec']) else float('nan'):.3f}"
        )

    print("INITIAL 2x2")
    print("-" * 104)
    for item in initial_plan:
        run_one(item, "initial")

    valid_count = sum(x["status"] == STATUS_PASS for x in summaries)
    fallback_triggered = valid_count < REQUIRED_VALID_ARMA
    fallback_stop_reason = "not_required"

    if fallback_triggered:
        print()
        print("NUMERICAL FALLBACK")
        print("-" * 104)
        print(
            f"Trigger: valid initial ARMA={valid_count} "
            f"< required={REQUIRED_VALID_ARMA}"
        )

        for item in fallback_plan:
            if valid_count >= REQUIRED_VALID_ARMA:
                fallback_stop_reason = "three_valid_arma_reached"
                break

            fallback_trace.append(
                {
                    "config": item["config"],
                    "p": int(item["p"]),
                    "q": int(item["q"]),
                    "ar_rank": int(item["ar_rank"]),
                    "ma_rank": int(item["ma_rank"]),
                    "rank_sum": int(item["rank_sum"]),
                    "rank_max": int(item["rank_max"]),
                }
            )
            run_one(item, "fallback")
            valid_count = sum(
                x["status"] == STATUS_PASS for x in summaries
            )

        if valid_count >= REQUIRED_VALID_ARMA:
            fallback_stop_reason = "three_valid_arma_reached"
        else:
            fallback_stop_reason = "fallback_pool_exhausted"

    summary_table = pd.DataFrame(summaries)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(
            columns=[
                "family","config","p","q","origin_timestamp",
                "target_timestamp","horizon_steps","horizon_minutes",
                "y_true","y_pred","persistence_pred","residual","abs_error"
            ]
        )
    )

    metrics = (
        compute_metrics(predictions, mase_scale)
        if not predictions.empty
        else pd.DataFrame()
    )
    summary_table = attach_scores(summary_table, metrics)

    if not predictions.empty:
        validate_common_timestamps(predictions, summary_table)

    ranking = deterministic_ranking(summary_table)
    valid_count = int((summary_table["status"] == STATUS_PASS).sum())

    overall_status = (
        STATUS_PASS
        if valid_count >= REQUIRED_VALID_ARMA
        else STATUS_BLOCKED
    )

    shortlist_out = ranking.head(REQUIRED_VALID_ARMA).copy()
    if not shortlist_out.empty:
        shortlist_out["shortlisted_for_april"] = (
            overall_status == STATUS_PASS
        )

    arima_seed = {
        "status": overall_status,
        "d_candidates": [1],
        "source_diagnostic": DIAGNOSTIC_ID,
        "top_3_arma_pairs": [],
    }
    if overall_status == STATUS_PASS:
        arima_seed["top_3_arma_pairs"] = [
            {
                "rank": int(row["rank"]),
                "p": int(row["p"]),
                "q": int(row["q"]),
                "arma_config": str(row["config"]),
                "arima_config": (
                    f"ARIMA({int(row['p'])},1,{int(row['q'])})"
                ),
            }
            for _, row in shortlist_out.sort_values("rank").iterrows()
        ]

    outputs = {
        "candidate_summary": (
            args.metrics_dir / f"{args.prefix}_candidate_summary.csv"
        ),
        "metrics": args.metrics_dir / f"{args.prefix}_metrics.csv",
        "ranking": args.metrics_dir / f"{args.prefix}_ranking.csv",
        "shortlist": args.metrics_dir / f"{args.prefix}_shortlist.csv",
        "fallback_trace": (
            args.metrics_dir / f"{args.prefix}_fallback_trace.json"
        ),
        "arima_seed": args.metrics_dir / f"{args.prefix}_arima_seed.json",
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
        "score": args.figures_dir / f"{args.prefix}_score",
        "mae": args.figures_dir / f"{args.prefix}_mae_horizon",
    }
    for base in figure_bases.values():
        for suffix in (".png", ".pdf"):
            ensure_output(base.with_suffix(suffix), args.overwrite)

    summary_table.to_csv(outputs["candidate_summary"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    ranking.to_csv(outputs["ranking"], index=False)
    shortlist_out.to_csv(outputs["shortlist"], index=False)
    outputs["fallback_trace"].write_text(
        json.dumps(
            {
                "triggered": fallback_triggered,
                "initial_valid_count": sum(
                    x["status"] == STATUS_PASS
                    for x in summaries
                    if x["source_stage"] == "initial"
                ),
                "attempted": fallback_trace,
                "stop_reason": fallback_stop_reason,
                "final_valid_count": valid_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outputs["arima_seed"].write_text(
        json.dumps(arima_seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_parquet(outputs["predictions_parquet"], index=False)
    predictions.to_csv(outputs["predictions_csv"], index=False)

    figure_paths = []
    if not ranking.empty:
        figure_paths += plot_score(
            ranking, figure_bases["score"], args.dpi
        )
        figure_paths += plot_mae(
            metrics, ranking, figure_bases["mae"], args.dpi
        )

    report = [
        "=" * 104,
        "UGR'16 MARCH — CRIBADO ARMA v2 + NUMERICAL FALLBACK",
        "=" * 104,
        f"Campaign:                 {CAMPAIGN_ID}",
        f"Run:                      {RUN_ID}",
        f"Protocol SHA-256:         {protocol_hash}",
        f"Amendment SHA-256:        {amendment_hash}",
        f"Input SHA-256:            {input_hash}",
        "",
        "INITIAL PLAN",
        "-" * 104,
        *[str(x["config"]) for x in initial_plan],
        "",
        "FALLBACK PLAN",
        "-" * 104,
        *[str(x["config"]) for x in fallback_plan],
        "",
        "EXECUTED CANDIDATES",
        "-" * 104,
    ]
    for _, row in summary_table.iterrows():
        report.append(
            f"{row['config']} "
            f"stage={row['source_stage']} "
            f"status={row['status']} "
            f"attempts={int(row['fit_attempts'])} "
            f"Score={row['score'] if pd.notna(row['score']) else 'NA'}"
        )

    report += [
        "",
        "FALLBACK",
        "-" * 104,
        f"triggered:                {fallback_triggered}",
        f"stop_reason:              {fallback_stop_reason}",
        f"valid_ARMA:               {valid_count}",
        "",
        "RANKING VALID ARMA",
        "-" * 104,
    ]
    if ranking.empty:
        report.append("NONE")
    else:
        for _, row in ranking.sort_values("rank").iterrows():
            report.append(
                f"rank={int(row['rank'])} "
                f"{row['config']} "
                f"Score={row['score']:.8f} "
                f"BIC={row['bic']:.6f}"
            )

    report += [
        "",
        "ARIMA SEED",
        "-" * 104,
    ]
    if arima_seed["top_3_arma_pairs"]:
        report += [
            item["arima_config"]
            for item in arima_seed["top_3_arma_pairs"]
        ]
    else:
        report.append("NOT GENERATED")

    report += [
        "",
        "GLOBAL RESULT",
        "-" * 104,
        f"STATUS: {overall_status}",
        "June used: NO",
        "",
    ]
    outputs["report"].write_text(
        "\n".join(report), encoding="utf-8"
    )

    artifact_paths = [
        outputs["candidate_summary"],
        outputs["metrics"],
        outputs["ranking"],
        outputs["shortlist"],
        outputs["fallback_trace"],
        outputs["arima_seed"],
        outputs["report"],
        outputs["predictions_parquet"],
        outputs["predictions_csv"],
        *figure_paths,
    ]

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": overall_status,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "sha256": protocol_hash,
            "state": "FROZEN",
        },
        "amendment": {
            "name": "TFM-STAT-AMENDMENT-ARMA-NUMERICAL-FALLBACK-001",
            "version": "1.0",
            "state": "FROZEN",
            "path": str(args.amendment_file),
            "sha256": amendment_hash,
        },
        "prerequisites": {
            "diagnostic_manifest_sha256": diagnostic_hash,
            "ar_ma_manifest_sha256": ar_ma_manifest_hash,
            "ar_ma_shortlist_sha256": shortlist_hash,
            "arma_seed_sha256": seed_hash,
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
        "initial_candidates": initial_plan,
        "fallback": {
            "triggered": fallback_triggered,
            "plan": fallback_plan,
            "attempted": fallback_trace,
            "stop_reason": fallback_stop_reason,
            "required_valid_arma": REQUIRED_VALID_ARMA,
            "final_valid_arma": valid_count,
        },
        "selection": {
            "primary_metric": "MAE",
            "score": "mean_h(MAE_model_h / MAE_persistence_h)",
            "practical_tie_relative": PRACTICAL_TIE_REL,
            "tiebreakers": [
                "BIC", "n_params", "training_time_sec"
            ],
            "arima_d": [1],
            "arima_seed": arima_seed,
        },
        "resource_policy": {
            "limit_seconds_per_candidate": RESOURCE_LIMIT_SECONDS,
            "single_retry_maxiter": RETRY_MAXITER,
        },
        "mase_scale_bps": mase_scale,
        "blindness": {
            "june_used": False,
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
    print("-" * 104)
    print(f"fallback_triggered={fallback_triggered}")
    print(f"fallback_stop_reason={fallback_stop_reason}")
    print(f"valid_ARMA={valid_count}")
    if not ranking.empty:
        print(
            "ranking=",
            ranking.sort_values("rank")["config"].tolist(),
        )
    print(
        "ARIMA candidates=",
        [
            x["arima_config"]
            for x in arima_seed["top_3_arma_pairs"]
        ],
    )
    print(f"STATUS: {overall_status}")

    return 0 if overall_status == STATUS_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
