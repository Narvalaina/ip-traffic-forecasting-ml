#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
Cribado March de ARIMA.

Autoridades:
- TFM-STAT-PROTOCOL-001 / v1.0 / FROZEN
- TFM-STAT-AMENDMENT-ARMA-NUMERICAL-FALLBACK-001 / v1.0 / FROZEN

Prerequisito:
- UGR16-STATISTICAL-ARMA-MARCH-002 = PASS

Candidatos autorizados:
se leen exclusivamente de:
results/metrics/ugr16_statistical_arma_march_v2_arima_seed.json

Con el estado actual son:
- ARIMA(24,1,3)
- ARIMA(6,1,12)
- ARIMA(12,1,6)

Política:
- March 5 min; train=512, validation=220.
- ARIMA(p,1,q), trend="n".
- enforce_stationarity=True
- enforce_invertibility=True
- R3: fit único, parámetros congelados, append(refit=False).
- H1/H3/H6/H12 desde el mismo origen.
- cobertura 100 %, mismos target timestamps.
- MAE principal + métricas congeladas.
- Score frente a persistencia.
- empate práctico 1 % -> BIC -> n_params -> training_time.
- máximo 3 candidatos pasan a April; si hay menos válidos, pasan los válidos.
- máximo 30 min/candidato.
- un único retry con el mismo orden y maxiter ampliado.
- no hay fallback predictivo adicional para ARIMA.

NO usa June.
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
RUN_ID = "UGR16-STATISTICAL-ARIMA-MARCH-001"
PREREQ_RUN_ID = "UGR16-STATISTICAL-ARMA-MARCH-002"

EXPECTED_PROTOCOL_SHA256 = (
    "031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699"
)
EXPECTED_AMENDMENT_SHA256 = (
    "03704a5a2e072e8f13a7296ef1623a8131755c809ccfb36e897dab195c5bea21"
)
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
SHORTLIST_MAX = 3
RESOURCE_LIMIT_SECONDS = 30 * 60
RETRY_MAXITER = 1000

STATUS_PASS = "PASS"
STATUS_INVALID_NUMERICAL = "INVALID_NUMERICAL"
STATUS_INVALID_COVERAGE = "INVALID_COVERAGE"
STATUS_INVALID_RESOURCE_LIMIT = "INVALID_RESOURCE_LIMIT"
STATUS_BLOCKED = "BLOCKED_NO_VALID_ARIMA"


class CandidateTimeoutError(TimeoutError):
    pass


@contextmanager
def candidate_timeout(seconds: int):
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
        description="Cribado March ARIMA para UGR16-STATISTICAL-MODELS-001."
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
        "--arma-manifest",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_arma_march_v2_manifest.json"
        ),
    )
    parser.add_argument(
        "--arima-seed",
        type=Path,
        default=Path(
            "results/metrics/"
            "ugr16_statistical_arma_march_v2_arima_seed.json"
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
        default=Path("results/figures/ugr16_statistical_arima_march"),
    )
    parser.add_argument(
        "--prefix",
        default="ugr16_statistical_arima_march",
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


def validate_prerequisites(
    manifest_path: Path,
    seed_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, int | str]]]:
    manifest, manifest_hash = load_json(manifest_path)
    seed, seed_hash = load_json(seed_path)

    checks = {
        "campaign": manifest.get("campaign_id") == CAMPAIGN_ID,
        "run": manifest.get("run_id") == PREREQ_RUN_ID,
        "pass": manifest.get("status") == "PASS",
        "protocol": (
            manifest.get("protocol", {}).get("sha256")
            == EXPECTED_PROTOCOL_SHA256
        ),
        "amendment": (
            manifest.get("amendment", {}).get("sha256")
            == EXPECTED_AMENDMENT_SHA256
        ),
        "input": (
            manifest.get("input", {}).get("sha256")
            == EXPECTED_MARCH_5MIN_SHA256
        ),
        "june_unused": (
            manifest.get("blindness", {}).get("june_used") is False
        ),
        "final_valid_arma": (
            manifest.get("fallback", {}).get("final_valid_arma") == 3
        ),
        "seed_status": seed.get("status") == "PASS",
        "seed_d": seed.get("d_candidates") == [1],
    }

    registered_seed_hash = None
    for artifact_path, artifact_hash in manifest.get("artifacts", {}).items():
        if Path(artifact_path).name == seed_path.name:
            registered_seed_hash = artifact_hash
            break

    checks["seed_registered"] = registered_seed_hash is not None
    if registered_seed_hash is not None:
        checks["seed_hash"] = seed_hash == registered_seed_hash

    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "ARMA prerequisite validation failed: " + ", ".join(failed)
        )

    pairs = seed.get("top_3_arma_pairs")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise RuntimeError("ARIMA seed must contain exactly 3 ARMA pairs.")

    candidates: list[dict[str, int | str]] = []
    for item in pairs:
        p = int(item["p"])
        q = int(item["q"])
        expected = f"ARIMA({p},1,{q})"
        if item.get("arima_config") != expected:
            raise RuntimeError(
                f"ARIMA seed config mismatch: {item.get('arima_config')} != {expected}"
            )
        candidates.append(
            {
                "p": p,
                "d": 1,
                "q": q,
                "config": expected,
                "source_arma": str(item["arma_config"]),
                "source_arma_rank": int(item["rank"]),
            }
        )

    return manifest, manifest_hash, seed, seed_hash, candidates


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
        raise ValueError(f"Missing target: {TARGET}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(f"Unexpected March row count: {len(frame)}")
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")
    values = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite target.")
    if (values < 0).any():
        raise ValueError("Negative target.")

    deltas = frame["timestamp"].diff().dropna()
    if not (deltas == EXPECTED_FREQUENCY).all():
        raise ValueError("March series is not exactly 5-minute continuous.")

    return frame, observed_hash


def split_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(frame) * TRAIN_FRACTION)
    if n_train != EXPECTED_TRAIN:
        raise RuntimeError(f"Unexpected train size: {n_train}")
    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:].copy()
    if len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError("Unexpected validation size.")
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise RuntimeError("Chronology violation.")
    return train, validation


def mase_scale_from_train(train_y: np.ndarray) -> float:
    scale = float(np.mean(np.abs(np.diff(train_y))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid MASE scale: {scale}")
    return scale


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
                order=(p, 1, q),
                trend="n",
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

        retvals = getattr(result, "mle_retvals", {})
        detail["mle_retvals"] = {
            str(key): (
                value.item() if isinstance(value, np.generic)
                else value.tolist() if isinstance(value, np.ndarray)
                else value
            )
            for key, value in retvals.items()
        }
        detail["converged"] = bool(retvals.get("converged", False))
        detail["finite_params"] = bool(
            np.isfinite(np.asarray(result.params, dtype=float)).all()
        )
        detail["elapsed_sec"] = float(time.perf_counter() - start)
        return result, detail

    except Exception as exc:
        detail["exception"] = f"{type(exc).__name__}: {exc}"
        detail["elapsed_sec"] = float(time.perf_counter() - start)
        return None, detail


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
) -> dict[str, Any]:
    residual = float(y_true - y_pred)
    return {
        "family": "ARIMA",
        "config": config,
        "p": p,
        "d": 1,
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


def evaluate_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    candidate: dict[str, int | str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    p = int(candidate["p"])
    q = int(candidate["q"])
    config = str(candidate["config"])
    total_start = time.perf_counter()

    train_index = pd.DatetimeIndex(
        train["timestamp"].to_numpy(), freq="5min"
    )
    validation_index = pd.DatetimeIndex(
        validation["timestamp"].to_numpy(), freq="5min"
    )
    train_series = pd.Series(
        train[TARGET].to_numpy(dtype=float),
        index=train_index,
        name=TARGET,
    )
    validation_series = pd.Series(
        validation[TARGET].to_numpy(dtype=float),
        index=validation_index,
        name=TARGET,
    )

    attempts: list[dict[str, Any]] = []
    final_result = None
    fit_time = 0.0

    try:
        with candidate_timeout(RESOURCE_LIMIT_SECONDS):
            for maxiter in (None, RETRY_MAXITER):
                result, detail = attempt_fit(
                    train_series=train_series,
                    p=p,
                    q=q,
                    maxiter=maxiter,
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
                        "family": "ARIMA",
                        "config": config,
                        "p": p,
                        "d": 1,
                        "q": q,
                        "source_arma": candidate["source_arma"],
                        "source_arma_rank": candidate["source_arma_rank"],
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_params": np.nan,
                        "converged": False,
                        "fit_attempts": len(attempts),
                        "warnings": " | ".join(
                            warning
                            for detail in attempts
                            for warning in detail.get("warnings", [])
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

            rows: list[dict[str, Any]] = []
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
                    current_result.forecast(steps=steps),
                    dtype=float,
                )
                if (
                    len(forecast_values) != steps
                    or not np.isfinite(forecast_values).all()
                ):
                    raise FloatingPointError(
                        "Invalid forecast length or non-finite forecast."
                    )

                for horizon in HORIZONS:
                    if horizon > steps:
                        continue
                    target_pos = k + horizon - 1
                    rows.append(
                        build_prediction_row(
                            config=config,
                            p=p,
                            q=q,
                            origin_timestamp=origin_timestamp,
                            target_timestamp=pd.Timestamp(
                                validation_series.index[target_pos]
                            ),
                            horizon=horizon,
                            y_true=float(validation_series.iloc[target_pos]),
                            y_pred=float(forecast_values[horizon - 1]),
                            persistence_pred=origin_value,
                        )
                    )

                # R3: incorporate only the newly observed real point,
                # preserving the fitted parameter vector.
                if k < len(validation_series) - 1:
                    new_observation = validation_series.iloc[k : k + 1]
                    current_result = current_result.append(
                        new_observation,
                        refit=False,
                    )

            inference_time = float(time.perf_counter() - inference_start)
            predictions = pd.DataFrame(rows)

            return (
                {
                    "family": "ARIMA",
                    "config": config,
                    "p": p,
                    "d": 1,
                    "q": q,
                    "source_arma": candidate["source_arma"],
                    "source_arma_rank": candidate["source_arma_rank"],
                    "aic": float(final_result.aic),
                    "bic": float(final_result.bic),
                    "n_params": int(len(final_result.params)),
                    "converged": True,
                    "fit_attempts": len(attempts),
                    "warnings": " | ".join(
                        warning
                        for detail in attempts
                        for warning in detail.get("warnings", [])
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
                "family": "ARIMA",
                "config": config,
                "p": p,
                "d": 1,
                "q": q,
                "source_arma": candidate["source_arma"],
                "source_arma_rank": candidate["source_arma_rank"],
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
                "family": "ARIMA",
                "config": config,
                "p": p,
                "d": 1,
                "q": q,
                "source_arma": candidate["source_arma"],
                "source_arma_rank": candidate["source_arma_rank"],
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


def validate_coverage(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[str, str]:
    if summary["status"] != STATUS_PASS:
        return summary["status"], ""

    if predictions.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."

    numeric = predictions[
        ["y_true", "y_pred", "persistence_pred"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf in predictions."

    problems: list[str] = []
    for horizon in HORIZONS:
        subset = predictions[
            predictions["horizon_steps"] == horizon
        ].sort_values("target_timestamp")

        if len(subset) != EXPECTED_COUNTS[horizon]:
            problems.append(
                f"H{horizon}: {len(subset)} != {EXPECTED_COUNTS[horizon]}"
            )
        if subset["target_timestamp"].duplicated().any():
            problems.append(f"H{horizon}: duplicate timestamps")

    if problems:
        return STATUS_INVALID_COVERAGE, " | ".join(problems)

    return STATUS_PASS, ""


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

    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"],
        sort=False,
    ):
        group = group.sort_values("target_timestamp")
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        persistence = group["persistence_pred"].to_numpy(dtype=float)

        residual = y_true - y_pred
        abs_error = np.abs(residual)
        mae = float(np.mean(abs_error))
        persistence_mae = float(np.mean(np.abs(y_true - persistence)))

        rows.append(
            {
                "family": "ARIMA",
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


def attach_scores(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()
    output["score"] = np.nan

    for idx, row in output.iterrows():
        if row["status"] != STATUS_PASS:
            continue

        subset = metrics[metrics["config"] == row["config"]]
        if set(subset["horizon_steps"].astype(int)) != set(HORIZONS):
            output.at[idx, "status"] = STATUS_INVALID_COVERAGE
            output.at[idx, "coverage_note"] = "Missing required horizon."
            continue

        ratios = (
            subset["mae_bps"].to_numpy(dtype=float)
            / subset["persistence_mae_bps"].to_numpy(dtype=float)
        )
        output.at[idx, "score"] = float(np.mean(ratios))

    return output


def deterministic_ranking(table: pd.DataFrame) -> pd.DataFrame:
    eligible = table[table["status"] == STATUS_PASS].copy()
    if eligible.empty:
        return eligible.assign(rank=pd.Series(dtype="Int64"))

    remaining = eligible.copy()
    ranked_parts: list[pd.DataFrame] = []
    next_rank = 1

    while not remaining.empty:
        best_score = float(remaining["score"].min())
        relative = (remaining["score"] - best_score) / best_score
        tie = remaining.loc[
            relative <= PRACTICAL_TIE_REL + 1e-15
        ].copy()

        tie = tie.sort_values(
            ["bic", "n_params", "training_time_sec", "config"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        tie["practical_tie_group_best_score"] = best_score
        tie["relative_to_group_best"] = (
            tie["score"] - best_score
        ) / best_score
        tie["rank"] = range(next_rank, next_rank + len(tie))

        ranked_parts.append(tie)
        next_rank += len(tie)
        remaining = remaining.drop(index=tie.index)

    return pd.concat(ranked_parts, ignore_index=True)


def validate_common_timestamps(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    eligible = summary[summary["status"] == STATUS_PASS]

    for horizon in HORIZONS:
        reference = None
        reference_config = None

        for _, row in eligible.iterrows():
            subset = predictions[
                (predictions["config"] == row["config"])
                & (predictions["horizon_steps"] == horizon)
            ].sort_values("target_timestamp")

            timestamps = tuple(
                pd.to_datetime(subset["target_timestamp"]).tolist()
            )

            if reference is None:
                reference = timestamps
                reference_config = row["config"]
            elif timestamps != reference:
                raise RuntimeError(
                    f"Timestamp mismatch H{horizon}: "
                    f"{row['config']} vs {reference_config}"
                )


def save_figure(base: Path, dpi: int) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def plot_score(
    ranking: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    if ranking.empty:
        return []

    ordered = ranking.sort_values("rank")
    plt.figure(figsize=(9, 5.5))
    plt.bar(ordered["config"], ordered["score"])
    plt.axhline(1.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Configuración")
    plt.ylabel("Score medio MAE / persistencia")
    plt.title("UGR'16 March — ARIMA — ranking por Score")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return save_figure(base, dpi)


def plot_mae(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
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
    plt.title("UGR'16 March — ARIMA — MAE frente a horizonte")
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def hash_artifacts(paths: list[Path]) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def main() -> int:
    args = parse_args()

    print("=" * 104)
    print("UGR'16 MARCH — CRIBADO ARIMA")
    print("=" * 104)
    print(f"Campaña: {CAMPAIGN_ID}")
    print(f"Run:     {RUN_ID}")
    print("June:    NO")
    print("d:       1")
    print("Refit:   NO — política R3")
    print()

    protocol_hash = validate_frozen_file(
        args.protocol_file,
        EXPECTED_PROTOCOL_SHA256,
        "Protocol",
    )
    amendment_hash = validate_frozen_file(
        args.amendment_file,
        EXPECTED_AMENDMENT_SHA256,
        "Amendment",
    )
    (
        arma_manifest,
        arma_manifest_hash,
        arima_seed,
        arima_seed_hash,
        candidates,
    ) = validate_prerequisites(
        args.arma_manifest,
        args.arima_seed,
    )

    frame, input_hash = load_series(args.input)
    train, validation = split_series(frame)
    mase_scale = mase_scale_from_train(
        train[TARGET].to_numpy(dtype=float)
    )

    print("PRECHECK")
    print("-" * 104)
    print(f"Protocol SHA-256:      {protocol_hash}")
    print(f"Amendment SHA-256:     {amendment_hash}")
    print(f"ARMA manifest SHA-256: {arma_manifest_hash}")
    print(f"ARIMA seed SHA-256:    {arima_seed_hash}")
    print(f"Input SHA-256:         {input_hash}")
    print(
        f"Rows: total={len(frame)} "
        f"train={len(train)} validation={len(validation)}"
    )
    print("Candidates:", [item["config"] for item in candidates])
    print()

    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    print("ARIMA GRID")
    print("-" * 104)

    for candidate in candidates:
        summary, predictions = evaluate_candidate(
            train=train,
            validation=validation,
            candidate=candidate,
        )
        status, note = validate_coverage(summary, predictions)
        summary["status"] = status
        summary["coverage_note"] = note
        summaries.append(summary)

        if not predictions.empty:
            prediction_frames.append(predictions)

        inference = summary["inference_time_sec"]
        inference_text = (
            f"{inference:.3f}"
            if pd.notna(inference)
            else "nan"
        )
        print(
            f"{summary['config']:17s} "
            f"status={summary['status']:24s} "
            f"attempts={summary['fit_attempts']} "
            f"fit={summary['training_time_sec']:.3f} "
            f"infer={inference_text}"
        )

    summary_table = pd.DataFrame(summaries)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(
            columns=[
                "family", "config", "p", "d", "q",
                "origin_timestamp", "target_timestamp",
                "horizon_steps", "horizon_minutes",
                "y_true", "y_pred", "persistence_pred",
                "residual", "abs_error",
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
    valid_count = int(
        (summary_table["status"] == STATUS_PASS).sum()
    )

    overall_status = STATUS_PASS if valid_count >= 1 else STATUS_BLOCKED

    shortlist = ranking.head(SHORTLIST_MAX).copy()
    if not shortlist.empty:
        shortlist["shortlisted_for_april"] = True

    outputs = {
        "candidate_summary": (
            args.metrics_dir / f"{args.prefix}_candidate_summary.csv"
        ),
        "metrics": args.metrics_dir / f"{args.prefix}_metrics.csv",
        "ranking": args.metrics_dir / f"{args.prefix}_ranking.csv",
        "shortlist": args.metrics_dir / f"{args.prefix}_shortlist.csv",
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
    shortlist.to_csv(outputs["shortlist"], index=False)
    predictions.to_parquet(outputs["predictions_parquet"], index=False)
    predictions.to_csv(outputs["predictions_csv"], index=False)

    figure_paths: list[Path] = []
    if not ranking.empty:
        figure_paths += plot_score(
            ranking,
            figure_bases["score"],
            args.dpi,
        )
        figure_paths += plot_mae(
            metrics,
            ranking,
            figure_bases["mae"],
            args.dpi,
        )

    report_lines = [
        "=" * 104,
        "UGR'16 MARCH — CRIBADO ARIMA",
        "=" * 104,
        f"Campaign:                 {CAMPAIGN_ID}",
        f"Run:                      {RUN_ID}",
        f"Protocol SHA-256:         {protocol_hash}",
        f"Amendment SHA-256:        {amendment_hash}",
        f"ARMA manifest SHA-256:    {arma_manifest_hash}",
        f"ARIMA seed SHA-256:       {arima_seed_hash}",
        f"Input SHA-256:            {input_hash}",
        "",
        "R3 / SPLIT",
        "-" * 104,
        f"Train:                    {len(train)}",
        f"Validation:               {len(validation)}",
        f"Origin_0:                 {train['timestamp'].max().isoformat()}",
        "d:                        1",
        "trend:                    n",
        "Refit validation:         NO",
        "Observed-state update:    YES",
        "June used:               NO",
        "",
        "CANDIDATES",
        "-" * 104,
        *[str(item["config"]) for item in candidates],
        "",
        "EXECUTION",
        "-" * 104,
    ]

    for _, row in summary_table.iterrows():
        report_lines.append(
            f"{row['config']} "
            f"status={row['status']} "
            f"attempts={int(row['fit_attempts'])} "
            f"Score={row['score'] if pd.notna(row['score']) else 'NA'}"
        )

    report_lines += [
        "",
        "RANKING VALID ARIMA",
        "-" * 104,
    ]

    if ranking.empty:
        report_lines.append("NONE")
    else:
        for _, row in ranking.sort_values("rank").iterrows():
            report_lines.append(
                f"rank={int(row['rank'])} "
                f"{row['config']} "
                f"Score={row['score']:.8f} "
                f"BIC={row['bic']:.6f}"
            )

    report_lines += [
        "",
        "SHORTLIST APRIL",
        "-" * 104,
        str(
            shortlist.sort_values("rank")["config"].tolist()
            if not shortlist.empty else []
        ),
        "",
        "GLOBAL RESULT",
        "-" * 104,
        f"STATUS: {overall_status}",
        f"valid_ARIMA: {valid_count}",
        "June used: NO",
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
            "sha256": amendment_hash,
        },
        "prerequisite": {
            "arma_manifest_path": str(args.arma_manifest),
            "arma_manifest_sha256": arma_manifest_hash,
            "arima_seed_path": str(args.arima_seed),
            "arima_seed_sha256": arima_seed_hash,
            "arma_run_status": arma_manifest["status"],
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
        "model_policy": {
            "family": "ARIMA",
            "d": 1,
            "trend": "n",
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        "candidates": candidates,
        "selection": {
            "primary_metric": "MAE",
            "score": "mean_h(MAE_model_h / MAE_persistence_h)",
            "practical_tie_relative": PRACTICAL_TIE_REL,
            "tiebreakers": [
                "BIC",
                "n_params",
                "training_time_sec",
            ],
            "shortlist_max": SHORTLIST_MAX,
            "valid_count": valid_count,
        },
        "horizons": {
            str(h): {
                "minutes": HORIZON_MINUTES[h],
                "expected_predictions": EXPECTED_COUNTS[h],
            }
            for h in HORIZONS
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
    print(f"valid_ARIMA={valid_count}")
    print(
        "ranking=",
        ranking.sort_values("rank")["config"].tolist()
        if not ranking.empty else [],
    )
    print(
        "April shortlist=",
        shortlist.sort_values("rank")["config"].tolist()
        if not shortlist.empty else [],
    )
    print(f"STATUS: {overall_status}")

    return 0 if overall_status == STATUS_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
