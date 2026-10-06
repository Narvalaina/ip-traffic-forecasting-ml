#!/usr/bin/env python3
"""
UGR16-STATISTICAL-MODELS-001
VAR March — deterministic lag screen + R3 evaluation.

Frozen protocol:
- 003 / TFM-STAT-PROTOCOL-001 / v1.0 / FROZEN

Diagnostic prerequisite:
- UGR16-STATISTICAL-MARCH-DIAGNOSTICS-001 = PASS
- All three VAR endogenous variables classified nonstationary.
- Representation = first_differences.

Scientific design:
- Variables: bitrate_bps, packet_rate_pps, flow_rate_fps.
- Target evaluated: bitrate_bps.
- March split: 512 training / 220 validation.
- First differences of all three variables.
- Standardization fitted ONLY on transformed training.
- Lag selection: AIC/BIC/HQIC/FPE, maxlags=12.
- Candidate construction exactly per frozen protocol.
- VAR(p) via statsmodels.tsa.api.VAR.
- Stability mandatory.
- R3: fit once; no parameter refit; append observed vector history only.
- Multi-horizon H1/H3/H6/H12 from same origin/state.
- 100 % coverage.
- Score normalized vs persistence.
- Practical tie 1 % -> BIC -> number of parameters -> training time.
- Maximum 3 March candidates advance to April.
- June is never used.

Implementation note:
For standardization, sigma_train is the population standard deviation (ddof=0),
consistent with z=(x-mu_train)/sigma_train. This numerical convention is
recorded in the manifest and is not tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
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
from statsmodels.tsa.api import VAR


CAMPAIGN_ID = "UGR16-STATISTICAL-MODELS-001"
RUN_ID = "UGR16-STATISTICAL-VAR-MARCH-001"
DIAGNOSTIC_ID = "UGR16-STATISTICAL-MARCH-DIAGNOSTICS-001"

EXPECTED_PROTOCOL_SHA256 = (
    "031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699"
)
EXPECTED_DIAGNOSTIC_MANIFEST_SHA256 = (
    "1e7af397f25081d534a72debfd00f38d5d7738710d2b216cfebeaab71e200d62"
)
EXPECTED_MARCH_5MIN_SHA256 = (
    "fd6daea6007f1b411617a5c67a489fa33c2efda35fdd13616cc66a40134fa1b1"
)

VAR_COLUMNS = (
    "bitrate_bps",
    "packet_rate_pps",
    "flow_rate_fps",
)
TARGET = "bitrate_bps"
TARGET_INDEX = 0

TRAIN_FRACTION = 0.70
EXPECTED_TOTAL = 732
EXPECTED_TRAIN = 512
EXPECTED_VALIDATION = 220
EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)

MAXLAGS = 12
MAX_CANDIDATES = 6
SHORTLIST_MAX = 3

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {h: EXPECTED_VALIDATION - (h - 1) for h in HORIZONS}

PRACTICAL_TIE_REL = 0.01
RESOURCE_LIMIT_SECONDS = 30 * 60

STATUS_PASS = "PASS"
STATUS_INVALID_NUMERICAL = "INVALID_NUMERICAL"
STATUS_INVALID_STABILITY = "INVALID_STABILITY"
STATUS_INVALID_COVERAGE = "INVALID_COVERAGE"
STATUS_INVALID_RESOURCE_LIMIT = "INVALID_RESOURCE_LIMIT"
STATUS_BLOCKED = "BLOCKED_NO_VALID_VAR"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="March VAR deterministic lag screen for UGR16."
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
        default=Path("results/figures/ugr16_statistical_var_march"),
    )
    parser.add_argument(
        "--prefix",
        default="ugr16_statistical_var_march",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true")
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

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, sha256_file(path)


def validate_diagnostic_manifest(
    path: Path,
) -> tuple[dict[str, Any], str]:
    manifest, observed_hash = load_json(path)

    if observed_hash != EXPECTED_DIAGNOSTIC_MANIFEST_SHA256:
        raise RuntimeError(
            "Diagnostic manifest SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_DIAGNOSTIC_MANIFEST_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    decisions = manifest.get(
        "protocol_decisions_from_diagnostics",
        {},
    )

    level_classes = decisions.get(
        "var_level_classifications",
        {},
    )

    checks = {
        "campaign": manifest.get("campaign_id") == CAMPAIGN_ID,
        "diagnostic_id": (
            manifest.get("diagnostic_id") == DIAGNOSTIC_ID
        ),
        "status": manifest.get("status") == "PASS",
        "protocol": (
            manifest.get("protocol", {}).get("sha256")
            == EXPECTED_PROTOCOL_SHA256
        ),
        "input": (
            manifest.get("input", {}).get("sha256")
            == EXPECTED_MARCH_5MIN_SHA256
        ),
        "n_train": (
            manifest.get("split", {}).get("n_train")
            == EXPECTED_TRAIN
        ),
        "n_validation": (
            manifest.get("split", {}).get("n_validation")
            == EXPECTED_VALIDATION
        ),
        "diagnostics_use_validation": (
            manifest.get("split", {}).get(
                "diagnostics_use_validation"
            )
            is False
        ),
        "june_unused": (
            manifest.get("blindness", {}).get("june_used")
            is False
        ),
        "representation": (
            decisions.get("var_representation_from_protocol")
            == "first_differences"
        ),
        "bitrate_nonstationary": (
            level_classes.get("bitrate_bps")
            == "nonstationary"
        ),
        "packet_nonstationary": (
            level_classes.get("packet_rate_pps")
            == "nonstationary"
        ),
        "flow_nonstationary": (
            level_classes.get("flow_rate_fps")
            == "nonstationary"
        ),
    }

    failed = [name for name, ok in checks.items() if not ok]

    if failed:
        raise RuntimeError(
            "Diagnostic prerequisite validation failed: "
            + ", ".join(failed)
        )

    return manifest, observed_hash


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

    frame = pd.read_parquet(path).copy()

    if "timestamp" not in frame.columns:
        if frame.index.name == "timestamp":
            frame = frame.reset_index()
        else:
            raise ValueError("Input has no timestamp column/index.")

    required = {"timestamp", *VAR_COLUMNS}
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
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

    frame = frame.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(
            f"Unexpected March rows: {len(frame)}"
        )

    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps.")

    values = frame[list(VAR_COLUMNS)].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite VAR values.")

    if (values < 0).any():
        raise ValueError("Negative VAR level values.")

    deltas = frame["timestamp"].diff().dropna()

    if not (deltas == EXPECTED_FREQUENCY).all():
        raise ValueError(
            "March series is not exactly 5-minute continuous."
        )

    return frame, observed_hash


def split_series(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(frame) * TRAIN_FRACTION)

    if n_train != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Unexpected training size: {n_train}"
        )

    train = frame.iloc[:n_train].copy()
    validation = frame.iloc[n_train:].copy()

    if len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError(
            f"Unexpected validation size: {len(validation)}"
        )

    if train["timestamp"].max() >= validation["timestamp"].min():
        raise RuntimeError("Chronology violation.")

    return train, validation


def build_training_representation(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    levels = train.set_index("timestamp")[list(VAR_COLUMNS)].astype(float)

    # Frozen diagnostic decision: first difference ALL endogenous variables.
    diff = levels.diff().dropna()

    if len(diff) != EXPECTED_TRAIN - 1:
        raise RuntimeError(
            f"Unexpected transformed training length: {len(diff)}"
        )

    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=0)

    if not np.isfinite(mean.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite scaler mean.")

    if (
        not np.isfinite(std.to_numpy(dtype=float)).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid scaler std.")

    standardized = (diff - mean) / std

    if not np.isfinite(
        standardized.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite standardized training.")

    scaler = {
        "representation": "first_differences",
        "ddof": 0,
        "columns": list(VAR_COLUMNS),
        "mean": {
            col: float(mean[col])
            for col in VAR_COLUMNS
        },
        "std": {
            col: float(std[col])
            for col in VAR_COLUMNS
        },
        "n_transformed_train": int(len(diff)),
        "fit_scope": "training_only",
    }

    return standardized, scaler


def build_lag_criteria_table(
    selection: Any,
) -> pd.DataFrame:
    rows = []

    for lag in range(MAXLAGS + 1):
        rows.append(
            {
                "lag": lag,
                "aic": float(selection.ics["aic"][lag]),
                "bic": float(selection.ics["bic"][lag]),
                "hqic": float(selection.ics["hqic"][lag]),
                "fpe": float(selection.ics["fpe"][lag]),
                "selected_aic": (
                    lag == int(selection.selected_orders["aic"])
                ),
                "selected_bic": (
                    lag == int(selection.selected_orders["bic"])
                ),
                "selected_hqic": (
                    lag == int(selection.selected_orders["hqic"])
                ),
                "selected_fpe": (
                    lag == int(selection.selected_orders["fpe"])
                ),
            }
        )

    return pd.DataFrame(rows)


def append_unique(sequence: list[int], value: int) -> None:
    if 1 <= value <= MAXLAGS and value not in sequence:
        sequence.append(value)


def build_candidate_lags(
    selected_orders: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    selected = {
        "AIC": int(selected_orders["aic"]),
        "BIC": int(selected_orders["bic"]),
        "HQIC": int(selected_orders["hqic"]),
        "FPE": int(selected_orders["fpe"]),
    }

    # Step 1-5 of frozen protocol.
    initial: list[int] = []

    for criterion in ("AIC", "BIC", "HQIC", "FPE"):
        append_unique(initial, selected[criterion])

    # Step 6: p=1 mandatory.
    append_unique(initial, 1)

    # Step 7-9: p-1 and p+1 around selected lags.
    expanded = list(initial)

    for criterion in ("AIC", "BIC", "HQIC", "FPE"):
        lag = selected[criterion]
        append_unique(expanded, lag - 1)
        append_unique(expanded, lag + 1)

    if len(expanded) <= MAX_CANDIDATES:
        final = expanded
        truncation_applied = False

    else:
        truncation_applied = True
        final: list[int] = []

        # Frozen priority when >6:
        # BIC -> AIC -> HQIC -> FPE -> p=1 -> neighbours.
        for criterion in ("BIC", "AIC", "HQIC", "FPE"):
            append_unique(final, selected[criterion])

        append_unique(final, 1)

        # Deterministic engineering tie inside the "neighbours" class:
        # source criterion follows frozen priority; lower neighbour first.
        for criterion in ("BIC", "AIC", "HQIC", "FPE"):
            lag = selected[criterion]
            for neighbour in sorted((lag - 1, lag + 1)):
                append_unique(final, neighbour)
                if len(final) >= MAX_CANDIDATES:
                    break

            if len(final) >= MAX_CANDIDATES:
                break

        final = final[:MAX_CANDIDATES]

    if not final:
        final = [1]

    if len(final) > MAX_CANDIDATES:
        raise RuntimeError("VAR candidate count exceeds protocol maximum.")

    trace = {
        "selected_orders": selected,
        "lag_zero_selected": {
            criterion: bool(lag == 0)
            for criterion, lag in selected.items()
        },
        "initial_after_selected_and_p1": initial,
        "expanded_with_neighbours": expanded,
        "truncation_applied": truncation_applied,
        "final_candidates": final,
        "max_candidates": MAX_CANDIDATES,
        "maxlags": MAXLAGS,
    }

    return final, trace


def fit_var_candidate(
    train_z: pd.DataFrame,
    lag: int,
) -> tuple[Any | None, dict[str, Any]]:
    start = time.perf_counter()

    detail = {
        "lag": lag,
        "warnings": [],
        "exception": None,
        "fit_time_sec": None,
        "finite_params": False,
        "stable": False,
        "min_abs_root": np.nan,
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            result = VAR(train_z).fit(
                maxlags=lag,
                ic=None,
                trend="c",
            )

            detail["warnings"] = [
                f"{item.category.__name__}: {item.message}"
                for item in caught
            ]

        detail["fit_time_sec"] = float(
            time.perf_counter() - start
        )

        params = np.asarray(
            result.params,
            dtype=float,
        )

        detail["finite_params"] = bool(
            np.isfinite(params).all()
        )

        roots = np.asarray(
            result.roots,
            dtype=complex,
        )

        if roots.size:
            detail["min_abs_root"] = float(
                np.min(np.abs(roots))
            )

        detail["stable"] = bool(
            result.is_stable(verbose=False)
        )

        return result, detail

    except Exception as exc:
        detail["fit_time_sec"] = float(
            time.perf_counter() - start
        )
        detail["exception"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return None, detail


def inverse_standardize_forecast(
    forecast_z: np.ndarray,
    scaler: dict[str, Any],
) -> np.ndarray:
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

    return forecast_z * std + mean


def standardize_observed_difference(
    diff_vector: np.ndarray,
    scaler: dict[str, Any],
) -> np.ndarray:
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

    return (diff_vector - mean) / std


def evaluate_var_candidate(
    result: Any,
    lag: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_z: pd.DataFrame,
    scaler: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    history_z = train_z.to_numpy(dtype=float).copy()

    train_levels = train[list(VAR_COLUMNS)].to_numpy(dtype=float)
    validation_levels = validation[list(VAR_COLUMNS)].to_numpy(dtype=float)

    rows = []
    inference_start = time.perf_counter()

    for k in range(len(validation)):
        remaining = len(validation) - k
        steps = min(max(HORIZONS), remaining)

        if k == 0:
            origin_timestamp = pd.Timestamp(
                train["timestamp"].iloc[-1]
            )
            origin_level = train_levels[-1].copy()
        else:
            origin_timestamp = pd.Timestamp(
                validation["timestamp"].iloc[k - 1]
            )
            origin_level = validation_levels[k - 1].copy()

        if len(history_z) < lag:
            raise RuntimeError(
                f"History shorter than VAR lag {lag}."
            )

        forecast_z = np.asarray(
            result.forecast(
                y=history_z[-lag:],
                steps=steps,
            ),
            dtype=float,
        )

        if (
            forecast_z.shape != (steps, len(VAR_COLUMNS))
            or not np.isfinite(forecast_z).all()
        ):
            raise FloatingPointError(
                "Invalid standardized VAR forecast."
            )

        forecast_diff = inverse_standardize_forecast(
            forecast_z,
            scaler,
        )

        # Reconstruct levels from the last REAL level at the origin.
        cumulative_bitrate = (
            origin_level[TARGET_INDEX]
            + np.cumsum(
                forecast_diff[:, TARGET_INDEX]
            )
        )

        for horizon in HORIZONS:
            if horizon > steps:
                continue

            target_pos = k + horizon - 1
            y_true = float(
                validation_levels[
                    target_pos,
                    TARGET_INDEX,
                ]
            )
            y_pred = float(
                cumulative_bitrate[horizon - 1]
            )
            persistence = float(
                origin_level[TARGET_INDEX]
            )
            residual = y_true - y_pred

            rows.append(
                {
                    "family": "VAR",
                    "config": f"VAR({lag})",
                    "lag": lag,
                    "origin_timestamp": origin_timestamp,
                    "target_timestamp": pd.Timestamp(
                        validation["timestamp"].iloc[target_pos]
                    ),
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "persistence_pred": persistence,
                    "residual": residual,
                    "abs_error": abs(residual),
                }
            )

        # R3 update: after forecasting from origin t, observe the next REAL
        # multivariate vector and append only its transformed/scaled difference.
        if k < len(validation) - 1:
            if k == 0:
                previous_real = train_levels[-1]
            else:
                previous_real = validation_levels[k - 1]

            current_real = validation_levels[k]
            observed_diff = current_real - previous_real

            observed_z = standardize_observed_difference(
                observed_diff,
                scaler,
            )

            if not np.isfinite(observed_z).all():
                raise FloatingPointError(
                    "Non-finite observed standardized VAR update."
                )

            history_z = np.vstack(
                [history_z, observed_z]
            )

    inference_time = float(
        time.perf_counter() - inference_start
    )

    return pd.DataFrame(rows), inference_time


def validate_coverage(
    predictions: pd.DataFrame,
) -> tuple[str, str]:
    if predictions.empty:
        return STATUS_INVALID_COVERAGE, "No predictions."

    numeric = predictions[
        ["y_true", "y_pred", "persistence_pred"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        return STATUS_INVALID_COVERAGE, "NaN/Inf in predictions."

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


def smape_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    contribution = np.zeros_like(
        denominator,
        dtype=float,
    )

    mask = denominator != 0

    contribution[mask] = (
        2.0
        * np.abs(y_true[mask] - y_pred[mask])
        / denominator[mask]
    )

    return float(
        100.0 * np.mean(contribution)
    )


def mase_scale_from_train(
    train_y: np.ndarray,
) -> float:
    scale = float(
        np.mean(
            np.abs(
                np.diff(train_y)
            )
        )
    )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Invalid MASE scale: {scale}"
        )

    return scale


def compute_metrics(
    predictions: pd.DataFrame,
    mase_scale: float,
) -> pd.DataFrame:
    rows = []

    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"],
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

        residual = y_true - y_pred
        abs_error = np.abs(residual)

        mae = float(
            np.mean(abs_error)
        )

        persistence_mae = float(
            np.mean(
                np.abs(
                    y_true - persistence
                )
            )
        )

        rows.append(
            {
                "family": "VAR",
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "n_predictions": int(len(group)),
                "expected_predictions": EXPECTED_COUNTS[int(horizon)],
                "coverage": float(
                    len(group)
                    / EXPECTED_COUNTS[int(horizon)]
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
                    mae / mase_scale
                ),
                "persistence_mae_bps": persistence_mae,
                "skill_vs_persistence": float(
                    1.0
                    - mae / persistence_mae
                ),
                "mean_bias_bps": float(
                    np.mean(residual)
                ),
                "underprediction_pct": float(
                    100.0
                    * np.mean(
                        y_true > y_pred
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


def attach_scores(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()
    output["score"] = np.nan

    for idx, row in output.iterrows():
        if row["status"] != STATUS_PASS:
            continue

        subset = metrics[
            metrics["config"]
            == row["config"]
        ]

        if set(
            subset["horizon_steps"].astype(int)
        ) != set(HORIZONS):
            output.at[
                idx,
                "status",
            ] = STATUS_INVALID_COVERAGE

            output.at[
                idx,
                "coverage_note",
            ] = "Missing required horizon."

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
            idx,
            "score",
        ] = float(
            np.mean(ratios)
        )

    return output


def deterministic_ranking(
    table: pd.DataFrame,
) -> pd.DataFrame:
    eligible = table[
        table["status"] == STATUS_PASS
    ].copy()

    if eligible.empty:
        return eligible.assign(
            rank=pd.Series(dtype="Int64")
        )

    remaining = eligible.copy()
    ranked_parts = []
    next_rank = 1

    while not remaining.empty:
        best_score = float(
            remaining["score"].min()
        )

        relative = (
            remaining["score"]
            - best_score
        ) / best_score

        tie = remaining.loc[
            relative
            <= PRACTICAL_TIE_REL + 1e-15
        ].copy()

        tie = tie.sort_values(
            [
                "bic",
                "n_params",
                "training_time_sec",
                "config",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
            kind="stable",
        )

        tie[
            "practical_tie_group_best_score"
        ] = best_score

        tie[
            "relative_to_group_best"
        ] = (
            tie["score"]
            - best_score
        ) / best_score

        tie["rank"] = range(
            next_rank,
            next_rank + len(tie),
        )

        ranked_parts.append(tie)

        next_rank += len(tie)

        remaining = remaining.drop(
            index=tie.index
        )

    return pd.concat(
        ranked_parts,
        ignore_index=True,
    )


def validate_common_timestamps(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    eligible = summary[
        summary["status"] == STATUS_PASS
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
                    subset["target_timestamp"]
                ).tolist()
            )

            if reference is None:
                reference = timestamps
                reference_config = row["config"]
            elif timestamps != reference:
                raise RuntimeError(
                    f"Target timestamp mismatch H{horizon}: "
                    f"{row['config']} vs {reference_config}"
                )


def granger_table(
    result: Any,
    best_config: str,
) -> pd.DataFrame:
    tests = [
        (
            "packet_rate_pps -> bitrate_bps",
            ["packet_rate_pps"],
        ),
        (
            "flow_rate_fps -> bitrate_bps",
            ["flow_rate_fps"],
        ),
        (
            "packet_rate_pps + flow_rate_fps -> bitrate_bps",
            ["packet_rate_pps", "flow_rate_fps"],
        ),
    ]

    rows = []

    for label, causing in tests:
        try:
            test = result.test_causality(
                caused="bitrate_bps",
                causing=causing,
                kind="f",
                signif=0.05,
            )

            rows.append(
                {
                    "config": best_config,
                    "test": label,
                    "kind": "F",
                    "statistic": float(
                        test.test_statistic
                    ),
                    "pvalue": float(
                        test.pvalue
                    ),
                    "critical_value": float(
                        test.crit_value
                    ),
                    "df": str(
                        test.df
                    ),
                    "alpha": float(
                        test.signif
                    ),
                    "conclusion": str(
                        test.conclusion
                    ),
                    "used_for_selection": False,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "config": best_config,
                    "test": label,
                    "kind": "F",
                    "statistic": np.nan,
                    "pvalue": np.nan,
                    "critical_value": np.nan,
                    "df": "",
                    "alpha": 0.05,
                    "conclusion": (
                        f"ERROR: {type(exc).__name__}: {exc}"
                    ),
                    "used_for_selection": False,
                }
            )

    return pd.DataFrame(rows)


def save_figure(
    base: Path,
    dpi: int,
) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")

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


def plot_information_criteria(
    criteria: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(figsize=(9.5, 5.6))

    for criterion in ("aic", "bic", "hqic"):
        plt.plot(
            criteria["lag"],
            criteria[criterion],
            marker="o",
            linewidth=1.1,
            label=criterion.upper(),
        )

    plt.xlabel("Lag VAR (intervalos de 5 min)")
    plt.ylabel("Criterio de información")
    plt.title(
        "UGR'16 March — VAR — selección de lag "
        "(AIC/BIC/HQIC)"
    )
    plt.xticks(range(0, MAXLAGS + 1))
    plt.legend()
    plt.grid(True, alpha=0.25)

    return save_figure(
        base,
        dpi,
    )


def plot_fpe(
    criteria: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(figsize=(9.5, 5.4))

    plt.plot(
        criteria["lag"],
        criteria["fpe"],
        marker="o",
        linewidth=1.1,
    )

    plt.xlabel("Lag VAR (intervalos de 5 min)")
    plt.ylabel("FPE")
    plt.title(
        "UGR'16 March — VAR — Final Prediction Error"
    )
    plt.xticks(range(0, MAXLAGS + 1))
    plt.grid(True, alpha=0.25)

    return save_figure(
        base,
        dpi,
    )


def plot_mae(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    if ranking.empty:
        return []

    plt.figure(figsize=(10, 5.8))

    for config in ranking.sort_values(
        "rank"
    )["config"].tolist():
        subset = metrics[
            metrics["config"]
            == config
        ].sort_values(
            "horizon_minutes"
        )

        plt.plot(
            subset["horizon_minutes"],
            subset["mae_bps"] / 1e6,
            marker="o",
            linewidth=1.1,
            label=config,
        )

    plt.xlabel("Horizonte (min)")
    plt.ylabel("MAE (Mbit/s)")
    plt.title(
        "UGR'16 March — VAR — MAE frente a horizonte"
    )
    plt.xticks([5, 15, 30, 60])
    plt.legend()
    plt.grid(True, alpha=0.25)

    return save_figure(
        base,
        dpi,
    )


def plot_score(
    ranking: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    if ranking.empty:
        return []

    ordered = ranking.sort_values(
        "rank"
    )

    plt.figure(figsize=(9.5, 5.5))

    plt.bar(
        ordered["config"],
        ordered["score"],
    )

    plt.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
    )

    plt.xlabel("Configuración")
    plt.ylabel("Score medio MAE / persistencia")
    plt.title(
        "UGR'16 March — VAR — ranking por Score"
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


def plot_stability(
    summary: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    eligible = summary[
        summary["min_abs_root"].notna()
    ].sort_values("lag")

    if eligible.empty:
        return []

    plt.figure(figsize=(9.5, 5.5))

    plt.bar(
        eligible["config"],
        eligible["min_abs_root"],
    )

    plt.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
    )

    plt.xlabel("Configuración")
    plt.ylabel("Mínimo |raíz característica|")
    plt.title(
        "UGR'16 March — VAR — estabilidad "
        "(estable si todas las raíces quedan fuera del círculo unidad)"
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


def hash_artifacts(
    paths: list[Path],
) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def main() -> int:
    args = parse_args()

    print("=" * 108)
    print("UGR'16 MARCH — VAR — DETERMINISTIC LAG SCREEN + R3")
    print("=" * 108)
    print(f"Campaign: {CAMPAIGN_ID}")
    print(f"Run:      {RUN_ID}")
    print("Representation: first_differences")
    print("Scaling: z=(x-mu_train)/sigma_train, ddof=0")
    print("maxlags: 12")
    print("June: NO")
    print("Refit: NO — R3")
    print()

    protocol_hash = validate_frozen_file(
        args.protocol_file,
        EXPECTED_PROTOCOL_SHA256,
        "Protocol 003",
    )

    diagnostic_manifest, diagnostic_hash = (
        validate_diagnostic_manifest(
            args.diagnostic_manifest
        )
    )

    frame, input_hash = load_series(
        args.input
    )

    train, validation = split_series(
        frame
    )

    train_z, scaler = (
        build_training_representation(
            train
        )
    )

    mase_scale = mase_scale_from_train(
        train[TARGET].to_numpy(
            dtype=float
        )
    )

    print("PRECHECK")
    print("-" * 108)
    print(f"Protocol SHA-256:   {protocol_hash}")
    print(f"Diagnostic SHA-256: {diagnostic_hash}")
    print(f"Input SHA-256:      {input_hash}")
    print(
        f"Rows: total={len(frame)} "
        f"train={len(train)} "
        f"validation={len(validation)}"
    )
    print(
        f"Transformed training rows: {len(train_z)}"
    )
    print(
        "VAR columns: "
        + ", ".join(VAR_COLUMNS)
    )
    print()

    # Lag selection is diagnostic and uses transformed/scaled training only.
    selection_start = time.perf_counter()

    with warnings.catch_warnings(record=True) as selection_warnings:
        warnings.simplefilter("always")

        selection = VAR(
            train_z
        ).select_order(
            maxlags=MAXLAGS,
            trend="c",
        )

    lag_selection_time = float(
        time.perf_counter()
        - selection_start
    )

    criteria = build_lag_criteria_table(
        selection
    )

    candidate_lags, lag_trace = (
        build_candidate_lags(
            selection.selected_orders
        )
    )

    print("LAG SELECTION")
    print("-" * 108)
    print(
        "Selected orders:",
        {
            key.upper(): int(value)
            for key, value
            in selection.selected_orders.items()
        },
    )
    print(
        "Candidate lags:",
        candidate_lags,
    )
    print(
        f"Lag selection time: "
        f"{lag_selection_time:.6f} s"
    )
    print()

    if args.preflight_only:
        print("PRECHECK GLOBAL: PASS")
        print("No VAR candidate fitted (--preflight-only).")
        return 0

    summaries = []
    prediction_frames = []
    fitted_results = {}

    print("VAR GRID")
    print("-" * 108)

    for lag in candidate_lags:
        total_start = time.perf_counter()

        result, fit_detail = fit_var_candidate(
            train_z=train_z,
            lag=lag,
        )

        status = STATUS_PASS
        coverage_note = ""
        predictions = pd.DataFrame()
        inference_time = np.nan

        if result is None:
            status = STATUS_INVALID_NUMERICAL

        elif not fit_detail["finite_params"]:
            status = STATUS_INVALID_NUMERICAL

        elif not fit_detail["stable"]:
            status = STATUS_INVALID_STABILITY

        elif (
            float(fit_detail["fit_time_sec"])
            > RESOURCE_LIMIT_SECONDS
        ):
            status = STATUS_INVALID_RESOURCE_LIMIT

        else:
            try:
                predictions, inference_time = (
                    evaluate_var_candidate(
                        result=result,
                        lag=lag,
                        train=train,
                        validation=validation,
                        train_z=train_z,
                        scaler=scaler,
                    )
                )

                coverage_status, coverage_note = (
                    validate_coverage(
                        predictions
                    )
                )

                status = coverage_status

            except Exception as exc:
                status = STATUS_INVALID_NUMERICAL
                coverage_note = (
                    f"{type(exc).__name__}: {exc}"
                )

        total_time = float(
            time.perf_counter()
            - total_start
        )

        if (
            total_time > RESOURCE_LIMIT_SECONDS
            and status == STATUS_PASS
        ):
            status = STATUS_INVALID_RESOURCE_LIMIT

        if (
            result is not None
            and status == STATUS_PASS
        ):
            fitted_results[lag] = result

        if not predictions.empty:
            prediction_frames.append(
                predictions
            )

        summary = {
            "family": "VAR",
            "config": f"VAR({lag})",
            "lag": lag,
            "representation": "first_differences",
            "trend": "c",
            "aic": (
                float(result.aic)
                if result is not None
                else np.nan
            ),
            "bic": (
                float(result.bic)
                if result is not None
                else np.nan
            ),
            "hqic": (
                float(result.hqic)
                if result is not None
                else np.nan
            ),
            "fpe": (
                float(result.fpe)
                if result is not None
                else np.nan
            ),
            "n_params": (
                int(
                    np.asarray(
                        result.params
                    ).size
                )
                if result is not None
                else np.nan
            ),
            "stable": bool(
                fit_detail["stable"]
            ),
            "min_abs_root": fit_detail[
                "min_abs_root"
            ],
            "finite_params": bool(
                fit_detail["finite_params"]
            ),
            "warnings": " | ".join(
                fit_detail["warnings"]
            ),
            "exception": fit_detail[
                "exception"
            ],
            "training_time_sec": fit_detail[
                "fit_time_sec"
            ],
            "inference_time_sec": inference_time,
            "total_time_sec": total_time,
            "status": status,
            "coverage_note": coverage_note,
        }

        summaries.append(
            summary
        )

        print(
            f"VAR({lag:<2d}) "
            f"status={status:24s} "
            f"stable={str(summary['stable']):5s} "
            f"min|root|="
            f"{summary['min_abs_root'] if pd.notna(summary['min_abs_root']) else float('nan'):.6f} "
            f"fit={summary['training_time_sec']:.4f} "
            f"infer="
            f"{summary['inference_time_sec'] if pd.notna(summary['inference_time_sec']) else float('nan'):.4f}"
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
            predictions=predictions,
            mase_scale=mase_scale,
        )
        if not predictions.empty
        else pd.DataFrame()
    )

    summary_table = attach_scores(
        summary=summary_table,
        metrics=metrics,
    )

    if not predictions.empty:
        validate_common_timestamps(
            predictions=predictions,
            summary=summary_table,
        )

    ranking = deterministic_ranking(
        summary_table
    )

    shortlist = ranking.head(
        SHORTLIST_MAX
    ).copy()

    if not shortlist.empty:
        shortlist[
            "shortlisted_for_april"
        ] = True

    valid_count = int(
        (
            summary_table["status"]
            == STATUS_PASS
        ).sum()
    )

    overall_status = (
        STATUS_PASS
        if valid_count >= 1
        else STATUS_BLOCKED
    )

    # Optional Granger evidence on the top-ranked March VAR.
    granger = pd.DataFrame()

    if not ranking.empty:
        best_lag = int(
            ranking.sort_values(
                "rank"
            ).iloc[0]["lag"]
        )

        if best_lag in fitted_results:
            granger = granger_table(
                result=fitted_results[
                    best_lag
                ],
                best_config=f"VAR({best_lag})",
            )

    outputs = {
        "lag_criteria": (
            args.metrics_dir
            / f"{args.prefix}_lag_criteria.csv"
        ),
        "lag_selection": (
            args.metrics_dir
            / f"{args.prefix}_lag_selection.json"
        ),
        "scaler": (
            args.metrics_dir
            / f"{args.prefix}_scaler.json"
        ),
        "candidate_summary": (
            args.metrics_dir
            / f"{args.prefix}_candidate_summary.csv"
        ),
        "metrics": (
            args.metrics_dir
            / f"{args.prefix}_metrics.csv"
        ),
        "ranking": (
            args.metrics_dir
            / f"{args.prefix}_ranking.csv"
        ),
        "shortlist": (
            args.metrics_dir
            / f"{args.prefix}_shortlist.csv"
        ),
        "granger": (
            args.metrics_dir
            / f"{args.prefix}_granger_best.csv"
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
        ensure_output(
            path,
            args.overwrite,
        )

    figure_bases = {
        "information_criteria": (
            args.figures_dir
            / f"{args.prefix}_lag_information_criteria"
        ),
        "fpe": (
            args.figures_dir
            / f"{args.prefix}_lag_fpe"
        ),
        "mae": (
            args.figures_dir
            / f"{args.prefix}_mae_horizon"
        ),
        "score": (
            args.figures_dir
            / f"{args.prefix}_score"
        ),
        "stability": (
            args.figures_dir
            / f"{args.prefix}_stability"
        ),
    }

    for base in figure_bases.values():
        for suffix in (".png", ".pdf"):
            ensure_output(
                base.with_suffix(suffix),
                args.overwrite,
            )

    criteria.to_csv(
        outputs["lag_criteria"],
        index=False,
    )

    lag_selection_payload = {
        "selected_orders": {
            key: int(value)
            for key, value
            in selection.selected_orders.items()
        },
        "trace": lag_trace,
        "selection_warnings": [
            f"{item.category.__name__}: {item.message}"
            for item in selection_warnings
        ],
        "lag_selection_time_sec": lag_selection_time,
        "used_validation": False,
        "used_june": False,
    }

    outputs["lag_selection"].write_text(
        json.dumps(
            lag_selection_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs["scaler"].write_text(
        json.dumps(
            scaler,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_table.to_csv(
        outputs["candidate_summary"],
        index=False,
    )

    metrics.to_csv(
        outputs["metrics"],
        index=False,
    )

    ranking.to_csv(
        outputs["ranking"],
        index=False,
    )

    shortlist.to_csv(
        outputs["shortlist"],
        index=False,
    )

    granger.to_csv(
        outputs["granger"],
        index=False,
    )

    predictions.to_parquet(
        outputs["predictions_parquet"],
        index=False,
    )

    predictions.to_csv(
        outputs["predictions_csv"],
        index=False,
    )

    figure_paths = []

    figure_paths += plot_information_criteria(
        criteria,
        figure_bases[
            "information_criteria"
        ],
        args.dpi,
    )

    figure_paths += plot_fpe(
        criteria,
        figure_bases["fpe"],
        args.dpi,
    )

    figure_paths += plot_mae(
        metrics,
        ranking,
        figure_bases["mae"],
        args.dpi,
    )

    figure_paths += plot_score(
        ranking,
        figure_bases["score"],
        args.dpi,
    )

    figure_paths += plot_stability(
        summary_table,
        figure_bases["stability"],
        args.dpi,
    )

    report_lines = [
        "=" * 108,
        "UGR'16 MARCH — VAR — DETERMINISTIC LAG SCREEN + R3",
        "=" * 108,
        f"Campaign:                 {CAMPAIGN_ID}",
        f"Run:                      {RUN_ID}",
        f"Protocol SHA-256:         {protocol_hash}",
        f"Diagnostic SHA-256:       {diagnostic_hash}",
        f"Input SHA-256:            {input_hash}",
        "",
        "REPRESENTATION",
        "-" * 108,
        "Endogenous:               bitrate_bps, packet_rate_pps, flow_rate_fps",
        "Target evaluated:         bitrate_bps",
        "Representation:           first_differences",
        "Scaling:                  training-only z-score",
        "Scaler std ddof:          0",
        f"Transformed train rows:    {len(train_z)}",
        "",
        "R3 / SPLIT",
        "-" * 108,
        f"Train levels:              {len(train)}",
        f"Validation levels:         {len(validation)}",
        f"Origin_0:                  {train['timestamp'].max().isoformat()}",
        "Parameter refit:          NO",
        "Observed vector update:   YES",
        "June used:                NO",
        "",
        "LAG SELECTION",
        "-" * 108,
        f"maxlags:                   {MAXLAGS}",
        f"AIC selected:              {int(selection.selected_orders['aic'])}",
        f"BIC selected:              {int(selection.selected_orders['bic'])}",
        f"HQIC selected:             {int(selection.selected_orders['hqic'])}",
        f"FPE selected:              {int(selection.selected_orders['fpe'])}",
        f"Candidates:                {candidate_lags}",
        "",
        "EXECUTION",
        "-" * 108,
    ]

    score_lookup = dict(
        zip(
            summary_table["config"],
            summary_table["score"],
        )
    )

    for _, row in summary_table.sort_values(
        "lag"
    ).iterrows():
        score = score_lookup.get(
            row["config"],
            np.nan,
        )

        score_text = (
            f"{score:.8f}"
            if pd.notna(score)
            else "NA"
        )

        report_lines.append(
            f"{row['config']} "
            f"status={row['status']} "
            f"stable={row['stable']} "
            f"min_abs_root={row['min_abs_root']} "
            f"Score={score_text}"
        )

    report_lines += [
        "",
        "RANKING VALID VAR",
        "-" * 108,
    ]

    if ranking.empty:
        report_lines.append(
            "NONE"
        )
    else:
        for _, row in ranking.sort_values(
            "rank"
        ).iterrows():
            report_lines.append(
                f"rank={int(row['rank'])} "
                f"{row['config']} "
                f"Score={row['score']:.8f} "
                f"BIC={row['bic']:.8f}"
            )

    report_lines += [
        "",
        "SHORTLIST APRIL",
        "-" * 108,
        str(
            shortlist.sort_values(
                "rank"
            )["config"].tolist()
            if not shortlist.empty
            else []
        ),
        "",
        "GLOBAL RESULT",
        "-" * 108,
        f"STATUS: {overall_status}",
        f"valid_VAR: {valid_count}",
        "June used: NO",
        "",
    ]

    outputs["report"].write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    artifact_paths = [
        outputs["lag_criteria"],
        outputs["lag_selection"],
        outputs["scaler"],
        outputs["candidate_summary"],
        outputs["metrics"],
        outputs["ranking"],
        outputs["shortlist"],
        outputs["granger"],
        outputs["report"],
        outputs["predictions_parquet"],
        outputs["predictions_csv"],
        *figure_paths,
    ]

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": overall_status,
        "created_at": now_iso(),
        "protocol": {
            "sha256": protocol_hash,
            "state": "FROZEN",
        },
        "diagnostic_prerequisite": {
            "diagnostic_id": DIAGNOSTIC_ID,
            "manifest_path": str(
                args.diagnostic_manifest
            ),
            "manifest_sha256": diagnostic_hash,
            "status": diagnostic_manifest[
                "status"
            ],
        },
        "input": {
            "path": str(args.input),
            "sha256": input_hash,
            "rows": len(frame),
        },
        "split": {
            "n_train_levels": len(train),
            "n_validation_levels": len(validation),
            "n_transformed_train": len(train_z),
            "origin_0": (
                train[
                    "timestamp"
                ].max().isoformat()
            ),
            "refit_validation": False,
            "observed_vector_update": True,
        },
        "model_policy": {
            "family": "VAR",
            "implementation": "statsmodels.tsa.api.VAR",
            "endogenous": list(VAR_COLUMNS),
            "target_evaluated": TARGET,
            "representation": "first_differences",
            "trend": "c",
            "standardization": {
                "formula": "z=(x-mu_train)/sigma_train",
                "fit_scope": "training_only",
                "ddof": 0,
            },
            "maxlags": MAXLAGS,
            "max_candidates": MAX_CANDIDATES,
            "stability_required": True,
        },
        "lag_selection": lag_selection_payload,
        "candidate_lags": candidate_lags,
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
        },
        "mase_scale_bps": mase_scale,
        "blindness": {
            "june_used": False,
            "validation_used_for_lag_selection": False,
        },
        "granger": {
            "generated_for_top_ranked_march_var": bool(
                not granger.empty
            ),
            "used_for_selection": False,
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
        "artifacts": hash_artifacts(
            artifact_paths
        ),
    }

    outputs["manifest"].write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("RESULT")
    print("-" * 108)
    print(f"valid_VAR={valid_count}")
    print(
        "ranking=",
        ranking.sort_values(
            "rank"
        )["config"].tolist()
        if not ranking.empty
        else [],
    )
    print(
        "April shortlist=",
        shortlist.sort_values(
            "rank"
        )["config"].tolist()
        if not shortlist.empty
        else [],
    )
    print(f"STATUS: {overall_status}")

    return (
        0
        if overall_status == STATUS_PASS
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
