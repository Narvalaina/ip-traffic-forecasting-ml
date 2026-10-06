#!/usr/bin/env python3
"""
UGR16-STATISTICAL-JUNE-INFERENCE-RESIDUALS-001

Post-hoc inferential comparison and residual analysis over the ALREADY GENERATED
June blind predictions from UGR16-STATISTICAL-JUNE-BLIND-001.

IMPORTANT
---------
This script DOES NOT fit, refit, train, tune, select, or substitute any model.
It only reads frozen June blind predictions and derives paired inference,
residual diagnostics, tables, figures, a report, and a SHA-256 manifest.

Inference deliberately reuses the established UGR'16 Phase-A methodology:
- absolute-error paired loss differential;
- positive differential = model improves over reference;
- circular moving-block bootstrap;
- 5,000 repetitions;
- primary block length 12 intervals (60 min at 5-min sampling);
- block-length sensitivity 6/12/24 intervals;
- 95 % confidence intervals;
- DM-style mean loss differential with Newey-West/HAC long-run variance;
- minimum HAC lag 12;
- two-sided normal-approximation p-values;
- no multiple-comparison correction (not part of the frozen historical method).

Primary inferential contrasts (pre-specified by the handoff):
1. VAR(5) vs ARIMA(6,1,12)
2. VAR(5) vs persistence
3. ARIMA(6,1,12) vs persistence
4. AR(24) vs ARMA(24,3)

Residual convention:
    residual = observed - predicted
    residual > 0 -> underprediction
    residual < 0 -> overprediction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAMPAIGN_ID = "UGR16-STATISTICAL-JUNE-INFERENCE-RESIDUALS-001"
SOURCE_RUN_ID = "UGR16-STATISTICAL-JUNE-BLIND-001"
RUNNER_VERSION = "1.0"

EXPECTED_FREEZE_008_SHA256 = (
    "cc98c4f7b74e2d0ce6515ac1a989bef2b20de0a921f63a5d0346591f2456ad1a"
)
EXPECTED_JUNE_MANIFEST_SHA256 = (
    "847f877144f2932e0b5967b1de0678898cb640434dbfb20f2f9805d7c29883a7"
)
EXPECTED_JUNE_PREDICTIONS_SHA256 = (
    "21ebddf20569533a9d1ed66547cc4393f9c3b7e6721515c68eb3eb9dfc95d353"
)
EXPECTED_JUNE_METRICS_SHA256 = (
    "7c61f308f71539f3bb8e8787c51e3cc85ee17440316db80ed5bdcafe241e52bf"
)
EXPECTED_JUNE_CANDIDATE_SUMMARY_SHA256 = (
    "7489593258e5750c31d1b0546ad20a7ca9f818be99baa41d15d2ed459ebce3a6"
)
EXPECTED_JUNE_DESCRIPTIVE_RANKING_SHA256 = (
    "e46e40c10c5b8424156f8cb7bc50ad4984c6dc18aa919011335f4c40d3f59ac3"
)

FROZEN_CONFIGS = [
    "AR(24)",
    "MA(12)",
    "ARMA(24,3)",
    "ARIMA(6,1,12)",
    "VAR(5)",
]
FAMILY_BY_CONFIG = {
    "AR(24)": "AR",
    "MA(12)": "MA",
    "ARMA(24,3)": "ARMA",
    "ARIMA(6,1,12)": "ARIMA",
    "VAR(5)": "VAR",
}

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
EXPECTED_PREDICTION_ROWS = sum(EXPECTED_COUNTS.values()) * len(FROZEN_CONFIGS)

PRIMARY_COMPARISONS = [
    {
        "comparison": "var_vs_arima",
        "model": "VAR(5)",
        "reference": "ARIMA(6,1,12)",
    },
    {
        "comparison": "var_vs_persistence",
        "model": "VAR(5)",
        "reference": "persistence",
    },
    {
        "comparison": "arima_vs_persistence",
        "model": "ARIMA(6,1,12)",
        "reference": "persistence",
    },
    {
        "comparison": "ar_vs_arma",
        "model": "AR(24)",
        "reference": "ARMA(24,3)",
    },
]

DEFAULT_METRICS_DIR = Path("results/metrics")
DEFAULT_PREDICTIONS_DIR = Path("results/predictions")
DEFAULT_FIGURES_DIR = Path("results/figures/ugr16_statistical_june_inference_residuals")
DEFAULT_PREFIX = "ugr16_statistical_june_inference_residuals"

TRAFFIC_LEVELS = ["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired inference and residual analysis over frozen UGR'16 June "
            "statistical-model predictions. No model training is performed."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            DEFAULT_PREDICTIONS_DIR
            / "ugr16_statistical_june_blind_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--june-manifest",
        type=Path,
        default=DEFAULT_METRICS_DIR / "ugr16_statistical_june_blind_manifest.json",
    )
    parser.add_argument(
        "--june-metrics",
        type=Path,
        default=DEFAULT_METRICS_DIR / "ugr16_statistical_june_blind_metrics.csv",
    )
    parser.add_argument(
        "--june-candidate-summary",
        type=Path,
        default=DEFAULT_METRICS_DIR / "ugr16_statistical_june_blind_candidate_summary.csv",
    )
    parser.add_argument(
        "--june-descriptive-ranking",
        type=Path,
        default=DEFAULT_METRICS_DIR / "ugr16_statistical_june_blind_descriptive_ranking.csv",
    )
    parser.add_argument(
        "--freeze-008",
        type=Path,
        default=Path(
            "docs/project_governance/"
            "008_statistical_final_representatives_freeze_2026-08-16.md"
        ),
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--campaign-id", type=str, default=CAMPAIGN_ID)

    # Reuse the already-established Phase-A inference specification.
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-block-length", type=int, default=12)
    parser.add_argument(
        "--bootstrap-sensitivity-blocks",
        type=int,
        nargs="+",
        default=[6, 12, 24],
    )
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20260717)
    parser.add_argument("--minimum-hac-lag", type=int, default=12)

    parser.add_argument("--acf-lags", type=int, nargs="+", default=[1, 3, 6, 12, 24])
    parser.add_argument("--acf-max-lag", type=int, default=48)
    parser.add_argument("--top-worst-per-group", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Falta {label}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 incorrecto para {label}.\n"
            f"Esperado: {expected}\n"
            f"Observado: {observed}\n"
            f"Ruta: {path}"
        )
    return observed


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON no válido (objeto esperado): {path}")
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    dataframe.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    dataframe.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def current_runner_sha() -> str:
    return sha256(Path(__file__).resolve())


def load_and_validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    hashes = {
        "freeze_008": require_hash(
            args.freeze_008, EXPECTED_FREEZE_008_SHA256, "Freeze 008"
        ),
        "june_manifest": require_hash(
            args.june_manifest,
            EXPECTED_JUNE_MANIFEST_SHA256,
            "June blind manifest",
        ),
        "june_predictions": require_hash(
            args.predictions,
            EXPECTED_JUNE_PREDICTIONS_SHA256,
            "June blind predictions",
        ),
        "june_metrics": require_hash(
            args.june_metrics,
            EXPECTED_JUNE_METRICS_SHA256,
            "June blind metrics",
        ),
        "june_candidate_summary": require_hash(
            args.june_candidate_summary,
            EXPECTED_JUNE_CANDIDATE_SUMMARY_SHA256,
            "June blind candidate summary",
        ),
        "june_descriptive_ranking": require_hash(
            args.june_descriptive_ranking,
            EXPECTED_JUNE_DESCRIPTIVE_RANKING_SHA256,
            "June blind descriptive ranking",
        ),
    }

    manifest = read_json(args.june_manifest)
    metrics = pd.read_csv(args.june_metrics)
    candidate_summary = pd.read_csv(args.june_candidate_summary)
    descriptive_ranking = pd.read_csv(args.june_descriptive_ranking)
    predictions = pd.read_parquet(args.predictions).copy()

    if manifest.get("campaign_id") != "UGR16-STATISTICAL-MODELS-001":
        raise RuntimeError("Campaign ID inesperado en June manifest.")
    if manifest.get("run_id") != SOURCE_RUN_ID:
        raise RuntimeError("Run ID inesperado en June manifest.")
    if manifest.get("status") != "PASS":
        raise RuntimeError("June blind source campaign no está en PASS.")

    evaluation = manifest.get("evaluation", {})
    if evaluation.get("selection_performed") is not False:
        raise RuntimeError("June manifest indica selección en June.")
    if evaluation.get("tuning_performed") is not False:
        raise RuntimeError("June manifest indica tuning en June.")
    if evaluation.get("substitution_allowed") is not False:
        raise RuntimeError("June manifest no conserva la prohibición de sustitución.")

    required_columns = {
        "family",
        "config",
        "origin_timestamp",
        "target_timestamp",
        "horizon_steps",
        "horizon_minutes",
        "y_true",
        "y_pred",
        "persistence_pred",
        "residual",
        "abs_error",
    }
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Faltan columnas en predictions: {sorted(missing)}")

    predictions["origin_timestamp"] = pd.to_datetime(
        predictions["origin_timestamp"], errors="raise"
    )
    predictions["target_timestamp"] = pd.to_datetime(
        predictions["target_timestamp"], errors="raise"
    )

    numeric_columns = [
        "horizon_steps",
        "horizon_minutes",
        "y_true",
        "y_pred",
        "persistence_pred",
        "residual",
        "abs_error",
    ]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")

    if len(predictions) != EXPECTED_PREDICTION_ROWS:
        raise RuntimeError(
            f"Filas de predictions={len(predictions)} != {EXPECTED_PREDICTION_ROWS}"
        )

    if set(predictions["config"].astype(str)) != set(FROZEN_CONFIGS):
        raise RuntimeError("El conjunto de configuraciones no coincide con Freeze 008.")

    for config in FROZEN_CONFIGS:
        families = set(
            predictions.loc[predictions["config"].eq(config), "family"].astype(str)
        )
        if families != {FAMILY_BY_CONFIG[config]}:
            raise RuntimeError(f"Familia incorrecta para {config}: {families}")

    if set(predictions["horizon_steps"].astype(int)) != set(HORIZONS):
        raise RuntimeError("Horizontes inesperados en predictions.")

    finite = predictions[
        ["y_true", "y_pred", "persistence_pred", "residual", "abs_error"]
    ].to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise RuntimeError("Predictions contiene NaN/Inf.")

    if not np.allclose(
        predictions["residual"].to_numpy(float),
        predictions["y_true"].to_numpy(float)
        - predictions["y_pred"].to_numpy(float),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Residual no reconcilia con observed - predicted.")

    if not np.allclose(
        predictions["abs_error"].to_numpy(float),
        np.abs(predictions["residual"].to_numpy(float)),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("abs_error no reconcilia con |residual|.")

    # Coverage and common-target validation.
    for horizon in HORIZONS:
        expected = EXPECTED_COUNTS[horizon]
        reference_targets: tuple[pd.Timestamp, ...] | None = None
        reference_truth: np.ndarray | None = None
        reference_persistence: np.ndarray | None = None

        for config in FROZEN_CONFIGS:
            subset = predictions[
                predictions["config"].eq(config)
                & predictions["horizon_steps"].eq(horizon)
            ].sort_values("target_timestamp")

            if len(subset) != expected:
                raise RuntimeError(
                    f"Cobertura {config} H{horizon}: {len(subset)} != {expected}"
                )
            if subset["target_timestamp"].duplicated().any():
                raise RuntimeError(f"Targets duplicados: {config} H{horizon}")

            targets = tuple(subset["target_timestamp"].tolist())
            truth = subset["y_true"].to_numpy(float)
            persistence = subset["persistence_pred"].to_numpy(float)

            if reference_targets is None:
                reference_targets = targets
                reference_truth = truth
                reference_persistence = persistence
            else:
                if targets != reference_targets:
                    raise RuntimeError(
                        f"Timestamps no emparejados en H{horizon}: {config}"
                    )
                if not np.allclose(truth, reference_truth, rtol=0.0, atol=1e-6):
                    raise RuntimeError(f"y_true difiere entre modelos en H{horizon}")
                if not np.allclose(
                    persistence, reference_persistence, rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError(
                        f"persistence_pred difiere entre modelos en H{horizon}"
                    )

    # Cross-check source metrics/candidate tables.
    if len(metrics) != len(FROZEN_CONFIGS) * len(HORIZONS):
        raise RuntimeError("Número inesperado de filas en June metrics.")
    if set(candidate_summary["config"].astype(str)) != set(FROZEN_CONFIGS):
        raise RuntimeError("candidate_summary no coincide con modelos frozen.")
    if not candidate_summary["status"].astype(str).eq("PASS").all():
        raise RuntimeError("Algún candidato frozen no está PASS en candidate_summary.")
    if set(descriptive_ranking["config"].astype(str)) != set(FROZEN_CONFIGS):
        raise RuntimeError("descriptive_ranking no coincide con modelos frozen.")
    if "used_for_selection" in descriptive_ranking.columns:
        values = descriptive_ranking["used_for_selection"].astype(str).str.lower()
        if not values.isin(["false", "0"]).all():
            raise RuntimeError("El ranking June aparece usado para selección.")

    return {
        "hashes": hashes,
        "manifest": manifest,
        "metrics": metrics,
        "candidate_summary": candidate_summary,
        "descriptive_ranking": descriptive_ranking,
        "predictions": predictions.sort_values(
            ["config", "horizon_steps", "target_timestamp"]
        ).reset_index(drop=True),
    }


def moving_block_bootstrap_means(
    values: np.ndarray,
    repetitions: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        raise ValueError("No hay observaciones para bootstrap.")
    if repetitions <= 0:
        raise ValueError("repetitions debe ser positivo.")
    if block_length <= 0:
        raise ValueError("block_length debe ser positivo.")
    block_length = min(block_length, n)
    blocks_needed = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(repetitions, blocks_needed))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(repetitions, -1)[:, :n]
    return values[indices].mean(axis=1)


def newey_west_lrv(values: np.ndarray, max_lag: int) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    n = centered.size
    if n == 0:
        return float("nan")
    max_lag = min(max_lag, n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * gamma
    return lrv


def dm_hac(values: np.ndarray, hac_lag: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    mean_value = float(np.mean(values))
    lrv = newey_west_lrv(values, hac_lag)
    if not np.isfinite(lrv) or lrv <= 0:
        return float("nan"), float("nan"), lrv
    statistic = mean_value / math.sqrt(lrv / n)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value, lrv


def aligned_pair(
    predictions: pd.DataFrame,
    horizon: int,
    model: str,
    reference: str,
) -> pd.DataFrame:
    model_data = predictions[
        predictions["config"].eq(model)
        & predictions["horizon_steps"].eq(horizon)
    ][
        [
            "target_timestamp",
            "y_true",
            "abs_error",
            "persistence_pred",
        ]
    ].rename(columns={"abs_error": "model_abs_error_bps"})

    if reference == "persistence":
        paired = model_data.copy()
        paired["reference_abs_error_bps"] = np.abs(
            paired["y_true"].to_numpy(float)
            - paired["persistence_pred"].to_numpy(float)
        )
    else:
        reference_data = predictions[
            predictions["config"].eq(reference)
            & predictions["horizon_steps"].eq(horizon)
        ][["target_timestamp", "y_true", "abs_error"]].rename(
            columns={
                "y_true": "reference_y_true",
                "abs_error": "reference_abs_error_bps",
            }
        )
        paired = model_data.merge(reference_data, on="target_timestamp", how="inner")
        if len(paired) != len(model_data) or len(paired) != len(reference_data):
            raise RuntimeError(
                f"Alineación incompleta H{horizon}: {model} vs {reference}."
            )
        if not np.allclose(
            paired["y_true"].to_numpy(float),
            paired["reference_y_true"].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(f"y_true no coincide: {model} vs {reference} H{horizon}")

    paired = paired.sort_values("target_timestamp").reset_index(drop=True)
    paired["loss_improvement_bps"] = (
        paired["reference_abs_error_bps"] - paired["model_abs_error_bps"]
    )
    return paired


def create_inference_tables(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    confidence = args.bootstrap_confidence
    if not (0.0 < confidence < 1.0):
        raise ValueError("--bootstrap-confidence debe estar entre 0 y 1.")
    alpha = 1.0 - confidence
    blocks = sorted(set(args.bootstrap_sensitivity_blocks))
    if args.bootstrap_block_length not in blocks:
        blocks = sorted(set([*blocks, args.bootstrap_block_length]))

    inference_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    base_rng = np.random.default_rng(args.random_seed)

    for spec in PRIMARY_COMPARISONS:
        for horizon in HORIZONS:
            paired = aligned_pair(
                predictions,
                horizon,
                spec["model"],
                spec["reference"],
            )
            differential = paired["loss_improvement_bps"].to_numpy(float)
            model_mae = float(paired["model_abs_error_bps"].mean())
            reference_mae = float(paired["reference_abs_error_bps"].mean())
            mean_improvement = float(differential.mean())
            median_improvement = float(np.median(differential))
            relative_improvement_pct = 100.0 * mean_improvement / reference_mae

            hac_lag = max(args.minimum_hac_lag, horizon - 1)
            if np.allclose(differential, 0.0):
                dm_stat, dm_p, lrv = 0.0, 1.0, 0.0
            else:
                dm_stat, dm_p, lrv = dm_hac(differential, hac_lag)

            child_seed = int(base_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(child_seed)
            bootstrap = moving_block_bootstrap_means(
                differential,
                args.bootstrap_repetitions,
                args.bootstrap_block_length,
                rng,
            )
            lower, upper = np.quantile(
                bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            prob_improvement = (
                0.5
                if np.allclose(differential, 0.0)
                else float(np.mean(bootstrap > 0.0))
            )

            if lower > 0:
                bootstrap_direction = "model_better"
            elif upper < 0:
                bootstrap_direction = "reference_better"
            else:
                bootstrap_direction = "not_distinguishable"

            wins = int(np.sum(differential > 0))
            losses = int(np.sum(differential < 0))
            ties = int(len(differential) - wins - losses)

            inference_rows.append(
                {
                    "comparison": spec["comparison"],
                    "model": spec["model"],
                    "reference_model": spec["reference"],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_pairs": len(paired),
                    "model_mae_mbps": model_mae / 1e6,
                    "reference_mae_mbps": reference_mae / 1e6,
                    "mean_loss_improvement_mbps": mean_improvement / 1e6,
                    "median_loss_improvement_mbps": median_improvement / 1e6,
                    "skill_vs_reference_pct": relative_improvement_pct,
                    "bootstrap_method": "circular moving-block bootstrap",
                    "bootstrap_block_length": args.bootstrap_block_length,
                    "bootstrap_repetitions": args.bootstrap_repetitions,
                    "bootstrap_confidence": confidence,
                    "bootstrap_ci_lower_mbps": float(lower) / 1e6,
                    "bootstrap_ci_upper_mbps": float(upper) / 1e6,
                    "bootstrap_probability_improvement": prob_improvement,
                    "bootstrap_direction": bootstrap_direction,
                    "hac_lag": hac_lag,
                    "dm_hac_statistic": dm_stat,
                    "dm_hac_p_value_two_sided": dm_p,
                    "dm_hac_significant_raw_0_05": bool(
                        np.isfinite(dm_p) and dm_p < 0.05
                    ),
                    "newey_west_lrv": lrv,
                    "multiple_testing_correction": "none_historical_protocol",
                    "random_seed": child_seed,
                }
            )

            pair_rows.append(
                {
                    "comparison": spec["comparison"],
                    "model": spec["model"],
                    "reference_model": spec["reference"],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_pairs": len(paired),
                    "wins_model_lower_abs_error": wins,
                    "ties_abs_error": ties,
                    "losses_model_higher_abs_error": losses,
                    "win_rate_pct": 100.0 * wins / len(paired),
                    "mean_improvement_mbps": mean_improvement / 1e6,
                    "median_improvement_mbps": median_improvement / 1e6,
                }
            )

            for block_length in blocks:
                sensitivity_seed = int(base_rng.integers(0, 2**32 - 1))
                sensitivity_rng = np.random.default_rng(sensitivity_seed)
                sensitivity = moving_block_bootstrap_means(
                    differential,
                    args.bootstrap_repetitions,
                    block_length,
                    sensitivity_rng,
                )
                sens_lower, sens_upper = np.quantile(
                    sensitivity, [alpha / 2.0, 1.0 - alpha / 2.0]
                )
                sensitivity_rows.append(
                    {
                        "comparison": spec["comparison"],
                        "model": spec["model"],
                        "reference_model": spec["reference"],
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "block_length": block_length,
                        "block_minutes": block_length * 5,
                        "mean_improvement_mbps": mean_improvement / 1e6,
                        "ci_lower_mbps": float(sens_lower) / 1e6,
                        "ci_upper_mbps": float(sens_upper) / 1e6,
                        "probability_improvement": (
                            0.5
                            if np.allclose(differential, 0.0)
                            else float(np.mean(sensitivity > 0.0))
                        ),
                        "random_seed": sensitivity_seed,
                    }
                )

    inference = pd.DataFrame(inference_rows).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["comparison", "horizon_steps", "block_length"]
    ).reset_index(drop=True)
    pair_summary = pd.DataFrame(pair_rows).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)
    return inference, sensitivity, pair_summary


def residual_autocorrelation(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if lag <= 0 or len(values) <= lag:
        return float("nan")
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"], sort=False
    ):
        group = group.sort_values("target_timestamp")
        residual = group["residual"].to_numpy(float)
        abs_error = group["abs_error"].to_numpy(float)
        rows.append(
            {
                "family": FAMILY_BY_CONFIG[str(config)],
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "n": len(group),
                "mae_mbps": float(np.mean(abs_error)) / 1e6,
                "rmse_mbps": float(np.sqrt(np.mean(residual**2))) / 1e6,
                "mean_residual_mbps": float(np.mean(residual)) / 1e6,
                "median_residual_mbps": float(np.median(residual)) / 1e6,
                "underprediction_pct": 100.0 * float(np.mean(residual > 0)),
                "overprediction_pct": 100.0 * float(np.mean(residual < 0)),
                "p95_abs_error_mbps": float(np.percentile(abs_error, 95)) / 1e6,
                "max_abs_error_mbps": float(np.max(abs_error)) / 1e6,
                "residual_acf_lag1": residual_autocorrelation(residual, 1),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "config"]
    ).reset_index(drop=True)


def residual_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"], sort=False
    ):
        residual_mbps = group["residual"].to_numpy(float) / 1e6
        series = pd.Series(residual_mbps)
        rows.append(
            {
                "family": FAMILY_BY_CONFIG[str(config)],
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "n": len(series),
                "mean_mbps": float(series.mean()),
                "std_mbps": float(series.std(ddof=0)),
                "q05_mbps": float(series.quantile(0.05)),
                "q25_mbps": float(series.quantile(0.25)),
                "median_mbps": float(series.quantile(0.50)),
                "q75_mbps": float(series.quantile(0.75)),
                "q95_mbps": float(series.quantile(0.95)),
                "skewness": float(series.skew()),
                "excess_kurtosis": float(series.kurt()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "config"]
    ).reset_index(drop=True)


def add_traffic_levels(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = predictions.copy()
    threshold_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        base = (
            annotated[
                annotated["config"].eq(FROZEN_CONFIGS[0])
                & annotated["horizon_steps"].eq(horizon)
            ][["target_timestamp", "y_true"]]
            .sort_values("target_timestamp")
            .reset_index(drop=True)
        )
        q1, q2, q3 = base["y_true"].quantile([0.25, 0.50, 0.75]).tolist()
        threshold_rows.append(
            {
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "q25_bps": q1,
                "q50_bps": q2,
                "q75_bps": q3,
                "q25_mbps": q1 / 1e6,
                "q50_mbps": q2 / 1e6,
                "q75_mbps": q3 / 1e6,
            }
        )

        mask = annotated["horizon_steps"].eq(horizon)
        y = annotated.loc[mask, "y_true"].to_numpy(float)
        labels = np.where(
            y <= q1,
            "Q1_low",
            np.where(y <= q2, "Q2_mid_low", np.where(y <= q3, "Q3_mid_high", "Q4_high")),
        )
        annotated.loc[mask, "traffic_level"] = labels

    annotated["traffic_level"] = pd.Categorical(
        annotated["traffic_level"], categories=TRAFFIC_LEVELS, ordered=True
    )
    thresholds = pd.DataFrame(threshold_rows)
    return annotated, thresholds


def traffic_level_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, horizon, level), group in annotated.groupby(
        ["config", "horizon_steps", "traffic_level"],
        observed=True,
        sort=False,
    ):
        residual = group["residual"].to_numpy(float)
        abs_error = group["abs_error"].to_numpy(float)
        rows.append(
            {
                "family": FAMILY_BY_CONFIG[str(config)],
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "traffic_level": str(level),
                "n": len(group),
                "mean_target_mbps": float(group["y_true"].mean()) / 1e6,
                "mae_mbps": float(np.mean(abs_error)) / 1e6,
                "mean_residual_mbps": float(np.mean(residual)) / 1e6,
                "underprediction_pct": 100.0 * float(np.mean(residual > 0)),
                "p95_abs_error_mbps": float(np.percentile(abs_error, 95)) / 1e6,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "config", "traffic_level"]
    ).reset_index(drop=True)


def extreme_summary(traffic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, horizon), group in traffic.groupby(["config", "horizon_steps"], sort=False):
        indexed = group.set_index("traffic_level")
        low = indexed.loc["Q1_low"]
        high = indexed.loc["Q4_high"]
        extreme_mae = 0.5 * (float(low["mae_mbps"]) + float(high["mae_mbps"]))
        middle = indexed.loc[["Q2_mid_low", "Q3_mid_high"], "mae_mbps"].mean()
        rows.append(
            {
                "family": FAMILY_BY_CONFIG[str(config)],
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "low_q1_mae_mbps": float(low["mae_mbps"]),
                "low_q1_bias_mbps": float(low["mean_residual_mbps"]),
                "high_q4_mae_mbps": float(high["mae_mbps"]),
                "high_q4_bias_mbps": float(high["mean_residual_mbps"]),
                "extreme_mean_mae_mbps": extreme_mae,
                "middle_mean_mae_mbps": float(middle),
                "extreme_to_middle_mae_ratio": (
                    float(extreme_mae / middle) if middle > 0 else float("nan")
                ),
                "regression_to_mean_direction": bool(
                    float(low["mean_residual_mbps"]) < 0
                    and float(high["mean_residual_mbps"]) > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "config"]
    ).reset_index(drop=True)


def hourly_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["target_hour"] = work["target_timestamp"].dt.hour.astype(int)
    rows: list[dict[str, Any]] = []
    for (config, horizon, hour), group in work.groupby(
        ["config", "horizon_steps", "target_hour"], sort=False
    ):
        rows.append(
            {
                "family": FAMILY_BY_CONFIG[str(config)],
                "config": config,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "target_hour": int(hour),
                "n": len(group),
                "mae_mbps": float(group["abs_error"].mean()) / 1e6,
                "mean_residual_mbps": float(group["residual"].mean()) / 1e6,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "config", "target_hour"]
    ).reset_index(drop=True)


def residual_acf_summary(
    predictions: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"], sort=False
    ):
        residual = (
            group.sort_values("target_timestamp")["residual"].to_numpy(float)
        )
        for lag in lags:
            rows.append(
                {
                    "family": FAMILY_BY_CONFIG[str(config)],
                    "config": config,
                    "horizon_steps": int(horizon),
                    "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                    "lag_steps": int(lag),
                    "lag_minutes": int(lag) * 5,
                    "residual_autocorrelation": residual_autocorrelation(
                        residual, int(lag)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["config", "horizon_steps", "lag_steps"]
    ).reset_index(drop=True)


def full_residual_acf(
    predictions: pd.DataFrame,
    configs: list[str],
    max_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for horizon in HORIZONS:
            residual = predictions[
                predictions["config"].eq(config)
                & predictions["horizon_steps"].eq(horizon)
            ].sort_values("target_timestamp")["residual"].to_numpy(float)
            for lag in range(1, max_lag + 1):
                rows.append(
                    {
                        "config": config,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "lag_steps": lag,
                        "lag_minutes": lag * 5,
                        "residual_autocorrelation": residual_autocorrelation(
                            residual, lag
                        ),
                    }
                )
    return pd.DataFrame(rows)


def worst_intervals(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames = []
    for (config, horizon), group in predictions.groupby(
        ["config", "horizon_steps"], sort=False
    ):
        frames.append(group.nlargest(top_n, "abs_error"))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["config", "horizon_steps", "abs_error"], ascending=[True, True, False]
    ).reset_index(drop=True)


def extreme_var_arima_pairwise(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for level in ("Q1_low", "Q4_high"):
            subset = annotated[
                annotated["horizon_steps"].eq(horizon)
                & annotated["traffic_level"].eq(level)
            ]
            var = subset[subset["config"].eq("VAR(5)")][
                ["target_timestamp", "abs_error"]
            ].rename(columns={"abs_error": "var_abs"})
            arima = subset[subset["config"].eq("ARIMA(6,1,12)")][
                ["target_timestamp", "abs_error"]
            ].rename(columns={"abs_error": "arima_abs"})
            paired = var.merge(arima, on="target_timestamp", how="inner")
            diff = paired["arima_abs"].to_numpy(float) - paired["var_abs"].to_numpy(float)
            rows.append(
                {
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "traffic_level": level,
                    "n_pairs": len(paired),
                    "mean_var_improvement_vs_arima_mbps": float(np.mean(diff)) / 1e6,
                    "median_var_improvement_vs_arima_mbps": float(np.median(diff)) / 1e6,
                    "var_win_rate_pct": 100.0 * float(np.mean(diff > 0)),
                }
            )
    return pd.DataFrame(rows)


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    prefix = args.prefix.strip()
    if not prefix:
        raise ValueError("--prefix vacío.")
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "inference": args.metrics_dir / f"{prefix}_paired_inference.csv",
        "sensitivity": args.metrics_dir / f"{prefix}_bootstrap_sensitivity.csv",
        "pair_summary": args.metrics_dir / f"{prefix}_paired_summary.csv",
        "residual_summary": args.metrics_dir / f"{prefix}_residual_summary.csv",
        "distribution": args.metrics_dir / f"{prefix}_residual_distribution.csv",
        "traffic_thresholds": args.metrics_dir / f"{prefix}_traffic_thresholds.csv",
        "traffic_level": args.metrics_dir / f"{prefix}_traffic_level.csv",
        "extreme_summary": args.metrics_dir / f"{prefix}_extreme_summary.csv",
        "hourly": args.metrics_dir / f"{prefix}_hourly.csv",
        "acf": args.metrics_dir / f"{prefix}_residual_acf.csv",
        "full_acf": args.metrics_dir / f"{prefix}_finalists_residual_acf_full.csv",
        "extreme_pairwise": args.metrics_dir / f"{prefix}_var_arima_extreme_pairwise.csv",
        "worst": args.metrics_dir / f"{prefix}_worst_intervals.csv",
        "annotated": args.metrics_dir / f"{prefix}_annotated_predictions.parquet",
        "report": args.metrics_dir / f"{prefix}_report.txt",
        "manifest": args.metrics_dir / f"{prefix}_manifest.json",
    }


def assert_outputs_available(paths: dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Ya existen salidas. Use --overwrite si desea regenerarlas:\n"
            + "\n".join(existing)
        )


def save_figure(base: Path, dpi: int) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def errorbar_from_inference(
    rows: pd.DataFrame,
    title: str,
    base: Path,
    dpi: int,
) -> list[Path]:
    plt.figure(figsize=(9.5, 5.7))
    for comparison, group in rows.groupby("comparison", sort=False):
        group = group.sort_values("horizon_minutes")
        y = group["mean_loss_improvement_mbps"].to_numpy(float)
        low = group["bootstrap_ci_lower_mbps"].to_numpy(float)
        high = group["bootstrap_ci_upper_mbps"].to_numpy(float)
        yerr = np.vstack([y - low, high - y])
        label = f"{group.iloc[0]['model']} vs {group.iloc[0]['reference_model']}"
        plt.errorbar(
            group["horizon_minutes"],
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=1.2,
            label=label,
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Horizonte (min)")
    plt.ylabel("Mejora MAE de modelo frente a referencia (Mbit/s)")
    plt.title(title)
    plt.xticks([5, 15, 30, 60])
    if rows["comparison"].nunique() > 1:
        plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def plot_sensitivity_var_arima(
    sensitivity: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    data = sensitivity[sensitivity["comparison"].eq("var_vs_arima")]
    plt.figure(figsize=(9.5, 5.7))
    for horizon, group in data.groupby("horizon_minutes", sort=True):
        group = group.sort_values("block_minutes")
        y = group["mean_improvement_mbps"].to_numpy(float)
        low = group["ci_lower_mbps"].to_numpy(float)
        high = group["ci_upper_mbps"].to_numpy(float)
        yerr = np.vstack([y - low, high - y])
        plt.errorbar(
            group["block_minutes"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.1,
            label=f"H={int(horizon)} min",
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Longitud de bloque bootstrap (min)")
    plt.ylabel("Mejora MAE VAR(5) frente a ARIMA(6,1,12) (Mbit/s)")
    plt.title("UGR'16 June blind — sensibilidad del bootstrap — VAR vs ARIMA")
    plt.xticks(sorted(data["block_minutes"].unique()))
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def plot_full_acf(data: pd.DataFrame, config: str, base: Path, dpi: int) -> list[Path]:
    subset = data[data["config"].eq(config)]
    plt.figure(figsize=(10, 5.8))
    for horizon, group in subset.groupby("horizon_minutes", sort=True):
        group = group.sort_values("lag_steps")
        plt.plot(
            group["lag_steps"],
            group["residual_autocorrelation"],
            linewidth=1.1,
            label=f"H={int(horizon)} min",
        )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Lag residual (intervalos de 5 min)")
    plt.ylabel("Autocorrelación residual")
    plt.title(f"UGR'16 June blind — ACF residual — {config}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def plot_traffic_heatmap(
    traffic: pd.DataFrame,
    config: str,
    value_column: str,
    value_label: str,
    title_suffix: str,
    base: Path,
    dpi: int,
) -> list[Path]:
    subset = traffic[traffic["config"].eq(config)].copy()
    pivot = subset.pivot(
        index="horizon_minutes", columns="traffic_level", values=value_column
    ).reindex(index=[5, 15, 30, 60], columns=TRAFFIC_LEVELS)
    values = pivot.to_numpy(float)
    plt.figure(figsize=(8.5, 5.3))
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(image, label=value_label)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    plt.yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center")
    plt.xlabel("Nivel de tráfico (cuartil de y observado)")
    plt.ylabel("Horizonte (min)")
    plt.title(f"UGR'16 June blind — {config} — {title_suffix}")
    return save_figure(base, dpi)


def plot_traffic_difference(
    traffic: pd.DataFrame,
    base: Path,
    dpi: int,
) -> list[Path]:
    var = traffic[traffic["config"].eq("VAR(5)")].pivot(
        index="horizon_minutes", columns="traffic_level", values="mae_mbps"
    )
    arima = traffic[traffic["config"].eq("ARIMA(6,1,12)")].pivot(
        index="horizon_minutes", columns="traffic_level", values="mae_mbps"
    )
    diff = (var - arima).reindex(index=[5, 15, 30, 60], columns=TRAFFIC_LEVELS)
    values = diff.to_numpy(float)
    plt.figure(figsize=(8.8, 5.3))
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(image, label="MAE VAR − MAE ARIMA (Mbit/s)")
    plt.xticks(range(len(diff.columns)), diff.columns, rotation=20, ha="right")
    plt.yticks(range(len(diff.index)), [str(v) for v in diff.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center")
    plt.xlabel("Nivel de tráfico")
    plt.ylabel("Horizonte (min)")
    plt.title("UGR'16 June blind — diferencia de MAE por nivel — VAR − ARIMA")
    return save_figure(base, dpi)


def plot_hourly_finalists(
    hourly: pd.DataFrame,
    horizon_minutes: int,
    base: Path,
    dpi: int,
) -> list[Path]:
    data = hourly[
        hourly["config"].isin(["VAR(5)", "ARIMA(6,1,12)"])
        & hourly["horizon_minutes"].eq(horizon_minutes)
    ]
    plt.figure(figsize=(10, 5.7))
    for config, group in data.groupby("config", sort=False):
        group = group.sort_values("target_hour")
        plt.plot(
            group["target_hour"],
            group["mae_mbps"],
            marker="o",
            linewidth=1.1,
            label=config,
        )
    plt.xlabel("Hora objetivo")
    plt.ylabel("MAE (Mbit/s)")
    plt.title(
        f"UGR'16 June blind — error por hora — VAR vs ARIMA — H={horizon_minutes} min"
    )
    plt.xticks(range(24))
    plt.legend()
    plt.grid(True, alpha=0.25)
    return save_figure(base, dpi)


def plot_residual_boxplot(
    predictions: pd.DataFrame,
    horizon_minutes: int,
    base: Path,
    dpi: int,
) -> list[Path]:
    horizon = next(k for k, value in HORIZON_MINUTES.items() if value == horizon_minutes)
    arrays = [
        predictions[
            predictions["config"].eq(config)
            & predictions["horizon_steps"].eq(horizon)
        ]["residual"].to_numpy(float)
        / 1e6
        for config in FROZEN_CONFIGS
    ]
    plt.figure(figsize=(10, 5.7))
    plt.boxplot(arrays, tick_labels=FROZEN_CONFIGS, showfliers=False)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.ylabel("Residual observado − predicho (Mbit/s)")
    plt.xlabel("Modelo")
    plt.title(f"UGR'16 June blind — distribución residual — H={horizon_minutes} min")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return save_figure(base, dpi)


def create_figures(
    inference: pd.DataFrame,
    sensitivity: pd.DataFrame,
    full_acf: pd.DataFrame,
    traffic: pd.DataFrame,
    hourly: pd.DataFrame,
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> list[Path]:
    base = args.figures_dir
    figures: list[Path] = []
    figures += errorbar_from_inference(
        inference[inference["comparison"].eq("var_vs_arima")],
        "UGR'16 June blind — inferencia pareada — VAR(5) vs ARIMA(6,1,12)",
        base / f"{args.prefix}_01_var_vs_arima_ci",
        args.dpi,
    )
    figures += errorbar_from_inference(
        inference[
            inference["comparison"].isin(
                ["var_vs_persistence", "arima_vs_persistence"]
            )
        ],
        "UGR'16 June blind — inferencia pareada — finalistas vs persistencia",
        base / f"{args.prefix}_02_finalists_vs_persistence_ci",
        args.dpi,
    )
    figures += errorbar_from_inference(
        inference[inference["comparison"].eq("ar_vs_arma")],
        "UGR'16 June blind — inferencia pareada — AR(24) vs ARMA(24,3)",
        base / f"{args.prefix}_03_ar_vs_arma_ci",
        args.dpi,
    )
    figures += plot_sensitivity_var_arima(
        sensitivity,
        base / f"{args.prefix}_04_var_vs_arima_bootstrap_sensitivity",
        args.dpi,
    )
    figures += plot_full_acf(
        full_acf,
        "VAR(5)",
        base / f"{args.prefix}_05_var_residual_acf",
        args.dpi,
    )
    figures += plot_full_acf(
        full_acf,
        "ARIMA(6,1,12)",
        base / f"{args.prefix}_06_arima_residual_acf",
        args.dpi,
    )
    figures += plot_traffic_heatmap(
        traffic,
        "VAR(5)",
        "mae_mbps",
        "MAE (Mbit/s)",
        "MAE por nivel de tráfico",
        base / f"{args.prefix}_07_var_traffic_level_mae",
        args.dpi,
    )
    figures += plot_traffic_heatmap(
        traffic,
        "ARIMA(6,1,12)",
        "mae_mbps",
        "MAE (Mbit/s)",
        "MAE por nivel de tráfico",
        base / f"{args.prefix}_08_arima_traffic_level_mae",
        args.dpi,
    )
    figures += plot_traffic_difference(
        traffic,
        base / f"{args.prefix}_09_var_minus_arima_traffic_level_mae",
        args.dpi,
    )
    figures += plot_hourly_finalists(
        hourly,
        5,
        base / f"{args.prefix}_10_hourly_var_arima_h1",
        args.dpi,
    )
    figures += plot_hourly_finalists(
        hourly,
        60,
        base / f"{args.prefix}_11_hourly_var_arima_h12",
        args.dpi,
    )
    figures += plot_residual_boxplot(
        predictions,
        5,
        base / f"{args.prefix}_12_residual_distribution_h1",
        args.dpi,
    )
    figures += plot_residual_boxplot(
        predictions,
        60,
        base / f"{args.prefix}_13_residual_distribution_h12",
        args.dpi,
    )
    return figures


def build_report(
    args: argparse.Namespace,
    runner_sha: str,
    inputs: dict[str, Any],
    inference: pd.DataFrame,
    residual: pd.DataFrame,
    extreme: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> str:
    lines = [
        "=" * 112,
        "UGR'16 JUNE BLIND — INFERENCIA PAREADA Y ANÁLISIS FINAL DE RESIDUOS",
        "=" * 112,
        f"Campaign: {args.campaign_id}",
        f"Source run: {SOURCE_RUN_ID}",
        f"Runner SHA-256: {runner_sha}",
        "Reentrenamiento realizado: NO",
        "Refit realizado: NO",
        "Tuning realizado: NO",
        "Selección realizada: NO",
        "Predicciones fuente modificadas: NO",
        "",
        "INFERENCIA",
        "-" * 112,
        f"Loss: absolute error",
        "Loss differential: abs_error_reference - abs_error_model",
        "Positive value: model improves over reference",
        f"Bootstrap: circular moving-block, {args.bootstrap_repetitions:,} réplicas",
        f"Primary block: {args.bootstrap_block_length} intervalos = {args.bootstrap_block_length*5} min",
        "Sensitivity blocks: "
        + ", ".join(
            f"{b} ({b*5} min)" for b in sorted(set(args.bootstrap_sensitivity_blocks))
        ),
        f"Confidence: {args.bootstrap_confidence:.2f}",
        f"Minimum HAC lag: {args.minimum_hac_lag}",
        "DM-HAC: mean loss differential + Newey-West/Bartlett LRV + normal approximation",
        "Multiple-testing correction: NONE (continuity with established historical protocol)",
        "",
        "PRIMARY PAIRED CONTRASTS",
        "-" * 112,
    ]

    for _, row in inference.sort_values(["comparison", "horizon_steps"]).iterrows():
        lines.append(
            f"{row['comparison']:<23s} H{int(row['horizon_steps']):>2} "
            f"({int(row['horizon_minutes']):>2} min): "
            f"{row['model']} vs {row['reference_model']} | "
            f"improvement={row['mean_loss_improvement_mbps']:+.4f} Mbit/s | "
            f"CI=[{row['bootstrap_ci_lower_mbps']:+.4f}, {row['bootstrap_ci_upper_mbps']:+.4f}] | "
            f"P(improvement)={row['bootstrap_probability_improvement']:.4f} | "
            f"DM-HAC p={row['dm_hac_p_value_two_sided']:.6g} | "
            f"bootstrap={row['bootstrap_direction']}"
        )

    lines += [
        "",
        "PAIRWISE WIN RATES",
        "-" * 112,
    ]
    for _, row in pair_summary.sort_values(["comparison", "horizon_steps"]).iterrows():
        lines.append(
            f"{row['comparison']:<23s} H{int(row['horizon_steps']):>2}: "
            f"wins={int(row['wins_model_lower_abs_error'])}/{int(row['n_pairs'])} "
            f"({row['win_rate_pct']:.2f} %) | "
            f"mean improvement={row['mean_improvement_mbps']:+.4f} Mbit/s"
        )

    lines += [
        "",
        "RESIDUAL SUMMARY — VAR / ARIMA",
        "-" * 112,
    ]
    finalists = residual[residual["config"].isin(["VAR(5)", "ARIMA(6,1,12)"])]
    for _, row in finalists.sort_values(["config", "horizon_steps"]).iterrows():
        lines.append(
            f"{row['config']:<15s} H{int(row['horizon_steps']):>2}: "
            f"MAE={row['mae_mbps']:.4f} | bias={row['mean_residual_mbps']:+.4f} | "
            f"under={row['underprediction_pct']:.2f}% | P95={row['p95_abs_error_mbps']:.4f} | "
            f"ACFres(1)={row['residual_acf_lag1']:+.4f}"
        )

    lines += [
        "",
        "EXTREMES — VAR / ARIMA",
        "-" * 112,
    ]
    finalist_extreme = extreme[
        extreme["config"].isin(["VAR(5)", "ARIMA(6,1,12)"])
    ]
    for _, row in finalist_extreme.sort_values(["config", "horizon_steps"]).iterrows():
        lines.append(
            f"{row['config']:<15s} H{int(row['horizon_steps']):>2}: "
            f"Q1 MAE={row['low_q1_mae_mbps']:.3f}, bias={row['low_q1_bias_mbps']:+.3f} | "
            f"Q4 MAE={row['high_q4_mae_mbps']:.3f}, bias={row['high_q4_bias_mbps']:+.3f} | "
            f"extreme/middle={row['extreme_to_middle_mae_ratio']:.3f} | "
            f"regression-to-mean-direction={row['regression_to_mean_direction']}"
        )

    lines += [
        "",
        "CAUTELAS",
        "-" * 112,
        "- Los contrastes son pareados y se realizan sobre exactamente los mismos target timestamps por horizonte.",
        "- Los IC bootstrap preservan dependencia local mediante bloques circulares.",
        "- Los p-values DM-HAC son bilaterales y NO llevan corrección por comparaciones múltiples, por continuidad con el protocolo histórico establecido.",
        "- La interpretación final debe considerar conjuntamente magnitud del efecto, IC bootstrap, sensibilidad de bloques, DM-HAC y relevancia práctica.",
        "- Este análisis NO autoriza retuning ni sustitución de ningún representante frozen.",
        "",
        "GLOBAL RESULT",
        "-" * 112,
        "STATUS: PASS",
        "No model fitted: TRUE",
        "No model refitted: TRUE",
        "No tuning: TRUE",
        "No selection: TRUE",
        "",
    ]
    return "\n".join(lines)


def validate_outputs(
    inference: pd.DataFrame,
    sensitivity: pd.DataFrame,
    residual: pd.DataFrame,
    traffic: pd.DataFrame,
    acf: pd.DataFrame,
    worst: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, bool]:
    expected_inference = len(PRIMARY_COMPARISONS) * len(HORIZONS)
    expected_sensitivity = (
        len(PRIMARY_COMPARISONS)
        * len(HORIZONS)
        * len(sorted(set([*args.bootstrap_sensitivity_blocks, args.bootstrap_block_length])))
    )
    checks = {
        "inference_rows": len(inference) == expected_inference,
        "sensitivity_rows": len(sensitivity) == expected_sensitivity,
        "residual_rows": len(residual) == len(FROZEN_CONFIGS) * len(HORIZONS),
        "traffic_rows": len(traffic)
        == len(FROZEN_CONFIGS) * len(HORIZONS) * len(TRAFFIC_LEVELS),
        "acf_rows": len(acf)
        == len(FROZEN_CONFIGS) * len(HORIZONS) * len(args.acf_lags),
        "worst_nonempty": len(worst) > 0,
        "n_pairs_positive": inference["n_pairs"].gt(0).all(),
        "ci_ordered": inference["bootstrap_ci_lower_mbps"].le(
            inference["bootstrap_ci_upper_mbps"]
        ).all(),
        "probabilities_valid": inference[
            "bootstrap_probability_improvement"
        ].between(0.0, 1.0).all(),
        "dm_p_values_valid": inference["dm_hac_p_value_two_sided"].dropna().between(
            0.0, 1.0
        ).all(),
        "coverage_expected": all(
            int(row.n_pairs) == EXPECTED_COUNTS[int(row.horizon_steps)]
            for row in inference.itertuples()
        ),
    }
    return {name: bool(value) for name, value in checks.items()}


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    print("=" * 112)
    print("UGR'16 JUNE BLIND — INFERENCIA PAREADA + ANÁLISIS FINAL DE RESIDUOS")
    print("=" * 112)
    print(f"Campaign: {args.campaign_id}")
    print(f"Source run: {SOURCE_RUN_ID}")
    print("Training/refit/tuning: NO / NO / NO")
    print()

    inputs = load_and_validate_inputs(args)
    predictions = inputs["predictions"]
    runner_sha = current_runner_sha()

    print("PRECHECK")
    print("-" * 112)
    print(f"Freeze 008 SHA-256:      {inputs['hashes']['freeze_008']}")
    print(f"June manifest SHA-256:   {inputs['hashes']['june_manifest']}")
    print(f"June predictions SHA-256:{inputs['hashes']['june_predictions']}")
    print(f"June metrics SHA-256:    {inputs['hashes']['june_metrics']}")
    print(f"Prediction rows:          {len(predictions)}")
    print("Frozen configs:           " + ", ".join(FROZEN_CONFIGS))
    print("Coverage:                 H1=602 H3=600 H6=597 H12=591 per model")
    print()
    print("INFERENCE SPECIFICATION")
    print("-" * 112)
    print(
        f"Bootstrap: circular moving-block | repetitions={args.bootstrap_repetitions} | "
        f"primary block={args.bootstrap_block_length} ({args.bootstrap_block_length*5} min)"
    )
    print(
        "Sensitivity blocks: "
        + ", ".join(
            f"{b} ({b*5} min)" for b in sorted(set(args.bootstrap_sensitivity_blocks))
        )
    )
    print(f"Confidence: {args.bootstrap_confidence}")
    print(f"Random seed: {args.random_seed} (historical cross-capture seed reused)")
    print(f"DM-HAC minimum lag: {args.minimum_hac_lag}")
    print("Multiple-testing correction: NONE — historical protocol continuity")
    print()
    print("PRIMARY CONTRASTS")
    print("-" * 112)
    for spec in PRIMARY_COMPARISONS:
        print(
            f"{spec['comparison']:<23s}: {spec['model']} vs {spec['reference']}"
        )

    if args.preflight_only:
        print()
        print("PRECHECK GLOBAL: PASS")
        print("No analysis output generated (--preflight-only).")
        print("No model fitted/refitted.")
        return 0

    paths = output_paths(args)
    assert_outputs_available(paths, args.overwrite)

    inference, sensitivity, pair_summary = create_inference_tables(predictions, args)
    residual = residual_summary(predictions)
    distribution = residual_distribution(predictions)
    annotated, thresholds = add_traffic_levels(predictions)
    traffic = traffic_level_summary(annotated)
    extreme = extreme_summary(traffic)
    hourly = hourly_summary(predictions)
    acf = residual_acf_summary(predictions, list(args.acf_lags))
    full_acf = full_residual_acf(
        predictions, ["VAR(5)", "ARIMA(6,1,12)"], args.acf_max_lag
    )
    extreme_pairwise = extreme_var_arima_pairwise(annotated)
    worst = worst_intervals(predictions, args.top_worst_per_group)

    validation = validate_outputs(
        inference, sensitivity, residual, traffic, acf, worst, args
    )
    if not all(validation.values()):
        failures = [name for name, ok in validation.items() if not ok]
        raise RuntimeError("Validación interna FAIL: " + ", ".join(failures))

    atomic_csv(paths["inference"], inference)
    atomic_csv(paths["sensitivity"], sensitivity)
    atomic_csv(paths["pair_summary"], pair_summary)
    atomic_csv(paths["residual_summary"], residual)
    atomic_csv(paths["distribution"], distribution)
    atomic_csv(paths["traffic_thresholds"], thresholds)
    atomic_csv(paths["traffic_level"], traffic)
    atomic_csv(paths["extreme_summary"], extreme)
    atomic_csv(paths["hourly"], hourly)
    atomic_csv(paths["acf"], acf)
    atomic_csv(paths["full_acf"], full_acf)
    atomic_csv(paths["extreme_pairwise"], extreme_pairwise)
    atomic_csv(paths["worst"], worst)
    atomic_parquet(paths["annotated"], annotated)

    figures = create_figures(
        inference, sensitivity, full_acf, traffic, hourly, predictions, args
    )

    report = build_report(
        args, runner_sha, inputs, inference, residual, extreme, pair_summary
    )
    atomic_text(paths["report"], report)

    artifact_paths = [
        path for name, path in paths.items() if name != "manifest"
    ] + figures
    artifacts = {
        str(path.resolve()): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in artifact_paths
        if path.is_file()
    }

    manifest = {
        "campaign_id": args.campaign_id,
        "source_run_id": SOURCE_RUN_ID,
        "status": "PASS",
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(Path(__file__).resolve()),
            "sha256": runner_sha,
        },
        "governance": {
            "model_training": False,
            "parameter_refit": False,
            "tuning": False,
            "selection": False,
            "model_substitution": False,
            "source_predictions_modified": False,
            "frozen_configs": FROZEN_CONFIGS,
        },
        "inputs": {
            "freeze_008": {
                "path": str(args.freeze_008.resolve()),
                "sha256": inputs["hashes"]["freeze_008"],
            },
            "june_manifest": {
                "path": str(args.june_manifest.resolve()),
                "sha256": inputs["hashes"]["june_manifest"],
            },
            "june_predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": inputs["hashes"]["june_predictions"],
                "rows": len(predictions),
            },
            "june_metrics": {
                "path": str(args.june_metrics.resolve()),
                "sha256": inputs["hashes"]["june_metrics"],
            },
            "june_candidate_summary": {
                "path": str(args.june_candidate_summary.resolve()),
                "sha256": inputs["hashes"]["june_candidate_summary"],
            },
            "june_descriptive_ranking": {
                "path": str(args.june_descriptive_ranking.resolve()),
                "sha256": inputs["hashes"]["june_descriptive_ranking"],
            },
        },
        "inference": {
            "loss": "absolute error",
            "loss_differential": "absolute_error_reference - absolute_error_model",
            "positive_value_meaning": "model improves over reference",
            "primary_comparisons": PRIMARY_COMPARISONS,
            "bootstrap_method": "circular moving-block bootstrap",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_block_length": args.bootstrap_block_length,
            "bootstrap_block_minutes": args.bootstrap_block_length * 5,
            "bootstrap_sensitivity_blocks": sorted(
                set(args.bootstrap_sensitivity_blocks)
            ),
            "bootstrap_confidence": args.bootstrap_confidence,
            "random_seed": args.random_seed,
            "seed_policy": "reuse historical UGR16 cross-capture inference seed",
            "dm_method": (
                "mean loss differential with Newey-West/HAC long-run variance "
                "and normal approximation"
            ),
            "newey_west_kernel": "Bartlett",
            "minimum_hac_lag": args.minimum_hac_lag,
            "hac_lag_rule": "max(minimum_hac_lag, horizon_steps - 1)",
            "multiple_testing_correction": "none_historical_protocol",
        },
        "residual_analysis": {
            "residual_definition": "observed - predicted",
            "positive_residual": "underprediction",
            "negative_residual": "overprediction",
            "traffic_levels": TRAFFIC_LEVELS,
            "traffic_level_definition": "per-horizon quartiles of observed y_true in June blind",
            "acf_summary_lags": list(args.acf_lags),
            "acf_max_lag_finalists": args.acf_max_lag,
            "top_worst_per_model_horizon": args.top_worst_per_group,
        },
        "validation": validation,
        "artifacts": artifacts,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "processing_seconds": float(time.perf_counter() - started),
    }
    atomic_json(paths["manifest"], manifest)

    print()
    print("PAIRED INFERENCE — PRIMARY BLOCK")
    print("-" * 112)
    for _, row in inference.iterrows():
        print(
            f"{row['comparison']:<23s} H{int(row['horizon_steps']):>2}: "
            f"improvement={row['mean_loss_improvement_mbps']:+.4f} Mbit/s | "
            f"CI=[{row['bootstrap_ci_lower_mbps']:+.4f}, {row['bootstrap_ci_upper_mbps']:+.4f}] | "
            f"P(improve)={row['bootstrap_probability_improvement']:.4f} | "
            f"DM-HAC p={row['dm_hac_p_value_two_sided']:.6g}"
        )

    print()
    print("RESULT")
    print("-" * 112)
    print("STATUS: PASS")
    print("Reentrenamiento: NO")
    print("Refit: NO")
    print("Tuning: NO")
    print("Selección: NO")
    print(f"Manifest: {paths['manifest'].resolve()}")
    print("VALIDACIÓN INTERNA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
