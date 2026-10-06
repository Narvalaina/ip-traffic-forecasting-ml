#!/usr/bin/env python3
"""
TRANSFORMER-INFERENCE-RESIDUALS-001
===================================

Paired inferential comparison and residual diagnostics over ALREADY FROZEN
June blind predictions from Phase E TRANSFORMER*.

Runner:
    analyze_transformer_inference_residuals_v1_2026-08-22.py

Internal version:
    1.0.0

This runner DOES NOT:
- fit, refit, train, tune or select any model;
- modify source predictions;
- replace TRANSFORMER*;
- change RECURRENT*, ARIMA*, VAR* or TREE*;
- alter the frozen Phase-E inferential methodology.

Binding authorities:
- 033_phase_e_transformer_lite_protocol_2026-08-21.md
- 044_transformer_june_blind_closure_2026-08-22.md

Frozen reference authorities:
- 028_recurrent_june_blind_closure_2026-08-21.md
- 021_tree_june_blind_closure_2026-08-19.md
- 008_statistical_final_representatives_freeze_2026-08-16.md
- 010_statistical_models_closure_freeze_2026-08-17.md

Frozen primary comparisons:
1. TRANSFORMER* vs Persistence x H1/H3/H6/H12
2. TRANSFORMER* vs RECURRENT*  x H1/H3/H6/H12
3. TRANSFORMER* vs ARIMA*      x H1/H3/H6/H12
4. TRANSFORMER* vs VAR*        x H1/H3/H6/H12
5. TRANSFORMER* vs TREE*       x H1/H3/H6/H12

Total:
    20 primary comparisons

Loss differential:
    absolute_error_reference - absolute_error_transformer

Positive:
    TRANSFORMER* improves over reference

Inference:
- circular moving-block bootstrap;
- 5000 repetitions;
- primary block length = 12;
- sensitivity = 6 / 12 / 24;
- 95 % percentile confidence interval;
- seed = 20260717;
- DM-HAC/Newey-West with Bartlett kernel;
- lag = max(12, horizon_steps - 1) = 12 here;
- two-sided unadjusted p-values;
- no multiplicity correction.

Residual convention:
    residual = observed - predicted
    positive -> underprediction
    negative -> overprediction

Residual diagnostics for TRANSFORMER*:
- bias;
- underprediction percentage;
- RMSE;
- P95 absolute error;
- residual distribution;
- residual ACF lags 1..48;
- traffic quartiles Q1/Q2/Q3/Q4;
- hour-of-day diagnostics;
- top-5 worst intervals per horizon.

With --preflight-only:
- verifies authorities and frozen source hashes;
- validates schemas, frozen identities, coverage and cross-family alignment;
- executes 0 bootstrap;
- executes 0 DM-HAC;
- writes 0 campaign artifacts.
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


RUNNER_VERSION = "1.0.0"
CAMPAIGN_ID = "TRANSFORMER-INFERENCE-RESIDUALS-001"
PARENT_CAMPAIGN_ID = "UGR16-TRANSFORMER-LITE-001"
FILE_TAG = "v1_2026-08-22"

ROOT = Path(__file__).resolve().parents[2]

P033 = ROOT / "docs/project_governance/033_phase_e_transformer_lite_protocol_2026-08-21.md"
P044 = ROOT / "docs/project_governance/044_transformer_june_blind_closure_2026-08-22.md"
P028 = ROOT / "docs/project_governance/028_recurrent_june_blind_closure_2026-08-21.md"
P021 = ROOT / "docs/project_governance/021_tree_june_blind_closure_2026-08-19.md"
P008 = ROOT / "docs/project_governance/008_statistical_final_representatives_freeze_2026-08-16.md"
P010 = ROOT / "docs/project_governance/010_statistical_models_closure_freeze_2026-08-17.md"

TRANSFORMER_PREDICTIONS = (
    ROOT
    / "results/metrics/transformer_lite/june_blind/v2_2026-08-22/"
      "recovered_outputs/ugr16_transformer_june_blind_predictions_v2_2026-08-22.parquet"
)

RECURRENT_PREDICTIONS = (
    ROOT
    / "results/metrics/recurrent_networks/june_blind/"
      "RECURRENT_JUNE_BLIND_001_RESULTS_V1_2026-08-21/"
      "ugr16_recurrent_june_blind_v1_2026-08-21_predictions.parquet"
)

TREE_MANIFEST = ROOT / "results/metrics/tree_ensembles/ugr16_tree_june_blind_manifest.json"
TREE_PREDICTIONS = ROOT / "results/predictions/tree_ensembles/ugr16_tree_june_blind_predictions.parquet"

STAT_MANIFEST = ROOT / "results/metrics/ugr16_statistical_june_blind_manifest.json"
STAT_PREDICTIONS = ROOT / "results/predictions/ugr16_statistical_june_blind_predictions.parquet"

EXPECTED_SHA = {
    "033": "4fcd61519612ae5fd3daa6d874a73df50b27b63531ac109f0782351875211552",
    "044": "2e32b0989c90a0af5558c86cac5367d75a7c5cc61b7ff2b0dc5b183ef14bf532",
    "028": "26379146225af9f0e804696e518cb31e81d4060614d96d1fdff4494cfec19e82",
    "021": "445ec43095bbf503d0a0ea45a2a209916fd705a0f197bb9b1961db2aeae74426",
    "008": "cc98c4f7b74e2d0ce6515ac1a989bef2b20de0a921f63a5d0346591f2456ad1a",
    "010": "aeb527f1c7966cb91aeececd67d2da8e1f114236608e09296d9885ad1de0f530",
    "transformer_predictions": "7195fd3c656a0f34a40a8bbc29d262593c9de8fff6c4f3bc4c2c0b90c11cf750",
    "recurrent_predictions": "a9f4f277bbf7f6415cd56ea1763e7afa74f307fd70f5d5c108844cbc1e964d19",
    "tree_manifest": "f0a296c21b3077b2f9c8788fbb118572a203021ae69074bb1a4a2b027f3d2b46",
    "tree_predictions": "7f787d74de5b593811e7988767d891d4aa70e0f8826bb1a302f20e4d0e7e61ad",
    "stat_manifest": "847f877144f2932e0b5967b1de0678898cb640434dbfb20f2f9805d7c29883a7",
    "stat_predictions": "21ebddf20569533a9d1ed66547cc4393f9c3b7e6721515c68eb3eb9dfc95d353",
}

HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

TRANSFORMER_EXPECTED_ROWS = 2390
RECURRENT_EXPECTED_ROWS = 7170
TREE_EXPECTED_ROWS = 7170
STAT_EXPECTED_ROWS = 11950
STAT_SELECTED_EXPECTED_ROWS = 4780
ONE_MODEL_ROWS = 2390

TRANSFORMER_STAR = "T01/L24"
TRANSFORMER_CONFIG_ID = "T01"
TRANSFORMER_LOOKBACK = 24
TRANSFORMER_CANONICAL_SEED = 20260820

RECURRENT_STAR = ("GRU", "N06")
TREE_STAR = "LGB01"
ARIMA_STAR = "ARIMA(6,1,12)"
VAR_STAR = "VAR(5)"

BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_PRIMARY_BLOCK = 12
BOOTSTRAP_SENSITIVITY_BLOCKS = [6, 12, 24]
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260717
HAC_MIN_LAG = 12

ACF_MAX_LAG = 48
TOP_WORST = 5

PRIMARY_COMPARISONS = [
    {
        "comparison": "transformer_vs_persistence",
        "display_model": "TRANSFORMER*",
        "model_key": "TRANSFORMER*",
        "reference_key": "Persistence",
    },
    {
        "comparison": "transformer_vs_recurrent",
        "display_model": "TRANSFORMER*",
        "model_key": "TRANSFORMER*",
        "reference_key": "RECURRENT*",
    },
    {
        "comparison": "transformer_vs_arima",
        "display_model": "TRANSFORMER*",
        "model_key": "TRANSFORMER*",
        "reference_key": "ARIMA*",
    },
    {
        "comparison": "transformer_vs_var",
        "display_model": "TRANSFORMER*",
        "model_key": "TRANSFORMER*",
        "reference_key": "VAR*",
    },
    {
        "comparison": "transformer_vs_tree",
        "display_model": "TRANSFORMER*",
        "model_key": "TRANSFORMER*",
        "reference_key": "TREE*",
    },
]

METRICS_DIR = (
    ROOT
    / "results/metrics/transformer_lite/inference_residuals/"
      "v1_2026-08-22"
)
FIGURES_DIR = (
    ROOT
    / "results/figures/transformer_lite/"
      "transformer_inference_residuals_v1_2026-08-22"
)

PREFIX = "ugr16_transformer_inference_residuals_v1_2026-08-22"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate frozen inputs, schemas, coverage and alignment only. "
            "Executes no bootstrap/DM-HAC and writes no campaign artifacts."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing file: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label}: SHA mismatch\nexpected={expected}\nactual={actual}\npath={path}"
        )
    return actual


def print_check(label: str, value: Any = None) -> None:
    if value is None:
        print(f"{label:38s}: PASS")
    else:
        print(f"{label:38s}: PASS | {value}")


def detect_stat_model_column(frame: pd.DataFrame) -> str:
    desired = {ARIMA_STAR, VAR_STAR}
    for column in ["config", "candidate_id", "model", "model_id", "configuration"]:
        if column in frame.columns:
            values = set(frame[column].dropna().astype(str).unique())
            if desired.issubset(values):
                return column
    raise RuntimeError(
        "Cannot detect statistical configuration column. "
        f"Columns: {list(frame.columns)}"
    )


def normalize_stat_schema(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    aliases = {
        "y_true_bps": "y_true",
        "y_pred_bps": "y_pred",
    }
    for canonical, legacy in aliases.items():
        c = canonical in normalized.columns
        l = legacy in normalized.columns
        if c and l:
            left = pd.to_numeric(normalized[canonical], errors="raise").to_numpy(float)
            right = pd.to_numeric(normalized[legacy], errors="raise").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"Stat schema aliases disagree: {canonical} vs {legacy}"
                )
        elif not c and l:
            normalized[canonical] = pd.to_numeric(
                normalized[legacy], errors="raise"
            )
    return normalized


def load_validate_sources() -> dict[str, Any]:
    hashes = {}
    for key, source in [
        ("033", P033),
        ("044", P044),
        ("028", P028),
        ("021", P021),
        ("008", P008),
        ("010", P010),
        ("transformer_predictions", TRANSFORMER_PREDICTIONS),
        ("recurrent_predictions", RECURRENT_PREDICTIONS),
        ("tree_manifest", TREE_MANIFEST),
        ("tree_predictions", TREE_PREDICTIONS),
        ("stat_manifest", STAT_MANIFEST),
        ("stat_predictions", STAT_PREDICTIONS),
    ]:
        hashes[key] = verify_sha(source, EXPECTED_SHA[key], key)

    transformer = pd.read_parquet(TRANSFORMER_PREDICTIONS).copy()
    recurrent = pd.read_parquet(RECURRENT_PREDICTIONS).copy()
    tree = pd.read_parquet(TREE_PREDICTIONS).copy()
    stat = normalize_stat_schema(pd.read_parquet(STAT_PREDICTIONS).copy())

    if len(transformer) != TRANSFORMER_EXPECTED_ROWS:
        raise RuntimeError(
            f"Transformer rows={len(transformer)} != {TRANSFORMER_EXPECTED_ROWS}"
        )
    if len(recurrent) != RECURRENT_EXPECTED_ROWS:
        raise RuntimeError(
            f"Recurrent rows={len(recurrent)} != {RECURRENT_EXPECTED_ROWS}"
        )
    if len(tree) != TREE_EXPECTED_ROWS:
        raise RuntimeError(f"Tree rows={len(tree)} != {TREE_EXPECTED_ROWS}")
    if len(stat) != STAT_EXPECTED_ROWS:
        raise RuntimeError(f"Stat rows={len(stat)} != {STAT_EXPECTED_ROWS}")

    transformer_required = {
        "config_id",
        "lookback",
        "canonical_seed",
        "horizon_steps",
        "horizon_minutes",
        "origin_timestamp",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
        "persistence_pred_bps",
    }
    recurrent_required = {
        "family",
        "config_id",
        "canonical_seed",
        "horizon_steps",
        "horizon_minutes",
        "origin_timestamp",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
        "persistence_pred_bps",
    }
    tree_required = {
        "candidate_id",
        "horizon_steps",
        "horizon_minutes",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
        "persistence_pred_bps",
    }
    stat_required = {
        "horizon_steps",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
    }

    for label, frame, required in [
        ("transformer", transformer, transformer_required),
        ("recurrent", recurrent, recurrent_required),
        ("tree", tree, tree_required),
        ("statistical", stat, stat_required),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"Missing {label} columns: {sorted(missing)}")

    for frame in (transformer, recurrent, tree, stat):
        frame["target_timestamp"] = pd.to_datetime(
            frame["target_timestamp"], errors="raise"
        )
        if "origin_timestamp" in frame.columns:
            frame["origin_timestamp"] = pd.to_datetime(
                frame["origin_timestamp"], errors="raise"
            )
        frame["horizon_steps"] = pd.to_numeric(
            frame["horizon_steps"], errors="raise"
        ).astype(int)

    transformer["config_id"] = transformer["config_id"].astype(str)
    recurrent["family"] = recurrent["family"].astype(str)
    recurrent["config_id"] = recurrent["config_id"].astype(str)
    tree["candidate_id"] = tree["candidate_id"].astype(str)

    if set(transformer["config_id"].unique()) != {TRANSFORMER_CONFIG_ID}:
        raise RuntimeError(
            f"Unexpected Transformer config set: {set(transformer['config_id'].unique())}"
        )
    if set(pd.to_numeric(transformer["lookback"], errors="raise").astype(int)) != {24}:
        raise RuntimeError("Unexpected Transformer lookback.")
    if set(pd.to_numeric(transformer["canonical_seed"], errors="raise").astype(int)) != {20260820}:
        raise RuntimeError("Unexpected Transformer canonical seed.")

    recurrent_star = recurrent[
        recurrent["family"].eq(RECURRENT_STAR[0])
        & recurrent["config_id"].eq(RECURRENT_STAR[1])
    ].copy()
    if len(recurrent_star) != ONE_MODEL_ROWS:
        raise RuntimeError(
            f"RECURRENT* rows={len(recurrent_star)} != {ONE_MODEL_ROWS}"
        )

    recurrent_seeds = set(
        pd.to_numeric(recurrent_star["canonical_seed"], errors="raise").astype(int)
    )
    if recurrent_seeds != {20260820}:
        raise RuntimeError(f"Unexpected RECURRENT* seed set: {recurrent_seeds}")

    if TREE_STAR not in set(tree["candidate_id"].unique()):
        raise RuntimeError("TREE*=LGB01 absent.")
    tree_star = tree[tree["candidate_id"].eq(TREE_STAR)].copy()
    if len(tree_star) != ONE_MODEL_ROWS:
        raise RuntimeError(f"TREE* rows={len(tree_star)} != {ONE_MODEL_ROWS}")

    stat_model_column = detect_stat_model_column(stat)
    stat[stat_model_column] = stat[stat_model_column].astype(str)
    if not {ARIMA_STAR, VAR_STAR}.issubset(set(stat[stat_model_column].unique())):
        raise RuntimeError("ARIMA*/VAR* absent from statistical predictions.")

    stat_selected = stat[
        stat[stat_model_column].isin([ARIMA_STAR, VAR_STAR])
    ].copy()
    if len(stat_selected) != STAT_SELECTED_EXPECTED_ROWS:
        raise RuntimeError(
            f"Selected statistical rows={len(stat_selected)} "
            f"!= {STAT_SELECTED_EXPECTED_ROWS}"
        )

    # Per-horizon row counts and no duplicate targets.
    for label, frame in [
        ("TRANSFORMER*", transformer),
        ("RECURRENT*", recurrent_star),
        ("TREE*", tree_star),
    ]:
        for horizon, expected in EXPECTED_COUNTS.items():
            sub = frame[frame["horizon_steps"].eq(horizon)]
            if len(sub) != expected:
                raise RuntimeError(
                    f"{label} H{horizon}: rows={len(sub)} != {expected}"
                )
            if sub["target_timestamp"].duplicated().any():
                raise RuntimeError(
                    f"{label} H{horizon}: duplicated target timestamps"
                )

    for config in [ARIMA_STAR, VAR_STAR]:
        model = stat_selected[stat_selected[stat_model_column].eq(config)]
        for horizon, expected in EXPECTED_COUNTS.items():
            sub = model[model["horizon_steps"].eq(horizon)]
            if len(sub) != expected:
                raise RuntimeError(
                    f"{config} H{horizon}: rows={len(sub)} != {expected}"
                )
            if sub["target_timestamp"].duplicated().any():
                raise RuntimeError(
                    f"{config} H{horizon}: duplicated target timestamps"
                )

    # Cross-family truth and Persistence alignment against TRANSFORMER*.
    key = ["horizon_steps", "target_timestamp"]

    for horizon, expected in EXPECTED_COUNTS.items():
        base = transformer[transformer["horizon_steps"].eq(horizon)][
            key + ["y_true_bps", "persistence_pred_bps"]
        ].sort_values(key).reset_index(drop=True)

        for label, ref in [
            ("RECURRENT*", recurrent_star),
            ("TREE*", tree_star),
        ]:
            comp = ref[ref["horizon_steps"].eq(horizon)][
                key + ["y_true_bps", "persistence_pred_bps"]
            ].sort_values(key).reset_index(drop=True)

            merged = base.merge(
                comp,
                on=key,
                suffixes=("_transformer", "_reference"),
                validate="one_to_one",
            )
            if len(merged) != expected:
                raise RuntimeError(
                    f"Alignment count mismatch {label} H{horizon}"
                )
            if not np.allclose(
                merged["y_true_bps_transformer"],
                merged["y_true_bps_reference"],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(f"Truth mismatch {label} H{horizon}")
            if not np.allclose(
                merged["persistence_pred_bps_transformer"],
                merged["persistence_pred_bps_reference"],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(f"Persistence mismatch {label} H{horizon}")

        for config in [ARIMA_STAR, VAR_STAR]:
            ref = stat_selected[
                stat_selected[stat_model_column].eq(config)
                & stat_selected["horizon_steps"].eq(horizon)
            ][key + ["y_true_bps"]].sort_values(key).reset_index(drop=True)

            merged = base.merge(
                ref,
                on=key,
                suffixes=("_transformer", "_reference"),
                validate="one_to_one",
            )
            if len(merged) != expected:
                raise RuntimeError(
                    f"Alignment count mismatch {config} H{horizon}"
                )
            if not np.allclose(
                merged["y_true_bps_transformer"],
                merged["y_true_bps_reference"],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(f"Truth mismatch {config} H{horizon}")

    return {
        "hashes": hashes,
        "transformer": transformer,
        "recurrent_star": recurrent_star,
        "tree_star": tree_star,
        "stat": stat,
        "stat_selected": stat_selected,
        "stat_model_column": stat_model_column,
    }


def make_long_aligned(data: dict[str, Any]) -> pd.DataFrame:
    transformer = data["transformer"]
    recurrent_star = data["recurrent_star"]
    tree_star = data["tree_star"]
    stat = data["stat_selected"]
    stat_model_column = data["stat_model_column"]

    parts = []

    # TRANSFORMER*
    t = transformer[
        [
            "horizon_steps",
            "horizon_minutes",
            "origin_timestamp",
            "target_timestamp",
            "y_true_bps",
            "y_pred_bps",
        ]
    ].copy()
    t["model_key"] = "TRANSFORMER*"
    t["model_source"] = "transformer"
    parts.append(t)

    # Persistence reconstructed from the same frozen Transformer source.
    p = transformer[
        [
            "horizon_steps",
            "horizon_minutes",
            "origin_timestamp",
            "target_timestamp",
            "y_true_bps",
            "persistence_pred_bps",
        ]
    ].copy()
    p = p.rename(columns={"persistence_pred_bps": "y_pred_bps"})
    p["model_key"] = "Persistence"
    p["model_source"] = "transformer_persistence"
    parts.append(p)

    # RECURRENT*=GRU/N06
    r = recurrent_star[
        [
            "horizon_steps",
            "horizon_minutes",
            "origin_timestamp",
            "target_timestamp",
            "y_true_bps",
            "y_pred_bps",
        ]
    ].copy()
    r["model_key"] = "RECURRENT*"
    r["model_source"] = "recurrent"
    parts.append(r)

    # TREE*=LGB01
    tr = tree_star[
        [
            "horizon_steps",
            "horizon_minutes",
            "target_timestamp",
            "y_true_bps",
            "y_pred_bps",
        ]
    ].copy()
    if "origin_timestamp" in tree_star.columns:
        tr["origin_timestamp"] = tree_star["origin_timestamp"].to_numpy()
    else:
        tr["origin_timestamp"] = pd.NaT
    tr["model_key"] = "TREE*"
    tr["model_source"] = "tree"
    parts.append(tr)

    # ARIMA* / VAR*
    for config, model_key in [(ARIMA_STAR, "ARIMA*"), (VAR_STAR, "VAR*")]:
        sub = stat[stat[stat_model_column].eq(config)].copy()
        part = sub[
            [
                "horizon_steps",
                "target_timestamp",
                "y_true_bps",
                "y_pred_bps",
            ]
        ].copy()
        if "horizon_minutes" in sub.columns:
            part["horizon_minutes"] = pd.to_numeric(
                sub["horizon_minutes"], errors="raise"
            ).astype(int)
        else:
            part["horizon_minutes"] = part["horizon_steps"].map(HORIZON_MINUTES)
        if "origin_timestamp" in sub.columns:
            part["origin_timestamp"] = sub["origin_timestamp"].to_numpy()
        else:
            part["origin_timestamp"] = pd.NaT
        part["model_key"] = model_key
        part["model_source"] = "statistical"
        parts.append(part)

    aligned = pd.concat(parts, ignore_index=True)
    aligned["horizon_steps"] = aligned["horizon_steps"].astype(int)
    aligned["horizon_minutes"] = aligned["horizon_minutes"].astype(int)
    aligned["target_timestamp"] = pd.to_datetime(
        aligned["target_timestamp"], errors="raise"
    )
    aligned["y_true_bps"] = pd.to_numeric(
        aligned["y_true_bps"], errors="raise"
    )
    aligned["y_pred_bps"] = pd.to_numeric(
        aligned["y_pred_bps"], errors="raise"
    )
    aligned["residual_bps"] = (
        aligned["y_true_bps"] - aligned["y_pred_bps"]
    )
    aligned["absolute_error_bps"] = np.abs(aligned["residual_bps"])

    expected_rows = 6 * ONE_MODEL_ROWS
    if len(aligned) != expected_rows:
        raise RuntimeError(
            f"Aligned long rows={len(aligned)} != {expected_rows}"
        )

    key = ["model_key", "horizon_steps", "target_timestamp"]
    if aligned.duplicated(key).any():
        raise RuntimeError(
            "Aligned long table has duplicate model/target keys."
        )

    for model in [
        "TRANSFORMER*",
        "Persistence",
        "RECURRENT*",
        "ARIMA*",
        "VAR*",
        "TREE*",
    ]:
        sub = aligned[aligned["model_key"].eq(model)]
        if len(sub) != ONE_MODEL_ROWS:
            raise RuntimeError(f"{model} aligned rows={len(sub)}")

    return aligned.sort_values(
        ["model_key", "horizon_steps", "target_timestamp"]
    ).reset_index(drop=True)


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def circular_moving_block_bootstrap_means(
    values: np.ndarray,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        raise RuntimeError("Series too short for bootstrap.")
    if block_length <= 0:
        raise RuntimeError("block_length must be positive.")

    blocks_needed = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    means = np.empty(repetitions, dtype=float)

    batch = 250
    written = 0
    while written < repetitions:
        current = min(batch, repetitions - written)
        starts = rng.integers(
            0,
            n,
            size=(current, blocks_needed),
            endpoint=False,
        )
        indices = (
            starts[..., None] + offsets[None, None, :]
        ) % n
        indices = indices.reshape(current, -1)[:, :n]
        samples = values[indices]
        means[written:written + current] = samples.mean(axis=1)
        written += current

    return means


def newey_west_dm(
    differential: np.ndarray,
    lag: int,
) -> dict[str, float]:
    d = np.asarray(differential, dtype=float)
    n = len(d)
    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.mean(centered * centered))
    lrv = gamma0

    for k in range(1, lag + 1):
        covariance = float(
            np.mean(centered[k:] * centered[:-k])
        )
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * covariance

    if lrv < 0 and abs(lrv) < 1e-12 * max(gamma0, 1.0):
        lrv = 0.0

    if lrv <= 0:
        return {
            "dm_hac_lrv": float(lrv),
            "dm_hac_standard_error_bps": float("nan"),
            "dm_hac_z": float("nan"),
            "dm_hac_p_value_two_sided": float("nan"),
        }

    standard_error = math.sqrt(lrv / n)
    z = mean_d / standard_error
    return {
        "dm_hac_lrv": float(lrv),
        "dm_hac_standard_error_bps": float(standard_error),
        "dm_hac_z": float(z),
        "dm_hac_p_value_two_sided": normal_two_sided_p(z),
    }


def comparison_arrays(
    aligned: pd.DataFrame,
    model_key: str,
    reference_key: str,
    horizon: int,
) -> pd.DataFrame:
    key = ["horizon_steps", "target_timestamp"]

    model = aligned[
        aligned["model_key"].eq(model_key)
        & aligned["horizon_steps"].eq(horizon)
    ][
        key + ["y_true_bps", "y_pred_bps", "absolute_error_bps"]
    ].rename(
        columns={
            "y_true_bps": "y_true_model",
            "y_pred_bps": "y_pred_model",
            "absolute_error_bps": "abs_error_model",
        }
    )

    reference = aligned[
        aligned["model_key"].eq(reference_key)
        & aligned["horizon_steps"].eq(horizon)
    ][
        key + ["y_true_bps", "y_pred_bps", "absolute_error_bps"]
    ].rename(
        columns={
            "y_true_bps": "y_true_reference",
            "y_pred_bps": "y_pred_reference",
            "absolute_error_bps": "abs_error_reference",
        }
    )

    paired = model.merge(
        reference,
        on=key,
        validate="one_to_one",
    ).sort_values("target_timestamp").reset_index(drop=True)

    expected = EXPECTED_COUNTS[horizon]
    if len(paired) != expected:
        raise RuntimeError(
            f"{model_key} vs {reference_key} H{horizon}: "
            f"{len(paired)} != {expected}"
        )

    if not np.allclose(
        paired["y_true_model"],
        paired["y_true_reference"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Truth mismatch {model_key} vs {reference_key} H{horizon}"
        )

    paired["loss_difference_bps"] = (
        paired["abs_error_reference"]
        - paired["abs_error_model"]
    )
    return paired


def run_inference(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows = []
    sensitivity_rows = []
    paired_rows = []

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for spec in PRIMARY_COMPARISONS:
        for horizon in HORIZONS:
            paired = comparison_arrays(
                aligned=aligned,
                model_key=spec["model_key"],
                reference_key=spec["reference_key"],
                horizon=horizon,
            )
            differential = paired["loss_difference_bps"].to_numpy(float)
            mean_improvement = float(differential.mean())

            paired_rows.append(
                {
                    "comparison": spec["comparison"],
                    "display_model": spec["display_model"],
                    "model_key": spec["model_key"],
                    "reference_key": spec["reference_key"],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_pairs": len(paired),
                    "model_wins": int(np.sum(differential > 0)),
                    "ties": int(np.sum(differential == 0)),
                    "model_losses": int(np.sum(differential < 0)),
                    "model_win_rate_pct": float(
                        100.0 * np.mean(differential > 0)
                    ),
                    "mean_improvement_bps": mean_improvement,
                    "median_improvement_bps": float(
                        np.median(differential)
                    ),
                }
            )

            primary_record = None

            for block in BOOTSTRAP_SENSITIVITY_BLOCKS:
                bootstrap_means = circular_moving_block_bootstrap_means(
                    differential,
                    block_length=block,
                    repetitions=BOOTSTRAP_REPETITIONS,
                    rng=rng,
                )

                alpha = 1.0 - BOOTSTRAP_CONFIDENCE
                lower = float(
                    np.quantile(bootstrap_means, alpha / 2.0)
                )
                upper = float(
                    np.quantile(bootstrap_means, 1.0 - alpha / 2.0)
                )
                probability = float(np.mean(bootstrap_means > 0))

                if lower > 0:
                    direction = "model_better"
                elif upper < 0:
                    direction = "reference_better"
                else:
                    direction = "not_distinguishable"

                record = {
                    "comparison": spec["comparison"],
                    "display_model": spec["display_model"],
                    "model_key": spec["model_key"],
                    "reference_key": spec["reference_key"],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_pairs": len(paired),
                    "mean_improvement_bps": mean_improvement,
                    "bootstrap_method": "circular_moving_block",
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "bootstrap_block_length": block,
                    "bootstrap_block_minutes": 5 * block,
                    "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
                    "bootstrap_ci_lower_bps": lower,
                    "bootstrap_ci_upper_bps": upper,
                    "bootstrap_probability_improvement": probability,
                    "bootstrap_direction": direction,
                    "bootstrap_seed_base": BOOTSTRAP_SEED,
                }
                sensitivity_rows.append(record)

                if block == BOOTSTRAP_PRIMARY_BLOCK:
                    primary_record = dict(record)

            if primary_record is None:
                raise RuntimeError("Primary bootstrap block not generated.")

            hac_lag = max(HAC_MIN_LAG, horizon - 1)
            dm = newey_west_dm(differential, lag=hac_lag)

            primary_record.update(
                {
                    "loss": "absolute_error",
                    "loss_differential_definition":
                        "absolute_error_reference - absolute_error_model",
                    "positive_value_meaning":
                        "TRANSFORMER*_improves_over_reference",
                    "dm_hac_lag": hac_lag,
                    "dm_hac_kernel": "Bartlett",
                    **dm,
                    "multiple_testing_correction": "none",
                }
            )
            p = primary_record["dm_hac_p_value_two_sided"]
            primary_record["dm_hac_distinguishable_0_05"] = bool(
                np.isfinite(p) and p < 0.05
            )
            primary_rows.append(primary_record)

    primary = pd.DataFrame(primary_rows).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)

    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["comparison", "horizon_steps", "bootstrap_block_length"]
    ).reset_index(drop=True)

    paired = pd.DataFrame(paired_rows).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)

    if len(primary) != 20:
        raise RuntimeError(f"Primary inference rows={len(primary)} != 20")
    if len(sensitivity) != 60:
        raise RuntimeError(f"Sensitivity rows={len(sensitivity)} != 60")
    if len(paired) != 20:
        raise RuntimeError(f"Paired summary rows={len(paired)} != 20")

    return primary, sensitivity, paired


def acf_values(values: np.ndarray, max_lag: int) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return [float("nan")] * max_lag

    out = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            out.append(float("nan"))
        else:
            out.append(
                float(np.dot(x[lag:], x[:-lag]) / denom)
            )
    return out


def transformer_annotated(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = aligned[
        aligned["model_key"].eq("TRANSFORMER*")
    ].copy()

    threshold_rows = []
    thresholds_by_horizon = {}

    for horizon in HORIZONS:
        y = physical[
            physical["horizon_steps"].eq(horizon)
        ]["y_true_bps"].to_numpy(float)

        q25, q50, q75 = np.quantile(y, [0.25, 0.50, 0.75])
        thresholds_by_horizon[horizon] = (q25, q50, q75)

        threshold_rows.append(
            {
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "q25_bps": float(q25),
                "q50_bps": float(q50),
                "q75_bps": float(q75),
            }
        )

    def level(row: pd.Series) -> str:
        q25, q50, q75 = thresholds_by_horizon[int(row["horizon_steps"])]
        value = float(row["y_true_bps"])
        if value <= q25:
            return "Q1"
        if value <= q50:
            return "Q2"
        if value <= q75:
            return "Q3"
        return "Q4"

    physical["traffic_level"] = physical.apply(level, axis=1)
    physical["target_hour"] = physical["target_timestamp"].dt.hour.astype(int)

    thresholds = pd.DataFrame(threshold_rows).sort_values(
        "horizon_steps"
    ).reset_index(drop=True)

    if len(physical) != ONE_MODEL_ROWS:
        raise RuntimeError(
            f"Annotated Transformer rows={len(physical)} != {ONE_MODEL_ROWS}"
        )
    if len(thresholds) != 4:
        raise RuntimeError("Traffic thresholds rows != 4.")

    return physical, thresholds


def residual_diagnostics(
    annotated: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary_rows = []
    distribution_rows = []
    traffic_rows = []
    hourly_rows = []
    acf_rows = []
    worst_rows = []

    for horizon in HORIZONS:
        group = annotated[
            annotated["horizon_steps"].eq(horizon)
        ].sort_values("target_timestamp").reset_index(drop=True)

        residual = group["residual_bps"].to_numpy(float)
        abs_error = group["absolute_error_bps"].to_numpy(float)

        summary_rows.append(
            {
                "model": "TRANSFORMER*",
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "n": len(group),
                "mae_bps": float(abs_error.mean()),
                "rmse_bps": float(np.sqrt(np.mean(residual ** 2))),
                "bias_bps": float(residual.mean()),
                "underprediction_pct": float(
                    100.0 * np.mean(residual > 0)
                ),
                "overprediction_pct": float(
                    100.0 * np.mean(residual < 0)
                ),
                "p95_abs_error_bps": float(
                    np.percentile(abs_error, 95)
                ),
            }
        )

        distribution_rows.append(
            {
                "model": "TRANSFORMER*",
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "residual_mean_bps": float(residual.mean()),
                "residual_std_bps": float(residual.std(ddof=0)),
                "residual_q05_bps": float(np.quantile(residual, 0.05)),
                "residual_q25_bps": float(np.quantile(residual, 0.25)),
                "residual_median_bps": float(np.quantile(residual, 0.50)),
                "residual_q75_bps": float(np.quantile(residual, 0.75)),
                "residual_q95_bps": float(np.quantile(residual, 0.95)),
            }
        )

        for traffic_level in ["Q1", "Q2", "Q3", "Q4"]:
            sub = group[group["traffic_level"].eq(traffic_level)]
            r = sub["residual_bps"].to_numpy(float)
            a = sub["absolute_error_bps"].to_numpy(float)
            traffic_rows.append(
                {
                    "model": "TRANSFORMER*",
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "traffic_level": traffic_level,
                    "n": len(sub),
                    "mae_bps": float(a.mean()),
                    "bias_bps": float(r.mean()),
                    "underprediction_pct": float(
                        100.0 * np.mean(r > 0)
                    ),
                    "p95_abs_error_bps": float(
                        np.percentile(a, 95)
                    ),
                }
            )

        for hour, sub in group.groupby("target_hour", sort=True):
            r = sub["residual_bps"].to_numpy(float)
            a = sub["absolute_error_bps"].to_numpy(float)
            hourly_rows.append(
                {
                    "model": "TRANSFORMER*",
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "target_hour": int(hour),
                    "n": len(sub),
                    "mae_bps": float(a.mean()),
                    "bias_bps": float(r.mean()),
                    "underprediction_pct": float(
                        100.0 * np.mean(r > 0)
                    ),
                }
            )

        for lag, acf in enumerate(
            acf_values(residual, ACF_MAX_LAG),
            start=1,
        ):
            acf_rows.append(
                {
                    "model": "TRANSFORMER*",
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "lag_steps": lag,
                    "lag_minutes": lag * 5,
                    "residual_acf": acf,
                }
            )

        worst = group.nlargest(
            TOP_WORST,
            "absolute_error_bps",
            keep="first",
        ).copy()
        worst = worst.sort_values(
            "absolute_error_bps",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

        for rank, row in enumerate(worst.itertuples(index=False), start=1):
            worst_rows.append(
                {
                    "model": "TRANSFORMER*",
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "rank_worst": rank,
                    "target_timestamp": row.target_timestamp,
                    "y_true_bps": row.y_true_bps,
                    "y_pred_bps": row.y_pred_bps,
                    "residual_bps": row.residual_bps,
                    "absolute_error_bps": row.absolute_error_bps,
                    "traffic_level": row.traffic_level,
                    "target_hour": row.target_hour,
                }
            )

    outputs = {
        "residual_summary": pd.DataFrame(summary_rows).sort_values(
            ["horizon_steps"]
        ).reset_index(drop=True),
        "residual_distribution": pd.DataFrame(distribution_rows).sort_values(
            ["horizon_steps"]
        ).reset_index(drop=True),
        "traffic_level": pd.DataFrame(traffic_rows).sort_values(
            ["horizon_steps", "traffic_level"]
        ).reset_index(drop=True),
        "hourly": pd.DataFrame(hourly_rows).sort_values(
            ["horizon_steps", "target_hour"]
        ).reset_index(drop=True),
        "residual_acf": pd.DataFrame(acf_rows).sort_values(
            ["horizon_steps", "lag_steps"]
        ).reset_index(drop=True),
        "worst_intervals": pd.DataFrame(worst_rows).sort_values(
            ["horizon_steps", "rank_worst"]
        ).reset_index(drop=True),
        "traffic_thresholds": thresholds.copy(),
    }

    if len(outputs["residual_summary"]) != 4:
        raise RuntimeError("Residual summary rows != 4.")
    if len(outputs["residual_distribution"]) != 4:
        raise RuntimeError("Residual distribution rows != 4.")
    if len(outputs["traffic_level"]) != 16:
        raise RuntimeError("Traffic-level rows != 16.")
    if len(outputs["residual_acf"]) != 192:
        raise RuntimeError("Residual ACF rows != 192.")
    if len(outputs["worst_intervals"]) != 20:
        raise RuntimeError("Worst intervals rows != 20.")

    return outputs


def output_paths() -> dict[str, Path]:
    return {
        "primary_inference": METRICS_DIR / f"{PREFIX}_primary_inference.csv",
        "bootstrap_sensitivity": METRICS_DIR / f"{PREFIX}_bootstrap_sensitivity.csv",
        "paired_summary": METRICS_DIR / f"{PREFIX}_paired_summary.csv",
        "residual_summary": METRICS_DIR / f"{PREFIX}_residual_summary.csv",
        "residual_distribution": METRICS_DIR / f"{PREFIX}_residual_distribution.csv",
        "traffic_thresholds": METRICS_DIR / f"{PREFIX}_traffic_thresholds.csv",
        "traffic_level": METRICS_DIR / f"{PREFIX}_traffic_level.csv",
        "hourly": METRICS_DIR / f"{PREFIX}_hourly.csv",
        "residual_acf": METRICS_DIR / f"{PREFIX}_residual_acf.csv",
        "worst_intervals": METRICS_DIR / f"{PREFIX}_worst_intervals.csv",
        "aligned_predictions": METRICS_DIR / f"{PREFIX}_aligned_predictions.parquet",
        "annotated_transformer_predictions":
            METRICS_DIR / f"{PREFIX}_annotated_transformer_predictions.parquet",
        "report": METRICS_DIR / f"{PREFIX}_report.txt",
        "manifest": METRICS_DIR / f"{PREFIX}_manifest.json",
    }


def expected_figure_paths() -> list[Path]:
    names = [
        "01_primary_inference",
        "02_transformer_bias",
        "03_transformer_underprediction",
        "04_transformer_residual_acf",
        "05_transformer_traffic_level_mae",
        "06_transformer_hourly_bias",
        "07_transformer_observed_predicted_h1_first24h",
        "08_transformer_residual_distribution",
    ]
    return [
        FIGURES_DIR / f"{PREFIX}_{stem}.{ext}"
        for stem in names
        for ext in ["png", "pdf"]
    ]


def refuse_existing_outputs() -> None:
    existing = [
        p
        for p in [*output_paths().values(), *expected_figure_paths()]
        if p.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing campaign outputs:\n"
            + "\n".join(str(p) for p in existing)
        )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(
            json_safe(obj),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    created = []
    for ext in ["png", "pdf"]:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        created.append(out)
    plt.close(fig)
    return created


def plot_outputs(
    primary: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    annotated: pd.DataFrame,
    aligned: pd.DataFrame,
) -> list[Path]:
    created = []

    fig, ax = plt.subplots(figsize=(12, 7))
    plot = primary.copy()
    plot["label"] = (
        plot["reference_key"]
        + " H"
        + plot["horizon_steps"].astype(str)
    )
    y = np.arange(len(plot))
    mean = plot["mean_improvement_bps"].to_numpy(float) / 1e6
    lower = plot["bootstrap_ci_lower_bps"].to_numpy(float) / 1e6
    upper = plot["bootstrap_ci_upper_bps"].to_numpy(float) / 1e6
    xerr = np.vstack([mean - lower, upper - mean])
    ax.errorbar(mean, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Mejora media de error absoluto (Mbit/s): referencia − TRANSFORMER*")
    ax.set_title("Contrastes inferenciales primarios — TRANSFORMER*")
    ax.grid(axis="x", alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_01_primary_inference",
    )

    summary = diagnostics["residual_summary"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        summary["horizon_minutes"],
        summary["bias_bps"] / 1e6,
        marker="o",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Bias observado − predicho (Mbit/s)")
    ax.set_title("Sesgo residual de TRANSFORMER* en June blind")
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_02_transformer_bias",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        summary["horizon_minutes"],
        summary["underprediction_pct"],
        marker="o",
    )
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Infrapredicción (%)")
    ax.set_title("Tasa de infrapredicción de TRANSFORMER*")
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_03_transformer_underprediction",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    acf = diagnostics["residual_acf"]
    for horizon, group in acf.groupby("horizon_steps", sort=True):
        ax.plot(
            group["lag_steps"],
            group["residual_acf"],
            label=f"H{horizon}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Lag (intervalos de 5 min)")
    ax.set_ylabel("ACF residual")
    ax.set_title("Autocorrelación residual de TRANSFORMER*")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_04_transformer_residual_acf",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    traffic = diagnostics["traffic_level"]
    levels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(len(levels))
    width = 0.18
    for i, horizon in enumerate(HORIZONS):
        g = traffic[
            traffic["horizon_steps"].eq(horizon)
        ].set_index("traffic_level")
        vals = [g.loc[level, "mae_bps"] / 1e6 for level in levels]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=f"H{horizon}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Cuartil de tráfico observado")
    ax.set_ylabel("MAE (Mbit/s)")
    ax.set_title("TRANSFORMER* — error por nivel de tráfico")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_05_transformer_traffic_level_mae",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    hourly = diagnostics["hourly"]
    for horizon, group in hourly.groupby("horizon_steps", sort=True):
        ax.plot(
            group["target_hour"],
            group["bias_bps"] / 1e6,
            marker="o",
            label=f"H{horizon}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Hora objetivo")
    ax.set_ylabel("Bias (Mbit/s)")
    ax.set_title("TRANSFORMER* — sesgo por hora")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_06_transformer_hourly_bias",
    )

    t_h1 = annotated[
        annotated["horizon_steps"].eq(1)
    ].sort_values("target_timestamp").head(288)
    p_h1 = aligned[
        aligned["model_key"].eq("Persistence")
        & aligned["horizon_steps"].eq(1)
    ].sort_values("target_timestamp").head(288)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        t_h1["target_timestamp"],
        t_h1["y_true_bps"] / 1e6,
        label="Observado",
    )
    ax.plot(
        t_h1["target_timestamp"],
        t_h1["y_pred_bps"] / 1e6,
        label="TRANSFORMER*",
    )
    ax.plot(
        p_h1["target_timestamp"],
        p_h1["y_pred_bps"] / 1e6,
        label="Persistence",
    )
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Bitrate (Mbit/s)")
    ax.set_title("June blind H1 — primeras 24 h")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_07_transformer_observed_predicted_h1_first24h",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for horizon in [1, 12]:
        values = (
            annotated[
                annotated["horizon_steps"].eq(horizon)
            ]["residual_bps"].to_numpy(float)
            / 1e6
        )
        ax.hist(
            values,
            bins=35,
            alpha=0.45,
            density=True,
            label=f"H{horizon}",
        )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Residual observado − predicho (Mbit/s)")
    ax.set_ylabel("Densidad")
    ax.set_title("TRANSFORMER* — distribución residual")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_08_transformer_residual_distribution",
    )

    if len(created) != 16:
        raise RuntimeError(f"Figure files={len(created)} != 16")

    return created


def build_report(
    hashes: dict[str, str],
    primary: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    stat_model_column: str,
    runtime_seconds: float,
) -> str:
    lines = [
        "=" * 126,
        "TRANSFORMER-INFERENCE-RESIDUALS-001 — INFERENCIA PAREADA Y DIAGNÓSTICO RESIDUAL JUNE",
        "=" * 126,
        f"Runner version:                 {RUNNER_VERSION}",
        f"Runner SHA-256:                 {sha256(Path(__file__).resolve())}",
        "",
        "GOBERNANZA",
        "-" * 126,
        "Reentrenamiento:                NO",
        "Refit:                          NO",
        "Tuning:                         NO",
        "Selección:                      NO",
        "Predicciones fuente modificadas:NO",
        "TRANSFORMER*:                   T01/L24",
        "RECURRENT*:                     GRU/N06",
        "ARIMA*:                         ARIMA(6,1,12)",
        "VAR*:                           VAR(5)",
        "TREE*:                          LGB01",
        "",
        "INPUTS",
        "-" * 126,
        f"033 SHA-256:                    {hashes['033']}",
        f"044 SHA-256:                    {hashes['044']}",
        f"028 SHA-256:                    {hashes['028']}",
        f"021 SHA-256:                    {hashes['021']}",
        f"008 SHA-256:                    {hashes['008']}",
        f"010 SHA-256:                    {hashes['010']}",
        f"Transformer predictions SHA:    {hashes['transformer_predictions']}",
        f"Recurrent predictions SHA:      {hashes['recurrent_predictions']}",
        f"Tree manifest SHA:              {hashes['tree_manifest']}",
        f"Tree predictions SHA:           {hashes['tree_predictions']}",
        f"Stat manifest SHA:              {hashes['stat_manifest']}",
        f"Stat predictions SHA:           {hashes['stat_predictions']}",
        f"Stat model column:              {stat_model_column}",
        "",
        "INFERENCIA",
        "-" * 126,
        "Loss:                           absolute error",
        "Differential:                   reference - TRANSFORMER*",
        "Positive:                       TRANSFORMER* improves",
        f"Bootstrap repetitions:          {BOOTSTRAP_REPETITIONS}",
        f"Primary block:                  {BOOTSTRAP_PRIMARY_BLOCK}",
        "Sensitivity blocks:             6 / 12 / 24",
        f"Confidence:                     {BOOTSTRAP_CONFIDENCE}",
        f"Random seed:                    {BOOTSTRAP_SEED}",
        "DM-HAC kernel:                  Bartlett",
        "DM-HAC lag:                     12 in H1/H3/H6/H12",
        "P-values:                       two-sided / unadjusted",
        "Multiple-testing correction:    NONE",
        "Primary comparisons:             20",
        "",
        "CONTRASTES PRIMARIOS",
        "-" * 126,
    ]

    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.comparison:28s} "
            f"H{row.horizon_steps:<2d} | "
            f"TRANSFORMER* vs {row.reference_key} | "
            f"improvement={row.mean_improvement_bps/1e6:+.4f} Mbit/s | "
            f"CI=[{row.bootstrap_ci_lower_bps/1e6:+.4f}, "
            f"{row.bootstrap_ci_upper_bps/1e6:+.4f}] | "
            f"P(improve)={row.bootstrap_probability_improvement:.4f} | "
            f"DM-HAC p={row.dm_hac_p_value_two_sided:.6g} | "
            f"bootstrap={row.bootstrap_direction}"
        )

    lines += [
        "",
        "RESIDUOS — TRANSFORMER*",
        "-" * 126,
    ]

    for row in diagnostics["residual_summary"].itertuples(index=False):
        lines.append(
            f"H{row.horizon_steps:<2d} | "
            f"MAE={row.mae_bps/1e6:.4f} | "
            f"RMSE={row.rmse_bps/1e6:.4f} | "
            f"bias={row.bias_bps/1e6:+.4f} | "
            f"underpred={row.underprediction_pct:.2f}% | "
            f"P95={row.p95_abs_error_bps/1e6:.4f} Mbit/s"
        )

    lines += [
        "",
        f"Runtime seconds:                {runtime_seconds:.3f}",
        "STATUS:                         PASS",
        "SOURCE PREDICTIONS MODIFIED:    NO",
        "SELECTION CHANGED:              NO",
        "TRAINING/REFIT/TUNING:          NO",
        "TRANSFORMER-CLOSURE-001:        NOT RUN",
        "PHASE F:                        NOT STARTED",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    print("=" * 126)
    print("TRANSFORMER-INFERENCE-RESIDUALS-001 — PREFLIGHT")
    print("=" * 126)

    data = load_validate_sources()

    hashes = data["hashes"]
    for key in [
        "033", "044", "028", "021", "008", "010",
        "transformer_predictions",
        "recurrent_predictions",
        "tree_manifest", "tree_predictions",
        "stat_manifest", "stat_predictions",
    ]:
        print_check(key, hashes[key])

    print()
    print("FUENTES CONGELADAS")
    print("-" * 126)
    print(f"Transformer rows:              {len(data['transformer'])}")
    print(f"RECURRENT* rows:               {len(data['recurrent_star'])}")
    print(f"TREE* rows:                    {len(data['tree_star'])}")
    print(f"Stat rows:                     {len(data['stat'])}")
    print(f"Stat selected ARIMA*/VAR*:     {len(data['stat_selected'])}")
    print(f"Stat model column:             {data['stat_model_column']}")
    print("TRANSFORMER*:                  T01/L24")
    print("Canonical seed:                20260820")
    print("RECURRENT*:                    GRU/N06")
    print("ARIMA*:                        ARIMA(6,1,12)")
    print("VAR*:                          VAR(5)")
    print("TREE*:                         LGB01")

    print()
    print("ALINEACIÓN")
    print("-" * 126)
    for horizon, expected in EXPECTED_COUNTS.items():
        print(
            f"H{horizon:<2d}: targets comunes={expected} | "
            "truth=PASS | Persistence=PASS"
        )

    print()
    print("INFERENCIA CONGELADA")
    print("-" * 126)
    print("Loss:                          absolute error")
    print("Differential:                  reference - TRANSFORMER*")
    print("Positive:                      TRANSFORMER* improves")
    print(f"Bootstrap repetitions:         {BOOTSTRAP_REPETITIONS}")
    print(f"Primary block:                 {BOOTSTRAP_PRIMARY_BLOCK}")
    print("Sensitivity blocks:            6 / 12 / 24")
    print(f"Confidence:                    {BOOTSTRAP_CONFIDENCE}")
    print(f"Random seed:                   {BOOTSTRAP_SEED}")
    print("DM-HAC kernel:                 Bartlett")
    print("DM-HAC lag:                    12")
    print("P-values:                      two-sided / unadjusted")
    print("Multiple-testing correction:   NONE")
    print("Primary comparisons:            20")
    print("Expected sensitivity rows:      60")

    print()
    print("RESIDUOS")
    print("-" * 126)
    print("Model:                         TRANSFORMER*")
    print("Residual:                      observed - predicted")
    print("Traffic levels:                Q1/Q2/Q3/Q4 per horizon")
    print("Hourly diagnostics:            YES")
    print("ACF:                           lags 1..48")
    print("Worst intervals:               top 5/horizon")

    print()
    print("REENTRENAMIENTO:                NO")
    print("REFIT:                          NO")
    print("TUNING:                         NO")
    print("SELECTION CHANGE:               NO")
    print("SOURCE PREDICTIONS MODIFIED:    NO")

    if args.preflight_only:
        print()
        print("TRANSFORMER-INFERENCE-RESIDUALS PREFLIGHT: PASS")
        print("BOOTSTRAP EJECUTADO EN PREFLIGHT: 0")
        print("DM-HAC EJECUTADO EN PREFLIGHT: 0")
        print("ARTEFACTOS GENERADOS EN PREFLIGHT: 0")
        return 0

    refuse_existing_outputs()

    print()
    print("=" * 126)
    print("INICIO TRANSFORMER-INFERENCE-RESIDUALS-001")
    print("=" * 126)
    print("No se entrenará ni reajustará ningún modelo.")
    print("Se usan exclusivamente predicciones June congeladas.")

    started = time.perf_counter()

    aligned = make_long_aligned(data)
    primary, sensitivity, paired = run_inference(aligned)
    annotated, thresholds = transformer_annotated(aligned)
    diagnostics = residual_diagnostics(annotated, thresholds)

    runtime_seconds = time.perf_counter() - started

    paths = output_paths()
    METRICS_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)

    atomic_csv(primary, paths["primary_inference"])
    atomic_csv(sensitivity, paths["bootstrap_sensitivity"])
    atomic_csv(paired, paths["paired_summary"])
    atomic_csv(diagnostics["residual_summary"], paths["residual_summary"])
    atomic_csv(
        diagnostics["residual_distribution"],
        paths["residual_distribution"],
    )
    atomic_csv(
        diagnostics["traffic_thresholds"],
        paths["traffic_thresholds"],
    )
    atomic_csv(diagnostics["traffic_level"], paths["traffic_level"])
    atomic_csv(diagnostics["hourly"], paths["hourly"])
    atomic_csv(diagnostics["residual_acf"], paths["residual_acf"])
    atomic_csv(
        diagnostics["worst_intervals"],
        paths["worst_intervals"],
    )
    atomic_parquet(aligned, paths["aligned_predictions"])
    atomic_parquet(
        annotated,
        paths["annotated_transformer_predictions"],
    )

    figure_paths = plot_outputs(
        primary=primary,
        diagnostics=diagnostics,
        annotated=annotated,
        aligned=aligned,
    )

    report = build_report(
        hashes=hashes,
        primary=primary,
        diagnostics=diagnostics,
        stat_model_column=data["stat_model_column"],
        runtime_seconds=runtime_seconds,
    )
    atomic_text(report, paths["report"])

    outputs = {}
    for output_name, output_path in paths.items():
        if output_name == "manifest":
            continue
        outputs[output_name] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "bytes": int(output_path.stat().st_size),
        }

    figure_outputs = [
        {
            "path": str(fig_path.resolve()),
            "sha256": sha256(fig_path),
            "bytes": int(fig_path.stat().st_size),
        }
        for fig_path in figure_paths
    ]

    validation = {
        "primary_rows_20": len(primary) == 20,
        "sensitivity_rows_60": len(sensitivity) == 60,
        "paired_summary_rows_20": len(paired) == 20,
        "residual_summary_rows_4":
            len(diagnostics["residual_summary"]) == 4,
        "residual_distribution_rows_4":
            len(diagnostics["residual_distribution"]) == 4,
        "traffic_threshold_rows_4":
            len(diagnostics["traffic_thresholds"]) == 4,
        "traffic_level_rows_16":
            len(diagnostics["traffic_level"]) == 16,
        "hourly_rows_96":
            len(diagnostics["hourly"]) == 96,
        "residual_acf_rows_192":
            len(diagnostics["residual_acf"]) == 192,
        "worst_intervals_rows_20":
            len(diagnostics["worst_intervals"]) == 20,
        "aligned_predictions_rows_14340":
            len(aligned) == 14340,
        "annotated_transformer_rows_2390":
            len(annotated) == 2390,
        "figure_files_16": len(figure_paths) == 16,
        "finite_primary_improvement":
            bool(np.isfinite(primary["mean_improvement_bps"]).all()),
        "source_transformer_hash_unchanged":
            sha256(TRANSFORMER_PREDICTIONS)
            == EXPECTED_SHA["transformer_predictions"],
        "source_recurrent_hash_unchanged":
            sha256(RECURRENT_PREDICTIONS)
            == EXPECTED_SHA["recurrent_predictions"],
        "source_tree_hash_unchanged":
            sha256(TREE_PREDICTIONS)
            == EXPECTED_SHA["tree_predictions"],
        "source_stat_hash_unchanged":
            sha256(STAT_PREDICTIONS)
            == EXPECTED_SHA["stat_predictions"],
        "selection_changed_false": True,
        "training_false": True,
        "refit_false": True,
        "tuning_false": True,
        "source_predictions_modified_false": True,
    }

    if not all(validation.values()):
        failed = [key for key, value in validation.items() if not value]
        raise RuntimeError(f"Final validation failures: {failed}")

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "runner": {
            "name": Path(__file__).name,
            "version": RUNNER_VERSION,
            "sha256": sha256(Path(__file__).resolve()),
        },
        "governance": {
            "training": False,
            "refit": False,
            "tuning": False,
            "selection_change": False,
            "source_predictions_modified": False,
            "transformer_star": "T01/L24",
            "canonical_seed": 20260820,
            "references": {
                "RECURRENT_star": "GRU/N06",
                "ARIMA_star": ARIMA_STAR,
                "VAR_star": VAR_STAR,
                "TREE_star": TREE_STAR,
            },
        },
        "authorities_and_sources": {
            key: {"sha256": value}
            for key, value in hashes.items()
        },
        "input_rows": {
            "transformer": len(data["transformer"]),
            "recurrent_star": len(data["recurrent_star"]),
            "tree_star": len(data["tree_star"]),
            "statistical_all": len(data["stat"]),
            "statistical_selected_arima_var":
                len(data["stat_selected"]),
        },
        "inference": {
            "loss": "absolute_error",
            "loss_differential":
                "absolute_error_reference - absolute_error_transformer",
            "positive_value_meaning":
                "TRANSFORMER*_improves_over_reference",
            "primary_comparisons": 20,
            "bootstrap_method": "circular_moving_block",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_primary_block": BOOTSTRAP_PRIMARY_BLOCK,
            "bootstrap_sensitivity_blocks":
                BOOTSTRAP_SENSITIVITY_BLOCKS,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "random_seed": BOOTSTRAP_SEED,
            "dm_method":
                "mean loss differential with Newey-West/HAC LRV and normal approximation",
            "newey_west_kernel": "Bartlett",
            "hac_lag_rule": "max(12, horizon_steps - 1)",
            "p_values": "two-sided_unadjusted",
            "multiple_testing_correction": "none",
        },
        "residual_diagnostics": {
            "models": ["TRANSFORMER*"],
            "residual_definition": "observed - predicted",
            "traffic_levels": ["Q1", "Q2", "Q3", "Q4"],
            "acf_max_lag": ACF_MAX_LAG,
            "worst_intervals_per_horizon": TOP_WORST,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seconds": runtime_seconds,
        },
        "validation": validation,
        "outputs": outputs,
        "figures": figure_outputs,
        "transformer_closure_run": False,
        "phase_f_started": False,
    }
    atomic_json(manifest, paths["manifest"])

    # Final source inmutability checks.
    for key, source in [
        ("transformer_predictions", TRANSFORMER_PREDICTIONS),
        ("recurrent_predictions", RECURRENT_PREDICTIONS),
        ("tree_predictions", TREE_PREDICTIONS),
        ("stat_predictions", STAT_PREDICTIONS),
    ]:
        if sha256(source) != EXPECTED_SHA[key]:
            raise RuntimeError(f"{key} source changed during run.")

    print()
    print(report)
    print("VALIDACIONES")
    print("-" * 126)
    for key, value in validation.items():
        print(f"{key:64s}: {'PASS' if value else 'FAIL'}")

    print()
    print(f"Manifest: {paths['manifest']}")
    print(f"Manifest SHA-256: {sha256(paths['manifest'])}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("SOURCE PREDICTIONS MODIFIED: NO")
    print("SELECTION CHANGED: NO")
    print("TRAINING/REFIT/TUNING: NO")
    print("TRANSFORMER-CLOSURE-001: NOT RUN")
    print("PHASE F: NOT STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
