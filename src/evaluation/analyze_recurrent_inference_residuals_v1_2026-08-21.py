#!/usr/bin/env python3
"""
RECURRENT-INFERENCE-RESIDUALS-001
=================================

Paired inferential comparison and residual diagnostics over ALREADY FROZEN
June blind predictions from Phase D recurrent networks.

Runner:
    analyze_recurrent_inference_residuals_v1_2026-08-21.py

Internal version:
    1.0.0

This runner DOES NOT:
- fit, refit, train, tune or select any model;
- modify source predictions;
- replace RNN*, LSTM*, GRU* or RECURRENT*;
- change ARIMA*, VAR* or TREE*;
- alter the historical inferential methodology.

Binding recurrent authorities:
- 023_recurrent_networks_protocol_2026-08-20.md
- 028_recurrent_june_blind_closure_2026-08-21.md

Frozen reference authorities:
- 008_statistical_final_representatives_freeze_2026-08-16.md
- 010_statistical_models_closure_freeze_2026-08-17.md
- 022_tree_phase_c_closure_2026-08-20.md

Frozen primary comparisons:
1. RNN*       vs Persistence  x H1/H3/H6/H12
2. LSTM*      vs Persistence  x H1/H3/H6/H12
3. GRU*       vs Persistence  x H1/H3/H6/H12
4. RECURRENT* vs ARIMA*       x H1/H3/H6/H12
5. RECURRENT* vs VAR*         x H1/H3/H6/H12
6. RECURRENT* vs TREE*        x H1/H3/H6/H12

Loss differential:
    absolute_error_reference - absolute_error_model
Positive:
    model improves over reference

Inference:
- circular moving-block bootstrap;
- 5000 repetitions;
- primary block length = 12;
- sensitivity = 6 / 12 / 24;
- 95 % percentile confidence interval;
- seed = 20260717;
- DM-HAC/Newey-West with Bartlett kernel;
- lag = max(12, horizon_steps - 1) = 12 here;
- no multiplicity correction.

Residual convention:
    residual = observed - predicted
    positive -> underprediction
    negative -> overprediction

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
CAMPAIGN_ID = "RECURRENT-INFERENCE-RESIDUALS-001"
PARENT_CAMPAIGN_ID = "UGR16-RECURRENT-NETWORKS-001"
FILE_TAG = "v1_2026-08-21"

ROOT = Path(__file__).resolve().parents[2]

P023 = ROOT / "docs/project_governance/023_recurrent_networks_protocol_2026-08-20.md"
P028 = ROOT / "docs/project_governance/028_recurrent_june_blind_closure_2026-08-21.md"
P008 = ROOT / "docs/project_governance/008_statistical_final_representatives_freeze_2026-08-16.md"
P010 = ROOT / "docs/project_governance/010_statistical_models_closure_freeze_2026-08-17.md"
P022 = ROOT / "docs/project_governance/022_tree_phase_c_closure_2026-08-20.md"

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
    "023": "af9d0d5984fd5e98c6660227f7bd98e246dbf934ad8df416439c2fd5bb59b174",
    "028": "26379146225af9f0e804696e518cb31e81d4060614d96d1fdff4494cfec19e82",
    "008": "cc98c4f7b74e2d0ce6515ac1a989bef2b20de0a921f63a5d0346591f2456ad1a",
    "010": "aeb527f1c7966cb91aeececd67d2da8e1f114236608e09296d9885ad1de0f530",
    "022": "fc8037337a5835c153f42f0f4eb53ef95cb25f7e6562b5aa65ff3e327c543de2",
    "recurrent_predictions": "a9f4f277bbf7f6415cd56ea1763e7afa74f307fd70f5d5c108844cbc1e964d19",
    "tree_manifest": "f0a296c21b3077b2f9c8788fbb118572a203021ae69074bb1a4a2b027f3d2b46",
    "tree_predictions": "7f787d74de5b593811e7988767d891d4aa70e0f8826bb1a302f20e4d0e7e61ad",
    "stat_manifest": "847f877144f2932e0b5967b1de0678898cb640434dbfb20f2f9805d7c29883a7",
    "stat_predictions": "21ebddf20569533a9d1ed66547cc4393f9c3b7e6721515c68eb3eb9dfc95d353",
}

HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

RECURRENT_EXPECTED_ROWS = 7170
TREE_EXPECTED_ROWS = 7170
STAT_EXPECTED_ROWS = 11950
STAT_SELECTED_EXPECTED_ROWS = 4780
PHYSICAL_RECURRENT_ROWS = 7170
ONE_REP_ROWS = 2390

RNN_STAR = ("SimpleRNN", "N03")
LSTM_STAR = ("LSTM", "N04")
GRU_STAR = ("GRU", "N06")
RECURRENT_STAR = GRU_STAR
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
        "comparison": "rnn_vs_persistence",
        "display_model": "RNN*",
        "model_key": "RNN*",
        "reference_key": "Persistence",
    },
    {
        "comparison": "lstm_vs_persistence",
        "display_model": "LSTM*",
        "model_key": "LSTM*",
        "reference_key": "Persistence",
    },
    {
        "comparison": "gru_vs_persistence",
        "display_model": "GRU*",
        "model_key": "GRU*",
        "reference_key": "Persistence",
    },
    {
        "comparison": "recurrent_vs_arima",
        "display_model": "RECURRENT*",
        "model_key": "GRU*",
        "reference_key": "ARIMA*",
    },
    {
        "comparison": "recurrent_vs_var",
        "display_model": "RECURRENT*",
        "model_key": "GRU*",
        "reference_key": "VAR*",
    },
    {
        "comparison": "recurrent_vs_tree",
        "display_model": "RECURRENT*",
        "model_key": "GRU*",
        "reference_key": "TREE*",
    },
]

METRICS_DIR = (
    ROOT
    / "results/metrics/recurrent_networks/inference_residuals/"
      "v1_2026-08-21"
)
FIGURES_DIR = (
    ROOT
    / "results/figures/recurrent_networks/"
      "recurrent_inference_residuals_v1_2026-08-21"
)

PREFIX = "ugr16_recurrent_inference_residuals_v1_2026-08-21"


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
        print(f"{label:58s}: PASS")
    else:
        print(f"{label:58s}: PASS | {value}")


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
    for key, path in [
        ("023", P023),
        ("028", P028),
        ("008", P008),
        ("010", P010),
        ("022", P022),
        ("recurrent_predictions", RECURRENT_PREDICTIONS),
        ("tree_manifest", TREE_MANIFEST),
        ("tree_predictions", TREE_PREDICTIONS),
        ("stat_manifest", STAT_MANIFEST),
        ("stat_predictions", STAT_PREDICTIONS),
    ]:
        hashes[key] = verify_sha(path, EXPECTED_SHA[key], key)

    rec = pd.read_parquet(RECURRENT_PREDICTIONS).copy()
    tree = pd.read_parquet(TREE_PREDICTIONS).copy()
    stat = normalize_stat_schema(pd.read_parquet(STAT_PREDICTIONS).copy())

    if len(rec) != RECURRENT_EXPECTED_ROWS:
        raise RuntimeError(
            f"Recurrent rows={len(rec)} != {RECURRENT_EXPECTED_ROWS}"
        )
    if len(tree) != TREE_EXPECTED_ROWS:
        raise RuntimeError(f"Tree rows={len(tree)} != {TREE_EXPECTED_ROWS}")
    if len(stat) != STAT_EXPECTED_ROWS:
        raise RuntimeError(f"Stat rows={len(stat)} != {STAT_EXPECTED_ROWS}")

    rec_required = {
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

    missing = rec_required.difference(rec.columns)
    if missing:
        raise RuntimeError(f"Missing recurrent columns: {sorted(missing)}")
    missing = tree_required.difference(tree.columns)
    if missing:
        raise RuntimeError(f"Missing tree columns: {sorted(missing)}")
    missing = stat_required.difference(stat.columns)
    if missing:
        raise RuntimeError(f"Missing statistical columns: {sorted(missing)}")

    for frame in (rec, tree, stat):
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

    rec["family"] = rec["family"].astype(str)
    rec["config_id"] = rec["config_id"].astype(str)
    tree["candidate_id"] = tree["candidate_id"].astype(str)

    observed_recurrent = set(
        rec[["family", "config_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_recurrent = {RNN_STAR, LSTM_STAR, GRU_STAR}
    if observed_recurrent != expected_recurrent:
        raise RuntimeError(
            f"Frozen recurrent representatives mismatch: {observed_recurrent}"
        )

    seeds = set(pd.to_numeric(rec["canonical_seed"], errors="raise").astype(int))
    if seeds != {20260820}:
        raise RuntimeError(f"Unexpected recurrent seed set: {seeds}")

    if TREE_STAR not in set(tree["candidate_id"].unique()):
        raise RuntimeError("TREE*=LGB01 absent.")

    stat_model_column = detect_stat_model_column(stat)
    stat[stat_model_column] = stat[stat_model_column].astype(str)

    if not {ARIMA_STAR, VAR_STAR}.issubset(
        set(stat[stat_model_column].unique())
    ):
        raise RuntimeError("ARIMA*/VAR* absent from statistical predictions.")

    stat_selected = stat[
        stat[stat_model_column].isin([ARIMA_STAR, VAR_STAR])
    ].copy()
    if len(stat_selected) != STAT_SELECTED_EXPECTED_ROWS:
        raise RuntimeError(
            f"ARIMA*/VAR* rows={len(stat_selected)} "
            f"!= {STAT_SELECTED_EXPECTED_ROWS}"
        )

    numeric_rec = rec[
        ["y_true_bps", "y_pred_bps", "persistence_pred_bps"]
    ].to_numpy(float)
    numeric_tree = tree[
        ["y_true_bps", "y_pred_bps", "persistence_pred_bps"]
    ].to_numpy(float)
    numeric_stat = stat_selected[
        ["y_true_bps", "y_pred_bps"]
    ].to_numpy(float)

    if not np.isfinite(numeric_rec).all():
        raise RuntimeError("Non-finite recurrent source values.")
    if not np.isfinite(numeric_tree).all():
        raise RuntimeError("Non-finite tree source values.")
    if not np.isfinite(numeric_stat).all():
        raise RuntimeError("Non-finite statistical source values.")

    # Physical recurrent coverage.
    recurrent_specs = {
        "RNN*": RNN_STAR,
        "LSTM*": LSTM_STAR,
        "GRU*": GRU_STAR,
    }
    for label, (family, config) in recurrent_specs.items():
        for horizon, expected in EXPECTED_COUNTS.items():
            n = len(
                rec[
                    rec["family"].eq(family)
                    & rec["config_id"].eq(config)
                    & rec["horizon_steps"].eq(horizon)
                ]
            )
            if n != expected:
                raise RuntimeError(
                    f"{label} H{horizon}: {n} != {expected}"
                )

    tree_star = tree[tree["candidate_id"].eq(TREE_STAR)].copy()
    if len(tree_star) != ONE_REP_ROWS:
        raise RuntimeError(f"TREE* rows={len(tree_star)} != {ONE_REP_ROWS}")

    for config in [ARIMA_STAR, VAR_STAR]:
        for horizon, expected in EXPECTED_COUNTS.items():
            n = len(
                stat_selected[
                    stat_selected[stat_model_column].eq(config)
                    & stat_selected["horizon_steps"].eq(horizon)
                ]
            )
            if n != expected:
                raise RuntimeError(
                    f"{config} H{horizon}: {n} != {expected}"
                )

    # Cross-family exact alignment.
    key = ["horizon_steps", "target_timestamp"]
    gru = rec[
        rec["family"].eq(GRU_STAR[0])
        & rec["config_id"].eq(GRU_STAR[1])
    ].copy()

    for horizon, expected in EXPECTED_COUNTS.items():
        base = gru[gru["horizon_steps"].eq(horizon)][
            key + ["y_true_bps", "persistence_pred_bps"]
        ].sort_values(key).reset_index(drop=True)

        refs = {
            "TREE*": tree_star[tree_star["horizon_steps"].eq(horizon)],
            "ARIMA*": stat_selected[
                stat_selected[stat_model_column].eq(ARIMA_STAR)
                & stat_selected["horizon_steps"].eq(horizon)
            ],
            "VAR*": stat_selected[
                stat_selected[stat_model_column].eq(VAR_STAR)
                & stat_selected["horizon_steps"].eq(horizon)
            ],
        }

        if base.duplicated(key).any():
            raise RuntimeError(f"RECURRENT* duplicate targets H{horizon}")

        for label, other_source in refs.items():
            other = other_source[
                key + ["y_true_bps"]
            ].sort_values(key).reset_index(drop=True)

            if other.duplicated(key).any():
                raise RuntimeError(f"{label} duplicate targets H{horizon}")

            merged = base.merge(
                other,
                on=key,
                suffixes=("_rec", "_ref"),
                validate="one_to_one",
            )
            if len(merged) != expected:
                raise RuntimeError(
                    f"{label} H{horizon} aligned rows={len(merged)} != {expected}"
                )
            if not np.allclose(
                merged["y_true_bps_rec"],
                merged["y_true_bps_ref"],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(f"Truth mismatch {label} H{horizon}")

        tree_p = tree_star[tree_star["horizon_steps"].eq(horizon)][
            key + ["persistence_pred_bps"]
        ].sort_values(key).reset_index(drop=True)

        pmerge = base.merge(
            tree_p,
            on=key,
            suffixes=("_rec", "_tree"),
            validate="one_to_one",
        )
        if len(pmerge) != expected:
            raise RuntimeError(f"Persistence alignment count H{horizon}")
        if not np.allclose(
            pmerge["persistence_pred_bps_rec"],
            pmerge["persistence_pred_bps_tree"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(f"Persistence mismatch H{horizon}")

    return {
        "hashes": hashes,
        "recurrent": rec,
        "tree": tree,
        "stat": stat,
        "stat_selected": stat_selected,
        "stat_model_column": stat_model_column,
    }


def make_long_aligned(data: dict[str, Any]) -> pd.DataFrame:
    rec = data["recurrent"]
    tree = data["tree"]
    stat = data["stat_selected"]
    stat_model_column = data["stat_model_column"]

    parts = []

    recurrent_map = {
        RNN_STAR: "RNN*",
        LSTM_STAR: "LSTM*",
        GRU_STAR: "GRU*",
    }
    for (family, config), model_key in recurrent_map.items():
        sub = rec[
            rec["family"].eq(family)
            & rec["config_id"].eq(config)
        ].copy()
        part = sub[
            [
                "horizon_steps",
                "horizon_minutes",
                "target_timestamp",
                "y_true_bps",
                "y_pred_bps",
            ]
        ].copy()
        if "origin_timestamp" in sub.columns:
            part["origin_timestamp"] = sub["origin_timestamp"].to_numpy()
        else:
            part["origin_timestamp"] = pd.NaT
        part["model_key"] = model_key
        part["model_source"] = "recurrent"
        parts.append(part)

    # Persistence reconstructed from frozen recurrent predictions.
    persistence = rec[
        rec["family"].eq(GRU_STAR[0])
        & rec["config_id"].eq(GRU_STAR[1])
    ][
        [
            "horizon_steps",
            "horizon_minutes",
            "target_timestamp",
            "y_true_bps",
            "persistence_pred_bps",
        ]
    ].copy()
    persistence = persistence.rename(
        columns={"persistence_pred_bps": "y_pred_bps"}
    )
    if "origin_timestamp" in rec.columns:
        origin = rec[
            rec["family"].eq(GRU_STAR[0])
            & rec["config_id"].eq(GRU_STAR[1])
        ]["origin_timestamp"].to_numpy()
        persistence["origin_timestamp"] = origin
    else:
        persistence["origin_timestamp"] = pd.NaT
    persistence["model_key"] = "Persistence"
    persistence["model_source"] = "recurrent_persistence"
    parts.append(persistence)

    tree_star = tree[tree["candidate_id"].eq(TREE_STAR)].copy()
    tree_part = tree_star[
        [
            "horizon_steps",
            "horizon_minutes",
            "target_timestamp",
            "y_true_bps",
            "y_pred_bps",
        ]
    ].copy()
    if "origin_timestamp" in tree_star.columns:
        tree_part["origin_timestamp"] = tree_star["origin_timestamp"].to_numpy()
    else:
        tree_part["origin_timestamp"] = pd.NaT
    tree_part["model_key"] = "TREE*"
    tree_part["model_source"] = "tree"
    parts.append(tree_part)

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

    # 7 distinct physical/reference tracks x 2390 rows.
    expected_rows = 7 * ONE_REP_ROWS
    if len(aligned) != expected_rows:
        raise RuntimeError(
            f"Aligned long rows={len(aligned)} != {expected_rows}"
        )

    key = ["model_key", "horizon_steps", "target_timestamp"]
    if aligned.duplicated(key).any():
        raise RuntimeError("Aligned long table has duplicate model/target keys.")

    for model in ["RNN*", "LSTM*", "GRU*", "Persistence", "TREE*", "ARIMA*", "VAR*"]:
        sub = aligned[aligned["model_key"].eq(model)]
        if len(sub) != ONE_REP_ROWS:
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

    # Reproduce historical methodology: one fixed RNG and deterministic iteration.
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
                        "model_improves_over_reference",
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

    if len(primary) != 24:
        raise RuntimeError(f"Primary inference rows={len(primary)} != 24")
    if len(sensitivity) != 72:
        raise RuntimeError(f"Sensitivity rows={len(sensitivity)} != 72")
    if len(paired) != 24:
        raise RuntimeError(f"Paired summary rows={len(paired)} != 24")

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


def recurrent_annotated(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = aligned[
        aligned["model_key"].isin(["RNN*", "LSTM*", "GRU*"])
    ].copy()

    threshold_rows = []
    thresholds_by_horizon = {}

    canonical = aligned[aligned["model_key"].eq("GRU*")].copy()

    for horizon in HORIZONS:
        y = canonical[
            canonical["horizon_steps"].eq(horizon)
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

    if len(physical) != PHYSICAL_RECURRENT_ROWS:
        raise RuntimeError(
            f"Annotated recurrent rows={len(physical)} != {PHYSICAL_RECURRENT_ROWS}"
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

    for model in ["RNN*", "LSTM*", "GRU*"]:
        for horizon in HORIZONS:
            group = annotated[
                annotated["model_key"].eq(model)
                & annotated["horizon_steps"].eq(horizon)
            ].sort_values("target_timestamp").reset_index(drop=True)

            residual = group["residual_bps"].to_numpy(float)
            abs_error = group["absolute_error_bps"].to_numpy(float)

            summary_rows.append(
                {
                    "model": model,
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
                    "model": model,
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
                        "model": model,
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
                        "model": model,
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
                        "model": model,
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
                        "model": model,
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
            ["model", "horizon_steps"]
        ).reset_index(drop=True),
        "residual_distribution": pd.DataFrame(distribution_rows).sort_values(
            ["model", "horizon_steps"]
        ).reset_index(drop=True),
        "traffic_level": pd.DataFrame(traffic_rows).sort_values(
            ["model", "horizon_steps", "traffic_level"]
        ).reset_index(drop=True),
        "hourly": pd.DataFrame(hourly_rows).sort_values(
            ["model", "horizon_steps", "target_hour"]
        ).reset_index(drop=True),
        "residual_acf": pd.DataFrame(acf_rows).sort_values(
            ["model", "horizon_steps", "lag_steps"]
        ).reset_index(drop=True),
        "worst_intervals": pd.DataFrame(worst_rows).sort_values(
            ["model", "horizon_steps", "rank_worst"]
        ).reset_index(drop=True),
        "traffic_thresholds": thresholds.copy(),
    }

    if len(outputs["residual_summary"]) != 12:
        raise RuntimeError("Residual summary rows != 12.")
    if len(outputs["residual_distribution"]) != 12:
        raise RuntimeError("Residual distribution rows != 12.")
    if len(outputs["traffic_level"]) != 48:
        raise RuntimeError("Traffic-level rows != 48.")
    if len(outputs["residual_acf"]) != 576:
        raise RuntimeError("Residual ACF rows != 576.")
    if len(outputs["worst_intervals"]) != 60:
        raise RuntimeError("Worst intervals rows != 60.")

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
        "annotated_recurrent_predictions": METRICS_DIR / f"{PREFIX}_annotated_recurrent_predictions.parquet",
        "report": METRICS_DIR / f"{PREFIX}_report.txt",
        "manifest": METRICS_DIR / f"{PREFIX}_manifest.json",
    }


def expected_figure_paths() -> list[Path]:
    names = [
        "01_primary_inference",
        "02_recurrent_bias",
        "03_recurrent_underprediction",
        "04_recurrent_residual_acf",
        "05_recurrent_traffic_level_mae",
        "06_recurrent_hourly_bias",
        "07_recurrent_star_observed_predicted_h1_first24h",
        "08_recurrent_star_residual_distribution",
    ]
    paths = []
    for name in names:
        for ext in ["png", "pdf"]:
            paths.append(FIGURES_DIR / f"{PREFIX}_{name}.{ext}")
    return paths


def refuse_existing_outputs() -> None:
    existing = [
        path for path in [
            *output_paths().values(),
            *expected_figure_paths(),
        ]
        if path.exists()
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


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "pdf"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_outputs(
    primary: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    annotated: pd.DataFrame,
    aligned: pd.DataFrame,
) -> list[Path]:
    created = []

    # 01 — primary inference
    fig, ax = plt.subplots(figsize=(12, 8))
    plot = primary.copy()
    plot["label"] = (
        plot["display_model"]
        + " vs "
        + plot["reference_key"]
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
    ax.set_xlabel("Mejora media de MAE (Mbit/s): referencia − modelo")
    ax.set_title("Contrastes inferenciales primarios — redes recurrentes")
    ax.grid(axis="x", alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_01_primary_inference",
    )

    # 02 — bias
    fig, ax = plt.subplots(figsize=(9, 6))
    summary = diagnostics["residual_summary"]
    for model, group in summary.groupby("model", sort=True):
        ax.plot(
            group["horizon_minutes"],
            group["bias_bps"] / 1e6,
            marker="o",
            label=model,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Bias observado − predicho (Mbit/s)")
    ax.set_title("Sesgo residual recurrente en June blind")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_02_recurrent_bias",
    )

    # 03 — underprediction
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in summary.groupby("model", sort=True):
        ax.plot(
            group["horizon_minutes"],
            group["underprediction_pct"],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Infrapredicción (%)")
    ax.set_title("Tasa de infrapredicción recurrente")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_03_recurrent_underprediction",
    )

    # 04 — ACF
    fig, ax = plt.subplots(figsize=(10, 6))
    acf = diagnostics["residual_acf"]
    for (model, horizon), group in acf.groupby(
        ["model", "horizon_steps"], sort=True
    ):
        ax.plot(
            group["lag_steps"],
            group["residual_acf"],
            label=f"{model} H{horizon}",
            alpha=0.8,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Lag (intervalos de 5 min)")
    ax.set_ylabel("ACF residual")
    ax.set_title("Autocorrelación residual recurrente")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_04_recurrent_residual_acf",
    )

    # 05 — traffic-level MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    traffic = diagnostics["traffic_level"]
    # Keep visual readable: RECURRENT*=GRU/N06, all horizons.
    sub = traffic[traffic["model"].eq("GRU*")]
    levels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(len(levels))
    width = 0.18
    for i, horizon in enumerate(HORIZONS):
        g = sub[sub["horizon_steps"].eq(horizon)].set_index("traffic_level")
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
    ax.set_title("RECURRENT*=GRU/N06 — error por nivel de tráfico")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_05_recurrent_traffic_level_mae",
    )

    # 06 — hourly bias RECURRENT*
    fig, ax = plt.subplots(figsize=(10, 6))
    hourly = diagnostics["hourly"]
    sub = hourly[hourly["model"].eq("GRU*")]
    for horizon, group in sub.groupby("horizon_steps", sort=True):
        ax.plot(
            group["target_hour"],
            group["bias_bps"] / 1e6,
            marker="o",
            label=f"H{horizon}",
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Hora objetivo")
    ax.set_ylabel("Bias (Mbit/s)")
    ax.set_title("RECURRENT*=GRU/N06 — sesgo por hora")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_06_recurrent_hourly_bias",
    )

    # 07 — H1 first 24 h
    gru_h1 = annotated[
        annotated["model_key"].eq("GRU*")
        & annotated["horizon_steps"].eq(1)
    ].sort_values("target_timestamp").head(288)
    persistence_h1 = aligned[
        aligned["model_key"].eq("Persistence")
        & aligned["horizon_steps"].eq(1)
    ].sort_values("target_timestamp").head(288)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        gru_h1["target_timestamp"],
        gru_h1["y_true_bps"] / 1e6,
        label="Observado",
    )
    ax.plot(
        gru_h1["target_timestamp"],
        gru_h1["y_pred_bps"] / 1e6,
        label="RECURRENT*=GRU/N06",
    )
    ax.plot(
        persistence_h1["target_timestamp"],
        persistence_h1["y_pred_bps"] / 1e6,
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
        FIGURES_DIR / f"{PREFIX}_07_recurrent_star_observed_predicted_h1_first24h",
    )

    # 08 — residual distribution H1/H12
    fig, ax = plt.subplots(figsize=(10, 6))
    gru = annotated[annotated["model_key"].eq("GRU*")]
    for horizon in [1, 12]:
        values = (
            gru[gru["horizon_steps"].eq(horizon)]["residual_bps"].to_numpy(float)
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
    ax.set_title("RECURRENT*=GRU/N06 — distribución residual")
    ax.legend()
    ax.grid(alpha=0.25)
    created += save_figure(
        fig,
        FIGURES_DIR / f"{PREFIX}_08_recurrent_star_residual_distribution",
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
        "RECURRENT-INFERENCE-RESIDUALS-001 — INFERENCIA PAREADA Y DIAGNÓSTICO RESIDUAL JUNE",
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
        "RNN* / LSTM* / GRU*:            CONGELADOS",
        "RECURRENT*:                     GRU/N06",
        "ARIMA*:                         ARIMA(6,1,12)",
        "VAR*:                           VAR(5)",
        "TREE*:                          LGB01",
        "",
        "INPUTS",
        "-" * 126,
        f"023 SHA-256:                    {hashes['023']}",
        f"028 SHA-256:                    {hashes['028']}",
        f"008 SHA-256:                    {hashes['008']}",
        f"010 SHA-256:                    {hashes['010']}",
        f"022 SHA-256:                    {hashes['022']}",
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
        "Differential:                   reference - model",
        "Positive:                       model improves",
        f"Bootstrap repetitions:          {BOOTSTRAP_REPETITIONS}",
        f"Primary block:                  {BOOTSTRAP_PRIMARY_BLOCK}",
        "Sensitivity blocks:             6 / 12 / 24",
        f"Confidence:                     {BOOTSTRAP_CONFIDENCE}",
        f"Random seed:                    {BOOTSTRAP_SEED}",
        "DM-HAC kernel:                  Bartlett",
        "DM-HAC lag:                     12 in H1/H3/H6/H12",
        "Multiple-testing correction:    NONE",
        "Primary comparisons:             24",
        "",
        "CONTRASTES PRIMARIOS",
        "-" * 126,
    ]

    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.comparison:24s} "
            f"H{row.horizon_steps:<2d} | "
            f"{row.display_model} vs {row.reference_key} | "
            f"improvement={row.mean_improvement_bps/1e6:+.4f} Mbit/s | "
            f"CI=[{row.bootstrap_ci_lower_bps/1e6:+.4f}, "
            f"{row.bootstrap_ci_upper_bps/1e6:+.4f}] | "
            f"P(improve)={row.bootstrap_probability_improvement:.4f} | "
            f"DM-HAC p={row.dm_hac_p_value_two_sided:.6g} | "
            f"bootstrap={row.bootstrap_direction}"
        )

    lines += [
        "",
        "RESIDUOS — REPRESENTANTES RECURRENTES",
        "-" * 126,
    ]

    for row in diagnostics["residual_summary"].itertuples(index=False):
        lines.append(
            f"{row.model:6s} H{row.horizon_steps:<2d} | "
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
        "RECURRENT-CLOSURE-001:          NOT RUN",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    print("=" * 126)
    print("RECURRENT-INFERENCE-RESIDUALS-001 — PREFLIGHT")
    print("=" * 126)

    data = load_validate_sources()

    hashes = data["hashes"]
    for key in [
        "023", "028", "008", "010", "022",
        "recurrent_predictions",
        "tree_manifest", "tree_predictions",
        "stat_manifest", "stat_predictions",
    ]:
        print_check(key, hashes[key])

    print()
    print("FUENTES CONGELADAS")
    print("-" * 126)
    print(f"Recurrent rows:                {len(data['recurrent'])}")
    print(f"Tree rows:                     {len(data['tree'])}")
    print(f"Stat rows:                     {len(data['stat'])}")
    print(f"Stat selected ARIMA*/VAR*:     {len(data['stat_selected'])}")
    print(f"Stat model column:             {data['stat_model_column']}")
    print("RNN*:                          SimpleRNN/N03")
    print("LSTM*:                         LSTM/N04")
    print("GRU*:                          GRU/N06")
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
    print("Differential:                  reference - model")
    print("Positive:                      model improves")
    print(f"Bootstrap repetitions:         {BOOTSTRAP_REPETITIONS}")
    print(f"Primary block:                 {BOOTSTRAP_PRIMARY_BLOCK}")
    print("Sensitivity blocks:            6 / 12 / 24")
    print(f"Confidence:                    {BOOTSTRAP_CONFIDENCE}")
    print(f"Random seed:                   {BOOTSTRAP_SEED}")
    print("DM-HAC kernel:                 Bartlett")
    print("DM-HAC lag:                    12")
    print("Multiple-testing correction:   NONE")
    print("Primary comparisons:            24")
    print("Expected sensitivity rows:      72")

    print()
    print("RESIDUOS")
    print("-" * 126)
    print("Models:                        RNN* / LSTM* / GRU*")
    print("Residual:                      observed - predicted")
    print("Traffic levels:                Q1/Q2/Q3/Q4 per horizon")
    print("Hourly diagnostics:            YES")
    print("ACF:                           lags 1..48")
    print("Worst intervals:               top 5/model/horizon")

    print()
    print("REENTRENAMIENTO:                NO")
    print("REFIT:                          NO")
    print("TUNING:                         NO")
    print("SELECTION CHANGE:               NO")
    print("SOURCE PREDICTIONS MODIFIED:    NO")

    if args.preflight_only:
        print()
        print("RECURRENT-INFERENCE-RESIDUALS PREFLIGHT: PASS")
        print("BOOTSTRAP EJECUTADO EN PREFLIGHT: 0")
        print("DM-HAC EJECUTADO EN PREFLIGHT: 0")
        print("ARTEFACTOS GENERADOS EN PREFLIGHT: 0")
        return 0

    refuse_existing_outputs()

    print()
    print("=" * 126)
    print("INICIO RECURRENT-INFERENCE-RESIDUALS-001")
    print("=" * 126)
    print("No se entrenará ni reajustará ningún modelo.")
    print("Se usan exclusivamente predicciones June congeladas.")

    started = time.perf_counter()

    aligned = make_long_aligned(data)
    primary, sensitivity, paired = run_inference(aligned)
    annotated, thresholds = recurrent_annotated(aligned)
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
        paths["annotated_recurrent_predictions"],
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
    for name, path in paths.items():
        if name == "manifest":
            continue
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }

    figure_outputs = []
    for path in figure_paths:
        figure_outputs.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )

    validation = {
        "primary_rows_24": len(primary) == 24,
        "sensitivity_rows_72": len(sensitivity) == 72,
        "paired_summary_rows_24": len(paired) == 24,
        "residual_summary_rows_12":
            len(diagnostics["residual_summary"]) == 12,
        "residual_distribution_rows_12":
            len(diagnostics["residual_distribution"]) == 12,
        "traffic_threshold_rows_4":
            len(diagnostics["traffic_thresholds"]) == 4,
        "traffic_level_rows_48":
            len(diagnostics["traffic_level"]) == 48,
        "residual_acf_rows_576":
            len(diagnostics["residual_acf"]) == 576,
        "worst_intervals_rows_60":
            len(diagnostics["worst_intervals"]) == 60,
        "aligned_predictions_rows_16730":
            len(aligned) == 16730,
        "annotated_recurrent_rows_7170":
            len(annotated) == 7170,
        "figure_files_16": len(figure_paths) == 16,
        "finite_primary_improvement":
            bool(np.isfinite(primary["mean_improvement_bps"]).all()),
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
            "recurrent_representatives": {
                "RNN_star": "SimpleRNN/N03",
                "LSTM_star": "LSTM/N04",
                "GRU_star": "GRU/N06",
                "RECURRENT_star": "GRU/N06",
            },
            "references": {
                "ARIMA_star": ARIMA_STAR,
                "VAR_star": VAR_STAR,
                "TREE_star": TREE_STAR,
            },
        },
        "authorities_and_sources": {
            key: {
                "sha256": value,
            }
            for key, value in hashes.items()
        },
        "input_rows": {
            "recurrent": len(data["recurrent"]),
            "tree": len(data["tree"]),
            "statistical_all": len(data["stat"]),
            "statistical_selected_arima_var":
                len(data["stat_selected"]),
        },
        "inference": {
            "loss": "absolute_error",
            "loss_differential":
                "absolute_error_reference - absolute_error_model",
            "positive_value_meaning":
                "model_improves_over_reference",
            "primary_comparisons": 24,
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
            "multiple_testing_correction": "none",
        },
        "residual_diagnostics": {
            "models": ["RNN*", "LSTM*", "GRU*"],
            "residual_definition": "observed - predicted",
            "traffic_levels": ["Q1", "Q2", "Q3", "Q4"],
            "acf_max_lag": ACF_MAX_LAG,
            "worst_intervals_per_model_horizon": TOP_WORST,
        },
        "runtime_seconds": runtime_seconds,
        "validation": validation,
        "outputs": outputs,
        "figures": figure_outputs,
        "recurrent_closure_run": False,
    }
    atomic_json(manifest, paths["manifest"])

    # Final self-hash checks.
    if sha256(RECURRENT_PREDICTIONS) != EXPECTED_SHA["recurrent_predictions"]:
        raise RuntimeError("Recurrent source changed during run.")
    if sha256(TREE_PREDICTIONS) != EXPECTED_SHA["tree_predictions"]:
        raise RuntimeError("Tree source changed during run.")
    if sha256(STAT_PREDICTIONS) != EXPECTED_SHA["stat_predictions"]:
        raise RuntimeError("Stat source changed during run.")

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
    print("RECURRENT-CLOSURE-001: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
