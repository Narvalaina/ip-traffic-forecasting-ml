#!/usr/bin/env python3
"""
GLOBAL-CROSS-FAMILY-ANALYSIS-001
================================

Phase F global cross-family synthesis over ALREADY FROZEN June predictions.

Binding protocol:
    049_phase_f_global_cross_family_protocol_2026-08-22.md

Preflight closure:
    050_global_cross_family_preflight_closure_2026-08-22.md

Primary Phase-F entries:
    Persistence
    ARIMA*       = ARIMA(6,1,12)
    VAR*         = VAR(5)
    TREE*        = LGB01
    RECURRENT*   = GRU/N06
    TRANSFORMER* = T01/L24

This runner DOES NOT:
- train, fit, refit, tune, select or reselect any model;
- execute a new bootstrap;
- execute a new DM-HAC test;
- modify frozen predictions;
- create a new GLOBAL* / BEST_GLOBAL_MODEL* identity.

It DOES:
- verify the frozen Phase-F authorities and preflight artifacts;
- load the frozen 14,340-row aligned June table;
- recompute common descriptive metrics;
- assemble the 60 historical primary pair/horizon conclusions;
- assemble the 180 historical bootstrap-sensitivity rows;
- independently reconcile all 60 historical mean loss differentials;
- compute global residual/extreme-error diagnostics;
- create stability and computational-cost inventories without inventing missing values;
- create the eight frozen Phase-F figure designs in PNG + PDF;
- produce a report and a scientific manifest.

With --preflight-only:
- verifies all frozen hashes and schemas;
- verifies exact row/key coverage;
- verifies that product output directories do not yet exist;
- executes 0 training/refit/tuning/selection;
- executes 0 bootstrap and 0 DM-HAC;
- writes 0 Phase-F scientific artifacts.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNNER_VERSION = "1.0.0"
CAMPAIGN_ID = "GLOBAL-CROSS-FAMILY-ANALYSIS-001"
PARENT_CAMPAIGN_ID = "UGR16-GLOBAL-CROSS-FAMILY-001"
FILE_TAG = "v1_2026-08-22"
PREFIX = f"ugr16_global_cross_family_{FILE_TAG}"

ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = (
    ROOT / "results/metrics/global_cross_family/analysis/v1_2026-08-22"
)
FIGURES_DIR = (
    ROOT / "results/figures/global_cross_family/v1_2026-08-22"
)

HORIZONS = (1, 3, 6, 12)
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
MODEL_ORDER = (
    "Persistence",
    "ARIMA*",
    "VAR*",
    "TREE*",
    "RECURRENT*",
    "TRANSFORMER*",
)
MODEL_INDEX = {model: i for i, model in enumerate(MODEL_ORDER)}
EXPECTED_ALIGNED_ROWS = 14340

AUTHORITY_HASHES = {
    "049": (
        ROOT / "docs/project_governance/049_phase_f_global_cross_family_protocol_2026-08-22.md",
        "755b54107cdf541e3fe01ed15731ac630ad0a2d91c5f7def3edd86baf9659917",
    ),
    "050": (
        ROOT / "docs/project_governance/050_global_cross_family_preflight_closure_2026-08-22.md",
        "5c4aac6a81d32d1e695a07cfc6e2bd5965be271e0c731bfbf67f557a74e6dea9",
    ),
    "010": (
        ROOT / "docs/project_governance/010_statistical_models_closure_freeze_2026-08-17.md",
        "aeb527f1c7966cb91aeececd67d2da8e1f114236608e09296d9885ad1de0f530",
    ),
    "022": (
        ROOT / "docs/project_governance/022_tree_phase_c_closure_2026-08-20.md",
        "fc8037337a5835c153f42f0f4eb53ef95cb25f7e6562b5aa65ff3e327c543de2",
    ),
    "029": (
        ROOT / "docs/project_governance/029_recurrent_inference_residuals_closure_2026-08-21.md",
        "ad1d3795cce0d126bc47ce4527988ef708daa148943bd35c3e4315c4e7c3bc40",
    ),
    "030": (
        ROOT / "docs/project_governance/030_recurrent_phase_d_closure_2026-08-21.md",
        "dbfeaba10c0d1360f2fdd971b5f381df730cbc6deab0a11c5da6d42fade89db9",
    ),
    "046": (
        ROOT / "docs/project_governance/046_transformer_inference_residuals_closure_2026-08-22.md",
        "9ce0f6d8237aae2f7cdda3852c7c0ddef01216b4e707e997b9d91c53a69f3eba",
    ),
    "047": (
        ROOT / "docs/project_governance/047_transformer_phase_e_closure_2026-08-22.md",
        "c46a51b12b7f8cea962bc282cd0451d6c6634d28ccceed3371ad8a2a331219c9",
    ),
    "048": (
        ROOT / "docs/project_governance/048_live_packages_v0_3_checkpoint_closure_2026-08-22.md",
        "93c89fd6f78397f0d7df557b24106aa038fb21e2d0a6d5b03a64171472fe8ea9",
    ),
}

PREFLIGHT_FILES = {
    "console": (
        ROOT / "results/metrics/global_cross_family/global_cross_family_preflight_console_v1_2026-08-22.txt",
        "c1f588bf6743d0d3018a1ea208932e0e806ea25c09860ecbe04a635da26aec8a",
    ),
    "registry": (
        ROOT / "results/metrics/global_cross_family/preflight/v1_2026-08-22/global_cross_family_preflight_registry_v1_2026-08-22.json",
        "17fc4e6d01c867018e57543c82f5392e375fe90685d44d83bfd66f6614c83bde",
    ),
    "report": (
        ROOT / "results/metrics/global_cross_family/preflight/v1_2026-08-22/global_cross_family_preflight_report_v1_2026-08-22.txt",
        "bb609699ab8973215730c5e79d8a5d3766804fa2f25a1d30cb7f6ed902570de8",
    ),
    "manifest": (
        ROOT / "results/metrics/global_cross_family/preflight/v1_2026-08-22/global_cross_family_preflight_manifest_v1_2026-08-22.json",
        "78f70e39565420dbaa8423552d6fb81ac7964c4b70348fb13fc99ff4dc2c0eaf",
    ),
}

JUNE_DATA = (
    ROOT / "data/processed/ugr16/june_week3_prepared_5min.parquet",
    "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3",
)

FROZEN_SOURCES = {
    "stat_predictions": (
        ROOT / "results/predictions/ugr16_statistical_june_blind_predictions.parquet",
        "21ebddf20569533a9d1ed66547cc4393f9c3b7e6721515c68eb3eb9dfc95d353",
    ),
    "tree_predictions": (
        ROOT / "results/predictions/tree_ensembles/ugr16_tree_june_blind_predictions.parquet",
        "7f787d74de5b593811e7988767d891d4aa70e0f8826bb1a302f20e4d0e7e61ad",
    ),
    "recurrent_predictions": (
        ROOT
        / "results/metrics/recurrent_networks/june_blind/"
          "RECURRENT_JUNE_BLIND_001_RESULTS_V1_2026-08-21/"
          "ugr16_recurrent_june_blind_v1_2026-08-21_predictions.parquet",
        "a9f4f277bbf7f6415cd56ea1763e7afa74f307fd70f5d5c108844cbc1e964d19",
    ),
    "transformer_predictions": (
        ROOT
        / "results/metrics/transformer_lite/june_blind/v2_2026-08-22/"
          "recovered_outputs/ugr16_transformer_june_blind_predictions_v2_2026-08-22.parquet",
        "7195fd3c656a0f34a40a8bbc29d262593c9de8fff6c4f3bc4c2c0b90c11cf750",
    ),
    "aligned_predictions": (
        ROOT
        / "results/metrics/transformer_lite/inference_residuals/v1_2026-08-22/"
          "ugr16_transformer_inference_residuals_v1_2026-08-22_aligned_predictions.parquet",
        "a4a84b409bb1b17c3ca4259ae9902fb0c1da557479faae6bdb0b1241b6d47075",
    ),
}

PAIR_MAP = {
    "arima_vs_persistence": ("ARIMA*", "Persistence"),
    "var_vs_persistence": ("VAR*", "Persistence"),
    "var_vs_arima": ("VAR*", "ARIMA*"),
    "lgbm_vs_persistence": ("TREE*", "Persistence"),
    "tree_vs_arima": ("TREE*", "ARIMA*"),
    "tree_vs_var": ("TREE*", "VAR*"),
    "gru_vs_persistence": ("RECURRENT*", "Persistence"),
    "recurrent_vs_arima": ("RECURRENT*", "ARIMA*"),
    "recurrent_vs_var": ("RECURRENT*", "VAR*"),
    "recurrent_vs_tree": ("RECURRENT*", "TREE*"),
    "transformer_vs_persistence": ("TRANSFORMER*", "Persistence"),
    "transformer_vs_arima": ("TRANSFORMER*", "ARIMA*"),
    "transformer_vs_var": ("TRANSFORMER*", "VAR*"),
    "transformer_vs_tree": ("TRANSFORMER*", "TREE*"),
    "transformer_vs_recurrent": ("TRANSFORMER*", "RECURRENT*"),
}

EXPECTED_PRIMARY_KEYS = set(PAIR_MAP)
EXPECTED_PRIMARY_ROW_COUNT = 60
EXPECTED_SENSITIVITY_ROW_COUNT = 180

# Historical exact stability data where a numeric summary is already frozen.
RECURRENT_STABILITY_SCORES = np.array(
    [0.917606657, 0.914613056, 0.935406855], dtype=float
)
TRANSFORMER_STABILITY_SCORES = np.array(
    [0.9923981236073321, 0.978038619, 0.996537875], dtype=float
)


class Gate:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any = "") -> None:
        ok = bool(condition)
        record = {"name": name, "pass": ok, "detail": str(detail)}
        self.checks.append(record)
        print(f"{name:<76} : {'PASS' if ok else 'FAIL'}"
              + (f" | {detail}" if str(detail) else ""))
        if not ok:
            raise RuntimeError(f"{name}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase F — síntesis global cross-family."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verifica el runner y las fuentes sin crear artefactos científicos.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI para las figuras PNG.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(gate: Gate, label: str, path: Path, expected_sha: str) -> str:
    gate.check(f"{label}: exists", path.is_file(), path)
    actual = sha256(path)
    gate.check(f"{label}: SHA-256", actual == expected_sha, actual)
    return actual


def output_paths() -> dict[str, Path]:
    names = {
        "global_metrics": "01_global_metrics.csv",
        "global_model_summary": "02_global_model_summary.csv",
        "global_pairwise_primary": "03_global_pairwise_primary.csv",
        "global_pairwise_sensitivity": "04_global_pairwise_sensitivity.csv",
        "dominance_by_horizon": "05_dominance_by_horizon.csv",
        "dominance_overall": "06_dominance_overall.csv",
        "residual_summary": "07_residual_summary.csv",
        "residual_distribution": "08_residual_distribution.csv",
        "traffic_thresholds": "09_traffic_thresholds.csv",
        "traffic_level": "10_traffic_level.csv",
        "hourly": "11_hourly.csv",
        "residual_acf": "12_residual_acf.csv",
        "worst_intervals": "13_worst_intervals.csv",
        "stability_inventory": "14_stability_inventory.csv",
        "cost_inventory": "15_cost_inventory.csv",
        "source_provenance": "16_source_provenance.csv",
        "aligned_predictions": "17_aligned_predictions.parquet",
        "report": "report.txt",
        "manifest": "manifest.json",
    }
    return {
        key: METRICS_DIR / f"{PREFIX}_{suffix}"
        for key, suffix in names.items()
    }


def figure_bases() -> dict[str, Path]:
    titles = {
        "01_mae_by_horizon": "01_mae_by_horizon",
        "02_ratio_skill_vs_persistence": "02_ratio_skill_vs_persistence",
        "03_global_score_mean_rank": "03_global_score_mean_rank",
        "04_primary_direction_matrix": "04_primary_direction_matrix",
        "05_bootstrap_sensitivity_global": "05_bootstrap_sensitivity_global",
        "06_p95_absolute_error": "06_p95_absolute_error",
        "07_bias_by_horizon": "07_bias_by_horizon",
        "08_residual_acf_summary": "08_residual_acf_summary",
    }
    return {
        key: FIGURES_DIR / f"{PREFIX}_{suffix}"
        for key, suffix in titles.items()
    }


def ensure_output_absent(gate: Gate) -> None:
    gate.check("Metrics output directory absent", not METRICS_DIR.exists(), METRICS_DIR)
    gate.check("Figures output directory absent", not FIGURES_DIR.exists(), FIGURES_DIR)


def verify_all_authorities(gate: Gate) -> None:
    for key, (path, expected) in AUTHORITY_HASHES.items():
        verify_file(gate, f"Governance {key}", path, expected)
    for key, (path, expected) in PREFLIGHT_FILES.items():
        verify_file(gate, f"Frozen preflight {key}", path, expected)
    verify_file(gate, "June dataset", JUNE_DATA[0], JUNE_DATA[1])
    for key, (path, expected) in FROZEN_SOURCES.items():
        verify_file(gate, f"Frozen source {key}", path, expected)


def load_registry(gate: Gate) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    registry_path, expected_sha = PREFLIGHT_FILES["registry"]
    gate.check("Registry SHA already verified", sha256(registry_path) == expected_sha)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    gate.check("Registry status PASS", registry.get("status") == "PASS")
    gate.check("Registry artifact count = 8", registry.get("artifact_count") == 8)
    artifacts = {
        item["source_id"]: item
        for item in registry["artifacts"]
    }
    gate.check("Registry unique source ids = 8", len(artifacts) == 8)
    return registry, artifacts


def validate_aligned(gate: Gate) -> pd.DataFrame:
    path, expected_sha = FROZEN_SOURCES["aligned_predictions"]
    gate.check("Aligned SHA already verified", sha256(path) == expected_sha)
    df = pd.read_parquet(path).copy()

    required = {
        "model_key",
        "horizon_steps",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
    }
    missing = sorted(required.difference(df.columns))
    gate.check("Aligned required schema", not missing, missing)

    df["model_key"] = df["model_key"].astype(str)
    df["horizon_steps"] = pd.to_numeric(
        df["horizon_steps"], errors="raise"
    ).astype(int)
    df["target_timestamp"] = pd.to_datetime(
        df["target_timestamp"], errors="raise"
    )
    df["y_true_bps"] = pd.to_numeric(df["y_true_bps"], errors="raise")
    df["y_pred_bps"] = pd.to_numeric(df["y_pred_bps"], errors="raise")

    gate.check("Aligned rows = 14340", len(df) == EXPECTED_ALIGNED_ROWS, len(df))
    gate.check("Aligned model set exact", set(df["model_key"]) == set(MODEL_ORDER))
    gate.check(
        "Aligned unique model/horizon/target keys",
        not df.duplicated(
            ["model_key", "horizon_steps", "target_timestamp"]
        ).any(),
    )

    for model in MODEL_ORDER:
        for horizon, expected in EXPECTED_COUNTS.items():
            sub = df[
                df["model_key"].eq(model)
                & df["horizon_steps"].eq(horizon)
            ]
            gate.check(
                f"{model} H{horizon} coverage",
                len(sub) == expected,
                len(sub),
            )

    # All models must share truth and targets within horizon.
    for horizon, expected in EXPECTED_COUNTS.items():
        base = df[
            df["model_key"].eq("Persistence")
            & df["horizon_steps"].eq(horizon)
        ][["target_timestamp", "y_true_bps"]].sort_values("target_timestamp")
        for model in MODEL_ORDER[1:]:
            other = df[
                df["model_key"].eq(model)
                & df["horizon_steps"].eq(horizon)
            ][["target_timestamp", "y_true_bps"]].sort_values("target_timestamp")
            merged = base.merge(
                other,
                on="target_timestamp",
                suffixes=("_base", "_other"),
                validate="one_to_one",
            )
            gate.check(
                f"Truth alignment {model} H{horizon}",
                len(merged) == expected
                and np.allclose(
                    merged["y_true_bps_base"],
                    merged["y_true_bps_other"],
                    rtol=0.0,
                    atol=1e-6,
                ),
            )

    df["residual_bps"] = df["y_true_bps"] - df["y_pred_bps"]
    df["absolute_error_bps"] = np.abs(df["residual_bps"])
    return df.sort_values(
        ["model_key", "horizon_steps", "target_timestamp"]
    ).reset_index(drop=True)


def june_mase_scale(gate: Gate) -> float:
    frame = pd.read_parquet(JUNE_DATA[0]).copy()
    gate.check("June rows = 2004", len(frame) == 2004, len(frame))
    gate.check("June target column", "bitrate_bps" in frame.columns)
    gate.check("June timestamp column", "timestamp" in frame.columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    train = pd.to_numeric(
        frame.loc[:1401, "bitrate_bps"], errors="raise"
    ).to_numpy(float)
    scale = float(np.mean(np.abs(np.diff(train))))
    gate.check("June common MASE scale finite", np.isfinite(scale) and scale > 0, scale)
    return scale


def get_first_numeric(
    row: pd.Series,
    candidates: tuple[str, ...],
    *,
    scale: float = 1.0,
    required: bool = True,
) -> float:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col]) * scale
    if required:
        raise RuntimeError(
            f"No se encuentra ninguna columna numérica entre {candidates}. "
            f"Disponibles: {list(row.index)}"
        )
    return float("nan")


def get_first_text(
    row: pd.Series,
    candidates: tuple[str, ...],
    *,
    default: str = "",
) -> str:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return default


def extract_improvement_bps(row: pd.Series) -> float:
    for col in ("mean_improvement_bps", "mean_loss_improvement_bps"):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    for col in ("mean_improvement_mbps", "mean_loss_improvement_mbps"):
        if col in row.index and pd.notna(row[col]):
            return float(row[col]) * 1e6
    raise RuntimeError("No se encuentra mean improvement en el artefacto histórico.")


def extract_ci_bps(row: pd.Series) -> tuple[float, float]:
    bps_pairs = (
        ("bootstrap_ci_lower_bps", "bootstrap_ci_upper_bps"),
        ("ci_lower_bps", "ci_upper_bps"),
    )
    mbps_pairs = (
        ("bootstrap_ci_lower_mbps", "bootstrap_ci_upper_mbps"),
        ("ci_lower_mbps", "ci_upper_mbps"),
    )
    for lo, hi in bps_pairs:
        if lo in row.index and hi in row.index:
            return float(row[lo]), float(row[hi])
    for lo, hi in mbps_pairs:
        if lo in row.index and hi in row.index:
            return float(row[lo]) * 1e6, float(row[hi]) * 1e6
    raise RuntimeError("No se encuentran límites CI en el artefacto histórico.")


def extract_block(row: pd.Series) -> int:
    for col in ("bootstrap_block_length", "block_length", "block"):
        if col in row.index and pd.notna(row[col]):
            return int(row[col])
    raise RuntimeError("No se encuentra block length.")


def derive_direction(lower: float, upper: float) -> str:
    if lower > 0:
        return "model_better"
    if upper < 0:
        return "reference_better"
    return "not_distinguishable"


def extract_direction(row: pd.Series, lower: float, upper: float) -> str:
    if "bootstrap_direction" in row.index and pd.notna(row["bootstrap_direction"]):
        direction = str(row["bootstrap_direction"])
        if direction in {"model_better", "reference_better", "not_distinguishable"}:
            return direction
    return derive_direction(lower, upper)


def canonical_pairwise_from_registry(
    gate: Gate,
    registry_artifacts: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for source_id, meta in registry_artifacts.items():
        path = Path(meta["exact_path"])
        expected_sha = meta["sha256"]
        gate.check(f"{source_id} exists", path.is_file(), path)
        gate.check(f"{source_id} SHA", sha256(path) == expected_sha, sha256(path))
        frame = pd.read_csv(path)
        frame["comparison"] = frame["comparison"].astype(str)
        frame["horizon_steps"] = pd.to_numeric(
            frame["horizon_steps"], errors="raise"
        ).astype(int)

        selected = frame[
            frame["comparison"].isin(meta["phase_f_comparison_keys"])
        ].copy()
        gate.check(
            f"{source_id} selected rows",
            len(selected) == meta["phase_f_selected_rows"],
            len(selected),
        )

        if meta["artifact_kind"] == "primary":
            for _, row in selected.iterrows():
                comparison = str(row["comparison"])
                if comparison not in PAIR_MAP:
                    raise RuntimeError(f"Comparison inesperada: {comparison}")
                model, reference = PAIR_MAP[comparison]
                horizon = int(row["horizon_steps"])
                lower, upper = extract_ci_bps(row)
                primary_rows.append(
                    {
                        "source_phase": meta["source_phase"],
                        "source_closure": meta["source_closure"],
                        "source_id": source_id,
                        "source_artifact": path.name,
                        "source_sha256": expected_sha,
                        "comparison": comparison,
                        "model": model,
                        "reference": reference,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "n_pairs": int(get_first_numeric(
                            row, ("n_pairs",), required=True
                        )),
                        "mean_improvement_bps_historical": extract_improvement_bps(row),
                        "bootstrap_method": get_first_text(
                            row,
                            ("bootstrap_method",),
                            default="circular moving-block bootstrap",
                        ),
                        "bootstrap_repetitions": int(get_first_numeric(
                            row, ("bootstrap_repetitions",), required=True
                        )),
                        "bootstrap_block_length": extract_block(row),
                        "bootstrap_confidence": get_first_numeric(
                            row, ("bootstrap_confidence",), required=True
                        ),
                        "bootstrap_ci_lower_bps": lower,
                        "bootstrap_ci_upper_bps": upper,
                        "bootstrap_probability_improvement": get_first_numeric(
                            row,
                            ("bootstrap_probability_improvement",),
                            required=True,
                        ),
                        "bootstrap_direction": extract_direction(row, lower, upper),
                        "dm_hac_lag": int(get_first_numeric(
                            row, ("dm_hac_lag", "hac_lag"), required=True
                        )),
                        "dm_hac_statistic": get_first_numeric(
                            row,
                            ("dm_hac_z", "dm_hac_statistic"),
                            required=False,
                        ),
                        "dm_hac_p_value_two_sided": get_first_numeric(
                            row,
                            ("dm_hac_p_value_two_sided",),
                            required=True,
                        ),
                        "multiple_testing_correction": get_first_text(
                            row,
                            ("multiple_testing_correction",),
                            default="none",
                        ),
                    }
                )
        else:
            for _, row in selected.iterrows():
                comparison = str(row["comparison"])
                if comparison not in PAIR_MAP:
                    raise RuntimeError(f"Comparison inesperada: {comparison}")
                model, reference = PAIR_MAP[comparison]
                horizon = int(row["horizon_steps"])
                lower, upper = extract_ci_bps(row)
                sensitivity_rows.append(
                    {
                        "source_phase": meta["source_phase"],
                        "source_closure": meta["source_closure"],
                        "source_id": source_id,
                        "source_artifact": path.name,
                        "source_sha256": expected_sha,
                        "comparison": comparison,
                        "model": model,
                        "reference": reference,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "bootstrap_block_length": extract_block(row),
                        "mean_improvement_bps_historical": extract_improvement_bps(row),
                        "bootstrap_ci_lower_bps": lower,
                        "bootstrap_ci_upper_bps": upper,
                        "bootstrap_probability_improvement": get_first_numeric(
                            row,
                            (
                                "bootstrap_probability_improvement",
                                "probability_improvement",
                            ),
                            required=True,
                        ),
                        "bootstrap_direction": extract_direction(row, lower, upper),
                    }
                )

    primary = pd.DataFrame(primary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)

    gate.check("Primary historical rows = 60", len(primary) == 60, len(primary))
    gate.check(
        "Sensitivity historical rows = 180",
        len(sensitivity) == 180,
        len(sensitivity),
    )
    gate.check(
        "Primary comparisons exact = 15",
        set(primary["comparison"]) == EXPECTED_PRIMARY_KEYS,
        sorted(set(primary["comparison"])),
    )
    gate.check(
        "Primary blocks all B12",
        set(primary["bootstrap_block_length"].astype(int)) == {12},
    )
    gate.check(
        "Sensitivity blocks all 6/12/24",
        set(sensitivity["bootstrap_block_length"].astype(int)) == {6, 12, 24},
    )
    gate.check(
        "Primary pair/horizon unique",
        not primary.duplicated(["comparison", "horizon_steps"]).any(),
    )
    gate.check(
        "Sensitivity pair/horizon/block unique",
        not sensitivity.duplicated(
            ["comparison", "horizon_steps", "bootstrap_block_length"]
        ).any(),
    )
    return (
        primary.sort_values(
            ["comparison", "horizon_steps"]
        ).reset_index(drop=True),
        sensitivity.sort_values(
            ["comparison", "horizon_steps", "bootstrap_block_length"]
        ).reset_index(drop=True),
    )


def reconcile_pairwise(
    gate: Gate,
    aligned: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = primary.copy()
    sensitivity = sensitivity.copy()

    recomputed: dict[tuple[str, int], float] = {}

    for comparison, (model, reference) in PAIR_MAP.items():
        for horizon, expected in EXPECTED_COUNTS.items():
            m = aligned[
                aligned["model_key"].eq(model)
                & aligned["horizon_steps"].eq(horizon)
            ][
                ["target_timestamp", "y_true_bps", "absolute_error_bps"]
            ].rename(
                columns={
                    "y_true_bps": "y_true_model",
                    "absolute_error_bps": "abs_error_model",
                }
            )
            r = aligned[
                aligned["model_key"].eq(reference)
                & aligned["horizon_steps"].eq(horizon)
            ][
                ["target_timestamp", "y_true_bps", "absolute_error_bps"]
            ].rename(
                columns={
                    "y_true_bps": "y_true_reference",
                    "absolute_error_bps": "abs_error_reference",
                }
            )
            pair = m.merge(
                r,
                on="target_timestamp",
                validate="one_to_one",
            )
            gate.check(
                f"Pair coverage {comparison} H{horizon}",
                len(pair) == expected,
                len(pair),
            )
            gate.check(
                f"Pair truth {comparison} H{horizon}",
                np.allclose(
                    pair["y_true_model"],
                    pair["y_true_reference"],
                    rtol=0.0,
                    atol=1e-6,
                ),
            )
            value = float(
                np.mean(
                    pair["abs_error_reference"].to_numpy(float)
                    - pair["abs_error_model"].to_numpy(float)
                )
            )
            recomputed[(comparison, horizon)] = value

    values = []
    deltas = []
    reconciled = []
    for _, row in primary.iterrows():
        key = (str(row["comparison"]), int(row["horizon_steps"]))
        value = recomputed[key]
        historical = float(row["mean_improvement_bps_historical"])
        ok = bool(np.isclose(
            value,
            historical,
            rtol=1e-12,
            atol=1e-6,
        ))
        gate.check(
            f"Mean differential reconcile {key[0]} H{key[1]}",
            ok,
            f"recomputed={value:.12f} historical={historical:.12f} delta={value-historical:.12g}",
        )
        values.append(value)
        deltas.append(value - historical)
        reconciled.append(ok)

    primary["mean_improvement_bps_recomputed"] = values
    primary["mean_improvement_delta_bps"] = deltas
    primary["mean_improvement_reconciled"] = reconciled

    # Sensitivity rows must carry exactly the same historical mean for the same pair/horizon.
    sens_values = []
    sens_deltas = []
    for _, row in sensitivity.iterrows():
        key = (str(row["comparison"]), int(row["horizon_steps"]))
        value = recomputed[key]
        hist = float(row["mean_improvement_bps_historical"])
        ok = bool(np.isclose(value, hist, rtol=1e-12, atol=1e-6))
        gate.check(
            f"Sensitivity mean reconcile {key[0]} H{key[1]} B{int(row['bootstrap_block_length'])}",
            ok,
            f"delta={value-hist:.12g}",
        )
        sens_values.append(value)
        sens_deltas.append(value - hist)
    sensitivity["mean_improvement_bps_recomputed"] = sens_values
    sensitivity["mean_improvement_delta_bps"] = sens_deltas

    # Global block-sensitivity classification.
    classes: dict[tuple[str, int], dict[str, Any]] = {}
    for (comparison, horizon), group in sensitivity.groupby(
        ["comparison", "horizon_steps"], sort=False
    ):
        directions = {
            int(row.bootstrap_block_length): str(row.bootstrap_direction)
            for row in group.itertuples()
        }
        labels = set(directions.values())
        if len(labels) == 1:
            cls = "strictly_consistent"
        elif "model_better" in labels and "reference_better" in labels:
            cls = "sign_reversal"
        else:
            cls = "weakened_not_reversed"
        classes[(str(comparison), int(horizon))] = {
            "sensitivity_class": cls,
            "direction_B6": directions.get(6, ""),
            "direction_B12": directions.get(12, ""),
            "direction_B24": directions.get(24, ""),
        }

    for col in (
        "sensitivity_class",
        "direction_B6",
        "direction_B12",
        "direction_B24",
    ):
        primary[col] = [
            classes[(str(r.comparison), int(r.horizon_steps))][col]
            for r in primary.itertuples()
        ]
        sensitivity[col] = [
            classes[(str(r.comparison), int(r.horizon_steps))][col]
            for r in sensitivity.itertuples()
        ]

    return primary, sensitivity


def smape_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(yhat)
    contrib = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    contrib[mask] = 2.0 * np.abs(y[mask] - yhat[mask]) / denom[mask]
    return float(100.0 * np.mean(contrib))


def compute_global_metrics(
    gate: Gate,
    aligned: pd.DataFrame,
    mase_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    persistence_mae = {}
    for horizon in HORIZONS:
        group = aligned[
            aligned["model_key"].eq("Persistence")
            & aligned["horizon_steps"].eq(horizon)
        ]
        persistence_mae[horizon] = float(group["absolute_error_bps"].mean())

    for model in MODEL_ORDER:
        for horizon in HORIZONS:
            group = aligned[
                aligned["model_key"].eq(model)
                & aligned["horizon_steps"].eq(horizon)
            ].sort_values("target_timestamp")
            y = group["y_true_bps"].to_numpy(float)
            yhat = group["y_pred_bps"].to_numpy(float)
            residual = y - yhat
            abs_error = np.abs(residual)
            mae = float(abs_error.mean())
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            ratio = mae / persistence_mae[horizon]
            rows.append(
                {
                    "model": model,
                    "model_order": MODEL_INDEX[model],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n": len(group),
                    "mae_bps": mae,
                    "mae_mbps": mae / 1e6,
                    "rmse_bps": rmse,
                    "rmse_mbps": rmse / 1e6,
                    "smape_pct": smape_pct(y, yhat),
                    "mase_common_june_scale": mae / mase_scale,
                    "mase_scale_bps": mase_scale,
                    "persistence_mae_bps": persistence_mae[horizon],
                    "mae_ratio_vs_persistence": ratio,
                    "skill_vs_persistence": 1.0 - ratio,
                    "bias_bps": float(residual.mean()),
                    "bias_mbps": float(residual.mean()) / 1e6,
                    "underprediction_pct": float(100.0 * np.mean(residual > 0)),
                    "overprediction_pct": float(100.0 * np.mean(residual < 0)),
                    "exact_prediction_pct": float(100.0 * np.mean(residual == 0)),
                    "p95_abs_error_bps": float(np.quantile(abs_error, 0.95)),
                    "p95_abs_error_mbps": float(np.quantile(abs_error, 0.95)) / 1e6,
                }
            )

    metrics = pd.DataFrame(rows)
    gate.check("Global metric rows = 24", len(metrics) == 24, len(metrics))

    metrics["mae_rank_within_horizon"] = metrics.groupby(
        "horizon_steps"
    )["mae_bps"].rank(method="average", ascending=True)

    summary_rows = []
    for model in MODEL_ORDER:
        group = metrics[metrics["model"].eq(model)]
        summary_rows.append(
            {
                "model": model,
                "model_order": MODEL_INDEX[model],
                "global_score_mean_mae_ratio_vs_persistence": float(
                    group["mae_ratio_vs_persistence"].mean()
                ),
                "mean_mae_rank_across_horizons": float(
                    group["mae_rank_within_horizon"].mean()
                ),
                "best_horizon_ratio_vs_persistence": float(
                    group["mae_ratio_vs_persistence"].min()
                ),
                "worst_horizon_ratio_vs_persistence": float(
                    group["mae_ratio_vs_persistence"].max()
                ),
                "horizons_ratio_below_1": int(
                    np.sum(group["mae_ratio_vs_persistence"] < 1.0)
                ),
                "horizons_ratio_equal_1": int(
                    np.sum(np.isclose(
                        group["mae_ratio_vs_persistence"], 1.0
                    ))
                ),
                "horizons": 4,
                "descriptive_only": True,
                "creates_global_star": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    gate.check("Global model summary rows = 6", len(summary) == 6, len(summary))
    return (
        metrics.sort_values(
            ["model_order", "horizon_steps"]
        ).reset_index(drop=True),
        summary.sort_values("model_order").reset_index(drop=True),
    )


def residual_outputs(
    gate: Gate,
    aligned: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    residual_rows = []
    dist_rows = []
    threshold_rows = []
    traffic_rows = []
    hourly_rows = []
    acf_rows = []
    worst_rows = []

    thresholds: dict[int, tuple[float, float, float]] = {}
    for horizon in HORIZONS:
        truth = aligned[
            aligned["model_key"].eq("Persistence")
            & aligned["horizon_steps"].eq(horizon)
        ]["y_true_bps"].to_numpy(float)
        q1, q2, q3 = np.quantile(truth, [0.25, 0.50, 0.75])
        thresholds[horizon] = (float(q1), float(q2), float(q3))
        threshold_rows.append(
            {
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "n_truth": len(truth),
                "q25_bps": float(q1),
                "q50_bps": float(q2),
                "q75_bps": float(q3),
                "definition": "per-horizon quartiles of common observed y_true",
            }
        )

    def acf(values: np.ndarray, lag: int) -> float:
        if len(values) <= lag:
            return float("nan")
        x = values[:-lag]
        y = values[lag:]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    for model in MODEL_ORDER:
        for horizon in HORIZONS:
            group = aligned[
                aligned["model_key"].eq(model)
                & aligned["horizon_steps"].eq(horizon)
            ].sort_values("target_timestamp").copy()
            residual = group["residual_bps"].to_numpy(float)
            abs_error = group["absolute_error_bps"].to_numpy(float)

            residual_rows.append(
                {
                    "model": model,
                    "model_order": MODEL_INDEX[model],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n": len(group),
                    "mae_bps": float(abs_error.mean()),
                    "rmse_bps": float(np.sqrt(np.mean(residual ** 2))),
                    "bias_bps": float(residual.mean()),
                    "underprediction_pct": float(100.0 * np.mean(residual > 0)),
                    "overprediction_pct": float(100.0 * np.mean(residual < 0)),
                    "p95_abs_error_bps": float(np.quantile(abs_error, 0.95)),
                }
            )
            s = pd.Series(residual)
            dist_rows.append(
                {
                    "model": model,
                    "model_order": MODEL_INDEX[model],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n": len(group),
                    "mean_bps": float(s.mean()),
                    "std_ddof1_bps": float(s.std(ddof=1)),
                    "median_bps": float(s.median()),
                    "q05_bps": float(s.quantile(0.05)),
                    "q25_bps": float(s.quantile(0.25)),
                    "q75_bps": float(s.quantile(0.75)),
                    "q95_bps": float(s.quantile(0.95)),
                    "min_bps": float(s.min()),
                    "max_bps": float(s.max()),
                    "skew": float(s.skew()),
                }
            )

            q1, q2, q3 = thresholds[horizon]
            y = group["y_true_bps"].to_numpy(float)
            labels = np.where(
                y <= q1,
                "Q1_low",
                np.where(
                    y <= q2,
                    "Q2_mid_low",
                    np.where(y <= q3, "Q3_mid_high", "Q4_high"),
                ),
            )
            group["traffic_level"] = labels
            for level in ("Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"):
                sub = group[group["traffic_level"].eq(level)]
                r = sub["residual_bps"].to_numpy(float)
                a = sub["absolute_error_bps"].to_numpy(float)
                traffic_rows.append(
                    {
                        "model": model,
                        "model_order": MODEL_INDEX[model],
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "traffic_level": level,
                        "n": len(sub),
                        "mae_bps": float(a.mean()) if len(sub) else float("nan"),
                        "rmse_bps": float(np.sqrt(np.mean(r ** 2))) if len(sub) else float("nan"),
                        "bias_bps": float(r.mean()) if len(sub) else float("nan"),
                        "underprediction_pct": (
                            float(100.0 * np.mean(r > 0)) if len(sub) else float("nan")
                        ),
                    }
                )

            group["hour"] = group["target_timestamp"].dt.hour.astype(int)
            for hour in range(24):
                sub = group[group["hour"].eq(hour)]
                r = sub["residual_bps"].to_numpy(float)
                a = sub["absolute_error_bps"].to_numpy(float)
                hourly_rows.append(
                    {
                        "model": model,
                        "model_order": MODEL_INDEX[model],
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "hour": hour,
                        "n": len(sub),
                        "mae_bps": float(a.mean()) if len(sub) else float("nan"),
                        "rmse_bps": float(np.sqrt(np.mean(r ** 2))) if len(sub) else float("nan"),
                        "bias_bps": float(r.mean()) if len(sub) else float("nan"),
                        "underprediction_pct": (
                            float(100.0 * np.mean(r > 0)) if len(sub) else float("nan")
                        ),
                    }
                )

            for lag in range(1, 49):
                acf_rows.append(
                    {
                        "model": model,
                        "model_order": MODEL_INDEX[model],
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "lag": lag,
                        "lag_minutes": lag * 5,
                        "residual_acf": acf(residual, lag),
                    }
                )

            worst = group.nlargest(
                5, "absolute_error_bps"
            ).sort_values(
                "absolute_error_bps", ascending=False
            )
            for rank, row in enumerate(worst.itertuples(), start=1):
                worst_rows.append(
                    {
                        "model": model,
                        "model_order": MODEL_INDEX[model],
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "rank": rank,
                        "target_timestamp": row.target_timestamp,
                        "y_true_bps": float(row.y_true_bps),
                        "y_pred_bps": float(row.y_pred_bps),
                        "residual_bps": float(row.residual_bps),
                        "absolute_error_bps": float(row.absolute_error_bps),
                    }
                )

    residual_df = pd.DataFrame(residual_rows)
    dist_df = pd.DataFrame(dist_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    traffic_df = pd.DataFrame(traffic_rows)
    hourly_df = pd.DataFrame(hourly_rows)
    acf_df = pd.DataFrame(acf_rows)
    worst_df = pd.DataFrame(worst_rows)

    expected = {
        "residual_summary": (residual_df, 24),
        "residual_distribution": (dist_df, 24),
        "traffic_thresholds": (threshold_df, 4),
        "traffic_level": (traffic_df, 96),
        "hourly": (hourly_df, 576),
        "residual_acf": (acf_df, 1152),
        "worst_intervals": (worst_df, 120),
    }
    for name, (frame, count) in expected.items():
        gate.check(f"{name} rows = {count}", len(frame) == count, len(frame))

    return (
        residual_df,
        dist_df,
        threshold_df,
        traffic_df,
        hourly_df,
        acf_df,
        worst_df,
    )


def dominance_tables(
    gate: Gate,
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for horizon in HORIZONS:
        h = primary[primary["horizon_steps"].eq(horizon)]
        for model in MODEL_ORDER:
            wins = losses = nd = 0
            for row in h.itertuples():
                if model not in {row.model, row.reference}:
                    continue
                if row.bootstrap_direction == "not_distinguishable":
                    nd += 1
                elif row.bootstrap_direction == "model_better":
                    if model == row.model:
                        wins += 1
                    else:
                        losses += 1
                elif row.bootstrap_direction == "reference_better":
                    if model == row.reference:
                        wins += 1
                    else:
                        losses += 1
                else:
                    raise RuntimeError(f"Dirección inesperada: {row.bootstrap_direction}")
            gate.check(
                f"Dominance opponents {model} H{horizon}",
                wins + losses + nd == 5,
                f"{wins}+{losses}+{nd}",
            )
            rows.append(
                {
                    "model": model,
                    "model_order": MODEL_INDEX[model],
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "wins": wins,
                    "losses": losses,
                    "not_distinguishable": nd,
                    "opponents": 5,
                }
            )
    by_horizon = pd.DataFrame(rows)
    overall = (
        by_horizon.groupby(["model", "model_order"], as_index=False)[
            ["wins", "losses", "not_distinguishable", "opponents"]
        ]
        .sum()
        .sort_values("model_order")
        .reset_index(drop=True)
    )
    gate.check("Dominance by horizon rows = 24", len(by_horizon) == 24, len(by_horizon))
    gate.check("Dominance overall rows = 6", len(overall) == 6, len(overall))
    gate.check(
        "Dominance overall opponents = 20 each",
        bool((overall["opponents"] == 20).all()),
        overall["opponents"].tolist(),
    )
    return by_horizon, overall


def stability_inventory(gate: Gate) -> pd.DataFrame:
    rec_mean = float(np.mean(RECURRENT_STABILITY_SCORES))
    rec_std = float(np.std(RECURRENT_STABILITY_SCORES, ddof=0))
    rec_range = float(np.ptp(RECURRENT_STABILITY_SCORES))
    tr_mean = float(np.mean(TRANSFORMER_STABILITY_SCORES))
    tr_std = float(np.std(TRANSFORMER_STABILITY_SCORES, ddof=0))
    tr_range = float(np.ptp(TRANSFORMER_STABILITY_SCORES))

    rows = [
        {
            "model": "Persistence",
            "family": "baseline",
            "stability_campaign_available": False,
            "seed_protocol": "not applicable / no-fit baseline",
            "canonical_seed": np.nan,
            "number_of_seeds": 0,
            "score_mean": np.nan,
            "score_std_ddof0": np.nan,
            "score_range": np.nan,
            "within_family_stability_summary": "No-fit baseline; seed stability not applicable.",
            "comparability_note": "Not comparable to seed-based fitted-model stability.",
            "source": "049_phase_f_global_cross_family_protocol_2026-08-22.md",
        },
        {
            "model": "ARIMA*",
            "family": "statistical",
            "stability_campaign_available": False,
            "seed_protocol": "seed-stability campaign not defined in frozen Phase B",
            "canonical_seed": np.nan,
            "number_of_seeds": 0,
            "score_mean": np.nan,
            "score_std_ddof0": np.nan,
            "score_range": np.nan,
            "within_family_stability_summary": "No seed-based stability campaign was defined; no value invented.",
            "comparability_note": "Statistical Phase-B protocol differs from stochastic multi-seed campaigns.",
            "source": "010_statistical_models_closure_freeze_2026-08-17.md",
        },
        {
            "model": "VAR*",
            "family": "statistical",
            "stability_campaign_available": False,
            "seed_protocol": "seed-stability campaign not defined in frozen Phase B",
            "canonical_seed": np.nan,
            "number_of_seeds": 0,
            "score_mean": np.nan,
            "score_std_ddof0": np.nan,
            "score_range": np.nan,
            "within_family_stability_summary": "No seed-based stability campaign was defined; no value invented.",
            "comparability_note": "Statistical Phase-B protocol differs from stochastic multi-seed campaigns.",
            "source": "010_statistical_models_closure_freeze_2026-08-17.md",
        },
        {
            "model": "TREE*",
            "family": "tree ensemble",
            "stability_campaign_available": True,
            "seed_protocol": "20260818 canonical; stability seeds 20260818/20260819/20260820; no best-seed selection",
            "canonical_seed": 20260818,
            "number_of_seeds": 3,
            "score_mean": np.nan,
            "score_std_ddof0": np.nan,
            "score_range": np.nan,
            "within_family_stability_summary": "Historical multi-seed stability completed; TREE*=LGB01 remained unchanged.",
            "comparability_note": "Do not compare dispersion numerically with D/E unless protocols are harmonized.",
            "source": "022_tree_phase_c_closure_2026-08-20.md",
        },
        {
            "model": "RECURRENT*",
            "family": "recurrent neural network",
            "stability_campaign_available": True,
            "seed_protocol": "20260820 canonical; seeds 20260820/20260821/20260822; no best-seed selection",
            "canonical_seed": 20260820,
            "number_of_seeds": 3,
            "score_mean": rec_mean,
            "score_std_ddof0": rec_std,
            "score_range": rec_range,
            "within_family_stability_summary": "GRU/N06 stability characterized across three frozen seeds; representative unchanged.",
            "comparability_note": "Within-family stability evidence only.",
            "source": "026_recurrent_stability_closure_2026-08-20.md",
        },
        {
            "model": "TRANSFORMER*",
            "family": "Transformer-lite",
            "stability_campaign_available": True,
            "seed_protocol": "20260820 canonical; seeds 20260820/20260821/20260822; no best-seed selection",
            "canonical_seed": 20260820,
            "number_of_seeds": 3,
            "score_mean": tr_mean,
            "score_std_ddof0": tr_std,
            "score_range": tr_range,
            "within_family_stability_summary": "T01/L24 stability characterized across three frozen seeds; canonical seed unchanged.",
            "comparability_note": "Within-family stability evidence only.",
            "source": "047_transformer_phase_e_closure_2026-08-22.md",
        },
    ]
    frame = pd.DataFrame(rows)
    gate.check("Stability inventory rows = 6", len(frame) == 6, len(frame))
    return frame


def cost_inventory(gate: Gate) -> pd.DataFrame:
    # Exact numbers are included only where they are already frozen in the historical evidence.
    # Missing cross-family-exact values are deliberately NA per 049.
    rows = [
        {
            "model": "Persistence",
            "family": "baseline",
            "hardware_environment": "not applicable / no-fit baseline",
            "fit_stages_or_count": "0",
            "fit_time_seconds": 0.0,
            "predict_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": 0,
            "source_artifact": "049_phase_f_global_cross_family_protocol_2026-08-22.md",
            "availability_status": "fit cost not applicable; prediction runtime not separately frozen",
            "comparability_note": "Baseline; no fitted model.",
        },
        {
            "model": "ARIMA*",
            "family": "statistical",
            "hardware_environment": "WSL2 / CPU statistical June run",
            "fit_stages_or_count": "initial June fit once; numerical attempts=2",
            "fit_time_seconds": 113.6166399989961,
            "predict_time_seconds": 56.444875916000456,
            "total_runtime_seconds": 170.06151591499655,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": np.nan,
            "source_artifact": "results/metrics/ugr16_statistical_june_blind_console.log",
            "availability_status": "exact frozen fit/predict time available",
            "comparability_note": "Raw CPU timing; not a fair universal speed ranking versus GPU phases.",
        },
        {
            "model": "VAR*",
            "family": "statistical",
            "hardware_environment": "WSL2 / CPU statistical June run",
            "fit_stages_or_count": "initial June fit once; numerical attempts=1",
            "fit_time_seconds": 0.007875800001784228,
            "predict_time_seconds": 0.5283532979956362,
            "total_runtime_seconds": 0.5362290979974204,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": np.nan,
            "source_artifact": "results/metrics/ugr16_statistical_june_blind_console.log",
            "availability_status": "exact frozen fit/predict time available",
            "comparability_note": "Raw CPU timing; not a fair universal speed ranking versus GPU phases.",
        },
        {
            "model": "TREE*",
            "family": "tree ensemble",
            "hardware_environment": "WSL2 / CPU; n_jobs=2 in frozen Phase C budget",
            "fit_stages_or_count": np.nan,
            "fit_time_seconds": np.nan,
            "predict_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": np.nan,
            "source_artifact": "022_tree_phase_c_closure_2026-08-20.md",
            "availability_status": "exact model-specific cross-family timing not populated here; no estimation",
            "comparability_note": "CPU Phase C; missing values intentionally left NA.",
        },
        {
            "model": "RECURRENT*",
            "family": "recurrent neural network",
            "hardware_environment": "Google Colab / Tesla T4 / CUDA 12.8",
            "fit_stages_or_count": "8 stages for GRU/N06 (4 inner epoch selections + 4 final refits)",
            "fit_time_seconds": np.nan,
            "predict_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": np.nan,
            "source_artifact": "028_recurrent_june_blind_closure_2026-08-21.md",
            "availability_status": "stage count frozen; exact aggregate model-specific timing left NA",
            "comparability_note": "GPU Phase D; raw runtime not comparable directly with CPU phases.",
        },
        {
            "model": "TRANSFORMER*",
            "family": "Transformer-lite",
            "hardware_environment": "Google Colab / Tesla T4 / CUDA 12.8",
            "fit_stages_or_count": "8 stages (4 inner epoch selections + 4 final refits)",
            "fit_time_seconds": np.nan,
            "predict_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "serialized_model_size_bytes": np.nan,
            "parameter_count": 8705,
            "source_artifact": "047_transformer_phase_e_closure_2026-08-22.md",
            "availability_status": "stage count and parameter count frozen; aggregate timing left NA",
            "comparability_note": "GPU Phase E; raw runtime not comparable directly with CPU phases.",
        },
    ]
    frame = pd.DataFrame(rows)
    gate.check("Cost inventory rows = 6", len(frame) == 6, len(frame))
    return frame


def provenance_table(
    registry_artifacts: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for key, (path, digest) in AUTHORITY_HASHES.items():
        rows.append(
            {
                "source_class": "governance",
                "source_id": key,
                "path": str(path),
                "basename": path.name,
                "sha256": digest,
                "role": "binding/frozen authority",
            }
        )
    for key, (path, digest) in PREFLIGHT_FILES.items():
        rows.append(
            {
                "source_class": "phase_f_preflight",
                "source_id": key,
                "path": str(path),
                "basename": path.name,
                "sha256": digest,
                "role": "frozen pre-full-run evidence",
            }
        )
    rows.append(
        {
            "source_class": "dataset",
            "source_id": "june_data",
            "path": str(JUNE_DATA[0]),
            "basename": JUNE_DATA[0].name,
            "sha256": JUNE_DATA[1],
            "role": "June frozen dataset",
        }
    )
    for key, (path, digest) in FROZEN_SOURCES.items():
        rows.append(
            {
                "source_class": "frozen_prediction_source",
                "source_id": key,
                "path": str(path),
                "basename": path.name,
                "sha256": digest,
                "role": "June frozen analytical source",
            }
        )
    for source_id, meta in registry_artifacts.items():
        rows.append(
            {
                "source_class": "historical_inference",
                "source_id": source_id,
                "path": meta["exact_path"],
                "basename": meta["basename"],
                "sha256": meta["sha256"],
                "role": f"{meta['source_phase']} {meta['artifact_kind']} inference source",
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    residual: pd.DataFrame,
    acf_df: pd.DataFrame,
    dpi: int,
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)
    bases = figure_bases()
    generated: list[Path] = []

    def save(fig: plt.Figure, base: Path) -> None:
        png = base.with_suffix(".png")
        pdf = base.with_suffix(".pdf")
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        generated.extend([png, pdf])

    # 01 MAE by horizon.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model in MODEL_ORDER:
        g = metrics[metrics["model"].eq(model)].sort_values("horizon_steps")
        ax.plot(g["horizon_minutes"], g["mae_mbps"], marker="o", label=model)
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("MAE (Mbit/s)")
    ax.set_title("Phase F — MAE June por horizonte")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, bases["01_mae_by_horizon"])

    # 02 Ratio vs Persistence.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model in MODEL_ORDER:
        g = metrics[metrics["model"].eq(model)].sort_values("horizon_steps")
        ax.plot(
            g["horizon_minutes"],
            g["mae_ratio_vs_persistence"],
            marker="o",
            label=model,
        )
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("MAE ratio vs Persistence")
    ax.set_title("Phase F — Ratio de MAE frente a Persistence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, bases["02_ratio_skill_vs_persistence"])

    # 03 GlobalScore vs mean rank.
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.scatter(
        summary["global_score_mean_mae_ratio_vs_persistence"],
        summary["mean_mae_rank_across_horizons"],
        s=70,
    )
    for row in summary.itertuples():
        ax.annotate(
            row.model,
            (
                row.global_score_mean_mae_ratio_vs_persistence,
                row.mean_mae_rank_across_horizons,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axvline(1.0, linewidth=1.0)
    ax.set_xlabel("GlobalScore descriptivo (media ratio MAE)")
    ax.set_ylabel("Mean MAE rank")
    ax.set_title("Phase F — GlobalScore descriptivo y ranking medio")
    ax.grid(True, alpha=0.25)
    save(fig, bases["03_global_score_mean_rank"])

    # 04 Direct-pair net-win matrix across four horizons.
    n = len(MODEL_ORDER)
    matrix = np.zeros((n, n), dtype=float)
    labels = [["" for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(MODEL_ORDER):
        for j, b in enumerate(MODEL_ORDER):
            if i == j:
                labels[i][j] = "—"
                continue
            wins_a = wins_b = nd = 0
            subset = primary[
                primary.apply(
                    lambda r: {r["model"], r["reference"]} == {a, b},
                    axis=1,
                )
            ]
            for row in subset.itertuples():
                if row.bootstrap_direction == "not_distinguishable":
                    nd += 1
                elif row.bootstrap_direction == "model_better":
                    if row.model == a:
                        wins_a += 1
                    else:
                        wins_b += 1
                elif row.bootstrap_direction == "reference_better":
                    if row.reference == a:
                        wins_a += 1
                    else:
                        wins_b += 1
            matrix[i, j] = wins_a - wins_b
            labels[i][j] = f"{wins_a}-{wins_b}-{nd}"
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax, label="Victorias fila − victorias columna")
    ax.set_xticks(range(n), MODEL_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(n), MODEL_ORDER)
    ax.set_title("Phase F — Matriz de contraste directo (W-L-ND, 4 horizontes)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
    save(fig, bases["04_primary_direction_matrix"])

    # 05 Sensitivity classes.
    sensitivity_class = (
        primary[["comparison", "horizon_steps", "sensitivity_class"]]
        .drop_duplicates()
        ["sensitivity_class"]
        .value_counts()
        .reindex(
            ["strictly_consistent", "weakened_not_reversed", "sign_reversal"],
            fill_value=0,
        )
    )
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.bar(sensitivity_class.index, sensitivity_class.values)
    ax.set_ylabel("Comparaciones pair/horizon")
    ax.set_title("Phase F — Robustez a block length 6/12/24")
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(sensitivity_class.values):
        ax.text(i, value, str(int(value)), ha="center", va="bottom")
    save(fig, bases["05_bootstrap_sensitivity_global"])

    # 06 P95 absolute error.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model in MODEL_ORDER:
        g = residual[residual["model"].eq(model)].sort_values("horizon_steps")
        ax.plot(
            g["horizon_minutes"],
            g["p95_abs_error_bps"] / 1e6,
            marker="o",
            label=model,
        )
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("P95 |error| (Mbit/s)")
    ax.set_title("Phase F — Error absoluto extremo (P95)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, bases["06_p95_absolute_error"])

    # 07 Bias.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model in MODEL_ORDER:
        g = residual[residual["model"].eq(model)].sort_values("horizon_steps")
        ax.plot(
            g["horizon_minutes"],
            g["bias_bps"] / 1e6,
            marker="o",
            label=model,
        )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Bias observado − predicho (Mbit/s)")
    ax.set_title("Phase F — Bias residual June")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, bases["07_bias_by_horizon"])

    # 08 Maximum absolute residual ACF over lags 1..48.
    summary_acf = (
        acf_df.assign(abs_acf=acf_df["residual_acf"].abs())
        .groupby(["model", "horizon_steps"], as_index=False)["abs_acf"]
        .max()
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model in MODEL_ORDER:
        g = summary_acf[summary_acf["model"].eq(model)].sort_values("horizon_steps")
        ax.plot(
            [HORIZON_MINUTES[int(x)] for x in g["horizon_steps"]],
            g["abs_acf"],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Horizonte (min)")
    ax.set_ylabel("Máx. |ACF residual|, lags 1..48")
    ax.set_title("Phase F — Dependencia residual")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, bases["08_residual_acf_summary"])

    return generated


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def write_report(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    dominance: pd.DataFrame,
    residual: pd.DataFrame,
    sensitivity: pd.DataFrame,
    source_unchanged: dict[str, bool],
    elapsed: float,
) -> str:
    score_order = summary.sort_values(
        "global_score_mean_mae_ratio_vs_persistence"
    )
    sensitivity_counts = (
        primary[["comparison", "horizon_steps", "sensitivity_class"]]
        .drop_duplicates()["sensitivity_class"]
        .value_counts()
        .to_dict()
    )
    direction_counts = primary["bootstrap_direction"].value_counts().to_dict()

    lines = [
        "=" * 122,
        "GLOBAL-CROSS-FAMILY-ANALYSIS-001 — REPORT",
        "=" * 122,
        f"Runner version: {RUNNER_VERSION}",
        "Status: PASS",
        "",
        "GOBERNANZA",
        "-" * 122,
        "Training: 0",
        "Refit: 0",
        "Tuning: 0",
        "Selection / reselection: 0",
        "New bootstrap: 0",
        "New DM-HAC: 0",
        "Source predictions modified: NO",
        "Creates GLOBAL*: NO",
        "",
        "COBERTURA",
        "-" * 122,
        "Models: 6",
        "Aligned rows: 14340",
        "H1/H3/H6/H12: 602 / 600 / 597 / 591",
        "",
        "MÉTRICAS DESCRIPTIVAS",
        "-" * 122,
        "GlobalScore = mean_h(MAE_model,h / MAE_Persistence,h), descriptive only",
    ]
    for row in score_order.itertuples():
        lines.append(
            f"{row.model:<14} "
            f"GlobalScore={row.global_score_mean_mae_ratio_vs_persistence:.9f} "
            f"mean_MAE_rank={row.mean_mae_rank_across_horizons:.3f} "
            f"horizons_ratio<1={int(row.horizons_ratio_below_1)}"
        )
    lines += [
        "",
        "INFERENCIA HISTÓRICA INTEGRADA",
        "-" * 122,
        f"Primary rows: {len(primary)}",
        f"Sensitivity rows: {len(sensitivity)}",
        f"Primary direction counts: {direction_counts}",
        f"Sensitivity-class counts: {sensitivity_counts}",
        "All 60 historical mean differentials independently reconciled: YES",
        "P-values: historical, two-sided, unadjusted; multiplicity correction = NONE",
        "",
        "DOMINANCIA DIRECTA (20 comparaciones/modelo)",
        "-" * 122,
    ]
    for row in dominance.sort_values("model_order").itertuples():
        lines.append(
            f"{row.model:<14} wins={int(row.wins):2d} "
            f"losses={int(row.losses):2d} "
            f"ND={int(row.not_distinguishable):2d}"
        )
    lines += [
        "",
        "INMUTABILIDAD DE FUENTES",
        "-" * 122,
    ]
    for key, value in source_unchanged.items():
        lines.append(f"{key}: {'PASS' if value else 'FAIL'}")
    lines += [
        "",
        "CAUTELAS",
        "-" * 122,
        "La comparación es entre pipelines congelados bajo el mismo June/target/horizonte.",
        "No todos los pipelines usan la misma representación de entrada.",
        "Los runtimes CPU/GPU se conservan como contexto ingenieril, no como ranking universal.",
        "GlobalScore es descriptivo y NO crea una identidad GLOBAL*.",
        "Phase G permanece NOT AUTHORIZED.",
        "",
        f"Runtime Phase-F synthesis: {elapsed:.6f} s",
        "",
        "GLOBAL-CROSS-FAMILY-ANALYSIS-001: PASS",
        "PHASE-F-CLOSURE-001: NOT RUN",
        "PHASE G: NOT AUTHORIZED",
        "",
    ]
    return "\n".join(lines)


def run_preflight() -> int:
    gate = Gate()
    print("=" * 122)
    print("GLOBAL-CROSS-FAMILY-ANALYSIS-001 — RUNNER PREFLIGHT ONLY")
    print("=" * 122)
    print(f"Runner: {Path(__file__).resolve()}")
    print(f"Runner SHA-256: {sha256(Path(__file__).resolve())}")
    print()

    verify_all_authorities(gate)
    ensure_output_absent(gate)
    registry, artifacts = load_registry(gate)
    aligned = validate_aligned(gate)
    _ = june_mase_scale(gate)
    primary, sensitivity = canonical_pairwise_from_registry(gate, artifacts)

    # Preflight validates that the means can be reconciled; this is not bootstrap/DM.
    # It creates no scientific artifact.
    primary, sensitivity = reconcile_pairwise(
        gate, aligned, primary, sensitivity
    )

    gate.check("Training executed = 0", True, 0)
    gate.check("Refit executed = 0", True, 0)
    gate.check("Tuning executed = 0", True, 0)
    gate.check("Selection/reselection executed = 0", True, 0)
    gate.check("Bootstrap executed = 0", True, 0)
    gate.check("DM-HAC executed = 0", True, 0)
    gate.check("Scientific artifacts written = 0", True, 0)

    print()
    print("RUNNER PREFLIGHT SUMMARY")
    print("-" * 122)
    print(f"Checks PASS/TOTAL: {sum(x['pass'] for x in gate.checks)}/{len(gate.checks)}")
    print(f"Aligned rows: {len(aligned)}")
    print(f"Primary historical rows: {len(primary)}")
    print(f"Sensitivity historical rows: {len(sensitivity)}")
    print("Training/refit/tuning/selection: 0")
    print("New bootstrap: 0")
    print("New DM-HAC: 0")
    print("Scientific artifacts: 0")
    print("GLOBAL-CROSS-FAMILY-ANALYSIS-001: NOT RUN")
    print("PHASE-F-CLOSURE-001: NOT RUN")
    print("PHASE G: NOT AUTHORIZED")
    print()
    print("GLOBAL-CROSS-FAMILY ANALYSIS RUNNER PREFLIGHT: PASS")
    return 0


def run_full(dpi: int) -> int:
    t0 = time.perf_counter()
    gate = Gate()

    print("=" * 122)
    print("START GLOBAL-CROSS-FAMILY-ANALYSIS-001")
    print("=" * 122)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {sha256(Path(__file__).resolve())}")
    print("New training/refit/tuning/selection: 0")
    print("New bootstrap: 0")
    print("New DM-HAC: 0")
    print()

    verify_all_authorities(gate)
    ensure_output_absent(gate)
    registry, artifacts = load_registry(gate)
    aligned = validate_aligned(gate)
    mase_scale = june_mase_scale(gate)

    primary, sensitivity = canonical_pairwise_from_registry(gate, artifacts)
    primary, sensitivity = reconcile_pairwise(
        gate, aligned, primary, sensitivity
    )

    metrics, model_summary = compute_global_metrics(
        gate, aligned, mase_scale
    )
    dominance_h, dominance_all = dominance_tables(gate, primary)

    (
        residual_summary,
        residual_distribution,
        traffic_thresholds,
        traffic_level,
        hourly,
        residual_acf,
        worst_intervals,
    ) = residual_outputs(gate, aligned)

    stability = stability_inventory(gate)
    cost = cost_inventory(gate)
    provenance = provenance_table(artifacts)

    gate.check("Source provenance nonempty", len(provenance) > 0, len(provenance))

    # Final expected row counts before writing.
    expected_frames = {
        "global_metrics": (metrics, 24),
        "global_model_summary": (model_summary, 6),
        "global_pairwise_primary": (primary, 60),
        "global_pairwise_sensitivity": (sensitivity, 180),
        "dominance_by_horizon": (dominance_h, 24),
        "dominance_overall": (dominance_all, 6),
        "residual_summary": (residual_summary, 24),
        "residual_distribution": (residual_distribution, 24),
        "traffic_thresholds": (traffic_thresholds, 4),
        "traffic_level": (traffic_level, 96),
        "hourly": (hourly, 576),
        "residual_acf": (residual_acf, 1152),
        "worst_intervals": (worst_intervals, 120),
        "stability_inventory": (stability, 6),
        "cost_inventory": (cost, 6),
        "aligned_predictions": (aligned, 14340),
    }
    for key, (frame, expected) in expected_frames.items():
        gate.check(f"Final {key} rows", len(frame) == expected, f"{len(frame)}/{expected}")

    gate.check(
        "All primary mean reconciliations TRUE",
        bool(primary["mean_improvement_reconciled"].all()),
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=False)
    paths = output_paths()

    save_csv(metrics, paths["global_metrics"])
    save_csv(model_summary, paths["global_model_summary"])
    save_csv(primary, paths["global_pairwise_primary"])
    save_csv(sensitivity, paths["global_pairwise_sensitivity"])
    save_csv(dominance_h, paths["dominance_by_horizon"])
    save_csv(dominance_all, paths["dominance_overall"])
    save_csv(residual_summary, paths["residual_summary"])
    save_csv(residual_distribution, paths["residual_distribution"])
    save_csv(traffic_thresholds, paths["traffic_thresholds"])
    save_csv(traffic_level, paths["traffic_level"])
    save_csv(hourly, paths["hourly"])
    save_csv(residual_acf, paths["residual_acf"])
    save_csv(worst_intervals, paths["worst_intervals"])
    save_csv(stability, paths["stability_inventory"])
    save_csv(cost, paths["cost_inventory"])
    save_csv(provenance, paths["source_provenance"])
    aligned.to_parquet(paths["aligned_predictions"], index=False)

    figure_paths = make_figures(
        metrics=metrics,
        summary=model_summary,
        primary=primary,
        sensitivity=sensitivity,
        residual=residual_summary,
        acf_df=residual_acf,
        dpi=dpi,
    )
    gate.check("Figure files = 16", len(figure_paths) == 16, len(figure_paths))
    gate.check("All figure files exist", all(p.is_file() for p in figure_paths))

    # Check source immutability after all scientific outputs have been generated.
    source_unchanged: dict[str, bool] = {}
    for key, (path, expected_sha) in FROZEN_SOURCES.items():
        actual = sha256(path)
        ok = actual == expected_sha
        source_unchanged[key] = ok
        gate.check(f"Source unchanged after run: {key}", ok, actual)

    elapsed = time.perf_counter() - t0
    report_text = write_report(
        metrics=metrics,
        summary=model_summary,
        primary=primary,
        dominance=dominance_all,
        residual=residual_summary,
        sensitivity=sensitivity,
        source_unchanged=source_unchanged,
        elapsed=elapsed,
    )
    paths["report"].write_text(report_text, encoding="utf-8", newline="\n")

    # Assemble artifact hashes (manifest excluded to avoid self-reference).
    artifacts_meta = {}
    for key, path in paths.items():
        if key == "manifest":
            continue
        gate.check(f"Output exists: {key}", path.is_file(), path)
        artifacts_meta[key] = artifact_meta(path)

    figures_meta = [
        artifact_meta(path)
        for path in sorted(figure_paths, key=lambda p: p.name)
    ]

    sensitivity_class_counts = (
        primary[["comparison", "horizon_steps", "sensitivity_class"]]
        .drop_duplicates()["sensitivity_class"]
        .value_counts()
        .to_dict()
    )

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "elapsed_seconds": elapsed,
        },
        "governance": {
            "protocol_049_sha256": AUTHORITY_HASHES["049"][1],
            "preflight_closure_050_sha256": AUTHORITY_HASHES["050"][1],
            "training": False,
            "refit": False,
            "tuning": False,
            "selection": False,
            "reselection": False,
            "new_bootstrap_executions": 0,
            "new_dm_hac_executions": 0,
            "source_predictions_modified": False,
            "creates_global_star": False,
            "phase_g_authorized": False,
        },
        "design": {
            "models": list(MODEL_ORDER),
            "horizons": list(HORIZONS),
            "expected_counts": EXPECTED_COUNTS,
            "aligned_rows": len(aligned),
            "mase_scale_bps": mase_scale,
            "primary_pairs": len(PAIR_MAP),
            "primary_pair_horizon_rows": len(primary),
            "sensitivity_rows": len(sensitivity),
            "bootstrap_primary_block": 12,
            "bootstrap_sensitivity_blocks": [6, 12, 24],
            "multiple_testing_correction": "NONE / historical unadjusted p-values",
        },
        "validation": {
            "global_metrics_rows_24": len(metrics) == 24,
            "global_model_summary_rows_6": len(model_summary) == 6,
            "global_pairwise_primary_rows_60": len(primary) == 60,
            "global_pairwise_sensitivity_rows_180": len(sensitivity) == 180,
            "dominance_by_horizon_rows_24": len(dominance_h) == 24,
            "dominance_overall_rows_6": len(dominance_all) == 6,
            "residual_summary_rows_24": len(residual_summary) == 24,
            "residual_distribution_rows_24": len(residual_distribution) == 24,
            "traffic_thresholds_rows_4": len(traffic_thresholds) == 4,
            "traffic_level_rows_96": len(traffic_level) == 96,
            "hourly_rows_576": len(hourly) == 576,
            "residual_acf_rows_1152": len(residual_acf) == 1152,
            "worst_intervals_rows_120": len(worst_intervals) == 120,
            "stability_inventory_rows_6": len(stability) == 6,
            "cost_inventory_rows_6": len(cost) == 6,
            "aligned_predictions_rows_14340": len(aligned) == 14340,
            "all_60_mean_differentials_reconciled": bool(
                primary["mean_improvement_reconciled"].all()
            ),
            "figure_files_16": len(figure_paths) == 16,
            "source_stat_predictions_unchanged": source_unchanged["stat_predictions"],
            "source_tree_predictions_unchanged": source_unchanged["tree_predictions"],
            "source_recurrent_predictions_unchanged": source_unchanged["recurrent_predictions"],
            "source_transformer_predictions_unchanged": source_unchanged["transformer_predictions"],
            "aligned_predictions_unchanged": source_unchanged["aligned_predictions"],
            "new_training_false": True,
            "new_bootstrap_false": True,
            "new_dm_hac_false": True,
        },
        "sensitivity_class_counts": sensitivity_class_counts,
        "outputs": artifacts_meta,
        "figures": figures_meta,
        "all_runtime_checks": gate.checks,
    }

    gate.check(
        "Manifest validation values all TRUE",
        all(manifest["validation"].values()),
        manifest["validation"],
    )
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    gate.check("Manifest written", paths["manifest"].is_file(), paths["manifest"])

    print()
    print(report_text)
    print("ARTEFACTOS")
    print("-" * 122)
    for key, path in paths.items():
        if path.exists():
            print(f"{key:<32} {path}")
            print(f"{'SHA-256':<32} {sha256(path)}")
    print(f"{'figure files':<32} {len(figure_paths)}")
    print()
    print("FINAL GOVERNANCE")
    print("-" * 122)
    print("training/refit/tuning/selection = 0")
    print("new bootstrap = 0")
    print("new DM-HAC = 0")
    print("source predictions modified = NO")
    print("GLOBAL* created = NO")
    print("PHASE-F-CLOSURE-001 = NOT RUN")
    print("PHASE G = NOT AUTHORIZED")
    print()
    print("GLOBAL-CROSS-FAMILY-ANALYSIS-001: PASS")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.preflight_only:
            return run_preflight()
        return run_full(dpi=args.dpi)
    except Exception as exc:
        print()
        print("=" * 122, file=sys.stderr)
        print("GLOBAL-CROSS-FAMILY ANALYSIS: FAIL / STOP", file=sys.stderr)
        print("=" * 122, file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("No se autoriza PHASE-F-CLOSURE-001.", file=sys.stderr)
        print("PHASE G permanece NOT AUTHORIZED.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
