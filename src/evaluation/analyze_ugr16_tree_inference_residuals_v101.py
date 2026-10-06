#!/usr/bin/env python3
"""
TREE-INFERENCE-RESIDUALS-001
============================

Inferencia pareada y análisis residual de los representantes congelados de
Phase C sobre las predicciones June ya cerradas.

NO entrena modelos.
NO hace refit.
NO hace tuning.
NO modifica predicciones fuente.
NO puede cambiar RF*, XGB*, LGBM* ni TREE*.

Fuentes:
- predicciones June congeladas de RF01 / XGB06 / LGB01;
- predicciones June congeladas de ARIMA(6,1,12) / VAR(5);
- Persistence reconstruida desde las predicciones Phase C.

Inferencia:
- pérdida absoluta;
- diferencial = pérdida_referencia - pérdida_modelo;
- valor positivo = mejora del modelo;
- circular moving-block bootstrap;
- 5000 réplicas;
- bloque principal = 12;
- sensibilidad = 6 / 12 / 24;
- IC 95 %;
- seed histórica = 20260717;
- DM-HAC/Newey-West con kernel Bartlett;
- lag = max(12, h-1) = 12 para H1/H3/H6/H12;
- sin corrección silenciosa por multiplicidad.

Residuos:
- e_t = observado - predicho;
- positivo = infrapredicción;
- negativo = sobrepredicción;
- resumen global;
- distribución;
- ACF;
- nivel de tráfico;
- hora del día;
- P95 absoluto;
- peores intervalos;
- diagnóstico descriptivo de regresión hacia la media.

Con --preflight-only:
- valida hashes y estructuras;
- carga las predicciones fuente;
- comprueba coberturas y alineación exacta;
- NO ejecuta bootstrap/DM;
- NO genera artefactos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNNER_VERSION = "1.0.1"
CAMPAIGN_ID = "TREE-INFERENCE-RESIDUALS-001"
PARENT_CAMPAIGN_ID = "UGR16-TREE-ENSEMBLES-001"
SOURCE_TREE_RUN = "TREE-JUNE-BLIND-001"
SOURCE_STAT_RUN = "UGR16-STATISTICAL-JUNE-BLIND-001"

# ---------------------------------------------------------------------
# Authorities / frozen inputs
# ---------------------------------------------------------------------

P017 = Path(
    "docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md"
)
P021 = Path(
    "docs/project_governance/021_tree_june_blind_closure_2026-08-19.md"
)
P008 = Path(
    "docs/project_governance/"
    "008_statistical_final_representatives_freeze_2026-08-16.md"
)
ENVIRONMENT = Path(
    "results/metrics/tree_ensembles/"
    "tree_environment_preflight_20260819.txt"
)
JUNE = Path(
    "data/processed/ugr16/june_week3_prepared_5min.parquet"
)

TREE_MANIFEST = Path(
    "results/metrics/tree_ensembles/"
    "ugr16_tree_june_blind_manifest.json"
)
TREE_PREDICTIONS = Path(
    "results/predictions/tree_ensembles/"
    "ugr16_tree_june_blind_predictions.parquet"
)
TREE_RUNTIME = Path(
    "results/metrics/tree_ensembles/"
    "ugr16_tree_june_blind_runtime.csv"
)

STAT_MANIFEST = Path(
    "results/metrics/ugr16_statistical_june_blind_manifest.json"
)
STAT_PREDICTIONS = Path(
    "results/predictions/"
    "ugr16_statistical_june_blind_predictions.parquet"
)

EXPECTED_SHA = {
    "017":
        "8c0a94e4ee80a84b78bf077d6abc3c18617ad1736b6d978047d0db1d52055003",
    "021":
        "445ec43095bbf503d0a0ea45a2a209916fd705a0f197bb9b1961db2aeae74426",
    "008":
        "cc98c4f7b74e2d0ce6515ac1a989bef2b20de0a921f63a5d0346591f2456ad1a",
    "environment":
        "82ab24602f66975929904bab802da8893bd059e51c7214bccf412a95275216b2",
    "june":
        "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3",
    "tree_manifest":
        "f0a296c21b3077b2f9c8788fbb118572a203021ae69074bb1a4a2b027f3d2b46",
    "tree_predictions":
        "7f787d74de5b593811e7988767d891d4aa70e0f8826bb1a302f20e4d0e7e61ad",
    "stat_manifest":
        "847f877144f2932e0b5967b1de0678898cb640434dbfb20f2f9805d7c29883a7",
    "stat_predictions":
        "21ebddf20569533a9d1ed66547cc4393f9c3b7e6721515c68eb3eb9dfc95d353",
}

TREE_REPRESENTATIVES = {
    "RF01": "random_forest",
    "XGB06": "xgboost",
    "LGB01": "lightgbm",
}
TREE_STAR = "LGB01"
ARIMA_STAR = "ARIMA(6,1,12)"
VAR_STAR = "VAR(5)"

HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
TREE_EXPECTED_ROWS = 7170
STAT_EXPECTED_ROWS = 11950
STAT_SELECTED_EXPECTED_ROWS = 4780

BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_PRIMARY_BLOCK = 12
BOOTSTRAP_SENSITIVITY_BLOCKS = [6, 12, 24]
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260717
HAC_MIN_LAG = 12
TOP_WORST = 5
ACF_MAX_LAG = 48
ACF_SUMMARY_LAGS = [1, 3, 6, 12, 24]

PRIMARY_COMPARISONS = [
    {
        "comparison": "rf_vs_persistence",
        "model": "RF01",
        "reference": "persistence",
    },
    {
        "comparison": "xgb_vs_persistence",
        "model": "XGB06",
        "reference": "persistence",
    },
    {
        "comparison": "lgbm_vs_persistence",
        "model": "LGB01",
        "reference": "persistence",
    },
    {
        "comparison": "tree_vs_arima",
        "model": "LGB01",
        "reference": ARIMA_STAR,
    },
    {
        "comparison": "tree_vs_var",
        "model": "LGB01",
        "reference": VAR_STAR,
    },
]

METRICS_DIR = Path("results/metrics/tree_ensembles")
PREDICTIONS_DIR = Path("results/predictions/tree_ensembles")
FIGURES_DIR = Path(
    "results/figures/tree_ensembles/"
    "ugr16_tree_inference_residuals"
)
PREFIX = "ugr16_tree_inference_residuals"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Valida inputs, esquemas, coberturas y alineación sin ejecutar "
            "inferencia ni generar salidas."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas previas de esta subcampaña.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 inesperado para {label}\n"
            f"esperado={expected}\n"
            f"actual=  {actual}"
        )
    return actual


def verify_sidecar_if_present(path: Path, label: str) -> bool:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return False
    token = sidecar.read_text(encoding="utf-8").split()[0]
    if token != sha256(path):
        raise RuntimeError(f"Sidecar SHA inválido para {label}: {sidecar}")
    return True


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def normal_two_sided_p(z_value: float) -> float:
    if not np.isfinite(z_value):
        return float("nan")
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


# ---------------------------------------------------------------------
# Load + normalize frozen predictions
# ---------------------------------------------------------------------

def detect_stat_model_column(frame: pd.DataFrame) -> str:
    desired = {ARIMA_STAR, VAR_STAR}
    candidates = ["config", "model", "candidate_id", "model_name"]

    for column in candidates:
        if column in frame.columns:
            values = set(frame[column].dropna().astype(str).unique())
            if desired.issubset(values):
                return column

    raise RuntimeError(
        "No se pudo detectar la columna de configuración del bloque "
        f"estadístico. Columnas disponibles: {list(frame.columns)}"
    )


def normalize_stat_prediction_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize frozen statistical June column names without changing values.

    The statistical June pipeline stores truth/prediction as ``y_true`` and
    ``y_pred``, while the tree June pipeline uses ``y_true_bps`` and
    ``y_pred_bps``.  This adapter creates the canonical ``*_bps`` aliases
    required by this cross-family analyzer.  If both naming variants are
    present, they must reconcile exactly (within the existing 1e-6 tolerance).
    """
    normalized = frame.copy()

    aliases = {
        "y_true_bps": "y_true",
        "y_pred_bps": "y_pred",
    }

    for canonical, legacy in aliases.items():
        canonical_present = canonical in normalized.columns
        legacy_present = legacy in normalized.columns

        if canonical_present and legacy_present:
            canonical_values = pd.to_numeric(
                normalized[canonical], errors="raise"
            ).to_numpy(float)
            legacy_values = pd.to_numeric(
                normalized[legacy], errors="raise"
            ).to_numpy(float)

            if not np.allclose(
                canonical_values,
                legacy_values,
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Columnas stat incompatibles: {canonical} vs {legacy}."
                )

        elif not canonical_present and legacy_present:
            normalized[canonical] = pd.to_numeric(
                normalized[legacy], errors="raise"
            )

    return normalized


def load_and_validate_inputs() -> tuple[
    dict[str, str],
    pd.DataFrame,
    pd.DataFrame,
    str,
    pd.DataFrame,
]:
    hashes = {}

    for key, path in [
        ("017", P017),
        ("021", P021),
        ("008", P008),
        ("environment", ENVIRONMENT),
        ("june", JUNE),
        ("tree_manifest", TREE_MANIFEST),
        ("tree_predictions", TREE_PREDICTIONS),
        ("stat_manifest", STAT_MANIFEST),
        ("stat_predictions", STAT_PREDICTIONS),
    ]:
        hashes[key] = verify_sha(
            path,
            EXPECTED_SHA[key],
            key,
        )

    # Los sidecars de gobernanza, cuando existen, deben reconciliar.
    for label, path in [
        ("017", P017),
        ("021", P021),
        ("008", P008),
        ("environment", ENVIRONMENT),
    ]:
        verify_sidecar_if_present(path, label)

    tree = pd.read_parquet(TREE_PREDICTIONS).copy()
    stat = pd.read_parquet(STAT_PREDICTIONS).copy()
    stat = normalize_stat_prediction_schema(stat)

    if len(tree) != TREE_EXPECTED_ROWS:
        raise RuntimeError(
            f"Tree predictions rows={len(tree)}; "
            f"esperadas={TREE_EXPECTED_ROWS}"
        )

    if len(stat) != STAT_EXPECTED_ROWS:
        raise RuntimeError(
            f"Stat predictions rows={len(stat)}; "
            f"esperadas={STAT_EXPECTED_ROWS}"
        )

    tree_required = {
        "candidate_id",
        "horizon_steps",
        "horizon_minutes",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
        "persistence_pred_bps",
    }
    missing_tree = tree_required.difference(tree.columns)
    if missing_tree:
        raise RuntimeError(
            f"Faltan columnas tree: {sorted(missing_tree)}"
        )

    stat_required = {
        "horizon_steps",
        "target_timestamp",
        "y_true_bps",
        "y_pred_bps",
    }
    missing_stat = stat_required.difference(stat.columns)
    if missing_stat:
        raise RuntimeError(
            f"Faltan columnas stat: {sorted(missing_stat)}"
        )

    tree["target_timestamp"] = pd.to_datetime(
        tree["target_timestamp"],
        errors="raise",
    )
    stat["target_timestamp"] = pd.to_datetime(
        stat["target_timestamp"],
        errors="raise",
    )

    if "origin_timestamp" in tree.columns:
        tree["origin_timestamp"] = pd.to_datetime(
            tree["origin_timestamp"],
            errors="raise",
        )
    if "origin_timestamp" in stat.columns:
        stat["origin_timestamp"] = pd.to_datetime(
            stat["origin_timestamp"],
            errors="raise",
        )

    tree["candidate_id"] = tree["candidate_id"].astype(str)
    stat_model_column = detect_stat_model_column(stat)
    stat[stat_model_column] = stat[stat_model_column].astype(str)

    tree = tree[
        tree["candidate_id"].isin(TREE_REPRESENTATIVES)
    ].copy()

    stat_selected = stat[
        stat[stat_model_column].isin(
            [ARIMA_STAR, VAR_STAR]
        )
    ].copy()

    if len(tree) != TREE_EXPECTED_ROWS:
        raise RuntimeError(
            "El filtrado de representantes tree perdió filas."
        )

    if len(stat_selected) != STAT_SELECTED_EXPECTED_ROWS:
        raise RuntimeError(
            f"Filas ARIMA*/VAR*={len(stat_selected)}; "
            f"esperadas={STAT_SELECTED_EXPECTED_ROWS}"
        )

    finite_tree = np.isfinite(
        tree[
            [
                "y_true_bps",
                "y_pred_bps",
                "persistence_pred_bps",
            ]
        ].to_numpy(float)
    ).all()
    if not finite_tree:
        raise RuntimeError("Tree contiene valores no finitos.")

    finite_stat = np.isfinite(
        stat_selected[
            ["y_true_bps", "y_pred_bps"]
        ].to_numpy(float)
    ).all()
    if not finite_stat:
        raise RuntimeError("Stat contiene valores no finitos.")

    # Cobertura exacta.
    for candidate in TREE_REPRESENTATIVES:
        for horizon, expected in EXPECTED_COUNTS.items():
            observed = len(
                tree[
                    tree["candidate_id"].eq(candidate)
                    & tree["horizon_steps"].eq(horizon)
                ]
            )
            if observed != expected:
                raise RuntimeError(
                    f"{candidate} H{horizon}: "
                    f"{observed} != {expected}"
                )

    for config in [ARIMA_STAR, VAR_STAR]:
        for horizon, expected in EXPECTED_COUNTS.items():
            observed = len(
                stat_selected[
                    stat_selected[stat_model_column].eq(config)
                    & stat_selected["horizon_steps"].eq(horizon)
                ]
            )
            if observed != expected:
                raise RuntimeError(
                    f"{config} H{horizon}: "
                    f"{observed} != {expected}"
                )

    # Canonical target + persistence table from LGB01.
    canonical = tree[
        tree["candidate_id"].eq(TREE_STAR)
    ][
        [
            "horizon_steps",
            "horizon_minutes",
            "target_timestamp",
            "y_true_bps",
            "persistence_pred_bps",
        ]
    ].copy()

    canonical = canonical.sort_values(
        ["horizon_steps", "target_timestamp"]
    ).reset_index(drop=True)

    if canonical.duplicated(
        ["horizon_steps", "target_timestamp"]
    ).any():
        raise RuntimeError(
            "Canonical tree contiene targets duplicados."
        )

    # Todas las familias tree deben compartir targets/observado/persistence.
    key = ["horizon_steps", "target_timestamp"]
    for candidate in TREE_REPRESENTATIVES:
        subset = tree[
            tree["candidate_id"].eq(candidate)
        ][
            key + ["y_true_bps", "persistence_pred_bps"]
        ].copy()

        merged = canonical.merge(
            subset,
            on=key,
            suffixes=("_canonical", "_candidate"),
            validate="one_to_one",
        )

        if len(merged) != len(canonical):
            raise RuntimeError(
                f"Alineación incompleta tree para {candidate}."
            )

        if not np.allclose(
            merged["y_true_bps_canonical"],
            merged["y_true_bps_candidate"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"y_true no coincide para {candidate}."
            )

        if not np.allclose(
            merged["persistence_pred_bps_canonical"],
            merged["persistence_pred_bps_candidate"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Persistence no coincide para {candidate}."
            )

    # ARIMA*/VAR* deben compartir exactamente targets y observado.
    for config in [ARIMA_STAR, VAR_STAR]:
        subset = stat_selected[
            stat_selected[stat_model_column].eq(config)
        ][
            key + ["y_true_bps"]
        ].copy()

        merged = canonical.merge(
            subset,
            on=key,
            suffixes=("_tree", "_stat"),
            validate="one_to_one",
        )

        if len(merged) != len(canonical):
            raise RuntimeError(
                f"Alineación incompleta {config} vs tree."
            )

        if not np.allclose(
            merged["y_true_bps_tree"],
            merged["y_true_bps_stat"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"y_true no coincide tree vs {config}."
            )

    return hashes, tree, stat_selected, stat_model_column, canonical


# ---------------------------------------------------------------------
# Unified prediction table
# ---------------------------------------------------------------------

def build_aligned_predictions(
    tree: pd.DataFrame,
    stat: pd.DataFrame,
    stat_model_column: str,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Base target/persistence.
    for record in canonical.itertuples(index=False):
        rows.append(
            {
                "source_family": "baseline",
                "model_key": "persistence",
                "horizon_steps": int(record.horizon_steps),
                "horizon_minutes": int(record.horizon_minutes),
                "target_timestamp": record.target_timestamp,
                "y_true_bps": float(record.y_true_bps),
                "y_pred_bps": float(record.persistence_pred_bps),
            }
        )

    # Tree representatives.
    for candidate, family in TREE_REPRESENTATIVES.items():
        subset = tree[
            tree["candidate_id"].eq(candidate)
        ].copy()

        for record in subset.itertuples(index=False):
            rows.append(
                {
                    "source_family": family,
                    "model_key": candidate,
                    "horizon_steps": int(record.horizon_steps),
                    "horizon_minutes": int(record.horizon_minutes),
                    "target_timestamp": record.target_timestamp,
                    "y_true_bps": float(record.y_true_bps),
                    "y_pred_bps": float(record.y_pred_bps),
                }
            )

    # Statistical comparators.
    for config in [ARIMA_STAR, VAR_STAR]:
        subset = stat[
            stat[stat_model_column].eq(config)
        ].copy()

        family = (
            "ARIMA" if config == ARIMA_STAR else "VAR"
        )

        for _, record in subset.iterrows():
            rows.append(
                {
                    "source_family": family,
                    "model_key": config,
                    "horizon_steps": int(
                        record["horizon_steps"]
                    ),
                    "horizon_minutes": int(
                        record.get(
                            "horizon_minutes",
                            HORIZON_MINUTES[
                                int(record["horizon_steps"])
                            ],
                        )
                    ),
                    "target_timestamp": record[
                        "target_timestamp"
                    ],
                    "y_true_bps": float(record["y_true_bps"]),
                    "y_pred_bps": float(record["y_pred_bps"]),
                }
            )

    aligned = pd.DataFrame(rows)
    aligned["residual_bps"] = (
        aligned["y_true_bps"] - aligned["y_pred_bps"]
    )
    aligned["absolute_error_bps"] = (
        aligned["residual_bps"].abs()
    )

    expected_models = {
        "persistence",
        *TREE_REPRESENTATIVES.keys(),
        ARIMA_STAR,
        VAR_STAR,
    }
    if set(aligned["model_key"]) != expected_models:
        raise RuntimeError("Modelos alineados inesperados.")

    if aligned.duplicated(
        ["model_key", "horizon_steps", "target_timestamp"]
    ).any():
        raise RuntimeError(
            "Predicciones alineadas contienen duplicados."
        )

    return aligned.sort_values(
        ["model_key", "horizon_steps", "target_timestamp"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def circular_moving_block_bootstrap_means(
    values: np.ndarray,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n <= 1:
        raise RuntimeError("Serie demasiado corta para bootstrap.")
    if block_length <= 0:
        raise RuntimeError("block_length debe ser positivo.")

    blocks_needed = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)

    means = np.empty(repetitions, dtype=float)

    # Vectorización por lotes para limitar memoria.
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

    # Tiny negative floating noise can arise; materially negative LRV is invalid.
    if lrv < 0 and abs(lrv) < 1e-12 * max(gamma0, 1.0):
        lrv = 0.0

    if lrv <= 0:
        return {
            "dm_hac_lrv": lrv,
            "dm_hac_standard_error_bps": float("nan"),
            "dm_hac_z": float("nan"),
            "dm_hac_p_value_two_sided": float("nan"),
        }

    standard_error = math.sqrt(lrv / n)
    z_value = mean_d / standard_error

    return {
        "dm_hac_lrv": float(lrv),
        "dm_hac_standard_error_bps": float(standard_error),
        "dm_hac_z": float(z_value),
        "dm_hac_p_value_two_sided": normal_two_sided_p(z_value),
    }


def comparison_arrays(
    aligned: pd.DataFrame,
    model: str,
    reference: str,
    horizon: int,
) -> pd.DataFrame:
    key = ["horizon_steps", "target_timestamp"]

    model_frame = aligned[
        aligned["model_key"].eq(model)
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

    reference_frame = aligned[
        aligned["model_key"].eq(reference)
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

    paired = model_frame.merge(
        reference_frame,
        on=key,
        validate="one_to_one",
    ).sort_values("target_timestamp").reset_index(drop=True)

    expected = EXPECTED_COUNTS[horizon]
    if len(paired) != expected:
        raise RuntimeError(
            f"{model} vs {reference} H{horizon}: "
            f"n={len(paired)} != {expected}"
        )

    if not np.allclose(
        paired["y_true_model"],
        paired["y_true_reference"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Targets no coinciden en {model} vs {reference} H{horizon}."
        )

    paired["loss_difference_bps"] = (
        paired["abs_error_reference"]
        - paired["abs_error_model"]
    )

    return paired


def run_inference(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sensitivity_rows = []
    primary_rows = []
    paired_summary_rows = []

    # Single historical seed and fixed deterministic iteration order.
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for comparison_spec in PRIMARY_COMPARISONS:
        comparison = comparison_spec["comparison"]
        model = comparison_spec["model"]
        reference = comparison_spec["reference"]

        for horizon in HORIZONS:
            paired = comparison_arrays(
                aligned,
                model,
                reference,
                horizon,
            )

            differential = paired[
                "loss_difference_bps"
            ].to_numpy(float)

            mean_improvement = float(np.mean(differential))
            wins = int(np.sum(differential > 0))
            ties = int(np.sum(differential == 0))
            losses = int(np.sum(differential < 0))

            paired_summary_rows.append(
                {
                    "comparison": comparison,
                    "model": model,
                    "reference": reference,
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_pairs": len(paired),
                    "model_wins": wins,
                    "ties": ties,
                    "model_losses": losses,
                    "model_win_rate_pct": 100.0 * wins / len(paired),
                    "mean_improvement_bps": mean_improvement,
                    "median_improvement_bps": float(
                        np.median(differential)
                    ),
                }
            )

            primary_record = None

            for block in BOOTSTRAP_SENSITIVITY_BLOCKS:
                bootstrap_means = (
                    circular_moving_block_bootstrap_means(
                        differential,
                        block_length=block,
                        repetitions=BOOTSTRAP_REPETITIONS,
                        rng=rng,
                    )
                )

                alpha = 1.0 - BOOTSTRAP_CONFIDENCE
                lower = float(
                    np.quantile(
                        bootstrap_means,
                        alpha / 2.0,
                    )
                )
                upper = float(
                    np.quantile(
                        bootstrap_means,
                        1.0 - alpha / 2.0,
                    )
                )
                probability = float(
                    np.mean(bootstrap_means > 0)
                )

                if lower > 0:
                    bootstrap_direction = "model_better"
                elif upper < 0:
                    bootstrap_direction = "reference_better"
                else:
                    bootstrap_direction = "not_distinguishable"

                sensitivity_row = {
                    "comparison": comparison,
                    "model": model,
                    "reference": reference,
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
                    "bootstrap_direction": bootstrap_direction,
                    "bootstrap_seed_base": BOOTSTRAP_SEED,
                }
                sensitivity_rows.append(sensitivity_row)

                if block == BOOTSTRAP_PRIMARY_BLOCK:
                    primary_record = dict(sensitivity_row)

            if primary_record is None:
                raise RuntimeError("No se generó bloque bootstrap primario.")

            hac_lag = max(HAC_MIN_LAG, horizon - 1)
            dm = newey_west_dm(
                differential,
                lag=hac_lag,
            )

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

            p_value = primary_record[
                "dm_hac_p_value_two_sided"
            ]
            primary_record["dm_hac_distinguishable_0_05"] = (
                bool(np.isfinite(p_value) and p_value < 0.05)
            )

            primary_rows.append(primary_record)

    primary = pd.DataFrame(primary_rows).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)

    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        [
            "comparison",
            "horizon_steps",
            "bootstrap_block_length",
        ]
    ).reset_index(drop=True)

    paired_summary = pd.DataFrame(
        paired_summary_rows
    ).sort_values(
        ["comparison", "horizon_steps"]
    ).reset_index(drop=True)

    if len(primary) != 20:
        raise RuntimeError(f"Primary inference rows={len(primary)} != 20")
    if len(sensitivity) != 60:
        raise RuntimeError(
            f"Bootstrap sensitivity rows={len(sensitivity)} != 60"
        )
    if len(paired_summary) != 20:
        raise RuntimeError(
            f"Paired summary rows={len(paired_summary)} != 20"
        )

    return primary, sensitivity, paired_summary


# ---------------------------------------------------------------------
# Residual diagnostics for tree representatives
# ---------------------------------------------------------------------

def residual_acf(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)

    if lag <= 0 or lag >= len(values):
        return float("nan")

    left = values[:-lag]
    right = values[lag:]

    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")

    return float(np.corrcoef(left, right)[0, 1])


def annotate_tree_predictions(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = aligned[
        aligned["model_key"].isin(TREE_REPRESENTATIVES)
    ].copy()

    thresholds_rows = []
    annotated_parts = []

    for horizon in HORIZONS:
        horizon_frame = tree[
            tree["horizon_steps"].eq(horizon)
        ].copy()

        unique_targets = horizon_frame[
            ["target_timestamp", "y_true_bps"]
        ].drop_duplicates("target_timestamp")

        if len(unique_targets) != EXPECTED_COUNTS[horizon]:
            raise RuntimeError(
                f"Unique targets H{horizon} inesperados."
            )

        q25, q50, q75 = np.quantile(
            unique_targets["y_true_bps"].to_numpy(float),
            [0.25, 0.50, 0.75],
        )

        thresholds_rows.append(
            {
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "q25_bps": q25,
                "q50_bps": q50,
                "q75_bps": q75,
            }
        )

        def level(value: float) -> str:
            if value <= q25:
                return "Q1_low"
            if value <= q50:
                return "Q2_mid_low"
            if value <= q75:
                return "Q3_mid_high"
            return "Q4_high"

        horizon_frame["traffic_level"] = (
            horizon_frame["y_true_bps"].map(level)
        )
        horizon_frame["target_hour"] = (
            horizon_frame["target_timestamp"].dt.hour
        )
        annotated_parts.append(horizon_frame)

    annotated = pd.concat(
        annotated_parts,
        ignore_index=True,
    )

    thresholds = pd.DataFrame(thresholds_rows)
    return annotated, thresholds


def residual_diagnostics(
    annotated: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_rows = []
    distribution_rows = []
    traffic_rows = []
    hourly_rows = []
    acf_rows = []
    regression_rows = []
    worst_rows = []

    for model in TREE_REPRESENTATIVES:
        for horizon in HORIZONS:
            group = annotated[
                annotated["model_key"].eq(model)
                & annotated["horizon_steps"].eq(horizon)
            ].sort_values(
                "target_timestamp"
            ).reset_index(drop=True)

            residual = group["residual_bps"].to_numpy(float)
            abs_error = np.abs(residual)

            summary_rows.append(
                {
                    "model": model,
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "n_predictions": len(group),
                    "bias_bps": float(np.mean(residual)),
                    "mae_bps": float(np.mean(abs_error)),
                    "rmse_bps": float(
                        np.sqrt(np.mean(residual ** 2))
                    ),
                    "underprediction_pct": float(
                        100.0 * np.mean(residual > 0)
                    ),
                    "overprediction_pct": float(
                        100.0 * np.mean(residual < 0)
                    ),
                    "zero_residual_pct": float(
                        100.0 * np.mean(residual == 0)
                    ),
                    "p95_absolute_error_bps": float(
                        np.quantile(abs_error, 0.95)
                    ),
                }
            )

            quantiles = np.quantile(
                residual,
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            )
            distribution_rows.append(
                {
                    "model": model,
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "mean_bps": float(np.mean(residual)),
                    "std_bps": float(np.std(residual, ddof=0)),
                    "q01_bps": float(quantiles[0]),
                    "q05_bps": float(quantiles[1]),
                    "q25_bps": float(quantiles[2]),
                    "q50_bps": float(quantiles[3]),
                    "q75_bps": float(quantiles[4]),
                    "q95_bps": float(quantiles[5]),
                    "q99_bps": float(quantiles[6]),
                    "skew": float(
                        pd.Series(residual).skew()
                    ),
                }
            )

            for traffic_level, level_group in group.groupby(
                "traffic_level",
                sort=False,
            ):
                level_residual = level_group[
                    "residual_bps"
                ].to_numpy(float)
                level_abs = np.abs(level_residual)

                traffic_rows.append(
                    {
                        "model": model,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "traffic_level": traffic_level,
                        "n": len(level_group),
                        "mean_observed_bps": float(
                            level_group["y_true_bps"].mean()
                        ),
                        "mean_predicted_bps": float(
                            level_group["y_pred_bps"].mean()
                        ),
                        "bias_bps": float(
                            np.mean(level_residual)
                        ),
                        "mae_bps": float(
                            np.mean(level_abs)
                        ),
                        "underprediction_pct": float(
                            100.0 * np.mean(level_residual > 0)
                        ),
                        "p95_absolute_error_bps": float(
                            np.quantile(level_abs, 0.95)
                        ),
                    }
                )

            for hour, hour_group in group.groupby(
                "target_hour",
                sort=True,
            ):
                hour_residual = hour_group[
                    "residual_bps"
                ].to_numpy(float)

                hourly_rows.append(
                    {
                        "model": model,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "target_hour": int(hour),
                        "n": len(hour_group),
                        "mean_observed_bps": float(
                            hour_group["y_true_bps"].mean()
                        ),
                        "mean_predicted_bps": float(
                            hour_group["y_pred_bps"].mean()
                        ),
                        "bias_bps": float(
                            np.mean(hour_residual)
                        ),
                        "mae_bps": float(
                            np.mean(np.abs(hour_residual))
                        ),
                        "underprediction_pct": float(
                            100.0 * np.mean(hour_residual > 0)
                        ),
                    }
                )

            for lag in range(1, ACF_MAX_LAG + 1):
                acf_rows.append(
                    {
                        "model": model,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "lag_steps": lag,
                        "lag_minutes": lag * 5,
                        "acf": residual_acf(
                            residual,
                            lag,
                        ),
                        "summary_lag":
                            lag in ACF_SUMMARY_LAGS,
                    }
                )

            # Descriptive regression-to-mean diagnostic.
            q1 = group[
                group["traffic_level"].eq("Q1_low")
            ]
            q4 = group[
                group["traffic_level"].eq("Q4_high")
            ]

            low_bias = float(q1["residual_bps"].mean())
            high_bias = float(q4["residual_bps"].mean())

            observed_span = float(
                q4["y_true_bps"].mean()
                - q1["y_true_bps"].mean()
            )
            predicted_span = float(
                q4["y_pred_bps"].mean()
                - q1["y_pred_bps"].mean()
            )

            regression_rows.append(
                {
                    "model": model,
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "q1_bias_bps": low_bias,
                    "q4_bias_bps": high_bias,
                    "q1_overprediction": low_bias < 0,
                    "q4_underprediction": high_bias > 0,
                    "regression_to_mean_pattern":
                        (low_bias < 0 and high_bias > 0),
                    "observed_q4_minus_q1_bps": observed_span,
                    "predicted_q4_minus_q1_bps": predicted_span,
                    "amplitude_ratio_predicted_vs_observed":
                        (
                            predicted_span / observed_span
                            if observed_span != 0
                            else float("nan")
                        ),
                }
            )

            worst = group.nlargest(
                TOP_WORST,
                "absolute_error_bps",
                keep="first",
            ).copy()

            worst["rank_worst"] = np.arange(
                1,
                len(worst) + 1,
            )

            for _, row in worst.iterrows():
                worst_rows.append(
                    {
                        "model": model,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "rank_worst": int(row["rank_worst"]),
                        "target_timestamp": row[
                            "target_timestamp"
                        ],
                        "traffic_level": row[
                            "traffic_level"
                        ],
                        "y_true_bps": float(row["y_true_bps"]),
                        "y_pred_bps": float(row["y_pred_bps"]),
                        "residual_bps": float(row["residual_bps"]),
                        "absolute_error_bps": float(
                            row["absolute_error_bps"]
                        ),
                    }
                )

    residual_summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "horizon_steps"]
    ).reset_index(drop=True)

    residual_distribution = pd.DataFrame(
        distribution_rows
    ).sort_values(
        ["model", "horizon_steps"]
    ).reset_index(drop=True)

    traffic_level = pd.DataFrame(traffic_rows).sort_values(
        ["model", "horizon_steps", "traffic_level"]
    ).reset_index(drop=True)

    hourly = pd.DataFrame(hourly_rows).sort_values(
        ["model", "horizon_steps", "target_hour"]
    ).reset_index(drop=True)

    residual_acf_full = pd.DataFrame(acf_rows).sort_values(
        ["model", "horizon_steps", "lag_steps"]
    ).reset_index(drop=True)

    regression_to_mean = pd.DataFrame(
        regression_rows
    ).sort_values(
        ["model", "horizon_steps"]
    ).reset_index(drop=True)

    worst_intervals = pd.DataFrame(worst_rows).sort_values(
        ["model", "horizon_steps", "rank_worst"]
    ).reset_index(drop=True)

    if len(residual_summary) != 12:
        raise RuntimeError("Residual summary debe tener 12 filas.")
    if len(residual_distribution) != 12:
        raise RuntimeError(
            "Residual distribution debe tener 12 filas."
        )
    if len(traffic_level) != 48:
        raise RuntimeError("Traffic level debe tener 48 filas.")
    if len(residual_acf_full) != 576:
        raise RuntimeError("Residual ACF debe tener 576 filas.")
    if len(regression_to_mean) != 12:
        raise RuntimeError(
            "Regression-to-mean debe tener 12 filas."
        )
    if len(worst_intervals) != 60:
        raise RuntimeError("Worst intervals debe tener 60 filas.")

    return (
        residual_summary,
        residual_distribution,
        traffic_level,
        hourly,
        residual_acf_full,
        regression_to_mean,
        worst_intervals,
    )


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def save_figure(
    figure: plt.Figure,
    stem: Path,
) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.tight_layout()
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def plot_inference_vs_persistence(
    primary: pd.DataFrame,
) -> list[Path]:
    subset = primary[
        primary["comparison"].isin(
            [
                "rf_vs_persistence",
                "xgb_vs_persistence",
                "lgbm_vs_persistence",
            ]
        )
    ].copy()

    figure, axis = plt.subplots(figsize=(10, 6))

    for model, group in subset.groupby("model", sort=False):
        group = group.sort_values("horizon_minutes")
        y = group["mean_improvement_bps"].to_numpy(float) / 1e6
        lower = group["bootstrap_ci_lower_bps"].to_numpy(float) / 1e6
        upper = group["bootstrap_ci_upper_bps"].to_numpy(float) / 1e6
        yerr = np.vstack([y - lower, upper - y])

        axis.errorbar(
            group["horizon_minutes"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=model,
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Horizonte de predicción (min)")
    axis.set_ylabel(
        "Mejora MAE frente a persistencia (Mbit/s)"
    )
    axis.set_title(
        "Métodos de conjunto basados en árboles frente a persistencia"
    )
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_01_tree_vs_persistence_ci",
    )


def plot_tree_vs_statistical(
    primary: pd.DataFrame,
) -> list[Path]:
    subset = primary[
        primary["comparison"].isin(
            ["tree_vs_arima", "tree_vs_var"]
        )
    ].copy()

    figure, axis = plt.subplots(figsize=(10, 6))

    for reference, group in subset.groupby(
        "reference",
        sort=False,
    ):
        group = group.sort_values("horizon_minutes")
        y = group["mean_improvement_bps"].to_numpy(float) / 1e6
        lower = group["bootstrap_ci_lower_bps"].to_numpy(float) / 1e6
        upper = group["bootstrap_ci_upper_bps"].to_numpy(float) / 1e6
        yerr = np.vstack([y - lower, upper - y])

        axis.errorbar(
            group["horizon_minutes"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=f"LGB01 vs {reference}",
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Horizonte de predicción (min)")
    axis.set_ylabel(
        "Mejora MAE de LGB01 frente a referencia (Mbit/s)"
    )
    axis.set_title(
        "TREE* frente a ARIMA* y VAR*: inferencia bootstrap"
    )
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_02_tree_vs_statistical_ci",
    )


def plot_bootstrap_sensitivity(
    sensitivity: pd.DataFrame,
) -> list[Path]:
    subset = sensitivity[
        sensitivity["comparison"].isin(
            ["tree_vs_arima", "tree_vs_var"]
        )
    ].copy()

    figure, axis = plt.subplots(figsize=(10, 6))

    for (reference, block), group in subset.groupby(
        ["reference", "bootstrap_block_length"],
        sort=True,
    ):
        group = group.sort_values("horizon_minutes")
        axis.plot(
            group["horizon_minutes"],
            group["bootstrap_ci_upper_bps"]
            .sub(group["bootstrap_ci_lower_bps"])
            .to_numpy(float)
            / 1e6,
            marker="o",
            label=f"{reference} | bloque {block}",
        )

    axis.set_xlabel("Horizonte de predicción (min)")
    axis.set_ylabel("Anchura IC bootstrap (Mbit/s)")
    axis.set_title(
        "Sensibilidad del IC bootstrap: TREE* frente a modelos estadísticos"
    )
    axis.legend(ncol=2)

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_03_bootstrap_sensitivity",
    )


def plot_tree_acf(
    acf: pd.DataFrame,
) -> list[Path]:
    subset = acf[
        acf["model"].eq(TREE_STAR)
    ].copy()

    figure, axis = plt.subplots(figsize=(10, 6))

    for horizon, group in subset.groupby(
        "horizon_steps",
        sort=True,
    ):
        group = group.sort_values("lag_steps")
        axis.plot(
            group["lag_steps"],
            group["acf"],
            label=f"H{horizon}",
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Lag residual (intervalos de 5 min)")
    axis.set_ylabel("ACF")
    axis.set_title("ACF residual de TREE* (LGB01)")
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_04_tree_residual_acf",
    )


def plot_tree_traffic_level(
    traffic: pd.DataFrame,
) -> list[Path]:
    subset = traffic[
        traffic["model"].eq(TREE_STAR)
    ].copy()

    level_order = [
        "Q1_low",
        "Q2_mid_low",
        "Q3_mid_high",
        "Q4_high",
    ]
    level_x = {name: index + 1 for index, name in enumerate(level_order)}

    figure, axis = plt.subplots(figsize=(10, 6))

    for horizon, group in subset.groupby(
        "horizon_steps",
        sort=True,
    ):
        group = group.copy()
        group["x"] = group["traffic_level"].map(level_x)
        group = group.sort_values("x")
        axis.plot(
            group["x"],
            group["mae_bps"].to_numpy(float) / 1e6,
            marker="o",
            label=f"H{horizon}",
        )

    axis.set_xticks(
        list(range(1, 5)),
        level_order,
    )
    axis.set_xlabel("Nivel de tráfico observado")
    axis.set_ylabel("MAE (Mbit/s)")
    axis.set_title("Error de TREE* por nivel de tráfico")
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_05_tree_traffic_level_mae",
    )


def plot_tree_hourly(
    hourly: pd.DataFrame,
) -> list[Path]:
    subset = hourly[
        hourly["model"].eq(TREE_STAR)
    ].copy()

    figure, axis = plt.subplots(figsize=(10, 6))

    for horizon, group in subset.groupby(
        "horizon_steps",
        sort=True,
    ):
        group = group.sort_values("target_hour")
        axis.plot(
            group["target_hour"],
            group["bias_bps"].to_numpy(float) / 1e6,
            marker="o",
            label=f"H{horizon}",
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Hora objetivo")
    axis.set_ylabel("Bias observado - predicho (Mbit/s)")
    axis.set_title("Sesgo horario de TREE*")
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_06_tree_hourly_bias",
    )


def plot_tree_observed_segment(
    annotated: pd.DataFrame,
) -> list[Path]:
    subset = annotated[
        annotated["model_key"].eq(TREE_STAR)
        & annotated["horizon_steps"].eq(1)
    ].sort_values("target_timestamp").copy()

    # Fixed, non-cherry-picked first 24 h of H1 blind targets.
    start = subset["target_timestamp"].min()
    end = start + pd.Timedelta(hours=24)
    segment = subset[
        subset["target_timestamp"] < end
    ].copy()

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        segment["target_timestamp"],
        segment["y_true_bps"].to_numpy(float) / 1e6,
        label="observado",
    )
    axis.plot(
        segment["target_timestamp"],
        segment["y_pred_bps"].to_numpy(float) / 1e6,
        label="LGB01",
    )
    axis.set_xlabel("Tiempo objetivo")
    axis.set_ylabel("Tasa equivalente (Mbit/s)")
    axis.set_title(
        "TREE*: observado vs predicho — primeras 24 h del test ciego, H1"
    )
    axis.legend()
    figure.autofmt_xdate()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_07_tree_observed_predicted_h1_first24h",
    )


def plot_tree_residual_distribution(
    annotated: pd.DataFrame,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 6))

    for horizon in [1, 12]:
        residual = annotated[
            annotated["model_key"].eq(TREE_STAR)
            & annotated["horizon_steps"].eq(horizon)
        ]["residual_bps"].to_numpy(float) / 1e6

        axis.hist(
            residual,
            bins=40,
            alpha=0.5,
            label=f"H{horizon}",
        )

    axis.axvline(0.0, linewidth=1.0)
    axis.set_xlabel("Residuo observado - predicho (Mbit/s)")
    axis.set_ylabel("Frecuencia")
    axis.set_title("Distribución residual de TREE*")
    axis.legend()

    return save_figure(
        figure,
        FIGURES_DIR / f"{PREFIX}_08_tree_residual_distribution",
    )


# ---------------------------------------------------------------------
# Report / manifest
# ---------------------------------------------------------------------

def build_report(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    residual_summary: pd.DataFrame,
    regression_to_mean: pd.DataFrame,
    hashes: dict[str, str],
    stat_model_column: str,
) -> str:
    lines = [
        "=" * 116,
        "TREE-INFERENCE-RESIDUALS-001 — INFERENCIA PAREADA Y ANÁLISIS RESIDUAL JUNE",
        "=" * 116,
        f"Campaña marco:                  {PARENT_CAMPAIGN_ID}",
        f"Subcampaña:                     {CAMPAIGN_ID}",
        f"Runner version:                 {RUNNER_VERSION}",
        f"Runner SHA-256:                 {sha256(Path(__file__).resolve())}",
        "",
        "GOBERNANZA",
        "-" * 116,
        "Reentrenamiento:                NO",
        "Refit:                          NO",
        "Tuning:                         NO",
        "Selección:                      NO",
        "Predicciones fuente modificadas:NO",
        "RF* / XGB* / LGBM* / TREE*:     CONGELADOS",
        "",
        "INPUTS",
        "-" * 116,
        f"017 SHA-256:                    {hashes['017']}",
        f"021 SHA-256:                    {hashes['021']}",
        f"008 SHA-256:                    {hashes['008']}",
        f"Tree manifest SHA-256:          {hashes['tree_manifest']}",
        f"Tree predictions SHA-256:       {hashes['tree_predictions']}",
        f"Stat manifest SHA-256:          {hashes['stat_manifest']}",
        f"Stat predictions SHA-256:       {hashes['stat_predictions']}",
        f"Stat model column detectada:    {stat_model_column}",
        "",
        "INFERENCIA",
        "-" * 116,
        "Loss:                           absolute error",
        "Loss differential:              reference - model",
        "Positive value:                 model improves over reference",
        f"Bootstrap:                      circular moving-block, {BOOTSTRAP_REPETITIONS} réplicas",
        f"Primary block:                  {BOOTSTRAP_PRIMARY_BLOCK} intervalos = 60 min",
        "Sensitivity blocks:             6 / 12 / 24",
        f"Confidence:                     {BOOTSTRAP_CONFIDENCE}",
        f"Random seed:                    {BOOTSTRAP_SEED}",
        "DM-HAC kernel:                  Bartlett",
        "DM-HAC lag:                     12 en H1/H3/H6/H12",
        "Multiple-testing correction:    NONE",
        "",
        "CONTRASTES PRIMARIOS",
        "-" * 116,
    ]

    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.comparison:24s} "
            f"H{row.horizon_steps:<2} | "
            f"{row.model} vs {row.reference} | "
            f"improvement={row.mean_improvement_bps/1e6:+.4f} Mbit/s | "
            f"CI=[{row.bootstrap_ci_lower_bps/1e6:+.4f}, "
            f"{row.bootstrap_ci_upper_bps/1e6:+.4f}] | "
            f"P(improve)={row.bootstrap_probability_improvement:.4f} | "
            f"DM-HAC p={row.dm_hac_p_value_two_sided:.6g} | "
            f"bootstrap={row.bootstrap_direction}"
        )

    lines.extend(
        [
            "",
            "RESIDUOS — RF01 / XGB06 / LGB01",
            "-" * 116,
        ]
    )

    for row in residual_summary.itertuples(index=False):
        lines.append(
            f"{row.model:6s} H{row.horizon_steps:<2} | "
            f"bias={row.bias_bps/1e6:+.3f} Mbit/s | "
            f"MAE={row.mae_bps/1e6:.3f} | "
            f"under={row.underprediction_pct:.2f}% | "
            f"P95={row.p95_absolute_error_bps/1e6:.3f} Mbit/s"
        )

    lines.extend(
        [
            "",
            "REGRESIÓN HACIA LA MEDIA — DIAGNÓSTICO DESCRIPTIVO",
            "-" * 116,
        ]
    )

    for row in regression_to_mean.itertuples(index=False):
        lines.append(
            f"{row.model:6s} H{row.horizon_steps:<2} | "
            f"Q1 bias={row.q1_bias_bps/1e6:+.3f} | "
            f"Q4 bias={row.q4_bias_bps/1e6:+.3f} | "
            f"pattern={bool(row.regression_to_mean_pattern)} | "
            f"amplitude_ratio={row.amplitude_ratio_predicted_vs_observed:.4f}"
        )

    lines.extend(
        [
            "",
            "CAUTELAS",
            "-" * 116,
            "- Los p-values son no ajustados; no se añade corrección múltiple post hoc.",
            "- Los diagnósticos residuales no pueden emplearse para retuning.",
            "- El ranking June no redefine TREE*.",
            "- Los cuartiles de tráfico se calculan por horizonte sobre y_true común.",
            "- La regresión hacia la media es un diagnóstico descriptivo, no una atribución causal.",
            "- El segmento observado/predicho se fija como las primeras 24 h de H1; no se selecciona por rendimiento.",
            "",
            "STATUS: PASS",
            "TREE-CLOSURE-001: NOT RUN",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------

def print_preflight(
    hashes: dict[str, str],
    tree: pd.DataFrame,
    stat: pd.DataFrame,
    stat_model_column: str,
    canonical: pd.DataFrame,
) -> None:
    print("=" * 108)
    print("TREE-INFERENCE-RESIDUALS-001 — PREFLIGHT SIN INFERENCIA")
    print("=" * 108)
    print(f"Runner version:              {RUNNER_VERSION}")
    print(f"Runner SHA-256:              {sha256(Path(__file__).resolve())}")
    print(f"017:                         PASS | {hashes['017']}")
    print(f"021:                         PASS | {hashes['021']}")
    print(f"008:                         PASS | {hashes['008']}")
    print(f"Environment:                 PASS | {hashes['environment']}")
    print(f"June:                        PASS | {hashes['june']}")
    print(f"Tree manifest:               PASS | {hashes['tree_manifest']}")
    print(f"Tree predictions:            PASS | {hashes['tree_predictions']}")
    print(f"Stat manifest:               PASS | {hashes['stat_manifest']}")
    print(f"Stat predictions:            PASS | {hashes['stat_predictions']}")
    print()
    print("FUENTES CONGELADAS")
    print("-" * 108)
    print(f"Tree rows:                   {len(tree)}")
    print(f"Stat rows seleccionadas:     {len(stat)}")
    print(f"Stat model column:           {stat_model_column}")
    print("Tree reps:                   RF01 / XGB06 / LGB01")
    print("TREE*:                       LGB01")
    print(f"ARIMA*:                      {ARIMA_STAR}")
    print(f"VAR*:                        {VAR_STAR}")
    print("Persistence:                 reconstruida de predicciones tree congeladas")
    print()
    print("ALINEACIÓN")
    print("-" * 108)
    for horizon in HORIZONS:
        n = len(
            canonical[
                canonical["horizon_steps"].eq(horizon)
            ]
        )
        print(
            f"H{horizon:<2}: targets comunes={n} | "
            f"expected={EXPECTED_COUNTS[horizon]} | alignment=PASS"
        )
    print()
    print("INFERENCIA CONGELADA")
    print("-" * 108)
    print("Loss:                        absolute error")
    print("Differential:                reference - model")
    print("Positive:                    model improves")
    print(f"Bootstrap repetitions:       {BOOTSTRAP_REPETITIONS}")
    print(f"Primary block:               {BOOTSTRAP_PRIMARY_BLOCK}")
    print("Sensitivity blocks:          6 / 12 / 24")
    print(f"Confidence:                  {BOOTSTRAP_CONFIDENCE}")
    print(f"Random seed:                 {BOOTSTRAP_SEED}")
    print("DM-HAC lag:                  12")
    print("Multiple-testing correction: NONE")
    print("Primary comparisons:         5")
    print("Expected primary rows:       20")
    print("Expected sensitivity rows:   60")
    print()
    print("RESIDUOS")
    print("-" * 108)
    print("Models:                      RF01 / XGB06 / LGB01")
    print("Residual:                    observed - predicted")
    print("Traffic levels:              Q1/Q2/Q3/Q4 por horizonte")
    print("ACF:                         lags 1..48")
    print("Worst intervals:             5/model/horizon")
    print("Regression-to-mean:          descriptiva")
    print()
    print("REENTRENAMIENTO:              NO")
    print("REFIT:                        NO")
    print("TUNING:                       NO")
    print("SELECTION CHANGE:             NO")
    print("SOURCE PREDICTIONS MODIFIED:  NO")
    print()
    print("TREE-INFERENCE-RESIDUALS PREFLIGHT: PASS")
    print("BOOTSTRAP EJECUTADO EN PREFLIGHT: 0")
    print("DM-HAC EJECUTADO EN PREFLIGHT: 0")
    print("ARTEFACTOS GENERADOS EN PREFLIGHT: 0")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    (
        hashes,
        tree,
        stat,
        stat_model_column,
        canonical,
    ) = load_and_validate_inputs()

    print_preflight(
        hashes,
        tree,
        stat,
        stat_model_column,
        canonical,
    )

    if args.preflight_only:
        return 0

    paths = {
        "paired_inference":
            METRICS_DIR / f"{PREFIX}_paired_inference.csv",
        "bootstrap_sensitivity":
            METRICS_DIR / f"{PREFIX}_bootstrap_sensitivity.csv",
        "paired_summary":
            METRICS_DIR / f"{PREFIX}_paired_summary.csv",
        "residual_summary":
            METRICS_DIR / f"{PREFIX}_residual_summary.csv",
        "residual_distribution":
            METRICS_DIR / f"{PREFIX}_residual_distribution.csv",
        "traffic_thresholds":
            METRICS_DIR / f"{PREFIX}_traffic_thresholds.csv",
        "traffic_level":
            METRICS_DIR / f"{PREFIX}_traffic_level.csv",
        "hourly":
            METRICS_DIR / f"{PREFIX}_hourly.csv",
        "residual_acf":
            METRICS_DIR / f"{PREFIX}_residual_acf.csv",
        "regression_to_mean":
            METRICS_DIR / f"{PREFIX}_regression_to_mean.csv",
        "worst_intervals":
            METRICS_DIR / f"{PREFIX}_worst_intervals.csv",
        "aligned_predictions":
            METRICS_DIR / f"{PREFIX}_aligned_predictions.parquet",
        "annotated_tree_predictions":
            METRICS_DIR / f"{PREFIX}_annotated_tree_predictions.parquet",
        "report":
            METRICS_DIR / f"{PREFIX}_report.txt",
        "manifest":
            METRICS_DIR / f"{PREFIX}_manifest.json",
    }

    expected_figures = [
        FIGURES_DIR / f"{PREFIX}_01_tree_vs_persistence_ci.png",
        FIGURES_DIR / f"{PREFIX}_01_tree_vs_persistence_ci.pdf",
        FIGURES_DIR / f"{PREFIX}_02_tree_vs_statistical_ci.png",
        FIGURES_DIR / f"{PREFIX}_02_tree_vs_statistical_ci.pdf",
        FIGURES_DIR / f"{PREFIX}_03_bootstrap_sensitivity.png",
        FIGURES_DIR / f"{PREFIX}_03_bootstrap_sensitivity.pdf",
        FIGURES_DIR / f"{PREFIX}_04_tree_residual_acf.png",
        FIGURES_DIR / f"{PREFIX}_04_tree_residual_acf.pdf",
        FIGURES_DIR / f"{PREFIX}_05_tree_traffic_level_mae.png",
        FIGURES_DIR / f"{PREFIX}_05_tree_traffic_level_mae.pdf",
        FIGURES_DIR / f"{PREFIX}_06_tree_hourly_bias.png",
        FIGURES_DIR / f"{PREFIX}_06_tree_hourly_bias.pdf",
        FIGURES_DIR / f"{PREFIX}_07_tree_observed_predicted_h1_first24h.png",
        FIGURES_DIR / f"{PREFIX}_07_tree_observed_predicted_h1_first24h.pdf",
        FIGURES_DIR / f"{PREFIX}_08_tree_residual_distribution.png",
        FIGURES_DIR / f"{PREFIX}_08_tree_residual_distribution.pdf",
    ]

    existing = [
        path for path in [*paths.values(), *expected_figures]
        if path.exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Ya existen salidas de TREE-INFERENCE-RESIDUALS-001. "
            "No se sobrescriben sin --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 108)
    print("INICIO TREE-INFERENCE-RESIDUALS-001")
    print("=" * 108)
    print("No se entrenará ni reajustará ningún modelo.")
    print("Se usan exclusivamente predicciones June congeladas.")
    print()

    aligned = build_aligned_predictions(
        tree,
        stat,
        stat_model_column,
        canonical,
    )

    primary, sensitivity, paired_summary = run_inference(
        aligned
    )

    annotated, thresholds = annotate_tree_predictions(
        aligned
    )

    (
        residual_summary,
        residual_distribution,
        traffic_level,
        hourly,
        residual_acf_full,
        regression_to_mean,
        worst_intervals,
    ) = residual_diagnostics(
        annotated,
        thresholds,
    )

    # Validations before write.
    checks = {
        "aligned_models_6":
            aligned["model_key"].nunique() == 6,
        "primary_rows_20":
            len(primary) == 20,
        "sensitivity_rows_60":
            len(sensitivity) == 60,
        "paired_summary_rows_20":
            len(paired_summary) == 20,
        "residual_summary_rows_12":
            len(residual_summary) == 12,
        "residual_distribution_rows_12":
            len(residual_distribution) == 12,
        "traffic_threshold_rows_4":
            len(thresholds) == 4,
        "traffic_level_rows_48":
            len(traffic_level) == 48,
        "acf_rows_576":
            len(residual_acf_full) == 576,
        "regression_rows_12":
            len(regression_to_mean) == 12,
        "worst_rows_60":
            len(worst_intervals) == 60,
        "n_pairs_positive":
            primary["n_pairs"].gt(0).all(),
        "ci_ordered":
            primary["bootstrap_ci_lower_bps"]
            .le(primary["bootstrap_ci_upper_bps"])
            .all(),
        "bootstrap_probability_valid":
            primary["bootstrap_probability_improvement"]
            .between(0.0, 1.0)
            .all(),
        "sensitivity_probability_valid":
            sensitivity["bootstrap_probability_improvement"]
            .between(0.0, 1.0)
            .all(),
        "sensitivity_ci_ordered":
            sensitivity["bootstrap_ci_lower_bps"]
            .le(sensitivity["bootstrap_ci_upper_bps"])
            .all(),
        "dm_p_values_all_finite":
            np.isfinite(
                primary["dm_hac_p_value_two_sided"]
                .to_numpy(float)
            ).all(),
        "dm_p_values_valid":
            primary["dm_hac_p_value_two_sided"]
            .between(0.0, 1.0)
            .all(),
        "coverage_expected":
            all(
                int(row.n_pairs)
                == EXPECTED_COUNTS[int(row.horizon_steps)]
                for row in primary.itertuples(index=False)
            ),
        "source_tree_hash_unchanged":
            sha256(TREE_PREDICTIONS)
            == EXPECTED_SHA["tree_predictions"],
        "source_stat_hash_unchanged":
            sha256(STAT_PREDICTIONS)
            == EXPECTED_SHA["stat_predictions"],
        "selection_changed_false":
            True,
        "training_false":
            True,
    }

    if not all(checks.values()):
        failed = [
            key for key, value in checks.items()
            if not value
        ]
        raise RuntimeError(
            "Validación interna FAIL: "
            + ", ".join(failed)
        )

    # Write tabular artifacts.
    atomic_csv(primary, paths["paired_inference"])
    atomic_csv(
        sensitivity,
        paths["bootstrap_sensitivity"],
    )
    atomic_csv(paired_summary, paths["paired_summary"])
    atomic_csv(
        residual_summary,
        paths["residual_summary"],
    )
    atomic_csv(
        residual_distribution,
        paths["residual_distribution"],
    )
    atomic_csv(
        thresholds,
        paths["traffic_thresholds"],
    )
    atomic_csv(
        traffic_level,
        paths["traffic_level"],
    )
    atomic_csv(hourly, paths["hourly"])
    atomic_csv(
        residual_acf_full,
        paths["residual_acf"],
    )
    atomic_csv(
        regression_to_mean,
        paths["regression_to_mean"],
    )
    atomic_csv(
        worst_intervals,
        paths["worst_intervals"],
    )
    atomic_parquet(
        aligned,
        paths["aligned_predictions"],
    )
    atomic_parquet(
        annotated,
        paths["annotated_tree_predictions"],
    )

    figure_paths = []
    figure_paths += plot_inference_vs_persistence(primary)
    figure_paths += plot_tree_vs_statistical(primary)
    figure_paths += plot_bootstrap_sensitivity(sensitivity)
    figure_paths += plot_tree_acf(residual_acf_full)
    figure_paths += plot_tree_traffic_level(traffic_level)
    figure_paths += plot_tree_hourly(hourly)
    figure_paths += plot_tree_observed_segment(annotated)
    figure_paths += plot_tree_residual_distribution(annotated)

    report = build_report(
        primary,
        sensitivity,
        residual_summary,
        regression_to_mean,
        hashes,
        stat_model_column,
    )
    atomic_text(report, paths["report"])

    output_records = {}

    for name, path in paths.items():
        if name == "manifest":
            continue
        output_records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }

    for index, path in enumerate(figure_paths, start=1):
        output_records[f"figure_{index:02d}"] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "source_tree_run": SOURCE_TREE_RUN,
        "source_statistical_run": SOURCE_STAT_RUN,
        "status": "PASS",
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "governance": {
            "model_training": False,
            "parameter_refit": False,
            "tuning": False,
            "selection": False,
            "model_substitution": False,
            "source_predictions_modified": False,
            "tree_representatives": TREE_REPRESENTATIVES,
            "tree_star": TREE_STAR,
            "arima_star": ARIMA_STAR,
            "var_star": VAR_STAR,
        },
        "inputs": {
            key: {
                "sha256": value,
            }
            for key, value in hashes.items()
        },
        "prediction_inputs": {
            "tree": {
                "path": str(TREE_PREDICTIONS.resolve()),
                "sha256": EXPECTED_SHA["tree_predictions"],
                "rows": TREE_EXPECTED_ROWS,
            },
            "statistical": {
                "path": str(STAT_PREDICTIONS.resolve()),
                "sha256": EXPECTED_SHA["stat_predictions"],
                "rows": STAT_EXPECTED_ROWS,
                "selected_rows_arima_var":
                    STAT_SELECTED_EXPECTED_ROWS,
                "model_column": stat_model_column,
                "schema_adapter": {
                    "y_true": "y_true_bps",
                    "y_pred": "y_pred_bps",
                    "source_values_modified": False,
                },
            },
        },
        "inference": {
            "loss": "absolute_error",
            "loss_differential":
                "absolute_error_reference - absolute_error_model",
            "positive_value_meaning":
                "model improves over reference",
            "primary_comparisons": PRIMARY_COMPARISONS,
            "bootstrap_method":
                "circular moving-block bootstrap",
            "bootstrap_repetitions":
                BOOTSTRAP_REPETITIONS,
            "bootstrap_block_length":
                BOOTSTRAP_PRIMARY_BLOCK,
            "bootstrap_sensitivity_blocks":
                BOOTSTRAP_SENSITIVITY_BLOCKS,
            "bootstrap_confidence":
                BOOTSTRAP_CONFIDENCE,
            "random_seed":
                BOOTSTRAP_SEED,
            "seed_policy":
                "reuse historical UGR16 inference seed",
            "dm_method":
                "mean loss differential with Newey-West/HAC LRV and normal approximation",
            "newey_west_kernel":
                "Bartlett",
            "minimum_hac_lag":
                HAC_MIN_LAG,
            "hac_lag_rule":
                "max(12, horizon_steps - 1)",
            "multiple_testing_correction":
                "none",
        },
        "residual_analysis": {
            "residual_definition":
                "observed - predicted",
            "positive_residual":
                "underprediction",
            "negative_residual":
                "overprediction",
            "traffic_levels": [
                "Q1_low",
                "Q2_mid_low",
                "Q3_mid_high",
                "Q4_high",
            ],
            "traffic_level_definition":
                "per-horizon quartiles of common observed y_true in June blind",
            "acf_max_lag":
                ACF_MAX_LAG,
            "acf_summary_lags":
                ACF_SUMMARY_LAGS,
            "top_worst_per_model_horizon":
                TOP_WORST,
            "regression_to_mean":
                "descriptive only",
            "observed_predicted_segment":
                "first 24 h of H1 blind targets; fixed non-cherry-picked segment",
        },
        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },
        "processing_seconds":
            time.perf_counter() - started,
        "tree_closure_run":
            False,
        "outputs":
            output_records,
    }

    atomic_json(manifest, paths["manifest"])

    print(report)
    print()
    print("VALIDACIONES")
    print("-" * 108)
    for key, value in checks.items():
        print(
            f"{key:48s}: "
            f"{'PASS' if value else 'FAIL'}"
        )
    print()
    print(f"Manifest: {paths['manifest'].resolve()}")
    print("TREE-INFERENCE-RESIDUALS-001: PASS")
    print("TRAINING/REFIT/TUNING: NO / NO / NO")
    print("SELECTION CHANGED: NO")
    print("TREE-CLOSURE-001: NOT RUN")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
