#!/usr/bin/env python3
"""
Análisis reproducible de residuos para la campaña baseline de UGR'16.

El script estudia, por modelo y horizonte:
- sesgo de predicción;
- frecuencia y magnitud de infrapredicción/sobrepredicción;
- error según nivel de tráfico;
- error según hora del día;
- intervalos con mayor error absoluto;
- comparación emparejada frente a persistencia;
- residuos de los mejores modelos por horizonte.

Convención del residuo
----------------------
    error_bps = y_true_bps - y_pred_bps

Por tanto:
- error positivo  -> infrapredicción;
- error negativo  -> sobrepredicción;
- error igual a 0 -> predicción exacta.

Entradas predeterminadas
------------------------
Predicciones:
    results/predictions/ugr16_march_week3_baselines_predictions.parquet

Métricas baseline:
    results/metrics/ugr16_march_week3_baselines_metrics.csv

Manifiesto baseline:
    results/metrics/ugr16_march_week3_baselines_manifest.json

Salidas predeterminadas
-----------------------
Tablas e informes:
    results/metrics/ugr16_march_week3_residuals_*.csv/json/txt

Figuras:
    results/figures/ugr16_residuals/*.png
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
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "results/predictions/ugr16_march_week3_baselines_predictions.parquet"
)
DEFAULT_BASELINE_METRICS = Path(
    "results/metrics/ugr16_march_week3_baselines_metrics.csv"
)
DEFAULT_BASELINE_MANIFEST = Path(
    "results/metrics/ugr16_march_week3_baselines_manifest.json"
)
DEFAULT_METRICS_DIR = Path("results/metrics")
DEFAULT_FIGURES_DIR = Path("results/figures/ugr16_residuals")
DEFAULT_PREFIX = "ugr16_march_week3_residuals"

REQUIRED_PREDICTION_COLUMNS = {
    "origin_timestamp",
    "target_timestamp",
    "origin_index",
    "target_index",
    "horizon_steps",
    "horizon_minutes",
    "model",
    "y_true_bps",
    "y_pred_bps",
    "error_bps",
    "absolute_error_bps",
}

TRAFFIC_LEVEL_ORDER = [
    "low_q1",
    "medium_low_q2",
    "medium_high_q3",
    "high_q4",
]


# ---------------------------------------------------------------------------
# Argumentos, rutas y utilidades de E/S
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza residuos, sesgo, infrapredicción, error por nivel de "
            "tráfico/hora y comparación emparejada frente a persistencia."
        )
    )

    parser.add_argument(
        "--predictions-parquet",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Predicciones Parquet de la campaña baseline.",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=DEFAULT_BASELINE_METRICS,
        help="CSV de métricas de la campaña baseline.",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help="Manifiesto JSON de la campaña baseline.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directorio de tablas, informe y manifiesto.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directorio de figuras.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Prefijo común de archivos de salida.",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default="UGR16-RESIDUALS-001",
        help="Identificador reproducible de la campaña de residuos.",
    )
    parser.add_argument(
        "--top-worst",
        type=int,
        default=10,
        help="Número de peores intervalos guardados por modelo y horizonte.",
    )
    parser.add_argument(
        "--tie-tolerance-bps",
        type=float,
        default=1e-6,
        help=(
            "Tolerancia absoluta para considerar empate en la comparación "
            "emparejada frente a persistencia."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Resolución de las figuras PNG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas existentes.",
    )

    return parser.parse_args()


def output_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    prefix = arguments.prefix

    paths: dict[str, Path] = {
        "summary_csv": arguments.metrics_dir / f"{prefix}_summary.csv",
        "traffic_level_csv": (
            arguments.metrics_dir / f"{prefix}_traffic_level.csv"
        ),
        "hourly_csv": arguments.metrics_dir / f"{prefix}_hourly.csv",
        "paired_csv": (
            arguments.metrics_dir / f"{prefix}_paired_vs_persistence.csv"
        ),
        "worst_intervals_csv": (
            arguments.metrics_dir / f"{prefix}_worst_intervals.csv"
        ),
        "best_model_residuals_csv": (
            arguments.metrics_dir / f"{prefix}_best_model_residuals.csv"
        ),
        "manifest_json": (
            arguments.metrics_dir / f"{prefix}_manifest.json"
        ),
        "report_txt": arguments.metrics_dir / f"{prefix}_report.txt",
        "figure_bias": (
            arguments.figures_dir / f"{prefix}_01_bias_by_horizon.png"
        ),
        "figure_underprediction": (
            arguments.figures_dir
            / f"{prefix}_02_underprediction_rate_by_horizon.png"
        ),
        "figure_paired_skill": (
            arguments.figures_dir
            / f"{prefix}_03_paired_skill_vs_persistence.png"
        ),
        "figure_traffic_level": (
            arguments.figures_dir
            / f"{prefix}_04_mae_by_traffic_level_best_models.png"
        ),
        "figure_hourly": (
            arguments.figures_dir
            / f"{prefix}_05_hourly_mae_best_models.png"
        ),
    }

    return paths


def add_horizon_figure_paths(
    paths: dict[str, Path],
    figures_dir: Path,
    prefix: str,
    horizons: Iterable[int],
) -> None:
    for position, horizon in enumerate(sorted(horizons), start=6):
        paths[f"figure_h{horizon}"] = (
            figures_dir
            / f"{prefix}_{position:02d}_h{horizon}_best_model_residuals.png"
        )


def validate_arguments(
    arguments: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    required_inputs = [
        arguments.predictions_parquet,
        arguments.baseline_metrics,
        arguments.baseline_manifest,
    ]

    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"No se encuentra la entrada: {path}")

    if arguments.top_worst <= 0:
        raise ValueError("--top-worst debe ser mayor que cero.")

    if arguments.tie_tolerance_bps < 0:
        raise ValueError("--tie-tolerance-bps no puede ser negativo.")

    if arguments.dpi <= 0:
        raise ValueError("--dpi debe ser mayor que cero.")

    if not arguments.prefix.strip():
        raise ValueError("--prefix no puede estar vacío.")

    arguments.campaign_id = arguments.campaign_id.strip()
    if not arguments.campaign_id:
        raise ValueError("--campaign-id no puede estar vacío.")

    if not arguments.overwrite:
        existing = [
            str(path)
            for path in paths.values()
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite o cambia el prefijo:\n"
                + "\n".join(existing)
            )


def ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def atomic_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    os.replace(temporary, path)


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_figure_atomic(
    figure: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, dpi=dpi, format="png")
    plt.close(figure)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------


def smape_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    numerator = 2.0 * np.abs(y_true - y_pred)

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float64"),
        where=denominator != 0,
    )

    return float(100.0 * np.mean(ratio))


def load_predictions(path: Path) -> pd.DataFrame:
    dataframe = pd.read_parquet(path)

    missing = sorted(
        REQUIRED_PREDICTION_COLUMNS - set(dataframe.columns)
    )
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en las predicciones: {missing}"
        )

    dataframe = dataframe.copy()

    for column in ["origin_timestamp", "target_timestamp"]:
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

    numeric_columns = [
        "origin_index",
        "target_index",
        "horizon_steps",
        "horizon_minutes",
        "y_true_bps",
        "y_pred_bps",
        "error_bps",
        "absolute_error_bps",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe[["origin_timestamp", "target_timestamp"]].isna().any().any():
        raise ValueError("Las predicciones contienen timestamps inválidos.")

    if dataframe[numeric_columns].isna().any().any():
        raise ValueError("Las predicciones contienen valores numéricos inválidos.")

    if not np.isfinite(dataframe[numeric_columns].to_numpy()).all():
        raise ValueError("Las predicciones contienen valores no finitos.")

    if (dataframe["y_true_bps"] < 0).any():
        raise ValueError("La variable observada contiene valores negativos.")

    if (dataframe["y_pred_bps"] < 0).any():
        raise ValueError("Las predicciones contienen valores negativos.")

    if not (
        dataframe["target_timestamp"] > dataframe["origin_timestamp"]
    ).all():
        raise ValueError(
            "Hay objetivos que no son posteriores al origen de predicción."
        )

    if not (
        dataframe["target_index"] - dataframe["origin_index"]
        == dataframe["horizon_steps"]
    ).all():
        raise ValueError(
            "La diferencia de índices no coincide con el horizonte."
        )

    expected_error = dataframe["y_true_bps"] - dataframe["y_pred_bps"]
    expected_absolute = np.abs(expected_error)

    if not np.allclose(
        dataframe["error_bps"],
        expected_error,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError("La columna error_bps no está reconciliada.")

    if not np.allclose(
        dataframe["absolute_error_bps"],
        expected_absolute,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError(
            "La columna absolute_error_bps no está reconciliada."
        )

    duplicate_key = [
        "horizon_steps",
        "model",
        "target_timestamp",
    ]
    duplicate_count = int(
        dataframe.duplicated(duplicate_key).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Hay {duplicate_count} predicciones duplicadas por clave."
        )

    if "persistence" not in set(dataframe["model"]):
        raise ValueError(
            "No existe el modelo persistence para la comparación emparejada."
        )

    # Para cada horizonte y timestamp, el valor observado debe ser único.
    consistency = (
        dataframe.groupby(
            ["horizon_steps", "target_timestamp"],
            sort=False,
        )["y_true_bps"]
        .nunique()
    )

    if (consistency != 1).any():
        raise ValueError(
            "Hay valores observados inconsistentes entre modelos."
        )

    return dataframe.sort_values(
        ["horizon_steps", "target_timestamp", "model"]
    ).reset_index(drop=True)


def load_baseline_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)

    required = {
        "horizon_steps",
        "horizon_minutes",
        "model",
        "n_predictions",
        "mae_bps",
        "rmse_bps",
        "smape_pct",
    }
    missing = sorted(required - set(metrics.columns))

    if missing:
        raise ValueError(
            f"Faltan columnas en las métricas baseline: {missing}"
        )

    return metrics.copy()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if "evaluation" not in manifest or "models" not in manifest:
        raise ValueError("El manifiesto baseline no tiene el esquema esperado.")

    return manifest


def validate_against_baseline_metrics(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    calculated_rows: list[dict[str, Any]] = []

    for (horizon, model), group in predictions.groupby(
        ["horizon_steps", "model"],
        sort=True,
    ):
        y_true = group["y_true_bps"].to_numpy(dtype="float64")
        y_pred = group["y_pred_bps"].to_numpy(dtype="float64")
        errors = y_true - y_pred

        calculated_rows.append(
            {
                "horizon_steps": int(horizon),
                "model": str(model),
                "n_predictions_calc": int(len(group)),
                "mae_bps_calc": float(np.mean(np.abs(errors))),
                "rmse_bps_calc": float(np.sqrt(np.mean(np.square(errors)))),
                "smape_pct_calc": smape_percent(y_true, y_pred),
            }
        )

    calculated = pd.DataFrame(calculated_rows)
    merged = metrics.merge(
        calculated,
        on=["horizon_steps", "model"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "Las combinaciones modelo/horizonte no coinciden entre "
            "predicciones y métricas."
        )

    checks = {
        "n_predictions": np.array_equal(
            merged["n_predictions"].astype(int),
            merged["n_predictions_calc"].astype(int),
        ),
        "mae_bps": np.allclose(
            merged["mae_bps"],
            merged["mae_bps_calc"],
            rtol=1e-10,
            atol=1e-6,
        ),
        "rmse_bps": np.allclose(
            merged["rmse_bps"],
            merged["rmse_bps_calc"],
            rtol=1e-10,
            atol=1e-6,
        ),
        "smape_pct": np.allclose(
            merged["smape_pct"],
            merged["smape_pct_calc"],
            rtol=1e-10,
            atol=1e-9,
        ),
    }

    failed = [name for name, result in checks.items() if not result]
    if failed:
        raise ValueError(
            "Las métricas no se reproducen desde las predicciones: "
            + ", ".join(failed)
        )


# ---------------------------------------------------------------------------
# Cálculos de residuos
# ---------------------------------------------------------------------------


def mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (horizon, model), group in predictions.groupby(
        ["horizon_steps", "model"],
        sort=True,
    ):
        errors = group["error_bps"].astype("float64")
        absolute = group["absolute_error_bps"].astype("float64")

        under = errors > 0
        over = errors < 0
        exact = errors == 0

        rows.append(
            {
                "horizon_steps": int(horizon),
                "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                "model": str(model),
                "n_predictions": int(len(group)),
                "mean_error_bps": float(errors.mean()),
                "mean_error_mbps": float(errors.mean() / 1_000_000.0),
                "median_error_bps": float(errors.median()),
                "median_error_mbps": float(errors.median() / 1_000_000.0),
                "mae_bps": float(absolute.mean()),
                "mae_mbps": float(absolute.mean() / 1_000_000.0),
                "rmse_bps": float(np.sqrt(np.mean(np.square(errors)))),
                "rmse_mbps": float(
                    np.sqrt(np.mean(np.square(errors))) / 1_000_000.0
                ),
                "underprediction_count": int(under.sum()),
                "underprediction_rate_pct": float(100.0 * under.mean()),
                "overprediction_count": int(over.sum()),
                "overprediction_rate_pct": float(100.0 * over.mean()),
                "exact_count": int(exact.sum()),
                "exact_rate_pct": float(100.0 * exact.mean()),
                "mean_underprediction_bps": mean_or_nan(errors.loc[under]),
                "mean_underprediction_mbps": (
                    mean_or_nan(errors.loc[under]) / 1_000_000.0
                ),
                "mean_overprediction_magnitude_bps": mean_or_nan(
                    -errors.loc[over]
                ),
                "mean_overprediction_magnitude_mbps": (
                    mean_or_nan(-errors.loc[over]) / 1_000_000.0
                ),
                "p50_absolute_error_bps": float(absolute.quantile(0.50)),
                "p90_absolute_error_bps": float(absolute.quantile(0.90)),
                "p95_absolute_error_bps": float(absolute.quantile(0.95)),
                "p99_absolute_error_bps": float(absolute.quantile(0.99)),
                "max_absolute_error_bps": float(absolute.max()),
                "max_absolute_error_mbps": float(
                    absolute.max() / 1_000_000.0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "mae_bps", "model"]
    ).reset_index(drop=True)


def determine_best_models(
    baseline_metrics: pd.DataFrame,
) -> dict[int, str]:
    best_rows = (
        baseline_metrics.sort_values(
            ["horizon_steps", "mae_bps", "model"]
        )
        .groupby("horizon_steps", as_index=False)
        .first()
    )

    return {
        int(row.horizon_steps): str(row.model)
        for row in best_rows.itertuples(index=False)
    }


def add_traffic_levels(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed = (
        predictions[["target_timestamp", "y_true_bps"]]
        .drop_duplicates()
        .sort_values("target_timestamp")
    )

    quantiles = observed["y_true_bps"].quantile([0.25, 0.50, 0.75])
    q25 = float(quantiles.loc[0.25])
    q50 = float(quantiles.loc[0.50])
    q75 = float(quantiles.loc[0.75])

    if not (q25 < q50 < q75):
        raise ValueError(
            "Los cuantiles de tráfico no permiten construir cuatro niveles."
        )

    enriched = predictions.copy()
    values = enriched["y_true_bps"]

    enriched["traffic_level"] = np.select(
        [
            values <= q25,
            values <= q50,
            values <= q75,
            values > q75,
        ],
        TRAFFIC_LEVEL_ORDER,
        default="unknown",
    )

    if (enriched["traffic_level"] == "unknown").any():
        raise RuntimeError("No se pudo asignar nivel de tráfico a todas las filas.")

    enriched["traffic_level"] = pd.Categorical(
        enriched["traffic_level"],
        categories=TRAFFIC_LEVEL_ORDER,
        ordered=True,
    )

    thresholds = {
        "q25_bps": q25,
        "q50_bps": q50,
        "q75_bps": q75,
        "q25_mbps": q25 / 1_000_000.0,
        "q50_mbps": q50 / 1_000_000.0,
        "q75_mbps": q75 / 1_000_000.0,
    }

    return enriched, thresholds


def grouped_error_table(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for keys, group in dataframe.groupby(
        group_columns,
        sort=True,
        observed=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))
        errors = group["error_bps"].astype("float64")
        absolute = group["absolute_error_bps"].astype("float64")

        row.update(
            {
                "n_predictions": int(len(group)),
                "mean_y_true_bps": float(group["y_true_bps"].mean()),
                "mean_y_true_mbps": float(
                    group["y_true_bps"].mean() / 1_000_000.0
                ),
                "mean_error_bps": float(errors.mean()),
                "mean_error_mbps": float(errors.mean() / 1_000_000.0),
                "mae_bps": float(absolute.mean()),
                "mae_mbps": float(absolute.mean() / 1_000_000.0),
                "rmse_bps": float(np.sqrt(np.mean(np.square(errors)))),
                "rmse_mbps": float(
                    np.sqrt(np.mean(np.square(errors))) / 1_000_000.0
                ),
                "underprediction_rate_pct": float(
                    100.0 * (errors > 0).mean()
                ),
                "overprediction_rate_pct": float(
                    100.0 * (errors < 0).mean()
                ),
                "p95_absolute_error_bps": float(
                    absolute.quantile(0.95)
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def exact_two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")

    k = min(wins, losses)
    tail = sum(
        math.comb(n, index) * (0.5 ** n)
        for index in range(k + 1)
    )
    return float(min(1.0, 2.0 * tail))


def paired_vs_persistence(
    predictions: pd.DataFrame,
    tie_tolerance_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for horizon, horizon_data in predictions.groupby(
        "horizon_steps",
        sort=True,
    ):
        persistence = horizon_data.loc[
            horizon_data["model"] == "persistence",
            [
                "target_timestamp",
                "absolute_error_bps",
                "error_bps",
            ],
        ].rename(
            columns={
                "absolute_error_bps": "persistence_absolute_error_bps",
                "error_bps": "persistence_error_bps",
            }
        )

        for model, model_data in horizon_data.groupby(
            "model",
            sort=True,
        ):
            if model == "persistence":
                continue

            paired = model_data[
                [
                    "target_timestamp",
                    "absolute_error_bps",
                    "error_bps",
                ]
            ].merge(
                persistence,
                on="target_timestamp",
                how="inner",
                validate="one_to_one",
            )

            expected_pairs = len(model_data)
            if len(paired) != expected_pairs:
                raise RuntimeError(
                    f"Emparejamiento incompleto para H{horizon}, {model}."
                )

            delta = (
                paired["absolute_error_bps"]
                - paired["persistence_absolute_error_bps"]
            )

            wins = int((delta < -tie_tolerance_bps).sum())
            ties = int((np.abs(delta) <= tie_tolerance_bps).sum())
            losses = int((delta > tie_tolerance_bps).sum())

            model_mae = float(paired["absolute_error_bps"].mean())
            persistence_mae = float(
                paired["persistence_absolute_error_bps"].mean()
            )

            rows.append(
                {
                    "horizon_steps": int(horizon),
                    "horizon_minutes": int(
                        model_data["horizon_minutes"].iloc[0]
                    ),
                    "model": str(model),
                    "n_pairs": int(len(paired)),
                    "model_mae_bps": model_mae,
                    "model_mae_mbps": model_mae / 1_000_000.0,
                    "persistence_mae_bps": persistence_mae,
                    "persistence_mae_mbps": (
                        persistence_mae / 1_000_000.0
                    ),
                    "paired_mae_delta_bps": float(delta.mean()),
                    "paired_mae_delta_mbps": float(
                        delta.mean() / 1_000_000.0
                    ),
                    "paired_median_delta_bps": float(delta.median()),
                    "paired_skill_pct": float(
                        100.0
                        * (persistence_mae - model_mae)
                        / persistence_mae
                    ),
                    "wins_vs_persistence": wins,
                    "ties_vs_persistence": ties,
                    "losses_vs_persistence": losses,
                    "win_rate_pct": float(100.0 * wins / len(paired)),
                    "tie_rate_pct": float(100.0 * ties / len(paired)),
                    "loss_rate_pct": float(100.0 * losses / len(paired)),
                    "sign_test_pvalue_two_sided": (
                        exact_two_sided_sign_test_pvalue(wins, losses)
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["horizon_steps", "paired_mae_delta_bps", "model"]
    ).reset_index(drop=True)


def worst_intervals(
    predictions: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for (horizon, model), group in predictions.groupby(
        ["horizon_steps", "model"],
        sort=True,
    ):
        selected = (
            group.nlargest(top_n, "absolute_error_bps")
            .sort_values("absolute_error_bps", ascending=False)
            .copy()
        )
        selected.insert(
            0,
            "rank_within_model_horizon",
            np.arange(1, len(selected) + 1),
        )
        selected["y_true_mbps"] = (
            selected["y_true_bps"] / 1_000_000.0
        )
        selected["y_pred_mbps"] = (
            selected["y_pred_bps"] / 1_000_000.0
        )
        selected["error_mbps"] = (
            selected["error_bps"] / 1_000_000.0
        )
        selected["absolute_error_mbps"] = (
            selected["absolute_error_bps"] / 1_000_000.0
        )
        selected["error_direction"] = np.where(
            selected["error_bps"] > 0,
            "underprediction",
            np.where(
                selected["error_bps"] < 0,
                "overprediction",
                "exact",
            ),
        )
        frames.append(selected)

    return pd.concat(frames, ignore_index=True)


def best_model_residuals(
    predictions: pd.DataFrame,
    best_models: dict[int, str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for horizon, model in sorted(best_models.items()):
        selected = predictions.loc[
            (predictions["horizon_steps"] == horizon)
            & (predictions["model"] == model)
        ].copy()
        selected["selected_best_model"] = model
        selected["y_true_mbps"] = selected["y_true_bps"] / 1_000_000.0
        selected["y_pred_mbps"] = selected["y_pred_bps"] / 1_000_000.0
        selected["error_mbps"] = selected["error_bps"] / 1_000_000.0
        selected["absolute_error_mbps"] = (
            selected["absolute_error_bps"] / 1_000_000.0
        )
        selected["error_direction"] = np.where(
            selected["error_bps"] > 0,
            "underprediction",
            np.where(
                selected["error_bps"] < 0,
                "overprediction",
                "exact",
            ),
        )
        frames.append(selected)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["horizon_steps", "target_timestamp"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def plot_metric_by_horizon(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
    reference_line: float,
) -> None:
    pivot = summary.pivot(
        index="horizon_minutes",
        columns="model",
        values=metric,
    ).sort_index()

    figure, axis = plt.subplots(figsize=(12, 6))

    for model in pivot.columns:
        axis.plot(
            pivot.index,
            pivot[model],
            marker="o",
            label=model,
        )

    axis.axhline(reference_line, linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Horizonte de predicción (minutos)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    save_figure_atomic(figure, path, dpi)


def plot_paired_skill(
    paired: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    pivot = paired.pivot(
        index="horizon_minutes",
        columns="model",
        values="paired_skill_pct",
    ).sort_index()

    figure, axis = plt.subplots(figsize=(12, 6))

    for model in pivot.columns:
        axis.plot(
            pivot.index,
            pivot[model],
            marker="o",
            label=model,
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Skill MAE emparejado frente a persistencia")
    axis.set_xlabel("Horizonte de predicción (minutos)")
    axis.set_ylabel("Mejora frente a persistencia (%)")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    save_figure_atomic(figure, path, dpi)


def plot_traffic_level_best_models(
    traffic_table: pd.DataFrame,
    best_models: dict[int, str],
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))

    for horizon, model in sorted(best_models.items()):
        subset = traffic_table.loc[
            (traffic_table["horizon_steps"] == horizon)
            & (traffic_table["model"] == model)
        ].copy()
        subset["traffic_level"] = pd.Categorical(
            subset["traffic_level"],
            categories=TRAFFIC_LEVEL_ORDER,
            ordered=True,
        )
        subset = subset.sort_values("traffic_level")

        axis.plot(
            subset["traffic_level"].astype(str),
            subset["mae_mbps"],
            marker="o",
            label=f"H{horizon}: {model}",
        )

    axis.set_title("MAE de los mejores modelos según nivel de tráfico")
    axis.set_xlabel("Nivel de tráfico observado")
    axis.set_ylabel("MAE (Mbit/s)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    save_figure_atomic(figure, path, dpi)


def plot_hourly_best_models(
    hourly_table: pd.DataFrame,
    best_models: dict[int, str],
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))

    for horizon, model in sorted(best_models.items()):
        subset = hourly_table.loc[
            (hourly_table["horizon_steps"] == horizon)
            & (hourly_table["model"] == model)
        ].sort_values("target_hour")

        axis.plot(
            subset["target_hour"],
            subset["mae_mbps"],
            marker="o",
            label=f"H{horizon}: {model}",
        )

    axis.set_title("MAE horario exploratorio de los mejores modelos")
    axis.set_xlabel("Hora del día del objetivo")
    axis.set_ylabel("MAE (Mbit/s)")
    axis.set_xticks(range(24))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    save_figure_atomic(figure, path, dpi)


def plot_best_model_residuals(
    best_residuals: pd.DataFrame,
    horizon: int,
    model: str,
    path: Path,
    dpi: int,
) -> None:
    subset = best_residuals.loc[
        best_residuals["horizon_steps"] == horizon
    ].sort_values("target_timestamp")

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        subset["target_timestamp"],
        subset["error_mbps"],
        linewidth=1.3,
        label="residuo",
    )
    axis.axhline(0.0, linewidth=1.0, label="error cero")
    axis.set_title(
        f"Residuos walk-forward — H{horizon} "
        f"({int(subset['horizon_minutes'].iloc[0])} min), {model}"
    )
    axis.set_xlabel("Tiempo objetivo")
    axis.set_ylabel("Residuo observado − predicho (Mbit/s)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    save_figure_atomic(figure, path, dpi)


# ---------------------------------------------------------------------------
# Informe y manifiesto
# ---------------------------------------------------------------------------


def build_report(
    predictions: pd.DataFrame,
    baseline_manifest: dict[str, Any],
    campaign_id: str,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    traffic_thresholds: dict[str, float],
    worst: pd.DataFrame,
    best_models: dict[int, str],
    paths: dict[str, Path],
    processing_seconds: float,
) -> str:
    source_baseline_campaign_id = str(
        baseline_manifest.get("campaign_id", "not_available")
    )
    capture_name = str(
        baseline_manifest.get("capture_name", "UGR'16")
    )
    evaluation_role = str(
        baseline_manifest.get(
            "evaluation_role",
            baseline_manifest.get(
                "evaluation",
                {},
            ).get("evaluation_role", "not_available"),
        )
    )

    baseline_input = baseline_manifest.get("input", {})
    input_rows = float(baseline_input.get("rows", 0))
    input_delta_seconds = float(
        baseline_input.get("expected_delta_seconds", 0)
    )
    coverage_hours = float(
        input_rows * input_delta_seconds / 3600.0
    )
    coverage_days = coverage_hours / 24.0

    if coverage_hours > 0:
        coverage_note = (
            f"- La captura baseline cubre {coverage_hours:.2f} horas "
            f"({coverage_days:.2f} días). El análisis horario se "
            "circunscribe al periodo objetivo evaluado y no demuestra "
            "por sí solo estabilidad entre semanas."
        )
    else:
        coverage_note = (
            "- No fue posible recuperar automáticamente la cobertura "
            "desde el manifiesto baseline; el análisis horario debe "
            "interpretarse según el periodo objetivo indicado."
        )

    if evaluation_role == "external_validation":
        role_note = (
            "- La campaña baseline tiene rol external_validation: los "
            "hiperparámetros se fijaron en una captura previa, aunque los "
            "coeficientes se reestimaron mediante walk-forward dentro de "
            "la captura actual."
        )
    else:
        role_note = (
            "- La campaña baseline tiene rol de desarrollo; los "
            "resultados no deben interpretarse como validación externa."
        )

    lines: list[str] = [
        "=" * 92,
        "ANÁLISIS DE RESIDUOS DE BASELINES Y RIDGE — UGR'16",
        "=" * 92,
        f"Campaña de residuos:               {campaign_id}",
        f"Campaña baseline fuente:           {source_baseline_campaign_id}",
        f"Captura:                           {capture_name}",
        f"Rol metodológico:                  {evaluation_role}",
        f"Filas de predicciones:              {len(predictions):,}",
        f"Modelos:                            {sorted(predictions['model'].unique())}",
        f"Horizontes:                         {sorted(predictions['horizon_steps'].unique())}",
        f"Inicio de objetivos:                {predictions['target_timestamp'].min().isoformat()}",
        f"Fin de objetivos:                   {predictions['target_timestamp'].max().isoformat()}",
        "Convención del residuo:             observado − predicho",
        "Residuo positivo:                   infrapredicción",
        "Residuo negativo:                   sobrepredicción",
        "",
        "NIVELES DE TRÁFICO OBSERVADO",
        "-" * 92,
        f"Q1 (25 %):                          {traffic_thresholds['q25_mbps']:.6f} Mbit/s",
        f"Mediana (50 %):                     {traffic_thresholds['q50_mbps']:.6f} Mbit/s",
        f"Q3 (75 %):                          {traffic_thresholds['q75_mbps']:.6f} Mbit/s",
        "Niveles:                            low_q1, medium_low_q2, medium_high_q3, high_q4",
        "",
        "MEJORES MODELOS DE LA CAMPAÑA BASELINE",
        "-" * 92,
    ]

    best_rows: list[pd.Series] = []

    for horizon, model in sorted(best_models.items()):
        row = summary.loc[
            (summary["horizon_steps"] == horizon)
            & (summary["model"] == model)
        ].iloc[0]
        best_rows.append(row)
        lines.append(
            f"H{horizon:>2} ({int(row['horizon_minutes']):>3} min): "
            f"{model:<20} | "
            f"MAE={row['mae_mbps']:.6f} Mbit/s | "
            f"sesgo={row['mean_error_mbps']:+.6f} Mbit/s | "
            f"infrapredicción={row['underprediction_rate_pct']:.3f} % | "
            f"P95 abs={row['p95_absolute_error_bps'] / 1_000_000.0:.6f} Mbit/s"
        )

    lines.extend(
        [
            "",
            "COMPARACIÓN EMPAREJADA DE LOS MEJORES MODELOS FRENTE A PERSISTENCIA",
            "-" * 92,
        ]
    )

    for horizon, model in sorted(best_models.items()):
        if model == "persistence":
            lines.append(
                f"H{horizon:>2}: persistence es el modelo de referencia y el mejor por MAE."
            )
            continue

        row = paired.loc[
            (paired["horizon_steps"] == horizon)
            & (paired["model"] == model)
        ].iloc[0]
        lines.append(
            f"H{horizon:>2}: {model:<20} | "
            f"skill={row['paired_skill_pct']:+.6f} % | "
            f"delta MAE={row['paired_mae_delta_mbps']:+.6f} Mbit/s | "
            f"wins/ties/losses={int(row['wins_vs_persistence'])}/"
            f"{int(row['ties_vs_persistence'])}/"
            f"{int(row['losses_vs_persistence'])} | "
            f"p(sign test)={row['sign_test_pvalue_two_sided']:.6g}"
        )

    lines.extend(
        [
            "",
            "PEORES INTERVALOS DE LOS MEJORES MODELOS",
            "-" * 92,
        ]
    )

    for horizon, model in sorted(best_models.items()):
        subset = worst.loc[
            (worst["horizon_steps"] == horizon)
            & (worst["model"] == model)
        ].head(5)

        lines.append(
            f"H{horizon} — {model}:"
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"  {row.target_timestamp} | "
                f"real={row.y_true_mbps:.3f} Mbit/s | "
                f"pred={row.y_pred_mbps:.3f} Mbit/s | "
                f"error={row.error_mbps:+.3f} Mbit/s | "
                f"{row.error_direction}"
            )

    input_info = baseline_manifest.get("input", {})
    evaluation_info = baseline_manifest.get("evaluation", {})

    lines.extend(
        [
            "",
            "VALIDACIONES METODOLÓGICAS",
            "-" * 92,
            "PASS — residuos reconciliados como observado − predicho.",
            "PASS — error absoluto reconciliado con |residuo|.",
            "PASS — timestamps objetivo posteriores al origen.",
            "PASS — diferencia de índices igual al horizonte.",
            "PASS — predicciones únicas por horizonte, modelo y objetivo.",
            "PASS — valores observados coherentes entre modelos.",
            "PASS — métricas baseline reproducidas desde las predicciones.",
            "PASS — comparación frente a persistencia realizada sobre pares exactos.",
            "",
            "CONTEXTO DE LA CAMPAÑA BASELINE",
            "-" * 92,
            f"Entrada original:                    {input_info.get('path')}",
            f"SHA-256 entrada original:            {input_info.get('sha256')}",
            f"Fracción inicial de entrenamiento:   {evaluation_info.get('initial_train_fraction')}",
            f"Filas iniciales:                     {evaluation_info.get('initial_train_rows')}",
            f"Fin entrenamiento inicial:           {evaluation_info.get('initial_train_end_timestamp')}",
            "",
            "LIMITACIONES DE INTERPRETACIÓN",
            "-" * 92,
            coverage_note,
            "- Los niveles de tráfico se definen mediante cuartiles de los objetivos evaluados.",
            "- El test de signos compara la frecuencia de victorias, no la magnitud del efecto.",
            "- Una diferencia estadística no implica por sí sola relevancia operacional.",
            "- La comparación entre horizontes debe interpretarse con cautela porque no comparten exactamente todos los timestamps objetivo.",
            role_note,
            "",
            "RENDIMIENTO",
            "-" * 92,
            f"Tiempo de procesamiento:              {processing_seconds:.3f} s",
            "",
            "SALIDAS",
            "-" * 92,
        ]
    )

    for name, path in paths.items():
        lines.append(f"{name}: {path.resolve()}")

    lines.extend(
        [
            "",
            "Análisis de residuos terminado correctamente.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------


def main() -> int:
    arguments = parse_arguments()
    start_time = time.perf_counter()

    try:
        paths = output_paths(arguments)
        validate_arguments(arguments, paths)
        ensure_directories(paths)

        predictions = load_predictions(arguments.predictions_parquet)
        baseline_metrics = load_baseline_metrics(arguments.baseline_metrics)
        baseline_manifest = load_manifest(arguments.baseline_manifest)

        horizons = sorted(
            int(value)
            for value in predictions["horizon_steps"].unique()
        )
        add_horizon_figure_paths(
            paths,
            arguments.figures_dir,
            arguments.prefix,
            horizons,
        )

        # Las rutas de figuras por horizonte se conocen después de leer la entrada.
        if not arguments.overwrite:
            existing = [
                str(path)
                for path in paths.values()
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Ya existen salidas. Usa --overwrite o cambia el prefijo:\n"
                    + "\n".join(existing)
                )

        ensure_directories(paths)

        validate_against_baseline_metrics(
            predictions,
            baseline_metrics,
        )

        manifest_models = set(baseline_manifest["models"])
        prediction_models = set(predictions["model"])

        if manifest_models != prediction_models:
            raise ValueError(
                "Los modelos del manifiesto no coinciden con las predicciones."
            )

        manifest_horizons = set(
            int(value)
            for value in baseline_manifest["evaluation"]["horizons"]
        )
        if manifest_horizons != set(horizons):
            raise ValueError(
                "Los horizontes del manifiesto no coinciden con las predicciones."
            )

        print("=" * 92)
        print("ANÁLISIS DE RESIDUOS — UGR'16")
        print("=" * 92)
        print(f"Predicciones: {len(predictions):,}")
        print(f"Modelos:      {len(prediction_models)}")
        print(f"Horizontes:   {horizons}")
        print()

        summary = residual_summary(predictions)
        best_models = determine_best_models(baseline_metrics)

        enriched, traffic_thresholds = add_traffic_levels(predictions)
        enriched["target_hour"] = enriched["target_timestamp"].dt.hour

        traffic_table = grouped_error_table(
            enriched,
            [
                "horizon_steps",
                "horizon_minutes",
                "model",
                "traffic_level",
            ],
        )
        traffic_table["traffic_level"] = traffic_table[
            "traffic_level"
        ].astype(str)

        hourly_table = grouped_error_table(
            enriched,
            [
                "horizon_steps",
                "horizon_minutes",
                "model",
                "target_hour",
            ],
        )

        paired = paired_vs_persistence(
            predictions,
            tie_tolerance_bps=arguments.tie_tolerance_bps,
        )
        worst = worst_intervals(
            enriched,
            top_n=arguments.top_worst,
        )
        best_residuals = best_model_residuals(
            enriched,
            best_models,
        )

        atomic_write_csv(summary, paths["summary_csv"])
        atomic_write_csv(traffic_table, paths["traffic_level_csv"])
        atomic_write_csv(hourly_table, paths["hourly_csv"])
        atomic_write_csv(paired, paths["paired_csv"])
        atomic_write_csv(worst, paths["worst_intervals_csv"])
        atomic_write_csv(
            best_residuals,
            paths["best_model_residuals_csv"],
        )

        plot_metric_by_horizon(
            summary=summary,
            metric="mean_error_mbps",
            ylabel="Sesgo medio: observado − predicho (Mbit/s)",
            title="Sesgo medio por modelo y horizonte",
            path=paths["figure_bias"],
            dpi=arguments.dpi,
            reference_line=0.0,
        )
        plot_metric_by_horizon(
            summary=summary,
            metric="underprediction_rate_pct",
            ylabel="Infrapredicciones (%)",
            title="Tasa de infrapredicción por modelo y horizonte",
            path=paths["figure_underprediction"],
            dpi=arguments.dpi,
            reference_line=50.0,
        )
        plot_paired_skill(
            paired=paired,
            path=paths["figure_paired_skill"],
            dpi=arguments.dpi,
        )
        plot_traffic_level_best_models(
            traffic_table=traffic_table,
            best_models=best_models,
            path=paths["figure_traffic_level"],
            dpi=arguments.dpi,
        )
        plot_hourly_best_models(
            hourly_table=hourly_table,
            best_models=best_models,
            path=paths["figure_hourly"],
            dpi=arguments.dpi,
        )

        for horizon, model in sorted(best_models.items()):
            plot_best_model_residuals(
                best_residuals=best_residuals,
                horizon=horizon,
                model=model,
                path=paths[f"figure_h{horizon}"],
                dpi=arguments.dpi,
            )

        processing_seconds = time.perf_counter() - start_time

        report = build_report(
            predictions=predictions,
            baseline_manifest=baseline_manifest,
            campaign_id=arguments.campaign_id,
            summary=summary,
            paired=paired,
            traffic_thresholds=traffic_thresholds,
            worst=worst,
            best_models=best_models,
            paths=paths,
            processing_seconds=processing_seconds,
        )
        atomic_write_text(report, paths["report_txt"])

        output_hashes: dict[str, dict[str, Any]] = {}
        for name, path in paths.items():
            if name == "manifest_json":
                continue
            output_hashes[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }

        campaign_id = arguments.campaign_id
        source_baseline_campaign_id = str(
            baseline_manifest.get(
                "campaign_id",
                "not_available",
            )
        )
        capture_name = str(
            baseline_manifest.get(
                "capture_name",
                "UGR'16",
            )
        )
        evaluation_role = str(
            baseline_manifest.get(
                "evaluation_role",
                baseline_manifest.get(
                    "evaluation",
                    {},
                ).get(
                    "evaluation_role",
                    "not_available",
                ),
            )
        )

        baseline_input = baseline_manifest.get("input", {})
        input_rows = float(baseline_input.get("rows", 0))
        input_delta_seconds = float(
            baseline_input.get(
                "expected_delta_seconds",
                0,
            )
        )
        coverage_hours = float(
            input_rows * input_delta_seconds / 3600.0
        )
        coverage_days = coverage_hours / 24.0

        manifest = {
            "campaign_id": campaign_id,
            "capture_name": capture_name,
            "evaluation_role": evaluation_role,
            "source_baseline_campaign_id": (
                source_baseline_campaign_id
            ),
            "inputs": {
                "predictions_parquet": {
                    "path": str(arguments.predictions_parquet.resolve()),
                    "sha256": sha256_file(arguments.predictions_parquet),
                    "rows": int(len(predictions)),
                },
                "baseline_metrics": {
                    "path": str(arguments.baseline_metrics.resolve()),
                    "sha256": sha256_file(arguments.baseline_metrics),
                    "rows": int(len(baseline_metrics)),
                },
                "baseline_manifest": {
                    "path": str(arguments.baseline_manifest.resolve()),
                    "sha256": sha256_file(arguments.baseline_manifest),
                },
            },
            "analysis": {
                "campaign_id": campaign_id,
                "capture_name": capture_name,
                "evaluation_role": evaluation_role,
                "coverage_hours": coverage_hours,
                "coverage_days": coverage_days,
                "error_definition": "y_true_bps - y_pred_bps",
                "positive_error_meaning": "underprediction",
                "negative_error_meaning": "overprediction",
                "horizons": horizons,
                "models": sorted(prediction_models),
                "best_models_by_horizon": {
                    str(horizon): model
                    for horizon, model in sorted(best_models.items())
                },
                "traffic_level_thresholds": traffic_thresholds,
                "traffic_level_order": TRAFFIC_LEVEL_ORDER,
                "top_worst_per_model_horizon": int(arguments.top_worst),
                "tie_tolerance_bps": float(arguments.tie_tolerance_bps),
                "paired_reference_model": "persistence",
                "paired_test": "two-sided exact sign test excluding ties",
                "processing_seconds": float(processing_seconds),
            },
            "validation": {
                "residual_reconciliation": True,
                "absolute_error_reconciliation": True,
                "future_targets_only": True,
                "index_horizon_reconciliation": True,
                "unique_prediction_keys": True,
                "consistent_observed_values": True,
                "baseline_metrics_reproduced": True,
                "exact_paired_alignment": True,
            },
            "outputs": output_hashes,
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
            },
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        atomic_write_json(manifest, paths["manifest_json"])

        print(report)
        print(f"Manifiesto: {paths['manifest_json'].resolve()}")

    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
