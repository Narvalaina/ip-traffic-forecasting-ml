#!/usr/bin/env python3
"""
Evaluación temporal de baselines y Ridge sobre UGR'16 (serie de 5 minutos).

Modelos:
- persistencia;
- medias móviles de 3, 6 y 12 observaciones;
- persistencia estacional de 1 h y 24 h;
- Ridge autorregresiva directa por horizonte.

La evaluación es walk-forward con ventana expansiva, sin barajado y sin fuga
futura. El alpha de Ridge puede seleccionarse mediante TimeSeriesSplit usando
solo la ventana inicial de entrenamiento o fijarse desde el manifiesto de una
campaña previa para realizar validación temporal externa sin retuning.
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
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path(
    "data/processed/ugr16/march_week3_prepared_5min.parquet"
)
DEFAULT_PREFIX = "ugr16_march_week3_baselines"
DEFAULT_CAPTURE_NAME = "UGR'16 March Week #3"
DEFAULT_CAMPAIGN_ID = "UGR16-MARCH-BASELINES-001"
DEFAULT_EVALUATION_ROLE = "development"
DEFAULT_HORIZONS = [1, 3, 6, 12]
DEFAULT_MOVING_WINDOWS = [3, 6, 12]
DEFAULT_SEASONAL_LAGS = [12, 288]
DEFAULT_RIDGE_LAGS = list(range(1, 13)) + [24, 72, 144, 288]
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa persistencia, medias móviles, baselines estacionales "
            "y Ridge mediante walk-forward temporal."
        )
    )
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", default="bitrate_bps")
    parser.add_argument(
        "--capture-name",
        default=DEFAULT_CAPTURE_NAME,
        help="Nombre legible de la captura para informes y figuras.",
    )
    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help="Identificador reproducible de la campaña.",
    )
    parser.add_argument(
        "--evaluation-role",
        choices=["development", "external_validation"],
        default=DEFAULT_EVALUATION_ROLE,
        help=(
            "development permite seleccionar alpha en la ventana inicial; "
            "external_validation exige fijarlo desde un manifiesto previo."
        ),
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS
    )
    parser.add_argument(
        "--initial-train-fraction", type=float, default=0.70
    )
    parser.add_argument(
        "--moving-average-windows",
        type=int,
        nargs="+",
        default=DEFAULT_MOVING_WINDOWS,
    )
    parser.add_argument(
        "--seasonal-lags",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONAL_LAGS,
    )
    parser.add_argument(
        "--ridge-lags",
        type=int,
        nargs="+",
        default=DEFAULT_RIDGE_LAGS,
        help=(
            "Retardos históricos. El valor actual se incluye siempre como "
            "característica adicional."
        ),
    )
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=DEFAULT_ALPHAS
    )
    parser.add_argument(
        "--fixed-ridge-alpha-manifest",
        type=Path,
        default=None,
        help=(
            "Manifiesto de una campaña previa con selected_ridge_alphas. "
            "Cuando se proporciona, no se realiza selección CV en la "
            "captura actual."
        ),
    )
    parser.add_argument("--ridge-cv-splits", type=int, default=5)
    parser.add_argument(
        "--disable-calendar-features", action="store_true"
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("results/metrics")
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/predictions"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/ugr16_baselines"),
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unique_positive_ints(values: Iterable[int], name: str) -> list[int]:
    result = sorted(set(int(value) for value in values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} debe contener enteros positivos.")
    return result


def unique_positive_floats(
    values: Iterable[float], name: str
) -> list[float]:
    result = sorted(set(float(value) for value in values))
    if not result or any(
        not np.isfinite(value) or value <= 0 for value in result
    ):
        raise ValueError(f"{name} debe contener valores positivos.")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_ridge_alphas(
    path: Path,
    horizons: list[int],
) -> tuple[dict[int, float], dict]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    raw = manifest.get("selected_ridge_alphas")
    if not isinstance(raw, dict):
        raise ValueError(
            "El manifiesto externo no contiene selected_ridge_alphas."
        )

    selected: dict[int, float] = {}
    for horizon in horizons:
        key = str(horizon)
        if key not in raw:
            raise ValueError(
                f"El manifiesto externo no contiene alpha para H{horizon}."
            )
        alpha = float(raw[key])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                f"Alpha externo inválido para H{horizon}: {alpha}"
            )
        selected[horizon] = alpha

    provenance = {
        "strategy": "external_manifest",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_campaign_id": manifest.get("campaign_id"),
        "source_capture_name": manifest.get("capture_name"),
        "selected_ridge_alphas": {
            str(horizon): alpha
            for horizon, alpha in selected.items()
        },
    }
    return selected, provenance


def make_paths(args: argparse.Namespace) -> dict[str, Path]:
    prefix = args.prefix
    paths = {
        "metrics_csv": args.metrics_dir / f"{prefix}_metrics.csv",
        "ridge_cv_csv": args.metrics_dir / f"{prefix}_ridge_cv.csv",
        "manifest_json": args.metrics_dir / f"{prefix}_manifest.json",
        "report_txt": args.metrics_dir / f"{prefix}_report.txt",
        "predictions_csv": (
            args.predictions_dir / f"{prefix}_predictions.csv"
        ),
        "predictions_parquet": (
            args.predictions_dir / f"{prefix}_predictions.parquet"
        ),
        "figure_mae": (
            args.figures_dir / f"{prefix}_01_mae_by_horizon.png"
        ),
        "figure_rmse": (
            args.figures_dir / f"{prefix}_02_rmse_by_horizon.png"
        ),
    }
    for position, horizon in enumerate(sorted(args.horizons), start=3):
        paths[f"figure_h{horizon}"] = (
            args.figures_dir
            / f"{prefix}_{position:02d}_h{horizon}_predictions.png"
        )
    return paths


def validate_args(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if not args.input_parquet.is_file():
        raise FileNotFoundError(
            f"No se encuentra la entrada: {args.input_parquet}"
        )
    if not 0.50 <= args.initial_train_fraction < 0.95:
        raise ValueError(
            "--initial-train-fraction debe estar entre 0.50 y 0.95."
        )
    if args.ridge_cv_splits < 2:
        raise ValueError("--ridge-cv-splits debe ser al menos 2.")
    if args.dpi <= 0:
        raise ValueError("--dpi debe ser positivo.")
    args.capture_name = args.capture_name.strip()
    args.campaign_id = args.campaign_id.strip()
    if not args.capture_name:
        raise ValueError("--capture-name no puede estar vacío.")
    if not args.campaign_id:
        raise ValueError("--campaign-id no puede estar vacío.")
    if (
        args.fixed_ridge_alpha_manifest is not None
        and not args.fixed_ridge_alpha_manifest.is_file()
    ):
        raise FileNotFoundError(
            "No se encuentra el manifiesto de alpha: "
            f"{args.fixed_ridge_alpha_manifest}"
        )
    if (
        args.evaluation_role == "external_validation"
        and args.fixed_ridge_alpha_manifest is None
    ):
        raise ValueError(
            "external_validation exige --fixed-ridge-alpha-manifest "
            "para evitar retuning en la captura de validación."
        )
    if not args.overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite o cambia el prefijo:\n"
                + "\n".join(existing)
            )


def ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_series(path: Path, target: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path).copy()
    required = {"timestamp", target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Hay timestamps inválidos.")
    if df[target].isna().any() or not np.isfinite(df[target]).all():
        raise ValueError(f"{target} contiene valores inválidos.")
    if (df[target] < 0).any():
        raise ValueError(f"{target} contiene valores negativos.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    duplicates = int(df["timestamp"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Hay {duplicates} timestamps duplicados.")

    differences = df["timestamp"].diff().dropna()
    if differences.empty:
        raise ValueError("La serie es demasiado corta.")
    expected_delta = differences.mode().iloc[0]
    discontinuities = int((differences != expected_delta).sum())
    if discontinuities:
        raise ValueError(
            f"Hay {discontinuities} discontinuidades temporales."
        )

    summary = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start": df["timestamp"].min().isoformat(),
        "end": df["timestamp"].max().isoformat(),
        "expected_delta_seconds": float(expected_delta.total_seconds()),
        "duplicate_timestamps": duplicates,
        "missing_intervals": discontinuities,
        "target": target,
        "target_min": float(df[target].min()),
        "target_max": float(df[target].max()),
        "target_mean": float(df[target].mean()),
    }
    return df, summary


def build_features(
    df: pd.DataFrame,
    target: str,
    lags: list[int],
    include_calendar: bool,
) -> pd.DataFrame:
    series = df[target].astype("float64")
    timestamps = df["timestamp"]
    features = pd.DataFrame(index=df.index)
    features["current"] = series
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)

    if include_calendar:
        hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
        dow = timestamps.dt.dayofweek.astype("float64")
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        features["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(
            "float64"
        )
        features["trend"] = np.arange(len(df), dtype="float64")
    return features


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def select_alpha(
    features: pd.DataFrame,
    target_values: np.ndarray,
    horizon: int,
    first_feature_origin: int,
    initial_train_end: int,
    alphas: list[float],
    requested_splits: int,
) -> tuple[float, list[dict]]:
    last_origin = initial_train_end - horizon
    origins = np.arange(first_feature_origin, last_origin + 1)
    if len(origins) < 20:
        raise ValueError(
            f"Solo hay {len(origins)} ejemplos iniciales para H{horizon}."
        )

    X = features.iloc[origins].to_numpy(dtype="float64")
    y = target_values[origins + horizon]
    if not np.isfinite(X).all():
        raise ValueError("La matriz inicial de Ridge contiene NaN o inf.")

    n_splits = min(requested_splits, len(X) - 1)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict] = []

    for alpha in alphas:
        fold_mae = []
        for train_idx, valid_idx in splitter.split(X):
            model = ridge_pipeline(alpha)
            model.fit(X[train_idx], y[train_idx])
            pred = np.maximum(model.predict(X[valid_idx]), 0.0)
            fold_mae.append(float(np.mean(np.abs(y[valid_idx] - pred))))
        results.append(
            {
                "horizon_steps": horizon,
                "alpha": alpha,
                "cv_splits": n_splits,
                "mean_mae_bps": float(np.mean(fold_mae)),
                "std_mae_bps": float(np.std(fold_mae)),
                "source": "initial_train_cv",
            }
        )

    results.sort(key=lambda item: (item["mean_mae_bps"], item["alpha"]))
    selected = float(results[0]["alpha"])
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
        item["selected"] = bool(math.isclose(item["alpha"], selected))
    return selected, results


def seasonal_name(lag: int, interval_minutes: int) -> str:
    minutes = lag * interval_minutes
    if minutes % 60 == 0:
        return f"seasonal_{minutes // 60}h"
    return f"seasonal_{minutes}min"


def walk_forward(
    df: pd.DataFrame,
    target: str,
    features: pd.DataFrame,
    horizons: list[int],
    moving_windows: list[int],
    seasonal_lags: list[int],
    initial_train_end: int,
    first_feature_origin: int,
    selected_alphas: dict[int, float],
    interval_minutes: int,
) -> tuple[pd.DataFrame, int, int]:
    values = df[target].to_numpy(dtype="float64")
    timestamps = df["timestamp"].to_numpy()
    records: list[dict] = []
    clipped = 0

    first_eval_origin = max(
        initial_train_end,
        first_feature_origin,
        max(seasonal_lags) - min(horizons),
        max(moving_windows) - 1,
    )

    for horizon in horizons:
        last_origin = len(df) - horizon - 1
        if first_eval_origin > last_origin:
            raise ValueError(f"No quedan orígenes para H{horizon}.")

        print(
            f"Evaluando H{horizon} ({horizon * interval_minutes} minutos)..."
        )
        alpha = selected_alphas[horizon]

        for origin in range(first_eval_origin, last_origin + 1):
            target_index = origin + horizon
            y_true = float(values[target_index])
            predictions = {"persistence": float(values[origin])}

            for window in moving_windows:
                predictions[f"moving_average_{window}"] = float(
                    values[origin - window + 1 : origin + 1].mean()
                )

            for lag in seasonal_lags:
                reference = origin + horizon - lag
                if reference < 0 or reference > origin:
                    raise RuntimeError(
                        f"Referencia estacional inválida: H{horizon}, lag={lag}."
                    )
                predictions[seasonal_name(lag, interval_minutes)] = float(
                    values[reference]
                )

            last_training_origin = origin - horizon
            train_origins = np.arange(
                first_feature_origin, last_training_origin + 1
            )
            X_train = features.iloc[train_origins].to_numpy(dtype="float64")
            y_train = values[train_origins + horizon]
            X_current = features.iloc[[origin]].to_numpy(dtype="float64")

            model = ridge_pipeline(alpha)
            model.fit(X_train, y_train)
            raw_pred = float(model.predict(X_current)[0])
            ridge_pred = max(raw_pred, 0.0)
            clipped += int(raw_pred < 0)
            predictions["ridge"] = ridge_pred

            for model_name, y_pred in predictions.items():
                records.append(
                    {
                        "origin_timestamp": pd.Timestamp(timestamps[origin]),
                        "target_timestamp": pd.Timestamp(
                            timestamps[target_index]
                        ),
                        "origin_index": origin,
                        "target_index": target_index,
                        "horizon_steps": horizon,
                        "horizon_minutes": horizon * interval_minutes,
                        "model": model_name,
                        "y_true_bps": y_true,
                        "y_pred_bps": y_pred,
                        "error_bps": y_true - y_pred,
                        "absolute_error_bps": abs(y_true - y_pred),
                        "ridge_alpha": (
                            alpha if model_name == "ridge" else np.nan
                        ),
                    }
                )

    predictions_df = pd.DataFrame(records).sort_values(
        ["horizon_steps", "target_timestamp", "model"]
    ).reset_index(drop=True)
    if predictions_df.empty:
        raise RuntimeError("No se generaron predicciones.")
    if not np.isfinite(
        predictions_df[
            ["y_true_bps", "y_pred_bps", "error_bps"]
        ].to_numpy()
    ).all():
        raise RuntimeError("Las predicciones contienen valores no finitos.")
    return predictions_df, clipped, first_eval_origin


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2.0 * np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(y_true, dtype="float64"),
        where=denominator != 0,
    )
    return float(100.0 * np.mean(ratio))


def compute_metrics(
    predictions: pd.DataFrame, mase_scale: float
) -> pd.DataFrame:
    rows = []
    for (horizon, model), group in predictions.groupby(
        ["horizon_steps", "model"], sort=True
    ):
        y_true = group["y_true_bps"].to_numpy(dtype="float64")
        y_pred = group["y_pred_bps"].to_numpy(dtype="float64")
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        rows.append(
            {
                "horizon_steps": int(horizon),
                "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                "model": model,
                "n_predictions": int(len(group)),
                "mae_bps": mae,
                "mae_mbps": mae / 1_000_000.0,
                "rmse_bps": rmse,
                "rmse_mbps": rmse / 1_000_000.0,
                "smape_pct": smape(y_true, y_pred),
                "mase": mae / mase_scale,
                "mean_error_bps": float(np.mean(errors)),
                "median_absolute_error_bps": float(
                    np.median(np.abs(errors))
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    persistence = metrics.loc[
        metrics["model"] == "persistence",
        ["horizon_steps", "mae_bps"],
    ].rename(columns={"mae_bps": "persistence_mae_bps"})
    metrics = metrics.merge(
        persistence,
        on="horizon_steps",
        how="left",
        validate="many_to_one",
    )
    metrics["mae_skill_vs_persistence_pct"] = 100.0 * (
        metrics["persistence_mae_bps"] - metrics["mae_bps"]
    ) / metrics["persistence_mae_bps"]
    metrics = metrics.drop(columns="persistence_mae_bps")
    return metrics.sort_values(
        ["horizon_steps", "mae_bps", "model"]
    ).reset_index(drop=True)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    df.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def plot_metric(
    metrics: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    pivot = metrics.pivot(
        index="horizon_minutes", columns="model", values=column
    ).sort_index()
    figure, axis = plt.subplots(figsize=(11, 6))
    for model in pivot.columns:
        axis.plot(pivot.index, pivot[model], marker="o", label=model)
    axis.set_title(title)
    axis.set_xlabel("Horizonte de predicción (minutos)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def plot_predictions(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    horizon: int,
    path: Path,
    dpi: int,
    capture_name: str,
) -> None:
    subset = predictions[predictions["horizon_steps"] == horizon].copy()
    horizon_metrics = metrics[
        metrics["horizon_steps"] == horizon
    ].sort_values("mae_bps")
    best_non_ridge = horizon_metrics[
        horizon_metrics["model"] != "ridge"
    ].iloc[0]["model"]
    models = list(dict.fromkeys(["persistence", best_non_ridge, "ridge"]))

    actual = subset[
        ["target_timestamp", "y_true_bps"]
    ].drop_duplicates("target_timestamp").sort_values("target_timestamp")

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        actual["target_timestamp"],
        actual["y_true_bps"] / 1_000_000.0,
        linewidth=2.0,
        label="observado",
    )
    for model in models:
        data = subset[subset["model"] == model].sort_values(
            "target_timestamp"
        )
        axis.plot(
            data["target_timestamp"],
            data["y_pred_bps"] / 1_000_000.0,
            linewidth=1.2,
            label=model,
        )
    minutes = int(subset["horizon_minutes"].iloc[0])
    axis.set_title(
        f"{capture_name} — predicciones walk-forward "
        f"H{horizon} ({minutes} minutos)"
    )
    axis.set_xlabel("Tiempo objetivo")
    axis.set_ylabel("Tasa equivalente (Mbit/s)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def build_report(
    capture_name: str,
    campaign_id: str,
    evaluation_role: str,
    input_summary: dict,
    evaluation: dict,
    metrics: pd.DataFrame,
    selected_alphas: dict[int, float],
    alpha_provenance: dict,
    paths: dict[str, Path],
) -> str:
    lines = [
        "=" * 88,
        "EVALUACIÓN TEMPORAL DE BASELINES Y RIDGE — UGR'16",
        "=" * 88,
        f"Campaña:                     {campaign_id}",
        f"Captura:                     {capture_name}",
        f"Rol metodológico:            {evaluation_role}",
        f"Entrada:                     {input_summary['path']}",
        f"SHA-256 entrada:             {input_summary['sha256']}",
        f"Objetivo:                    {input_summary['target']}",
        f"Filas:                       {input_summary['rows']:,}",
        f"Inicio:                      {input_summary['start']}",
        f"Fin:                         {input_summary['end']}",
        f"Resolución:                  {input_summary['expected_delta_seconds'] / 60:.0f} minutos",
        "",
        "DISEÑO TEMPORAL",
        "-" * 88,
        f"Fracción inicial:            {evaluation['initial_train_fraction']:.2f}",
        f"Filas iniciales:             {evaluation['initial_train_rows']:,}",
        f"Fin entrenamiento inicial:   {evaluation['initial_train_end_timestamp']}",
        f"Primer origen evaluado:      {evaluation['first_evaluation_origin_timestamp']}",
        "Evaluación:                  walk-forward expansiva",
        "Barajado:                    no",
        (
            "Selección alpha:             "
            + evaluation["ridge_alpha_strategy_label"]
        ),
        "Escalado Ridge:              ajustado en cada origen",
        "Coeficientes Ridge:          reestimados en cada origen con "
        "ventana expansiva de la captura actual",
        f"Escala MASE:                 {evaluation['mase_scale_bps']:.6f} bps",
        f"Predicciones Ridge recortadas:{evaluation['ridge_negative_predictions_clipped']:,}",
        "",
        "ALPHA RIDGE POR HORIZONTE",
        "-" * 88,
    ]
    if alpha_provenance["strategy"] == "external_manifest":
        lines += [
            "Origen:                      manifiesto externo",
            f"Manifiesto fuente:          {alpha_provenance['path']}",
            f"SHA-256 manifiesto:         {alpha_provenance['sha256']}",
        ]
        if alpha_provenance.get("source_campaign_id"):
            lines.append(
                "Campaña fuente:             "
                f"{alpha_provenance['source_campaign_id']}"
            )
        if alpha_provenance.get("source_capture_name"):
            lines.append(
                "Captura fuente:             "
                f"{alpha_provenance['source_capture_name']}"
            )
    else:
        lines.append(
            "Origen:                      TimeSeriesSplit sobre "
            "la ventana inicial"
        )
    interval_minutes = int(input_summary["expected_delta_seconds"] / 60)
    for horizon in sorted(selected_alphas):
        lines.append(
            f"H{horizon:>2} ({horizon * interval_minutes:>3} min): "
            f"alpha={selected_alphas[horizon]}"
        )

    columns = [
        "horizon_steps",
        "horizon_minutes",
        "model",
        "n_predictions",
        "mae_mbps",
        "rmse_mbps",
        "smape_pct",
        "mase",
        "mae_skill_vs_persistence_pct",
    ]
    lines += [
        "",
        "MÉTRICAS",
        "-" * 88,
        metrics[columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "MEJOR MODELO POR HORIZONTE SEGÚN MAE",
        "-" * 88,
    ]
    for horizon, group in metrics.groupby("horizon_steps", sort=True):
        best = group.sort_values("mae_bps").iloc[0]
        lines.append(
            f"H{int(horizon):>2} ({int(best['horizon_minutes']):>3} min): "
            f"{best['model']} | MAE={best['mae_mbps']:.6f} Mbit/s | "
            f"RMSE={best['rmse_mbps']:.6f} Mbit/s | "
            f"sMAPE={best['smape_pct']:.6f} % | MASE={best['mase']:.6f}"
        )

    coverage_hours = (
        input_summary["rows"]
        * input_summary["expected_delta_seconds"]
        / 3600.0
    )
    coverage_days = coverage_hours / 24.0
    if coverage_days < 7.0:
        coverage_limitation = (
            f"- La captura cubre {coverage_hours:.2f} horas "
            f"({coverage_days:.2f} días); la evidencia de estacionalidad "
            "diaria y semanal debe interpretarse con cautela."
        )
    else:
        coverage_limitation = (
            f"- La captura cubre {coverage_hours:.2f} horas "
            f"({coverage_days:.2f} días); una única semana no permite "
            "demostrar estabilidad entre semanas."
        )

    if evaluation_role == "external_validation":
        role_limitation = (
            "- Los hiperparámetros Ridge se fijaron antes de observar esta "
            "campaña y no se re-seleccionaron sobre la captura de validación."
        )
    else:
        role_limitation = (
            "- Los hiperparámetros Ridge se seleccionaron únicamente con "
            "la ventana inicial de esta captura."
        )

    lines += [
        "",
        "LIMITACIONES",
        "-" * 88,
        coverage_limitation,
        role_limitation,
        "- La persistencia diaria se evalúa como baseline, no como prueba "
        "por sí sola de estacionalidad estable.",
        "- Ridge se ajusta de forma directa e independiente para cada horizonte.",
        "- MASE usa una escala fija calculada solo con la ventana inicial.",
        "- Los resultados corresponden a esta captura y deben contrastarse "
        "con otras semanas sin concatenar segmentos discontinuos.",
        "",
        "RENDIMIENTO",
        "-" * 88,
        f"Tiempo de procesamiento:     {evaluation['processing_seconds']:.2f} s",
        "",
        "SALIDAS",
        "-" * 88,
    ]
    for name, path in paths.items():
        lines.append(f"{name}: {path.resolve()}")
    lines += ["", "Evaluación terminada correctamente.", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    try:
        args.horizons = unique_positive_ints(args.horizons, "--horizons")
        args.moving_average_windows = unique_positive_ints(
            args.moving_average_windows, "--moving-average-windows"
        )
        args.seasonal_lags = unique_positive_ints(
            args.seasonal_lags, "--seasonal-lags"
        )
        args.ridge_lags = unique_positive_ints(
            args.ridge_lags, "--ridge-lags"
        )
        args.ridge_alphas = unique_positive_floats(
            args.ridge_alphas, "--ridge-alphas"
        )

        paths = make_paths(args)
        validate_args(args, paths)
        ensure_directories(paths)
        df, input_summary = load_series(args.input_parquet, args.target)

        interval_minutes_float = (
            input_summary["expected_delta_seconds"] / 60.0
        )
        if not interval_minutes_float.is_integer():
            raise ValueError("La resolución no es un número entero de minutos.")
        interval_minutes = int(interval_minutes_float)
        if interval_minutes != 5:
            print(
                f"ADVERTENCIA: la entrada tiene {interval_minutes} minutos; "
                "la campaña fue diseñada para 5 minutos.",
                file=sys.stderr,
            )

        initial_train_rows = int(
            math.floor(len(df) * args.initial_train_fraction)
        )
        initial_train_end = initial_train_rows - 1
        first_feature_origin = max(args.ridge_lags)
        if initial_train_end - max(args.horizons) <= first_feature_origin:
            raise ValueError(
                "La ventana inicial no deja suficientes ejemplos para Ridge."
            )

        include_calendar = not args.disable_calendar_features
        features = build_features(
            df, args.target, args.ridge_lags, include_calendar
        )
        values = df[args.target].to_numpy(dtype="float64")
        mase_scale = float(
            np.mean(np.abs(np.diff(values[: initial_train_end + 1])))
        )
        if not np.isfinite(mase_scale) or mase_scale <= 0:
            raise ValueError("La escala MASE no es válida.")

        selected_alphas: dict[int, float]
        cv_rows: list[dict] = []
        if args.fixed_ridge_alpha_manifest is not None:
            print("=" * 88)
            print("ALPHAS RIDGE FIJADOS DESDE UNA CAMPAÑA PREVIA")
            print("=" * 88)
            selected_alphas, alpha_provenance = load_fixed_ridge_alphas(
                args.fixed_ridge_alpha_manifest,
                args.horizons,
            )
            for horizon in args.horizons:
                alpha = selected_alphas[horizon]
                cv_rows.append(
                    {
                        "horizon_steps": horizon,
                        "alpha": alpha,
                        "cv_splits": np.nan,
                        "mean_mae_bps": np.nan,
                        "std_mae_bps": np.nan,
                        "source": "external_manifest",
                        "rank": 1,
                        "selected": True,
                    }
                )
                print(
                    f"H{horizon} "
                    f"({horizon * interval_minutes} min): alpha={alpha}"
                )
            ridge_alpha_strategy_label = (
                "fijado desde manifiesto externo; sin retuning"
            )
        else:
            print("=" * 88)
            print("SELECCIÓN TEMPORAL DE ALPHA PARA RIDGE")
            print("=" * 88)
            selected_alphas = {}
            for horizon in args.horizons:
                selected, rows = select_alpha(
                    features,
                    values,
                    horizon,
                    first_feature_origin,
                    initial_train_end,
                    args.ridge_alphas,
                    args.ridge_cv_splits,
                )
                selected_alphas[horizon] = selected
                cv_rows.extend(rows)
                print(
                    f"H{horizon} "
                    f"({horizon * interval_minutes} min): "
                    f"alpha={selected}"
                )
            alpha_provenance = {
                "strategy": "initial_train_cv",
                "path": None,
                "sha256": None,
                "source_campaign_id": args.campaign_id,
                "source_capture_name": args.capture_name,
                "selected_ridge_alphas": {
                    str(horizon): alpha
                    for horizon, alpha in selected_alphas.items()
                },
            }
            ridge_alpha_strategy_label = (
                "TimeSeriesSplit solo sobre la ventana inicial"
            )

        print()
        print("=" * 88)
        print("EVALUACIÓN WALK-FORWARD")
        print("=" * 88)
        predictions, clipped, first_eval_origin = walk_forward(
            df,
            args.target,
            features,
            args.horizons,
            args.moving_average_windows,
            args.seasonal_lags,
            initial_train_end,
            first_feature_origin,
            selected_alphas,
            interval_minutes,
        )
        metrics = compute_metrics(predictions, mase_scale)
        cv_df = pd.DataFrame(cv_rows).sort_values(
            ["horizon_steps", "rank", "alpha"]
        ).reset_index(drop=True)

        atomic_csv(metrics, paths["metrics_csv"])
        atomic_csv(cv_df, paths["ridge_cv_csv"])
        atomic_csv(predictions, paths["predictions_csv"])
        atomic_parquet(predictions, paths["predictions_parquet"])

        plot_metric(
            metrics,
            "mae_mbps",
            "MAE (Mbit/s)",
            f"{args.capture_name} — MAE por modelo y horizonte",
            paths["figure_mae"],
            args.dpi,
        )
        plot_metric(
            metrics,
            "rmse_mbps",
            "RMSE (Mbit/s)",
            f"{args.capture_name} — RMSE por modelo y horizonte",
            paths["figure_rmse"],
            args.dpi,
        )
        for horizon in args.horizons:
            plot_predictions(
                predictions,
                metrics,
                horizon,
                paths[f"figure_h{horizon}"],
                args.dpi,
                args.capture_name,
            )

        processing_seconds = time.perf_counter() - start_time
        evaluation = {
            "capture_name": args.capture_name,
            "campaign_id": args.campaign_id,
            "evaluation_role": args.evaluation_role,
            "ridge_alpha_strategy": alpha_provenance["strategy"],
            "ridge_alpha_strategy_label": ridge_alpha_strategy_label,
            "ridge_alpha_provenance": alpha_provenance,
            "ridge_weights_strategy": (
                "expanding_refit_within_current_capture"
            ),
            "initial_train_fraction": args.initial_train_fraction,
            "initial_train_rows": initial_train_rows,
            "initial_train_end_timestamp": df.loc[
                initial_train_end, "timestamp"
            ].isoformat(),
            "first_evaluation_origin_timestamp": df.loc[
                first_eval_origin, "timestamp"
            ].isoformat(),
            "mase_scale_bps": mase_scale,
            "horizons": args.horizons,
            "moving_average_windows": args.moving_average_windows,
            "seasonal_lags": args.seasonal_lags,
            "ridge_lags": args.ridge_lags,
            "ridge_alphas": args.ridge_alphas,
            "ridge_cv_splits": args.ridge_cv_splits,
            "calendar_features": include_calendar,
            "ridge_negative_predictions_clipped": clipped,
            "processing_seconds": processing_seconds,
        }

        report = build_report(
            args.capture_name,
            args.campaign_id,
            args.evaluation_role,
            input_summary,
            evaluation,
            metrics,
            selected_alphas,
            alpha_provenance,
            paths,
        )
        atomic_text(report, paths["report_txt"])

        output_hashes = {}
        for name, path in paths.items():
            if name == "manifest_json":
                continue
            output_hashes[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }

        manifest = {
            "campaign_id": args.campaign_id,
            "capture_name": args.capture_name,
            "evaluation_role": args.evaluation_role,
            "input": input_summary,
            "evaluation": evaluation,
            "selected_ridge_alphas": {
                str(horizon): alpha
                for horizon, alpha in selected_alphas.items()
            },
            "models": [
                "persistence",
                *[
                    f"moving_average_{window}"
                    for window in args.moving_average_windows
                ],
                *[
                    seasonal_name(lag, interval_minutes)
                    for lag in args.seasonal_lags
                ],
                "ridge",
            ],
            "outputs": output_hashes,
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "scikit_learn": sklearn.__version__,
                "platform": platform.platform(),
            },
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        atomic_json(manifest, paths["manifest_json"])

        print()
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
