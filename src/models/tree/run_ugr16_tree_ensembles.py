#!/usr/bin/env python3
"""
Phase C — TREE-MARCH-SCREEN-001
================================

Cribado formal de Random Forest, XGBoost y LightGBM sobre UGR'16 March
Week #3, estrictamente limitado a la captura March.

Protocolo de autoridad:
    docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md

Diseño congelado:
- target: bitrate_bps
- resolución: 5 minutos
- horizontes: H1/H3/H6/H12
- 23 features: current_value + 16 lags + 6 temporales
- TimeSeriesSplit(n_splits=3, gap=12)
- 6 candidatos por familia
- semilla canónica: 20260818
- una configuración común por familia para los cuatro horizontes
- Score = media_h(MAE_modelo_h / MAE_persistencia_h)
- top 3 candidatos por familia
- sin clipping de predicciones
- sin April
- sin June
- CPU, n_jobs=2, concurrencia de candidatos=1
- timeout duro por fit: 600 s

Este runner NO contiene rutas de April ni June y no puede abrir esas capturas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


CAMPAIGN_ID = "TREE-MARCH-SCREEN-001"
PARENT_CAMPAIGN_ID = "UGR16-TREE-ENSEMBLES-001"

DEFAULT_MARCH_SERIES = Path(
    "data/processed/ugr16/march_week3_prepared_5min.parquet"
)
DEFAULT_PROTOCOL = Path(
    "docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md"
)
DEFAULT_ENVIRONMENT_PREFLIGHT = Path(
    "results/metrics/tree_ensembles/"
    "tree_environment_preflight_20260819.txt"
)
DEFAULT_METRICS_DIR = Path("results/metrics/tree_ensembles")
DEFAULT_PREDICTIONS_DIR = Path("results/predictions/tree_ensembles")
DEFAULT_PREFIX = "ugr16_tree_march_screen"

EXPECTED_PROTOCOL_SHA256 = (
    "8c0a94e4ee80a84b78bf077d6abc3c18617ad1736b6d978047d0db1d52055003"
)

TARGET = "bitrate_bps"
INTERVAL_MINUTES = 5
HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 72, 144, 288]
CV_SPLITS = 3
CV_GAP = 12
CANONICAL_SEED = 20260818
N_JOBS = 2
FIT_TIMEOUT_SECONDS = 600

EXPECTED_MARCH_ROWS = 732
EXPECTED_MARCH_START = pd.Timestamp("2016-03-18 11:00:00")
EXPECTED_MARCH_END = pd.Timestamp("2016-03-20 23:55:00")

EXPECTED_FEATURE_COLUMNS = [
    "current_value",
    *[f"lag_{lag}" for lag in LAGS],
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
]

RF_GRID: list[dict[str, Any]] = [
    {
        "candidate_id": "RF01",
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "max_features": 0.7,
    },
    {
        "candidate_id": "RF02",
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "candidate_id": "RF03",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "max_features": 0.7,
    },
    {
        "candidate_id": "RF04",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "max_features": 1.0,
    },
    {
        "candidate_id": "RF05",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "candidate_id": "RF06",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 1.0,
    },
]

XGB_GRID: list[dict[str, Any]] = [
    {
        "candidate_id": "XGB01",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "XGB02",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "XGB03",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "XGB04",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2,
    },
    {
        "candidate_id": "XGB05",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
    },
    {
        "candidate_id": "XGB06",
        "n_estimators": 300,
        "learning_rate": 0.10,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
    },
]

LGB_GRID: list[dict[str, Any]] = [
    {
        "candidate_id": "LGB01",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 30,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "LGB02",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "LGB03",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1,
    },
    {
        "candidate_id": "LGB04",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2,
    },
    {
        "candidate_id": "LGB05",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
    },
    {
        "candidate_id": "LGB06",
        "n_estimators": 300,
        "learning_rate": 0.10,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
    },
]

GRIDS: dict[str, list[dict[str, Any]]] = {
    "random_forest": RF_GRID,
    "xgboost": XGB_GRID,
    "lightgbm": LGB_GRID,
}


@dataclass(frozen=True)
class OutputPaths:
    grid_csv: Path
    fold_metrics_csv: Path
    candidate_horizon_metrics_csv: Path
    candidate_scores_csv: Path
    top3_csv: Path
    runtime_csv: Path
    predictions_parquet: Path
    report_txt: Path
    manifest_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta TREE-MARCH-SCREEN-001 exclusivamente sobre "
            "UGR'16 March Week #3."
        )
    )
    parser.add_argument(
        "--march-series",
        type=Path,
        default=DEFAULT_MARCH_SERIES,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--environment-preflight",
        type=Path,
        default=DEFAULT_ENVIRONMENT_PREFLIGHT,
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Valida protocolo, entorno, March, features y folds sin entrenar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas previas de esta subcampaña.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_sidecar(path: Path) -> tuple[bool, str | None]:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return False, None
    text = sidecar.read_text(encoding="utf-8").strip()
    if not text:
        return False, None
    expected = text.split()[0].strip()
    actual = sha256_file(path)
    return expected == actual, actual


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def output_paths(args: argparse.Namespace) -> OutputPaths:
    prefix = args.prefix
    return OutputPaths(
        grid_csv=args.metrics_dir / f"{prefix}_grid.csv",
        fold_metrics_csv=args.metrics_dir / f"{prefix}_fold_metrics.csv",
        candidate_horizon_metrics_csv=(
            args.metrics_dir / f"{prefix}_candidate_horizon_metrics.csv"
        ),
        candidate_scores_csv=(
            args.metrics_dir / f"{prefix}_candidate_scores.csv"
        ),
        top3_csv=args.metrics_dir / f"{prefix}_top3.csv",
        runtime_csv=args.metrics_dir / f"{prefix}_runtime.csv",
        predictions_parquet=(
            args.predictions_dir / f"{prefix}_predictions.parquet"
        ),
        report_txt=args.metrics_dir / f"{prefix}_report.txt",
        manifest_json=args.metrics_dir / f"{prefix}_manifest.json",
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def validate_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el protocolo: {path}")
    actual = sha256_file(path)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "El SHA-256 de 017 no coincide con el protocolo congelado.\n"
            f"Esperado: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual:   {actual}"
        )
    text = path.read_text(encoding="utf-8")
    required_tokens = [
        "TFM-TREE-PROTOCOL-001",
        "Status = FROZEN",
        "TimeSeriesSplit",
        "gap = 12",
        "20260818",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    ]
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise RuntimeError(
            "El protocolo 017 no contiene tokens esperados: "
            + ", ".join(missing)
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "status": "PASS",
    }


def validate_environment_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el preflight de entorno: {path}"
        )
    text = path.read_text(encoding="utf-8")
    required = [
        "All installed packages are compatible",
        "RandomForest PASS",
        "XGBoost      PASS",
        "LightGBM     PASS",
        "TREE ENSEMBLES SMOKE TEST: PASS",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(
            "El preflight no contiene todos los PASS requeridos: "
            + ", ".join(missing)
        )
    sidecar_ok, actual = verify_sha256_sidecar(path)
    if not sidecar_ok:
        raise RuntimeError(
            "El sidecar SHA-256 del preflight no existe o no valida."
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "sidecar_valid": True,
        "status": "PASS",
    }


def load_march_series(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe March 5min: {path}")

    frame = pd.read_parquet(path)
    required = {"timestamp", TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Faltan columnas requeridas en March: {sorted(missing)}"
        )

    result = frame[["timestamp", TARGET]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result[TARGET] = pd.to_numeric(result[TARGET], errors="raise")
    result = result.reset_index(drop=True)

    if len(result) != EXPECTED_MARCH_ROWS:
        raise ValueError(
            f"March debe contener {EXPECTED_MARCH_ROWS} filas; "
            f"obtenidas={len(result)}."
        )
    if result["timestamp"].iloc[0] != EXPECTED_MARCH_START:
        raise ValueError(
            "Inicio March inesperado: "
            f"{result['timestamp'].iloc[0]}"
        )
    if result["timestamp"].iloc[-1] != EXPECTED_MARCH_END:
        raise ValueError(
            "Fin March inesperado: "
            f"{result['timestamp'].iloc[-1]}"
        )
    if result["timestamp"].duplicated().any():
        raise ValueError("March contiene timestamps duplicados.")
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("March contiene timestamps desordenados.")

    delta = result["timestamp"].diff().dropna()
    if not bool(delta.eq(pd.Timedelta(minutes=INTERVAL_MINUTES)).all()):
        raise ValueError("March no tiene cadencia exacta de 5 minutos.")

    values = result[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("March contiene target no finito.")
    if np.any(values < 0):
        raise ValueError("March contiene target negativo.")

    summary = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(result),
        "start": result["timestamp"].iloc[0].isoformat(),
        "end": result["timestamp"].iloc[-1].isoformat(),
        "interval_minutes": INTERVAL_MINUTES,
        "target": TARGET,
    }
    return result, summary


def make_feature_frame(series: pd.DataFrame) -> pd.DataFrame:
    timestamps = series["timestamp"]
    y = series[TARGET].astype(float)

    features = pd.DataFrame(index=series.index)
    features["current_value"] = y

    for lag in LAGS:
        features[f"lag_{lag}"] = y.shift(lag)

    minute_of_day = (
        timestamps.dt.hour * 60 + timestamps.dt.minute
    ).astype(float)
    day_of_week = timestamps.dt.dayofweek.astype(float)

    features["hour_sin"] = np.sin(
        2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    )
    features["hour_cos"] = np.cos(
        2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    )
    features["dow_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 7.0
    )
    features["dow_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 7.0
    )
    features["is_weekend"] = (
        timestamps.dt.dayofweek >= 5
    ).astype(float)
    features["trend"] = np.arange(len(features), dtype=float)

    if list(features.columns) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError(
            "Las columnas de features no coinciden exactamente con 017."
        )
    if len(features.columns) != 23:
        raise RuntimeError(
            f"Se esperaban 23 features; obtenidas={len(features.columns)}."
        )

    return features


def supervised_rows(
    features: pd.DataFrame,
    series: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    result = features.copy()
    result["origin_index"] = np.arange(len(series), dtype=int)
    result["target_index"] = result["origin_index"] + horizon
    result["origin_timestamp"] = series["timestamp"]
    result["target_timestamp"] = series["timestamp"].shift(-horizon)
    result["target_value"] = series[TARGET].shift(-horizon)
    result["persistence_value"] = series[TARGET]
    result = result.dropna().reset_index(drop=True)

    result["origin_index"] = result["origin_index"].astype(int)
    result["target_index"] = result["target_index"].astype(int)

    if not (
        result["target_index"] - result["origin_index"]
    ).eq(horizon).all():
        raise RuntimeError(
            f"H{horizon}: desalineación origin/target."
        )

    expected_delta = pd.to_timedelta(
        HORIZON_MINUTES[horizon], unit="min"
    )
    actual_delta = (
        result["target_timestamp"] - result["origin_timestamp"]
    )
    if not bool(actual_delta.eq(expected_delta).all()):
        raise RuntimeError(
            f"H{horizon}: timestamps objetivo incorrectos."
        )

    matrix = result[EXPECTED_FEATURE_COLUMNS].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise RuntimeError(
            f"H{horizon}: matriz supervisada con NaN/inf."
        )
    if not np.isfinite(
        result[["target_value", "persistence_value"]].to_numpy(float)
    ).all():
        raise RuntimeError(
            f"H{horizon}: target/persistencia no finitos."
        )

    return result


def build_grid_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, grid in GRIDS.items():
        for parameters in grid:
            row = {"family": family, **parameters}
            if family == "random_forest":
                row.update(
                    {
                        "criterion": "absolute_error",
                        "bootstrap": True,
                        "n_jobs": N_JOBS,
                    }
                )
            elif family == "xgboost":
                row.update(
                    {
                        "booster": "gbtree",
                        "tree_method": "hist",
                        "objective": "reg:absoluteerror",
                        "gamma": 0,
                        "reg_alpha": 0,
                        "n_jobs": N_JOBS,
                        "early_stopping": False,
                    }
                )
            elif family == "lightgbm":
                row.update(
                    {
                        "boosting_type": "gbdt",
                        "objective": "regression_l1",
                        "reg_alpha": 0,
                        "n_jobs": N_JOBS,
                        "subsample_freq": (
                            1 if parameters["subsample"] < 1.0 else 0
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def validate_grid() -> None:
    if set(GRIDS) != {"random_forest", "xgboost", "lightgbm"}:
        raise RuntimeError("Familias de grid inesperadas.")
    if any(len(grid) != 6 for grid in GRIDS.values()):
        raise RuntimeError("Cada familia debe tener exactamente 6 candidatos.")

    ids = [
        item["candidate_id"]
        for grid in GRIDS.values()
        for item in grid
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Hay candidate_id duplicados.")


def build_model(
    family: str,
    parameters: dict[str, Any],
    seed: int,
) -> Any:
    params = {
        key: value
        for key, value in parameters.items()
        if key != "candidate_id"
    }

    if family == "random_forest":
        return RandomForestRegressor(
            criterion="absolute_error",
            bootstrap=True,
            random_state=seed,
            n_jobs=N_JOBS,
            **params,
        )

    if family == "xgboost":
        return XGBRegressor(
            booster="gbtree",
            tree_method="hist",
            objective="reg:absoluteerror",
            gamma=0,
            reg_alpha=0,
            random_state=seed,
            n_jobs=N_JOBS,
            verbosity=0,
            **params,
        )

    if family == "lightgbm":
        subsample = float(params["subsample"])
        return LGBMRegressor(
            boosting_type="gbdt",
            objective="regression_l1",
            reg_alpha=0,
            random_state=seed,
            n_jobs=N_JOBS,
            verbosity=-1,
            subsample_freq=1 if subsample < 1.0 else 0,
            **params,
        )

    raise ValueError(f"Familia no soportada: {family}")


def _fit_worker(
    connection: Any,
    family: str,
    parameters: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
) -> None:
    try:
        model = build_model(family, parameters, seed)

        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        predictions = np.asarray(
            model.predict(X_validation),
            dtype=float,
        )
        predict_seconds = time.perf_counter() - predict_start

        serialized_model_bytes = len(
            pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        )

        connection.send(
            {
                "status": "PASS",
                "predictions": predictions,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "serialized_model_bytes": serialized_model_bytes,
                "negative_predictions": int(np.sum(predictions < 0)),
            }
        )
    except BaseException as error:
        connection.send(
            {
                "status": "ERROR",
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def fit_with_hard_timeout(
    family: str,
    parameters: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    timeout_seconds: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)

    process = context.Process(
        target=_fit_worker,
        args=(
            child_connection,
            family,
            parameters,
            seed,
            X_train,
            y_train,
            X_validation,
        ),
    )
    process.start()
    child_connection.close()

    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        parent_connection.close()
        return {
            "status": "RESOURCE_LIMIT",
            "timeout_seconds": timeout_seconds,
        }

    if not parent_connection.poll(2):
        exitcode = process.exitcode
        parent_connection.close()
        return {
            "status": "ERROR",
            "error_type": "WorkerNoResult",
            "error_message": (
                "El proceso terminó sin devolver resultado. "
                f"exitcode={exitcode}"
            ),
        }

    result = parent_connection.recv()
    parent_connection.close()
    return result


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    values = np.divide(
        2.0 * np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(y_true, dtype=float),
        where=denominator != 0,
    )
    return float(100.0 * np.mean(values))


def fold_mase_scale(
    series: pd.DataFrame,
    supervised: pd.DataFrame,
    train_indices: np.ndarray,
) -> float:
    last_train_target_index = int(
        supervised.iloc[train_indices]["target_index"].max()
    )
    history = series.loc[
        :last_train_target_index,
        TARGET,
    ].to_numpy(float)
    scale = float(np.mean(np.abs(np.diff(history))))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Escala MASE de fold no válida.")
    return scale


def validate_folds(
    supervised_by_horizon: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        supervised = supervised_by_horizon[horizon]
        splitter = TimeSeriesSplit(
            n_splits=CV_SPLITS,
            gap=CV_GAP,
        )

        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(supervised),
            start=1,
        ):
            if len(train_idx) == 0 or len(valid_idx) == 0:
                raise RuntimeError(
                    f"H{horizon} fold {fold}: train/validation vacío."
                )

            train = supervised.iloc[train_idx]
            validation = supervised.iloc[valid_idx]

            max_train_target = int(train["target_index"].max())
            min_validation_origin = int(
                validation["origin_index"].min()
            )

            if max_train_target >= min_validation_origin:
                raise RuntimeError(
                    f"H{horizon} fold {fold}: leakage temporal; "
                    f"max_train_target={max_train_target}, "
                    f"min_validation_origin={min_validation_origin}."
                )

            rows.append(
                {
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[horizon],
                    "fold": fold,
                    "supervised_rows": len(supervised),
                    "train_rows": len(train_idx),
                    "validation_rows": len(valid_idx),
                    "train_first_origin_index": int(
                        train["origin_index"].min()
                    ),
                    "train_last_origin_index": int(
                        train["origin_index"].max()
                    ),
                    "train_last_target_index": max_train_target,
                    "validation_first_origin_index": (
                        min_validation_origin
                    ),
                    "validation_last_origin_index": int(
                        validation["origin_index"].max()
                    ),
                    "validation_first_target_timestamp": (
                        validation["target_timestamp"].iloc[0]
                    ),
                    "validation_last_target_timestamp": (
                        validation["target_timestamp"].iloc[-1]
                    ),
                    "gap_steps": CV_GAP,
                    "leakage_check": True,
                }
            )

    return pd.DataFrame(rows)


def preflight(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[int, pd.DataFrame],
    pd.DataFrame,
]:
    validate_grid()

    protocol = validate_protocol(args.protocol)
    environment = validate_environment_preflight(
        args.environment_preflight
    )
    march, march_summary = load_march_series(args.march_series)
    features = make_feature_frame(march)

    supervised_by_horizon = {
        horizon: supervised_rows(features, march, horizon)
        for horizon in HORIZONS
    }

    folds = validate_folds(supervised_by_horizon)
    grid = build_grid_frame()

    if set(grid["family"]) != set(GRIDS):
        raise RuntimeError("Grid frame incompleto.")

    print("=" * 96)
    print("TREE-MARCH-SCREEN-001 — PREFLIGHT")
    print("=" * 96)
    print(f"Protocolo 017:               PASS | {protocol['sha256']}")
    print(
        "Environment preflight:      "
        f"PASS | {environment['sha256']}"
    )
    print(
        f"March:                      PASS | "
        f"{march_summary['rows']} filas | "
        f"{march_summary['start']} → {march_summary['end']}"
    )
    print(
        f"Features:                   PASS | "
        f"{len(EXPECTED_FEATURE_COLUMNS)}"
    )
    print(
        f"Lags:                       PASS | "
        f"{','.join(str(value) for value in LAGS)}"
    )
    print(
        f"Horizontes:                 PASS | "
        f"{','.join('H'+str(value) for value in HORIZONS)}"
    )
    print(
        f"CV:                         PASS | "
        f"TimeSeriesSplit({CV_SPLITS}, gap={CV_GAP})"
    )
    print("Leakage folds:              PASS")
    print(
        f"Grid:                       PASS | "
        f"{len(grid)} candidatos totales"
    )
    print(f"Seed canónica:              {CANONICAL_SEED}")
    print(f"n_jobs:                     {N_JOBS}")
    print(f"Timeout por fit:            {FIT_TIMEOUT_SECONDS} s")
    print("April accessible by runner: NO")
    print("June accessible by runner:  NO")
    print()
    print("Filas supervisadas:")
    for horizon in HORIZONS:
        print(
            f"  H{horizon:<2} "
            f"({HORIZON_MINUTES[horizon]:>2} min): "
            f"{len(supervised_by_horizon[horizon]):,}"
        )
    print()
    print("TREE-MARCH-SCREEN PREFLIGHT: PASS")

    return (
        march,
        {
            "protocol": protocol,
            "environment": environment,
            "march": march_summary,
        },
        grid,
        supervised_by_horizon,
        folds,
    )


def evaluate_screen(
    march: pd.DataFrame,
    grid: pd.DataFrame,
    supervised_by_horizon: dict[int, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    total_fits = len(grid) * len(HORIZONS) * CV_SPLITS
    completed_fits = 0

    for family in ["random_forest", "xgboost", "lightgbm"]:
        family_grid = GRIDS[family]

        print()
        print("=" * 96)
        print(f"FAMILIA: {family}")
        print("=" * 96)

        for parameters in family_grid:
            candidate_id = str(parameters["candidate_id"])
            print(f"\n{candidate_id}")

            candidate_resource_limited = False

            for horizon in HORIZONS:
                supervised = supervised_by_horizon[horizon]
                splitter = TimeSeriesSplit(
                    n_splits=CV_SPLITS,
                    gap=CV_GAP,
                )

                for fold, (train_idx, valid_idx) in enumerate(
                    splitter.split(supervised),
                    start=1,
                ):
                    completed_fits += 1
                    print(
                        f"  H{horizon:<2} fold {fold} "
                        f"[{completed_fits}/{total_fits}] ... ",
                        end="",
                        flush=True,
                    )

                    X_train = supervised.iloc[
                        train_idx
                    ][EXPECTED_FEATURE_COLUMNS].to_numpy(float)
                    y_train = supervised.iloc[
                        train_idx
                    ]["target_value"].to_numpy(float)
                    X_valid = supervised.iloc[
                        valid_idx
                    ][EXPECTED_FEATURE_COLUMNS].to_numpy(float)
                    y_valid = supervised.iloc[
                        valid_idx
                    ]["target_value"].to_numpy(float)
                    persistence = supervised.iloc[
                        valid_idx
                    ]["persistence_value"].to_numpy(float)

                    scale = fold_mase_scale(
                        march,
                        supervised,
                        train_idx,
                    )

                    result = fit_with_hard_timeout(
                        family=family,
                        parameters=parameters,
                        seed=CANONICAL_SEED,
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_valid,
                        timeout_seconds=FIT_TIMEOUT_SECONDS,
                    )

                    status = result["status"]

                    runtime_row = {
                        "family": family,
                        "candidate_id": candidate_id,
                        "horizon_steps": horizon,
                        "horizon_minutes": HORIZON_MINUTES[horizon],
                        "fold": fold,
                        "seed": CANONICAL_SEED,
                        "status": status,
                        "fit_seconds": np.nan,
                        "predict_seconds": np.nan,
                        "serialized_model_bytes": np.nan,
                        "negative_predictions": np.nan,
                        "timeout_seconds": FIT_TIMEOUT_SECONDS,
                    }

                    if status == "RESOURCE_LIMIT":
                        candidate_resource_limited = True
                        runtime_rows.append(runtime_row)
                        print("RESOURCE_LIMIT")
                        continue

                    if status != "PASS":
                        raise RuntimeError(
                            f"{candidate_id} H{horizon} fold {fold}: "
                            f"{result.get('error_type')}: "
                            f"{result.get('error_message')}\n"
                            f"{result.get('traceback', '')}"
                        )

                    model_predictions = np.asarray(
                        result["predictions"],
                        dtype=float,
                    )

                    if len(model_predictions) != len(valid_idx):
                        raise RuntimeError(
                            f"{candidate_id} H{horizon} fold {fold}: "
                            "número de predicciones incorrecto."
                        )
                    if not np.isfinite(model_predictions).all():
                        raise RuntimeError(
                            f"{candidate_id} H{horizon} fold {fold}: "
                            "predicciones no finitas."
                        )

                    runtime_row.update(
                        {
                            "fit_seconds": float(
                                result["fit_seconds"]
                            ),
                            "predict_seconds": float(
                                result["predict_seconds"]
                            ),
                            "serialized_model_bytes": int(
                                result["serialized_model_bytes"]
                            ),
                            "negative_predictions": int(
                                result["negative_predictions"]
                            ),
                        }
                    )
                    runtime_rows.append(runtime_row)

                    errors = y_valid - model_predictions
                    persistence_errors = y_valid - persistence

                    fold_metric_rows.append(
                        {
                            "family": family,
                            "candidate_id": candidate_id,
                            "horizon_steps": horizon,
                            "horizon_minutes": HORIZON_MINUTES[
                                horizon
                            ],
                            "fold": fold,
                            "seed": CANONICAL_SEED,
                            "status": "PASS",
                            "n_predictions": len(y_valid),
                            "model_mae_bps": float(
                                np.mean(np.abs(errors))
                            ),
                            "persistence_mae_bps": float(
                                np.mean(
                                    np.abs(persistence_errors)
                                )
                            ),
                            "mae_ratio_vs_persistence": float(
                                np.mean(np.abs(errors))
                                / np.mean(
                                    np.abs(persistence_errors)
                                )
                            ),
                            "model_rmse_bps": float(
                                np.sqrt(np.mean(errors ** 2))
                            ),
                            "persistence_rmse_bps": float(
                                np.sqrt(
                                    np.mean(
                                        persistence_errors ** 2
                                    )
                                )
                            ),
                            "model_smape_pct": smape(
                                y_valid,
                                model_predictions,
                            ),
                            "persistence_smape_pct": smape(
                                y_valid,
                                persistence,
                            ),
                            "model_mase": float(
                                np.mean(np.abs(errors) / scale)
                            ),
                            "persistence_mase": float(
                                np.mean(
                                    np.abs(persistence_errors)
                                    / scale
                                )
                            ),
                            "model_bias_bps": float(
                                np.mean(errors)
                            ),
                            "model_underprediction_pct": float(
                                100.0 * np.mean(errors > 0)
                            ),
                            "model_p95_absolute_error_bps": float(
                                np.quantile(
                                    np.abs(errors),
                                    0.95,
                                )
                            ),
                            "coverage": 1.0,
                            "mase_scale_bps": scale,
                        }
                    )

                    validation = supervised.iloc[
                        valid_idx
                    ].reset_index(drop=True)

                    for position in range(len(validation)):
                        prediction_rows.append(
                            {
                                "capture_key": "march",
                                "capture_name": (
                                    "UGR'16 March Week #3"
                                ),
                                "evaluation_role": "screening",
                                "family": family,
                                "candidate_id": candidate_id,
                                "seed": CANONICAL_SEED,
                                "horizon_steps": horizon,
                                "horizon_minutes": HORIZON_MINUTES[
                                    horizon
                                ],
                                "fold": fold,
                                "origin_index": int(
                                    validation.at[
                                        position,
                                        "origin_index",
                                    ]
                                ),
                                "target_index": int(
                                    validation.at[
                                        position,
                                        "target_index",
                                    ]
                                ),
                                "origin_timestamp": (
                                    validation.at[
                                        position,
                                        "origin_timestamp",
                                    ]
                                ),
                                "target_timestamp": (
                                    validation.at[
                                        position,
                                        "target_timestamp",
                                    ]
                                ),
                                "y_true_bps": float(
                                    y_valid[position]
                                ),
                                "y_pred_bps": float(
                                    model_predictions[position]
                                ),
                                "persistence_pred_bps": float(
                                    persistence[position]
                                ),
                                "error_bps": float(
                                    errors[position]
                                ),
                                "absolute_error_bps": float(
                                    abs(errors[position])
                                ),
                                "persistence_error_bps": float(
                                    persistence_errors[position]
                                ),
                                "persistence_absolute_error_bps": (
                                    float(
                                        abs(
                                            persistence_errors[
                                                position
                                            ]
                                        )
                                    )
                                ),
                            }
                        )

                    print(
                        "PASS | "
                        f"MAE={np.mean(np.abs(errors))/1e6:.3f} "
                        "Mbit/s"
                    )

            if candidate_resource_limited:
                print(
                    f"  {candidate_id}: candidato con al menos un "
                    "RESOURCE_LIMIT; quedará sin Score válido."
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    runtime = pd.DataFrame(runtime_rows)

    if predictions.empty:
        raise RuntimeError("No se generaron predicciones válidas.")
    if fold_metrics.empty:
        raise RuntimeError("No se generaron métricas de folds.")

    candidate_horizon_metrics = summarize_candidate_horizons(
        predictions
    )
    candidate_scores = summarize_scores(
        candidate_horizon_metrics,
        runtime,
    )
    top3 = select_top3(candidate_scores)
    grid_output = grid.copy()

    return (
        grid_output,
        fold_metrics,
        candidate_horizon_metrics,
        candidate_scores,
        top3,
        runtime,
        predictions,
    )


def summarize_candidate_horizons(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        family,
        candidate_id,
        horizon,
    ), group in predictions.groupby(
        ["family", "candidate_id", "horizon_steps"],
        sort=True,
    ):
        y_true = group["y_true_bps"].to_numpy(float)
        y_pred = group["y_pred_bps"].to_numpy(float)
        persistence = group[
            "persistence_pred_bps"
        ].to_numpy(float)

        errors = y_true - y_pred
        persistence_errors = y_true - persistence

        model_mae = float(np.mean(np.abs(errors)))
        persistence_mae = float(
            np.mean(np.abs(persistence_errors))
        )

        expected_rows = len(group)
        if expected_rows <= 0:
            raise RuntimeError(
                f"{candidate_id} H{horizon}: cobertura vacía."
            )

        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "horizon_steps": int(horizon),
                "horizon_minutes": HORIZON_MINUTES[int(horizon)],
                "seed": CANONICAL_SEED,
                "n_predictions": expected_rows,
                "model_mae_bps": model_mae,
                "model_mae_mbps": model_mae / 1e6,
                "persistence_mae_bps": persistence_mae,
                "persistence_mae_mbps": persistence_mae / 1e6,
                "mae_ratio_vs_persistence": (
                    model_mae / persistence_mae
                ),
                "skill_vs_persistence_pct": (
                    100.0
                    * (persistence_mae - model_mae)
                    / persistence_mae
                ),
                "model_rmse_bps": float(
                    np.sqrt(np.mean(errors ** 2))
                ),
                "model_rmse_mbps": float(
                    np.sqrt(np.mean(errors ** 2)) / 1e6
                ),
                "model_smape_pct": smape(y_true, y_pred),
                "model_bias_bps": float(np.mean(errors)),
                "model_underprediction_pct": float(
                    100.0 * np.mean(errors > 0)
                ),
                "model_p95_absolute_error_bps": float(
                    np.quantile(np.abs(errors), 0.95)
                ),
                "negative_predictions": int(
                    np.sum(y_pred < 0)
                ),
                "coverage": 1.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["family", "candidate_id", "horizon_steps"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_scores(
    candidate_horizon_metrics: pd.DataFrame,
    runtime: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    all_candidate_ids = [
        (family, item["candidate_id"])
        for family, grid in GRIDS.items()
        for item in grid
    ]

    for family, candidate_id in all_candidate_ids:
        metrics = candidate_horizon_metrics[
            candidate_horizon_metrics["family"].eq(family)
            & candidate_horizon_metrics[
                "candidate_id"
            ].eq(candidate_id)
        ].copy()

        runtime_candidate = runtime[
            runtime["family"].eq(family)
            & runtime["candidate_id"].eq(candidate_id)
        ].copy()

        resource_limited = bool(
            runtime_candidate["status"]
            .eq("RESOURCE_LIMIT")
            .any()
        )

        complete = (
            set(metrics["horizon_steps"]) == set(HORIZONS)
            and len(metrics) == len(HORIZONS)
            and not resource_limited
        )

        row: dict[str, Any] = {
            "family": family,
            "candidate_id": candidate_id,
            "seed": CANONICAL_SEED,
            "status": "PASS" if complete else (
                "RESOURCE_LIMIT"
                if resource_limited
                else "INCOMPLETE"
            ),
            "score": np.nan,
            "total_fit_seconds": float(
                runtime_candidate["fit_seconds"].sum(
                    skipna=True
                )
            ),
            "total_predict_seconds": float(
                runtime_candidate["predict_seconds"].sum(
                    skipna=True
                )
            ),
            "mean_serialized_model_bytes": float(
                runtime_candidate[
                    "serialized_model_bytes"
                ].mean(skipna=True)
            ),
            "max_serialized_model_bytes": float(
                runtime_candidate[
                    "serialized_model_bytes"
                ].max(skipna=True)
            ),
            "negative_predictions": int(
                runtime_candidate[
                    "negative_predictions"
                ].sum(skipna=True)
            ),
        }

        for horizon in HORIZONS:
            subset = metrics[
                metrics["horizon_steps"].eq(horizon)
            ]
            row[f"h{horizon}_mae_ratio"] = (
                float(
                    subset.iloc[0][
                        "mae_ratio_vs_persistence"
                    ]
                )
                if len(subset) == 1
                else np.nan
            )

        if complete:
            ratios = [
                row[f"h{horizon}_mae_ratio"]
                for horizon in HORIZONS
            ]
            row["score"] = float(np.mean(ratios))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["family", "score", "candidate_id"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def select_top3(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for family in ["random_forest", "xgboost", "lightgbm"]:
        valid = candidate_scores[
            candidate_scores["family"].eq(family)
            & candidate_scores["status"].eq("PASS")
            & candidate_scores["score"].notna()
        ].sort_values(
            ["score", "candidate_id"],
            kind="stable",
        )

        if len(valid) < 3:
            raise RuntimeError(
                f"{family}: solo {len(valid)} candidatos válidos; "
                "se requieren al menos 3 para continuar a April."
            )

        selected = valid.head(3).copy()
        selected.insert(
            2,
            "march_rank",
            np.arange(1, len(selected) + 1),
        )
        rows.append(selected)

    top3 = pd.concat(rows, ignore_index=True)
    if len(top3) != 9:
        raise RuntimeError(
            f"Top3 debe contener 9 filas; obtenidas={len(top3)}."
        )
    return top3


def validate_outputs(
    fold_metrics: pd.DataFrame,
    candidate_horizon_metrics: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    top3: pd.DataFrame,
    runtime: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, bool]:
    valid_candidates = candidate_scores[
        candidate_scores["status"].eq("PASS")
    ]

    return {
        "fold_metrics_nonempty": not fold_metrics.empty,
        "candidate_horizon_metrics_nonempty": (
            not candidate_horizon_metrics.empty
        ),
        "candidate_scores_18_rows": len(candidate_scores) == 18,
        "top3_9_rows": len(top3) == 9,
        "top3_three_per_family": bool(
            top3.groupby("family").size().eq(3).all()
        ),
        "runtime_nonempty": not runtime.empty,
        "predictions_nonempty": not predictions.empty,
        "only_march": set(predictions["capture_key"]) == {"march"},
        "all_horizons_for_valid_candidates": all(
            set(
                candidate_horizon_metrics[
                    candidate_horizon_metrics[
                        "candidate_id"
                    ].eq(candidate_id)
                ]["horizon_steps"]
            )
            == set(HORIZONS)
            for candidate_id in valid_candidates[
                "candidate_id"
            ]
        ),
        "future_targets": bool(
            (
                predictions["target_timestamp"]
                > predictions["origin_timestamp"]
            ).all()
        ),
        "index_horizon": bool(
            (
                predictions["target_index"]
                - predictions["origin_index"]
            ).eq(predictions["horizon_steps"]).all()
        ),
        "finite_predictions": bool(
            np.isfinite(predictions["y_pred_bps"]).all()
        ),
        "finite_targets": bool(
            np.isfinite(predictions["y_true_bps"]).all()
        ),
        "coverage_one": bool(
            candidate_horizon_metrics["coverage"].eq(1.0).all()
        ),
        "scores_finite_for_pass": bool(
            np.isfinite(
                valid_candidates["score"].to_numpy(float)
            ).all()
        ),
        "canonical_seed_only": set(
            predictions["seed"]
        ) == {CANONICAL_SEED},
    }


def build_report(
    input_records: dict[str, Any],
    folds: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    top3: pd.DataFrame,
    runtime: pd.DataFrame,
    validations: dict[str, bool],
    processing_seconds: float,
    paths: OutputPaths,
) -> str:
    lines = [
        "=" * 104,
        "TREE-MARCH-SCREEN-001 — PHASE C TREE ENSEMBLES",
        "=" * 104,
        f"Campaña marco:                  {PARENT_CAMPAIGN_ID}",
        f"Subcampaña:                     {CAMPAIGN_ID}",
        "Estado:                         PASS",
        "",
        "PROTOCOLO",
        "-" * 104,
        f"017:                            {input_records['protocol']['path']}",
        f"SHA-256 017:                    {input_records['protocol']['sha256']}",
        f"Preflight entorno:              {input_records['environment']['path']}",
        f"SHA-256 preflight:              {input_records['environment']['sha256']}",
        "",
        "MARCH",
        "-" * 104,
        f"Entrada:                        {input_records['march']['path']}",
        f"SHA-256 entrada:                {input_records['march']['sha256']}",
        f"Filas:                          {input_records['march']['rows']}",
        f"Inicio:                         {input_records['march']['start']}",
        f"Fin:                            {input_records['march']['end']}",
        f"Target:                         {TARGET}",
        f"Resolución:                     {INTERVAL_MINUTES} min",
        "",
        "DISEÑO CONGELADO",
        "-" * 104,
        f"Features:                       {len(EXPECTED_FEATURE_COLUMNS)}",
        f"Lags:                           {LAGS}",
        f"Horizontes:                     {HORIZONS}",
        f"CV:                             TimeSeriesSplit(n_splits={CV_SPLITS}, gap={CV_GAP})",
        f"Semilla:                        {CANONICAL_SEED}",
        f"n_jobs:                         {N_JOBS}",
        f"Timeout por fit:                {FIT_TIMEOUT_SECONDS} s",
        "Clipping:                       NO",
        "April:                          NO ACCESIBLE EN ESTE RUNNER",
        "June:                           NO ACCESIBLE EN ESTE RUNNER",
        "",
        "FOLDS",
        "-" * 104,
        folds[
            [
                "horizon_steps",
                "fold",
                "train_rows",
                "validation_rows",
                "train_last_target_index",
                "validation_first_origin_index",
                "leakage_check",
            ]
        ].to_string(index=False),
        "",
        "SCORE POR CANDIDATO",
        "-" * 104,
    ]

    display_columns = [
        "family",
        "candidate_id",
        "status",
        "score",
        "h1_mae_ratio",
        "h3_mae_ratio",
        "h6_mae_ratio",
        "h12_mae_ratio",
        "total_fit_seconds",
    ]
    lines.append(
        candidate_scores[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    lines.extend(
        [
            "",
            "TOP 3 MARCH POR FAMILIA",
            "-" * 104,
            top3[
                [
                    "family",
                    "march_rank",
                    "candidate_id",
                    "score",
                    "total_fit_seconds",
                    "mean_serialized_model_bytes",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "VALIDACIONES",
            "-" * 104,
        ]
    )

    for name, value in validations.items():
        lines.append(
            f"{name:40s}: {'PASS' if value else 'FAIL'}"
        )

    resource_limits = int(
        runtime["status"].eq("RESOURCE_LIMIT").sum()
    )
    fit_errors = int(runtime["status"].eq("ERROR").sum())

    lines.extend(
        [
            "",
            "RECURSOS",
            "-" * 104,
            f"Fits registrados:               {len(runtime)}",
            f"RESOURCE_LIMIT:                 {resource_limits}",
            f"ERROR:                          {fit_errors}",
            f"Tiempo total de campaña:        {processing_seconds:.3f} s",
            "",
            "SALIDAS",
            "-" * 104,
        ]
    )

    for field in paths.__dataclass_fields__:
        value = getattr(paths, field)
        lines.append(f"{field}: {value.resolve()}")

    lines.extend(
        [
            "",
            "DECISIÓN",
            "-" * 104,
            "March screening completado.",
            "Los top 3 por familia quedan identificados para la futura subcampaña April.",
            "Este resultado NO selecciona todavía RF*, XGB*, LGBM* ni TREE*.",
            "TREE-JUNE-BLIND-001 permanece NO AUTORIZADO.",
            "",
        ]
    )

    return "\n".join(lines)


def save_outputs(
    paths: OutputPaths,
    grid: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    candidate_horizon_metrics: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    top3: pd.DataFrame,
    runtime: pd.DataFrame,
    predictions: pd.DataFrame,
    report: str,
    manifest: dict[str, Any],
) -> None:
    atomic_csv(grid, paths.grid_csv)
    atomic_csv(fold_metrics, paths.fold_metrics_csv)
    atomic_csv(
        candidate_horizon_metrics,
        paths.candidate_horizon_metrics_csv,
    )
    atomic_csv(candidate_scores, paths.candidate_scores_csv)
    atomic_csv(top3, paths.top3_csv)
    atomic_csv(runtime, paths.runtime_csv)
    atomic_parquet(predictions, paths.predictions_parquet)
    atomic_text(report, paths.report_txt)

    output_records: dict[str, Any] = {}
    for field in paths.__dataclass_fields__:
        if field == "manifest_json":
            continue
        path = getattr(paths, field)
        output_records[field] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }

    manifest["outputs"] = output_records
    atomic_json(manifest, paths.manifest_json)


def ensure_output_state(
    paths: OutputPaths,
    overwrite: bool,
) -> None:
    all_paths = [
        getattr(paths, field)
        for field in paths.__dataclass_fields__
    ]
    existing = [path for path in all_paths if path.exists()]

    if existing and not overwrite:
        raise FileExistsError(
            "Ya existen salidas de TREE-MARCH-SCREEN-001. "
            "No se sobrescriben sin --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    paths.grid_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions_parquet.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    try:
        (
            march,
            input_records,
            grid,
            supervised_by_horizon,
            folds,
        ) = preflight(args)

        if args.preflight_only:
            return 0

        paths = output_paths(args)
        ensure_output_state(paths, args.overwrite)

        print()
        print("=" * 96)
        print("INICIO TREE-MARCH-SCREEN-001")
        print("=" * 96)
        print(
            "Aviso: esta ejecución puede realizar hasta "
            f"{len(grid) * len(HORIZONS) * CV_SPLITS} fits."
        )
        print("April: NO")
        print("June:  NO")
        print()

        (
            grid_output,
            fold_metrics,
            candidate_horizon_metrics,
            candidate_scores,
            top3,
            runtime,
            predictions,
        ) = evaluate_screen(
            march=march,
            grid=grid,
            supervised_by_horizon=supervised_by_horizon,
        )

        validations = validate_outputs(
            fold_metrics=fold_metrics,
            candidate_horizon_metrics=candidate_horizon_metrics,
            candidate_scores=candidate_scores,
            top3=top3,
            runtime=runtime,
            predictions=predictions,
        )

        if not all(validations.values()):
            failed = [
                name
                for name, value in validations.items()
                if not value
            ]
            raise RuntimeError(
                "Fallaron validaciones internas: "
                + ", ".join(failed)
            )

        processing_seconds = time.perf_counter() - started
        paths = output_paths(args)

        report = build_report(
            input_records=input_records,
            folds=folds,
            candidate_scores=candidate_scores,
            top3=top3,
            runtime=runtime,
            validations=validations,
            processing_seconds=processing_seconds,
            paths=paths,
        )

        manifest = {
            "campaign_id": CAMPAIGN_ID,
            "parent_campaign_id": PARENT_CAMPAIGN_ID,
            "status": "PASS",
            "protocol": input_records["protocol"],
            "environment_preflight": input_records[
                "environment"
            ],
            "input": input_records["march"],
            "design": {
                "capture": "march",
                "capture_role": "development / screening",
                "april_accessed": False,
                "june_accessed": False,
                "target": TARGET,
                "interval_minutes": INTERVAL_MINUTES,
                "horizons_steps": HORIZONS,
                "horizons_minutes": [
                    HORIZON_MINUTES[h]
                    for h in HORIZONS
                ],
                "features": EXPECTED_FEATURE_COLUMNS,
                "feature_count": len(EXPECTED_FEATURE_COLUMNS),
                "lags": LAGS,
                "direct_model_per_horizon": True,
                "common_configuration_across_horizons": True,
                "cv": {
                    "class": "TimeSeriesSplit",
                    "n_splits": CV_SPLITS,
                    "gap": CV_GAP,
                    "shuffle": False,
                },
                "selection_metric": (
                    "mean_h(MAE_model_h / MAE_persistence_h)"
                ),
                "top_k_per_family": 3,
                "clipping": False,
                "refit_within_validation": False,
                "candidate_concurrency": 1,
                "hard_timeout_per_fit_seconds": (
                    FIT_TIMEOUT_SECONDS
                ),
            },
            "random_seed": CANONICAL_SEED,
            "n_jobs": N_JOBS,
            "grids": GRIDS,
            "folds": folds.to_dict(orient="records"),
            "top3": top3.to_dict(orient="records"),
            "validation": validations,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": package_version(
                    "scikit-learn"
                ),
                "xgboost": package_version("xgboost-cpu"),
                "lightgbm": package_version("lightgbm"),
                "pyarrow": package_version("pyarrow"),
                "platform": platform.platform(),
            },
            "processing_seconds": processing_seconds,
            "june_blind_authorized": False,
            "next_step": "TREE-APRIL-SELECTION-001",
        }

        manifest = json_safe(manifest)

        save_outputs(
            paths=paths,
            grid=grid_output,
            fold_metrics=fold_metrics,
            candidate_horizon_metrics=(
                candidate_horizon_metrics
            ),
            candidate_scores=candidate_scores,
            top3=top3,
            runtime=runtime,
            predictions=predictions,
            report=report,
            manifest=manifest,
        )

        print()
        print(report)
        print(
            f"Manifiesto: {paths.manifest_json.resolve()}"
        )
        print("TREE-MARCH-SCREEN-001: PASS")
        print("TREE-JUNE-BLIND-001: NOT AUTHORIZED")
        return 0

    except Exception as error:
        print(
            f"ERROR: {error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
