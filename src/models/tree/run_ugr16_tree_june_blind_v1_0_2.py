#!/usr/bin/env python3
"""
TREE-JUNE-BLIND-001
===================

Evaluación externa ciega de los representantes congelados de los métodos de
conjunto basados en árboles sobre UGR'16 June Week #3.

Diseño congelado:
- RF*   = RF01
- XGB*  = XGB06
- LGBM* = LGB01
- TREE* = LGB01 / LightGBM
- 23 variables explicativas
- granularidad 5 min
- H1/H3/H6/H12
- seed canónica 20260818
- 70 % inicial para fit, 30 % final para test ciego
- un fit por representante y horizonte
- sin retuning
- sin recalibración
- sin refit por origen
- actualización causal de variables explicativas
- sin clipping
- CPU, n_jobs=2, timeout 600 s por fit

Este runner NO carga resultados June del bloque estadístico. Los contrastes con
ARIMA* y VAR* pertenecen a TREE-INFERENCE-RESIDUALS-001, posterior a esta
campaña.

Con --preflight-only:
- valida gobernanza, entorno, identidad de June y manifiesto del split;
- inspecciona únicamente metadatos/esquema del Parquet;
- NO carga la serie completa;
- NO entrena;
- NO genera predicciones.
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


RUNNER_VERSION = "1.0.2"
CAMPAIGN_ID = "TREE-JUNE-BLIND-001"
PARENT_CAMPAIGN_ID = "UGR16-TREE-ENSEMBLES-001"

P017 = Path(
    "docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md"
)
P019 = Path(
    "docs/project_governance/019_tree_freeze_2026-08-19.md"
)
P020 = Path(
    "docs/project_governance/020_tree_stability_closure_2026-08-19.md"
)
ENVIRONMENT = Path(
    "results/metrics/tree_ensembles/tree_environment_preflight_20260819.txt"
)
JUNE = Path(
    "data/processed/ugr16/june_week3_prepared_5min.parquet"
)
JUNE_SPLIT = Path(
    "results/metrics/ugr16_final_split_manifest.json"
)

METRICS_DIR = Path("results/metrics/tree_ensembles")
PREDICTIONS_DIR = Path("results/predictions/tree_ensembles")
PREFIX = "ugr16_tree_june_blind"

EXPECTED_SHA = {
    "017": (
        "8c0a94e4ee80a84b78bf077d6abc3c18617ad1736b6d978047d0db1d52055003"
    ),
    "019": (
        "cdf5f1b46f183b19835d7d0159292a129c3c1f621894625627ea2fda0cd24ae3"
    ),
    "020": (
        "43b71c386c004fcadf69ba5585897f7bb96406543f73ad8fcea940337b488f94"
    ),
    "environment": (
        "82ab24602f66975929904bab802da8893bd059e51c7214bccf412a95275216b2"
    ),
    "june": (
        "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3"
    ),
    "june_split": (
        "a641f0bf011c705cfb8fb25c9042a198461918b01dddb284644a309b783adb39"
    ),
}

TARGET = "bitrate_bps"
INTERVAL_MINUTES = 5
HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
LAGS = [1,2,3,4,5,6,7,8,9,10,11,12,24,72,144,288]
FEATURES = [
    "current_value",
    *[f"lag_{lag}" for lag in LAGS],
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
]

CANONICAL_SEED = 20260818
N_JOBS = 2
FIT_TIMEOUT_SECONDS = 600

EXPECTED_ROWS = 2004
EXPECTED_START = pd.Timestamp("2016-06-13 01:00:00")
EXPECTED_END = pd.Timestamp("2016-06-19 23:55:00")
BOUNDARY = pd.Timestamp("2016-06-17 21:50:00")
TRAIN_END_TIMESTAMP = pd.Timestamp("2016-06-17 21:45:00")
TRAIN_ROWS = 1402
TRAIN_END_INDEX = 1401
BLIND_ROWS = 602
EXPECTED_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

REPRESENTATIVES = {
    "random_forest": "RF01",
    "xgboost": "XGB06",
    "lightgbm": "LGB01",
}
TREE_STAR = ("lightgbm", "LGB01")

CONFIGS: dict[str, dict[str, Any]] = {
    "RF01": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "max_features": 0.7,
    },
    "XGB06": {
        "n_estimators": 300,
        "learning_rate": 0.10,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
    },
    "LGB01": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 30,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Valida la campaña sin cargar la serie completa ni entrenar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas previas de esta campaña.",
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
            f"SHA-256 inesperado para {label}.\n"
            f"Esperado: {expected}\n"
            f"Actual:   {actual}"
        )
    return actual


def verify_sidecar(path: Path, label: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"Falta sidecar SHA de {label}: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected != sha256(path):
        raise RuntimeError(f"Sidecar SHA inválido para {label}.")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def validate_governance() -> dict[str, str]:
    """
    Valida la gobernanza exclusivamente mediante identidad criptográfica.

    017, 019 y 020 están congelados y sus SHA-256 exactos forman parte del
    runner. Por tanto, volver a exigir frases literales dentro de esos
    documentos es redundante y frágil: una diferencia de redacción no cambia
    su identidad ni su contenido aprobado.

    Los sidecars se comprueban además como control de integridad local.
    """
    records = {}

    for key, path in [
        ("017", P017),
        ("019", P019),
        ("020", P020),
        ("environment", ENVIRONMENT),
    ]:
        records[key] = verify_sha(
            path,
            EXPECTED_SHA[key],
            key,
        )
        verify_sidecar(path, key)

    return records

def validate_june_metadata() -> dict[str, Any]:
    june_sha = verify_sha(
        JUNE,
        EXPECTED_SHA["june"],
        "June 5min",
    )
    split_sha = verify_sha(
        JUNE_SPLIT,
        EXPECTED_SHA["june_split"],
        "June split manifest",
    )

    parquet = pq.ParquetFile(JUNE)
    schema_names = set(parquet.schema_arrow.names)

    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"June debe tener {EXPECTED_ROWS} filas; "
            f"metadatos={parquet.metadata.num_rows}."
        )

    required_columns = {"timestamp", TARGET}
    missing = required_columns.difference(schema_names)
    if missing:
        raise RuntimeError(
            f"June no contiene columnas requeridas: {sorted(missing)}"
        )

    with JUNE_SPLIT.open(encoding="utf-8") as file:
        split_manifest = json.load(file)

    # No se depende de una estructura JSON concreta para la validez científica:
    # la identidad del manifiesto queda fijada por SHA. El preflight registra
    # además los valores congelados de split establecidos antes de la evaluación.

    return {
        "june_sha256": june_sha,
        "split_sha256": split_sha,
        "rows_from_parquet_metadata": int(parquet.metadata.num_rows),
        "schema_has_timestamp": "timestamp" in schema_names,
        "schema_has_target": TARGET in schema_names,
        "split_manifest_loaded": isinstance(split_manifest, dict),
    }


def print_preflight(
    governance: dict[str, str],
    june_meta: dict[str, Any],
) -> None:
    current_runner = Path(__file__).resolve()
    current_runner_sha = sha256(current_runner)

    print("=" * 104)
    print("TREE-JUNE-BLIND-001 — PREFLIGHT SIN ENTRENAMIENTO")
    print("=" * 104)
    print(f"Runner version:              {RUNNER_VERSION}")
    print(f"Runner SHA-256:              {current_runner_sha}")
    print(f"017:                         PASS | {governance['017']}")
    print(f"019:                         PASS | {governance['019']}")
    print(f"020:                         PASS | {governance['020']}")
    print(
        f"Environment:                 PASS | "
        f"{governance['environment']}"
    )
    print(
        f"June identity:               PASS | "
        f"{june_meta['june_sha256']}"
    )
    print(
        f"June split manifest:         PASS | "
        f"{june_meta['split_sha256']}"
    )
    print(
        f"Parquet metadata rows:       PASS | "
        f"{june_meta['rows_from_parquet_metadata']}"
    )
    print("Parquet schema:              PASS | timestamp + bitrate_bps")
    print()
    print("DISEÑO CONGELADO")
    print("-" * 104)
    print("Captura:                     UGR'16 June Week #3")
    print(f"Filas totales:               {EXPECTED_ROWS}")
    print(f"Train:                       {TRAIN_ROWS}")
    print(f"Blind test físico:           {BLIND_ROWS}")
    print(f"Boundary:                    {BOUNDARY.isoformat()}")
    print(f"Train end:                   {TRAIN_END_TIMESTAMP.isoformat()}")
    print("Granularidad:                5 min")
    print("Features:                    23")
    print("Horizontes:                  H1 / H3 / H6 / H12")
    print(
        "Cobertura esperada:           "
        "H1=602 H3=600 H6=597 H12=591"
    )
    print("RF*:                         RF01")
    print("XGB*:                        XGB06")
    print("LGBM*:                       LGB01")
    print("TREE*:                       LGB01 / LightGBM")
    print(f"Seed:                        {CANONICAL_SEED}")
    print("Fits previstos:              3 × 4 = 12")
    print("Fit:                         único por representante/horizonte")
    print("Retuning:                    NO")
    print("Recalibración:               NO")
    print("Refit por origen:            NO")
    print("Clipping:                    NO")
    print("Resultados estadísticos June:NO ACCEDIDOS")
    print("Inferencia/residuos:         POSTERIOR")
    print()
    print("TREE-JUNE-BLIND PREFLIGHT: PASS")
    print("MODELOS ENTRENADOS EN PREFLIGHT: 0")
    print("PREDICCIONES GENERADAS: 0")


def load_june() -> pd.DataFrame:
    frame = pd.read_parquet(JUNE)[["timestamp", TARGET]].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="raise",
    )
    frame[TARGET] = pd.to_numeric(
        frame[TARGET],
        errors="raise",
    )
    frame = frame.reset_index(drop=True)

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError("Número de filas June inesperado.")
    if frame["timestamp"].iloc[0] != EXPECTED_START:
        raise RuntimeError("Inicio June inesperado.")
    if frame["timestamp"].iloc[-1] != EXPECTED_END:
        raise RuntimeError("Fin June inesperado.")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("June contiene timestamps duplicados.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("June no está ordenado temporalmente.")
    if not frame["timestamp"].diff().dropna().eq(
        pd.Timedelta(minutes=5)
    ).all():
        raise RuntimeError("June no tiene cadencia exacta de 5 min.")

    values = frame[TARGET].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("June contiene target no finito.")
    if np.any(values < 0):
        raise RuntimeError("June contiene target negativo.")

    boundary_index = frame.index[
        frame["timestamp"].eq(BOUNDARY)
    ].tolist()
    if boundary_index != [TRAIN_ROWS]:
        raise RuntimeError(
            f"Boundary June inesperado: índices={boundary_index}"
        )
    if frame.at[TRAIN_END_INDEX, "timestamp"] != TRAIN_END_TIMESTAMP:
        raise RuntimeError("Último timestamp de train inesperado.")

    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = frame["timestamp"]
    y = frame[TARGET].astype(float)

    features = pd.DataFrame(index=frame.index)
    features["current_value"] = y

    for lag in LAGS:
        features[f"lag_{lag}"] = y.shift(lag)

    minute_of_day = (
        timestamps.dt.hour * 60 + timestamps.dt.minute
    ).astype(float)
    dow = timestamps.dt.dayofweek.astype(float)

    features["hour_sin"] = np.sin(
        2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    )
    features["hour_cos"] = np.cos(
        2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    )
    features["dow_sin"] = np.sin(
        2.0 * np.pi * dow / 7.0
    )
    features["dow_cos"] = np.cos(
        2.0 * np.pi * dow / 7.0
    )
    features["is_weekend"] = (
        timestamps.dt.dayofweek >= 5
    ).astype(float)
    features["trend"] = np.arange(
        len(features),
        dtype=float,
    )

    if list(features.columns) != FEATURES:
        raise RuntimeError("Las 23 variables explicativas no coinciden.")
    if len(features.columns) != 23:
        raise RuntimeError("Número de variables explicativas distinto de 23.")

    return features


def supervised(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = features.copy()
    data["origin_index"] = np.arange(len(frame))
    data["target_index"] = data["origin_index"] + horizon
    data["origin_timestamp"] = frame["timestamp"]
    data["target_timestamp"] = frame["timestamp"].shift(-horizon)
    data["target_value"] = frame[TARGET].shift(-horizon)
    data["persistence_value"] = frame[TARGET]
    data = data.dropna().reset_index(drop=True)

    data["origin_index"] = data["origin_index"].astype(int)
    data["target_index"] = data["target_index"].astype(int)

    train = data[
        data["target_index"] <= TRAIN_END_INDEX
    ].copy()

    blind = data[
        data["origin_index"] >= TRAIN_END_INDEX
    ].copy()

    if len(blind) != EXPECTED_COUNTS[horizon]:
        raise RuntimeError(
            f"H{horizon}: cobertura ciega inesperada "
            f"{len(blind)} != {EXPECTED_COUNTS[horizon]}"
        )

    if int(train["target_index"].max()) != TRAIN_END_INDEX:
        raise RuntimeError(
            f"H{horizon}: último target de train incorrecto."
        )

    if int(blind["origin_index"].min()) != TRAIN_END_INDEX:
        raise RuntimeError(
            f"H{horizon}: primer origen ciego incorrecto."
        )

    if int(blind["target_index"].min()) <= TRAIN_END_INDEX:
        raise RuntimeError(
            f"H{horizon}: el primer target no está en el bloque ciego."
        )

    if int(train["target_index"].max()) > int(
        blind["origin_index"].min()
    ):
        raise RuntimeError(f"H{horizon}: leakage temporal.")

    delta = (
        blind["target_timestamp"] - blind["origin_timestamp"]
    )
    if not delta.eq(
        pd.Timedelta(minutes=HORIZON_MINUTES[horizon])
    ).all():
        raise RuntimeError(
            f"H{horizon}: timestamps objetivo desalineados."
        )

    matrix = data[FEATURES].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise RuntimeError(
            f"H{horizon}: variables explicativas no finitas."
        )

    info = {
        "horizon_steps": horizon,
        "horizon_minutes": HORIZON_MINUTES[horizon],
        "supervised_rows": len(data),
        "train_rows": len(train),
        "blind_rows": len(blind),
        "train_last_target_index": int(
            train["target_index"].max()
        ),
        "blind_first_origin_index": int(
            blind["origin_index"].min()
        ),
        "blind_first_target_index": int(
            blind["target_index"].min()
        ),
        "blind_first_target_timestamp": (
            blind["target_timestamp"].iloc[0].isoformat()
        ),
        "blind_last_target_timestamp": (
            blind["target_timestamp"].iloc[-1].isoformat()
        ),
        "leakage_check": True,
    }

    return train, blind, info


def model_for(
    family: str,
    candidate_id: str,
) -> Any:
    params = CONFIGS[candidate_id]

    if family == "random_forest":
        return RandomForestRegressor(
            criterion="absolute_error",
            bootstrap=True,
            random_state=CANONICAL_SEED,
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
            random_state=CANONICAL_SEED,
            n_jobs=N_JOBS,
            verbosity=0,
            **params,
        )

    if family == "lightgbm":
        return LGBMRegressor(
            boosting_type="gbdt",
            objective="regression_l1",
            reg_alpha=0,
            random_state=CANONICAL_SEED,
            n_jobs=N_JOBS,
            verbosity=-1,
            subsample_freq=0,
            **params,
        )

    raise RuntimeError(f"Familia no soportada: {family}")


def worker(
    connection: Any,
    family: str,
    candidate_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_blind: np.ndarray,
) -> None:
    try:
        model = model_for(family, candidate_id)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = np.asarray(
            model.predict(X_blind),
            dtype=float,
        )
        predict_seconds = time.perf_counter() - start

        serialized_bytes = len(
            pickle.dumps(
                model,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

        connection.send(
            {
                "status": "PASS",
                "predictions": predictions,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "serialized_model_bytes": serialized_bytes,
                "negative_predictions": int(
                    np.sum(predictions < 0)
                ),
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


def fit_with_timeout(
    family: str,
    candidate_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_blind: np.ndarray,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)

    process = context.Process(
        target=worker,
        args=(
            child,
            family,
            candidate_id,
            X_train,
            y_train,
            X_blind,
        ),
    )

    process.start()
    child.close()
    process.join(FIT_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        return {"status": "RESOURCE_LIMIT"}

    if not parent.poll(2):
        exitcode = process.exitcode
        parent.close()
        return {
            "status": "ERROR",
            "error_type": "WorkerNoResult",
            "error_message": f"exitcode={exitcode}",
        }

    result = parent.recv()
    parent.close()
    return result


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2.0 * np.abs(y_true - y_pred),
        denominator,
        out=np.zeros_like(y_true, dtype=float),
        where=denominator != 0,
    )
    return float(100.0 * np.mean(ratio))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    governance = validate_governance()
    june_meta = validate_june_metadata()
    print_preflight(governance, june_meta)

    if args.preflight_only:
        return 0

    paths = {
        "horizon_metrics": (
            METRICS_DIR / f"{PREFIX}_horizon_metrics.csv"
        ),
        "candidate_summary": (
            METRICS_DIR / f"{PREFIX}_candidate_summary.csv"
        ),
        "runtime": (
            METRICS_DIR / f"{PREFIX}_runtime.csv"
        ),
        "predictions": (
            PREDICTIONS_DIR / f"{PREFIX}_predictions.parquet"
        ),
        "report": (
            METRICS_DIR / f"{PREFIX}_report.txt"
        ),
        "manifest": (
            METRICS_DIR / f"{PREFIX}_manifest.json"
        ),
    }

    existing = [
        path for path in paths.values()
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Ya existen salidas de TREE-JUNE-BLIND-001. "
            "No se sobrescriben sin --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_june()
    features = build_features(frame)

    train_by_horizon = {}
    blind_by_horizon = {}
    split_records = []

    for horizon in HORIZONS:
        train, blind, info = supervised(
            features,
            frame,
            horizon,
        )
        train_by_horizon[horizon] = train
        blind_by_horizon[horizon] = blind
        split_records.append(info)

    mase_scale = float(
        np.mean(
            np.abs(
                np.diff(
                    frame.loc[
                        :TRAIN_END_INDEX,
                        TARGET,
                    ].to_numpy(float)
                )
            )
        )
    )
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise RuntimeError("Escala MASE June no válida.")

    metric_rows = []
    runtime_rows = []
    prediction_rows = []

    print()
    print("=" * 104)
    print("INICIO TREE-JUNE-BLIND-001")
    print("=" * 104)
    print("A partir de este punto se abre el test externo June.")
    print("Representantes e hiperparámetros permanecen congelados.")
    print("Fits previstos: 3 representantes × 4 horizontes = 12")
    print()

    done = 0
    total = 12

    for family, candidate_id in REPRESENTATIVES.items():
        print()
        print(f"{family} / {candidate_id}")

        for horizon in HORIZONS:
            done += 1

            train = train_by_horizon[horizon]
            blind = blind_by_horizon[horizon]

            X_train = train[FEATURES].to_numpy(float)
            y_train = train["target_value"].to_numpy(float)
            X_blind = blind[FEATURES].to_numpy(float)
            y_true = blind["target_value"].to_numpy(float)
            persistence = blind[
                "persistence_value"
            ].to_numpy(float)

            print(
                f"  H{horizon:<2} [{done}/{total}] ... ",
                end="",
                flush=True,
            )

            result = fit_with_timeout(
                family,
                candidate_id,
                X_train,
                y_train,
                X_blind,
            )

            runtime_row = {
                "family": family,
                "candidate_id": candidate_id,
                "horizon_steps": horizon,
                "horizon_minutes": HORIZON_MINUTES[horizon],
                "seed": CANONICAL_SEED,
                "status": result["status"],
                "fit_seconds": np.nan,
                "predict_seconds": np.nan,
                "serialized_model_bytes": np.nan,
                "negative_predictions": np.nan,
                "train_rows": len(train),
                "blind_rows": len(blind),
            }

            if result["status"] == "RESOURCE_LIMIT":
                runtime_rows.append(runtime_row)
                print("RESOURCE_LIMIT")
                continue

            if result["status"] != "PASS":
                raise RuntimeError(
                    f"{candidate_id} H{horizon}: "
                    f"{result.get('error_type')}: "
                    f"{result.get('error_message')}\n"
                    f"{result.get('traceback', '')}"
                )

            y_pred = np.asarray(
                result["predictions"],
                dtype=float,
            )
            if len(y_pred) != len(blind):
                raise RuntimeError(
                    f"{candidate_id} H{horizon}: cobertura incorrecta."
                )
            if not np.isfinite(y_pred).all():
                raise RuntimeError(
                    f"{candidate_id} H{horizon}: predicción no finita."
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

            error = y_true - y_pred
            persistence_error = y_true - persistence

            mae = float(np.mean(np.abs(error)))
            persistence_mae = float(
                np.mean(np.abs(persistence_error))
            )
            ratio = mae / persistence_mae

            metric_rows.append(
                {
                    "family": family,
                    "candidate_id": candidate_id,
                    "tree_star": (
                        family,
                        candidate_id,
                    ) == TREE_STAR,
                    "seed": CANONICAL_SEED,
                    "horizon_steps": horizon,
                    "horizon_minutes": HORIZON_MINUTES[
                        horizon
                    ],
                    "n_predictions": len(blind),
                    "model_mae_bps": mae,
                    "model_mae_mbps": mae / 1e6,
                    "persistence_mae_bps": persistence_mae,
                    "persistence_mae_mbps": (
                        persistence_mae / 1e6
                    ),
                    "mae_ratio_vs_persistence": ratio,
                    "skill_vs_persistence_pct": (
                        100.0
                        * (persistence_mae - mae)
                        / persistence_mae
                    ),
                    "model_rmse_bps": float(
                        np.sqrt(np.mean(error ** 2))
                    ),
                    "model_rmse_mbps": float(
                        np.sqrt(np.mean(error ** 2)) / 1e6
                    ),
                    "model_smape_pct": smape(
                        y_true,
                        y_pred,
                    ),
                    "model_mase": mae / mase_scale,
                    "model_bias_bps": float(
                        np.mean(error)
                    ),
                    "model_underprediction_pct": float(
                        100.0 * np.mean(error > 0)
                    ),
                    "model_p95_absolute_error_bps": float(
                        np.quantile(
                            np.abs(error),
                            0.95,
                        )
                    ),
                    "negative_predictions": int(
                        np.sum(y_pred < 0)
                    ),
                    "coverage": 1.0,
                    "mase_scale_bps": mase_scale,
                }
            )

            blind_reset = blind.reset_index(drop=True)

            for position in range(len(blind_reset)):
                prediction_rows.append(
                    {
                        "capture_key": "june",
                        "capture_name": (
                            "UGR'16 June Week #3"
                        ),
                        "evaluation_role": (
                            "blind_external_test"
                        ),
                        "family": family,
                        "candidate_id": candidate_id,
                        "tree_star": (
                            family,
                            candidate_id,
                        ) == TREE_STAR,
                        "seed": CANONICAL_SEED,
                        "horizon_steps": horizon,
                        "horizon_minutes": (
                            HORIZON_MINUTES[horizon]
                        ),
                        "origin_index": int(
                            blind_reset.at[
                                position,
                                "origin_index",
                            ]
                        ),
                        "target_index": int(
                            blind_reset.at[
                                position,
                                "target_index",
                            ]
                        ),
                        "origin_timestamp": (
                            blind_reset.at[
                                position,
                                "origin_timestamp",
                            ]
                        ),
                        "target_timestamp": (
                            blind_reset.at[
                                position,
                                "target_timestamp",
                            ]
                        ),
                        "y_true_bps": float(
                            y_true[position]
                        ),
                        "y_pred_bps": float(
                            y_pred[position]
                        ),
                        "persistence_pred_bps": float(
                            persistence[position]
                        ),
                        "error_bps": float(
                            error[position]
                        ),
                        "absolute_error_bps": float(
                            abs(error[position])
                        ),
                        "persistence_error_bps": float(
                            persistence_error[position]
                        ),
                        "persistence_absolute_error_bps": float(
                            abs(persistence_error[position])
                        ),
                    }
                )

            print(
                "PASS | "
                f"MAE={mae/1e6:.3f} Mbit/s | "
                f"ratio={ratio:.6f}"
            )

    metrics = pd.DataFrame(metric_rows)
    runtime = pd.DataFrame(runtime_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary = (
        metrics
        .groupby(
            [
                "family",
                "candidate_id",
                "tree_star",
            ],
            as_index=False,
        )["mae_ratio_vs_persistence"]
        .mean()
        .rename(
            columns={
                "mae_ratio_vs_persistence":
                    "score_june_descriptive"
            }
        )
    )

    fit_summary = (
        runtime
        .groupby(
            ["family", "candidate_id"],
            as_index=False,
        )
        .agg(
            total_fit_seconds=("fit_seconds", "sum"),
            total_predict_seconds=("predict_seconds", "sum"),
            total_serialized_model_bytes=(
                "serialized_model_bytes",
                "sum",
            ),
        )
    )

    summary = summary.merge(
        fit_summary,
        on=["family", "candidate_id"],
        validate="one_to_one",
    )
    summary["selection_changed"] = False
    summary["descriptive_rank"] = (
        summary["score_june_descriptive"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    expected_prediction_rows = (
        len(REPRESENTATIVES)
        * sum(EXPECTED_COUNTS.values())
    )

    checks = {
        "metrics_12":
            len(metrics) == 12,
        "runtime_12":
            len(runtime) == 12,
        "summary_3":
            len(summary) == 3,
        "predictions_7170":
            len(predictions) == expected_prediction_rows,
        "only_june":
            set(predictions["capture_key"]) == {"june"},
        "representatives_exact":
            set(predictions["candidate_id"])
            == set(REPRESENTATIVES.values()),
        "canonical_seed_only":
            set(predictions["seed"]) == {CANONICAL_SEED},
        "all_horizons":
            set(predictions["horizon_steps"])
            == set(HORIZONS),
        "no_resource_limit":
            not runtime["status"].eq(
                "RESOURCE_LIMIT"
            ).any(),
        "no_error":
            not runtime["status"].eq("ERROR").any(),
        "finite_predictions":
            np.isfinite(
                predictions["y_pred_bps"].to_numpy(float)
            ).all(),
        "finite_targets":
            np.isfinite(
                predictions["y_true_bps"].to_numpy(float)
            ).all(),
        "future_targets":
            (
                pd.to_datetime(
                    predictions["target_timestamp"]
                )
                >
                pd.to_datetime(
                    predictions["origin_timestamp"]
                )
            ).all(),
        "index_horizon":
            (
                predictions["target_index"]
                - predictions["origin_index"]
            ).eq(
                predictions["horizon_steps"]
            ).all(),
        "no_duplicates":
            not predictions.duplicated(
                [
                    "candidate_id",
                    "horizon_steps",
                    "target_timestamp",
                ]
            ).any(),
        "coverage_one":
            metrics["coverage"].eq(1.0).all(),
        "selection_never_changed":
            not summary["selection_changed"].any(),
        "tree_star_still_lgb01":
            summary.loc[
                summary["tree_star"],
                "candidate_id",
            ].tolist() == ["LGB01"],
    }

    if not all(checks.values()):
        failed = [
            name for name, value in checks.items()
            if not value
        ]
        raise RuntimeError(
            "Fallaron validaciones: "
            + ", ".join(failed)
        )

    report_lines = [
        "=" * 112,
        "TREE-JUNE-BLIND-001 — EVALUACIÓN EXTERNA CIEGA",
        "=" * 112,
        f"Campaña marco:                  {PARENT_CAMPAIGN_ID}",
        f"Subcampaña:                     {CAMPAIGN_ID}",
        "Estado:                         PASS",
        "",
        "JUNE",
        "-" * 112,
        f"SHA-256:                        {EXPECTED_SHA['june']}",
        f"Filas:                          {EXPECTED_ROWS}",
        f"Inicio:                         {EXPECTED_START.isoformat()}",
        f"Fin:                            {EXPECTED_END.isoformat()}",
        f"Boundary:                       {BOUNDARY.isoformat()}",
        f"Train:                          {TRAIN_ROWS}",
        f"Blind test físico:              {BLIND_ROWS}",
        "",
        "DISEÑO CONGELADO",
        "-" * 112,
        "RF*   = RF01",
        "XGB*  = XGB06",
        "LGBM* = LGB01",
        "TREE* = LGB01 / LightGBM",
        f"Seed =                          {CANONICAL_SEED}",
        "Fit único =                    YES",
        "Retuning =                     NO",
        "Recalibración =                NO",
        "Refit por origen =             NO",
        "Clipping =                     NO",
        "",
        "MÉTRICAS POR HORIZONTE",
        "-" * 112,
        metrics[
            [
                "family",
                "candidate_id",
                "horizon_steps",
                "n_predictions",
                "model_mae_mbps",
                "persistence_mae_mbps",
                "mae_ratio_vs_persistence",
                "skill_vs_persistence_pct",
                "model_rmse_mbps",
                "model_smape_pct",
                "model_mase",
                "model_bias_bps",
                "model_underprediction_pct",
                "model_p95_absolute_error_bps",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.9f}",
        ),
        "",
        "RESUMEN DESCRIPTIVO JUNE",
        "-" * 112,
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.9f}",
        ),
        "",
        "VALIDACIONES",
        "-" * 112,
    ]

    for name, value in checks.items():
        report_lines.append(
            f"{name:42s}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    report_lines.extend(
        [
            "",
            "INTERPRETACIÓN DE ESTADO",
            "-" * 112,
            "El ranking June es únicamente descriptivo.",
            "RF*, XGB*, LGBM* y TREE* NO pueden cambiar.",
            "Los contrastes con ARIMA*/VAR* NO se ejecutan aquí.",
            "La inferencia pareada y residuos pertenecen a TREE-INFERENCE-RESIDUALS-001.",
            "",
        ]
    )

    report = "\n".join(report_lines)

    atomic_csv(
        metrics,
        paths["horizon_metrics"],
    )
    atomic_csv(
        summary,
        paths["candidate_summary"],
    )
    atomic_csv(
        runtime,
        paths["runtime"],
    )
    atomic_parquet(
        predictions,
        paths["predictions"],
    )
    atomic_text(
        report,
        paths["report"],
    )

    output_records = {}
    for name, path in paths.items():
        if name == "manifest":
            continue
        output_records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }

    manifest = {
        "runner_version": RUNNER_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "capture": "june",
        "evaluation_role": "blind_external_test",
        "governance": {
            "017_sha256": governance["017"],
            "019_sha256": governance["019"],
            "020_sha256": governance["020"],
            "environment_sha256": governance[
                "environment"
            ],
        },
        "input": {
            "path": str(JUNE.resolve()),
            "sha256": EXPECTED_SHA["june"],
            "rows": EXPECTED_ROWS,
            "start": EXPECTED_START.isoformat(),
            "end": EXPECTED_END.isoformat(),
        },
        "split": {
            "manifest_path": str(
                JUNE_SPLIT.resolve()
            ),
            "manifest_sha256": (
                EXPECTED_SHA["june_split"]
            ),
            "train_rows": TRAIN_ROWS,
            "blind_rows": BLIND_ROWS,
            "boundary": BOUNDARY.isoformat(),
            "train_end": (
                TRAIN_END_TIMESTAMP.isoformat()
            ),
            "expected_counts": EXPECTED_COUNTS,
        },
        "design": {
            "target": TARGET,
            "interval_minutes": INTERVAL_MINUTES,
            "features": FEATURES,
            "feature_count": len(FEATURES),
            "lags": LAGS,
            "horizons_steps": HORIZONS,
            "horizons_minutes": [
                HORIZON_MINUTES[h]
                for h in HORIZONS
            ],
            "representatives": REPRESENTATIVES,
            "tree_star": {
                "family": TREE_STAR[0],
                "candidate_id": TREE_STAR[1],
            },
            "seed": CANONICAL_SEED,
            "single_fit_per_model_horizon": True,
            "retuning": False,
            "recalibration": False,
            "refit_per_origin": False,
            "causal_feature_updates": True,
            "clipping": False,
            "n_jobs": N_JOBS,
            "hard_timeout_seconds": FIT_TIMEOUT_SECONDS,
            "statistical_june_results_accessed": False,
        },
        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },
        "split_records": split_records,
        "candidate_summary": (
            summary.to_dict(orient="records")
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": package_version(
                "scikit-learn"
            ),
            "xgboost": package_version(
                "xgboost-cpu"
            ),
            "lightgbm": package_version(
                "lightgbm"
            ),
            "pyarrow": package_version(
                "pyarrow"
            ),
            "platform": platform.platform(),
        },
        "processing_seconds": (
            time.perf_counter() - started
        ),
        "selection_changed": False,
        "inference_residuals_run": False,
        "next_step": (
            "independent audit, then "
            "TREE-INFERENCE-RESIDUALS-001"
        ),
        "outputs": output_records,
    }

    atomic_json(
        manifest,
        paths["manifest"],
    )

    print()
    print(report)
    print(
        f"Manifiesto: {paths['manifest'].resolve()}"
    )
    print("TREE-JUNE-BLIND-001: PASS")
    print("SELECTION CHANGED: NO")
    print("TREE-INFERENCE-RESIDUALS-001: NOT RUN")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
