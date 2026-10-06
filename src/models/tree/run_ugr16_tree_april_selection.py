#!/usr/bin/env python3
"""
TREE-APRIL-SELECTION-001

Selección formal de RF*, XGB*, LGBM* y TREE* sobre UGR'16 April Week #3,
conforme a 017_tree_ensembles_protocol_2026-08-18.md y al cierre 018.

- April exclusivamente; June no está referenciado.
- 70 % inicial para fit y 30 % final para validation/selection.
- Sin refit durante validation; features actualizadas causalmente.
- 23 features congeladas, H1/H3/H6/H12.
- Solo los 9 supervivientes de March.
- Score = media_h(MAE_modelo_h / MAE_persistencia_h).
- Empate práctico <= 1 %.
- Semilla 20260818, n_jobs=2, timeout 600 s por fit, sin clipping.
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
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

CAMPAIGN_ID = "TREE-APRIL-SELECTION-001"
PARENT_CAMPAIGN_ID = "UGR16-TREE-ENSEMBLES-001"

DEFAULT_APRIL_SERIES = Path("data/processed/ugr16/april_week3_prepared_5min.parquet")
DEFAULT_PROTOCOL = Path("docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md")
DEFAULT_MARCH_CLOSURE = Path("docs/project_governance/018_tree_march_screen_closure_2026-08-19.md")
DEFAULT_MARCH_TOP3 = Path("results/metrics/tree_ensembles/ugr16_tree_march_screen_top3.csv")
DEFAULT_ENV_PREFLIGHT = Path("results/metrics/tree_ensembles/tree_environment_preflight_20260819.txt")
DEFAULT_METRICS_DIR = Path("results/metrics/tree_ensembles")
DEFAULT_PREDICTIONS_DIR = Path("results/predictions/tree_ensembles")
DEFAULT_PREFIX = "ugr16_tree_april_selection"

EXPECTED_PROTOCOL_SHA256 = "8c0a94e4ee80a84b78bf077d6abc3c18617ad1736b6d978047d0db1d52055003"
EXPECTED_MARCH_CLOSURE_SHA256 = "53445e2f95fcdc74a66240c5368cc5ad64d826e0a3300d95a61da0c91ca94e66"
EXPECTED_MARCH_TOP3_SHA256 = "a3aec331f30bafb592f1ff51c7652de7744f34d6b0daf064bc7d93fbef4ed3c4"
EXPECTED_ENV_PREFLIGHT_SHA256 = "82ab24602f66975929904bab802da8893bd059e51c7214bccf412a95275216b2"
EXPECTED_APRIL_SHA256 = "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0"

TARGET = "bitrate_bps"
INTERVAL_MINUTES = 5
HORIZONS = [1, 3, 6, 12]
HORIZON_MINUTES = {1: 5, 3: 15, 6: 30, 12: 60}
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 72, 144, 288]
INITIAL_TRAIN_FRACTION = 0.70
CANONICAL_SEED = 20260818
N_JOBS = 2
FIT_TIMEOUT_SECONDS = 600
PRACTICAL_TIE_RELATIVE = 0.01

EXPECTED_APRIL_ROWS = 2004
EXPECTED_APRIL_START = pd.Timestamp("2016-04-11 01:00:00")
EXPECTED_APRIL_END = pd.Timestamp("2016-04-17 23:55:00")
EXPECTED_INITIAL_TRAIN_ROWS = 1402
EXPECTED_INITIAL_TRAIN_END_INDEX = 1401
EXPECTED_INITIAL_TRAIN_END_TIMESTAMP = pd.Timestamp("2016-04-15 21:45:00")
EXPECTED_VALIDATION_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}

EXPECTED_FEATURE_COLUMNS = [
    "current_value",
    *[f"lag_{lag}" for lag in LAGS],
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "trend",
]
EXPECTED_MARCH_TOP3 = {
    "random_forest": ["RF01", "RF03", "RF02"],
    "xgboost": ["XGB03", "XGB06", "XGB05"],
    "lightgbm": ["LGB01", "LGB06", "LGB04"],
}
SURVIVOR_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "RF01": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 10, "max_features": 0.7},
        "RF03": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 3, "max_features": 0.7},
        "RF02": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "max_features": 0.7},
    },
    "xgboost": {
        "XGB03": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1},
        "XGB06": {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 3, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
        "XGB05": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
    },
    "lightgbm": {
        "LGB01": {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3, "min_child_samples": 30, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1},
        "LGB06": {"n_estimators": 300, "learning_rate": 0.10, "num_leaves": 15, "max_depth": 4, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
        "LGB04": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2},
    },
}

@dataclass(frozen=True)
class OutputPaths:
    survivor_grid_csv: Path
    candidate_horizon_metrics_csv: Path
    candidate_scores_csv: Path
    family_winners_csv: Path
    tree_winner_csv: Path
    runtime_csv: Path
    predictions_parquet: Path
    report_txt: Path
    manifest_json: Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TREE-APRIL-SELECTION-001 sobre April exclusivamente.")
    p.add_argument("--april-series", type=Path, default=DEFAULT_APRIL_SERIES)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--march-closure", type=Path, default=DEFAULT_MARCH_CLOSURE)
    p.add_argument("--march-top3", type=Path, default=DEFAULT_MARCH_TOP3)
    p.add_argument("--environment-preflight", type=Path, default=DEFAULT_ENV_PREFLIGHT)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()

def verify_exact_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 inesperado para {label}. Esperado={expected} Actual={actual} Ruta={path}")
    return {"path": str(path.resolve()), "sha256": actual, "status": "PASS"}

def verify_sidecar(path: Path) -> bool:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == sha256_file(path)

def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"

def output_paths(args: argparse.Namespace) -> OutputPaths:
    p = args.prefix
    return OutputPaths(
        args.metrics_dir / f"{p}_survivor_grid.csv",
        args.metrics_dir / f"{p}_candidate_horizon_metrics.csv",
        args.metrics_dir / f"{p}_candidate_scores.csv",
        args.metrics_dir / f"{p}_family_winners.csv",
        args.metrics_dir / f"{p}_tree_winner.csv",
        args.metrics_dir / f"{p}_runtime.csv",
        args.predictions_dir / f"{p}_predictions.parquet",
        args.metrics_dir / f"{p}_report.txt",
        args.metrics_dir / f"{p}_manifest.json",
    )

def ensure_output_state(paths: OutputPaths, overwrite: bool) -> None:
    existing = [getattr(paths, f) for f in paths.__dataclass_fields__ if getattr(paths, f).exists()]
    if existing and not overwrite:
        raise FileExistsError("Ya existen salidas; no se sobrescriben sin --overwrite:\n" + "\n".join(map(str, existing)))
    paths.survivor_grid_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions_parquet.parent.mkdir(parents=True, exist_ok=True)

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, path)

def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)

def json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    return str(v)

def validate_governance(args: argparse.Namespace) -> dict[str, Any]:
    protocol = verify_exact_sha(args.protocol, EXPECTED_PROTOCOL_SHA256, "017")
    closure = verify_exact_sha(args.march_closure, EXPECTED_MARCH_CLOSURE_SHA256, "018")
    if not verify_sidecar(args.protocol):
        raise RuntimeError("Sidecar de 017 no valida.")
    if not verify_sidecar(args.march_closure):
        raise RuntimeError("Sidecar de 018 no valida.")
    ptxt = args.protocol.read_text(encoding="utf-8")
    ctxt = args.march_closure.read_text(encoding="utf-8")
    for token in ["TFM-TREE-PROTOCOL-001", "Status = FROZEN", "70 % inicial", "30 % final", "TREE*"]:
        if token not in ptxt:
            raise RuntimeError(f"017 no contiene token: {token}")
    for token in ["TREE-MARCH-SCREEN-001", "Closed = TRUE", "TREE-APRIL-SELECTION-001", "AUTHORIZED = TRUE"]:
        if token not in ctxt:
            raise RuntimeError(f"018 no contiene token: {token}")
    return {"protocol": protocol, "march_closure": closure}

def validate_environment(path: Path) -> dict[str, Any]:
    rec = verify_exact_sha(path, EXPECTED_ENV_PREFLIGHT_SHA256, "preflight de entorno")
    if not verify_sidecar(path):
        raise RuntimeError("Sidecar del preflight de entorno no valida.")
    text = path.read_text(encoding="utf-8")
    for token in ["All installed packages are compatible", "RandomForest PASS", "XGBoost      PASS", "LightGBM     PASS", "TREE ENSEMBLES SMOKE TEST: PASS"]:
        if token not in text:
            raise RuntimeError(f"Preflight de entorno incompleto: {token}")
    return rec

def validate_march_top3(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rec = verify_exact_sha(path, EXPECTED_MARCH_TOP3_SHA256, "top3 March")
    df = pd.read_csv(path)
    if len(df) != 9:
        raise RuntimeError(f"Top3 March debe tener 9 filas; obtenidas={len(df)}")
    for family, expected in EXPECTED_MARCH_TOP3.items():
        observed = df[df["family"].eq(family)].sort_values("march_rank")["candidate_id"].tolist()
        if observed != expected:
            raise RuntimeError(f"Top3 inesperado {family}: {observed} != {expected}")
    return df, rec

def load_april(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rec = verify_exact_sha(path, EXPECTED_APRIL_SHA256, "April 5min")
    df = pd.read_parquet(path)
    missing = {"timestamp", TARGET}.difference(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas en April: {sorted(missing)}")
    out = df[["timestamp", TARGET]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="raise")
    out = out.reset_index(drop=True)
    if len(out) != EXPECTED_APRIL_ROWS:
        raise RuntimeError(f"April rows={len(out)}; esperado={EXPECTED_APRIL_ROWS}")
    if out["timestamp"].iloc[0] != EXPECTED_APRIL_START or out["timestamp"].iloc[-1] != EXPECTED_APRIL_END:
        raise RuntimeError("Cobertura temporal April inesperada.")
    if out["timestamp"].duplicated().any() or not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Timestamps April inválidos.")
    if not out["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("Cadencia April no es exactamente 5 min.")
    values = out[TARGET].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("Target April contiene valores inválidos.")
    train_rows = math.floor(len(out) * INITIAL_TRAIN_FRACTION)
    if train_rows != EXPECTED_INITIAL_TRAIN_ROWS:
        raise RuntimeError(f"Train rows inesperado: {train_rows}")
    end_idx = train_rows - 1
    if end_idx != EXPECTED_INITIAL_TRAIN_END_INDEX or out.at[end_idx, "timestamp"] != EXPECTED_INITIAL_TRAIN_END_TIMESTAMP:
        raise RuntimeError("Boundary 70/30 inesperado.")
    rec.update({
        "rows": len(out), "start": out["timestamp"].iloc[0].isoformat(), "end": out["timestamp"].iloc[-1].isoformat(),
        "target": TARGET, "interval_minutes": 5, "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "initial_train_rows": train_rows, "initial_train_end_index": end_idx,
        "initial_train_end_timestamp": out.at[end_idx, "timestamp"].isoformat(),
    })
    return out, rec

def make_features(series: pd.DataFrame) -> pd.DataFrame:
    ts = series["timestamp"]
    y = series[TARGET].astype(float)
    f = pd.DataFrame(index=series.index)
    f["current_value"] = y
    for lag in LAGS:
        f[f"lag_{lag}"] = y.shift(lag)
    minute = (ts.dt.hour * 60 + ts.dt.minute).astype(float)
    dow = ts.dt.dayofweek.astype(float)
    f["hour_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    f["hour_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    f["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    f["trend"] = np.arange(len(f), dtype=float)
    if list(f.columns) != EXPECTED_FEATURE_COLUMNS or len(f.columns) != 23:
        raise RuntimeError("Features no coinciden con 017.")
    return f

def make_supervised(features: pd.DataFrame, series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    r = features.copy()
    r["origin_index"] = np.arange(len(series), dtype=int)
    r["target_index"] = r["origin_index"] + horizon
    r["origin_timestamp"] = series["timestamp"]
    r["target_timestamp"] = series["timestamp"].shift(-horizon)
    r["target_value"] = series[TARGET].shift(-horizon)
    r["persistence_value"] = series[TARGET]
    r = r.dropna().reset_index(drop=True)
    r["origin_index"] = r["origin_index"].astype(int)
    r["target_index"] = r["target_index"].astype(int)
    if not (r["target_index"] - r["origin_index"]).eq(horizon).all():
        raise RuntimeError(f"H{horizon}: desalineación de índices.")
    if not (r["target_timestamp"] - r["origin_timestamp"]).eq(pd.to_timedelta(HORIZON_MINUTES[horizon], unit="min")).all():
        raise RuntimeError(f"H{horizon}: desalineación temporal.")
    if not np.isfinite(r[EXPECTED_FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError(f"H{horizon}: features no finitas.")
    return r

def split_supervised(s: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = s[s["target_index"] <= EXPECTED_INITIAL_TRAIN_END_INDEX].copy()
    valid = s[s["origin_index"] >= EXPECTED_INITIAL_TRAIN_END_INDEX].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"H{horizon}: split vacío.")
    if int(train["target_index"].max()) != EXPECTED_INITIAL_TRAIN_END_INDEX:
        raise RuntimeError(f"H{horizon}: último target train incorrecto.")
    if int(valid["origin_index"].min()) != EXPECTED_INITIAL_TRAIN_END_INDEX:
        raise RuntimeError(f"H{horizon}: primer origen validation incorrecto.")
    if int(valid["target_index"].min()) <= EXPECTED_INITIAL_TRAIN_END_INDEX:
        raise RuntimeError(f"H{horizon}: leakage target validation.")
    if len(valid) != EXPECTED_VALIDATION_COUNTS[horizon]:
        raise RuntimeError(f"H{horizon}: validation={len(valid)}; esperado={EXPECTED_VALIDATION_COUNTS[horizon]}")
    if int(train["target_index"].max()) > int(valid["origin_index"].min()):
        raise RuntimeError(f"H{horizon}: leakage temporal en boundary.")
    info = {
        "horizon_steps": horizon, "horizon_minutes": HORIZON_MINUTES[horizon], "supervised_rows": len(s),
        "training_rows": len(train), "training_first_origin_index": int(train["origin_index"].min()),
        "training_last_origin_index": int(train["origin_index"].max()), "training_last_target_index": int(train["target_index"].max()),
        "validation_rows": len(valid), "validation_first_origin_index": int(valid["origin_index"].min()),
        "validation_last_origin_index": int(valid["origin_index"].max()), "validation_first_target_index": int(valid["target_index"].min()),
        "validation_last_target_index": int(valid["target_index"].max()),
        "validation_first_origin_timestamp": valid["origin_timestamp"].iloc[0].isoformat(),
        "validation_first_target_timestamp": valid["target_timestamp"].iloc[0].isoformat(),
        "validation_last_target_timestamp": valid["target_timestamp"].iloc[-1].isoformat(), "leakage_check": True,
    }
    return train, valid, info

def survivor_grid_frame(top3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ["random_forest", "xgboost", "lightgbm"]:
        rank = top3[top3["family"].eq(family)].sort_values("march_rank")
        for _, m in rank.iterrows():
            cid = str(m["candidate_id"])
            row = {"family": family, "candidate_id": cid, "march_rank": int(m["march_rank"]), "march_score": float(m["score"]), **SURVIVOR_CONFIGS[family][cid]}
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 9:
        raise RuntimeError("Survivor grid April debe tener 9 candidatos.")
    return frame

def build_model(family: str, cid: str, seed: int) -> Any:
    p = SURVIVOR_CONFIGS[family][cid]
    if family == "random_forest":
        return RandomForestRegressor(criterion="absolute_error", bootstrap=True, random_state=seed, n_jobs=N_JOBS, **p)
    if family == "xgboost":
        return XGBRegressor(booster="gbtree", tree_method="hist", objective="reg:absoluteerror", gamma=0, reg_alpha=0, random_state=seed, n_jobs=N_JOBS, verbosity=0, **p)
    if family == "lightgbm":
        return LGBMRegressor(boosting_type="gbdt", objective="regression_l1", reg_alpha=0, random_state=seed, n_jobs=N_JOBS, verbosity=-1, subsample_freq=(1 if p["subsample"] < 1.0 else 0), **p)
    raise ValueError(f"Familia no soportada: {family}")

def _fit_worker(conn: Any, family: str, cid: str, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray) -> None:
    try:
        model = build_model(family, cid, CANONICAL_SEED)
        t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = np.asarray(model.predict(Xv), dtype=float); pred_s = time.perf_counter() - t0
        size = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        conn.send({"status": "PASS", "predictions": pred, "fit_seconds": fit_s, "predict_seconds": pred_s, "serialized_model_bytes": size, "negative_predictions": int(np.sum(pred < 0))})
    except BaseException as exc:
        conn.send({"status": "ERROR", "error_type": exc.__class__.__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    finally:
        conn.close()

def fit_with_timeout(family: str, cid: str, Xtr: np.ndarray, ytr: np.ndarray, Xv: np.ndarray) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_fit_worker, args=(child, family, cid, Xtr, ytr, Xv))
    proc.start(); child.close(); proc.join(FIT_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate(); proc.join(10)
        if proc.is_alive(): proc.kill(); proc.join()
        parent.close()
        return {"status": "RESOURCE_LIMIT", "timeout_seconds": FIT_TIMEOUT_SECONDS}
    if not parent.poll(2):
        code = proc.exitcode; parent.close()
        return {"status": "ERROR", "error_type": "WorkerNoResult", "error_message": f"Worker sin resultado; exitcode={code}"}
    result = parent.recv(); parent.close(); return result

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.abs(y_true) + np.abs(y_pred)
    v = np.divide(2 * np.abs(y_true - y_pred), den, out=np.zeros_like(y_true, dtype=float), where=den != 0)
    return float(100 * np.mean(v))

def preflight(args: argparse.Namespace):
    gov = validate_governance(args)
    env = validate_environment(args.environment_preflight)
    top3, top3_rec = validate_march_top3(args.march_top3)
    april, april_rec = load_april(args.april_series)
    features = make_features(april)
    train_by_h, valid_by_h, split_records = {}, {}, []
    for h in HORIZONS:
        supervised = make_supervised(features, april, h)
        train, valid, info = split_supervised(supervised, h)
        train_by_h[h], valid_by_h[h] = train, valid
        split_records.append(info)
    grid = survivor_grid_frame(top3)
    records = {**gov, "environment": env, "march_top3": top3_rec, "april": april_rec}
    print("=" * 100)
    print("TREE-APRIL-SELECTION-001 — PREFLIGHT")
    print("=" * 100)
    print(f"017:                           PASS | {gov['protocol']['sha256']}")
    print(f"018:                           PASS | {gov['march_closure']['sha256']}")
    print(f"March top3:                    PASS | {top3_rec['sha256']}")
    print(f"Environment preflight:         PASS | {env['sha256']}")
    print(f"April:                         PASS | {len(april)} filas | {april_rec['start']} → {april_rec['end']}")
    print(f"April SHA-256:                 {april_rec['sha256']}")
    print("Features:                      PASS | 23")
    print("Horizontes:                    PASS | H1,H3,H6,H12")
    print("Split:                         PASS | 70 % fit / 30 % validation")
    print(f"Train inicial:                 {EXPECTED_INITIAL_TRAIN_ROWS} filas | fin={EXPECTED_INITIAL_TRAIN_END_TIMESTAMP.isoformat()}")
    print("Refit en validation:           NO")
    print("Clipping:                      NO")
    print(f"Seed canónica:                 {CANONICAL_SEED}")
    print(f"n_jobs:                        {N_JOBS}")
    print(f"Timeout por fit:               {FIT_TIMEOUT_SECONDS} s")
    print("Candidatos April:              PASS | 9")
    print("June accessible by runner:     NO")
    print("\nSUPERVISED / TRAIN / VALIDATION")
    print("-" * 100)
    for x in split_records:
        print(f"H{x['horizon_steps']:<2} ({x['horizon_minutes']:>2} min): supervised={x['supervised_rows']:,} | train={x['training_rows']:,} | validation={x['validation_rows']:,} | first_origin={x['validation_first_origin_index']} | leakage=PASS")
    print("\nSUPERVIVIENTES MARCH")
    print("-" * 100)
    for family, ids in EXPECTED_MARCH_TOP3.items(): print(f"{family:16s}: {', '.join(ids)}")
    print("\nREGLA DE DESEMPATE")
    print("-" * 100)
    print("Empate práctico: diferencia relativa de Score <= 1 %.")
    print("Serialized model size: suma de bytes de H1/H3/H6/H12.")
    print("Intrafamilia: fit time total → serialized bytes total → candidate_id.")
    print("TREE*: fit time total → serialized bytes total; empate exacto persistente => STOP.")
    print("\nTREE-APRIL-SELECTION PREFLIGHT: PASS")
    print("TREE-JUNE-BLIND-001: NOT AUTHORIZED")
    return april, grid, train_by_h, valid_by_h, top3, split_records, records

def evaluate(april: pd.DataFrame, grid: pd.DataFrame, train_by_h: dict[int, pd.DataFrame], valid_by_h: dict[int, pd.DataFrame]):
    pred_rows, metric_rows, runtime_rows = [], [], []
    scale = float(np.mean(np.abs(np.diff(april.loc[:EXPECTED_INITIAL_TRAIN_END_INDEX, TARGET].to_numpy(float)))))
    if not np.isfinite(scale) or scale <= 0: raise RuntimeError("Escala MASE April no válida.")
    total, done = 36, 0
    for family in ["random_forest", "xgboost", "lightgbm"]:
        print("\n" + "=" * 100); print(f"FAMILIA: {family}"); print("=" * 100)
        fam = grid[grid["family"].eq(family)].sort_values("march_rank")
        for _, row in fam.iterrows():
            cid = str(row["candidate_id"]); print(f"\n{cid}")
            for h in HORIZONS:
                done += 1
                tr, va = train_by_h[h], valid_by_h[h]
                Xtr = tr[EXPECTED_FEATURE_COLUMNS].to_numpy(float); ytr = tr["target_value"].to_numpy(float)
                Xv = va[EXPECTED_FEATURE_COLUMNS].to_numpy(float); yt = va["target_value"].to_numpy(float); pers = va["persistence_value"].to_numpy(float)
                print(f"  H{h:<2} [{done}/{total}] ... ", end="", flush=True)
                result = fit_with_timeout(family, cid, Xtr, ytr, Xv)
                rt = {"family": family, "candidate_id": cid, "horizon_steps": h, "horizon_minutes": HORIZON_MINUTES[h], "seed": CANONICAL_SEED, "status": result["status"], "fit_seconds": np.nan, "predict_seconds": np.nan, "serialized_model_bytes": np.nan, "negative_predictions": np.nan, "training_rows": len(tr), "validation_rows": len(va), "timeout_seconds": FIT_TIMEOUT_SECONDS}
                if result["status"] == "RESOURCE_LIMIT": runtime_rows.append(rt); print("RESOURCE_LIMIT"); continue
                if result["status"] != "PASS": raise RuntimeError(f"{cid} H{h}: {result.get('error_type')}: {result.get('error_message')}\n{result.get('traceback','')}")
                yp = np.asarray(result["predictions"], dtype=float)
                if len(yp) != len(va) or not np.isfinite(yp).all(): raise RuntimeError(f"{cid} H{h}: predicciones inválidas.")
                rt.update({"fit_seconds": float(result["fit_seconds"]), "predict_seconds": float(result["predict_seconds"]), "serialized_model_bytes": int(result["serialized_model_bytes"]), "negative_predictions": int(result["negative_predictions"])})
                runtime_rows.append(rt)
                err, perr = yt - yp, yt - pers
                mae, pmae = float(np.mean(np.abs(err))), float(np.mean(np.abs(perr)))
                metric_rows.append({"family": family, "candidate_id": cid, "horizon_steps": h, "horizon_minutes": HORIZON_MINUTES[h], "seed": CANONICAL_SEED, "status": "PASS", "n_predictions": len(va), "model_mae_bps": mae, "model_mae_mbps": mae/1e6, "persistence_mae_bps": pmae, "persistence_mae_mbps": pmae/1e6, "mae_ratio_vs_persistence": mae/pmae, "skill_vs_persistence_pct": 100*(pmae-mae)/pmae, "model_rmse_bps": float(np.sqrt(np.mean(err**2))), "model_rmse_mbps": float(np.sqrt(np.mean(err**2))/1e6), "model_smape_pct": smape(yt, yp), "model_mase": mae/scale, "model_bias_bps": float(np.mean(err)), "model_underprediction_pct": float(100*np.mean(err>0)), "model_p95_absolute_error_bps": float(np.quantile(np.abs(err),0.95)), "negative_predictions": int(np.sum(yp<0)), "coverage": 1.0, "mase_scale_bps": scale})
                vr = va.reset_index(drop=True)
                for i in range(len(vr)):
                    pred_rows.append({"capture_key": "april", "capture_name": "UGR'16 April Week #3", "evaluation_role": "selection / validation", "family": family, "candidate_id": cid, "seed": CANONICAL_SEED, "horizon_steps": h, "horizon_minutes": HORIZON_MINUTES[h], "origin_index": int(vr.at[i,"origin_index"]), "target_index": int(vr.at[i,"target_index"]), "origin_timestamp": vr.at[i,"origin_timestamp"], "target_timestamp": vr.at[i,"target_timestamp"], "y_true_bps": float(yt[i]), "y_pred_bps": float(yp[i]), "persistence_pred_bps": float(pers[i]), "error_bps": float(err[i]), "absolute_error_bps": float(abs(err[i])), "persistence_error_bps": float(perr[i]), "persistence_absolute_error_bps": float(abs(perr[i]))})
                print(f"PASS | MAE={mae/1e6:.3f} Mbit/s | ratio={mae/pmae:.4f}")
    return pd.DataFrame(metric_rows), pd.DataFrame(runtime_rows), pd.DataFrame(pred_rows)

def candidate_scores(metrics: pd.DataFrame, runtime: pd.DataFrame, top3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, ids in EXPECTED_MARCH_TOP3.items():
        for cid in ids:
            m = metrics[metrics["family"].eq(family) & metrics["candidate_id"].eq(cid)]
            r = runtime[runtime["family"].eq(family) & runtime["candidate_id"].eq(cid)]
            resource = r["status"].eq("RESOURCE_LIMIT").any()
            complete = set(m["horizon_steps"]) == set(HORIZONS) and len(m) == 4 and not resource
            mr = top3[top3["family"].eq(family) & top3["candidate_id"].eq(cid)].iloc[0]
            row = {"family": family, "candidate_id": cid, "march_rank": int(mr["march_rank"]), "march_score": float(mr["score"]), "seed": CANONICAL_SEED, "status": "PASS" if complete else ("RESOURCE_LIMIT" if resource else "INCOMPLETE"), "score": np.nan, "total_fit_seconds": float(r["fit_seconds"].sum(skipna=True)), "total_predict_seconds": float(r["predict_seconds"].sum(skipna=True)), "total_serialized_model_bytes": float(r["serialized_model_bytes"].sum(skipna=True)), "negative_predictions": int(r["negative_predictions"].sum(skipna=True))}
            for h in HORIZONS:
                one = m[m["horizon_steps"].eq(h)]
                row[f"h{h}_mae_ratio"] = float(one.iloc[0]["mae_ratio_vs_persistence"]) if len(one)==1 else np.nan
            if complete: row["score"] = float(np.mean([row[f"h{h}_mae_ratio"] for h in HORIZONS]))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["family","score","candidate_id"], na_position="last", kind="stable").reset_index(drop=True)

def tie_mask(df: pd.DataFrame, best: float) -> pd.Series:
    return ((df["score"] - best) / abs(best)).le(PRACTICAL_TIE_RELATIVE + 1e-15)

def select_family_winners(scores: pd.DataFrame):
    scored = scores.copy(); scored["practical_tie_with_family_best"] = False; scored["selected_family_winner"] = False
    winners = []
    for family in ["random_forest", "xgboost", "lightgbm"]:
        f = scored[scored["family"].eq(family) & scored["status"].eq("PASS") & scored["score"].notna()].copy()
        if len(f) != 3: raise RuntimeError(f"{family}: se requieren 3 candidatos válidos.")
        best = float(f["score"].min()); mask = tie_mask(f, best); tied = f.loc[mask].copy()
        ids = tied["candidate_id"].tolist(); scored.loc[scored["family"].eq(family) & scored["candidate_id"].isin(ids), "practical_tie_with_family_best"] = True
        tied = tied.sort_values(["total_fit_seconds","total_serialized_model_bytes","candidate_id"], kind="stable")
        w = tied.iloc[0].copy(); w["family_best_score"] = best; w["practical_tie_count"] = len(tied); w["selection_rule"] = "score; <=1% tie: total_fit_seconds, total_serialized_model_bytes, candidate_id"
        winners.append(w); scored.loc[scored["family"].eq(family) & scored["candidate_id"].eq(w["candidate_id"]), "selected_family_winner"] = True
    return scored, pd.DataFrame(winners).reset_index(drop=True)

def select_tree_winner(winners: pd.DataFrame) -> pd.DataFrame:
    if len(winners) != 3: raise RuntimeError("TREE*: se requieren 3 family winners.")
    best = float(winners["score"].min()); tied = winners.loc[tie_mask(winners, best)].copy().sort_values(["total_fit_seconds","total_serialized_model_bytes"], kind="stable")
    if len(tied) > 1:
        a,b = tied.iloc[0], tied.iloc[1]
        if math.isclose(float(a["total_fit_seconds"]), float(b["total_fit_seconds"]), rel_tol=0, abs_tol=1e-12) and math.isclose(float(a["total_serialized_model_bytes"]), float(b["total_serialized_model_bytes"]), rel_tol=0, abs_tol=0.5):
            raise RuntimeError("TREE*: empate exacto persiste tras criterios preespecificados; se requiere enmienda.")
    out = tied.iloc[[0]].copy(); out["tree_best_score"] = best; out["practical_tie_count"] = len(tied); out["selected_tree_winner"] = True; out["selection_rule_tree"] = "score; <=1% tie: total_fit_seconds, total_serialized_model_bytes"
    return out.reset_index(drop=True)

def validate_results(metrics, runtime, predictions, scores, winners, tree):
    expected = 9 * sum(EXPECTED_VALIDATION_COUNTS.values())
    valid = scores[scores["status"].eq("PASS")]
    return {
        "metrics_36_rows": len(metrics)==36,
        "runtime_36_rows": len(runtime)==36,
        "predictions_21510_rows": len(predictions)==expected,
        "only_april": set(predictions["capture_key"])=={"april"},
        "all_9_candidates": predictions["candidate_id"].nunique()==9,
        "all_4_horizons": set(predictions["horizon_steps"])==set(HORIZONS),
        "canonical_seed_only": set(predictions["seed"])=={CANONICAL_SEED},
        "no_resource_limit": not runtime["status"].eq("RESOURCE_LIMIT").any(),
        "no_error": not runtime["status"].eq("ERROR").any(),
        "finite_predictions": bool(np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all()),
        "finite_targets": bool(np.isfinite(predictions["y_true_bps"].to_numpy(float)).all()),
        "future_targets": bool((predictions["target_timestamp"]>predictions["origin_timestamp"]).all()),
        "index_horizon": bool((predictions["target_index"]-predictions["origin_index"]).eq(predictions["horizon_steps"]).all()),
        "no_duplicates": not predictions.duplicated(["family","candidate_id","horizon_steps","target_timestamp"]).any(),
        "coverage_one": metrics["coverage"].eq(1.0).all(),
        "scores_9_rows": len(scores)==9,
        "scores_all_pass": len(valid)==9,
        "scores_finite": bool(np.isfinite(valid["score"].to_numpy(float)).all()),
        "family_winners_3": len(winners)==3,
        "one_winner_per_family": winners["family"].nunique()==3,
        "tree_winner_1": len(tree)==1,
        "tree_winner_is_family_winner": bool(tree.iloc[0]["candidate_id"] in set(winners["candidate_id"])),
    }

def build_report(records, split_records, scores, winners, tree, runtime, checks, elapsed, outputs):
    lines = ["="*112, "TREE-APRIL-SELECTION-001 — PHASE C TREE ENSEMBLES", "="*112,
             f"Campaña marco:                  {PARENT_CAMPAIGN_ID}", f"Subcampaña:                     {CAMPAIGN_ID}", "Estado:                         PASS", "",
             "GOBERNANZA", "-"*112, f"017 SHA-256:                    {records['protocol']['sha256']}", f"018 SHA-256:                    {records['march_closure']['sha256']}", f"March top3 SHA-256:             {records['march_top3']['sha256']}", f"Environment SHA-256:            {records['environment']['sha256']}", "",
             "APRIL", "-"*112, f"Entrada:                        {records['april']['path']}", f"SHA-256:                        {records['april']['sha256']}", f"Filas:                          {records['april']['rows']}", f"Inicio:                         {records['april']['start']}", f"Fin:                            {records['april']['end']}", f"Target:                         {TARGET}", "Resolución:                     5 min", f"Train inicial:                  {EXPECTED_INITIAL_TRAIN_ROWS}", f"Fin train:                      {EXPECTED_INITIAL_TRAIN_END_TIMESTAMP.isoformat()}", "",
             "DISEÑO", "-"*112, "Split:                           70 % fit / 30 % validation", "Refit validation:                NO", "Features:                        23", f"Lags:                            {LAGS}", f"Horizontes:                      {HORIZONS}", f"Seed:                            {CANONICAL_SEED}", f"n_jobs:                          {N_JOBS}", f"Timeout por fit:                 {FIT_TIMEOUT_SECONDS} s", "Clipping:                        NO", "Serialized model size:           suma de H1/H3/H6/H12", "June:                            NO ACCESIBLE EN ESTE RUNNER", "", "COBERTURA", "-"*112]
    for x in split_records:
        lines.append(f"H{x['horizon_steps']:>2} ({x['horizon_minutes']:>2} min): train={x['training_rows']:,} | validation={x['validation_rows']:,} | first target={x['validation_first_target_timestamp']} | leakage=PASS")
    cols = ["family","candidate_id","march_rank","march_score","score","h1_mae_ratio","h3_mae_ratio","h6_mae_ratio","h12_mae_ratio","total_fit_seconds","total_serialized_model_bytes","practical_tie_with_family_best","selected_family_winner"]
    lines += ["", "SCORES APRIL", "-"*112, scores[cols].to_string(index=False, float_format=lambda x:f"{x:.6f}"), "", "REPRESENTANTES DE FAMILIA", "-"*112, winners[["family","candidate_id","score","total_fit_seconds","total_serialized_model_bytes","practical_tie_count"]].to_string(index=False, float_format=lambda x:f"{x:.6f}"), "", "TREE*", "-"*112, tree[["family","candidate_id","score","total_fit_seconds","total_serialized_model_bytes","practical_tie_count"]].to_string(index=False, float_format=lambda x:f"{x:.6f}"), "", "VALIDACIONES", "-"*112]
    for k,v in checks.items(): lines.append(f"{k:42s}: {'PASS' if v else 'FAIL'}")
    lines += ["", "RECURSOS", "-"*112, f"Fits registrados:               {len(runtime)}", f"RESOURCE_LIMIT:                 {int(runtime['status'].eq('RESOURCE_LIMIT').sum())}", f"ERROR:                          {int(runtime['status'].eq('ERROR').sum())}", f"Tiempo total de campaña:        {elapsed:.3f} s", "", "DECISIÓN", "-"*112,
              f"RF*:                            {winners.loc[winners['family'].eq('random_forest'),'candidate_id'].iloc[0]}", f"XGB*:                           {winners.loc[winners['family'].eq('xgboost'),'candidate_id'].iloc[0]}", f"LGBM*:                          {winners.loc[winners['family'].eq('lightgbm'),'candidate_id'].iloc[0]}", f"TREE*:                          {tree.iloc[0]['candidate_id']} ({tree.iloc[0]['family']})", "", "La selección queda basada exclusivamente en April.", "TREE-FREEZE-001 sigue PENDING.", "TREE-JUNE-BLIND-001 sigue NOT AUTHORIZED.", "", "SALIDAS", "-"*112]
    for field in outputs.__dataclass_fields__: lines.append(f"{field}: {getattr(outputs,field).resolve()}")
    return "\n".join(lines)+"\n"

def save_outputs(outputs, grid, metrics, scores, winners, tree, runtime, predictions, report, manifest):
    atomic_csv(grid, outputs.survivor_grid_csv); atomic_csv(metrics, outputs.candidate_horizon_metrics_csv); atomic_csv(scores, outputs.candidate_scores_csv); atomic_csv(winners, outputs.family_winners_csv); atomic_csv(tree, outputs.tree_winner_csv); atomic_csv(runtime, outputs.runtime_csv); atomic_parquet(predictions, outputs.predictions_parquet); atomic_text(report, outputs.report_txt)
    records = {}
    for field in outputs.__dataclass_fields__:
        if field == "manifest_json": continue
        p = getattr(outputs, field); records[field] = {"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": int(p.stat().st_size)}
    manifest["outputs"] = records; atomic_json(manifest, outputs.manifest_json)

def main() -> int:
    args = parse_args(); started = time.perf_counter()
    try:
        april, grid, train_by_h, valid_by_h, top3, split_records, records = preflight(args)
        if args.preflight_only: return 0
        outputs = output_paths(args); ensure_output_state(outputs, args.overwrite)
        print("\n"+"="*100); print("INICIO TREE-APRIL-SELECTION-001"); print("="*100); print("Fits previstos: 9 × 4 = 36"); print("Refit validation: NO"); print("June: NO\n")
        metrics, runtime, predictions = evaluate(april, grid, train_by_h, valid_by_h)
        scores = candidate_scores(metrics, runtime, top3)
        scores, winners = select_family_winners(scores)
        tree = select_tree_winner(winners)
        checks = validate_results(metrics, runtime, predictions, scores, winners, tree)
        if not all(checks.values()):
            raise RuntimeError("Fallaron validaciones internas: " + ", ".join(k for k,v in checks.items() if not v))
        elapsed = time.perf_counter()-started
        report = build_report(records, split_records, scores, winners, tree, runtime, checks, elapsed, outputs)
        manifest = json_safe({
            "campaign_id": CAMPAIGN_ID, "parent_campaign_id": PARENT_CAMPAIGN_ID, "status": "PASS",
            "protocol": records["protocol"], "march_closure": records["march_closure"], "march_top3": records["march_top3"], "environment_preflight": records["environment"], "input": records["april"],
            "design": {"capture": "april", "capture_role": "selection / validation", "june_accessed": False, "target": TARGET, "interval_minutes": 5, "initial_train_fraction": INITIAL_TRAIN_FRACTION, "initial_train_rows": EXPECTED_INITIAL_TRAIN_ROWS, "initial_train_end_index": EXPECTED_INITIAL_TRAIN_END_INDEX, "initial_train_end_timestamp": EXPECTED_INITIAL_TRAIN_END_TIMESTAMP.isoformat(), "horizons_steps": HORIZONS, "horizons_minutes": [HORIZON_MINUTES[h] for h in HORIZONS], "validation_counts": EXPECTED_VALIDATION_COUNTS, "features": EXPECTED_FEATURE_COLUMNS, "feature_count": 23, "lags": LAGS, "direct_model_per_horizon": True, "common_configuration_across_horizons": True, "model_refit_during_validation": False, "causal_feature_updates": True, "clipping": False, "selection_metric": "mean_h(MAE_model_h / MAE_persistence_h)", "practical_tie_relative_threshold": PRACTICAL_TIE_RELATIVE, "family_tie_break": ["total_fit_seconds","total_serialized_model_bytes","candidate_id"], "tree_tie_break": ["total_fit_seconds","total_serialized_model_bytes"], "serialized_model_size_definition": "sum of serialized bytes of H1/H3/H6/H12", "candidate_concurrency": 1, "hard_timeout_per_fit_seconds": FIT_TIMEOUT_SECONDS},
            "random_seed": CANONICAL_SEED, "n_jobs": N_JOBS, "survivor_configs": SURVIVOR_CONFIGS, "split_records": split_records, "family_winners": winners.to_dict(orient="records"), "tree_winner": tree.to_dict(orient="records"), "validation": checks,
            "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": package_version("scikit-learn"), "xgboost": package_version("xgboost-cpu"), "lightgbm": package_version("lightgbm"), "pyarrow": package_version("pyarrow"), "platform": platform.platform()},
            "processing_seconds": elapsed, "tree_freeze_completed": False, "june_blind_authorized": False, "next_step": "TREE-FREEZE-001",
        })
        save_outputs(outputs, grid, metrics, scores, winners, tree, runtime, predictions, report, manifest)
        print("\n"+report); print(f"Manifiesto: {outputs.manifest_json.resolve()}"); print("TREE-APRIL-SELECTION-001: PASS"); print("TREE-FREEZE-001: PENDING"); print("TREE-JUNE-BLIND-001: NOT AUTHORIZED")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc.__class__.__name__}: {exc}", file=sys.stderr); return 1

if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
