#!/usr/bin/env python3
"""
RECURRENT-STABILITY-001
Phase D — stability characterization of the April-selected recurrent representatives.

Runner filename:
run_recurrent_stability_v1_2026-08-20.py

Internal version:
1.0.0

Binding authorities:
- 023_recurrent_networks_protocol_2026-08-20.md
- 025_recurrent_april_selection_closure_2026-08-20.md

Governance:
- April = stability characterization only.
- Representatives are frozen and cannot change.
- Seeds = 20260820, 20260821, 20260822.
- Canonical seed remains 20260820.
- No best-seed selection.
- June must be physically absent.
- --preflight-only performs 0 training fits and writes 0 campaign artifacts.
"""

from __future__ import annotations

# --- PHASE 1D: portable public-repository path bootstrap ---
import os as _tfm_os
from pathlib import Path as _TFMPath


def _tfm_public_find_repo_root() -> _TFMPath:
    """Resolve the repository root without depending on a personal absolute path."""
    _start = _TFMPath(__file__).resolve().parent
    for _candidate in (_start, *_start.parents):
        if (_candidate / "src").is_dir() and (_candidate / "results").is_dir():
            return _candidate
    return _start


_TFM_PUBLIC_PROJECT_ROOT = _TFMPath(
    _tfm_os.environ.get("TFM_PROJECT_ROOT", str(_tfm_public_find_repo_root()))
).expanduser().resolve()
_TFM_PUBLIC_COLAB_WORKDIR = _TFMPath(
    _tfm_os.environ.get(
        "TFM_COLAB_WORKDIR",
        "/content" if _TFMPath("/content").is_dir() else str(_TFM_PUBLIC_PROJECT_ROOT),
    )
).expanduser().resolve()
# --- end PHASE 1D bootstrap ---


import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RUNNER_VERSION = "1.0.0"
CAMPAIGN_ID = "RECURRENT-STABILITY-001"
PARENT_CAMPAIGN_ID = "UGR16-RECURRENT-NETWORKS-001"
FILE_TAG = "v1_2026-08-20"

PROTOCOL_NAME = "023_recurrent_networks_protocol_2026-08-20.md"
PROTOCOL_SHA = "af9d0d5984fd5e98c6660227f7bd98e246dbf934ad8df416439c2fd5bb59b174"

APRIL_CLOSURE_NAME = "025_recurrent_april_selection_closure_2026-08-20.md"
APRIL_CLOSURE_SHA = "21c14840ea6a67fba52f6dbb114221e0b48cc62dafb1a2039201f37829093cb2"

REPRESENTATIVES_NAME = "ugr16_recurrent_april_selection_v1_2026-08-20_representatives.csv"
REPRESENTATIVES_SHA = "1cb3ee5ba20bb7e11fef2b736db50adb5edf515bc815a0396df5096b08e73a1b"

APRIL_NAME = "april_week3_prepared_5min.parquet"
APRIL_SHA = "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0"
APRIL_ROWS = 2004
APRIL_START = "2016-04-11 01:00:00"
APRIL_END = "2016-04-17 23:55:00"

TARGET = "bitrate_bps"
TIME_COL = "timestamp"
L_MAX = 288
HORIZONS = (1, 3, 6, 12)

CANONICAL_SEED = 20260820
STABILITY_SEEDS = (20260820, 20260821, 20260822)

GAP_1 = 12
GAP_2 = 12
FIT_FRACTION = 0.70
EARLY_STOP_FRACTION = 0.15

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 0.0
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
HARD_FIT_SECONDS = 600.0

EXPECTED_SOFTWARE = {
    "python": "3.12.13",
    "torch": "2.11.0+cu128",
    "torch_cuda": "12.8",
    "cudnn": 91900,
    "numpy": "2.0.2",
    "pandas": "2.2.3",
}

REQUIRED_COLUMNS = {
    "timestamp", "flows_total", "packets_total", "bytes_total",
    "bitrate_bps", "packet_rate_pps", "flow_rate_fps",
    "mean_flow_duration", "max_flow_bytes", "max_flow_packets",
    "flows_background", "packets_background", "bytes_background",
    "flows_blacklist", "packets_blacklist", "bytes_blacklist",
    "flows_anomaly", "packets_anomaly", "bytes_anomaly",
    "tcp_flows", "tcp_packets", "tcp_bytes",
    "udp_flows", "udp_packets", "udp_bytes",
    "icmp_flows", "icmp_packets", "icmp_bytes",
    "other_protocol_flows", "other_protocol_packets",
    "other_protocol_bytes", "duration_sum",
}

GRID = {
    "N01": {"lookback": 12,  "hidden_size": 16, "n_layers": 1, "dropout": 0.00},
    "N02": {"lookback": 24,  "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N03": {"lookback": 72,  "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N04": {"lookback": 144, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N05": {"lookback": 288, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N06": {"lookback": 72,  "hidden_size": 64, "n_layers": 2, "dropout": 0.20},
}

FROZEN_REPRESENTATIVES = {
    "SimpleRNN": "N03",
    "LSTM": "N04",
    "GRU": "N06",
}
FROZEN_RECURRENT_STAR = ("GRU", "N06")

EXPECTED_COMMON_COUNTS = {
    1: 1716,
    3: 1714,
    6: 1711,
    12: 1705,
}

EXPECTED_SPLIT_COUNTS = {
    1:  {"fit": 1184, "early_stop": 253, "selection": 255},
    3:  {"fit": 1183, "early_stop": 253, "selection": 254},
    6:  {"fit": 1180, "early_stop": 253, "selection": 254},
    12: {"fit": 1176, "early_stop": 252, "selection": 253},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{name:58s}: {status}" + (f" | {detail}" if detail else ""))
    if not condition:
        raise RuntimeError(f"{name}: FAIL | {detail}")


def exact_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_software() -> dict:
    info = {
        "python": exact_python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available() else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    check("software_python", info["python"] == EXPECTED_SOFTWARE["python"], info["python"])
    check("software_torch", info["torch"] == EXPECTED_SOFTWARE["torch"], info["torch"])
    check("software_torch_cuda", info["torch_cuda"] == EXPECTED_SOFTWARE["torch_cuda"], str(info["torch_cuda"]))
    check("software_cudnn", info["cudnn"] == EXPECTED_SOFTWARE["cudnn"], str(info["cudnn"]))
    check("software_numpy", info["numpy"] == EXPECTED_SOFTWARE["numpy"], info["numpy"])
    check("software_pandas", info["pandas"] == EXPECTED_SOFTWARE["pandas"], info["pandas"])
    check("cuda_available", info["cuda_available"], str(info["gpu_name"]))
    check("cuda_device_count_positive", info["device_count"] >= 1, str(info["device_count"]))
    check("CUBLAS_WORKSPACE_CONFIG", info["cublas_workspace_config"] == ":4096:8",
          str(info["cublas_workspace_config"]))
    return info


def find_forbidden_june(bundle_root: Path) -> list[Path]:
    return [
        p for p in bundle_root.rglob("*")
        if p.is_file() and "june" in p.name.lower()
    ]


def validate_april(path: Path) -> pd.DataFrame:
    check("April exists", path.is_file(), str(path))
    actual_sha = sha256(path)
    check("April SHA-256", actual_sha == APRIL_SHA, actual_sha)

    frame = pd.read_parquet(path)
    check("April rows", len(frame) == APRIL_ROWS, str(len(frame)))
    check("April columns", set(frame.columns) == REQUIRED_COLUMNS, f"{len(frame.columns)} cols")
    check("April target finite", np.isfinite(frame[TARGET].to_numpy(float)).all())

    ts = pd.to_datetime(frame[TIME_COL], errors="raise")
    check("April timestamps unique", not ts.duplicated().any())
    check("April timestamps sorted", ts.is_monotonic_increasing)
    check("April start", str(ts.min()) == APRIL_START, str(ts.min()))
    check("April end", str(ts.max()) == APRIL_END, str(ts.max()))
    check("April spacing 5 min",
          bool((ts.diff().dropna() == pd.Timedelta(minutes=5)).all()))
    return frame


def validate_representatives(path: Path) -> pd.DataFrame:
    check("April representatives exists", path.is_file(), str(path))
    actual_sha = sha256(path)
    check("April representatives SHA-256", actual_sha == REPRESENTATIVES_SHA, actual_sha)

    frame = pd.read_csv(path)
    check("representative rows = 3", len(frame) == 3, str(len(frame)))
    check("family column exists", "family" in frame.columns)
    check("representative_config column exists", "representative_config" in frame.columns)
    check("is_recurrent_star column exists", "is_recurrent_star" in frame.columns)

    observed = dict(
        zip(frame["family"].astype(str), frame["representative_config"].astype(str))
    )
    check("frozen family representatives exact",
          observed == FROZEN_REPRESENTATIVES, str(observed))

    star_rows = frame[frame["is_recurrent_star"].astype(bool)]
    check("exactly one RECURRENT*", len(star_rows) == 1, str(len(star_rows)))
    star = (
        str(star_rows.iloc[0]["family"]),
        str(star_rows.iloc[0]["representative_config"]),
    )
    check("RECURRENT* exact", star == FROZEN_RECURRENT_STAR, str(star))
    return frame


def common_sample_indices(n_rows: int, horizon: int) -> np.ndarray:
    n = n_rows - L_MAX - horizon + 1
    if n <= 0:
        raise RuntimeError("No common samples")
    return np.arange(n, dtype=np.int64)


def origin_from_common_index(i: int) -> int:
    return (L_MAX - 1) + int(i)


def april_split(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_available = n_samples - GAP_1 - GAP_2
    if n_available <= 0:
        raise RuntimeError("Insufficient samples after gaps")

    fit_count = math.floor(FIT_FRACTION * n_available)
    early_count = math.floor(EARLY_STOP_FRACTION * n_available)
    selection_count = n_available - fit_count - early_count

    fit_end = fit_count
    early_start = fit_end + GAP_1
    early_end = early_start + early_count
    selection_start = early_end + GAP_2
    selection_end = selection_start + selection_count

    check("split consumes all common samples", selection_end == n_samples,
          f"{selection_end}/{n_samples}")

    fit_idx = np.arange(0, fit_end, dtype=np.int64)
    early_idx = np.arange(early_start, early_end, dtype=np.int64)
    selection_idx = np.arange(selection_start, selection_end, dtype=np.int64)
    return fit_idx, early_idx, selection_idx


@dataclass
class ScalarStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ScalarStandardizer":
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        scale = float(arr.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(scale):
            raise RuntimeError("Non-finite scaler parameters")
        if scale == 0.0:
            scale = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def build_xy(y: np.ndarray, sample_indices: np.ndarray, lookback: int,
             horizon: int, scaler: ScalarStandardizer):
    xs, ys, origins, targets = [], [], [], []
    for i in sample_indices:
        origin = origin_from_common_index(int(i))
        target_idx = origin + horizon
        start = origin - lookback + 1
        if start < 0 or target_idx >= len(y):
            raise RuntimeError("Invalid supervised window bounds")
        xs.append(
            scaler.transform(y[start:origin + 1])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        ys.append(
            np.float32(
                scaler.transform(np.array([y[target_idx]], dtype=np.float64))[0]
            )
        )
        origins.append(origin)
        targets.append(target_idx)

    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32).reshape(-1, 1),
        np.asarray(origins, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


class RecurrentRegressor(nn.Module):
    def __init__(self, family: str, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        if family == "SimpleRNN":
            recurrent_cls = nn.RNN
        elif family == "LSTM":
            recurrent_cls = nn.LSTM
        elif family == "GRU":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f"Unknown family: {family}")

        self.recurrent = recurrent_cls(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.external_dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        last = sequence[:, -1, :]
        return self.output(self.external_dropout(last))


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_loader(x: np.ndarray, y: np.ndarray) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.L1Loss(reduction="sum")
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        total += float(loss_fn(pred, yb).item())
        n += int(len(xb))
    return total / n


@torch.no_grad()
def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    model.eval()
    chunks = []
    for (xb,) in loader:
        chunks.append(model(xb.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def train_one_fit(
    family: str,
    config_id: str,
    horizon: int,
    seed: int,
    y: np.ndarray,
    fit_idx: np.ndarray,
    early_idx: np.ndarray,
    selection_idx: np.ndarray,
    device: torch.device,
):
    cfg = GRID[config_id]
    lookback = int(cfg["lookback"])

    fit_target_indices = np.asarray(
        [origin_from_common_index(i) + horizon for i in fit_idx],
        dtype=np.int64,
    )
    scaler_cutoff = int(fit_target_indices.max())
    scaler = ScalarStandardizer.fit(y[:scaler_cutoff + 1])

    x_fit, y_fit, _, _ = build_xy(y, fit_idx, lookback, horizon, scaler)
    x_early, y_early, _, _ = build_xy(y, early_idx, lookback, horizon, scaler)
    x_sel, _, origins_sel, targets_sel = build_xy(
        y, selection_idx, lookback, horizon, scaler
    )

    set_seed(seed)
    model = RecurrentRegressor(
        family=family,
        hidden_size=int(cfg["hidden_size"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()
    fit_loader = make_loader(x_fit, y_fit)
    early_loader = make_loader(x_early, y_early)

    best_val = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in fit_loader:
            if time.perf_counter() - started > HARD_FIT_SECONDS:
                raise TimeoutError(
                    f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}/seed={seed}"
                )

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss {family}/{config_id}/H{horizon}/seed={seed}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        validation_mae = eval_loss(model, early_loader, device)

        if validation_mae < (best_val - MIN_DELTA):
            best_val = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    runtime_seconds = time.perf_counter() - started

    if runtime_seconds > HARD_FIT_SECONDS:
        raise TimeoutError(
            f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}/seed={seed}"
        )

    if best_state is None or best_epoch is None:
        raise RuntimeError("No valid early-stopping checkpoint")

    model.load_state_dict(best_state)

    pred_scaled = predict_scaled(model, x_sel, device)
    y_pred = scaler.inverse(pred_scaled)
    y_true = y[targets_sel].astype(np.float64)
    persistence = y[origins_sel].astype(np.float64)

    predictions = pd.DataFrame({
        "family": family,
        "config_id": config_id,
        "seed": int(seed),
        "is_canonical_seed": bool(seed == CANONICAL_SEED),
        "horizon_steps": int(horizon),
        "sample_index": selection_idx.astype(int),
        "origin_index": origins_sel.astype(int),
        "target_index": targets_sel.astype(int),
        "y_true_bps": y_true,
        "y_pred_bps": y_pred,
        "persistence_pred_bps": persistence,
        "residual_bps": y_true - y_pred,
        "abs_error_bps": np.abs(y_true - y_pred),
        "persistence_abs_error_bps": np.abs(y_true - persistence),
    })

    mase_scale = float(np.mean(np.abs(np.diff(y[:scaler_cutoff + 1]))))
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise RuntimeError("Invalid MASE scale")

    runtime_record = {
        "family": family,
        "config_id": config_id,
        "seed": int(seed),
        "is_canonical_seed": bool(seed == CANONICAL_SEED),
        "horizon_steps": int(horizon),
        "lookback": lookback,
        "hidden_size": int(cfg["hidden_size"]),
        "n_layers": int(cfg["n_layers"]),
        "dropout": float(cfg["dropout"]),
        "n_fit_samples": int(len(fit_idx)),
        "n_early_stop_samples": int(len(early_idx)),
        "n_selection_samples": int(len(selection_idx)),
        "scaler_cutoff_index": scaler_cutoff,
        "scaler_mean_bps": scaler.mean,
        "scaler_scale_bps": scaler.scale,
        "mase_scale_bps": mase_scale,
        "best_epoch": int(best_epoch),
        "best_early_stop_mae_scaled": float(best_val),
        "parameter_count": parameter_count(model),
        "runtime_seconds": float(runtime_seconds),
        "status": "PASS",
    }

    return runtime_record, predictions


def summarize_metrics(predictions: pd.DataFrame, runtime_record: pd.Series) -> dict:
    yt = predictions["y_true_bps"].to_numpy(float)
    yp = predictions["y_pred_bps"].to_numpy(float)
    pp = predictions["persistence_pred_bps"].to_numpy(float)

    residual = yt - yp
    abs_error = np.abs(residual)
    persistence_abs = np.abs(yt - pp)

    mae = float(abs_error.mean())
    persistence_mae = float(persistence_abs.mean())
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    denominator = np.abs(yt) + np.abs(yp)
    smape_terms = np.zeros_like(denominator, dtype=float)
    mask = denominator > 0
    smape_terms[mask] = 2.0 * abs_error[mask] / denominator[mask]

    mase_scale = float(runtime_record["mase_scale_bps"])

    return {
        "n_predictions": int(len(predictions)),
        "mae_bps": mae,
        "persistence_mae_bps": persistence_mae,
        "mae_ratio_vs_persistence": float(mae / persistence_mae),
        "skill_vs_persistence": float(1.0 - mae / persistence_mae),
        "rmse_bps": rmse,
        "smape_pct": float(100.0 * smape_terms.mean()),
        "mase": float(mae / mase_scale),
        "bias_bps": float(residual.mean()),
        "underprediction_pct": float(100.0 * np.mean(yt > yp)),
        "p95_abs_error_bps": float(np.percentile(abs_error, 95)),
    }


def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(obj: object, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def preflight(bundle_root: Path):
    print("=" * 122)
    print(f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO")
    print("=" * 122)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {sha256(Path(__file__).resolve())}")
    print()

    software = validate_software()

    june_like = find_forbidden_june(bundle_root)
    check("June physically absent", not june_like, str([str(p) for p in june_like]))

    protocol = bundle_root / "governance" / PROTOCOL_NAME
    closure = bundle_root / "governance" / APRIL_CLOSURE_NAME
    reps_path = bundle_root / "evidence" / REPRESENTATIVES_NAME
    april_path = bundle_root / "inputs" / APRIL_NAME

    check("023 exists", protocol.is_file(), str(protocol))
    check("023 SHA-256", sha256(protocol) == PROTOCOL_SHA, sha256(protocol))

    check("025 exists", closure.is_file(), str(closure))
    check("025 SHA-256", sha256(closure) == APRIL_CLOSURE_SHA, sha256(closure))

    reps = validate_representatives(reps_path)
    april = validate_april(april_path)

    check("stability seeds exact",
          tuple(STABILITY_SEEDS) == (20260820, 20260821, 20260822),
          str(STABILITY_SEEDS))
    check("canonical seed fixed", CANONICAL_SEED == 20260820, str(CANONICAL_SEED))

    for horizon in HORIZONS:
        common = common_sample_indices(len(april), horizon)
        check(
            f"April common count H{horizon}",
            len(common) == EXPECTED_COMMON_COUNTS[horizon],
            str(len(common)),
        )
        fit_idx, early_idx, selection_idx = april_split(len(common))
        expected = EXPECTED_SPLIT_COUNTS[horizon]
        check(f"H{horizon} fit count",
              len(fit_idx) == expected["fit"], str(len(fit_idx)))
        check(f"H{horizon} early-stop count",
              len(early_idx) == expected["early_stop"], str(len(early_idx)))
        check(f"H{horizon} selection count",
              len(selection_idx) == expected["selection"], str(len(selection_idx)))
        check(f"H{horizon} gap 1",
              int(early_idx[0] - fit_idx[-1] - 1) == GAP_1)
        check(f"H{horizon} gap 2",
              int(selection_idx[0] - early_idx[-1] - 1) == GAP_2)

    set_seed(CANONICAL_SEED)
    device = torch.device("cuda")
    x = torch.randn(4, 12, 1, device=device)
    for family, config_id in FROZEN_REPRESENTATIVES.items():
        cfg = GRID[config_id]
        model = RecurrentRegressor(
            family,
            int(cfg["hidden_size"]),
            int(cfg["n_layers"]),
            float(cfg["dropout"]),
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        check(f"{family}/{config_id} forward smoke",
              tuple(out.shape) == (4, 1), str(tuple(out.shape)))

    print("TRAINING EXECUTED IN PREFLIGHT: 0")
    print("ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")

    return software, april, reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/recurrent_stability_v1_2026-08-20_outputs')),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)

    software, april, reps = preflight(bundle_root)

    if args.preflight_only:
        return 0

    output_names = {
        "runtime": f"ugr16_recurrent_stability_{FILE_TAG}_runtime.csv",
        "predictions": f"ugr16_recurrent_stability_{FILE_TAG}_predictions.parquet",
        "horizon_metrics": f"ugr16_recurrent_stability_{FILE_TAG}_horizon_metrics.csv",
        "seed_scores": f"ugr16_recurrent_stability_{FILE_TAG}_seed_scores.csv",
        "summary": f"ugr16_recurrent_stability_{FILE_TAG}_summary.csv",
        "report": f"ugr16_recurrent_stability_{FILE_TAG}_report.txt",
        "manifest": f"ugr16_recurrent_stability_{FILE_TAG}_manifest.json",
        "results_zip": f"recurrent_stability_001_runner_results_{FILE_TAG}.zip",
    }
    paths = {k: output_dir / v for k, v in output_names.items()}

    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing outputs; refusing overwrite:\n" +
            "\n".join(str(p) for p in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    y = april[TARGET].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(april[TIME_COL]).reset_index(drop=True)
    device = torch.device("cuda")

    runtime_records = []
    prediction_frames = []
    campaign_started = time.perf_counter()

    print()
    print("=" * 122)
    print(f"START {CAMPAIGN_ID}")
    print("=" * 122)
    print("Expected fits = 3 representatives × 4 horizons × 3 seeds = 36")
    print("No best-seed selection. Representatives are immutable.")
    print("June absent.")
    print()

    for family, config_id in FROZEN_REPRESENTATIVES.items():
        for seed in STABILITY_SEEDS:
            for horizon in HORIZONS:
                common = common_sample_indices(len(april), horizon)
                fit_idx, early_idx, selection_idx = april_split(len(common))

                print(
                    f"[FIT] {family:10s} {config_id} "
                    f"seed={seed} H{horizon:<2d} "
                    f"fit={len(fit_idx)} early={len(early_idx)} "
                    f"selection={len(selection_idx)}",
                    flush=True,
                )

                rr, preds = train_one_fit(
                    family=family,
                    config_id=config_id,
                    horizon=horizon,
                    seed=seed,
                    y=y,
                    fit_idx=fit_idx,
                    early_idx=early_idx,
                    selection_idx=selection_idx,
                    device=device,
                )

                preds["origin_timestamp"] = timestamps.iloc[
                    preds["origin_index"].to_numpy(int)
                ].to_numpy()
                preds["target_timestamp"] = timestamps.iloc[
                    preds["target_index"].to_numpy(int)
                ].to_numpy()

                runtime_records.append(rr)
                prediction_frames.append(preds)

                print(
                    f"      PASS best_epoch={rr['best_epoch']} "
                    f"runtime={rr['runtime_seconds']:.2f}s",
                    flush=True,
                )

    runtime = pd.DataFrame(runtime_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    check("fit count 36", len(runtime) == 36, str(len(runtime)))
    check("all fit status PASS", runtime["status"].eq("PASS").all())
    check("predictions finite",
          np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all())
    check("April hash unchanged",
          sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA)
    check("023 hash unchanged",
          sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA)
    check("025 hash unchanged",
          sha256(bundle_root / "governance" / APRIL_CLOSURE_NAME) == APRIL_CLOSURE_SHA)
    check("representatives hash unchanged",
          sha256(bundle_root / "evidence" / REPRESENTATIVES_NAME) == REPRESENTATIVES_SHA)
    check("June still absent", not find_forbidden_june(bundle_root))

    horizon_rows = []
    for (family, config_id, seed, horizon), sub in predictions.groupby(
        ["family", "config_id", "seed", "horizon_steps"],
        sort=True,
    ):
        rr = runtime[
            runtime["family"].eq(family)
            & runtime["config_id"].eq(config_id)
            & runtime["seed"].eq(seed)
            & runtime["horizon_steps"].eq(horizon)
        ].iloc[0]
        horizon_rows.append({
            "family": family,
            "config_id": config_id,
            "seed": int(seed),
            "is_canonical_seed": bool(seed == CANONICAL_SEED),
            "horizon_steps": int(horizon),
            **summarize_metrics(sub, rr),
        })

    horizon_metrics = pd.DataFrame(horizon_rows)
    check("horizon metric rows = 36", len(horizon_metrics) == 36,
          str(len(horizon_metrics)))

    seed_score_rows = []
    for (family, config_id, seed), sub in horizon_metrics.groupby(
        ["family", "config_id", "seed"],
        sort=True,
    ):
        ratios = {
            int(r.horizon_steps): float(r.mae_ratio_vs_persistence)
            for r in sub.itertuples(index=False)
        }
        check(f"{family}/{config_id}/seed={seed}: four horizons",
              set(ratios) == set(HORIZONS), str(sorted(ratios)))
        score = float(np.mean([ratios[h] for h in HORIZONS]))
        seed_score_rows.append({
            "family": family,
            "config_id": config_id,
            "seed": int(seed),
            "is_canonical_seed": bool(seed == CANONICAL_SEED),
            "score": score,
            "ratio_H1": ratios[1],
            "ratio_H3": ratios[3],
            "ratio_H6": ratios[6],
            "ratio_H12": ratios[12],
        })

    seed_scores = pd.DataFrame(seed_score_rows)
    check("seed score rows = 9", len(seed_scores) == 9, str(len(seed_scores)))
    check("canonical rows = 3",
          int(seed_scores["is_canonical_seed"].sum()) == 3,
          str(int(seed_scores["is_canonical_seed"].sum())))

    summary_rows = []
    for (family, config_id), sub in seed_scores.groupby(
        ["family", "config_id"], sort=True
    ):
        scores = sub["score"].to_numpy(float)
        canonical_score = float(
            sub[sub["seed"].eq(CANONICAL_SEED)]["score"].iloc[0]
        )
        summary_rows.append({
            "family": family,
            "config_id": config_id,
            "canonical_seed": CANONICAL_SEED,
            "canonical_score": canonical_score,
            "mean_score_3seeds": float(scores.mean()),
            "std_score_3seeds_ddof0": float(scores.std(ddof=0)),
            "min_score_3seeds": float(scores.min()),
            "max_score_3seeds": float(scores.max()),
            "range_score_3seeds": float(scores.max() - scores.min()),
            "representative_identity_immutable": True,
        })

    summary = pd.DataFrame(summary_rows)
    check("stability summary rows = 3", len(summary) == 3, str(len(summary)))

    # No ranking or best-seed decisions are created.
    check("no best seed column",
          "best_seed" not in seed_scores.columns and "best_seed" not in summary.columns)
    check("representatives remain exact",
          dict(zip(summary["family"], summary["config_id"])) == FROZEN_REPRESENTATIVES)

    campaign_seconds = time.perf_counter() - campaign_started

    report_lines = [
        "=" * 122,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 122,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"023 SHA-256: {PROTOCOL_SHA}",
        f"025 SHA-256: {APRIL_CLOSURE_SHA}",
        "",
        "GOVERNANCE",
        "Purpose: stability characterization only",
        f"Seeds: {list(STABILITY_SEEDS)}",
        f"Canonical seed remains: {CANONICAL_SEED}",
        "Best-seed selection: NO",
        "Representative changes: NO",
        "June present: NO",
        "",
        "FROZEN REPRESENTATIVES",
        "RNN*       = SimpleRNN/N03",
        "LSTM*      = LSTM/N04",
        "GRU*       = GRU/N06",
        "RECURRENT* = GRU/N06",
        "",
        "SEED SCORES",
    ]

    for row in seed_scores.sort_values(
        ["family", "seed"], kind="mergesort"
    ).itertuples(index=False):
        report_lines.append(
            f"{row.family:10s} {row.config_id} "
            f"seed={row.seed} score={row.score:.9f} "
            f"H1={row.ratio_H1:.9f} H3={row.ratio_H3:.9f} "
            f"H6={row.ratio_H6:.9f} H12={row.ratio_H12:.9f}"
        )

    report_lines += [
        "",
        "STABILITY SUMMARY",
    ]

    for row in summary.sort_values("family").itertuples(index=False):
        report_lines.append(
            f"{row.family:10s} {row.config_id} "
            f"canonical={row.canonical_score:.9f} "
            f"mean={row.mean_score_3seeds:.9f} "
            f"std={row.std_score_3seeds_ddof0:.9f} "
            f"range={row.range_score_3seeds:.9f}"
        )

    report_lines += [
        "",
        f"Fits executed: {len(runtime)}",
        f"Campaign seconds: {campaign_seconds:.3f}",
        "STATUS: PASS",
        "REPRESENTATIVES CHANGED: NO",
        "BEST SEED SELECTED: NO",
        "RECURRENT-FREEZE-001: NOT RUN",
        "JUNE: NOT PRESENT / NOT RUN",
    ]
    report = "\n".join(report_lines) + "\n"

    atomic_csv(runtime, paths["runtime"])
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(horizon_metrics, paths["horizon_metrics"])
    atomic_csv(seed_scores, paths["seed_scores"])
    atomic_csv(summary, paths["summary"])
    atomic_text(report, paths["report"])

    output_records = {}
    for key in ("runtime", "predictions", "horizon_metrics",
                "seed_scores", "summary", "report"):
        p = paths[key]
        output_records[key] = {
            "name": p.name,
            "sha256": sha256(p),
            "bytes": int(p.stat().st_size),
        }

    validation = {
        "fit_count_36": len(runtime) == 36,
        "all_fit_status_pass": bool(runtime["status"].eq("PASS").all()),
        "horizon_metric_rows_36": len(horizon_metrics) == 36,
        "seed_score_rows_9": len(seed_scores) == 9,
        "summary_rows_3": len(summary) == 3,
        "canonical_rows_3": int(seed_scores["is_canonical_seed"].sum()) == 3,
        "no_best_seed_selection":
            "best_seed" not in seed_scores.columns and "best_seed" not in summary.columns,
        "representatives_unchanged":
            dict(zip(summary["family"], summary["config_id"])) == FROZEN_REPRESENTATIVES,
        "june_absent_after_run": not bool(find_forbidden_june(bundle_root)),
        "april_hash_unchanged":
            sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA,
        "protocol_hash_unchanged":
            sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA,
        "april_closure_hash_unchanged":
            sha256(bundle_root / "governance" / APRIL_CLOSURE_NAME) == APRIL_CLOSURE_SHA,
        "representatives_hash_unchanged":
            sha256(bundle_root / "evidence" / REPRESENTATIVES_NAME) == REPRESENTATIVES_SHA,
    }
    check("all final validations PASS", all(validation.values()), str(validation))

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "runner": {
            "name": runner_path.name,
            "version": RUNNER_VERSION,
            "sha256": runner_sha,
        },
        "authorities": {
            "023": {"name": PROTOCOL_NAME, "sha256": PROTOCOL_SHA},
            "025": {"name": APRIL_CLOSURE_NAME, "sha256": APRIL_CLOSURE_SHA},
        },
        "inputs": {
            "april": {
                "name": APRIL_NAME,
                "sha256": APRIL_SHA,
                "rows": APRIL_ROWS,
            },
            "representatives": {
                "name": REPRESENTATIVES_NAME,
                "sha256": REPRESENTATIVES_SHA,
            },
        },
        "software": software,
        "governance": {
            "purpose": "stability_characterization_only",
            "canonical_seed": CANONICAL_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "best_seed_selection": False,
            "representative_changes_allowed": False,
            "representatives_changed": False,
            "june_present": False,
            "freeze_run": False,
        },
        "representatives": {
            "RNN_star": {"family": "SimpleRNN", "config_id": "N03"},
            "LSTM_star": {"family": "LSTM", "config_id": "N04"},
            "GRU_star": {"family": "GRU", "config_id": "N06"},
            "RECURRENT_star": {"family": "GRU", "config_id": "N06"},
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "L1Loss",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "gradient_clipping": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "shuffle": False,
            "hard_fit_seconds": HARD_FIT_SECONDS,
            "gap_1": GAP_1,
            "gap_2": GAP_2,
            "fit_fraction": FIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
        },
        "validation": validation,
        "campaign_seconds": float(campaign_seconds),
        "outputs": output_records,
    }

    atomic_json(manifest, paths["manifest"])

    results_zip_path = paths["results_zip"]
    if results_zip_path.exists():
        results_zip_path.unlink()

    with zipfile.ZipFile(
        results_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for key in (
            "runtime", "predictions", "horizon_metrics", "seed_scores",
            "summary", "report", "manifest"
        ):
            p = paths[key]
            zf.write(p, arcname=p.name)

    print()
    print(report)
    print("VALIDATIONS")
    print("-" * 122)
    for key, value in validation.items():
        print(f"{key:58s}: {'PASS' if value else 'FAIL'}")

    print()
    print(f"Results ZIP: {results_zip_path}")
    print(f"Results ZIP SHA-256: {sha256(results_zip_path)}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("REPRESENTATIVES CHANGED: NO")
    print("BEST SEED SELECTED: NO")
    print("RECURRENT-FREEZE-001: NOT RUN")
    print("JUNE: NOT PRESENT / NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
