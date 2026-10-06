#!/usr/bin/env python3
"""
RECURRENT-APRIL-SELECTION-001
Phase D — April selection for frozen March top-2 recurrent candidates.

Runner filename:
run_recurrent_april_selection_v1_2026-08-20.py

Internal version:
1.0.0

Binding authorities:
- 023_recurrent_networks_protocol_2026-08-20.md
- 024_recurrent_march_screen_closure_2026-08-20.md

Governance:
- April = selection/validation.
- Only the six March-frozen candidates are permitted.
- June must be physically absent.
- No new candidate generation.
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

# Must be set before importing/initializing torch/CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RUNNER_VERSION = "1.0.0"
CAMPAIGN_ID = "RECURRENT-APRIL-SELECTION-001"
PARENT_CAMPAIGN_ID = "UGR16-RECURRENT-NETWORKS-001"

DATE_TAG = "2026-08-20"
FILE_TAG = "v1_2026-08-20"

PROTOCOL_NAME = "023_recurrent_networks_protocol_2026-08-20.md"
PROTOCOL_SHA = "af9d0d5984fd5e98c6660227f7bd98e246dbf934ad8df416439c2fd5bb59b174"

MARCH_CLOSURE_NAME = "024_recurrent_march_screen_closure_2026-08-20.md"
MARCH_CLOSURE_SHA = "f877831a0078eb1dd57bbc10e245e2bb19064d4da24630c3853b4b4e58399c9f"

MARCH_CANDIDATE_SUMMARY_NAME = "ugr16_recurrent_march_screen_candidate_summary.csv"
MARCH_CANDIDATE_SUMMARY_SHA = "6f0ec5b88e8146f704dca35322ad2dbe9659c2d35382174ddd24488d98b0df8d"

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

FROZEN_CANDIDATES = {
    "SimpleRNN": ("N03", "N04"),
    "LSTM": ("N02", "N04"),
    "GRU": ("N06", "N02"),
}

FAMILY_TIE_PRIORITY = {
    "SimpleRNN": 0,
    "GRU": 1,
    "LSTM": 2,
}

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
    print(f"{name:54s}: {status}" + (f" | {detail}" if detail else ""))
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
    check(
        "April spacing 5 min",
        bool((ts.diff().dropna() == pd.Timedelta(minutes=5)).all())
    )
    return frame


def validate_march_candidate_summary(path: Path) -> pd.DataFrame:
    check("March candidate summary exists", path.is_file(), str(path))
    actual_sha = sha256(path)
    check("March candidate summary SHA-256",
          actual_sha == MARCH_CANDIDATE_SUMMARY_SHA, actual_sha)

    frame = pd.read_csv(path)
    check("March candidate summary rows", len(frame) == 18, str(len(frame)))
    check("selected_top2 column", "selected_top2" in frame.columns)

    selected = frame[frame["selected_top2"].astype(bool)].copy()
    check("March selected rows = 6", len(selected) == 6, str(len(selected)))

    observed = {
        family: tuple(
            selected[selected["family"].eq(family)]
            .sort_values(["rank_within_family", "config_id"])["config_id"]
            .astype(str)
            .tolist()
        )
        for family in FROZEN_CANDIDATES
    }

    for family, expected in FROZEN_CANDIDATES.items():
        check(
            f"Frozen candidates {family}",
            observed.get(family) == expected,
            f"observed={observed.get(family)}, expected={expected}",
        )
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

    fit_start = 0
    fit_end = fit_start + fit_count

    gap1_start = fit_end
    gap1_end = gap1_start + GAP_1

    early_start = gap1_end
    early_end = early_start + early_count

    gap2_start = early_end
    gap2_end = gap2_start + GAP_2

    selection_start = gap2_end
    selection_end = selection_start + selection_count

    check("split consumes all common samples", selection_end == n_samples,
          f"{selection_end}/{n_samples}")

    fit_idx = np.arange(fit_start, fit_end, dtype=np.int64)
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
        x_raw = y[start:origin + 1]
        target_raw = y[target_idx]
        xs.append(scaler.transform(x_raw).astype(np.float32).reshape(-1, 1))
        ys.append(np.float32(scaler.transform(np.array([target_raw]))[0]))
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
    y: np.ndarray,
    fit_idx: np.ndarray,
    early_idx: np.ndarray,
    selection_idx: np.ndarray,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    cfg = GRID[config_id]
    lookback = int(cfg["lookback"])

    # Scaler is fitted using raw observations available only through the latest
    # supervised training target.
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

    set_seed(CANONICAL_SEED)
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
                    f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}"
                )

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss {family}/{config_id}/H{horizon}"
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
        raise TimeoutError(f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}")

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
        "horizon_steps": horizon,
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
        "horizon_steps": horizon,
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


def exact_tie(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def select_family_representative(frame: pd.DataFrame, family: str) -> pd.Series:
    sub = frame[frame["family"].eq(family)].copy()
    check(f"{family}: exactly two candidates", len(sub) == 2, str(len(sub)))

    sub = sub.sort_values(["score", "config_id"], kind="mergesort")
    first = sub.iloc[0]
    second = sub.iloc[1]

    if exact_tie(first["score"], second["score"]):
        return sub.sort_values("config_id", kind="mergesort").iloc[0]

    return first


def select_recurrent_star(representatives: pd.DataFrame) -> pd.Series:
    ordered = representatives.sort_values(
        ["score", "parameter_count", "family_priority", "family"],
        kind="mergesort",
    )

    best_score = float(ordered.iloc[0]["score"])
    tie_group = ordered[
        np.abs(ordered["score"].astype(float) - best_score) <= 1e-12
    ].copy()

    if len(tie_group) == 1:
        return tie_group.iloc[0]

    min_params = int(tie_group["parameter_count"].min())
    tie_group = tie_group[tie_group["parameter_count"].eq(min_params)].copy()

    if len(tie_group) == 1:
        return tie_group.iloc[0]

    tie_group = tie_group.sort_values(
        ["family_priority", "family"], kind="mergesort"
    )
    return tie_group.iloc[0]


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
    print("=" * 120)
    print(f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO")
    print("=" * 120)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {sha256(Path(__file__).resolve())}")
    print()

    software = validate_software()

    june_like = find_forbidden_june(bundle_root)
    check("June physically absent", not june_like, str([str(p) for p in june_like]))

    protocol = bundle_root / "governance" / PROTOCOL_NAME
    closure = bundle_root / "governance" / MARCH_CLOSURE_NAME
    evidence = bundle_root / "evidence" / MARCH_CANDIDATE_SUMMARY_NAME
    april_path = bundle_root / "inputs" / APRIL_NAME

    check("023 exists", protocol.is_file(), str(protocol))
    check("023 SHA-256", sha256(protocol) == PROTOCOL_SHA, sha256(protocol))

    check("024 exists", closure.is_file(), str(closure))
    check("024 SHA-256", sha256(closure) == MARCH_CLOSURE_SHA, sha256(closure))

    march_candidates = validate_march_candidate_summary(evidence)
    april = validate_april(april_path)

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

        check(
            f"H{horizon} gap 1",
            int(early_idx[0] - fit_idx[-1] - 1) == GAP_1,
            str(int(early_idx[0] - fit_idx[-1] - 1)),
        )
        check(
            f"H{horizon} gap 2",
            int(selection_idx[0] - early_idx[-1] - 1) == GAP_2,
            str(int(selection_idx[0] - early_idx[-1] - 1)),
        )

    set_seed(CANONICAL_SEED)
    device = torch.device("cuda")
    x = torch.randn(4, 12, 1, device=device)

    for family in FROZEN_CANDIDATES:
        model = RecurrentRegressor(family, 8, 1, 0.0).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        check(f"{family} forward smoke", tuple(out.shape) == (4, 1),
              str(tuple(out.shape)))

    print("TRAINING EXECUTED IN PREFLIGHT: 0")
    print("ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")

    return software, april, march_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/recurrent_april_selection_v1_2026-08-20_outputs')),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)

    software, april, march_candidates = preflight(bundle_root)

    if args.preflight_only:
        return 0

    output_names = {
        "runtime": f"ugr16_recurrent_april_selection_{FILE_TAG}_runtime.csv",
        "predictions": f"ugr16_recurrent_april_selection_{FILE_TAG}_predictions.parquet",
        "horizon_metrics": f"ugr16_recurrent_april_selection_{FILE_TAG}_horizon_metrics.csv",
        "candidate_summary": f"ugr16_recurrent_april_selection_{FILE_TAG}_candidate_summary.csv",
        "representatives": f"ugr16_recurrent_april_selection_{FILE_TAG}_representatives.csv",
        "report": f"ugr16_recurrent_april_selection_{FILE_TAG}_report.txt",
        "manifest": f"ugr16_recurrent_april_selection_{FILE_TAG}_manifest.json",
        "results_zip": f"recurrent_april_selection_001_runner_results_{FILE_TAG}.zip",
    }

    expected_paths = {
        key: output_dir / name
        for key, name in output_names.items()
    }

    existing = [p for p in expected_paths.values() if p.exists()]
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
    print("=" * 120)
    print(f"START {CAMPAIGN_ID}")
    print("=" * 120)
    print("Expected fits = 6 candidates × 4 horizons = 24")
    print("June absent. No new candidates permitted.")
    print()

    for family, configs in FROZEN_CANDIDATES.items():
        for config_id in configs:
            for horizon in HORIZONS:
                common = common_sample_indices(len(april), horizon)
                fit_idx, early_idx, selection_idx = april_split(len(common))

                print(
                    f"[FIT] {family:10s} {config_id} H{horizon:<2d} "
                    f"fit={len(fit_idx)} early={len(early_idx)} "
                    f"selection={len(selection_idx)}",
                    flush=True,
                )

                runtime_record, predictions = train_one_fit(
                    family=family,
                    config_id=config_id,
                    horizon=horizon,
                    y=y,
                    fit_idx=fit_idx,
                    early_idx=early_idx,
                    selection_idx=selection_idx,
                    device=device,
                )

                predictions["origin_timestamp"] = timestamps.iloc[
                    predictions["origin_index"].to_numpy(int)
                ].to_numpy()

                predictions["target_timestamp"] = timestamps.iloc[
                    predictions["target_index"].to_numpy(int)
                ].to_numpy()

                runtime_records.append(runtime_record)
                prediction_frames.append(predictions)

                print(
                    f"      PASS best_epoch={runtime_record['best_epoch']} "
                    f"runtime={runtime_record['runtime_seconds']:.2f}s",
                    flush=True,
                )

    runtime = pd.DataFrame(runtime_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    check("fit count 24", len(runtime) == 24, str(len(runtime)))
    check("all fit status PASS", runtime["status"].eq("PASS").all())
    check("predictions finite",
          np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all())
    check("April hash unchanged",
          sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA)
    check("023 hash unchanged",
          sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA)
    check("024 hash unchanged",
          sha256(bundle_root / "governance" / MARCH_CLOSURE_NAME) == MARCH_CLOSURE_SHA)
    check("March candidate summary hash unchanged",
          sha256(bundle_root / "evidence" / MARCH_CANDIDATE_SUMMARY_NAME)
          == MARCH_CANDIDATE_SUMMARY_SHA)
    check("June still absent", not find_forbidden_june(bundle_root))

    horizon_rows = []

    for (family, config_id, horizon), sub in predictions.groupby(
        ["family", "config_id", "horizon_steps"],
        sort=True,
    ):
        rr = runtime[
            runtime["family"].eq(family)
            & runtime["config_id"].eq(config_id)
            & runtime["horizon_steps"].eq(horizon)
        ].iloc[0]

        horizon_rows.append({
            "family": family,
            "config_id": config_id,
            "horizon_steps": int(horizon),
            **summarize_metrics(sub, rr),
        })

    horizon_metrics = pd.DataFrame(horizon_rows)

    candidate_rows = []

    for (family, config_id), sub in horizon_metrics.groupby(
        ["family", "config_id"],
        sort=True,
    ):
        ratios = {
            int(r.horizon_steps): float(r.mae_ratio_vs_persistence)
            for r in sub.itertuples(index=False)
        }

        check(
            f"{family}/{config_id}: four horizons",
            set(ratios) == set(HORIZONS),
            str(sorted(ratios)),
        )

        representative_runtime = runtime[
            runtime["family"].eq(family)
            & runtime["config_id"].eq(config_id)
        ]

        parameter_values = representative_runtime["parameter_count"].unique()
        check(
            f"{family}/{config_id}: parameter count stable",
            len(parameter_values) == 1,
            str(parameter_values),
        )

        score = float(np.mean([ratios[h] for h in HORIZONS]))

        candidate_rows.append({
            "family": family,
            "config_id": config_id,
            "score": score,
            "ratio_H1": ratios[1],
            "ratio_H3": ratios[3],
            "ratio_H6": ratios[6],
            "ratio_H12": ratios[12],
            "parameter_count": int(parameter_values[0]),
        })

    candidate_summary = pd.DataFrame(candidate_rows)

    check("candidate rows 6", len(candidate_summary) == 6, str(len(candidate_summary)))

    representative_rows = []

    for family in FROZEN_CANDIDATES:
        winner = select_family_representative(candidate_summary, family)

        representative_rows.append({
            "family": family,
            "representative_config": str(winner["config_id"]),
            "score": float(winner["score"]),
            "parameter_count": int(winner["parameter_count"]),
            "family_priority": FAMILY_TIE_PRIORITY[family],
        })

    representatives = pd.DataFrame(representative_rows)

    recurrent_star = select_recurrent_star(representatives)

    representatives["is_recurrent_star"] = (
        representatives["family"].eq(str(recurrent_star["family"]))
    )

    check("representative rows 3", len(representatives) == 3, str(len(representatives)))
    check("exactly one RECURRENT*", int(representatives["is_recurrent_star"].sum()) == 1)

    rnn_star = representatives[
        representatives["family"].eq("SimpleRNN")
    ].iloc[0]["representative_config"]

    lstm_star = representatives[
        representatives["family"].eq("LSTM")
    ].iloc[0]["representative_config"]

    gru_star = representatives[
        representatives["family"].eq("GRU")
    ].iloc[0]["representative_config"]

    recurrent_family = str(recurrent_star["family"])
    recurrent_config = str(recurrent_star["representative_config"])

    campaign_seconds = time.perf_counter() - campaign_started

    report_lines = [
        "=" * 120,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 120,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"023 SHA-256: {PROTOCOL_SHA}",
        f"024 SHA-256: {MARCH_CLOSURE_SHA}",
        "",
        "GOVERNANCE",
        "April role: selection / validation",
        "Frozen March candidates only: YES",
        "June present: NO",
        "New candidates created: NO",
        f"Canonical seed: {CANONICAL_SEED}",
        "",
        "CANDIDATE SCORES",
    ]

    for row in candidate_summary.sort_values(
        ["family", "score", "config_id"],
        kind="mergesort",
    ).itertuples(index=False):
        report_lines.append(
            f"{row.family:10s} {row.config_id} "
            f"score={row.score:.9f} "
            f"H1={row.ratio_H1:.9f} "
            f"H3={row.ratio_H3:.9f} "
            f"H6={row.ratio_H6:.9f} "
            f"H12={row.ratio_H12:.9f}"
        )

    report_lines += [
        "",
        "REPRESENTATIVES SELECTED ON APRIL",
        f"RNN*       = SimpleRNN/{rnn_star}",
        f"LSTM*      = LSTM/{lstm_star}",
        f"GRU*       = GRU/{gru_star}",
        f"RECURRENT* = {recurrent_family}/{recurrent_config}",
        "",
        f"Fits executed: {len(runtime)}",
        f"Campaign seconds: {campaign_seconds:.3f}",
        "STATUS: PASS",
        "RECURRENT-STABILITY-001: NOT RUN",
        "RECURRENT-FREEZE-001: NOT RUN",
        "JUNE: NOT PRESENT / NOT RUN",
    ]

    report = "\n".join(report_lines) + "\n"

    atomic_csv(runtime, expected_paths["runtime"])
    atomic_parquet(predictions, expected_paths["predictions"])
    atomic_csv(horizon_metrics, expected_paths["horizon_metrics"])
    atomic_csv(candidate_summary, expected_paths["candidate_summary"])
    atomic_csv(representatives, expected_paths["representatives"])
    atomic_text(report, expected_paths["report"])

    output_records = {}

    for key in (
        "runtime",
        "predictions",
        "horizon_metrics",
        "candidate_summary",
        "representatives",
        "report",
    ):
        path = expected_paths[key]
        output_records[key] = {
            "name": path.name,
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
        }

    validation = {
        "fit_count_24": len(runtime) == 24,
        "all_fit_status_pass": bool(runtime["status"].eq("PASS").all()),
        "candidate_rows_6": len(candidate_summary) == 6,
        "representative_rows_3": len(representatives) == 3,
        "exactly_one_recurrent_star": int(representatives["is_recurrent_star"].sum()) == 1,
        "june_absent_after_run": not bool(find_forbidden_june(bundle_root)),
        "april_hash_unchanged":
            sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA,
        "protocol_hash_unchanged":
            sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA,
        "march_closure_hash_unchanged":
            sha256(bundle_root / "governance" / MARCH_CLOSURE_NAME) == MARCH_CLOSURE_SHA,
        "march_candidate_summary_hash_unchanged":
            sha256(bundle_root / "evidence" / MARCH_CANDIDATE_SUMMARY_NAME)
            == MARCH_CANDIDATE_SUMMARY_SHA,
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
            "023": {
                "name": PROTOCOL_NAME,
                "sha256": PROTOCOL_SHA,
            },
            "024": {
                "name": MARCH_CLOSURE_NAME,
                "sha256": MARCH_CLOSURE_SHA,
            },
        },
        "inputs": {
            "april": {
                "name": APRIL_NAME,
                "sha256": APRIL_SHA,
                "rows": APRIL_ROWS,
            },
            "march_candidate_summary": {
                "name": MARCH_CANDIDATE_SUMMARY_NAME,
                "sha256": MARCH_CANDIDATE_SUMMARY_SHA,
            },
        },
        "software": software,
        "governance": {
            "april_role": "selection_validation",
            "frozen_candidates_only": True,
            "june_present": False,
            "new_candidates_created": False,
            "canonical_seed": CANONICAL_SEED,
            "stability_run": False,
            "freeze_run": False,
        },
        "frozen_candidates": FROZEN_CANDIDATES,
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
        "selection": {
            "score": "mean_h(MAE_model_h / MAE_persistence_h)",
            "exact_tie_tolerance": 1e-12,
            "family_tie_break": "config_id ascending",
            "recurrent_star_exact_tie_break": [
                "lower parameter_count",
                "SimpleRNN",
                "GRU",
                "LSTM",
            ],
        },
        "representatives": {
            "RNN_star": {
                "family": "SimpleRNN",
                "config_id": str(rnn_star),
            },
            "LSTM_star": {
                "family": "LSTM",
                "config_id": str(lstm_star),
            },
            "GRU_star": {
                "family": "GRU",
                "config_id": str(gru_star),
            },
            "RECURRENT_star": {
                "family": recurrent_family,
                "config_id": recurrent_config,
            },
        },
        "validation": validation,
        "campaign_seconds": float(campaign_seconds),
        "outputs": output_records,
    }

    atomic_json(manifest, expected_paths["manifest"])

    results_zip_path = expected_paths["results_zip"]
    if results_zip_path.exists():
        results_zip_path.unlink()

    with zipfile.ZipFile(
        results_zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for key in (
            "runtime",
            "predictions",
            "horizon_metrics",
            "candidate_summary",
            "representatives",
            "report",
            "manifest",
        ):
            path = expected_paths[key]
            zf.write(path, arcname=path.name)

    print()
    print(report)
    print("VALIDATIONS")
    print("-" * 120)

    for key, value in validation.items():
        print(f"{key:54s}: {'PASS' if value else 'FAIL'}")

    print()
    print(f"Results ZIP: {results_zip_path}")
    print(f"Results ZIP SHA-256: {sha256(results_zip_path)}")
    print(f"{CAMPAIGN_ID}: PASS")
    print(f"RNN*       = SimpleRNN/{rnn_star}")
    print(f"LSTM*      = LSTM/{lstm_star}")
    print(f"GRU*       = GRU/{gru_star}")
    print(f"RECURRENT* = {recurrent_family}/{recurrent_config}")
    print("RECURRENT-STABILITY-001: NOT RUN")
    print("RECURRENT-FREEZE-001: NOT RUN")
    print("JUNE: NOT PRESENT / NOT RUN")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
