#!/usr/bin/env python3
"""
RECURRENT-JUNE-BLIND-001
Phase D — final blind external evaluation of frozen recurrent representatives.

Runner:
run_recurrent_june_blind_v1_2026-08-21.py

Internal version:
1.0.0

Binding authorities:
- 023_recurrent_networks_protocol_2026-08-20.md
- 024_recurrent_march_screen_closure_2026-08-20.md
- 025_recurrent_april_selection_closure_2026-08-20.md
- 026_recurrent_stability_closure_2026-08-20.md
- 027_recurrent_freeze_2026-08-20.md

Governance:
- June = blind external evaluation only.
- Representatives are immutable.
- Canonical seed = 20260820.
- No best-seed selection.
- No tuning or representative changes.
- Inner early stopping uses only the initial June training block.
- Final refit is preplanned and uses only the initial June training block.
- Blind targets never affect model fitting, epoch selection, scaling, or weights.
- --preflight-only performs 0 training stages, 0 blind predictions,
  and writes 0 campaign artifacts.
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
CAMPAIGN_ID = "RECURRENT-JUNE-BLIND-001"
PARENT_CAMPAIGN_ID = "UGR16-RECURRENT-NETWORKS-001"
FILE_TAG = "v1_2026-08-21"

PROTOCOL_NAME = "023_recurrent_networks_protocol_2026-08-20.md"
PROTOCOL_SHA = "af9d0d5984fd5e98c6660227f7bd98e246dbf934ad8df416439c2fd5bb59b174"

MARCH_CLOSURE_NAME = "024_recurrent_march_screen_closure_2026-08-20.md"
MARCH_CLOSURE_SHA = "f877831a0078eb1dd57bbc10e245e2bb19064d4da24630c3853b4b4e58399c9f"

APRIL_CLOSURE_NAME = "025_recurrent_april_selection_closure_2026-08-20.md"
APRIL_CLOSURE_SHA = "21c14840ea6a67fba52f6dbb114221e0b48cc62dafb1a2039201f37829093cb2"

STABILITY_CLOSURE_NAME = "026_recurrent_stability_closure_2026-08-20.md"
STABILITY_CLOSURE_SHA = "329c3068fb60f97789cc3b973ec0a13e4c6770d95706ac2688f8017fa78f289f"

FREEZE_NAME = "027_recurrent_freeze_2026-08-20.md"
FREEZE_SHA = "27d7e96fdbc8575f241638eb817a31f28673cf512d30dfac8efb34835bf41de4"

REPRESENTATIVES_NAME = "ugr16_recurrent_april_selection_v1_2026-08-20_representatives.csv"
REPRESENTATIVES_SHA = "1cb3ee5ba20bb7e11fef2b736db50adb5edf515bc815a0396df5096b08e73a1b"

STABILITY_SUMMARY_NAME = "ugr16_recurrent_stability_v1_2026-08-20_summary.csv"
STABILITY_SUMMARY_SHA = "2d86057330992f73bbee728c904e2f5c549dde6c996fb4247964db9e44c35a82"

JUNE_NAME = "june_week3_prepared_5min.parquet"
JUNE_SHA = "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3"
JUNE_ROWS = 2004
JUNE_START = "2016-06-13 01:00:00"
JUNE_END = "2016-06-19 23:55:00"

BUNDLE_MANIFEST_NAME = "phase_d_recurrent_june_blind_exec_bundle_manifest_v1_2026-08-21.json"

TARGET = "bitrate_bps"
TIME_COL = "timestamp"

TRAIN_ROWS = 1402
TRAIN_START = "2016-06-13 01:00:00"
TRAIN_END = "2016-06-17 21:45:00"
BLIND_START = "2016-06-17 21:50:00"
BLIND_ROWS = 602

L_MAX = 288
HORIZONS = (1, 3, 6, 12)
EXPECTED_BLIND_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
EXPECTED_TOTAL_BLIND_PREDICTIONS = 3 * sum(EXPECTED_BLIND_COUNTS.values())

CANONICAL_SEED = 20260820

INNER_GAP = 12
INNER_EARLY_STOP_FRACTION = 0.20

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 0.0
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
HARD_STAGE_SECONDS = 600.0

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
    "N03": {"lookback": 72,  "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N04": {"lookback": 144, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N06": {"lookback": 72,  "hidden_size": 64, "n_layers": 2, "dropout": 0.20},
}

FROZEN_REPRESENTATIVES = {
    "SimpleRNN": "N03",
    "LSTM": "N04",
    "GRU": "N06",
}
FROZEN_RECURRENT_STAR = ("GRU", "N06")

EXPECTED_TRAIN_COMMON_COUNTS = {
    1: 1114,
    3: 1112,
    6: 1109,
    12: 1103,
}

EXPECTED_INNER_SPLITS = {
    1:  {"fit": 882, "gap": 12, "early_stop": 220},
    3:  {"fit": 880, "gap": 12, "early_stop": 220},
    6:  {"fit": 878, "gap": 12, "early_stop": 219},
    12: {"fit": 873, "gap": 12, "early_stop": 218},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{name:62s}: {status}" + (f" | {detail}" if detail else ""))
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
    check("CUBLAS_WORKSPACE_CONFIG",
          info["cublas_workspace_config"] == ":4096:8",
          str(info["cublas_workspace_config"]))
    return info


def validate_hash_file(path: Path, expected_sha: str, label: str) -> None:
    check(f"{label} exists", path.is_file(), str(path))
    actual = sha256(path)
    check(f"{label} SHA-256", actual == expected_sha, actual)


def validate_june(path: Path) -> pd.DataFrame:
    validate_hash_file(path, JUNE_SHA, "June")
    frame = pd.read_parquet(path)
    check("June rows", len(frame) == JUNE_ROWS, str(len(frame)))
    check("June columns", set(frame.columns) == REQUIRED_COLUMNS, f"{len(frame.columns)} cols")
    check("June target finite", np.isfinite(frame[TARGET].to_numpy(float)).all())

    ts = pd.to_datetime(frame[TIME_COL], errors="raise")
    check("June timestamps unique", not ts.duplicated().any())
    check("June timestamps sorted", ts.is_monotonic_increasing)
    check("June start", str(ts.min()) == JUNE_START, str(ts.min()))
    check("June end", str(ts.max()) == JUNE_END, str(ts.max()))
    check("June spacing 5 min",
          bool((ts.diff().dropna() == pd.Timedelta(minutes=5)).all()))

    check("June training rows frozen", TRAIN_ROWS == 1402, str(TRAIN_ROWS))
    check("June train start", str(ts.iloc[0]) == TRAIN_START, str(ts.iloc[0]))
    check("June train end", str(ts.iloc[TRAIN_ROWS - 1]) == TRAIN_END,
          str(ts.iloc[TRAIN_ROWS - 1]))
    check("June blind start", str(ts.iloc[TRAIN_ROWS]) == BLIND_START,
          str(ts.iloc[TRAIN_ROWS]))
    check("June blind rows", len(frame) - TRAIN_ROWS == BLIND_ROWS,
          str(len(frame) - TRAIN_ROWS))
    return frame


def validate_representatives(path: Path) -> pd.DataFrame:
    validate_hash_file(path, REPRESENTATIVES_SHA, "April representatives")
    frame = pd.read_csv(path)
    check("representative rows = 3", len(frame) == 3, str(len(frame)))
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


def validate_stability_summary(path: Path) -> pd.DataFrame:
    validate_hash_file(path, STABILITY_SUMMARY_SHA, "Stability summary")
    frame = pd.read_csv(path)
    check("stability summary rows = 3", len(frame) == 3, str(len(frame)))
    observed = dict(zip(frame["family"].astype(str), frame["config_id"].astype(str)))
    check("stability identities unchanged", observed == FROZEN_REPRESENTATIVES, str(observed))
    check("no best_seed column in stability summary", "best_seed" not in frame.columns)
    return frame


def validate_bundle_manifest(path: Path, runner_sha: str) -> dict:
    check("bundle manifest exists", path.is_file(), str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    check("bundle manifest id",
          obj.get("bundle_id") == "PHASE-D-RECURRENT-JUNE-BLIND-EXEC-BUNDLE-V1-2026-08-21",
          str(obj.get("bundle_id")))
    check("bundle manifest campaign",
          obj.get("campaign_id") == CAMPAIGN_ID,
          str(obj.get("campaign_id")))
    check("bundle manifest status",
          obj.get("status") == "PRE-EXECUTION / FROZEN_INPUT_BUNDLE",
          str(obj.get("status")))
    check("bundle manifest runner SHA",
          obj["runner"]["sha256"] == runner_sha,
          str(obj["runner"]["sha256"]))
    check("bundle manifest June SHA",
          obj["june"]["sha256"] == JUNE_SHA,
          str(obj["june"]["sha256"]))
    check("bundle manifest canonical seed",
          obj["frozen_configuration"]["canonical_seed"] == CANONICAL_SEED,
          str(obj["frozen_configuration"]["canonical_seed"]))
    check("bundle manifest no training yet",
          obj["pre_execution_assertions"]["training_executed"] is False)
    check("bundle manifest no selection allowed",
          obj["pre_execution_assertions"]["selection_allowed"] is False)
    return obj


def training_common_origins(horizon: int) -> np.ndarray:
    # Common origins use L_MAX and targets must remain inside initial June training block.
    return np.arange(L_MAX - 1, TRAIN_ROWS - horizon, dtype=np.int64)


def inner_split(origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(origins)
    n_available = n - INNER_GAP
    early_count = math.floor(INNER_EARLY_STOP_FRACTION * n_available)
    fit_count = n_available - early_count
    fit = origins[:fit_count]
    early = origins[fit_count + INNER_GAP:]
    check("inner split consumes all permitted samples",
          len(fit) + INNER_GAP + len(early) == n,
          f"{len(fit)}+{INNER_GAP}+{len(early)}={n}")
    return fit, early


def blind_origins(horizon: int) -> np.ndarray:
    # The first common blind forecast origin is the final training observation.
    # Subsequent real blind observations may become lag inputs once revealed.
    return np.arange(TRAIN_ROWS - 1, JUNE_ROWS - horizon, dtype=np.int64)


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


def build_xy_from_origins(
    y: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    horizon: int,
    scaler: ScalarStandardizer,
    with_targets: bool = True,
):
    xs, ys, targets = [], [], []
    for origin in origins:
        origin = int(origin)
        target_idx = origin + horizon
        start = origin - lookback + 1
        if start < 0 or target_idx >= len(y):
            raise RuntimeError("Invalid supervised window bounds")

        # Causal input only: observations at or before forecast origin.
        xs.append(
            scaler.transform(y[start:origin + 1])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        targets.append(target_idx)

        if with_targets:
            ys.append(
                np.float32(
                    scaler.transform(np.array([y[target_idx]], dtype=np.float64))[0]
                )
            )

    x = np.stack(xs).astype(np.float32)
    target_idx = np.asarray(targets, dtype=np.int64)
    if with_targets:
        yy = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    else:
        yy = None
    return x, yy, target_idx


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


def select_best_epoch(
    family: str,
    config_id: str,
    horizon: int,
    y: np.ndarray,
    fit_origins: np.ndarray,
    early_origins: np.ndarray,
    device: torch.device,
) -> dict:
    cfg = GRID[config_id]

    fit_target_max = int(fit_origins[-1] + horizon)
    scaler = ScalarStandardizer.fit(y[:fit_target_max + 1])

    x_fit, y_fit, _ = build_xy_from_origins(
        y, fit_origins, int(cfg["lookback"]), horizon, scaler, True
    )
    x_early, y_early, _ = build_xy_from_origins(
        y, early_origins, int(cfg["lookback"]), horizon, scaler, True
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
        if time.perf_counter() - started > HARD_STAGE_SECONDS:
            raise TimeoutError(
                f"RESOURCE_LIMIT inner {family}/{config_id}/H{horizon}"
            )

        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite inner loss {family}/{config_id}/H{horizon}"
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

    runtime = time.perf_counter() - started
    if runtime > HARD_STAGE_SECONDS:
        raise TimeoutError(
            f"RESOURCE_LIMIT inner {family}/{config_id}/H{horizon}"
        )
    if best_epoch is None or best_state is None:
        raise RuntimeError("No valid inner early-stopping checkpoint")

    return {
        "best_epoch": int(best_epoch),
        "best_early_stop_mae_scaled": float(best_val),
        "inner_scaler_cutoff_index": fit_target_max,
        "inner_scaler_mean_bps": scaler.mean,
        "inner_scaler_scale_bps": scaler.scale,
        "inner_runtime_seconds": float(runtime),
        "parameter_count": parameter_count(model),
    }


def final_refit(
    family: str,
    config_id: str,
    horizon: int,
    best_epoch: int,
    y: np.ndarray,
    train_origins: np.ndarray,
    device: torch.device,
):
    cfg = GRID[config_id]

    # Frozen rule: scaler is refitted on the complete permitted June training block.
    scaler = ScalarStandardizer.fit(y[:TRAIN_ROWS])

    x_train, y_train, _ = build_xy_from_origins(
        y, train_origins, int(cfg["lookback"]), horizon, scaler, True
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
    loader = make_loader(x_train, y_train)

    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        if time.perf_counter() - started > HARD_STAGE_SECONDS:
            raise TimeoutError(
                f"RESOURCE_LIMIT final_refit {family}/{config_id}/H{horizon}"
            )

        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite final-refit loss {family}/{config_id}/H{horizon}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    runtime = time.perf_counter() - started
    if runtime > HARD_STAGE_SECONDS:
        raise TimeoutError(
            f"RESOURCE_LIMIT final_refit {family}/{config_id}/H{horizon}"
        )

    return model, scaler, float(runtime), parameter_count(model)


def blind_predict(
    model: nn.Module,
    scaler: ScalarStandardizer,
    family: str,
    config_id: str,
    horizon: int,
    best_epoch: int,
    y: np.ndarray,
    timestamps: pd.Series,
    device: torch.device,
) -> pd.DataFrame:
    cfg = GRID[config_id]
    origins = blind_origins(horizon)
    x_blind, _, targets = build_xy_from_origins(
        y,
        origins,
        int(cfg["lookback"]),
        horizon,
        scaler,
        with_targets=False,
    )

    # Predictions are generated from lag inputs only. Blind targets are accessed
    # only after the forward pass for evaluation.
    pred_scaled = predict_scaled(model, x_blind, device)
    y_pred = scaler.inverse(pred_scaled)

    y_true = y[targets].astype(np.float64)
    persistence = y[origins].astype(np.float64)

    out = pd.DataFrame({
        "family": family,
        "config_id": config_id,
        "is_recurrent_star": bool((family, config_id) == FROZEN_RECURRENT_STAR),
        "canonical_seed": CANONICAL_SEED,
        "horizon_steps": int(horizon),
        "horizon_minutes": int(horizon * 5),
        "best_epoch_from_inner_training": int(best_epoch),
        "origin_index": origins.astype(int),
        "target_index": targets.astype(int),
        "origin_timestamp": timestamps.iloc[origins].to_numpy(),
        "target_timestamp": timestamps.iloc[targets].to_numpy(),
        "y_true_bps": y_true,
        "y_pred_bps": y_pred,
        "persistence_pred_bps": persistence,
        "residual_bps": y_true - y_pred,
        "abs_error_bps": np.abs(y_true - y_pred),
        "persistence_abs_error_bps": np.abs(y_true - persistence),
    })
    return out


def metric_row(predictions: pd.DataFrame, mase_scale: float) -> dict:
    yt = predictions["y_true_bps"].to_numpy(float)
    yp = predictions["y_pred_bps"].to_numpy(float)
    pp = predictions["persistence_pred_bps"].to_numpy(float)

    residual = yt - yp
    abs_error = np.abs(residual)
    persistence_abs = np.abs(yt - pp)

    mae = float(abs_error.mean())
    p_mae = float(persistence_abs.mean())
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    denominator = np.abs(yt) + np.abs(yp)
    smape = np.zeros_like(denominator, dtype=float)
    mask = denominator > 0
    smape[mask] = 2.0 * abs_error[mask] / denominator[mask]

    return {
        "n_predictions": int(len(predictions)),
        "mae_bps": mae,
        "persistence_mae_bps": p_mae,
        "mae_ratio_vs_persistence": float(mae / p_mae),
        "skill_vs_persistence": float(1.0 - mae / p_mae),
        "rmse_bps": rmse,
        "smape_pct": float(100.0 * smape.mean()),
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
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
    print("=" * 126)
    print(f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO NI PREDICCIONES BLIND")
    print("=" * 126)

    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {runner_sha}")
    print()

    software = validate_software()

    governance = bundle_root / "governance"
    evidence = bundle_root / "evidence"
    inputs = bundle_root / "inputs"

    validate_hash_file(governance / PROTOCOL_NAME, PROTOCOL_SHA, "023")
    validate_hash_file(governance / MARCH_CLOSURE_NAME, MARCH_CLOSURE_SHA, "024")
    validate_hash_file(governance / APRIL_CLOSURE_NAME, APRIL_CLOSURE_SHA, "025")
    validate_hash_file(governance / STABILITY_CLOSURE_NAME, STABILITY_CLOSURE_SHA, "026")
    validate_hash_file(governance / FREEZE_NAME, FREEZE_SHA, "027")

    reps = validate_representatives(evidence / REPRESENTATIVES_NAME)
    stability = validate_stability_summary(evidence / STABILITY_SUMMARY_NAME)
    june = validate_june(inputs / JUNE_NAME)
    bundle_manifest = validate_bundle_manifest(
        bundle_root / "manifest" / BUNDLE_MANIFEST_NAME,
        runner_sha,
    )

    check("canonical seed frozen", CANONICAL_SEED == 20260820, str(CANONICAL_SEED))

    for h in HORIZONS:
        train_origins = training_common_origins(h)
        check(f"H{h} train common count",
              len(train_origins) == EXPECTED_TRAIN_COMMON_COUNTS[h],
              str(len(train_origins)))
        fit_origins, early_origins = inner_split(train_origins)
        exp = EXPECTED_INNER_SPLITS[h]
        check(f"H{h} inner fit count", len(fit_origins) == exp["fit"], str(len(fit_origins)))
        check(f"H{h} inner gap", int(early_origins[0] - fit_origins[-1] - 1) == exp["gap"])
        check(f"H{h} inner early-stop count",
              len(early_origins) == exp["early_stop"],
              str(len(early_origins)))

        b_origins = blind_origins(h)
        check(f"H{h} blind count",
              len(b_origins) == EXPECTED_BLIND_COUNTS[h],
              str(len(b_origins)))
        check(f"H{h} first blind origin is last train row",
              int(b_origins[0]) == TRAIN_ROWS - 1,
              str(int(b_origins[0])))
        check(f"H{h} first target is inside blind block",
              int(b_origins[0] + h) >= TRAIN_ROWS,
              str(int(b_origins[0] + h)))
        check(f"H{h} final target is June final row",
              int(b_origins[-1] + h) == JUNE_ROWS - 1,
              str(int(b_origins[-1] + h)))

    set_seed(CANONICAL_SEED)
    device = torch.device("cuda")
    for family, config_id in FROZEN_REPRESENTATIVES.items():
        cfg = GRID[config_id]
        x = torch.randn(4, int(cfg["lookback"]), 1, device=device)
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
              tuple(out.shape) == (4, 1),
              str(tuple(out.shape)))

    print("TRAINING STAGES EXECUTED IN PREFLIGHT: 0")
    print("BLIND PREDICTIONS GENERATED IN PREFLIGHT: 0")
    print("CAMPAIGN ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")

    return software, june, reps, stability, bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/recurrent_june_blind_v1_2026-08-21_outputs')),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)

    software, june, _, _, bundle_manifest = preflight(bundle_root)

    if args.preflight_only:
        return 0

    output_names = {
        "runtime": f"ugr16_recurrent_june_blind_{FILE_TAG}_runtime.csv",
        "epoch_selection": f"ugr16_recurrent_june_blind_{FILE_TAG}_epoch_selection.csv",
        "predictions": f"ugr16_recurrent_june_blind_{FILE_TAG}_predictions.parquet",
        "horizon_metrics": f"ugr16_recurrent_june_blind_{FILE_TAG}_horizon_metrics.csv",
        "report": f"ugr16_recurrent_june_blind_{FILE_TAG}_report.txt",
        "manifest": f"ugr16_recurrent_june_blind_{FILE_TAG}_manifest.json",
        "results_zip": f"recurrent_june_blind_001_runner_results_{FILE_TAG}.zip",
    }
    paths = {k: output_dir / v for k, v in output_names.items()}

    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing outputs; refusing overwrite:\n" +
            "\n".join(str(p) for p in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    y = june[TARGET].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(june[TIME_COL]).reset_index(drop=True)
    device = torch.device("cuda")

    # MASE scale uses only the complete permitted initial June training block.
    mase_scale = float(np.mean(np.abs(np.diff(y[:TRAIN_ROWS]))))
    check("June training MASE scale valid",
          np.isfinite(mase_scale) and mase_scale > 0,
          str(mase_scale))

    runtime_rows = []
    epoch_rows = []
    pred_frames = []

    campaign_started = time.perf_counter()

    print()
    print("=" * 126)
    print(f"START {CAMPAIGN_ID}")
    print("=" * 126)
    print("Frozen representatives = 3")
    print("Horizons = 4")
    print("Inner epoch-selection fits = 12")
    print("Final refits = 12")
    print("Total training stages = 24")
    print(f"Expected blind predictions = {EXPECTED_TOTAL_BLIND_PREDICTIONS}")
    print("Canonical seed = 20260820")
    print("No tuning. No selection changes. No blind-target-driven fitting.")
    print()

    for family, config_id in FROZEN_REPRESENTATIVES.items():
        for h in HORIZONS:
            train_origins = training_common_origins(h)
            fit_origins, early_origins = inner_split(train_origins)

            print(
                f"[INNER] {family:10s} {config_id} H{h:<2d} "
                f"fit={len(fit_origins)} gap={INNER_GAP} early={len(early_origins)}",
                flush=True,
            )
            epoch_info = select_best_epoch(
                family=family,
                config_id=config_id,
                horizon=h,
                y=y,
                fit_origins=fit_origins,
                early_origins=early_origins,
                device=device,
            )
            print(
                f"        PASS best_epoch={epoch_info['best_epoch']} "
                f"runtime={epoch_info['inner_runtime_seconds']:.2f}s",
                flush=True,
            )

            print(
                f"[REFIT] {family:10s} {config_id} H{h:<2d} "
                f"epochs={epoch_info['best_epoch']} "
                f"train_samples={len(train_origins)}",
                flush=True,
            )
            model, final_scaler, final_runtime, params = final_refit(
                family=family,
                config_id=config_id,
                horizon=h,
                best_epoch=epoch_info["best_epoch"],
                y=y,
                train_origins=train_origins,
                device=device,
            )
            print(
                f"        PASS epochs={epoch_info['best_epoch']} "
                f"runtime={final_runtime:.2f}s",
                flush=True,
            )

            preds = blind_predict(
                model=model,
                scaler=final_scaler,
                family=family,
                config_id=config_id,
                horizon=h,
                best_epoch=epoch_info["best_epoch"],
                y=y,
                timestamps=timestamps,
                device=device,
            )

            check(f"{family}/{config_id}/H{h} blind prediction count",
                  len(preds) == EXPECTED_BLIND_COUNTS[h],
                  str(len(preds)))
            check(f"{family}/{config_id}/H{h} predictions finite",
                  np.isfinite(preds["y_pred_bps"].to_numpy(float)).all())

            runtime_rows.append({
                "family": family,
                "config_id": config_id,
                "is_recurrent_star": bool((family, config_id) == FROZEN_RECURRENT_STAR),
                "canonical_seed": CANONICAL_SEED,
                "horizon_steps": h,
                "lookback": GRID[config_id]["lookback"],
                "hidden_size": GRID[config_id]["hidden_size"],
                "n_layers": GRID[config_id]["n_layers"],
                "dropout": GRID[config_id]["dropout"],
                "parameter_count": params,
                "train_common_samples": len(train_origins),
                "inner_fit_samples": len(fit_origins),
                "inner_gap": INNER_GAP,
                "inner_early_stop_samples": len(early_origins),
                "best_epoch": epoch_info["best_epoch"],
                "inner_runtime_seconds": epoch_info["inner_runtime_seconds"],
                "final_refit_epochs": epoch_info["best_epoch"],
                "final_refit_runtime_seconds": final_runtime,
                "final_scaler_mean_bps": final_scaler.mean,
                "final_scaler_scale_bps": final_scaler.scale,
                "status": "PASS",
            })

            epoch_rows.append({
                "family": family,
                "config_id": config_id,
                "horizon_steps": h,
                "canonical_seed": CANONICAL_SEED,
                "inner_fit_samples": len(fit_origins),
                "inner_gap": INNER_GAP,
                "inner_early_stop_samples": len(early_origins),
                "inner_scaler_cutoff_index": epoch_info["inner_scaler_cutoff_index"],
                "inner_scaler_mean_bps": epoch_info["inner_scaler_mean_bps"],
                "inner_scaler_scale_bps": epoch_info["inner_scaler_scale_bps"],
                "best_epoch": epoch_info["best_epoch"],
                "best_early_stop_mae_scaled": epoch_info["best_early_stop_mae_scaled"],
            })

            pred_frames.append(preds)

    runtime = pd.DataFrame(runtime_rows)
    epoch_selection = pd.DataFrame(epoch_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)

    check("runtime rows = 12", len(runtime) == 12, str(len(runtime)))
    check("epoch-selection rows = 12", len(epoch_selection) == 12,
          str(len(epoch_selection)))
    check("all runtime status PASS", runtime["status"].eq("PASS").all())
    check("total blind predictions",
          len(predictions) == EXPECTED_TOTAL_BLIND_PREDICTIONS,
          str(len(predictions)))

    # Verify paired coverage among the three representatives for each horizon.
    for h in HORIZONS:
        expected_origins = tuple(blind_origins(h).tolist())
        expected_targets = tuple((blind_origins(h) + h).tolist())
        for family, config_id in FROZEN_REPRESENTATIVES.items():
            sub = predictions[
                predictions["family"].eq(family)
                & predictions["config_id"].eq(config_id)
                & predictions["horizon_steps"].eq(h)
            ]
            check(f"{family}/{config_id}/H{h} paired origin coverage",
                  tuple(sub["origin_index"].astype(int)) == expected_origins)
            check(f"{family}/{config_id}/H{h} paired target coverage",
                  tuple(sub["target_index"].astype(int)) == expected_targets)

    metric_rows = []
    for (family, config_id, h), sub in predictions.groupby(
        ["family", "config_id", "horizon_steps"],
        sort=True,
    ):
        metric_rows.append({
            "family": family,
            "config_id": config_id,
            "is_recurrent_star": bool((family, config_id) == FROZEN_RECURRENT_STAR),
            "canonical_seed": CANONICAL_SEED,
            "horizon_steps": int(h),
            "horizon_minutes": int(h * 5),
            "best_epoch": int(
                epoch_selection[
                    epoch_selection["family"].eq(family)
                    & epoch_selection["config_id"].eq(config_id)
                    & epoch_selection["horizon_steps"].eq(h)
                ]["best_epoch"].iloc[0]
            ),
            **metric_row(sub, mase_scale),
        })

    horizon_metrics = pd.DataFrame(metric_rows)
    check("horizon metric rows = 12", len(horizon_metrics) == 12,
          str(len(horizon_metrics)))

    # Integrity of frozen authorities/inputs after execution.
    validate_hash_file(bundle_root / "governance" / PROTOCOL_NAME, PROTOCOL_SHA, "023 post-run")
    validate_hash_file(bundle_root / "governance" / MARCH_CLOSURE_NAME, MARCH_CLOSURE_SHA, "024 post-run")
    validate_hash_file(bundle_root / "governance" / APRIL_CLOSURE_NAME, APRIL_CLOSURE_SHA, "025 post-run")
    validate_hash_file(bundle_root / "governance" / STABILITY_CLOSURE_NAME, STABILITY_CLOSURE_SHA, "026 post-run")
    validate_hash_file(bundle_root / "governance" / FREEZE_NAME, FREEZE_SHA, "027 post-run")
    validate_hash_file(bundle_root / "inputs" / JUNE_NAME, JUNE_SHA, "June post-run")
    validate_hash_file(bundle_root / "evidence" / REPRESENTATIVES_NAME, REPRESENTATIVES_SHA,
                       "representatives post-run")
    validate_hash_file(bundle_root / "evidence" / STABILITY_SUMMARY_NAME, STABILITY_SUMMARY_SHA,
                       "stability summary post-run")

    campaign_seconds = time.perf_counter() - campaign_started

    report_lines = [
        "=" * 126,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 126,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"023 SHA-256: {PROTOCOL_SHA}",
        f"024 SHA-256: {MARCH_CLOSURE_SHA}",
        f"025 SHA-256: {APRIL_CLOSURE_SHA}",
        f"026 SHA-256: {STABILITY_CLOSURE_SHA}",
        f"027 SHA-256: {FREEZE_SHA}",
        "",
        "GOVERNANCE",
        "June role: blind external evaluation",
        "Representatives changed: NO",
        "Best-seed selection: NO",
        "Retuning: NO",
        f"Canonical seed: {CANONICAL_SEED}",
        "Blind-target-driven fitting: NO",
        "",
        "FROZEN REPRESENTATIVES",
        "RNN*       = SimpleRNN/N03",
        "LSTM*      = LSTM/N04",
        "GRU*       = GRU/N06",
        "RECURRENT* = GRU/N06",
        "",
        "JUNE SPLIT",
        f"train rows = {TRAIN_ROWS}",
        f"blind rows = {BLIND_ROWS}",
        f"blind starts = {BLIND_START}",
        "",
        "BLIND METRICS",
    ]

    for row in horizon_metrics.sort_values(
        ["family", "horizon_steps"], kind="mergesort"
    ).itertuples(index=False):
        report_lines.append(
            f"{row.family:10s} {row.config_id} H{row.horizon_steps:<2d} "
            f"n={row.n_predictions} best_epoch={row.best_epoch} "
            f"MAE={row.mae_bps:.6f} "
            f"RMSE={row.rmse_bps:.6f} "
            f"ratio_vs_Persistence={row.mae_ratio_vs_persistence:.9f} "
            f"skill={row.skill_vs_persistence:.9f} "
            f"bias={row.bias_bps:.6f} "
            f"underpred_pct={row.underprediction_pct:.6f} "
            f"P95={row.p95_abs_error_bps:.6f}"
        )

    report_lines += [
        "",
        f"Runtime rows: {len(runtime)}",
        f"Epoch-selection rows: {len(epoch_selection)}",
        f"Horizon metric rows: {len(horizon_metrics)}",
        f"Blind prediction rows: {len(predictions)}",
        f"Campaign seconds: {campaign_seconds:.3f}",
        "STATUS: PASS",
        "REPRESENTATIVES CHANGED: NO",
        "BEST SEED SELECTED: NO",
        "RETUNING: NO",
        "RECURRENT-INFERENCE-RESIDUALS-001: NOT RUN",
        "RECURRENT-CLOSURE-001: NOT RUN",
    ]
    report = "\n".join(report_lines) + "\n"

    atomic_csv(runtime, paths["runtime"])
    atomic_csv(epoch_selection, paths["epoch_selection"])
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(horizon_metrics, paths["horizon_metrics"])
    atomic_text(report, paths["report"])

    output_records = {}
    for key in ("runtime", "epoch_selection", "predictions",
                "horizon_metrics", "report"):
        p = paths[key]
        output_records[key] = {
            "name": p.name,
            "sha256": sha256(p),
            "bytes": int(p.stat().st_size),
        }

    validation = {
        "runtime_rows_12": len(runtime) == 12,
        "epoch_selection_rows_12": len(epoch_selection) == 12,
        "all_runtime_status_pass": bool(runtime["status"].eq("PASS").all()),
        "horizon_metric_rows_12": len(horizon_metrics) == 12,
        "blind_prediction_rows_expected":
            len(predictions) == EXPECTED_TOTAL_BLIND_PREDICTIONS,
        "finite_predictions":
            bool(np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all()),
        "representatives_unchanged": True,
        "canonical_seed_unchanged": CANONICAL_SEED == 20260820,
        "no_best_seed_selection": True,
        "no_retuning": True,
        "june_hash_unchanged":
            sha256(bundle_root / "inputs" / JUNE_NAME) == JUNE_SHA,
        "protocol_hash_unchanged":
            sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA,
        "freeze_hash_unchanged":
            sha256(bundle_root / "governance" / FREEZE_NAME) == FREEZE_SHA,
        "representatives_hash_unchanged":
            sha256(bundle_root / "evidence" / REPRESENTATIVES_NAME) == REPRESENTATIVES_SHA,
        "stability_summary_hash_unchanged":
            sha256(bundle_root / "evidence" / STABILITY_SUMMARY_NAME) == STABILITY_SUMMARY_SHA,
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
        "bundle_manifest": bundle_manifest,
        "authorities": {
            "023": {"name": PROTOCOL_NAME, "sha256": PROTOCOL_SHA},
            "024": {"name": MARCH_CLOSURE_NAME, "sha256": MARCH_CLOSURE_SHA},
            "025": {"name": APRIL_CLOSURE_NAME, "sha256": APRIL_CLOSURE_SHA},
            "026": {"name": STABILITY_CLOSURE_NAME, "sha256": STABILITY_CLOSURE_SHA},
            "027": {"name": FREEZE_NAME, "sha256": FREEZE_SHA},
        },
        "inputs": {
            "june": {
                "name": JUNE_NAME,
                "sha256": JUNE_SHA,
                "rows": JUNE_ROWS,
                "coverage": [JUNE_START, JUNE_END],
                "train_rows": TRAIN_ROWS,
                "blind_rows": BLIND_ROWS,
                "blind_start": BLIND_START,
            },
            "representatives": {
                "name": REPRESENTATIVES_NAME,
                "sha256": REPRESENTATIVES_SHA,
            },
            "stability_summary": {
                "name": STABILITY_SUMMARY_NAME,
                "sha256": STABILITY_SUMMARY_SHA,
            },
        },
        "software": software,
        "frozen_configuration": {
            "canonical_seed": CANONICAL_SEED,
            "representatives": {
                "RNN_star": {"family": "SimpleRNN", "config_id": "N03"},
                "LSTM_star": {"family": "LSTM", "config_id": "N04"},
                "GRU_star": {"family": "GRU", "config_id": "N06"},
                "RECURRENT_star": {"family": "GRU", "config_id": "N06"},
            },
            "horizons": list(HORIZONS),
            "L_MAX": L_MAX,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "L1Loss",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "gradient_clip_norm": MAX_GRAD_NORM,
            "shuffle": False,
            "inner_gap": INNER_GAP,
            "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
            "final_refit_uses_complete_training_block_scaler": True,
            "final_refit_exact_best_epoch": True,
            "blind_online_weight_updates": False,
            "blind_real_observations_as_revealed_lags": True,
        },
        "campaign": {
            "inner_epoch_selection_fits": 12,
            "final_refits": 12,
            "total_training_stages": 24,
            "blind_model_horizon_evaluations": 12,
            "blind_prediction_rows": int(len(predictions)),
            "expected_counts_per_horizon": EXPECTED_BLIND_COUNTS,
            "representatives_changed": False,
            "best_seed_selection": False,
            "retuning": False,
            "blind_targets_used_for_fitting": False,
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
            "runtime", "epoch_selection", "predictions",
            "horizon_metrics", "report", "manifest"
        ):
            p = paths[key]
            zf.write(p, arcname=p.name)

    print()
    print(report)
    print("VALIDATIONS")
    print("-" * 126)
    for key, value in validation.items():
        print(f"{key:62s}: {'PASS' if value else 'FAIL'}")

    print()
    print(f"Results ZIP: {results_zip_path}")
    print(f"Results ZIP SHA-256: {sha256(results_zip_path)}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("REPRESENTATIVES CHANGED: NO")
    print("BEST SEED SELECTED: NO")
    print("RETUNING: NO")
    print("RECURRENT-INFERENCE-RESIDUALS-001: NOT RUN")
    print("RECURRENT-CLOSURE-001: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
