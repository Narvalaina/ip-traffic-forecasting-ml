#!/usr/bin/env python3
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
RUNNER_DATE = "2026-08-22"
CAMPAIGN_ID = "TRANSFORMER-STABILITY-001"
PARENT_CAMPAIGN_ID = "UGR16-TRANSFORMER-LITE-001"

PROTOCOL_NAME = "033_phase_e_transformer_lite_protocol_2026-08-21.md"
PROTOCOL_SHA = "4fcd61519612ae5fd3daa6d874a73df50b27b63531ac109f0782351875211552"

PREFLIGHT_CLOSURE_NAME = "034_phase_e_colab_preflight_closure_2026-08-21.md"
PREFLIGHT_CLOSURE_SHA = "9d89f0bbaa13448c56217c05f1663fa57fdb4505d3a395860186e2235e9cc081"

MARCH_RUNNER_PREFLIGHT_NAME = "035_transformer_march_runner_preflight_closure_2026-08-21.md"
MARCH_RUNNER_PREFLIGHT_SHA = "2f90c65e1855c98a20470d76229be68359cad8fb33b0b213c5f351eb9e21322e"

RUNTIME_COMPATIBILITY_NAME = "036_phase_e_colab_runtime_compatibility_2026-08-21.md"
RUNTIME_COMPATIBILITY_SHA = "8e8aeb3b0783c5928468bbbf074c1c42359837b57461890c4eb5e1145ca9dbf3"

MARCH_CLOSURE_NAME = "037_transformer_march_screen_closure_2026-08-22.md"
MARCH_CLOSURE_SHA = "9650e5aeb7ea8653e17baaa584237da8ab3d1fb41db9a150b0f98e29e075e03e"

APRIL_PREFLIGHT_CLOSURE_NAME = "038_transformer_april_selection_preflight_closure_2026-08-22.md"
APRIL_PREFLIGHT_CLOSURE_SHA = "99b9fbe0592729ec2c5e5d653124b62f52a423c917bda58568ab80ea67e5cc7c"

APRIL_CLOSURE_NAME = "039_transformer_april_selection_closure_2026-08-22.md"
APRIL_CLOSURE_SHA = "f4998ca12d138c91cc9d67f53b1e95a5bd4310cdb0637aa32e7690ec5836ab99"

APRIL_CANDIDATE_SUMMARY_NAME = (
    "ugr16_transformer_april_selection_candidate_summary_v1_2026-08-22.csv"
)
APRIL_CANDIDATE_SUMMARY_SHA = (
    "5272e6733f60b6cf24281d9fe3d73cba62918636e582ad7ed7fcb2445728ff14"
)

APRIL_SCIENTIFIC_MANIFEST_NAME = (
    "ugr16_transformer_april_selection_manifest_v1_2026-08-22.json"
)
APRIL_SCIENTIFIC_MANIFEST_SHA = (
    "c687f0a4f09487d5a90baaefa86a034d53a26921c32014f38a150d41a2db0679"
)

APRIL_NAME = "april_week3_prepared_5min.parquet"
APRIL_SHA = "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0"
APRIL_ROWS = 2004
APRIL_START = "2016-04-11 01:00:00"
APRIL_END = "2016-04-17 23:55:00"

TARGET = "bitrate_bps"
TIME_COL = "timestamp"

L_MAX = 288
TRANSFORMER_STAR_CONFIG = "T01"
TRANSFORMER_STAR_LOOKBACK = 24
CANONICAL_SEED = 20260820
STABILITY_SEEDS = (20260820, 20260821, 20260822)
HORIZONS = (1, 3, 6, 12)
GAP = 12

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 0.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
HARD_FIT_SECONDS = 600.0
EXPECTED_PARAMETER_COUNT = 8705

EXPECTED_SOFTWARE = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torch_cuda": "12.8",
    "cudnn": 91900,
    "numpy": "2.1.3",
    "pandas": "2.2.3",
}

REQUIRED_COLUMNS = {
    "timestamp", "flows_total", "packets_total", "bytes_total", "bitrate_bps",
    "packet_rate_pps", "flow_rate_fps", "mean_flow_duration", "max_flow_bytes",
    "max_flow_packets", "flows_background", "packets_background", "bytes_background",
    "flows_blacklist", "packets_blacklist", "bytes_blacklist", "flows_anomaly",
    "packets_anomaly", "bytes_anomaly", "tcp_flows", "tcp_packets", "tcp_bytes",
    "udp_flows", "udp_packets", "udp_bytes", "icmp_flows", "icmp_packets",
    "icmp_bytes", "other_protocol_flows", "other_protocol_packets",
    "other_protocol_bytes", "duration_sum",
}

EXPECTED_APRIL_COMMON_COUNTS = {
    1: 1716,
    3: 1714,
    6: 1711,
    12: 1705,
}

EXPECTED_APRIL_SPLITS = {
    1: (1184, 253, 255),
    3: (1183, 253, 254),
    6: (1180, 253, 254),
    12: (1176, 252, 253),
}

EXPECTED_FITS = 12
EXPECTED_HORIZON_METRIC_ROWS = 12
EXPECTED_SEED_SUMMARY_ROWS = 3
EXPECTED_STABILITY_SUMMARY_ROWS = 1
EXPECTED_PREDICTIONS = 3048

OUTPUT_PREFIX = "ugr16_transformer_stability"
OUTPUT_SUFFIX = "v1_2026-08-22"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{name:64s}: {status}" + (f" | {detail}" if detail else ""))
    if not condition:
        raise RuntimeError(f"{name}: FAIL | {detail}")


def exact_python_version() -> str:
    return (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


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
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_vram_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None
        ),
    }

    check("software_python", info["python"] == EXPECTED_SOFTWARE["python"], info["python"])
    check("software_torch", info["torch"] == EXPECTED_SOFTWARE["torch"], info["torch"])
    check("software_torch_cuda", info["torch_cuda"] == EXPECTED_SOFTWARE["torch_cuda"], str(info["torch_cuda"]))
    check("software_cudnn", info["cudnn"] == EXPECTED_SOFTWARE["cudnn"], str(info["cudnn"]))
    check("software_numpy", info["numpy"] == EXPECTED_SOFTWARE["numpy"], info["numpy"])
    check("software_pandas", info["pandas"] == EXPECTED_SOFTWARE["pandas"], info["pandas"])
    check("cuda_available", info["cuda_available"], str(info["gpu_name"]))
    check("cuda_device_count_positive", info["device_count"] >= 1, str(info["device_count"]))

    return info


def find_forbidden_june(bundle_root: Path) -> list[Path]:
    return [
        p
        for p in bundle_root.rglob("*")
        if p.is_file() and "june" in p.name.lower()
    ]


def validate_frame(
    path: Path,
    expected_sha: str,
    expected_rows: int,
    expected_start: str,
    expected_end: str,
) -> pd.DataFrame:
    check(f"{path.name}: exists", path.is_file(), str(path))

    actual_sha = sha256(path)
    check(f"{path.name}: sha256", actual_sha == expected_sha, actual_sha)

    frame = pd.read_parquet(path)
    check(f"{path.name}: rows", len(frame) == expected_rows, str(len(frame)))
    check(
        f"{path.name}: columns",
        set(frame.columns) == REQUIRED_COLUMNS,
        f"{len(frame.columns)} cols",
    )
    check(
        f"{path.name}: target finite",
        np.isfinite(frame[TARGET].to_numpy(float)).all(),
    )

    ts = pd.to_datetime(frame[TIME_COL], errors="raise")
    check(f"{path.name}: timestamps unique", not ts.duplicated().any())
    check(f"{path.name}: timestamps sorted", ts.is_monotonic_increasing)
    check(f"{path.name}: start", str(ts.min()) == expected_start, str(ts.min()))
    check(f"{path.name}: end", str(ts.max()) == expected_end, str(ts.max()))

    diffs = ts.diff().dropna()
    check(
        f"{path.name}: spacing 5 min",
        bool((diffs == pd.Timedelta(minutes=5)).all()),
    )

    return frame


def common_sample_indices(n_rows: int, horizon: int) -> np.ndarray:
    n = n_rows - L_MAX - horizon + 1
    if n <= 0:
        raise RuntimeError("No common samples")
    return np.arange(n, dtype=np.int64)


def origin_from_common_index(i: int) -> int:
    return (L_MAX - 1) + int(i)


def april_split_indices(
    n_common: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_available = n_common - 2 * GAP
    fit_n = int(math.floor(0.70 * n_available))
    early_n = int(math.floor(0.15 * n_available))
    selection_n = n_available - fit_n - early_n

    fit_start = 0
    fit_end = fit_start + fit_n

    gap1_start = fit_end
    gap1_end = gap1_start + GAP

    early_start = gap1_end
    early_end = early_start + early_n

    gap2_start = early_end
    gap2_end = gap2_start + GAP

    selection_start = gap2_end
    selection_end = selection_start + selection_n

    if selection_end != n_common:
        raise RuntimeError(
            f"April split does not consume common samples: "
            f"{selection_end} != {n_common}"
        )

    return (
        np.arange(fit_start, fit_end, dtype=np.int64),
        np.arange(gap1_start, gap1_end, dtype=np.int64),
        np.arange(early_start, early_end, dtype=np.int64),
        np.arange(gap2_start, gap2_end, dtype=np.int64),
        np.arange(selection_start, selection_end, dtype=np.int64),
    )


@dataclass
class ScalarStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        scale = float(arr.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(scale):
            raise RuntimeError("Non-finite scaler")
        if scale == 0.0:
            scale = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float64) - self.mean
        ) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def build_xy(
    y: np.ndarray,
    sample_indices: np.ndarray,
    lookback: int,
    horizon: int,
    scaler: ScalarStandardizer,
):
    xs, ys, origins, targets = [], [], [], []

    for i in sample_indices:
        origin = origin_from_common_index(int(i))
        target_idx = origin + horizon
        start = origin - lookback + 1

        xs.append(
            scaler.transform(
                y[start : origin + 1]
            ).astype(np.float32).reshape(-1, 1)
        )
        ys.append(
            np.float32(
                scaler.transform(
                    np.array([y[target_idx]])
                )[0]
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


class FixedSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 32, max_len: int = 288):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(
            dtype=x.dtype,
            device=x.device,
        )


class TransformerLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(1, 32, bias=True)
        self.positional_encoding = FixedSinusoidalPositionalEncoding(32, 288)

        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.final_norm = nn.LayerNorm(32)
        self.external_dropout = nn.Dropout(0.10)
        self.head = nn.Linear(32, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x)
        z = self.positional_encoding(z)
        z = self.encoder(z)
        z = z[:, -1, :]
        z = self.final_norm(z)
        z = self.external_dropout(z)
        return self.head(z)


def parameter_count(model: nn.Module) -> int:
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def make_loader(x, y):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    loss_fn = nn.L1Loss(reduction="sum")

    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        total += float(
            loss_fn(model(xb), yb).item()
        )
        n += len(xb)

    return total / n


@torch.no_grad()
def predict_scaled(model, x, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    chunks = []

    for (xb,) in loader:
        chunks.append(
            model(xb.to(device))
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return np.concatenate(chunks)


def train_one_fit(
    seed: int,
    horizon: int,
    y: np.ndarray,
    fit_idx: np.ndarray,
    early_idx: np.ndarray,
    selection_idx: np.ndarray,
    device,
):
    lookback = TRANSFORMER_STAR_LOOKBACK

    fit_targets = np.array(
        [
            origin_from_common_index(i) + horizon
            for i in fit_idx
        ]
    )

    scaler_cutoff = int(fit_targets.max())

    scaler = ScalarStandardizer.fit(
        y[: scaler_cutoff + 1]
    )

    x_fit, y_fit, _, _ = build_xy(
        y,
        fit_idx,
        lookback,
        horizon,
        scaler,
    )

    x_early, y_early, _, _ = build_xy(
        y,
        early_idx,
        lookback,
        horizon,
        scaler,
    )

    x_selection, _, origins_sel, targets_sel = build_xy(
        y,
        selection_idx,
        lookback,
        horizon,
        scaler,
    )

    set_seed(seed)
    model = TransformerLite().to(device)

    check(
        f"seed={seed}/H{horizon}: parameter count",
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        str(parameter_count(model)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.L1Loss()

    fit_loader = make_loader(
        x_fit,
        y_fit,
    )

    early_loader = make_loader(
        x_early,
        y_early,
    )

    best_val = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for xb, yb in fit_loader:
            if (
                time.perf_counter() - started
                > HARD_FIT_SECONDS
            ):
                raise TimeoutError(
                    f"RESOURCE_LIMIT seed={seed}/H{horizon}"
                )

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = loss_fn(
                model(xb),
                yb,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

        early_mae = eval_loss(
            model,
            early_loader,
            device,
        )

        if early_mae < (best_val - MIN_DELTA):
            best_val = early_mae
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    runtime_s = time.perf_counter() - started

    if runtime_s > HARD_FIT_SECONDS:
        raise TimeoutError(
            f"RESOURCE_LIMIT seed={seed}/H{horizon}"
        )

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint"
        )

    model.load_state_dict(best_state)

    y_pred = scaler.inverse(
        predict_scaled(
            model,
            x_selection,
            device,
        )
    )

    y_true = y[
        targets_sel
    ].astype(np.float64)

    persistence = y[
        origins_sel
    ].astype(np.float64)

    predictions = pd.DataFrame(
        {
            "seed": seed,
            "config_id": TRANSFORMER_STAR_CONFIG,
            "lookback": TRANSFORMER_STAR_LOOKBACK,
            "horizon_steps": horizon,
            "sample_index": selection_idx.astype(int),
            "origin_index": origins_sel.astype(int),
            "target_index": targets_sel.astype(int),
            "y_true_bps": y_true,
            "y_pred_bps": y_pred,
            "persistence_pred_bps": persistence,
            "residual_bps": y_true - y_pred,
            "abs_error_bps": np.abs(
                y_true - y_pred
            ),
            "persistence_abs_error_bps": np.abs(
                y_true - persistence
            ),
        }
    )

    mase_scale = float(
        np.mean(
            np.abs(
                np.diff(
                    y[: scaler_cutoff + 1]
                )
            )
        )
    )

    if (
        not np.isfinite(mase_scale)
        or mase_scale <= 0
    ):
        raise RuntimeError(
            "Invalid MASE scale"
        )

    fit_record = {
        "seed": seed,
        "config_id": TRANSFORMER_STAR_CONFIG,
        "lookback": TRANSFORMER_STAR_LOOKBACK,
        "horizon_steps": horizon,
        "n_fit_samples": len(fit_idx),
        "n_early_stop_samples": len(early_idx),
        "n_selection_samples": len(selection_idx),
        "scaler_cutoff_index": scaler_cutoff,
        "scaler_mean_bps": scaler.mean,
        "scaler_scale_bps": scaler.scale,
        "mase_scale_bps": mase_scale,
        "best_epoch": int(best_epoch),
        "best_early_stop_mae_scaled": float(best_val),
        "parameter_count": parameter_count(model),
        "runtime_seconds": float(runtime_s),
        "status": "PASS",
    }

    return fit_record, predictions


def summarize_metrics(
    frame: pd.DataFrame,
    mase_scale: float,
) -> dict:
    yt = frame["y_true_bps"].to_numpy(float)
    yp = frame["y_pred_bps"].to_numpy(float)
    pp = frame["persistence_pred_bps"].to_numpy(float)

    err = yt - yp
    ae = np.abs(err)
    pae = np.abs(yt - pp)

    mae = float(ae.mean())
    persistence_mae = float(pae.mean())

    denom = np.abs(yt) + np.abs(yp)
    smape_terms = np.zeros_like(denom)
    mask = denom > 0
    smape_terms[mask] = (
        2 * ae[mask] / denom[mask]
    )

    return {
        "n_predictions": len(frame),
        "mae_bps": mae,
        "persistence_mae_bps": persistence_mae,
        "mae_ratio_vs_persistence": (
            mae / persistence_mae
        ),
        "skill_vs_persistence": (
            1 - mae / persistence_mae
        ),
        "rmse_bps": float(
            np.sqrt(
                np.mean(
                    err ** 2
                )
            )
        ),
        "smape_pct": float(
            100 * smape_terms.mean()
        ),
        "mase": float(
            mae / mase_scale
        ),
        "bias_bps": float(
            err.mean()
        ),
        "underprediction_pct": float(
            100 * np.mean(
                yt > yp
            )
        ),
        "p95_abs_error_bps": float(
            np.percentile(
                ae,
                95,
            )
        ),
    }


def validate_april_evidence(
    bundle_root: Path,
) -> tuple[pd.DataFrame, dict]:
    summary_path = (
        bundle_root
        / "evidence"
        / APRIL_CANDIDATE_SUMMARY_NAME
    )

    manifest_path = (
        bundle_root
        / "evidence"
        / APRIL_SCIENTIFIC_MANIFEST_NAME
    )

    check(
        "April candidate summary exists",
        summary_path.is_file(),
        str(summary_path),
    )

    check(
        "April candidate summary SHA-256",
        sha256(summary_path)
        == APRIL_CANDIDATE_SUMMARY_SHA,
        sha256(summary_path),
    )

    check(
        "April scientific manifest exists",
        manifest_path.is_file(),
        str(manifest_path),
    )

    check(
        "April scientific manifest SHA-256",
        sha256(manifest_path)
        == APRIL_SCIENTIFIC_MANIFEST_SHA,
        sha256(manifest_path),
    )

    summary = pd.read_csv(
        summary_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    check(
        "April candidate rows = 2",
        len(summary) == 2,
        str(len(summary)),
    )

    check(
        "selected_transformer_star column",
        "selected_transformer_star"
        in summary.columns,
    )

    selected = summary[
        summary[
            "selected_transformer_star"
        ].astype(bool)
    ]

    check(
        "Exactly one TRANSFORMER* row",
        len(selected) == 1,
        str(len(selected)),
    )

    row = selected.iloc[0]

    check(
        "TRANSFORMER* config exact T01",
        str(row["config_id"])
        == TRANSFORMER_STAR_CONFIG,
        str(row["config_id"]),
    )

    check(
        "TRANSFORMER* lookback exact 24",
        int(row["lookback"])
        == TRANSFORMER_STAR_LOOKBACK,
        str(row["lookback"]),
    )

    check(
        "TRANSFORMER* rank exact 1",
        int(row["rank"]) == 1,
        str(row["rank"]),
    )

    check(
        "April manifest campaign PASS",
        manifest.get("campaign_id")
        == "TRANSFORMER-APRIL-SELECTION-001"
        and manifest.get("status")
        == "PASS",
        repr(
            (
                manifest.get("campaign_id"),
                manifest.get("status"),
            )
        ),
    )

    winner = (
        manifest.get(
            "selection",
            {},
        ).get(
            "winner",
            {},
        )
    )

    check(
        "April manifest winner T01",
        winner.get("config_id")
        == TRANSFORMER_STAR_CONFIG,
        repr(winner),
    )

    check(
        "April manifest winner L24",
        winner.get("lookback")
        == TRANSFORMER_STAR_LOOKBACK,
        repr(winner),
    )

    gov = manifest.get(
        "governance",
        {},
    )

    check(
        "April manifest TRANSFORMER* selected",
        gov.get(
            "transformer_star_selected"
        )
        is True,
    )

    check(
        "April manifest stability not run",
        gov.get(
            "stability_run"
        )
        is False,
    )

    check(
        "April manifest freeze not run",
        gov.get(
            "freeze_run"
        )
        is False,
    )

    check(
        "April manifest June absent",
        gov.get(
            "june_present"
        )
        is False,
    )

    return (
        summary,
        manifest,
    )


def preflight(bundle_root: Path):
    print("=" * 120)
    print(
        f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO"
    )
    print("=" * 120)

    print(
        f"Runner version: {RUNNER_VERSION}"
    )

    print(
        "Runner SHA-256:",
        sha256(
            Path(__file__).resolve()
        ),
    )

    print()

    software = validate_software()

    forbidden = find_forbidden_june(
        bundle_root
    )

    check(
        "June physically absent",
        not forbidden,
        repr(
            [
                str(p)
                for p in forbidden
            ]
        ),
    )

    authority_files = [
        (
            PROTOCOL_NAME,
            PROTOCOL_SHA,
            "033",
        ),
        (
            PREFLIGHT_CLOSURE_NAME,
            PREFLIGHT_CLOSURE_SHA,
            "034",
        ),
        (
            MARCH_RUNNER_PREFLIGHT_NAME,
            MARCH_RUNNER_PREFLIGHT_SHA,
            "035",
        ),
        (
            RUNTIME_COMPATIBILITY_NAME,
            RUNTIME_COMPATIBILITY_SHA,
            "036",
        ),
        (
            MARCH_CLOSURE_NAME,
            MARCH_CLOSURE_SHA,
            "037",
        ),
        (
            APRIL_PREFLIGHT_CLOSURE_NAME,
            APRIL_PREFLIGHT_CLOSURE_SHA,
            "038",
        ),
        (
            APRIL_CLOSURE_NAME,
            APRIL_CLOSURE_SHA,
            "039",
        ),
    ]

    for (
        name,
        expected_sha,
        label,
    ) in authority_files:
        path = (
            bundle_root
            / "governance"
            / name
        )

        check(
            f"{label} exists",
            path.is_file(),
            str(path),
        )

        check(
            f"{label} SHA-256",
            sha256(path)
            == expected_sha,
            sha256(path),
        )

    april_summary, april_manifest = (
        validate_april_evidence(
            bundle_root
        )
    )

    april = validate_frame(
        bundle_root
        / "inputs"
        / APRIL_NAME,
        APRIL_SHA,
        APRIL_ROWS,
        APRIL_START,
        APRIL_END,
    )

    for h in HORIZONS:
        n_common = len(
            common_sample_indices(
                len(april),
                h,
            )
        )

        check(
            f"April common count H{h}",
            n_common
            == EXPECTED_APRIL_COMMON_COUNTS[h],
            str(n_common),
        )

        (
            fit_idx,
            gap1,
            early_idx,
            gap2,
            selection_idx,
        ) = april_split_indices(
            n_common
        )

        observed = (
            len(fit_idx),
            len(early_idx),
            len(selection_idx),
        )

        check(
            f"H{h} frozen April split",
            observed
            == EXPECTED_APRIL_SPLITS[h],
            repr(observed),
        )

        check(
            f"H{h} first gap = 12",
            len(gap1) == GAP,
            str(len(gap1)),
        )

        check(
            f"H{h} second gap = 12",
            len(gap2) == GAP,
            str(len(gap2)),
        )

        check(
            f"H{h} fit->early separation",
            int(
                early_idx[0]
                - fit_idx[-1]
                - 1
            )
            == GAP,
        )

        check(
            f"H{h} early->selection separation",
            int(
                selection_idx[0]
                - early_idx[-1]
                - 1
            )
            == GAP,
        )

    check(
        "TRANSFORMER* exact T01/L24",
        TRANSFORMER_STAR_CONFIG
        == "T01"
        and TRANSFORMER_STAR_LOOKBACK
        == 24,
    )

    check(
        "Stability seeds exact",
        STABILITY_SEEDS
        == (
            20260820,
            20260821,
            20260822,
        ),
        repr(STABILITY_SEEDS),
    )

    check(
        "Canonical seed remains 20260820",
        CANONICAL_SEED
        == 20260820,
        str(CANONICAL_SEED),
    )

    device = torch.device(
        "cuda"
    )

    for seed in STABILITY_SEEDS:
        set_seed(seed)

        model = TransformerLite().to(
            device
        )

        check(
            f"seed={seed}: parameter count",
            parameter_count(model)
            == EXPECTED_PARAMETER_COUNT,
            str(
                parameter_count(
                    model
                )
            ),
        )

        model.eval()

        with torch.no_grad():
            out = model(
                torch.randn(
                    4,
                    TRANSFORMER_STAR_LOOKBACK,
                    1,
                    device=device,
                )
            )

        check(
            f"seed={seed}: forward shape",
            tuple(out.shape)
            == (4, 1),
            str(
                tuple(
                    out.shape
                )
            ),
        )

        check(
            f"seed={seed}: forward finite",
            torch.isfinite(
                out
            ).all().item(),
        )

    check(
        "expected fit count",
        len(STABILITY_SEEDS)
        * len(HORIZONS)
        == EXPECTED_FITS,
        str(
            len(STABILITY_SEEDS)
            * len(HORIZONS)
        ),
    )

    print(
        "TRAINING EXECUTED IN PREFLIGHT: 0"
    )

    print(
        "SCIENTIFIC ARTIFACTS WRITTEN IN PREFLIGHT: 0"
    )

    print(
        "BEST-SEED SELECTION: NO"
    )

    print(
        "CANONICAL SEED REMAINS: 20260820"
    )

    print(
        "TRANSFORMER-FREEZE-001: NOT RUN"
    )

    print(
        "TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED"
    )

    print(
        f"{CAMPAIGN_ID} PREFLIGHT: PASS"
    )

    return (
        software,
        april,
        april_summary,
        april_manifest,
    )


def deterministic_zip(
    zip_path: Path,
    members: list[Path],
    arc_root: Path,
) -> None:
    fixed_time = (
        2026,
        8,
        22,
        12,
        0,
        0,
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for path in sorted(
            members,
            key=lambda p: (
                p.relative_to(
                    arc_root
                ).as_posix()
            ),
        ):
            arcname = path.relative_to(
                arc_root
            ).as_posix()

            info = zipfile.ZipInfo(
                arcname
            )

            info.date_time = fixed_time
            info.compress_type = (
                zipfile.ZIP_DEFLATED
            )
            info.external_attr = (
                0o644 << 16
            )

            zf.writestr(
                info,
                path.read_bytes(),
            )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            (str(_TFM_PUBLIC_COLAB_WORKDIR) + '/transformer_stability_001_outputs_v1_2026-08-22')
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    bundle_root = (
        args.bundle_root.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    runner_sha = sha256(
        Path(__file__).resolve()
    )

    (
        software,
        april,
        april_summary,
        april_manifest,
    ) = preflight(
        bundle_root
    )

    if args.preflight_only:
        return 0

    del april_summary
    del april_manifest

    expected_paths = {
        "fit_runtime": (
            output_dir
            / f"{OUTPUT_PREFIX}_fit_runtime_{OUTPUT_SUFFIX}.csv"
        ),
        "predictions": (
            output_dir
            / f"{OUTPUT_PREFIX}_predictions_{OUTPUT_SUFFIX}.parquet"
        ),
        "horizon_metrics": (
            output_dir
            / f"{OUTPUT_PREFIX}_horizon_metrics_{OUTPUT_SUFFIX}.csv"
        ),
        "seed_summary": (
            output_dir
            / f"{OUTPUT_PREFIX}_seed_summary_{OUTPUT_SUFFIX}.csv"
        ),
        "stability_summary": (
            output_dir
            / f"{OUTPUT_PREFIX}_summary_{OUTPUT_SUFFIX}.csv"
        ),
        "report": (
            output_dir
            / f"{OUTPUT_PREFIX}_report_{OUTPUT_SUFFIX}.txt"
        ),
        "manifest": (
            output_dir
            / f"{OUTPUT_PREFIX}_manifest_{OUTPUT_SUFFIX}.json"
        ),
        "outputs_zip": (
            output_dir
            / f"{OUTPUT_PREFIX}_outputs_{OUTPUT_SUFFIX}.zip"
        ),
    }

    if output_dir.exists():
        existing = [
            p
            for p in output_dir.rglob("*")
            if p.is_file()
        ]

        if existing:
            raise FileExistsError(
                "Output directory already contains files; "
                "refusing overwrite:\n"
                + "\n".join(
                    map(
                        str,
                        existing,
                    )
                )
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y = april[
        TARGET
    ].to_numpy(
        np.float64
    )

    timestamps = pd.to_datetime(
        april[
            TIME_COL
        ]
    ).reset_index(
        drop=True
    )

    device = torch.device(
        "cuda"
    )

    fit_records = []
    pred_frames = []

    started = time.perf_counter()

    print("=" * 120)
    print(
        f"START {CAMPAIGN_ID}"
    )
    print(
        f"TRANSFORMER* = "
        f"{TRANSFORMER_STAR_CONFIG}/"
        f"L{TRANSFORMER_STAR_LOOKBACK}"
    )
    print(
        "Stability seeds = "
        "20260820, 20260821, 20260822"
    )
    print(
        "Expected fits = 12"
    )
    print(
        "April split reuse = YES"
    )
    print(
        "Best-seed selection = NO"
    )
    print(
        "Canonical seed remains = 20260820"
    )
    print(
        "Freeze = NOT RUN"
    )
    print(
        "June present = NO"
    )
    print("=" * 120)

    for seed in STABILITY_SEEDS:
        for horizon in HORIZONS:
            n_common = len(
                common_sample_indices(
                    len(april),
                    horizon,
                )
            )

            (
                fit_idx,
                gap1,
                early_idx,
                gap2,
                selection_idx,
            ) = april_split_indices(
                n_common
            )

            del gap1
            del gap2

            print(
                f"[FIT] seed={seed} "
                f"H{horizon}",
                flush=True,
            )

            (
                fit_record,
                predictions,
            ) = train_one_fit(
                seed,
                horizon,
                y,
                fit_idx,
                early_idx,
                selection_idx,
                device,
            )

            predictions[
                "origin_timestamp"
            ] = timestamps.iloc[
                predictions[
                    "origin_index"
                ].to_numpy(int)
            ].to_numpy()

            predictions[
                "target_timestamp"
            ] = timestamps.iloc[
                predictions[
                    "target_index"
                ].to_numpy(int)
            ].to_numpy()

            fit_records.append(
                fit_record
            )

            pred_frames.append(
                predictions
            )

            print(
                "      PASS "
                f"epoch="
                f"{fit_record['best_epoch']} "
                f"runtime="
                f"{fit_record['runtime_seconds']:.2f}s",
                flush=True,
            )

    fit_runtime = pd.DataFrame(
        fit_records
    )

    predictions = pd.concat(
        pred_frames,
        ignore_index=True,
    )

    check(
        "fit count 12",
        len(fit_runtime)
        == EXPECTED_FITS,
        str(
            len(
                fit_runtime
            )
        ),
    )

    check(
        "all fit status PASS",
        fit_runtime[
            "status"
        ].eq(
            "PASS"
        ).all(),
    )

    check(
        "predictions rows 3048",
        len(predictions)
        == EXPECTED_PREDICTIONS,
        str(
            len(
                predictions
            )
        ),
    )

    check(
        "predictions finite",
        np.isfinite(
            predictions[
                "y_pred_bps"
            ].to_numpy(float)
        ).all(),
    )

    check(
        "April hash unchanged",
        sha256(
            bundle_root
            / "inputs"
            / APRIL_NAME
        )
        == APRIL_SHA,
    )

    authority_hashes = [
        (
            PROTOCOL_NAME,
            PROTOCOL_SHA,
            "033",
        ),
        (
            PREFLIGHT_CLOSURE_NAME,
            PREFLIGHT_CLOSURE_SHA,
            "034",
        ),
        (
            MARCH_RUNNER_PREFLIGHT_NAME,
            MARCH_RUNNER_PREFLIGHT_SHA,
            "035",
        ),
        (
            RUNTIME_COMPATIBILITY_NAME,
            RUNTIME_COMPATIBILITY_SHA,
            "036",
        ),
        (
            MARCH_CLOSURE_NAME,
            MARCH_CLOSURE_SHA,
            "037",
        ),
        (
            APRIL_PREFLIGHT_CLOSURE_NAME,
            APRIL_PREFLIGHT_CLOSURE_SHA,
            "038",
        ),
        (
            APRIL_CLOSURE_NAME,
            APRIL_CLOSURE_SHA,
            "039",
        ),
    ]

    for (
        name,
        expected_sha,
        label,
    ) in authority_hashes:
        check(
            f"{label} hash unchanged",
            sha256(
                bundle_root
                / "governance"
                / name
            )
            == expected_sha,
        )

    check(
        "April candidate summary unchanged",
        sha256(
            bundle_root
            / "evidence"
            / APRIL_CANDIDATE_SUMMARY_NAME
        )
        == APRIL_CANDIDATE_SUMMARY_SHA,
    )

    check(
        "April scientific manifest unchanged",
        sha256(
            bundle_root
            / "evidence"
            / APRIL_SCIENTIFIC_MANIFEST_NAME
        )
        == APRIL_SCIENTIFIC_MANIFEST_SHA,
    )

    check(
        "June still absent",
        not find_forbidden_june(
            bundle_root
        ),
    )

    horizon_rows = []

    for (
        seed,
        horizon,
    ), sub in predictions.groupby(
        [
            "seed",
            "horizon_steps",
        ],
        sort=True,
    ):
        fit_row = fit_runtime[
            fit_runtime[
                "seed"
            ].eq(
                seed
            )
            & fit_runtime[
                "horizon_steps"
            ].eq(
                horizon
            )
        ]

        check(
            f"seed={seed}/H{horizon}: one fit record",
            len(fit_row) == 1,
            str(
                len(
                    fit_row
                )
            ),
        )

        mase_scale = float(
            fit_row[
                "mase_scale_bps"
            ].iloc[0]
        )

        horizon_rows.append(
            {
                "seed": int(seed),
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": TRANSFORMER_STAR_LOOKBACK,
                "horizon_steps": int(horizon),
                **summarize_metrics(
                    sub,
                    mase_scale,
                ),
            }
        )

    horizon_metrics = pd.DataFrame(
        horizon_rows
    )

    check(
        "horizon metric rows 12",
        len(horizon_metrics)
        == EXPECTED_HORIZON_METRIC_ROWS,
        str(
            len(
                horizon_metrics
            )
        ),
    )

    seed_rows = []

    for seed, sub in (
        horizon_metrics.groupby(
            "seed",
            sort=True,
        )
    ):
        ratios = {
            int(
                row.horizon_steps
            ): float(
                row.mae_ratio_vs_persistence
            )
            for row
            in sub.itertuples()
        }

        check(
            f"seed={seed}: four horizons",
            set(
                ratios
            )
            == set(
                HORIZONS
            ),
        )

        seed_rows.append(
            {
                "seed": int(seed),
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": TRANSFORMER_STAR_LOOKBACK,
                "score": float(
                    np.mean(
                        [
                            ratios[h]
                            for h
                            in HORIZONS
                        ]
                    )
                ),
                "ratio_H1": ratios[1],
                "ratio_H3": ratios[3],
                "ratio_H6": ratios[6],
                "ratio_H12": ratios[12],
                "is_canonical_seed": bool(
                    int(seed)
                    == CANONICAL_SEED
                ),
            }
        )

    seed_summary = pd.DataFrame(
        seed_rows
    ).sort_values(
        "seed",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    check(
        "seed summary rows 3",
        len(seed_summary)
        == EXPECTED_SEED_SUMMARY_ROWS,
        str(
            len(
                seed_summary
            )
        ),
    )

    check(
        "exactly one canonical seed row",
        int(
            seed_summary[
                "is_canonical_seed"
            ].sum()
        )
        == 1,
    )

    check(
        "canonical seed row = 20260820",
        int(
            seed_summary.loc[
                seed_summary[
                    "is_canonical_seed"
                ],
                "seed",
            ].iloc[0]
        )
        == CANONICAL_SEED,
    )

    scores = seed_summary[
        "score"
    ].to_numpy(
        float
    )

    stability_summary = pd.DataFrame(
        [
            {
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": TRANSFORMER_STAR_LOOKBACK,
                "canonical_seed": CANONICAL_SEED,
                "n_seeds": len(STABILITY_SEEDS),
                "n_horizons": len(HORIZONS),
                "n_fits": EXPECTED_FITS,
                "mean_score": float(
                    np.mean(
                        scores
                    )
                ),
                "std_score_ddof0": float(
                    np.std(
                        scores,
                        ddof=0,
                    )
                ),
                "min_score": float(
                    np.min(
                        scores
                    )
                ),
                "max_score": float(
                    np.max(
                        scores
                    )
                ),
                "score_range": float(
                    np.max(
                        scores
                    )
                    - np.min(
                        scores
                    )
                ),
                "best_seed_selection": False,
                "canonical_seed_changed": False,
            }
        ]
    )

    check(
        "stability summary rows 1",
        len(stability_summary)
        == EXPECTED_STABILITY_SUMMARY_ROWS,
        str(
            len(
                stability_summary
            )
        ),
    )

    check(
        "best-seed selection false",
        stability_summary[
            "best_seed_selection"
        ].eq(
            False
        ).all(),
    )

    check(
        "canonical seed unchanged",
        stability_summary[
            "canonical_seed_changed"
        ].eq(
            False
        ).all(),
    )

    report_lines = [
        "=" * 120,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 120,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"Protocol 033 SHA-256: {PROTOCOL_SHA}",
        f"Preflight closure 034 SHA-256: {PREFLIGHT_CLOSURE_SHA}",
        f"March runner preflight 035 SHA-256: {MARCH_RUNNER_PREFLIGHT_SHA}",
        f"Runtime compatibility 036 SHA-256: {RUNTIME_COMPATIBILITY_SHA}",
        f"March closure 037 SHA-256: {MARCH_CLOSURE_SHA}",
        f"April preflight closure 038 SHA-256: {APRIL_PREFLIGHT_CLOSURE_SHA}",
        f"April closure 039 SHA-256: {APRIL_CLOSURE_SHA}",
        "",
        "GOVERNANCE",
        "Stability role: seed robustness assessment",
        "TRANSFORMER*: T01 / L24",
        "April split reuse: YES",
        "Seeds: 20260820, 20260821, 20260822",
        "Best-seed selection: NO",
        "Canonical seed remains: 20260820",
        "Freeze: NOT RUN",
        "June present: NO",
        "June access authorized: NO",
        "",
        "SCORE PER SEED",
    ]

    for row in seed_summary.itertuples():
        report_lines.append(
            f"seed={int(row.seed)} "
            f"score={float(row.score):.9f} "
            f"H1={float(row.ratio_H1):.9f} "
            f"H3={float(row.ratio_H3):.9f} "
            f"H6={float(row.ratio_H6):.9f} "
            f"H12={float(row.ratio_H12):.9f}"
        )

    stability_row = stability_summary.iloc[0]

    report_lines.extend(
        [
            "",
            "STABILITY SUMMARY",
            f"mean Score: {float(stability_row['mean_score']):.9f}",
            f"std Score ddof=0: {float(stability_row['std_score_ddof0']):.9f}",
            f"Score range: {float(stability_row['score_range']):.9f}",
            f"min Score: {float(stability_row['min_score']):.9f}",
            f"max Score: {float(stability_row['max_score']):.9f}",
            "",
            f"Fits executed: {len(fit_runtime)}",
            f"Predictions: {len(predictions)}",
            f"Campaign seconds: {time.perf_counter() - started:.3f}",
            "STATUS: PASS",
            "TRANSFORMER-STABILITY-001: PASS",
            "BEST-SEED SELECTION: NO",
            "CANONICAL SEED: 20260820",
            "TRANSFORMER-FREEZE-001: NOT RUN",
            "TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED",
        ]
    )

    report = (
        "\n".join(
            report_lines
        )
        + "\n"
    )

    paths = expected_paths

    atomic_csv(
        fit_runtime,
        paths[
            "fit_runtime"
        ],
    )

    atomic_parquet(
        predictions,
        paths[
            "predictions"
        ],
    )

    atomic_csv(
        horizon_metrics,
        paths[
            "horizon_metrics"
        ],
    )

    atomic_csv(
        seed_summary,
        paths[
            "seed_summary"
        ],
    )

    atomic_csv(
        stability_summary,
        paths[
            "stability_summary"
        ],
    )

    atomic_text(
        report,
        paths[
            "report"
        ],
    )

    output_records = {
        key: {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for key, path
        in paths.items()
        if key not in {
            "manifest",
            "outputs_zip",
        }
    }

    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "runner": {
            "name": Path(__file__).name,
            "version": RUNNER_VERSION,
            "sha256": runner_sha,
        },
        "authority": {
            "protocol_033": {
                "name": PROTOCOL_NAME,
                "sha256": PROTOCOL_SHA,
            },
            "preflight_closure_034": {
                "name": PREFLIGHT_CLOSURE_NAME,
                "sha256": PREFLIGHT_CLOSURE_SHA,
            },
            "march_runner_preflight_035": {
                "name": MARCH_RUNNER_PREFLIGHT_NAME,
                "sha256": MARCH_RUNNER_PREFLIGHT_SHA,
            },
            "runtime_compatibility_036": {
                "name": RUNTIME_COMPATIBILITY_NAME,
                "sha256": RUNTIME_COMPATIBILITY_SHA,
            },
            "march_closure_037": {
                "name": MARCH_CLOSURE_NAME,
                "sha256": MARCH_CLOSURE_SHA,
            },
            "april_preflight_closure_038": {
                "name": APRIL_PREFLIGHT_CLOSURE_NAME,
                "sha256": APRIL_PREFLIGHT_CLOSURE_SHA,
            },
            "april_closure_039": {
                "name": APRIL_CLOSURE_NAME,
                "sha256": APRIL_CLOSURE_SHA,
            },
        },
        "governance": {
            "stability_role": "seed robustness assessment",
            "transformer_star_config": TRANSFORMER_STAR_CONFIG,
            "transformer_star_lookback": TRANSFORMER_STAR_LOOKBACK,
            "april_split_reused_exactly": True,
            "stability_seeds": list(STABILITY_SEEDS),
            "best_seed_selection": False,
            "canonical_seed": CANONICAL_SEED,
            "canonical_seed_changed": False,
            "freeze_run": False,
            "june_present": False,
            "june_access_authorized": False,
            "tuning": False,
            "architecture_search": False,
            "lookback_search": False,
        },
        "inputs": {
            "april": {
                "name": APRIL_NAME,
                "sha256": APRIL_SHA,
                "rows": APRIL_ROWS,
                "role": "stability / exact April split reuse",
                "used_for_training": True,
                "used_for_scoring": True,
            },
            "april_candidate_summary": {
                "name": APRIL_CANDIDATE_SUMMARY_NAME,
                "sha256": APRIL_CANDIDATE_SUMMARY_SHA,
                "role": "frozen TRANSFORMER* evidence only",
                "used_for_training_or_scoring": False,
            },
            "april_scientific_manifest": {
                "name": APRIL_SCIENTIFIC_MANIFEST_NAME,
                "sha256": APRIL_SCIENTIFIC_MANIFEST_SHA,
                "role": "frozen provenance evidence only",
                "used_for_training_or_scoring": False,
            },
            "june_present": False,
        },
        "software": software,
        "architecture": {
            "family": "TransformerEncoder-lite",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "encoder_dropout": 0.10,
            "external_dropout": 0.10,
            "activation": "GELU",
            "batch_first": True,
            "norm_first": True,
            "positional_encoding": "fixed_sinusoidal",
            "aggregation": "last_temporal_representation",
            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        },
        "transformer_star": {
            "config_id": TRANSFORMER_STAR_CONFIG,
            "lookback": TRANSFORMER_STAR_LOOKBACK,
        },
        "stability": {
            "seeds": list(STABILITY_SEEDS),
            "canonical_seed": CANONICAL_SEED,
            "horizons": list(HORIZONS),
            "expected_fits": EXPECTED_FITS,
            "best_seed_selection": False,
            "summary": {
                "mean_score": float(
                    stability_row[
                        "mean_score"
                    ]
                ),
                "std_score_ddof0": float(
                    stability_row[
                        "std_score_ddof0"
                    ]
                ),
                "min_score": float(
                    stability_row[
                        "min_score"
                    ]
                ),
                "max_score": float(
                    stability_row[
                        "max_score"
                    ]
                ),
                "score_range": float(
                    stability_row[
                        "score_range"
                    ]
                ),
            },
            "score_per_seed": [
                {
                    "seed": int(
                        row.seed
                    ),
                    "score": float(
                        row.score
                    ),
                    "ratio_H1": float(
                        row.ratio_H1
                    ),
                    "ratio_H3": float(
                        row.ratio_H3
                    ),
                    "ratio_H6": float(
                        row.ratio_H6
                    ),
                    "ratio_H12": float(
                        row.ratio_H12
                    ),
                    "is_canonical_seed": bool(
                        row.is_canonical_seed
                    ),
                }
                for row
                in seed_summary.itertuples()
            ],
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "L1Loss",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "gradient_clip_norm": MAX_GRAD_NORM,
            "dataloader_shuffle": False,
            "hard_fit_seconds": HARD_FIT_SECONDS,
            "gap_before_early_stop": GAP,
            "gap_before_selection": GAP,
        },
        "validation": {
            "fit_count_12": (
                len(
                    fit_runtime
                )
                == EXPECTED_FITS
            ),
            "all_fit_status_pass": bool(
                fit_runtime[
                    "status"
                ].eq(
                    "PASS"
                ).all()
            ),
            "prediction_rows_3048": (
                len(
                    predictions
                )
                == EXPECTED_PREDICTIONS
            ),
            "horizon_metric_rows_12": (
                len(
                    horizon_metrics
                )
                == EXPECTED_HORIZON_METRIC_ROWS
            ),
            "seed_summary_rows_3": (
                len(
                    seed_summary
                )
                == EXPECTED_SEED_SUMMARY_ROWS
            ),
            "stability_summary_rows_1": (
                len(
                    stability_summary
                )
                == EXPECTED_STABILITY_SUMMARY_ROWS
            ),
            "best_seed_selection_false": (
                stability_summary[
                    "best_seed_selection"
                ].eq(
                    False
                ).all()
            ),
            "canonical_seed_unchanged": (
                stability_summary[
                    "canonical_seed_changed"
                ].eq(
                    False
                ).all()
            ),
            "april_hash_unchanged": (
                sha256(
                    bundle_root
                    / "inputs"
                    / APRIL_NAME
                )
                == APRIL_SHA
            ),
            "april_candidate_summary_hash_unchanged": (
                sha256(
                    bundle_root
                    / "evidence"
                    / APRIL_CANDIDATE_SUMMARY_NAME
                )
                == APRIL_CANDIDATE_SUMMARY_SHA
            ),
            "april_scientific_manifest_hash_unchanged": (
                sha256(
                    bundle_root
                    / "evidence"
                    / APRIL_SCIENTIFIC_MANIFEST_NAME
                )
                == APRIL_SCIENTIFIC_MANIFEST_SHA
            ),
            "protocol_033_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / PROTOCOL_NAME
                )
                == PROTOCOL_SHA
            ),
            "closure_034_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / PREFLIGHT_CLOSURE_NAME
                )
                == PREFLIGHT_CLOSURE_SHA
            ),
            "closure_035_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / MARCH_RUNNER_PREFLIGHT_NAME
                )
                == MARCH_RUNNER_PREFLIGHT_SHA
            ),
            "closure_036_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / RUNTIME_COMPATIBILITY_NAME
                )
                == RUNTIME_COMPATIBILITY_SHA
            ),
            "closure_037_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / MARCH_CLOSURE_NAME
                )
                == MARCH_CLOSURE_SHA
            ),
            "closure_038_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / APRIL_PREFLIGHT_CLOSURE_NAME
                )
                == APRIL_PREFLIGHT_CLOSURE_SHA
            ),
            "closure_039_hash_unchanged": (
                sha256(
                    bundle_root
                    / "governance"
                    / APRIL_CLOSURE_NAME
                )
                == APRIL_CLOSURE_SHA
            ),
            "june_absent_after_run": (
                not bool(
                    find_forbidden_june(
                        bundle_root
                    )
                )
            ),
        },
        "campaign_seconds": float(
            time.perf_counter()
            - started
        ),
        "outputs": output_records,
    }

    if not all(
        manifest[
            "validation"
        ].values()
    ):
        raise RuntimeError(
            "Final validation failed"
        )

    atomic_json(
        manifest,
        paths[
            "manifest"
        ],
    )

    deterministic_zip(
        paths[
            "outputs_zip"
        ],
        [
            path
            for key, path
            in paths.items()
            if key
            != "outputs_zip"
        ],
        output_dir,
    )

    print(report)

    print(
        "VALIDATIONS"
    )

    for key, value in (
        manifest[
            "validation"
        ].items()
    ):
        print(
            f"{key:72s}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print(
        "Outputs ZIP:",
        paths[
            "outputs_zip"
        ],
    )

    print(
        "Outputs ZIP SHA-256:",
        sha256(
            paths[
                "outputs_zip"
            ]
        ),
    )

    print(
        "TRANSFORMER-STABILITY-001: PASS"
    )

    print(
        "TRANSFORMER*: T01 / L24"
    )

    print(
        "BEST-SEED SELECTION: NO"
    )

    print(
        "CANONICAL SEED: 20260820"
    )

    print(
        "TRANSFORMER-FREEZE-001: NOT RUN"
    )

    print(
        "TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
