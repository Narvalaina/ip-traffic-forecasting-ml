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
CAMPAIGN_ID = "TRANSFORMER-APRIL-SELECTION-001"
PARENT_CAMPAIGN_ID = "UGR16-TRANSFORMER-LITE-001"

PROTOCOL_NAME = "033_phase_e_transformer_lite_protocol_2026-08-21.md"
PROTOCOL_SHA = "4fcd61519612ae5fd3daa6d874a73df50b27b63531ac109f0782351875211552"

PREFLIGHT_CLOSURE_NAME = "034_phase_e_colab_preflight_closure_2026-08-21.md"
PREFLIGHT_CLOSURE_SHA = "9d89f0bbaa13448c56217c05f1663fa57fdb4505d3a395860186e2235e9cc081"

RUNNER_PREFLIGHT_CLOSURE_NAME = "035_transformer_march_runner_preflight_closure_2026-08-21.md"
RUNNER_PREFLIGHT_CLOSURE_SHA = "2f90c65e1855c98a20470d76229be68359cad8fb33b0b213c5f351eb9e21322e"

RUNTIME_COMPATIBILITY_NAME = "036_phase_e_colab_runtime_compatibility_2026-08-21.md"
RUNTIME_COMPATIBILITY_SHA = "8e8aeb3b0783c5928468bbbf074c1c42359837b57461890c4eb5e1145ca9dbf3"

MARCH_CLOSURE_NAME = "037_transformer_march_screen_closure_2026-08-22.md"
MARCH_CLOSURE_SHA = "9650e5aeb7ea8653e17baaa584237da8ab3d1fb41db9a150b0f98e29e075e03e"

MARCH_CANDIDATE_SUMMARY_NAME = (
    "ugr16_transformer_march_screen_candidate_summary_v2_2026-08-22.csv"
)
MARCH_CANDIDATE_SUMMARY_SHA = (
    "c461bce8d18c2b194a8c5f87db66a5efc3b29557522504bc75e6c34c0cea7e90"
)

MARCH_SCIENTIFIC_MANIFEST_NAME = (
    "ugr16_transformer_march_screen_manifest_v2_2026-08-22.json"
)
MARCH_SCIENTIFIC_MANIFEST_SHA = (
    "44853373cb979b8ce63f56aa9c0020c4a82ecc60821b5156e82637aa04b6357f"
)

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

FROZEN_CANDIDATES = {
    "T04": {"lookback": 288, "march_rank": 1},
    "T01": {"lookback": 24, "march_rank": 2},
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

OUTPUT_PREFIX = "ugr16_transformer_april_selection"
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
    print(f"{name:58s}: {status}" + (f" | {detail}" if detail else ""))
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
    check(
        "software_python",
        info["python"] == EXPECTED_SOFTWARE["python"],
        info["python"],
    )
    check(
        "software_torch",
        info["torch"] == EXPECTED_SOFTWARE["torch"],
        info["torch"],
    )
    check(
        "software_torch_cuda",
        info["torch_cuda"] == EXPECTED_SOFTWARE["torch_cuda"],
        str(info["torch_cuda"]),
    )
    check(
        "software_cudnn",
        info["cudnn"] == EXPECTED_SOFTWARE["cudnn"],
        str(info["cudnn"]),
    )
    check(
        "software_numpy",
        info["numpy"] == EXPECTED_SOFTWARE["numpy"],
        info["numpy"],
    )
    check(
        "software_pandas",
        info["pandas"] == EXPECTED_SOFTWARE["pandas"],
        info["pandas"],
    )
    check(
        "cuda_available",
        info["cuda_available"],
        str(info["gpu_name"]),
    )
    check(
        "cuda_device_count_positive",
        info["device_count"] >= 1,
        str(info["device_count"]),
    )
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
    check(
        f"{path.name}: start",
        str(ts.min()) == expected_start,
        str(ts.min()),
    )
    check(
        f"{path.name}: end",
        str(ts.max()) == expected_end,
        str(ts.max()),
    )

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
        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, d_model, 2, dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(
            max_len, d_model, dtype=torch.float32
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe.unsqueeze(0), persistent=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[
            :, : x.size(1), :
        ].to(dtype=x.dtype, device=x.device)


class TransformerLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(1, 32, bias=True)
        self.positional_encoding = (
            FixedSinusoidalPositionalEncoding(32, 288)
        )

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
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
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
    config_id: str,
    horizon: int,
    y: np.ndarray,
    fit_idx: np.ndarray,
    early_idx: np.ndarray,
    selection_idx: np.ndarray,
    device,
):
    lookback = int(
        FROZEN_CANDIDATES[config_id]["lookback"]
    )

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
        y, fit_idx, lookback, horizon, scaler
    )
    x_early, y_early, _, _ = build_xy(
        y, early_idx, lookback, horizon, scaler
    )
    x_selection, _, origins_sel, targets_sel = build_xy(
        y, selection_idx, lookback, horizon, scaler
    )

    set_seed(CANONICAL_SEED)
    model = TransformerLite().to(device)

    check(
        f"{config_id}/H{horizon}: parameter count",
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        str(parameter_count(model)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
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
            if (
                time.perf_counter() - started
                > HARD_FIT_SECONDS
            ):
                raise TimeoutError(
                    f"RESOURCE_LIMIT {config_id}/H{horizon}"
                )

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)

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
            model, early_loader, device
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
            f"RESOURCE_LIMIT {config_id}/H{horizon}"
        )

    if best_state is None:
        raise RuntimeError("No valid checkpoint")

    model.load_state_dict(best_state)

    y_pred = scaler.inverse(
        predict_scaled(
            model, x_selection, device
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
            "config_id": config_id,
            "lookback": lookback,
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
        raise RuntimeError("Invalid MASE scale")

    fit_record = {
        "config_id": config_id,
        "lookback": lookback,
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
            np.sqrt(np.mean(err ** 2))
        ),
        "smape_pct": float(
            100 * smape_terms.mean()
        ),
        "mase": float(
            mae / mase_scale
        ),
        "bias_bps": float(err.mean()),
        "underprediction_pct": float(
            100 * np.mean(yt > yp)
        ),
        "p95_abs_error_bps": float(
            np.percentile(ae, 95)
        ),
    }


def validate_march_evidence(
    bundle_root: Path,
) -> tuple[pd.DataFrame, dict]:
    summary_path = (
        bundle_root
        / "evidence"
        / MARCH_CANDIDATE_SUMMARY_NAME
    )
    manifest_path = (
        bundle_root
        / "evidence"
        / MARCH_SCIENTIFIC_MANIFEST_NAME
    )

    check(
        "March candidate summary exists",
        summary_path.is_file(),
        str(summary_path),
    )
    check(
        "March candidate summary SHA-256",
        sha256(summary_path)
        == MARCH_CANDIDATE_SUMMARY_SHA,
        sha256(summary_path),
    )
    check(
        "March scientific manifest exists",
        manifest_path.is_file(),
        str(manifest_path),
    )
    check(
        "March scientific manifest SHA-256",
        sha256(manifest_path)
        == MARCH_SCIENTIFIC_MANIFEST_SHA,
        sha256(manifest_path),
    )

    summary = pd.read_csv(summary_path)
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    check(
        "March candidate rows = 4",
        len(summary) == 4,
        str(len(summary)),
    )
    check(
        "March selected_top2 column",
        "selected_top2" in summary.columns,
    )

    selected = summary[
        summary["selected_top2"].astype(bool)
    ].sort_values("rank")

    observed = [
        (
            str(r.config_id),
            int(r.lookback),
            int(r.rank),
        )
        for r in selected.itertuples()
    ]
    expected = [
        ("T04", 288, 1),
        ("T01", 24, 2),
    ]

    check(
        "Frozen March top-2 exact",
        observed == expected,
        repr(observed),
    )
    check(
        "March manifest campaign PASS",
        manifest.get("campaign_id")
        == "TRANSFORMER-MARCH-SCREEN-001"
        and manifest.get("status") == "PASS",
        repr(
            (
                manifest.get("campaign_id"),
                manifest.get("status"),
            )
        ),
    )
    check(
        "March manifest selected top-2",
        manifest.get(
            "governance", {}
        ).get(
            "top2_for_april_selected"
        )
        is True,
    )
    check(
        "March manifest TRANSFORMER* absent",
        manifest.get(
            "governance", {}
        ).get(
            "transformer_star_selected"
        )
        is False,
    )

    return summary, manifest


def preflight(bundle_root: Path):
    print("=" * 120)
    print(
        f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO"
    )
    print("=" * 120)
    print(f"Runner version: {RUNNER_VERSION}")
    print(
        "Runner SHA-256:",
        sha256(Path(__file__).resolve()),
    )
    print()

    software = validate_software()

    forbidden = find_forbidden_june(
        bundle_root
    )
    check(
        "June physically absent",
        not forbidden,
        repr([str(p) for p in forbidden]),
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
            RUNNER_PREFLIGHT_CLOSURE_NAME,
            RUNNER_PREFLIGHT_CLOSURE_SHA,
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
    ]

    for name, expected_sha, label in authority_files:
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
            sha256(path) == expected_sha,
            sha256(path),
        )

    march_summary, march_manifest = (
        validate_march_evidence(
            bundle_root
        )
    )

    april = validate_frame(
        bundle_root / "inputs" / APRIL_NAME,
        APRIL_SHA,
        APRIL_ROWS,
        APRIL_START,
        APRIL_END,
    )

    for h in HORIZONS:
        n_common = len(
            common_sample_indices(
                len(april), h
            )
        )
        check(
            f"April common count H{h}",
            n_common
            == EXPECTED_APRIL_COMMON_COUNTS[h],
            str(n_common),
        )

        fit_idx, gap1, early_idx, gap2, selection_idx = (
            april_split_indices(
                n_common
            )
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
        "Frozen candidates exact",
        tuple(FROZEN_CANDIDATES.keys())
        == ("T04", "T01"),
        repr(FROZEN_CANDIDATES),
    )
    check(
        "Frozen lookbacks exact",
        [
            FROZEN_CANDIDATES["T04"]["lookback"],
            FROZEN_CANDIDATES["T01"]["lookback"],
        ]
        == [288, 24],
    )

    device = torch.device("cuda")

    for config_id in ("T04", "T01"):
        set_seed(CANONICAL_SEED)
        model = TransformerLite().to(
            device
        )

        check(
            f"{config_id}: parameter count",
            parameter_count(model)
            == EXPECTED_PARAMETER_COUNT,
            str(parameter_count(model)),
        )

        lookback = int(
            FROZEN_CANDIDATES[
                config_id
            ]["lookback"]
        )

        model.eval()
        with torch.no_grad():
            out = model(
                torch.randn(
                    4,
                    lookback,
                    1,
                    device=device,
                )
            )

        check(
            f"{config_id}: forward shape",
            tuple(out.shape) == (4, 1),
            str(tuple(out.shape)),
        )
        check(
            f"{config_id}: forward finite",
            torch.isfinite(
                out
            ).all().item(),
        )

    expected_fits = (
        len(FROZEN_CANDIDATES)
        * len(HORIZONS)
    )
    check(
        "expected fit count",
        expected_fits == 8,
        str(expected_fits),
    )

    print("TRAINING EXECUTED IN PREFLIGHT: 0")
    print(
        "SCIENTIFIC ARTIFACTS WRITTEN IN PREFLIGHT: 0"
    )
    print(
        "TRANSFORMER*: NOT SELECTED"
    )
    print(
        "TRANSFORMER-STABILITY-001: NOT RUN"
    )
    print(
        "TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED"
    )
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")

    return (
        software,
        april,
        march_summary,
        march_manifest,
    )


def deterministic_zip(
    zip_path: Path,
    members: list[Path],
    arc_root: Path,
) -> None:
    fixed_time = (2026, 8, 22, 12, 0, 0)

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
            (str(_TFM_PUBLIC_COLAB_WORKDIR) + '/transformer_april_selection_001_outputs_v1_2026-08-22')
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = parser.parse_args()

    bundle_root = (
        args.bundle_root
        .resolve()
    )
    output_dir = (
        args.output_dir
        .resolve()
    )

    runner_sha = sha256(
        Path(__file__).resolve()
    )

    (
        software,
        april,
        march_summary,
        march_manifest,
    ) = preflight(bundle_root)

    if args.preflight_only:
        return 0

    del march_summary
    del march_manifest

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
        "candidate_summary": (
            output_dir
            / f"{OUTPUT_PREFIX}_candidate_summary_{OUTPUT_SUFFIX}.csv"
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
                "Output directory already "
                "contains files; refusing overwrite:\n"
                + "\n".join(
                    map(str, existing)
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
        april[TIME_COL]
    ).reset_index(
        drop=True
    )

    device = torch.device("cuda")

    fit_records = []
    pred_frames = []

    started = time.perf_counter()

    print("=" * 120)
    print(f"START {CAMPAIGN_ID}")
    print("Expected fits = 8")
    print(
        "Frozen candidates = "
        "T04/L288, T01/L24"
    )
    print(
        "April training/scoring = YES"
    )
    print(
        "March retraining/scoring = NO"
    )
    print(
        "Stability = NOT RUN"
    )
    print(
        "June present = NO"
    )
    print("=" * 120)

    for config_id in (
        "T04",
        "T01",
    ):
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
                f"[FIT] {config_id} "
                f"H{horizon}",
                flush=True,
            )

            fit_record, predictions = (
                train_one_fit(
                    config_id,
                    horizon,
                    y,
                    fit_idx,
                    early_idx,
                    selection_idx,
                    device,
                )
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
                f"epoch={fit_record['best_epoch']} "
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
        "fit count 8",
        len(fit_runtime) == 8,
        str(len(fit_runtime)),
    )
    check(
        "all fit status PASS",
        fit_runtime[
            "status"
        ].eq("PASS").all(),
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
            RUNNER_PREFLIGHT_CLOSURE_NAME,
            RUNNER_PREFLIGHT_CLOSURE_SHA,
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
    ]

    for name, expected_sha, label in authority_hashes:
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
        "March evidence summary unchanged",
        sha256(
            bundle_root
            / "evidence"
            / MARCH_CANDIDATE_SUMMARY_NAME
        )
        == MARCH_CANDIDATE_SUMMARY_SHA,
    )

    check(
        "March scientific manifest unchanged",
        sha256(
            bundle_root
            / "evidence"
            / MARCH_SCIENTIFIC_MANIFEST_NAME
        )
        == MARCH_SCIENTIFIC_MANIFEST_SHA,
    )

    check(
        "June still absent",
        not find_forbidden_june(
            bundle_root
        ),
    )

    horizon_rows = []

    for (
        config_id,
        horizon,
    ), sub in predictions.groupby(
        [
            "config_id",
            "horizon_steps",
        ],
        sort=True,
    ):
        fit_row = fit_runtime[
            fit_runtime[
                "config_id"
            ].eq(config_id)
            & fit_runtime[
                "horizon_steps"
            ].eq(horizon)
        ]

        check(
            f"{config_id}/H{horizon}: one fit record",
            len(fit_row) == 1,
            str(len(fit_row)),
        )

        mase_scale = float(
            fit_row[
                "mase_scale_bps"
            ].iloc[0]
        )

        horizon_rows.append(
            {
                "config_id": config_id,
                "lookback": int(
                    FROZEN_CANDIDATES[
                        config_id
                    ]["lookback"]
                ),
                "horizon_steps": int(
                    horizon
                ),
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
        "horizon metric rows 8",
        len(horizon_metrics) == 8,
        str(len(horizon_metrics)),
    )

    candidate_rows = []

    for config_id, sub in (
        horizon_metrics.groupby(
            "config_id",
            sort=True,
        )
    ):
        ratios = {
            int(row.horizon_steps): float(
                row.mae_ratio_vs_persistence
            )
            for row in sub.itertuples()
        }

        check(
            f"{config_id}: four horizons",
            set(ratios)
            == set(HORIZONS),
        )

        candidate_rows.append(
            {
                "config_id": config_id,
                "lookback": int(
                    FROZEN_CANDIDATES[
                        config_id
                    ]["lookback"]
                ),
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
            }
        )

    candidate_summary = pd.DataFrame(
        candidate_rows
    )

    ranked = candidate_summary.sort_values(
        [
            "score",
            "lookback",
            "config_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    )

    candidate_summary["rank"] = 0
    candidate_summary[
        "selected_transformer_star"
    ] = False

    candidate_summary.loc[
        ranked.index,
        "rank",
    ] = np.arange(
        1,
        len(ranked) + 1,
    )

    winner_index = ranked.index[0]

    candidate_summary.loc[
        winner_index,
        "selected_transformer_star",
    ] = True

    candidate_summary[
        "rank"
    ] = candidate_summary[
        "rank"
    ].astype(int)

    check(
        "candidate rows 2",
        len(candidate_summary) == 2,
        str(len(candidate_summary)),
    )

    check(
        "TRANSFORMER* rows 1",
        int(
            candidate_summary[
                "selected_transformer_star"
            ].sum()
        )
        == 1,
    )

    winner = candidate_summary[
        candidate_summary[
            "selected_transformer_star"
        ]
    ].iloc[0]

    transformer_star = {
        "config_id": str(
            winner["config_id"]
        ),
        "lookback": int(
            winner["lookback"]
        ),
        "score": float(
            winner["score"]
        ),
        "rank": int(
            winner["rank"]
        ),
    }

    report_lines = [
        "=" * 120,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 120,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        (
            "Protocol 033 SHA-256: "
            f"{PROTOCOL_SHA}"
        ),
        (
            "Preflight closure 034 SHA-256: "
            f"{PREFLIGHT_CLOSURE_SHA}"
        ),
        (
            "Runner preflight closure 035 SHA-256: "
            f"{RUNNER_PREFLIGHT_CLOSURE_SHA}"
        ),
        (
            "Runtime compatibility 036 SHA-256: "
            f"{RUNTIME_COMPATIBILITY_SHA}"
        ),
        (
            "March closure 037 SHA-256: "
            f"{MARCH_CLOSURE_SHA}"
        ),
        "",
        "GOVERNANCE",
        "April role: selection",
        "Frozen candidates: T04/L288, T01/L24",
        "March retraining/scoring: NO",
        "June present: NO",
        "Tuning outside frozen candidates: NO",
        f"Canonical seed: {CANONICAL_SEED}",
        "Architecture search: NO",
        "",
        "TRANSFORMER*",
        (
            f"config={transformer_star['config_id']} "
            f"lookback={transformer_star['lookback']} "
            f"score={transformer_star['score']:.9f}"
        ),
        "",
        f"Fits executed: {len(fit_runtime)}",
        (
            "Campaign seconds: "
            f"{time.perf_counter() - started:.3f}"
        ),
        "STATUS: PASS",
        (
            "TRANSFORMER-APRIL-SELECTION-001: PASS"
        ),
        (
            "TRANSFORMER-STABILITY-001: NOT RUN"
        ),
        (
            "TRANSFORMER-JUNE-BLIND-001: "
            "NOT AUTHORIZED"
        ),
    ]

    report = (
        "\n".join(report_lines)
        + "\n"
    )

    paths = expected_paths

    atomic_csv(
        fit_runtime,
        paths["fit_runtime"],
    )
    atomic_parquet(
        predictions,
        paths["predictions"],
    )
    atomic_csv(
        horizon_metrics,
        paths["horizon_metrics"],
    )
    atomic_csv(
        candidate_summary,
        paths["candidate_summary"],
    )
    atomic_text(
        report,
        paths["report"],
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
            "runner_preflight_closure_035": {
                "name": RUNNER_PREFLIGHT_CLOSURE_NAME,
                "sha256": RUNNER_PREFLIGHT_CLOSURE_SHA,
            },
            "runtime_compatibility_036": {
                "name": RUNTIME_COMPATIBILITY_NAME,
                "sha256": RUNTIME_COMPATIBILITY_SHA,
            },
            "march_closure_037": {
                "name": MARCH_CLOSURE_NAME,
                "sha256": MARCH_CLOSURE_SHA,
            },
        },
        "governance": {
            "april_role": "selection",
            "march_used_for_training_or_scoring": False,
            "june_present": False,
            "canonical_seed": CANONICAL_SEED,
            "frozen_candidates": [
                "T04",
                "T01",
            ],
            "tuning_outside_033": False,
            "architecture_search": False,
            "transformer_star_selected": True,
            "stability_run": False,
            "freeze_run": False,
            "june_access_authorized": False,
        },
        "inputs": {
            "april": {
                "name": APRIL_NAME,
                "sha256": APRIL_SHA,
                "rows": APRIL_ROWS,
                "role": "selection",
                "used_for_training": True,
                "used_for_scoring": True,
            },
            "march_candidate_summary": {
                "name": MARCH_CANDIDATE_SUMMARY_NAME,
                "sha256": MARCH_CANDIDATE_SUMMARY_SHA,
                "role": "frozen candidate evidence only",
                "used_for_training_or_scoring": False,
            },
            "march_scientific_manifest": {
                "name": MARCH_SCIENTIFIC_MANIFEST_NAME,
                "sha256": MARCH_SCIENTIFIC_MANIFEST_SHA,
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
        "candidates": FROZEN_CANDIDATES,
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
        "selection": {
            "score": (
                "mean_h("
                "MAE_candidate_selection_h / "
                "MAE_persistence_selection_h"
                ")"
            ),
            "winner": transformer_star,
            "tie_break": [
                "lower score",
                "shorter lookback",
                "config_id ascending",
            ],
            "transformer_star_selected": True,
        },
        "validation": {
            "fit_count_8": (
                len(fit_runtime) == 8
            ),
            "all_fit_status_pass": bool(
                fit_runtime[
                    "status"
                ].eq("PASS").all()
            ),
            "candidate_rows_2": (
                len(candidate_summary) == 2
            ),
            "transformer_star_rows_1": (
                int(
                    candidate_summary[
                        "selected_transformer_star"
                    ].sum()
                )
                == 1
            ),
            "horizon_metric_rows_8": (
                len(horizon_metrics) == 8
            ),
            "april_hash_unchanged": (
                sha256(
                    bundle_root
                    / "inputs"
                    / APRIL_NAME
                )
                == APRIL_SHA
            ),
            "march_candidate_summary_hash_unchanged": (
                sha256(
                    bundle_root
                    / "evidence"
                    / MARCH_CANDIDATE_SUMMARY_NAME
                )
                == MARCH_CANDIDATE_SUMMARY_SHA
            ),
            "march_scientific_manifest_hash_unchanged": (
                sha256(
                    bundle_root
                    / "evidence"
                    / MARCH_SCIENTIFIC_MANIFEST_NAME
                )
                == MARCH_SCIENTIFIC_MANIFEST_SHA
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
                    / RUNNER_PREFLIGHT_CLOSURE_NAME
                )
                == RUNNER_PREFLIGHT_CLOSURE_SHA
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
            "june_absent_after_run": (
                not bool(
                    find_forbidden_june(
                        bundle_root
                    )
                )
            ),
        },
        "campaign_seconds": float(
            time.perf_counter() - started
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
        paths["manifest"],
    )

    deterministic_zip(
        paths["outputs_zip"],
        [
            path
            for key, path
            in paths.items()
            if key != "outputs_zip"
        ],
        output_dir,
    )

    print(report)

    print("VALIDATIONS")
    for key, value in (
        manifest[
            "validation"
        ].items()
    ):
        print(
            f"{key:68s}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print(
        "Outputs ZIP:",
        paths["outputs_zip"],
    )
    print(
        "Outputs ZIP SHA-256:",
        sha256(
            paths["outputs_zip"]
        ),
    )
    print(
        "TRANSFORMER-APRIL-SELECTION-001: PASS"
    )
    print(
        "TRANSFORMER*: "
        f"{transformer_star['config_id']}"
    )
    print(
        "TRANSFORMER-STABILITY-001: NOT RUN"
    )
    print(
        "TRANSFORMER-JUNE-BLIND-001: "
        "NOT AUTHORIZED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
