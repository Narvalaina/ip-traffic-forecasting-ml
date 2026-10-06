#!/usr/bin/env python3
"""
TRANSFORMER-JUNE-BLIND-001
Phase E — final blind external evaluation of frozen TRANSFORMER*.

Runner:
run_transformer_june_blind_v2_2026-08-22.py

Internal version:
2.0.0

Version note:
- v2 is operationally identical scientifically to v1.
- v2 only tracks the corrected June bundle-manifest/bundle version after the
  v1 builder aborted on an over-broad June-like filename guard.
- No scientific result was produced by v1.

Binding authorities:
- 033_phase_e_transformer_lite_protocol_2026-08-21.md
- 034_phase_e_colab_preflight_closure_2026-08-21.md
- 035_transformer_march_runner_preflight_closure_2026-08-21.md
- 036_phase_e_colab_runtime_compatibility_2026-08-21.md
- 037_transformer_march_screen_closure_2026-08-22.md
- 038_transformer_april_selection_preflight_closure_2026-08-22.md
- 039_transformer_april_selection_closure_2026-08-22.md
- 040_transformer_stability_preflight_closure_2026-08-22.md
- 041_transformer_stability_closure_2026-08-22.md
- 042_transformer_pre_june_freeze_2026-08-22.md
- transformer_pre_june_freeze_manifest_v1_2026-08-22.json

Frozen scientific identity:
- TRANSFORMER* = T01 / L24
- canonical seed = 20260820
- architecture/hyperparameters from 033/042
- 4 inner best_epoch fits + 4 final refits = 8 training stages
- expected blind predictions = 2390
- no tuning, no re-selection, no seed changes
- blind targets never affect fitting, epoch selection, scaling, or weights
- --preflight-only performs 0 training stages, 0 blind predictions,
  and writes 0 campaign artifacts

JSON serialization rule after 041:
- numpy.bool_, numpy.integer and numpy.floating are explicitly converted
  to native Python scalars before json.dumps().
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

RUNNER_VERSION = "2.0.0"
CAMPAIGN_ID = "TRANSFORMER-JUNE-BLIND-001"
PARENT_CAMPAIGN_ID = "UGR16-TRANSFORMER-LITE-001"
FILE_TAG = "v2_2026-08-22"

GOVERNANCE = {
    "033_phase_e_transformer_lite_protocol_2026-08-21.md":
        "4fcd61519612ae5fd3daa6d874a73df50b27b63531ac109f0782351875211552",
    "034_phase_e_colab_preflight_closure_2026-08-21.md":
        "9d89f0bbaa13448c56217c05f1663fa57fdb4505d3a395860186e2235e9cc081",
    "035_transformer_march_runner_preflight_closure_2026-08-21.md":
        "2f90c65e1855c98a20470d76229be68359cad8fb33b0b213c5f351eb9e21322e",
    "036_phase_e_colab_runtime_compatibility_2026-08-21.md":
        "8e8aeb3b0783c5928468bbbf074c1c42359837b57461890c4eb5e1145ca9dbf3",
    "037_transformer_march_screen_closure_2026-08-22.md":
        "9650e5aeb7ea8653e17baaa584237da8ab3d1fb41db9a150b0f98e29e075e03e",
    "038_transformer_april_selection_preflight_closure_2026-08-22.md":
        "99b9fbe0592729ec2c5e5d653124b62f52a423c917bda58568ab80ea67e5cc7c",
    "039_transformer_april_selection_closure_2026-08-22.md":
        "f4998ca12d138c91cc9d67f53b1e95a5bd4310cdb0637aa32e7690ec5836ab99",
    "040_transformer_stability_preflight_closure_2026-08-22.md":
        "31c1c4c3810746defb11b7f7bc26b99998cf51d432216e3dcda00c1413a525d9",
    "041_transformer_stability_closure_2026-08-22.md":
        "ffa68b530d479fe9d1510a7f74598c63b14a714c45316b86ad83454dd24d4c88",
    "042_transformer_pre_june_freeze_2026-08-22.md":
        "341050a9f41856792572764153ba75b1706b6ffad3252215d519b078f11531b0",
}

FREEZE_MANIFEST_NAME = "transformer_pre_june_freeze_manifest_v1_2026-08-22.json"
FREEZE_MANIFEST_SHA = "42f996fed2698e5f87e3feb9965e3155cc7331f86b6598f1b9f8fa4c2746182e"

APRIL_CANDIDATE_SUMMARY_NAME = (
    "ugr16_transformer_april_selection_candidate_summary_v1_2026-08-22.csv"
)
APRIL_CANDIDATE_SUMMARY_SHA = (
    "5272e6733f60b6cf24281d9fe3d73cba62918636e582ad7ed7fcb2445728ff14"
)
APRIL_MANIFEST_NAME = "ugr16_transformer_april_selection_manifest_v1_2026-08-22.json"
APRIL_MANIFEST_SHA = "c687f0a4f09487d5a90baaefa86a034d53a26921c32014f38a150d41a2db0679"

STABILITY_SEED_SUMMARY_NAME = "ugr16_transformer_stability_seed_summary_v1_2026-08-22.csv"
STABILITY_SEED_SUMMARY_SHA = "e97086f62fd14dbae7e1c4ba39b57a9487317733febd02c538221553c7837ab2"
STABILITY_SUMMARY_NAME = "ugr16_transformer_stability_summary_v1_2026-08-22.csv"
STABILITY_SUMMARY_SHA = "4e8dd5fc874602ecf74ea401b9da8ca714d91f7c182f679cca34e5e58d0bf7cc"
STABILITY_RECOVERY_MANIFEST_NAME = (
    "ugr16_transformer_stability_manifest_recovery_v1_2026-08-22.json"
)
STABILITY_RECOVERY_MANIFEST_SHA = (
    "8076b4aba74b0a1528d84a55b8e4cd07f0e36416690eb5761a3487357a91a77d"
)
STABILITY_RECOVERY_EXEC_NAME = (
    "transformer_stability_full_colab_exec_recovery_v1_2026-08-22_execution_manifest.json"
)
STABILITY_RECOVERY_EXEC_SHA = (
    "dc016f76c4b8153838bca7b8ac9559b4b490a86053dca7816b0e3c4ec957e861"
)

JUNE_NAME = "june_week3_prepared_5min.parquet"
JUNE_SHA = "d528047b6d93d9b93f03b016c408c4a660447eb9c1ab1bd680f6dcbf628b56f3"
JUNE_ROWS = 2004
JUNE_START = "2016-06-13 01:00:00"
JUNE_END = "2016-06-19 23:55:00"

BUNDLE_MANIFEST_NAME = "phase_e_transformer_june_blind_exec_bundle_manifest_v2_2026-08-22.json"
BUNDLE_ID = "PHASE-E-TRANSFORMER-JUNE-BLIND-EXEC-V2-2026-08-22"

TARGET = "bitrate_bps"
TIME_COL = "timestamp"

TRAIN_ROWS = 1402
TRAIN_START = "2016-06-13 01:00:00"
TRAIN_END = "2016-06-17 21:45:00"
BLIND_START = "2016-06-17 21:50:00"
BLIND_ROWS = 602

L_MAX = 288
LOOKBACK = 24
TRANSFORMER_STAR_CONFIG = "T01"
CANONICAL_SEED = 20260820
EXPECTED_PARAMETER_COUNT = 8705

HORIZONS = (1, 3, 6, 12)
EXPECTED_BLIND_COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
EXPECTED_TOTAL_BLIND_PREDICTIONS = sum(EXPECTED_BLIND_COUNTS.values())
EXPECTED_TRAIN_COMMON_COUNTS = {1: 1114, 3: 1112, 6: 1109, 12: 1103}
EXPECTED_INNER_SPLITS = {
    1:  {"fit": 882, "gap": 12, "early_stop": 220},
    3:  {"fit": 880, "gap": 12, "early_stop": 220},
    6:  {"fit": 878, "gap": 12, "early_stop": 219},
    12: {"fit": 873, "gap": 12, "early_stop": 218},
}

INNER_GAP = 12
INNER_EARLY_STOP_FRACTION = 0.20

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 0.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
HARD_STAGE_SECONDS = 600.0

EXPECTED_SOFTWARE = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torch_cuda": "12.8",
    "cudnn": 91900,
    "numpy": "2.1.3",
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, condition: bool, detail: str = "") -> None:
    condition = bool(condition)
    status = "PASS" if condition else "FAIL"
    print(f"{name:72s}: {status}" + (f" | {detail}" if detail else ""))
    if not condition:
        raise RuntimeError(f"{name}: FAIL | {detail}")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


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
    for key in ("python", "torch", "torch_cuda", "cudnn", "numpy", "pandas"):
        check(f"software_{key}", info[key] == EXPECTED_SOFTWARE[key], str(info[key]))
    check("cuda_available", info["cuda_available"], str(info["gpu_name"]))
    check("cuda_device_count_positive", info["device_count"] >= 1, str(info["device_count"]))
    check(
        "CUBLAS_WORKSPACE_CONFIG",
        info["cublas_workspace_config"] == ":4096:8",
        str(info["cublas_workspace_config"]),
    )
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
    check("June spacing 5 min", bool((ts.diff().dropna() == pd.Timedelta(minutes=5)).all()))

    check("June training rows frozen", TRAIN_ROWS == 1402, str(TRAIN_ROWS))
    check("June train start", str(ts.iloc[0]) == TRAIN_START, str(ts.iloc[0]))
    check("June train end", str(ts.iloc[TRAIN_ROWS - 1]) == TRAIN_END, str(ts.iloc[TRAIN_ROWS - 1]))
    check("June blind start", str(ts.iloc[TRAIN_ROWS]) == BLIND_START, str(ts.iloc[TRAIN_ROWS]))
    check("June blind rows", len(frame) - TRAIN_ROWS == BLIND_ROWS, str(len(frame) - TRAIN_ROWS))
    return frame


def validate_evidence(evidence: Path) -> None:
    expected = {
        APRIL_CANDIDATE_SUMMARY_NAME: APRIL_CANDIDATE_SUMMARY_SHA,
        APRIL_MANIFEST_NAME: APRIL_MANIFEST_SHA,
        STABILITY_SEED_SUMMARY_NAME: STABILITY_SEED_SUMMARY_SHA,
        STABILITY_SUMMARY_NAME: STABILITY_SUMMARY_SHA,
        STABILITY_RECOVERY_MANIFEST_NAME: STABILITY_RECOVERY_MANIFEST_SHA,
        STABILITY_RECOVERY_EXEC_NAME: STABILITY_RECOVERY_EXEC_SHA,
    }
    for name, expected_sha in expected.items():
        validate_hash_file(evidence / name, expected_sha, name)

    april = pd.read_csv(evidence / APRIL_CANDIDATE_SUMMARY_NAME)
    check("April candidate rows = 2", len(april) == 2, str(len(april)))
    check("selected_transformer_star column", "selected_transformer_star" in april.columns)
    selected = april[april["selected_transformer_star"].astype(bool)]
    check("exactly one TRANSFORMER* row", len(selected) == 1, str(len(selected)))
    row = selected.iloc[0]
    check("TRANSFORMER* config = T01", str(row["config_id"]) == "T01", str(row["config_id"]))
    check("TRANSFORMER* lookback = 24", int(row["lookback"]) == 24, str(row["lookback"]))

    stability = pd.read_csv(evidence / STABILITY_SUMMARY_NAME)
    check("stability summary rows = 1", len(stability) == 1, str(len(stability)))
    check(
        "best-seed selection false",
        bool(stability["best_seed_selection"].eq(False).all()),
    )
    check(
        "canonical seed unchanged",
        bool(stability["canonical_seed_changed"].eq(False).all()),
    )
    check(
        "canonical seed = 20260820",
        int(stability["canonical_seed"].iloc[0]) == CANONICAL_SEED,
        str(stability["canonical_seed"].iloc[0]),
    )


def validate_freeze_manifest(path: Path) -> dict:
    validate_hash_file(path, FREEZE_MANIFEST_SHA, "Freeze manifest")
    obj = json.loads(path.read_text(encoding="utf-8"))
    check("freeze campaign id", obj.get("campaign_id") == "TRANSFORMER-FREEZE-001", str(obj.get("campaign_id")))
    star = obj.get("transformer_star", {})
    check("freeze TRANSFORMER* T01", star.get("config_id") == "T01", repr(star))
    check("freeze lookback 24", star.get("lookback") == 24, repr(star))
    check("freeze canonical seed", star.get("canonical_seed") == CANONICAL_SEED, repr(star))
    check("freeze parameter count", star.get("trainable_parameters") == EXPECTED_PARAMETER_COUNT, repr(star))
    after = obj.get("state_after_effective_local_verification", {})
    check(
        "freeze authorizes June preparation",
        after.get("TRANSFORMER-JUNE-BLIND-001") == "AUTHORIZED FOR PREPARATION / NOT RUN",
        repr(after),
    )
    return obj


def validate_bundle_manifest(path: Path, runner_sha: str) -> dict:
    check("bundle manifest exists", path.is_file(), str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    check("bundle manifest id", obj.get("bundle_id") == BUNDLE_ID, str(obj.get("bundle_id")))
    check("bundle manifest campaign", obj.get("campaign_id") == CAMPAIGN_ID, str(obj.get("campaign_id")))
    check(
        "bundle manifest status",
        obj.get("status") == "PRE-EXECUTION / FROZEN_INPUT_BUNDLE",
        str(obj.get("status")),
    )
    check("bundle manifest runner SHA", obj["runner"]["sha256"] == runner_sha, str(obj["runner"]["sha256"]))
    check("bundle manifest June SHA", obj["inputs"]["june"]["sha256"] == JUNE_SHA, str(obj["inputs"]["june"]["sha256"]))
    check("bundle manifest canonical seed", obj["frozen_configuration"]["canonical_seed"] == CANONICAL_SEED)
    check("bundle manifest TRANSFORMER*", obj["frozen_configuration"]["config_id"] == "T01")
    check("bundle manifest lookback", obj["frozen_configuration"]["lookback"] == 24)
    check("bundle manifest expected stages", obj["expected_execution"]["training_stages"] == 8)
    check("bundle manifest expected predictions", obj["expected_execution"]["blind_predictions"] == 2390)
    check("bundle manifest no training yet", obj["pre_execution_assertions"]["training_executed"] is False)
    check("bundle manifest no selection", obj["pre_execution_assertions"]["selection_allowed"] is False)
    check("bundle manifest no retuning", obj["pre_execution_assertions"]["retuning_allowed"] is False)
    return obj


def training_common_origins(horizon: int) -> np.ndarray:
    return np.arange(L_MAX - 1, TRAIN_ROWS - horizon, dtype=np.int64)


def inner_split(origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(origins)
    n_available = n - INNER_GAP
    early_count = math.floor(INNER_EARLY_STOP_FRACTION * n_available)
    fit_count = n_available - early_count
    fit = origins[:fit_count]
    early = origins[fit_count + INNER_GAP:]
    check(
        "inner split consumes all permitted samples",
        len(fit) + INNER_GAP + len(early) == n,
        f"{len(fit)}+{INNER_GAP}+{len(early)}={n}",
    )
    return fit, early


def blind_origins(horizon: int) -> np.ndarray:
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
    horizon: int,
    scaler: ScalarStandardizer,
    with_targets: bool = True,
):
    xs, ys, targets = [], [], []
    for origin in origins:
        origin = int(origin)
        target_idx = origin + horizon
        start = origin - LOOKBACK + 1
        if start < 0 or target_idx >= len(y):
            raise RuntimeError("Invalid supervised window bounds")

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
    yy = (
        np.asarray(ys, dtype=np.float32).reshape(-1, 1)
        if with_targets
        else None
    )
    return x, yy, target_idx


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
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)


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
        total += float(loss_fn(model(xb), yb).item())
        n += len(xb)
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
    horizon: int,
    y: np.ndarray,
    fit_origins: np.ndarray,
    early_origins: np.ndarray,
    device: torch.device,
) -> dict:
    fit_target_max = int(fit_origins[-1] + horizon)
    scaler = ScalarStandardizer.fit(y[:fit_target_max + 1])

    x_fit, y_fit, _ = build_xy_from_origins(y, fit_origins, horizon, scaler, True)
    x_early, y_early, _ = build_xy_from_origins(y, early_origins, horizon, scaler, True)

    set_seed(CANONICAL_SEED)
    model = TransformerLite().to(device)
    check(
        f"T01/H{horizon}: parameter count",
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
        if time.perf_counter() - started > HARD_STAGE_SECONDS:
            raise TimeoutError(f"RESOURCE_LIMIT inner T01/H{horizon}")

        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite inner loss T01/H{horizon}")

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
        raise TimeoutError(f"RESOURCE_LIMIT inner T01/H{horizon}")
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
    horizon: int,
    best_epoch: int,
    y: np.ndarray,
    train_origins: np.ndarray,
    device: torch.device,
):
    scaler = ScalarStandardizer.fit(y[:TRAIN_ROWS])

    x_train, y_train, _ = build_xy_from_origins(
        y, train_origins, horizon, scaler, True
    )

    set_seed(CANONICAL_SEED)
    model = TransformerLite().to(device)
    check(
        f"T01/H{horizon} refit parameter count",
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        str(parameter_count(model)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.L1Loss()
    loader = make_loader(x_train, y_train)

    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        if time.perf_counter() - started > HARD_STAGE_SECONDS:
            raise TimeoutError(f"RESOURCE_LIMIT final_refit T01/H{horizon}")

        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite final-refit loss T01/H{horizon}")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    runtime = time.perf_counter() - started
    if runtime > HARD_STAGE_SECONDS:
        raise TimeoutError(f"RESOURCE_LIMIT final_refit T01/H{horizon}")

    return model, scaler, float(runtime), parameter_count(model)


def blind_predict(
    model: nn.Module,
    scaler: ScalarStandardizer,
    horizon: int,
    best_epoch: int,
    y: np.ndarray,
    timestamps: pd.Series,
    device: torch.device,
) -> pd.DataFrame:
    origins = blind_origins(horizon)

    x_blind, _, targets = build_xy_from_origins(
        y, origins, horizon, scaler, with_targets=False
    )

    # Strict causal order:
    # 1) construct lag windows using only observations through current origin;
    # 2) run forward pass;
    # 3) only then access target values for evaluation.
    pred_scaled = predict_scaled(model, x_blind, device)
    y_pred = scaler.inverse(pred_scaled)

    y_true = y[targets].astype(np.float64)
    persistence = y[origins].astype(np.float64)

    return pd.DataFrame(
        {
            "config_id": TRANSFORMER_STAR_CONFIG,
            "lookback": LOOKBACK,
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
        }
    )


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
        json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
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

    governance_dir = bundle_root / "governance"
    evidence_dir = bundle_root / "evidence"
    inputs_dir = bundle_root / "inputs"

    for name, expected_sha in GOVERNANCE.items():
        validate_hash_file(governance_dir / name, expected_sha, name[:3])

    freeze_manifest = validate_freeze_manifest(
        governance_dir / FREEZE_MANIFEST_NAME
    )
    validate_evidence(evidence_dir)
    june = validate_june(inputs_dir / JUNE_NAME)

    bundle_manifest = validate_bundle_manifest(
        bundle_root / "manifest" / BUNDLE_MANIFEST_NAME,
        runner_sha,
    )

    check("canonical seed frozen", CANONICAL_SEED == 20260820, str(CANONICAL_SEED))
    check("TRANSFORMER* frozen", TRANSFORMER_STAR_CONFIG == "T01")
    check("lookback frozen", LOOKBACK == 24)
    check("parameter count frozen", EXPECTED_PARAMETER_COUNT == 8705)

    for h in HORIZONS:
        train_origins = training_common_origins(h)
        check(
            f"H{h} train common count",
            len(train_origins) == EXPECTED_TRAIN_COMMON_COUNTS[h],
            str(len(train_origins)),
        )
        fit_origins, early_origins = inner_split(train_origins)
        exp = EXPECTED_INNER_SPLITS[h]
        check(f"H{h} inner fit count", len(fit_origins) == exp["fit"], str(len(fit_origins)))
        check(
            f"H{h} inner gap",
            int(early_origins[0] - fit_origins[-1] - 1) == exp["gap"],
            str(int(early_origins[0] - fit_origins[-1] - 1)),
        )
        check(
            f"H{h} inner early-stop count",
            len(early_origins) == exp["early_stop"],
            str(len(early_origins)),
        )

        b_origins = blind_origins(h)
        check(f"H{h} blind count", len(b_origins) == EXPECTED_BLIND_COUNTS[h], str(len(b_origins)))
        check(f"H{h} first blind origin", int(b_origins[0]) == TRAIN_ROWS - 1, str(int(b_origins[0])))
        check(f"H{h} first target in blind", int(b_origins[0] + h) >= TRAIN_ROWS, str(int(b_origins[0] + h)))
        check(f"H{h} final target", int(b_origins[-1] + h) == JUNE_ROWS - 1, str(int(b_origins[-1] + h)))

    set_seed(CANONICAL_SEED)
    device = torch.device("cuda")
    x = torch.randn(4, LOOKBACK, 1, device=device)
    model = TransformerLite().to(device)
    check("Transformer parameter count", parameter_count(model) == EXPECTED_PARAMETER_COUNT, str(parameter_count(model)))
    model.eval()
    with torch.no_grad():
        out = model(x)
    check("Transformer forward shape", tuple(out.shape) == (4, 1), str(tuple(out.shape)))
    check("Transformer forward finite", bool(torch.isfinite(out).all().item()))

    print("TRAINING STAGES EXECUTED IN PREFLIGHT: 0")
    print("BLIND PREDICTIONS GENERATED IN PREFLIGHT: 0")
    print("CAMPAIGN ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print("TRANSFORMER* RESELECTION IN PREFLIGHT: NO")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")

    return software, june, freeze_manifest, bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/transformer_june_blind_v2_2026-08-22_outputs')),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)

    software, june, freeze_manifest, bundle_manifest = preflight(bundle_root)

    if args.preflight_only:
        return 0

    output_names = {
        "runtime": f"ugr16_transformer_june_blind_runtime_{FILE_TAG}.csv",
        "epoch_selection": f"ugr16_transformer_june_blind_epoch_selection_{FILE_TAG}.csv",
        "predictions": f"ugr16_transformer_june_blind_predictions_{FILE_TAG}.parquet",
        "horizon_metrics": f"ugr16_transformer_june_blind_horizon_metrics_{FILE_TAG}.csv",
        "report": f"ugr16_transformer_june_blind_report_{FILE_TAG}.txt",
        "manifest": f"ugr16_transformer_june_blind_manifest_{FILE_TAG}.json",
        "results_zip": f"ugr16_transformer_june_blind_outputs_{FILE_TAG}.zip",
    }
    paths = {k: output_dir / v for k, v in output_names.items()}

    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing outputs; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    y = june[TARGET].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(june[TIME_COL]).reset_index(drop=True)
    device = torch.device("cuda")

    mase_scale = float(np.mean(np.abs(np.diff(y[:TRAIN_ROWS]))))
    check(
        "June training MASE scale valid",
        np.isfinite(mase_scale) and mase_scale > 0,
        str(mase_scale),
    )

    runtime_rows = []
    epoch_rows = []
    pred_frames = []

    campaign_started = time.perf_counter()

    print()
    print("=" * 126)
    print(f"START {CAMPAIGN_ID}")
    print("=" * 126)
    print("TRANSFORMER* = T01/L24")
    print("Horizons = 4")
    print("Inner epoch-selection fits = 4")
    print("Final refits = 4")
    print("Total training stages = 8")
    print(f"Expected blind predictions = {EXPECTED_TOTAL_BLIND_PREDICTIONS}")
    print("Canonical seed = 20260820")
    print("No tuning. No re-selection. No blind-target-driven fitting.")
    print()

    for h in HORIZONS:
        train_origins = training_common_origins(h)
        fit_origins, early_origins = inner_split(train_origins)

        print(
            f"[INNER] T01/L24 H{h:<2d} "
            f"fit={len(fit_origins)} gap={INNER_GAP} early={len(early_origins)}",
            flush=True,
        )
        epoch_info = select_best_epoch(
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
            f"[REFIT] T01/L24 H{h:<2d} "
            f"epochs={epoch_info['best_epoch']} train_samples={len(train_origins)}",
            flush=True,
        )
        model, final_scaler, final_runtime, params = final_refit(
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
            horizon=h,
            best_epoch=epoch_info["best_epoch"],
            y=y,
            timestamps=timestamps,
            device=device,
        )

        check(
            f"T01/H{h} blind prediction count",
            len(preds) == EXPECTED_BLIND_COUNTS[h],
            str(len(preds)),
        )
        check(
            f"T01/H{h} predictions finite",
            np.isfinite(preds["y_pred_bps"].to_numpy(float)).all(),
        )

        runtime_rows.append(
            {
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": LOOKBACK,
                "canonical_seed": CANONICAL_SEED,
                "horizon_steps": h,
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
            }
        )

        epoch_rows.append(
            {
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": LOOKBACK,
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
            }
        )

        pred_frames.append(preds)

    runtime = pd.DataFrame(runtime_rows)
    epoch_selection = pd.DataFrame(epoch_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)

    check("runtime rows = 4", len(runtime) == 4, str(len(runtime)))
    check("epoch-selection rows = 4", len(epoch_selection) == 4, str(len(epoch_selection)))
    check("all runtime status PASS", bool(runtime["status"].eq("PASS").all()))
    check(
        "total blind predictions",
        len(predictions) == EXPECTED_TOTAL_BLIND_PREDICTIONS,
        str(len(predictions)),
    )

    for h in HORIZONS:
        sub = predictions[predictions["horizon_steps"].eq(h)]
        check(
            f"H{h} origin coverage",
            tuple(sub["origin_index"].astype(int)) == tuple(blind_origins(h).tolist()),
        )
        check(
            f"H{h} target coverage",
            tuple(sub["target_index"].astype(int)) == tuple((blind_origins(h) + h).tolist()),
        )

    metric_rows = []
    for h, sub in predictions.groupby("horizon_steps", sort=True):
        metric_rows.append(
            {
                "config_id": TRANSFORMER_STAR_CONFIG,
                "lookback": LOOKBACK,
                "canonical_seed": CANONICAL_SEED,
                "horizon_steps": int(h),
                "horizon_minutes": int(h * 5),
                "best_epoch": int(
                    epoch_selection.loc[
                        epoch_selection["horizon_steps"].eq(h),
                        "best_epoch",
                    ].iloc[0]
                ),
                **metric_row(sub, mase_scale),
            }
        )

    horizon_metrics = pd.DataFrame(metric_rows)
    check("horizon metric rows = 4", len(horizon_metrics) == 4, str(len(horizon_metrics)))

    # Frozen authority/input integrity after execution.
    governance_dir = bundle_root / "governance"
    evidence_dir = bundle_root / "evidence"
    for name, expected_sha in GOVERNANCE.items():
        validate_hash_file(governance_dir / name, expected_sha, f"{name[:3]} post-run")
    validate_hash_file(governance_dir / FREEZE_MANIFEST_NAME, FREEZE_MANIFEST_SHA, "freeze manifest post-run")
    validate_hash_file(bundle_root / "inputs" / JUNE_NAME, JUNE_SHA, "June post-run")

    expected_evidence = {
        APRIL_CANDIDATE_SUMMARY_NAME: APRIL_CANDIDATE_SUMMARY_SHA,
        APRIL_MANIFEST_NAME: APRIL_MANIFEST_SHA,
        STABILITY_SEED_SUMMARY_NAME: STABILITY_SEED_SUMMARY_SHA,
        STABILITY_SUMMARY_NAME: STABILITY_SUMMARY_SHA,
        STABILITY_RECOVERY_MANIFEST_NAME: STABILITY_RECOVERY_MANIFEST_SHA,
        STABILITY_RECOVERY_EXEC_NAME: STABILITY_RECOVERY_EXEC_SHA,
    }
    for name, expected_sha in expected_evidence.items():
        validate_hash_file(evidence_dir / name, expected_sha, f"{name} post-run")

    campaign_seconds = time.perf_counter() - campaign_started

    report_lines = [
        "=" * 126,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 126,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"042 SHA-256: {GOVERNANCE['042_transformer_pre_june_freeze_2026-08-22.md']}",
        f"Freeze manifest SHA-256: {FREEZE_MANIFEST_SHA}",
        "",
        "GOVERNANCE",
        "June role: final blind external evaluation",
        "TRANSFORMER* changed: NO",
        "Best-seed selection: NO",
        "Retuning: NO",
        f"Canonical seed: {CANONICAL_SEED}",
        "Blind-target-driven fitting: NO",
        "",
        "FROZEN TRANSFORMER*",
        "config_id = T01",
        "lookback = 24",
        "trainable parameters = 8705",
        "",
        "JUNE SPLIT",
        f"train rows = {TRAIN_ROWS}",
        f"blind rows = {BLIND_ROWS}",
        f"blind starts = {BLIND_START}",
        "",
        "BLIND METRICS",
    ]

    for row in horizon_metrics.sort_values("horizon_steps").itertuples(index=False):
        report_lines.append(
            f"T01 L24 H{row.horizon_steps:<2d} "
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
        "TRANSFORMER* CHANGED: NO",
        "BEST SEED SELECTED: NO",
        "RETUNING: NO",
        "TRANSFORMER-INFERENCE-RESIDUALS-001: NOT RUN",
        "TRANSFORMER-CLOSURE-001: NOT RUN",
    ]
    report = "\n".join(report_lines) + "\n"

    atomic_csv(runtime, paths["runtime"])
    atomic_csv(epoch_selection, paths["epoch_selection"])
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(horizon_metrics, paths["horizon_metrics"])
    atomic_text(report, paths["report"])

    output_records = {}
    for key in ("runtime", "epoch_selection", "predictions", "horizon_metrics", "report"):
        p = paths[key]
        output_records[key] = {
            "name": p.name,
            "sha256": sha256(p),
            "bytes": int(p.stat().st_size),
        }

    validation = {
        "runtime_rows_4": len(runtime) == 4,
        "epoch_selection_rows_4": len(epoch_selection) == 4,
        "all_runtime_status_pass": bool(runtime["status"].eq("PASS").all()),
        "horizon_metric_rows_4": len(horizon_metrics) == 4,
        "blind_prediction_rows_2390": len(predictions) == 2390,
        "finite_predictions": bool(np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all()),
        "transformer_star_unchanged": True,
        "canonical_seed_unchanged": CANONICAL_SEED == 20260820,
        "no_best_seed_selection": True,
        "no_retuning": True,
        "blind_targets_used_for_fitting": False,
        "june_hash_unchanged": sha256(bundle_root / "inputs" / JUNE_NAME) == JUNE_SHA,
        "freeze_042_hash_unchanged":
            sha256(governance_dir / "042_transformer_pre_june_freeze_2026-08-22.md")
            == GOVERNANCE["042_transformer_pre_june_freeze_2026-08-22.md"],
        "freeze_manifest_hash_unchanged":
            sha256(governance_dir / FREEZE_MANIFEST_NAME) == FREEZE_MANIFEST_SHA,
    }
    check("all final validations PASS", all(validation.values()), repr(validation))

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
        "freeze_manifest": freeze_manifest,
        "authorities": {
            name: {"name": name, "sha256": expected_sha}
            for name, expected_sha in GOVERNANCE.items()
        },
        "freeze_manifest_authority": {
            "name": FREEZE_MANIFEST_NAME,
            "sha256": FREEZE_MANIFEST_SHA,
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
        },
        "software": software,
        "frozen_configuration": {
            "config_id": TRANSFORMER_STAR_CONFIG,
            "lookback": LOOKBACK,
            "canonical_seed": CANONICAL_SEED,
            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
            "horizons": list(HORIZONS),
            "L_MAX": L_MAX,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.10,
            "activation": "GELU",
            "positional_encoding": "fixed_sinusoidal",
            "aggregation": "last_temporal_representation",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
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
            "blind_scaler_updates": False,
            "blind_retuning": False,
            "blind_real_observations_as_revealed_lags": True,
        },
        "campaign": {
            "inner_epoch_selection_fits": 4,
            "final_refits": 4,
            "total_training_stages": 8,
            "blind_model_horizon_evaluations": 4,
            "blind_prediction_rows": int(len(predictions)),
            "expected_counts_per_horizon": EXPECTED_BLIND_COUNTS,
            "transformer_star_changed": False,
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
        if not args.overwrite:
            raise FileExistsError(f"Refusing overwrite: {results_zip_path}")
        results_zip_path.unlink()

    with zipfile.ZipFile(
        results_zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for key in (
            "runtime",
            "epoch_selection",
            "predictions",
            "horizon_metrics",
            "report",
            "manifest",
        ):
            p = paths[key]
            zf.write(p, arcname=p.name)

    print()
    print(report)
    print("VALIDATIONS")
    print("-" * 126)
    for key, value in validation.items():
        print(f"{key:72s}: {'PASS' if bool(value) else 'FAIL'}")

    print()
    print(f"Results ZIP: {results_zip_path}")
    print(f"Results ZIP SHA-256: {sha256(results_zip_path)}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("TRANSFORMER* CHANGED: NO")
    print("BEST SEED SELECTED: NO")
    print("RETUNING: NO")
    print("TRANSFORMER-INFERENCE-RESIDUALS-001: NOT RUN")
    print("TRANSFORMER-CLOSURE-001: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
