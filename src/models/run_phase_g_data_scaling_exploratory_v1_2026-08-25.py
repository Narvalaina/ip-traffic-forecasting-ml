#!/usr/bin/env python3
from __future__ import annotations

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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Debe fijarse antes de cualquier uso CUDA determinista.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import statsmodels
import torch
from statsmodels.tsa.api import VAR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RUNNER_VERSION = "1.0.0"
CAMPAIGN_ID = "DATA-SCALING-EXPLORATORY-001"
PREFLIGHT_ID = "PHASE-G-SCIENTIFIC-RUNNER-PREFLIGHT-001"
DATE_TAG = "2026-08-25"

DEFAULT_PROJECT = Path.home() / "TFM_IP_Traffic_Forecasting"

GOVERNANCE = {
    "031_phase_g_data_scaling_decision_2026-08-21.md":
        "78723a948e6dc17e440622044e6eaa270077cae6d1ee7a38dd36524917d41bc1",
    "055_post_phase_f_data_scaling_decision_2026-08-22.md":
        "9d8ec6e54c2bda070f1a70e67fd8b4c1c0598369e6bddbd1deaf053047d1bd7e",
    "071_phase_g_aggregation_final_closure_continuity_series_preparation_authorization_2026-08-25.md":
        "c30d65fae44bdb1ad110c878ae62b699f2fa32b124d20c32603c6a3e3ad3b34b",
    "072_phase_g_continuity_boundary_incident_method_repair_authorization_2026-08-25.md":
        "ad5b57a7f34f031878d89e59195d858726003c75406615a1d25001589b815955",
    "073_phase_g_continuity_series_v2_closure_scientific_protocol_freeze_2026-08-25.md":
        "c97bce8004f4b1be5357a6a09d8c81f9655147da9f9a85e51ed0eca2872bc6fb",
}

CONTINUITY_MANIFEST_REL = (
    "results/metrics/phase_g/continuity_series_preparation/v2_2026-08-25/"
    "phase_g_continuity_series_preparation_v2_2026-08-25_manifest.json"
)
CONTINUITY_MANIFEST_SHA = "5423845bb70a85331d8b780fec883cfd74c175288872674240fb6054ab689f7c"

PREFLIGHT_OUT_REL = "results/metrics/phase_g/scientific_runner_preflight/v1_2026-08-25"
SCIENTIFIC_OUT_REL = "results/phase_g/data_scaling_exploratory/v1_2026-08-25"

TARGET = "bitrate_bps"
TIME_COL = "timestamp"
VAR_COLUMNS = ("bitrate_bps", "packet_rate_pps", "flow_rate_fps")

HISTORIES = (1, 2, 4, 8)
HISTORY_ROWS = {1: 2016, 2: 4032, 4: 8064, 8: 16128}

TEST_ROWS = 1947
TEST_START = pd.Timestamp("2016-06-20T00:10:00")
TEST_END_EXCLUSIVE = pd.Timestamp("2016-06-26T18:25:00")
TEST_END = TEST_END_EXCLUSIVE - pd.Timedelta(minutes=5)

TRAIN_END_EXCLUSIVE = TEST_START
TRAIN_STARTS = {
    1: pd.Timestamp("2016-06-13T00:10:00"),
    2: pd.Timestamp("2016-06-06T00:10:00"),
    4: pd.Timestamp("2016-05-23T00:10:00"),
    8: pd.Timestamp("2016-04-25T00:10:00"),
}
TRAIN_END = TRAIN_END_EXCLUSIVE - pd.Timedelta(minutes=5)

INPUT_KEYS = {
    1: "train_1w",
    2: "train_2w",
    4: "train_4w",
    8: "train_8w",
}
EXPECTED_BASENAMES = {
    1: "phase_g_train_1w_5min.parquet",
    2: "phase_g_train_2w_5min.parquet",
    4: "phase_g_train_4w_5min.parquet",
    8: "phase_g_train_8w_5min.parquet",
    "test": "phase_g_test_june_week4_5min.parquet",
}

L_MAX = 288
HORIZONS = (1, 3, 6, 12)
EXPECTED_TEST_PREDICTIONS = {1: 1947, 3: 1945, 6: 1942, 12: 1936}
EXPECTED_TOTAL_PER_MODEL_HISTORY = sum(EXPECTED_TEST_PREDICTIONS.values())

CANONICAL_SEED = 20260820
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

RECURRENT_CONFIG = {
    "family": "GRU",
    "config_id": "N06",
    "lookback": 72,
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.20,
    "expected_parameter_count": 37889,
}

TRANSFORMER_CONFIG = {
    "config_id": "T01",
    "lookback": 24,
    "expected_parameter_count": 8705,
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


class StopError(RuntimeError):
    pass


class Gate:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: Any, detail: Any = "") -> None:
        ok = bool(condition)
        self.checks.append({"name": name, "pass": ok, "detail": str(detail)})
        print(
            f"{name:<94}: {'PASS' if ok else 'FAIL'}"
            + (f" | {detail}" if str(detail) else ""),
            flush=True,
        )
        if not ok:
            raise StopError(f"{name}: {detail}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase G UGR'16: data-scaling exploratorio 1/2/4/8 semanas. "
            "El preflight no entrena ni genera predicciones científicas."
        )
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")

    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Opcional. Si se especifica, los Parquet se resuelven por basename "
            "dentro de este directorio; útil para un bundle Colab."
        ),
    )
    p.add_argument("--preflight-out-dir", type=Path, default=None)
    p.add_argument("--scientific-out-dir", type=Path, default=None)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--hard-stage-seconds",
        type=float,
        default=HARD_STAGE_SECONDS,
    )

    # Obligatorios únicamente con --execute. Se rellenarán después del cierre 074.
    p.add_argument("--authorization-074", type=Path, default=None)
    p.add_argument("--authorization-074-sha", type=str, default=None)

    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(obj), indent=2, ensure_ascii=False) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


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


def software_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
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
        "statsmodels": statsmodels.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise StopError("--device cuda solicitado pero CUDA no está disponible.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            raise StopError("Parámetros no finitos en StandardScaler escalar.")
        if scale == 0.0:
            scale = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


class RecurrentRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(
            input_size=1,
            hidden_size=RECURRENT_CONFIG["hidden_size"],
            num_layers=RECURRENT_CONFIG["n_layers"],
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.external_dropout = nn.Dropout(p=RECURRENT_CONFIG["dropout"])
        self.output = nn.Linear(RECURRENT_CONFIG["hidden_size"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        last = sequence[:, -1, :]
        return self.output(self.external_dropout(last))


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
        return x + self.pe[:, : x.size(1), :].to(
            dtype=x.dtype,
            device=x.device,
        )


class TransformerLite(nn.Module):
    def __init__(self) -> None:
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


def validate_governance(project_root: Path, gate: Gate) -> dict[str, Any]:
    out: dict[str, Any] = {}
    root = project_root / "docs/project_governance"
    for name, expected in GOVERNANCE.items():
        path = root / name
        gate.check(f"Governance {name}: existe", path.is_file(), path)
        actual = sha256(path)
        gate.check(f"Governance {name}: SHA exacto", actual == expected, actual)
        out[name] = {"path": str(path), "sha256": actual}
    return out


def validate_execution_authorization(
    path: Path | None,
    expected_sha: str | None,
    gate: Gate,
) -> dict[str, Any]:
    gate.check("074 path supplied", path is not None, path)
    gate.check("074 expected SHA supplied", bool(expected_sha), expected_sha)
    assert path is not None
    assert expected_sha is not None
    gate.check("074 exists", path.is_file(), path)
    actual = sha256(path)
    gate.check("074 SHA exact", actual == expected_sha, actual)
    text = path.read_text(encoding="utf-8")
    gate.check(
        "074 autoriza DATA-SCALING-EXPLORATORY-001",
        "DATA-SCALING-EXPLORATORY-001 = AUTHORIZED FOR EXECUTION" in text,
    )
    return {"path": str(path), "sha256": actual}


def load_continuity_manifest(
    project_root: Path,
    manifest_arg: Path | None,
    gate: Gate,
) -> tuple[Path, dict[str, Any]]:
    path = (
        manifest_arg
        if manifest_arg is not None
        else project_root / CONTINUITY_MANIFEST_REL
    )
    gate.check("Manifest continuidad v2 existe", path.is_file(), path)
    actual = sha256(path)
    gate.check(
        "Manifest continuidad v2 SHA exacto",
        actual == CONTINUITY_MANIFEST_SHA,
        actual,
    )
    obj = json.loads(path.read_text(encoding="utf-8"))
    gate.check(
        "Manifest continuidad v2 campaign exact",
        obj.get("campaign_id") == "PHASE-G-CONTINUITY-SERIES-PREPARATION-V2",
        obj.get("campaign_id"),
    )
    gate.check("Manifest continuidad v2 status PASS", obj.get("status") == "PASS")
    return path, obj


def resolve_input(
    manifest: dict[str, Any],
    key: str,
    input_dir: Path | None,
) -> tuple[Path, str, str]:
    meta = manifest["outputs"][key]
    recorded_path = Path(meta["parquet_path"])
    expected_sha = str(meta["parquet_sha256"])
    path = (
        input_dir / recorded_path.name
        if input_dir is not None
        else recorded_path
    )
    return path, expected_sha, recorded_path.name


def validate_frame(
    path: Path,
    expected_sha: str,
    expected_basename: str,
    expected_rows: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    label: str,
    gate: Gate,
) -> pd.DataFrame:
    gate.check(f"{label}: basename exacto", path.name == expected_basename, path.name)
    gate.check(f"{label}: existe", path.is_file(), path)
    actual = sha256(path)
    gate.check(f"{label}: SHA exacto", actual == expected_sha, actual)

    frame = pd.read_parquet(path).copy()
    gate.check(f"{label}: filas exactas", len(frame) == expected_rows, len(frame))
    gate.check(
        f"{label}: esquema exacto",
        set(frame.columns) == REQUIRED_COLUMNS,
        f"{len(frame.columns)} columnas",
    )

    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    gate.check(f"{label}: timestamps únicos", not frame[TIME_COL].duplicated().any())
    gate.check(f"{label}: timestamps ordenados", frame[TIME_COL].is_monotonic_increasing)
    gate.check(
        f"{label}: cadencia 5min exacta",
        bool((frame[TIME_COL].diff().dropna() == pd.Timedelta(minutes=5)).all()),
    )
    gate.check(f"{label}: inicio exacto", frame[TIME_COL].iloc[0] == expected_start, frame[TIME_COL].iloc[0])
    gate.check(f"{label}: fin exacto", frame[TIME_COL].iloc[-1] == expected_end, frame[TIME_COL].iloc[-1])

    values = frame[list(VAR_COLUMNS)].to_numpy(dtype=float)
    gate.check(f"{label}: variables VAR finitas", np.isfinite(values).all())
    gate.check(f"{label}: variables VAR no negativas", (values >= 0).all())

    return frame


def expected_train_common_count(train_rows: int, horizon: int) -> int:
    return int(train_rows - horizon - L_MAX + 1)


def training_common_origins(train_rows: int, horizon: int) -> np.ndarray:
    return np.arange(L_MAX - 1, train_rows - horizon, dtype=np.int64)


def inner_split(origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(origins)
    n_available = n - INNER_GAP
    if n_available <= 0:
        raise StopError("No hay suficientes orígenes para inner split.")
    early_count = math.floor(INNER_EARLY_STOP_FRACTION * n_available)
    fit_count = n_available - early_count
    fit = origins[:fit_count]
    early = origins[fit_count + INNER_GAP:]
    if len(fit) + INNER_GAP + len(early) != n:
        raise StopError("Inner split no reconcilia.")
    if len(fit) == 0 or len(early) == 0:
        raise StopError("Inner split vacío.")
    return fit, early


def blind_origins(train_rows: int, test_rows: int, horizon: int) -> np.ndarray:
    return np.arange(
        train_rows - 1,
        train_rows + test_rows - horizon,
        dtype=np.int64,
    )


def build_xy_from_origins(
    y: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    horizon: int,
    scaler: ScalarStandardizer,
    with_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.float32] = []
    targets: list[int] = []

    for origin_value in origins:
        origin = int(origin_value)
        target_idx = origin + horizon
        start = origin - lookback + 1
        if start < 0 or target_idx >= len(y):
            raise StopError("Ventana supervisada fuera de límites.")
        xs.append(
            scaler.transform(y[start:origin + 1])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        targets.append(target_idx)
        if with_targets:
            ys.append(
                np.float32(
                    scaler.transform(
                        np.array([y[target_idx]], dtype=np.float64)
                    )[0]
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
        n += int(len(xb))
    if n == 0:
        raise StopError("Early-stop loader vacío.")
    return total / n


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for (xb,) in loader:
        chunks.append(model(xb.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def make_model(model_name: str, device: torch.device) -> tuple[nn.Module, int]:
    if model_name == "RECURRENT*":
        model = RecurrentRegressor().to(device)
        expected = int(RECURRENT_CONFIG["expected_parameter_count"])
    elif model_name == "TRANSFORMER*":
        model = TransformerLite().to(device)
        expected = int(TRANSFORMER_CONFIG["expected_parameter_count"])
    else:
        raise StopError(f"Modelo neuronal desconocido: {model_name}")

    observed = parameter_count(model)
    if observed != expected:
        raise StopError(
            f"{model_name}: parameter count {observed} != {expected}"
        )
    return model, observed


def model_lookback(model_name: str) -> int:
    if model_name == "RECURRENT*":
        return int(RECURRENT_CONFIG["lookback"])
    if model_name == "TRANSFORMER*":
        return int(TRANSFORMER_CONFIG["lookback"])
    raise StopError(f"Modelo neuronal desconocido: {model_name}")


def select_best_epoch(
    model_name: str,
    horizon: int,
    y_train: np.ndarray,
    fit_origins: np.ndarray,
    early_origins: np.ndarray,
    device: torch.device,
    hard_stage_seconds: float,
) -> dict[str, Any]:
    lookback = model_lookback(model_name)

    fit_target_max = int(fit_origins[-1] + horizon)
    scaler = ScalarStandardizer.fit(y_train[:fit_target_max + 1])

    x_fit, y_fit, _ = build_xy_from_origins(
        y_train, fit_origins, lookback, horizon, scaler, True
    )
    x_early, y_early, _ = build_xy_from_origins(
        y_train, early_origins, lookback, horizon, scaler, True
    )
    assert y_fit is not None and y_early is not None

    set_seed(CANONICAL_SEED)
    model, params = make_model(model_name, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=(WEIGHT_DECAY if model_name == "TRANSFORMER*" else 0.0),
    )
    loss_fn = nn.L1Loss()
    fit_loader = make_loader(x_fit, y_fit)
    early_loader = make_loader(x_early, y_early)

    best_val = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        if time.perf_counter() - started > hard_stage_seconds:
            raise TimeoutError(
                f"RESOURCE_LIMIT inner {model_name}/H{horizon}"
            )

        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise StopError(
                    f"Loss no finita inner {model_name}/H{horizon}"
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

    runtime = float(time.perf_counter() - started)
    if runtime > hard_stage_seconds:
        raise TimeoutError(
            f"RESOURCE_LIMIT inner {model_name}/H{horizon}"
        )
    if best_epoch is None or best_state is None:
        raise StopError("No se obtuvo checkpoint válido de early stopping.")

    return {
        "best_epoch": int(best_epoch),
        "best_early_stop_mae_scaled": float(best_val),
        "inner_scaler_cutoff_index": fit_target_max,
        "inner_scaler_mean_bps": scaler.mean,
        "inner_scaler_scale_bps": scaler.scale,
        "inner_runtime_seconds": runtime,
        "parameter_count": params,
    }


def final_refit(
    model_name: str,
    horizon: int,
    best_epoch: int,
    y_train: np.ndarray,
    train_origins: np.ndarray,
    device: torch.device,
    hard_stage_seconds: float,
) -> tuple[nn.Module, ScalarStandardizer, float, int]:
    lookback = model_lookback(model_name)

    scaler = ScalarStandardizer.fit(y_train)

    x_train, y_supervised, _ = build_xy_from_origins(
        y_train,
        train_origins,
        lookback,
        horizon,
        scaler,
        True,
    )
    assert y_supervised is not None

    set_seed(CANONICAL_SEED)
    model, params = make_model(model_name, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=(WEIGHT_DECAY if model_name == "TRANSFORMER*" else 0.0),
    )
    loss_fn = nn.L1Loss()
    loader = make_loader(x_train, y_supervised)

    started = time.perf_counter()

    for _epoch in range(1, best_epoch + 1):
        if time.perf_counter() - started > hard_stage_seconds:
            raise TimeoutError(
                f"RESOURCE_LIMIT final_refit {model_name}/H{horizon}"
            )

        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise StopError(
                    f"Loss no finita final_refit {model_name}/H{horizon}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    runtime = float(time.perf_counter() - started)
    if runtime > hard_stage_seconds:
        raise TimeoutError(
            f"RESOURCE_LIMIT final_refit {model_name}/H{horizon}"
        )

    return model, scaler, runtime, params


def prediction_frame(
    history_weeks: int,
    model_name: str,
    config_id: str,
    horizon: int,
    origins: np.ndarray,
    targets: np.ndarray,
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence: np.ndarray,
    best_epoch: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame({
        "training_history_weeks": int(history_weeks),
        "model": model_name,
        "config_id": config_id,
        "canonical_seed": (
            CANONICAL_SEED
            if model_name in {"RECURRENT*", "TRANSFORMER*"}
            else np.nan
        ),
        "horizon_steps": int(horizon),
        "horizon_minutes": int(horizon * 5),
        "best_epoch_from_inner_training": (
            int(best_epoch) if best_epoch is not None else np.nan
        ),
        "origin_index": origins.astype(int),
        "target_index": targets.astype(int),
        "origin_timestamp": timestamps.iloc[origins].to_numpy(),
        "target_timestamp": timestamps.iloc[targets].to_numpy(),
        "y_true_bps": y_true.astype(np.float64),
        "y_pred_bps": y_pred.astype(np.float64),
        "persistence_pred_bps": persistence.astype(np.float64),
        "residual_bps": (y_true - y_pred).astype(np.float64),
        "abs_error_bps": np.abs(y_true - y_pred).astype(np.float64),
        "persistence_abs_error_bps": np.abs(y_true - persistence).astype(np.float64),
    })


def neural_predict(
    model_name: str,
    history_weeks: int,
    model: nn.Module,
    scaler: ScalarStandardizer,
    horizon: int,
    best_epoch: int,
    y_full: np.ndarray,
    timestamps: pd.Series,
    train_rows: int,
    test_rows: int,
    device: torch.device,
) -> pd.DataFrame:
    origins = blind_origins(train_rows, test_rows, horizon)
    x_blind, _, targets = build_xy_from_origins(
        y_full,
        origins,
        model_lookback(model_name),
        horizon,
        scaler,
        with_targets=False,
    )

    # Igual que en las fases congeladas: forward causal primero;
    # lectura de targets solo después para evaluación.
    pred_scaled = predict_scaled(model, x_blind, device)
    y_pred = scaler.inverse(pred_scaled)

    y_true = y_full[targets].astype(np.float64)
    persistence = y_full[origins].astype(np.float64)

    config_id = (
        RECURRENT_CONFIG["config_id"]
        if model_name == "RECURRENT*"
        else TRANSFORMER_CONFIG["config_id"]
    )

    return prediction_frame(
        history_weeks,
        model_name,
        str(config_id),
        horizon,
        origins,
        targets,
        timestamps,
        y_true,
        y_pred,
        persistence,
        best_epoch,
    )


def persistence_predictions(
    history_weeks: int,
    y_full: np.ndarray,
    timestamps: pd.Series,
    train_rows: int,
    test_rows: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for h in HORIZONS:
        origins = blind_origins(train_rows, test_rows, h)
        targets = origins + h
        y_true = y_full[targets].astype(np.float64)
        y_pred = y_full[origins].astype(np.float64)
        frames.append(
            prediction_frame(
                history_weeks,
                "Persistence",
                "Persistence",
                h,
                origins,
                targets,
                timestamps,
                y_true,
                y_pred,
                y_pred.copy(),
                None,
            )
        )
    return pd.concat(frames, ignore_index=True)


def fit_var(
    train: pd.DataFrame,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    levels = train.set_index(TIME_COL)[list(VAR_COLUMNS)].astype(float)
    diff = levels.diff().dropna()

    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=0)

    if (
        not np.isfinite(mean.to_numpy(dtype=float)).all()
        or not np.isfinite(std.to_numpy(dtype=float)).all()
        or (std <= 0).any()
    ):
        raise StopError("Scaler VAR inválido.")

    standardized = (diff - mean) / std
    if not np.isfinite(standardized.to_numpy(dtype=float)).all():
        raise StopError("Training VAR estandarizado no finito.")

    scaler = {
        "representation": "first_differences",
        "ddof": 0,
        "columns": list(VAR_COLUMNS),
        "mean": {c: float(mean[c]) for c in VAR_COLUMNS},
        "std": {c: float(std[c]) for c in VAR_COLUMNS},
        "n_transformed_train": int(len(diff)),
    }

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = VAR(standardized).fit(
            maxlags=5,
            ic=None,
            trend="c",
        )
    elapsed = float(time.perf_counter() - started)

    params = np.asarray(result.params, dtype=float)
    roots = np.asarray(result.roots, dtype=complex)

    detail = {
        "attempts": 1,
        "warnings": [
            f"{item.category.__name__}: {item.message}"
            for item in caught
        ],
        "converged": True,
        "finite_params": bool(np.isfinite(params).all()),
        "stable": bool(result.is_stable(verbose=False)),
        "min_abs_root": (
            float(np.min(np.abs(roots))) if roots.size else np.nan
        ),
        "training_time_seconds": elapsed,
        "n_params": int(params.size),
    }
    return result, detail, standardized, scaler


def evaluate_var(
    history_weeks: int,
    result: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_z: pd.DataFrame,
    scaler: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    lag = 5
    history_z = train_z.to_numpy(dtype=float).copy()
    train_levels = train[list(VAR_COLUMNS)].to_numpy(dtype=float)
    test_levels = test[list(VAR_COLUMNS)].to_numpy(dtype=float)

    mean = np.array(
        [scaler["mean"][c] for c in VAR_COLUMNS],
        dtype=float,
    )
    std = np.array(
        [scaler["std"][c] for c in VAR_COLUMNS],
        dtype=float,
    )

    full_timestamps = pd.concat(
        [train[TIME_COL], test[TIME_COL]],
        ignore_index=True,
    )
    train_rows = len(train)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for k in range(len(test)):
        remaining = len(test) - k
        steps = min(max(HORIZONS), remaining)

        origin_idx = train_rows - 1 + k
        origin_level = train_levels[-1] if k == 0 else test_levels[k - 1]

        forecast_z = np.asarray(
            result.forecast(
                history_z[-lag:],
                steps=steps,
            ),
            dtype=float,
        )
        if (
            forecast_z.shape != (steps, len(VAR_COLUMNS))
            or not np.isfinite(forecast_z).all()
        ):
            raise StopError("Forecast VAR inválido.")

        forecast_diff = forecast_z * std + mean
        bitrate_levels = (
            origin_level[0]
            + np.cumsum(forecast_diff[:, 0])
        )

        for h in HORIZONS:
            if h > steps:
                continue
            target_k = k + h - 1
            target_idx = train_rows + target_k
            y_true = float(test_levels[target_k, 0])
            y_pred = float(bitrate_levels[h - 1])
            persistence = float(origin_level[0])

            rows.append({
                "training_history_weeks": int(history_weeks),
                "model": "VAR*",
                "config_id": "VAR(5)",
                "canonical_seed": np.nan,
                "horizon_steps": int(h),
                "horizon_minutes": int(h * 5),
                "best_epoch_from_inner_training": np.nan,
                "origin_index": int(origin_idx),
                "target_index": int(target_idx),
                "origin_timestamp": full_timestamps.iloc[origin_idx],
                "target_timestamp": full_timestamps.iloc[target_idx],
                "y_true_bps": y_true,
                "y_pred_bps": y_pred,
                "persistence_pred_bps": persistence,
                "residual_bps": y_true - y_pred,
                "abs_error_bps": abs(y_true - y_pred),
                "persistence_abs_error_bps": abs(y_true - persistence),
            })

        # R3: se revela el vector real siguiente; no hay refit.
        if k < len(test) - 1:
            previous_real = train_levels[-1] if k == 0 else test_levels[k - 1]
            current_real = test_levels[k]
            observed_diff = current_real - previous_real
            observed_z = (observed_diff - mean) / std
            if not np.isfinite(observed_z).all():
                raise StopError("Actualización R3 VAR no finita.")
            history_z = np.vstack([history_z, observed_z])

    return pd.DataFrame(rows), float(time.perf_counter() - started)


def metric_row(
    predictions: pd.DataFrame,
    mase_scale: float,
) -> dict[str, Any]:
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

    ratio = float(mae / p_mae) if p_mae > 0 else np.nan
    skill = float(1.0 - ratio) if np.isfinite(ratio) else np.nan

    return {
        "n_predictions": int(len(predictions)),
        "mae_bps": mae,
        "persistence_mae_bps": p_mae,
        "mae_ratio_vs_persistence": ratio,
        "skill_vs_persistence": skill,
        "rmse_bps": rmse,
        "smape_pct": float(100.0 * smape.mean()),
        "mase": float(mae / mase_scale),
        "bias_bps": float(residual.mean()),
        "underprediction_pct": float(100.0 * np.mean(yt > yp)),
        "p95_abs_error_bps": float(np.percentile(abs_error, 95)),
    }


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    history_weeks: int,
    model_name: str,
    gate: Gate | None = None,
) -> None:
    for h in HORIZONS:
        sub = predictions[
            predictions["horizon_steps"].eq(h)
        ].sort_values("target_timestamp")
        expected = EXPECTED_TEST_PREDICTIONS[h]
        condition = len(sub) == expected
        if gate is not None:
            gate.check(
                f"{history_weeks}w/{model_name}/H{h}: prediction count",
                condition,
                f"{len(sub)}/{expected}",
            )
        elif not condition:
            raise StopError(
                f"{history_weeks}w/{model_name}/H{h}: "
                f"{len(sub)} != {expected}"
            )

        if len(sub):
            # Política histórica direct multi-horizon:
            # el primer origen es la última observación de training.
            # Por ello H>1 comienza más tarde dentro del test, mientras
            # que el último target siempre coincide con el último punto
            # disponible del test.
            expected_first = TEST_START + pd.Timedelta(minutes=5 * (h - 1))
            expected_last = TEST_END
            if pd.Timestamp(sub["target_timestamp"].iloc[0]) != expected_first:
                raise StopError(
                    f"{history_weeks}w/{model_name}/H{h}: first target incorrecto"
                )
            if pd.Timestamp(sub["target_timestamp"].iloc[-1]) != expected_last:
                raise StopError(
                    f"{history_weeks}w/{model_name}/H{h}: last target incorrecto"
                )


def prepare_inputs(
    project_root: Path,
    manifest_path_arg: Path | None,
    input_dir: Path | None,
    gate: Gate,
) -> tuple[Path, dict[str, Any], dict[int, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    manifest_path, manifest = load_continuity_manifest(
        project_root,
        manifest_path_arg,
        gate,
    )

    trains: dict[int, pd.DataFrame] = {}
    input_registry: dict[str, Any] = {}

    for weeks in HISTORIES:
        key = INPUT_KEYS[weeks]
        path, expected_sha, recorded_basename = resolve_input(
            manifest,
            key,
            input_dir,
        )
        gate.check(
            f"{weeks}w: basename manifest exacto",
            recorded_basename == EXPECTED_BASENAMES[weeks],
            recorded_basename,
        )
        frame = validate_frame(
            path=path,
            expected_sha=expected_sha,
            expected_basename=EXPECTED_BASENAMES[weeks],
            expected_rows=HISTORY_ROWS[weeks],
            expected_start=TRAIN_STARTS[weeks],
            expected_end=TRAIN_END,
            label=f"Train {weeks}w",
            gate=gate,
        )
        trains[weeks] = frame
        input_registry[f"train_{weeks}w"] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(frame),
            "start": frame[TIME_COL].iloc[0].isoformat(),
            "end": frame[TIME_COL].iloc[-1].isoformat(),
        }

    test_path, test_sha, recorded_basename = resolve_input(
        manifest,
        "test_june_week4",
        input_dir,
    )
    gate.check(
        "Test: basename manifest exacto",
        recorded_basename == EXPECTED_BASENAMES["test"],
        recorded_basename,
    )
    test = validate_frame(
        path=test_path,
        expected_sha=test_sha,
        expected_basename=EXPECTED_BASENAMES["test"],
        expected_rows=TEST_ROWS,
        expected_start=TEST_START,
        expected_end=TEST_END,
        label="Test JuneW4",
        gate=gate,
    )
    input_registry["test_june_week4"] = {
        "path": str(test_path),
        "sha256": sha256(test_path),
        "rows": len(test),
        "start": test[TIME_COL].iloc[0].isoformat(),
        "end": test[TIME_COL].iloc[-1].isoformat(),
    }

    # Nesting exacto de ventanas.
    gate.check(
        "Nesting 1w == tail(2w)",
        trains[2].tail(len(trains[1])).reset_index(drop=True).equals(
            trains[1].reset_index(drop=True)
        ),
    )
    gate.check(
        "Nesting 2w == tail(4w)",
        trains[4].tail(len(trains[2])).reset_index(drop=True).equals(
            trains[2].reset_index(drop=True)
        ),
    )
    gate.check(
        "Nesting 4w == tail(8w)",
        trains[8].tail(len(trains[4])).reset_index(drop=True).equals(
            trains[4].reset_index(drop=True)
        ),
    )
    gate.check(
        "Frontera train/test contigua",
        trains[8][TIME_COL].iloc[-1] + pd.Timedelta(minutes=5)
        == test[TIME_COL].iloc[0],
        f"{trains[8][TIME_COL].iloc[-1]} -> {test[TIME_COL].iloc[0]}",
    )

    return manifest_path, manifest, trains, test, input_registry


def structural_model_preflight(
    gate: Gate,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(CANONICAL_SEED)

    recurrent, recurrent_params = make_model("RECURRENT*", device)
    transformer, transformer_params = make_model("TRANSFORMER*", device)

    gate.check(
        "GRU/N06 parameter count",
        recurrent_params == RECURRENT_CONFIG["expected_parameter_count"],
        recurrent_params,
    )
    gate.check(
        "T01/L24 parameter count",
        transformer_params == TRANSFORMER_CONFIG["expected_parameter_count"],
        transformer_params,
    )

    recurrent.eval()
    transformer.eval()
    with torch.no_grad():
        r_out = recurrent(torch.zeros((2, 72, 1), dtype=torch.float32, device=device))
        t_out = transformer(torch.zeros((2, 24, 1), dtype=torch.float32, device=device))

    gate.check("GRU/N06 forward shape", tuple(r_out.shape) == (2, 1), tuple(r_out.shape))
    gate.check("GRU/N06 forward finite", bool(torch.isfinite(r_out).all().item()))
    gate.check("T01/L24 forward shape", tuple(t_out.shape) == (2, 1), tuple(t_out.shape))
    gate.check("T01/L24 forward finite", bool(torch.isfinite(t_out).all().item()))

    del recurrent, transformer, r_out, t_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "recurrent_parameter_count": recurrent_params,
        "transformer_parameter_count": transformer_params,
        "forward_device": str(device),
    }


def preflight(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve()
    gate = Gate()

    print("=" * 132)
    print("PHASE G — SCIENTIFIC RUNNER PREFLIGHT")
    print("=" * 132)
    print("Training científico = 0")
    print("Predicciones científicas = 0")
    print("No se leen TAR.GZ raw.")
    print()

    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {runner_sha}")
    print()

    governance_registry = validate_governance(project_root, gate)

    manifest_path, manifest, trains, test, input_registry = prepare_inputs(
        project_root,
        args.manifest,
        args.input_dir,
        gate,
    )

    # Conteos exactos derivados del protocolo.
    derived_counts: dict[str, Any] = {}
    for weeks in HISTORIES:
        for h in HORIZONS:
            train_rows = HISTORY_ROWS[weeks]
            origins = training_common_origins(train_rows, h)
            expected_common = expected_train_common_count(train_rows, h)
            gate.check(
                f"{weeks}w/H{h}: common-origin count",
                len(origins) == expected_common,
                f"{len(origins)}/{expected_common}",
            )
            fit, early = inner_split(origins)
            gap = int(early[0] - fit[-1] - 1)
            gate.check(
                f"{weeks}w/H{h}: inner gap 12",
                gap == INNER_GAP,
                gap,
            )
            gate.check(
                f"{weeks}w/H{h}: inner split reconcilia",
                len(fit) + INNER_GAP + len(early) == len(origins),
                f"{len(fit)}+{INNER_GAP}+{len(early)}={len(origins)}",
            )
            derived_counts[f"{weeks}w_H{h}"] = {
                "train_common": len(origins),
                "inner_fit": len(fit),
                "inner_gap": INNER_GAP,
                "inner_early_stop": len(early),
                "expected_test_predictions": EXPECTED_TEST_PREDICTIONS[h],
            }

    gate.check(
        "Total predicciones por model-history",
        EXPECTED_TOTAL_PER_MODEL_HISTORY == 7770,
        EXPECTED_TOTAL_PER_MODEL_HISTORY,
    )
    gate.check(
        "Neural direct cells",
        2 * len(HISTORIES) * len(HORIZONS) == 32,
        2 * len(HISTORIES) * len(HORIZONS),
    )
    gate.check(
        "Neural training stages",
        2 * 2 * len(HISTORIES) * len(HORIZONS) == 64,
        2 * 2 * len(HISTORIES) * len(HORIZONS),
    )
    gate.check("VAR fits", len(HISTORIES) == 4, len(HISTORIES))

    info = software_info()
    gate.check("Python major >= 3", sys.version_info.major == 3, info["python"])
    gate.check("PyTorch importado", bool(info["torch"]), info["torch"])
    gate.check("statsmodels importado", bool(info["statsmodels"]), info["statsmodels"])
    gate.check(
        "CUBLAS_WORKSPACE_CONFIG",
        info["cublas_workspace_config"] == ":4096:8",
        info["cublas_workspace_config"],
    )

    device = resolve_device(args.device)
    model_registry = structural_model_preflight(gate, device)

    preflight_out = (
        args.preflight_out_dir
        if args.preflight_out_dir is not None
        else project_root / PREFLIGHT_OUT_REL
    )
    scientific_out = (
        args.scientific_out_dir
        if args.scientific_out_dir is not None
        else project_root / SCIENTIFIC_OUT_REL
    )

    gate.check("Preflight output dir ausente", not preflight_out.exists(), preflight_out)
    gate.check("Scientific output dir ausente", not scientific_out.exists(), scientific_out)

    # Confirmaciones de cero ejecución científica.
    gate.check("Training ejecutado en preflight = 0", True, 0)
    gate.check("Predicciones científicas en preflight = 0", True, 0)
    gate.check("DATA-SCALING-EXPLORATORY-001 no ejecutado", True)

    preflight_out.mkdir(parents=True, exist_ok=False)

    manifest_out = preflight_out / "phase_g_scientific_runner_preflight_v1_2026-08-25_manifest.json"
    report_out = preflight_out / "phase_g_scientific_runner_preflight_v1_2026-08-25_report.txt"

    preflight_manifest = {
        "preflight_id": PREFLIGHT_ID,
        "campaign_id": CAMPAIGN_ID,
        "runner_version": RUNNER_VERSION,
        "runner_path": str(runner_path),
        "runner_sha256": runner_sha,
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "governance": governance_registry,
        "continuity_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "inputs": input_registry,
        "frozen_design": {
            "histories_weeks": list(HISTORIES),
            "history_rows": HISTORY_ROWS,
            "test_rows": TEST_ROWS,
            "test_start": TEST_START,
            "test_end_exclusive": TEST_END_EXCLUSIVE,
            "horizons": list(HORIZONS),
            "expected_test_predictions": EXPECTED_TEST_PREDICTIONS,
            "expected_total_per_model_history": EXPECTED_TOTAL_PER_MODEL_HISTORY,
            "models": ["Persistence", "VAR*", "RECURRENT*", "TRANSFORMER*"],
            "var": {
                "config": "VAR(5)",
                "columns": list(VAR_COLUMNS),
                "representation": "first_differences",
                "ddof": 0,
                "trend": "c",
            },
            "recurrent": RECURRENT_CONFIG,
            "transformer": TRANSFORMER_CONFIG,
            "canonical_seed": CANONICAL_SEED,
            "L_MAX": L_MAX,
            "inner_gap": INNER_GAP,
            "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "derived_counts": derived_counts,
        "model_preflight": model_registry,
        "environment": info,
        "checks_pass": len(gate.checks),
        "checks_total": len(gate.checks),
        "assertions": {
            "training_executed": False,
            "scientific_predictions_generated": False,
            "scientific_outputs_written": False,
            "raw_tar_access": False,
            "model_reselection": False,
            "retuning": False,
            "best_seed_selection": False,
            "phase_a_to_f_modified": False,
            "data_scaling_exploratory_executed": False,
        },
        "execution_gate": {
            "status": "BLOCKED_PENDING_074",
            "required_args": [
                "--authorization-074",
                "--authorization-074-sha",
            ],
        },
    }

    report = "\n".join([
        "=" * 132,
        "PHASE G — SCIENTIFIC RUNNER PREFLIGHT",
        "=" * 132,
        "Status: PASS",
        f"Checks PASS/TOTAL: {len(gate.checks)}/{len(gate.checks)}",
        f"Runner SHA-256: {runner_sha}",
        f"Continuity manifest SHA-256: {sha256(manifest_path)}",
        f"Device usado para forward estructural: {device}",
        f"CUDA disponible: {info['cuda_available']}",
        f"GPU: {info['gpu_name']}",
        "Training científico ejecutado: 0",
        "Predicciones científicas generadas: 0",
        "DATA-SCALING-EXPLORATORY-001: NOT RUN",
        "Scientific execution: BLOCKED_PENDING_074",
        "",
        f"{PREFLIGHT_ID}: PASS",
        "",
    ])

    atomic_json(manifest_out, preflight_manifest)
    atomic_text(report_out, report)

    print()
    print("=" * 132)
    print("FINAL PREFLIGHT SUMMARY")
    print("=" * 132)
    print(f"Checks PASS/TOTAL: {len(gate.checks)}/{len(gate.checks)}")
    print("Training científico ejecutado: 0")
    print("Predicciones científicas generadas: 0")
    print("DATA-SCALING-EXPLORATORY-001: NOT RUN")
    print(f"{PREFLIGHT_ID}: PASS")
    print("SCIENTIFIC EXECUTION: BLOCKED_PENDING_074")
    print()
    print(manifest_out)
    print(f"SHA-256 = {sha256(manifest_out)}")
    print(report_out)
    print(f"SHA-256 = {sha256(report_out)}")
    return 0


def scientific_execution(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve()
    gate = Gate()

    print("=" * 132)
    print("PHASE G — DATA-SCALING-EXPLORATORY-001")
    print("=" * 132)

    runner_path = Path(__file__).resolve()
    runner_sha = sha256(runner_path)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {runner_sha}")
    print()

    governance_registry = validate_governance(project_root, gate)
    authorization = validate_execution_authorization(
        args.authorization_074,
        args.authorization_074_sha,
        gate,
    )

    manifest_path, manifest, trains, test, input_registry = prepare_inputs(
        project_root,
        args.manifest,
        args.input_dir,
        gate,
    )

    scientific_out = (
        args.scientific_out_dir
        if args.scientific_out_dir is not None
        else project_root / SCIENTIFIC_OUT_REL
    )
    gate.check("Scientific output dir ausente", not scientific_out.exists(), scientific_out)

    device = resolve_device(args.device)
    info = software_info()
    print(f"Execution device: {device}")
    print(f"GPU: {info['gpu_name']}")
    print()

    # Forward estructural antes del primer fit.
    structural_model_preflight(gate, device)

    # El directorio científico se crea únicamente al publicar los artefactos
    # finales. Si una ejecución inesperada falla durante training, no deja un
    # directorio científico vacío que bloquee una repetición controlada.
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    cell_status: list[dict[str, Any]] = []

    campaign_started = time.perf_counter()

    for weeks in HISTORIES:
        print()
        print("#" * 132)
        print(f"HISTORY = {weeks} WEEK(S)")
        print("#" * 132)

        train = trains[weeks].copy()
        train_rows = len(train)
        test_rows = len(test)

        full = pd.concat([train, test], ignore_index=True)
        full[TIME_COL] = pd.to_datetime(full[TIME_COL], errors="raise")
        y_train = train[TARGET].to_numpy(dtype=float)
        y_full = full[TARGET].to_numpy(dtype=float)
        timestamps = full[TIME_COL]

        mase_scale = float(np.mean(np.abs(np.diff(y_train))))
        if not np.isfinite(mase_scale) or mase_scale <= 0:
            raise StopError(f"{weeks}w: MASE scale inválida.")

        # Persistence.
        started = time.perf_counter()
        p_pred = persistence_predictions(
            weeks,
            y_full,
            timestamps,
            train_rows,
            test_rows,
        )
        p_runtime = float(time.perf_counter() - started)
        validate_prediction_coverage(p_pred, weeks, "Persistence")
        prediction_frames.append(p_pred)

        for h in HORIZONS:
            sub = p_pred[p_pred["horizon_steps"].eq(h)]
            metric_rows.append({
                "training_history_weeks": weeks,
                "model": "Persistence",
                "config_id": "Persistence",
                "horizon_steps": h,
                "horizon_minutes": h * 5,
                "status": "PASS",
                **metric_row(sub, mase_scale),
            })
        runtime_rows.append({
            "training_history_weeks": weeks,
            "model": "Persistence",
            "config_id": "Persistence",
            "horizon_steps": np.nan,
            "status": "PASS",
            "training_time_seconds": 0.0,
            "inference_time_seconds": p_runtime,
            "total_time_seconds": p_runtime,
            "parameter_count": 0,
        })
        cell_status.append({
            "training_history_weeks": weeks,
            "model": "Persistence",
            "status": "PASS",
        })

        # VAR*.
        var_started = time.perf_counter()
        try:
            result, detail, train_z, scaler = fit_var(train)
            if not detail["finite_params"]:
                raise StopError(f"{weeks}w VAR*: parámetros no finitos.")
            if not detail["stable"]:
                raise StopError(
                    f"{weeks}w VAR*: VAR(5) inestable; no se sustituye lag."
                )

            var_pred, var_infer = evaluate_var(
                weeks,
                result,
                train,
                test,
                train_z,
                scaler,
            )
            validate_prediction_coverage(var_pred, weeks, "VAR*")
            prediction_frames.append(var_pred)

            for h in HORIZONS:
                sub = var_pred[var_pred["horizon_steps"].eq(h)]
                metric_rows.append({
                    "training_history_weeks": weeks,
                    "model": "VAR*",
                    "config_id": "VAR(5)",
                    "horizon_steps": h,
                    "horizon_minutes": h * 5,
                    "status": "PASS",
                    **metric_row(sub, mase_scale),
                })

            runtime_rows.append({
                "training_history_weeks": weeks,
                "model": "VAR*",
                "config_id": "VAR(5)",
                "horizon_steps": np.nan,
                "status": "PASS",
                "training_time_seconds": detail["training_time_seconds"],
                "inference_time_seconds": var_infer,
                "total_time_seconds": float(time.perf_counter() - var_started),
                "parameter_count": detail["n_params"],
                "min_abs_root": detail["min_abs_root"],
                "stable": detail["stable"],
            })
            cell_status.append({
                "training_history_weeks": weeks,
                "model": "VAR*",
                "status": "PASS",
            })
        except Exception as exc:
            runtime_rows.append({
                "training_history_weeks": weeks,
                "model": "VAR*",
                "config_id": "VAR(5)",
                "horizon_steps": np.nan,
                "status": "FAIL",
                "training_time_seconds": np.nan,
                "inference_time_seconds": np.nan,
                "total_time_seconds": float(time.perf_counter() - var_started),
                "parameter_count": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            })
            cell_status.append({
                "training_history_weeks": weeks,
                "model": "VAR*",
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[VAR*] {weeks}w FAIL: {type(exc).__name__}: {exc}", flush=True)

        # Modelos neuronales directos.
        for model_name in ("RECURRENT*", "TRANSFORMER*"):
            for h in HORIZONS:
                print(
                    f"[{model_name}] {weeks}w H{h} — inner + final refit",
                    flush=True,
                )
                cell_started = time.perf_counter()
                try:
                    train_origins = training_common_origins(train_rows, h)
                    fit_origins, early_origins = inner_split(train_origins)

                    epoch_info = select_best_epoch(
                        model_name=model_name,
                        horizon=h,
                        y_train=y_train,
                        fit_origins=fit_origins,
                        early_origins=early_origins,
                        device=device,
                        hard_stage_seconds=float(args.hard_stage_seconds),
                    )

                    model, final_scaler, final_runtime, params = final_refit(
                        model_name=model_name,
                        horizon=h,
                        best_epoch=epoch_info["best_epoch"],
                        y_train=y_train,
                        train_origins=train_origins,
                        device=device,
                        hard_stage_seconds=float(args.hard_stage_seconds),
                    )

                    infer_started = time.perf_counter()
                    pred = neural_predict(
                        model_name=model_name,
                        history_weeks=weeks,
                        model=model,
                        scaler=final_scaler,
                        horizon=h,
                        best_epoch=epoch_info["best_epoch"],
                        y_full=y_full,
                        timestamps=timestamps,
                        train_rows=train_rows,
                        test_rows=test_rows,
                        device=device,
                    )
                    inference_runtime = float(time.perf_counter() - infer_started)

                    expected = EXPECTED_TEST_PREDICTIONS[h]
                    if len(pred) != expected:
                        raise StopError(
                            f"{model_name}/{weeks}w/H{h}: "
                            f"{len(pred)} != {expected}"
                        )

                    prediction_frames.append(pred)
                    metric_rows.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "config_id": (
                            RECURRENT_CONFIG["config_id"]
                            if model_name == "RECURRENT*"
                            else TRANSFORMER_CONFIG["config_id"]
                        ),
                        "horizon_steps": h,
                        "horizon_minutes": h * 5,
                        "status": "PASS",
                        **metric_row(pred, mase_scale),
                    })

                    epoch_rows.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "config_id": (
                            RECURRENT_CONFIG["config_id"]
                            if model_name == "RECURRENT*"
                            else TRANSFORMER_CONFIG["config_id"]
                        ),
                        "canonical_seed": CANONICAL_SEED,
                        "horizon_steps": h,
                        "train_common_samples": len(train_origins),
                        "inner_fit_samples": len(fit_origins),
                        "inner_gap": INNER_GAP,
                        "inner_early_stop_samples": len(early_origins),
                        "best_epoch": epoch_info["best_epoch"],
                        "best_early_stop_mae_scaled": epoch_info["best_early_stop_mae_scaled"],
                        "inner_scaler_cutoff_index": epoch_info["inner_scaler_cutoff_index"],
                        "inner_scaler_mean_bps": epoch_info["inner_scaler_mean_bps"],
                        "inner_scaler_scale_bps": epoch_info["inner_scaler_scale_bps"],
                        "final_scaler_mean_bps": final_scaler.mean,
                        "final_scaler_scale_bps": final_scaler.scale,
                        "parameter_count": params,
                        "status": "PASS",
                    })

                    runtime_rows.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "config_id": (
                            RECURRENT_CONFIG["config_id"]
                            if model_name == "RECURRENT*"
                            else TRANSFORMER_CONFIG["config_id"]
                        ),
                        "horizon_steps": h,
                        "status": "PASS",
                        "training_time_seconds": (
                            epoch_info["inner_runtime_seconds"]
                            + final_runtime
                        ),
                        "inner_runtime_seconds": epoch_info["inner_runtime_seconds"],
                        "final_refit_runtime_seconds": final_runtime,
                        "inference_time_seconds": inference_runtime,
                        "total_time_seconds": float(time.perf_counter() - cell_started),
                        "parameter_count": params,
                        "best_epoch": epoch_info["best_epoch"],
                    })

                    cell_status.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "horizon_steps": h,
                        "status": "PASS",
                    })

                    print(
                        f"    PASS best_epoch={epoch_info['best_epoch']} "
                        f"params={params} total={time.perf_counter()-cell_started:.2f}s",
                        flush=True,
                    )

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as exc:
                    runtime_rows.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "config_id": (
                            RECURRENT_CONFIG["config_id"]
                            if model_name == "RECURRENT*"
                            else TRANSFORMER_CONFIG["config_id"]
                        ),
                        "horizon_steps": h,
                        "status": "FAIL",
                        "training_time_seconds": np.nan,
                        "inference_time_seconds": np.nan,
                        "total_time_seconds": float(time.perf_counter() - cell_started),
                        "parameter_count": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    epoch_rows.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "config_id": (
                            RECURRENT_CONFIG["config_id"]
                            if model_name == "RECURRENT*"
                            else TRANSFORMER_CONFIG["config_id"]
                        ),
                        "canonical_seed": CANONICAL_SEED,
                        "horizon_steps": h,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    cell_status.append({
                        "training_history_weeks": weeks,
                        "model": model_name,
                        "horizon_steps": h,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(
                        f"    FAIL {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    runtime = pd.DataFrame(runtime_rows)
    epochs = pd.DataFrame(epoch_rows)
    status_frame = pd.DataFrame(cell_status)

    # Verificaciones globales.
    failures = status_frame[status_frame["status"].ne("PASS")]
    overall_status = "PASS" if failures.empty else "PARTIAL_FAIL"

    # Para las celdas PASS, cobertura exacta.
    for weeks in HISTORIES:
        for model_name in ("Persistence", "VAR*", "RECURRENT*", "TRANSFORMER*"):
            model_status = status_frame[
                status_frame["training_history_weeks"].eq(weeks)
                & status_frame["model"].eq(model_name)
            ]
            if model_status.empty or not model_status["status"].eq("PASS").all():
                continue
            subset = predictions[
                predictions["training_history_weeks"].eq(weeks)
                & predictions["model"].eq(model_name)
            ]
            validate_prediction_coverage(subset, weeks, model_name)

    # Scaling summary: ratio de MAE frente a la misma identidad con 1 semana.
    summary_rows: list[dict[str, Any]] = []
    for model_name in ("Persistence", "VAR*", "RECURRENT*", "TRANSFORMER*"):
        for h in HORIZONS:
            base = metrics[
                metrics["training_history_weeks"].eq(1)
                & metrics["model"].eq(model_name)
                & metrics["horizon_steps"].eq(h)
                & metrics["status"].eq("PASS")
            ]
            base_mae = (
                float(base["mae_bps"].iloc[0])
                if len(base) == 1
                else np.nan
            )
            for weeks in HISTORIES:
                current = metrics[
                    metrics["training_history_weeks"].eq(weeks)
                    & metrics["model"].eq(model_name)
                    & metrics["horizon_steps"].eq(h)
                    & metrics["status"].eq("PASS")
                ]
                if len(current) != 1:
                    summary_rows.append({
                        "model": model_name,
                        "horizon_steps": h,
                        "horizon_minutes": h * 5,
                        "training_history_weeks": weeks,
                        "status": "NOT_AVAILABLE",
                        "mae_bps": np.nan,
                        "mae_ratio_vs_same_model_1w": np.nan,
                    })
                    continue
                mae = float(current["mae_bps"].iloc[0])
                ratio = (
                    float(mae / base_mae)
                    if np.isfinite(base_mae) and base_mae > 0
                    else np.nan
                )
                summary_rows.append({
                    "model": model_name,
                    "horizon_steps": h,
                    "horizon_minutes": h * 5,
                    "training_history_weeks": weeks,
                    "status": "PASS",
                    "mae_bps": mae,
                    "mae_ratio_vs_same_model_1w": ratio,
                })

    scaling_summary = pd.DataFrame(summary_rows)

    # Reconciliaciones básicas.
    if not predictions.empty:
        if not np.isfinite(
            predictions[
                ["y_true_bps", "y_pred_bps", "residual_bps", "abs_error_bps"]
            ].to_numpy(float)
        ).all():
            raise StopError("Predicciones/resultados no finitos.")
        if not np.allclose(
            predictions["residual_bps"].to_numpy(float),
            predictions["y_true_bps"].to_numpy(float)
            - predictions["y_pred_bps"].to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise StopError("Residuos no reconcilian.")

    gate.check("Scientific output dir sigue ausente antes de publicar", not scientific_out.exists(), scientific_out)
    scientific_out.mkdir(parents=True, exist_ok=False)

    predictions_path = scientific_out / "phase_g_data_scaling_predictions_v1_2026-08-25.parquet"
    metrics_path = scientific_out / "phase_g_data_scaling_metrics_v1_2026-08-25.csv"
    summary_path = scientific_out / "phase_g_data_scaling_summary_v1_2026-08-25.csv"
    runtime_path = scientific_out / "phase_g_data_scaling_runtime_v1_2026-08-25.csv"
    epochs_path = scientific_out / "phase_g_data_scaling_neural_epoch_registry_v1_2026-08-25.csv"
    status_path = scientific_out / "phase_g_data_scaling_cell_status_v1_2026-08-25.csv"
    report_path = scientific_out / "phase_g_data_scaling_report_v1_2026-08-25.txt"
    manifest_out = scientific_out / "phase_g_data_scaling_manifest_v1_2026-08-25.json"

    atomic_parquet(predictions_path, predictions)
    atomic_csv(metrics_path, metrics)
    atomic_csv(summary_path, scaling_summary)
    atomic_csv(runtime_path, runtime)
    atomic_csv(epochs_path, epochs)
    atomic_csv(status_path, status_frame)

    campaign_seconds = float(time.perf_counter() - campaign_started)

    report_lines = [
        "=" * 132,
        "PHASE G — DATA-SCALING-EXPLORATORY-001",
        "=" * 132,
        f"Status: {overall_status}",
        f"Campaign seconds: {campaign_seconds:.6f}",
        f"Histories: {list(HISTORIES)}",
        f"Models: Persistence / VAR(5) / GRU-N06 / Transformer-T01-L24",
        f"Horizons: {list(HORIZONS)}",
        f"Test rows: {TEST_ROWS}",
        f"Predictions rows: {len(predictions)}",
        f"Metric rows: {len(metrics)}",
        f"Neural epoch rows: {len(epochs)}",
        f"Failed cells: {len(failures)}",
        "Phase A-F modified: NO",
        "Model reselection: NO",
        "Retuning: NO",
        "Best-seed selection: NO",
        "",
        "DATA-SCALING-EXPLORATORY-001 = " + overall_status,
        "",
    ]
    atomic_text(report_path, "\n".join(report_lines))

    artifact_paths = {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "scaling_summary": summary_path,
        "runtime": runtime_path,
        "neural_epoch_registry": epochs_path,
        "cell_status": status_path,
        "report": report_path,
    }

    manifest_obj = {
        "campaign_id": CAMPAIGN_ID,
        "runner_version": RUNNER_VERSION,
        "runner_sha256": runner_sha,
        "status": overall_status,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_seconds": campaign_seconds,
        "governance": governance_registry,
        "execution_authorization": authorization,
        "continuity_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "inputs": input_registry,
        "environment": info,
        "device": str(device),
        "frozen_design": {
            "histories_weeks": list(HISTORIES),
            "history_rows": HISTORY_ROWS,
            "test_rows": TEST_ROWS,
            "test_start": TEST_START,
            "test_end_exclusive": TEST_END_EXCLUSIVE,
            "horizons": list(HORIZONS),
            "expected_test_predictions": EXPECTED_TEST_PREDICTIONS,
            "models": ["Persistence", "VAR*", "RECURRENT*", "TRANSFORMER*"],
            "var": "VAR(5)",
            "recurrent": RECURRENT_CONFIG,
            "transformer": TRANSFORMER_CONFIG,
            "canonical_seed": CANONICAL_SEED,
            "L_MAX": L_MAX,
            "inner_gap": INNER_GAP,
            "inner_early_stop_fraction": INNER_EARLY_STOP_FRACTION,
        },
        "counts": {
            "prediction_rows": len(predictions),
            "metric_rows": len(metrics),
            "scaling_summary_rows": len(scaling_summary),
            "runtime_rows": len(runtime),
            "neural_epoch_rows": len(epochs),
            "cell_status_rows": len(status_frame),
            "failed_cells": len(failures),
        },
        "assertions": {
            "phase_a_to_f_modified": False,
            "model_reselection": False,
            "retuning": False,
            "best_seed_selection": False,
            "raw_tar_access": False,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }
    atomic_json(manifest_out, manifest_obj)

    print()
    print("=" * 132)
    print("FINAL SCIENTIFIC SUMMARY")
    print("=" * 132)
    print(f"Status: {overall_status}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Metric rows: {len(metrics)}")
    print(f"Failed cells: {len(failures)}")
    print(f"Campaign seconds: {campaign_seconds:.3f}")
    print(f"Manifest SHA-256: {sha256(manifest_out)}")
    print(f"Report SHA-256: {sha256(report_path)}")
    print(f"{CAMPAIGN_ID} = {overall_status}")

    return 0 if overall_status == "PASS" else 2


def main() -> int:
    args = parse_args()

    if args.hard_stage_seconds <= 0:
        raise StopError("--hard-stage-seconds debe ser > 0.")

    if args.preflight_only:
        if args.authorization_074 is not None or args.authorization_074_sha is not None:
            raise StopError("Preflight no debe recibir autorización 074.")
        return preflight(args)

    # --execute
    return scientific_execution(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StopError, FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print()
        print("=" * 132)
        print("PHASE G RUNNER: FAIL / STOP")
        print("=" * 132)
        print(f"{type(exc).__name__}: {exc}")
        print("No continuar automáticamente.")
        raise SystemExit(1)
