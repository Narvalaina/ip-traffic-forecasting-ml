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


import argparse, copy, hashlib, json, math, os, platform, random, sys, time, zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RUNNER_VERSION = "2.0.0"
RUNNER_DATE = "2026-08-22"
CAMPAIGN_ID = "TRANSFORMER-MARCH-SCREEN-001"
PARENT_CAMPAIGN_ID = "UGR16-TRANSFORMER-LITE-001"

PROTOCOL_NAME = "033_phase_e_transformer_lite_protocol_2026-08-21.md"
PROTOCOL_SHA = "4fcd61519612ae5fd3daa6d874a73df50b27b63531ac109f0782351875211552"
PREFLIGHT_CLOSURE_NAME = "034_phase_e_colab_preflight_closure_2026-08-21.md"
PREFLIGHT_CLOSURE_SHA = "9d89f0bbaa13448c56217c05f1663fa57fdb4505d3a395860186e2235e9cc081"
RUNNER_PREFLIGHT_CLOSURE_NAME = "035_transformer_march_runner_preflight_closure_2026-08-21.md"
RUNNER_PREFLIGHT_CLOSURE_SHA = "2f90c65e1855c98a20470d76229be68359cad8fb33b0b213c5f351eb9e21322e"
RUNTIME_COMPATIBILITY_NAME = "036_phase_e_colab_runtime_compatibility_2026-08-21.md"
RUNTIME_COMPATIBILITY_SHA = "8e8aeb3b0783c5928468bbbf074c1c42359837b57461890c4eb5e1145ca9dbf3"

MARCH_NAME = "march_week3_prepared_5min.parquet"
MARCH_SHA = "fd6daea6007f1b411617a5c67a489fa33c2efda35fdd13616cc66a40134fa1b1"
MARCH_ROWS = 732
MARCH_START = "2016-03-18 11:00:00"
MARCH_END = "2016-03-20 23:55:00"

APRIL_NAME = "april_week3_prepared_5min.parquet"
APRIL_SHA = "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0"
APRIL_ROWS = 2004
APRIL_START = "2016-04-11 01:00:00"
APRIL_END = "2016-04-17 23:55:00"

TARGET, TIME_COL = "bitrate_bps", "timestamp"
L_MAX = 288
HORIZONS = (1, 3, 6, 12)
CANONICAL_SEED = 20260820
N_SPLITS, OUTER_GAP, INNER_GAP = 3, 12, 12
INNER_VAL_FRACTION = 0.15

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

GRID = {
    "T01": {"lookback": 24},
    "T02": {"lookback": 72},
    "T03": {"lookback": 144},
    "T04": {"lookback": 288},
}

ARCHITECTURE = {
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
}

EXPECTED_MARCH_COMMON_COUNTS = {1: 444, 3: 442, 6: 439, 12: 433}
EXPECTED_APRIL_COMMON_COUNTS = {1: 1716, 3: 1714, 6: 1711, 12: 1705}
EXPECTED_MARCH_SPLITS = {
    1:  [(73, 14, 111), (167, 31, 111), (261, 48, 111)],
    3:  [(73, 15, 110), (167, 31, 110), (260, 48, 110)],
    6:  [(73, 15, 109), (166, 31, 109), (259, 47, 109)],
    12: [(71, 14, 108), (163, 30, 108), (255, 46, 108)],
}
EXPECTED_APRIL_SPLITS = {
    1: (1184, 253, 255),
    3: (1183, 253, 254),
    6: (1180, 253, 254),
    12: (1176, 252, 253),
}

OUTPUT_PREFIX = "ugr16_transformer_march_screen"
OUTPUT_SUFFIX = "v2_2026-08-22"



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
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    print(f"{name:48s}: {status}" + (f" | {detail}" if detail else ""))
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
    return [p for p in bundle_root.rglob("*") if p.is_file() and "june" in p.name.lower()]


def validate_frame(path: Path, expected_sha: str, expected_rows: int,
                   expected_start: str, expected_end: str) -> pd.DataFrame:
    check(f"{path.name}: exists", path.is_file(), str(path))
    actual_sha = sha256(path)
    check(f"{path.name}: sha256", actual_sha == expected_sha, actual_sha)
    frame = pd.read_parquet(path)
    check(f"{path.name}: rows", len(frame) == expected_rows, str(len(frame)))
    check(f"{path.name}: columns", set(frame.columns) == REQUIRED_COLUMNS, f"{len(frame.columns)} cols")
    check(f"{path.name}: target finite", np.isfinite(frame[TARGET].to_numpy(float)).all())
    ts = pd.to_datetime(frame[TIME_COL], errors="raise")
    check(f"{path.name}: timestamps unique", not ts.duplicated().any())
    check(f"{path.name}: timestamps sorted", ts.is_monotonic_increasing)
    check(f"{path.name}: start", str(ts.min()) == expected_start, str(ts.min()))
    check(f"{path.name}: end", str(ts.max()) == expected_end, str(ts.max()))
    diffs = ts.diff().dropna()
    check(f"{path.name}: spacing 5 min", bool((diffs == pd.Timedelta(minutes=5)).all()))
    return frame


def common_sample_indices(n_rows: int, horizon: int) -> np.ndarray:
    n = n_rows - L_MAX - horizon + 1
    if n <= 0:
        raise RuntimeError("No common samples")
    return np.arange(n, dtype=np.int64)


def origin_from_common_index(i: int) -> int:
    return (L_MAX - 1) + int(i)


def manual_time_series_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size = n_samples // (N_SPLITS + 1)
    starts = range(n_samples - N_SPLITS * test_size, n_samples, test_size)
    result = []
    for test_start in starts:
        train_end = test_start - OUTER_GAP
        tr = np.arange(0, train_end, dtype=np.int64)
        te = np.arange(test_start, min(test_start + test_size, n_samples), dtype=np.int64)
        if len(tr) == 0 or len(te) == 0:
            raise RuntimeError("Invalid outer split")
        result.append((tr, te))
    return result


def inner_split(outer_train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(outer_train_idx)
    val_n = max(1, int(math.floor(INNER_VAL_FRACTION * n)))
    val_start = n - val_n
    fit_end = val_start - INNER_GAP
    if fit_end <= 0:
        raise RuntimeError("Invalid inner split")
    return outer_train_idx[:fit_end], outer_train_idx[val_start:]


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
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def build_xy(y: np.ndarray, sample_indices: np.ndarray, lookback: int, horizon: int,
             scaler: ScalarStandardizer):
    xs, ys, origins, targets = [], [], [], []
    for i in sample_indices:
        origin = origin_from_common_index(int(i))
        target_idx = origin + horizon
        start = origin - lookback + 1
        xs.append(scaler.transform(y[start:origin + 1]).astype(np.float32).reshape(-1, 1))
        ys.append(np.float32(scaler.transform(np.array([y[target_idx]]))[0]))
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
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :].to(dtype=x.dtype, device=x.device)


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


def make_loader(x, y):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    loss_fn = nn.L1Loss(reduction="sum")
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += float(loss_fn(model(xb), yb).item())
        n += len(xb)
    return total / n


@torch.no_grad()
def predict_scaled(model, x, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH_SIZE,
                        shuffle=False, drop_last=False)
    model.eval()
    chunks = []
    for (xb,) in loader:
        chunks.append(model(xb.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def train_one_fit(config_id, horizon, fold, y, fit_idx, val_idx, test_idx, device):
    lookback = int(GRID[config_id]["lookback"])
    fit_targets = np.array([origin_from_common_index(i) + horizon for i in fit_idx])
    scaler_cutoff = int(fit_targets.max())
    scaler = ScalarStandardizer.fit(y[:scaler_cutoff + 1])

    x_fit, y_fit, _, _ = build_xy(y, fit_idx, lookback, horizon, scaler)
    x_val, y_val, _, _ = build_xy(y, val_idx, lookback, horizon, scaler)
    x_test, _, origins_test, targets_test = build_xy(
        y, test_idx, lookback, horizon, scaler
    )

    set_seed(CANONICAL_SEED)
    model = TransformerLite().to(device)
    check(
        f"{config_id}/H{horizon}/fold{fold}: parameter count",
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        str(parameter_count(model)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.L1Loss()
    fit_loader, val_loader = make_loader(x_fit, y_fit), make_loader(x_val, y_val)

    best_val, best_epoch, best_state = float("inf"), None, None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in fit_loader:
            if time.perf_counter() - started > HARD_FIT_SECONDS:
                raise TimeoutError(
                    f"RESOURCE_LIMIT {config_id}/H{horizon}/fold{fold}"
                )
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        val_mae = eval_loss(model, val_loader, device)
        if val_mae < (best_val - MIN_DELTA):
            best_val = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    runtime_s = time.perf_counter() - started
    if runtime_s > HARD_FIT_SECONDS:
        raise TimeoutError(f"RESOURCE_LIMIT {config_id}/H{horizon}/fold{fold}")
    if best_state is None:
        raise RuntimeError("No valid checkpoint")

    model.load_state_dict(best_state)
    y_pred = scaler.inverse(predict_scaled(model, x_test, device))
    y_true = y[targets_test].astype(np.float64)
    persistence = y[origins_test].astype(np.float64)

    preds = pd.DataFrame({
        "config_id": config_id,
        "lookback": lookback,
        "horizon_steps": horizon,
        "fold": fold,
        "sample_index": test_idx.astype(int),
        "origin_index": origins_test.astype(int),
        "target_index": targets_test.astype(int),
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

    rec = {
        "config_id": config_id,
        "horizon_steps": horizon,
        "fold": fold,
        "lookback": lookback,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "encoder_dropout": 0.10,
        "external_dropout": 0.10,
        "n_fit_samples": len(fit_idx),
        "n_val_samples": len(val_idx),
        "n_test_samples": len(test_idx),
        "scaler_cutoff_index": scaler_cutoff,
        "scaler_mean_bps": scaler.mean,
        "scaler_scale_bps": scaler.scale,
        "mase_scale_bps": mase_scale,
        "best_epoch": int(best_epoch),
        "best_val_mae_scaled": float(best_val),
        "parameter_count": parameter_count(model),
        "runtime_seconds": float(runtime_s),
        "status": "PASS",
    }
    return rec, preds


def summarize_metrics(frame, fold_records):
    yt = frame["y_true_bps"].to_numpy(float)
    yp = frame["y_pred_bps"].to_numpy(float)
    pp = frame["persistence_pred_bps"].to_numpy(float)
    err = yt - yp
    ae = np.abs(err)
    pae = np.abs(yt - pp)
    mae, p_mae = float(ae.mean()), float(pae.mean())
    denom = np.abs(yt) + np.abs(yp)
    smape_terms = np.zeros_like(denom)
    mask = denom > 0
    smape_terms[mask] = 2 * ae[mask] / denom[mask]

    weighted, total_n = 0.0, 0
    for fold, sub in frame.groupby("fold"):
        scale = float(fold_records.loc[fold_records["fold"].eq(fold), "mase_scale_bps"].iloc[0])
        n = len(sub)
        weighted += float(sub["abs_error_bps"].mean() / scale) * n
        total_n += n

    return {
        "n_predictions": len(frame), "mae_bps": mae, "persistence_mae_bps": p_mae,
        "mae_ratio_vs_persistence": mae / p_mae, "skill_vs_persistence": 1 - mae / p_mae,
        "rmse_bps": float(np.sqrt(np.mean(err ** 2))),
        "smape_pct": float(100 * smape_terms.mean()),
        "mase_weighted": weighted / total_n, "bias_bps": float(err.mean()),
        "underprediction_pct": float(100 * np.mean(yt > yp)),
        "p95_abs_error_bps": float(np.percentile(ae, 95)),
    }


def expected_april_split(n_common: int) -> tuple[int, int, int]:
    n_available = n_common - 24
    fit = int(math.floor(0.70 * n_available))
    early = int(math.floor(0.15 * n_available))
    selection = n_available - fit - early
    return fit, early, selection


def preflight(bundle_root: Path):
    print("=" * 118)
    print(f"{CAMPAIGN_ID} — PREFLIGHT SIN ENTRENAMIENTO")
    print("=" * 118)
    print(f"Runner version: {RUNNER_VERSION}")
    print(f"Runner SHA-256: {sha256(Path(__file__).resolve())}")
    print()

    software = validate_software()
    forbidden = find_forbidden_june(bundle_root)
    check("June physically absent", not forbidden, str([str(p) for p in forbidden]))

    protocol = bundle_root / "governance" / PROTOCOL_NAME
    closure = bundle_root / "governance" / PREFLIGHT_CLOSURE_NAME
    runner_preflight_closure = bundle_root / "governance" / RUNNER_PREFLIGHT_CLOSURE_NAME
    runtime_compatibility = bundle_root / "governance" / RUNTIME_COMPATIBILITY_NAME
    check("033 exists", protocol.is_file(), str(protocol))
    check("033 SHA-256", sha256(protocol) == PROTOCOL_SHA, sha256(protocol))
    check("034 exists", closure.is_file(), str(closure))
    check("034 SHA-256", sha256(closure) == PREFLIGHT_CLOSURE_SHA, sha256(closure))
    check("035 exists", runner_preflight_closure.is_file(), str(runner_preflight_closure))
    check("035 SHA-256", sha256(runner_preflight_closure) == RUNNER_PREFLIGHT_CLOSURE_SHA, sha256(runner_preflight_closure))
    check("036 exists", runtime_compatibility.is_file(), str(runtime_compatibility))
    check("036 SHA-256", sha256(runtime_compatibility) == RUNTIME_COMPATIBILITY_SHA, sha256(runtime_compatibility))

    march = validate_frame(
        bundle_root / "inputs" / MARCH_NAME,
        MARCH_SHA, MARCH_ROWS, MARCH_START, MARCH_END
    )
    april = validate_frame(
        bundle_root / "inputs" / APRIL_NAME,
        APRIL_SHA, APRIL_ROWS, APRIL_START, APRIL_END
    )

    for h in HORIZONS:
        march_n = len(common_sample_indices(len(march), h))
        april_n = len(common_sample_indices(len(april), h))
        check(
            f"March common count H{h}",
            march_n == EXPECTED_MARCH_COMMON_COUNTS[h],
            str(march_n),
        )
        check(
            f"April common count H{h}",
            april_n == EXPECTED_APRIL_COMMON_COUNTS[h],
            str(april_n),
        )

        splits = manual_time_series_splits(march_n)
        actual_tuples = []
        for fold, (tr, te) in enumerate(splits, 1):
            fit, val = inner_split(tr)
            actual_tuples.append((len(fit), len(val), len(te)))
            check(
                f"H{h}/fold{fold} inner gap",
                int(val[0] - fit[-1] - 1) == INNER_GAP,
            )
            check(
                f"H{h}/fold{fold} outer gap",
                int(te[0] - tr[-1] - 1) == OUTER_GAP,
            )
        check(
            f"H{h} frozen March split tuples",
            actual_tuples == EXPECTED_MARCH_SPLITS[h],
            repr(actual_tuples),
        )
        april_tuple = expected_april_split(april_n)
        check(
            f"H{h} frozen April split tuple",
            april_tuple == EXPECTED_APRIL_SPLITS[h],
            repr(april_tuple),
        )

    device = torch.device("cuda")
    for lookback in (24, 288):
        set_seed(CANONICAL_SEED)
        model = TransformerLite().to(device)
        check(
            f"Transformer L={lookback} parameter count",
            parameter_count(model) == EXPECTED_PARAMETER_COUNT,
            str(parameter_count(model)),
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, lookback, 1, device=device))
        check(
            f"Transformer L={lookback} forward shape",
            tuple(out.shape) == (4, 1),
            str(tuple(out.shape)),
        )
        check(
            f"Transformer L={lookback} forward finite",
            torch.isfinite(out).all().item(),
        )

    check(
        "grid exactly T01-T04",
        tuple(GRID.keys()) == ("T01", "T02", "T03", "T04"),
        repr(GRID),
    )
    check(
        "expected fit count",
        len(GRID) * len(HORIZONS) * N_SPLITS == 48,
        str(len(GRID) * len(HORIZONS) * N_SPLITS),
    )

    print("TRAINING EXECUTED IN PREFLIGHT: 0")
    print("SCIENTIFIC ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")
    return software, march, april


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["lookback"] = work["config_id"].map(lambda c: int(GRID[c]["lookback"]))
    return work.sort_values(
        ["score", "lookback", "config_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )


def deterministic_zip(zip_path: Path, members: list[Path], arc_root: Path) -> None:
    fixed_time = (2026, 8, 21, 12, 0, 0)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for p in sorted(
            members, key=lambda x: x.relative_to(arc_root).as_posix()
        ):
            arcname = p.relative_to(arc_root).as_posix()
            info = zipfile.ZipInfo(arcname)
            info.date_time = fixed_time
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/transformer_march_screen_001_outputs_v1_2026-08-21')),
    )
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_sha = sha256(Path(__file__).resolve())

    software, march, april = preflight(bundle_root)
    if args.preflight_only:
        return 0

    del april

    expected_paths = {
        "fold_runtime": output_dir / f"{OUTPUT_PREFIX}_fold_runtime_{OUTPUT_SUFFIX}.csv",
        "predictions": output_dir / f"{OUTPUT_PREFIX}_predictions_{OUTPUT_SUFFIX}.parquet",
        "horizon_metrics": output_dir / f"{OUTPUT_PREFIX}_horizon_metrics_{OUTPUT_SUFFIX}.csv",
        "candidate_summary": output_dir / f"{OUTPUT_PREFIX}_candidate_summary_{OUTPUT_SUFFIX}.csv",
        "report": output_dir / f"{OUTPUT_PREFIX}_report_{OUTPUT_SUFFIX}.txt",
        "manifest": output_dir / f"{OUTPUT_PREFIX}_manifest_{OUTPUT_SUFFIX}.json",
        "outputs_zip": output_dir / f"{OUTPUT_PREFIX}_outputs_{OUTPUT_SUFFIX}.zip",
    }

    if output_dir.exists():
        existing = [p for p in output_dir.rglob("*") if p.is_file()]
        if existing:
            raise FileExistsError(
                "Output directory already contains files; refusing overwrite:\\n"
                + "\\n".join(map(str, existing))
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    y = march[TARGET].to_numpy(np.float64)
    timestamps = pd.to_datetime(march[TIME_COL]).reset_index(drop=True)
    device = torch.device("cuda")
    fit_records, pred_frames = [], []
    started = time.perf_counter()

    print("=" * 118)
    print(f"START {CAMPAIGN_ID}")
    print("Expected fits = 48")
    print("March training/scoring = YES")
    print("April training/scoring = NO")
    print("June present = NO")
    print("=" * 118)

    for config_id in GRID:
        for h in HORIZONS:
            splits = manual_time_series_splits(
                len(common_sample_indices(len(march), h))
            )
            for fold, (outer_tr, test_idx) in enumerate(splits, 1):
                fit_idx, val_idx = inner_split(outer_tr)
                print(f"[FIT] {config_id} H{h:<2d} fold={fold}", flush=True)
                rec, preds = train_one_fit(
                    config_id, h, fold, y, fit_idx, val_idx, test_idx, device
                )
                preds["origin_timestamp"] = timestamps.iloc[
                    preds["origin_index"].to_numpy(int)
                ].to_numpy()
                preds["target_timestamp"] = timestamps.iloc[
                    preds["target_index"].to_numpy(int)
                ].to_numpy()
                fit_records.append(rec)
                pred_frames.append(preds)
                print(
                    f"      PASS epoch={rec['best_epoch']} "
                    f"runtime={rec['runtime_seconds']:.2f}s",
                    flush=True,
                )

    fit_runtime = pd.DataFrame(fit_records)
    predictions = pd.concat(pred_frames, ignore_index=True)

    check("fit count 48", len(fit_runtime) == 48, str(len(fit_runtime)))
    check("all fit status PASS", fit_runtime["status"].eq("PASS").all())
    check(
        "predictions finite",
        np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all(),
    )
    check(
        "March hash unchanged",
        sha256(bundle_root / "inputs" / MARCH_NAME) == MARCH_SHA,
    )
    check(
        "April hash unchanged",
        sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA,
    )
    check(
        "033 hash unchanged",
        sha256(bundle_root / "governance" / PROTOCOL_NAME) == PROTOCOL_SHA,
    )
    check(
        "034 hash unchanged",
        sha256(bundle_root / "governance" / PREFLIGHT_CLOSURE_NAME)
        == PREFLIGHT_CLOSURE_SHA,
    )
    check(
        "035 hash unchanged",
        sha256(bundle_root / "governance" / RUNNER_PREFLIGHT_CLOSURE_NAME)
        == RUNNER_PREFLIGHT_CLOSURE_SHA,
    )
    check(
        "036 hash unchanged",
        sha256(bundle_root / "governance" / RUNTIME_COMPATIBILITY_NAME)
        == RUNTIME_COMPATIBILITY_SHA,
    )
    check("June still absent", not find_forbidden_june(bundle_root))

    horizon_rows = []
    for (config_id, h), sub in predictions.groupby(
        ["config_id", "horizon_steps"], sort=True
    ):
        folds = fit_runtime[
            fit_runtime["config_id"].eq(config_id)
            & fit_runtime["horizon_steps"].eq(h)
        ]
        horizon_rows.append(
            {
                "config_id": config_id,
                "lookback": int(GRID[config_id]["lookback"]),
                "horizon_steps": int(h),
                **summarize_metrics(sub, folds),
            }
        )
    horizon_metrics = pd.DataFrame(horizon_rows)

    rows = []
    for config_id, sub in horizon_metrics.groupby("config_id", sort=True):
        ratios = {
            int(r.horizon_steps): float(r.mae_ratio_vs_persistence)
            for r in sub.itertuples()
        }
        check(f"{config_id}: four horizons", set(ratios) == set(HORIZONS))
        rows.append(
            {
                "config_id": config_id,
                "lookback": int(GRID[config_id]["lookback"]),
                "score": float(np.mean([ratios[h] for h in HORIZONS])),
                "ratio_H1": ratios[1],
                "ratio_H3": ratios[3],
                "ratio_H6": ratios[6],
                "ratio_H12": ratios[12],
            }
        )

    candidate_summary = pd.DataFrame(rows)
    ranked = rank_candidates(candidate_summary)
    candidate_summary["rank"] = 0
    candidate_summary["selected_top2"] = False
    candidate_summary.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    candidate_summary.loc[ranked.index[:2], "selected_top2"] = True
    candidate_summary["rank"] = candidate_summary["rank"].astype(int)

    check("candidate rows 4", len(candidate_summary) == 4, str(len(candidate_summary)))
    check(
        "top2 rows 2",
        int(candidate_summary["selected_top2"].sum()) == 2,
    )

    report_lines = [
        "=" * 118,
        f"{CAMPAIGN_ID} — REPORT",
        "=" * 118,
        f"Runner version: {RUNNER_VERSION}",
        f"Runner SHA-256: {runner_sha}",
        f"Protocol 033 SHA-256: {PROTOCOL_SHA}",
        f"Preflight closure 034 SHA-256: {PREFLIGHT_CLOSURE_SHA}",
        f"Runner preflight closure 035 SHA-256: {RUNNER_PREFLIGHT_CLOSURE_SHA}",
        f"Runtime compatibility 036 SHA-256: {RUNTIME_COMPATIBILITY_SHA}",
        "",
        "GOVERNANCE",
        "March role: screening",
        "April training/scoring: NO",
        "June present: NO",
        "Tuning outside frozen grid: NO",
        f"Canonical seed: {CANONICAL_SEED}",
        "Architecture search: NO",
        "Search dimension: lookback only",
        "",
        "TOP-2 FOR APRIL",
    ]
    top2 = candidate_summary[
        candidate_summary["selected_top2"]
    ].sort_values("rank")
    for r in top2.itertuples():
        report_lines.append(
            f"rank={int(r.rank)} config={r.config_id} "
            f"lookback={int(r.lookback)} score={r.score:.9f}"
        )
    report_lines += [
        "",
        f"Fits executed: {len(fit_runtime)}",
        f"Campaign seconds: {time.perf_counter() - started:.3f}",
        "STATUS: PASS",
        "TRANSFORMER*: NOT SELECTED IN MARCH",
        "TRANSFORMER-APRIL-SELECTION-001: NOT RUN",
        "TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED",
    ]
    report = "\\n".join(report_lines) + "\\n"

    paths = expected_paths
    atomic_csv(fit_runtime, paths["fold_runtime"])
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(horizon_metrics, paths["horizon_metrics"])
    atomic_csv(candidate_summary, paths["candidate_summary"])
    atomic_text(report, paths["report"])

    output_records = {
        k: {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
        for k, p in paths.items()
        if k not in {"manifest", "outputs_zip"}
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
            "protocol_033": {"name": PROTOCOL_NAME, "sha256": PROTOCOL_SHA},
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
        },
        "governance": {
            "march_role": "screening",
            "april_used_for_training_or_scoring": False,
            "june_present": False,
            "tuning_outside_033": False,
            "architecture_search": False,
            "search_dimension": "lookback_only",
            "canonical_seed": CANONICAL_SEED,
            "transformer_star_selected": False,
            "top2_for_april_selected": True,
        },
        "inputs": {
            "march": {"sha256": MARCH_SHA, "rows": MARCH_ROWS},
            "april": {
                "sha256": APRIL_SHA,
                "rows": APRIL_ROWS,
                "used_for_training_or_scoring": False,
            },
        },
        "software": software,
        "architecture": {
            **ARCHITECTURE,
            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        },
        "grid": GRID,
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
            "outer_splits": N_SPLITS,
            "outer_gap": OUTER_GAP,
            "inner_validation_fraction": INNER_VAL_FRACTION,
            "inner_gap": INNER_GAP,
        },
        "selection": {
            "score": "mean_h(MAE_candidate_h / MAE_persistence_h)",
            "advance": "top-2 valid candidates",
            "tie_break": [
                "lower score",
                "shorter lookback",
                "config_id ascending",
            ],
            "transformer_star_selected": False,
        },
        "validation": {
            "fit_count_48": len(fit_runtime) == 48,
            "all_fit_status_pass": bool(
                fit_runtime["status"].eq("PASS").all()
            ),
            "candidate_rows_4": len(candidate_summary) == 4,
            "top2_rows_2": int(
                candidate_summary["selected_top2"].sum()
            )
            == 2,
            "june_absent_after_run": not bool(
                find_forbidden_june(bundle_root)
            ),
            "march_hash_unchanged": (
                sha256(bundle_root / "inputs" / MARCH_NAME) == MARCH_SHA
            ),
            "april_hash_unchanged": (
                sha256(bundle_root / "inputs" / APRIL_NAME) == APRIL_SHA
            ),
            "protocol_hash_unchanged": (
                sha256(bundle_root / "governance" / PROTOCOL_NAME)
                == PROTOCOL_SHA
            ),
            "preflight_closure_hash_unchanged": (
                sha256(
                    bundle_root / "governance" / PREFLIGHT_CLOSURE_NAME
                )
                == PREFLIGHT_CLOSURE_SHA
            ),
            "runner_preflight_closure_hash_unchanged": (
                sha256(
                    bundle_root / "governance" / RUNNER_PREFLIGHT_CLOSURE_NAME
                )
                == RUNNER_PREFLIGHT_CLOSURE_SHA
            ),
            "runtime_compatibility_hash_unchanged": (
                sha256(
                    bundle_root / "governance" / RUNTIME_COMPATIBILITY_NAME
                )
                == RUNTIME_COMPATIBILITY_SHA
            ),
        },
        "campaign_seconds": float(time.perf_counter() - started),
        "outputs": output_records,
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError("Final validation failed")
    atomic_json(manifest, paths["manifest"])

    deterministic_zip(
        paths["outputs_zip"],
        [p for k, p in paths.items() if k != "outputs_zip"],
        output_dir,
    )

    print(report)
    print("VALIDATIONS")
    for k, v in manifest["validation"].items():
        print(f"{k:58s}: {'PASS' if v else 'FAIL'}")
    print(f"Outputs ZIP: {paths['outputs_zip']}")
    print(f"Outputs ZIP SHA-256: {sha256(paths['outputs_zip'])}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("TRANSFORMER*: NOT SELECTED IN MARCH")
    print("TRANSFORMER-APRIL-SELECTION-001: NOT RUN")
    print("TRANSFORMER-JUNE-BLIND-001: NOT AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
