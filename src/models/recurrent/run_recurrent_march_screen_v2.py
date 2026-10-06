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

RUNNER_VERSION = "1.0.1"
CAMPAIGN_ID = "RECURRENT-MARCH-SCREEN-001"
PARENT_CAMPAIGN_ID = "UGR16-RECURRENT-NETWORKS-001"

PROTOCOL_NAME = "023_recurrent_networks_protocol_2026-08-20.md"
PROTOCOL_SHA = "af9d0d5984fd5e98c6660227f7bd98e246dbf934ad8df416439c2fd5bb59b174"

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
    "N01": {"lookback": 12, "hidden_size": 16, "n_layers": 1, "dropout": 0.00},
    "N02": {"lookback": 24, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N03": {"lookback": 72, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N04": {"lookback": 144, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N05": {"lookback": 288, "hidden_size": 32, "n_layers": 1, "dropout": 0.10},
    "N06": {"lookback": 72, "hidden_size": 64, "n_layers": 2, "dropout": 0.20},
}
FAMILIES = ("SimpleRNN", "LSTM", "GRU")


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


class RecurrentRegressor(nn.Module):
    def __init__(self, family: str, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        cls = {"SimpleRNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[family]
        self.recurrent = cls(
            input_size=1, hidden_size=hidden_size, num_layers=n_layers,
            bias=True, batch_first=True, dropout=0.0, bidirectional=False
        )
        self.external_dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        seq, _ = self.recurrent(x)
        return self.output(self.external_dropout(seq[:, -1, :]))


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


def train_one_fit(family, config_id, horizon, fold, y, fit_idx, val_idx, test_idx, device):
    cfg = GRID[config_id]
    lookback = int(cfg["lookback"])
    fit_targets = np.array([origin_from_common_index(i) + horizon for i in fit_idx])
    scaler_cutoff = int(fit_targets.max())
    scaler = ScalarStandardizer.fit(y[:scaler_cutoff + 1])

    x_fit, y_fit, _, _ = build_xy(y, fit_idx, lookback, horizon, scaler)
    x_val, y_val, _, _ = build_xy(y, val_idx, lookback, horizon, scaler)
    x_test, _, origins_test, targets_test = build_xy(y, test_idx, lookback, horizon, scaler)

    set_seed(CANONICAL_SEED)
    model = RecurrentRegressor(
        family, int(cfg["hidden_size"]), int(cfg["n_layers"]), float(cfg["dropout"])
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()
    fit_loader, val_loader = make_loader(x_fit, y_fit), make_loader(x_val, y_val)

    best_val, best_epoch, best_state = float("inf"), None, None
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in fit_loader:
            if time.perf_counter() - started > HARD_FIT_SECONDS:
                raise TimeoutError(f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}/fold{fold}")
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
        raise TimeoutError(f"RESOURCE_LIMIT {family}/{config_id}/H{horizon}/fold{fold}")
    if best_state is None:
        raise RuntimeError("No valid checkpoint")

    model.load_state_dict(best_state)
    y_pred = scaler.inverse(predict_scaled(model, x_test, device))
    y_true = y[targets_test].astype(np.float64)
    persistence = y[origins_test].astype(np.float64)

    preds = pd.DataFrame({
        "family": family, "config_id": config_id, "horizon_steps": horizon,
        "fold": fold, "sample_index": test_idx.astype(int),
        "origin_index": origins_test.astype(int), "target_index": targets_test.astype(int),
        "y_true_bps": y_true, "y_pred_bps": y_pred,
        "persistence_pred_bps": persistence,
        "residual_bps": y_true - y_pred,
        "abs_error_bps": np.abs(y_true - y_pred),
        "persistence_abs_error_bps": np.abs(y_true - persistence),
    })

    mase_scale = float(np.mean(np.abs(np.diff(y[:scaler_cutoff + 1]))))
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise RuntimeError("Invalid MASE scale")

    rec = {
        "family": family, "config_id": config_id, "horizon_steps": horizon, "fold": fold,
        "lookback": lookback, "hidden_size": int(cfg["hidden_size"]),
        "n_layers": int(cfg["n_layers"]), "dropout": float(cfg["dropout"]),
        "n_fit_samples": len(fit_idx), "n_val_samples": len(val_idx),
        "n_test_samples": len(test_idx), "scaler_cutoff_index": scaler_cutoff,
        "scaler_mean_bps": scaler.mean, "scaler_scale_bps": scaler.scale,
        "mase_scale_bps": mase_scale, "best_epoch": int(best_epoch),
        "best_val_mae_scaled": float(best_val), "parameter_count": parameter_count(model),
        "runtime_seconds": float(runtime_s), "status": "PASS",
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
    check("023 exists", protocol.is_file(), str(protocol))
    check("023 SHA-256", sha256(protocol) == PROTOCOL_SHA, sha256(protocol))

    march = validate_frame(bundle_root/"inputs"/MARCH_NAME, MARCH_SHA, MARCH_ROWS, MARCH_START, MARCH_END)
    april = validate_frame(bundle_root/"inputs"/APRIL_NAME, APRIL_SHA, APRIL_ROWS, APRIL_START, APRIL_END)

    expected_counts = {1: 444, 3: 442, 6: 439, 12: 433}
    for h in HORIZONS:
        n = len(common_sample_indices(len(march), h))
        check(f"March common count H{h}", n == expected_counts[h], str(n))
        splits = manual_time_series_splits(n)
        check(f"March outer splits H{h}", len(splits) == 3)
        for fold, (tr, te) in enumerate(splits, 1):
            fit, val = inner_split(tr)
            check(f"H{h}/fold{fold} fit>0", len(fit) > 0, str(len(fit)))
            check(f"H{h}/fold{fold} val>0", len(val) > 0, str(len(val)))
            check(f"H{h}/fold{fold} test>0", len(te) > 0, str(len(te)))
            check(f"H{h}/fold{fold} inner gap", int(val[0]-fit[-1]-1) == INNER_GAP)
            check(f"H{h}/fold{fold} outer gap", int(te[0]-tr[-1]-1) == OUTER_GAP)

    device = torch.device("cuda")
    set_seed(CANONICAL_SEED)
    x = torch.randn(4, 12, 1, device=device)
    for family in FAMILIES:
        m = RecurrentRegressor(family, 8, 1, 0.0).to(device)
        with torch.no_grad():
            out = m(x)
        check(f"{family} forward smoke", tuple(out.shape) == (4,1), str(tuple(out.shape)))

    print("TRAINING EXECUTED IN PREFLIGHT: 0")
    print("ARTIFACTS WRITTEN IN PREFLIGHT: 0")
    print(f"{CAMPAIGN_ID} PREFLIGHT: PASS")
    return software, march, april



def rank_with_tolerance(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Rank candidates by score ascending.
    Scores within 1e-12 of the current tie-group anchor are treated as an
    exact numerical tie and ordered by config_id ascending, per protocol 023.
    """
    work = frame.sort_values(["score", "config_id"], kind="mergesort").copy()
    ordered = []
    i = 0
    rows = list(work.itertuples())
    while i < len(rows):
        anchor = float(rows[i].score)
        group = [rows[i]]
        j = i + 1
        while j < len(rows) and abs(float(rows[j].score) - anchor) <= 1e-12:
            group.append(rows[j])
            j += 1
        group = sorted(group, key=lambda r: str(r.config_id))
        ordered.extend(group)
        i = j
    order_index = [r.Index for r in ordered]
    return frame.loc[order_index].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path((str(_TFM_PUBLIC_COLAB_WORKDIR) + '/recurrent_march_screen_v1_outputs')))
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    runner_sha = sha256(Path(__file__).resolve())

    software, march, april = preflight(bundle_root)
    if args.preflight_only:
        return 0

    del april
    expected = [
        output_dir/"ugr16_recurrent_march_screen_fold_runtime.csv",
        output_dir/"ugr16_recurrent_march_screen_predictions.parquet",
        output_dir/"ugr16_recurrent_march_screen_horizon_metrics.csv",
        output_dir/"ugr16_recurrent_march_screen_candidate_summary.csv",
        output_dir/"ugr16_recurrent_march_screen_report.txt",
        output_dir/"ugr16_recurrent_march_screen_manifest.json",
        output_dir/"ugr16_recurrent_march_screen_outputs_v1.zip",
    ]
    existing = [p for p in expected if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Existing outputs; refusing overwrite:\n" + "\n".join(map(str, existing)))

    output_dir.mkdir(parents=True, exist_ok=True)

    y = march[TARGET].to_numpy(np.float64)
    timestamps = pd.to_datetime(march[TIME_COL]).reset_index(drop=True)
    device = torch.device("cuda")
    fit_records, pred_frames = [], []
    started = time.perf_counter()

    print("=" * 118)
    print(f"START {CAMPAIGN_ID}")
    print("Expected fits = 216")
    print("=" * 118)

    for family in FAMILIES:
        for config_id in GRID:
            for h in HORIZONS:
                splits = manual_time_series_splits(len(common_sample_indices(len(march), h)))
                for fold, (outer_tr, test_idx) in enumerate(splits, 1):
                    fit_idx, val_idx = inner_split(outer_tr)
                    print(f"[FIT] {family:10s} {config_id} H{h:<2d} fold={fold}", flush=True)
                    rec, preds = train_one_fit(
                        family, config_id, h, fold, y, fit_idx, val_idx, test_idx, device
                    )
                    preds["origin_timestamp"] = timestamps.iloc[preds["origin_index"].to_numpy(int)].to_numpy()
                    preds["target_timestamp"] = timestamps.iloc[preds["target_index"].to_numpy(int)].to_numpy()
                    fit_records.append(rec)
                    pred_frames.append(preds)
                    print(f"      PASS epoch={rec['best_epoch']} runtime={rec['runtime_seconds']:.2f}s", flush=True)

    fit_runtime = pd.DataFrame(fit_records)
    predictions = pd.concat(pred_frames, ignore_index=True)
    check("fit count 216", len(fit_runtime) == 216, str(len(fit_runtime)))
    check("all fit status PASS", fit_runtime["status"].eq("PASS").all())
    check("predictions finite", np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all())
    check("March hash unchanged", sha256(bundle_root/"inputs"/MARCH_NAME) == MARCH_SHA)
    check("April hash unchanged", sha256(bundle_root/"inputs"/APRIL_NAME) == APRIL_SHA)
    check("023 hash unchanged", sha256(bundle_root/"governance"/PROTOCOL_NAME) == PROTOCOL_SHA)
    check("June still absent", not find_forbidden_june(bundle_root))

    horizon_rows = []
    for (family, config_id, h), sub in predictions.groupby(["family","config_id","horizon_steps"], sort=True):
        folds = fit_runtime[
            fit_runtime["family"].eq(family) &
            fit_runtime["config_id"].eq(config_id) &
            fit_runtime["horizon_steps"].eq(h)
        ]
        horizon_rows.append({
            "family": family, "config_id": config_id, "horizon_steps": int(h),
            **summarize_metrics(sub, folds)
        })
    horizon_metrics = pd.DataFrame(horizon_rows)

    rows = []
    for (family, config_id), sub in horizon_metrics.groupby(["family","config_id"], sort=True):
        ratios = {int(r.horizon_steps): float(r.mae_ratio_vs_persistence) for r in sub.itertuples()}
        check(f"{family}/{config_id}: four horizons", set(ratios) == set(HORIZONS))
        rows.append({
            "family": family, "config_id": config_id,
            "score": float(np.mean([ratios[h] for h in HORIZONS])),
            "ratio_H1": ratios[1], "ratio_H3": ratios[3],
            "ratio_H6": ratios[6], "ratio_H12": ratios[12],
        })
    candidate_summary = pd.DataFrame(rows)
    candidate_summary["rank_within_family"] = 0
    candidate_summary["selected_top2"] = False

    for family in FAMILIES:
        sub = rank_with_tolerance(
            candidate_summary[candidate_summary["family"].eq(family)]
        )
        candidate_summary.loc[sub.index, "rank_within_family"] = np.arange(
            1, len(sub) + 1
        )
        candidate_summary.loc[sub.index[:2], "selected_top2"] = True

    candidate_summary["rank_within_family"] = candidate_summary["rank_within_family"].astype(int)
    check("candidate rows 18", len(candidate_summary) == 18, str(len(candidate_summary)))
    check("top2 rows 6", int(candidate_summary["selected_top2"].sum()) == 6)

    report_lines = [
        "="*118, f"{CAMPAIGN_ID} — REPORT", "="*118,
        f"Runner version: {RUNNER_VERSION}", f"Runner SHA-256: {runner_sha}",
        f"Protocol SHA-256: {PROTOCOL_SHA}", "",
        "GOVERNANCE", "March role: screening", "April training/scoring: NO",
        "June present: NO", "Tuning outside frozen grid: NO",
        f"Canonical seed: {CANONICAL_SEED}", "", "TOP-2 PER FAMILY",
    ]
    for family in FAMILIES:
        sub = candidate_summary[
            candidate_summary["family"].eq(family) & candidate_summary["selected_top2"]
        ].sort_values(["rank_within_family","config_id"])
        for r in sub.itertuples():
            report_lines.append(
                f"{family:10s} rank={int(r.rank_within_family)} config={r.config_id} score={r.score:.9f}"
            )
    report_lines += [
        "", f"Fits executed: {len(fit_runtime)}",
        f"Campaign seconds: {time.perf_counter()-started:.3f}",
        "STATUS: PASS",
        "RNN*/LSTM*/GRU*/RECURRENT*: NOT SELECTED IN MARCH",
        "RECURRENT-APRIL-SELECTION-001: NOT RUN",
    ]
    report = "\n".join(report_lines) + "\n"

    paths = {
        "fold_runtime": output_dir/"ugr16_recurrent_march_screen_fold_runtime.csv",
        "predictions": output_dir/"ugr16_recurrent_march_screen_predictions.parquet",
        "horizon_metrics": output_dir/"ugr16_recurrent_march_screen_horizon_metrics.csv",
        "candidate_summary": output_dir/"ugr16_recurrent_march_screen_candidate_summary.csv",
        "report": output_dir/"ugr16_recurrent_march_screen_report.txt",
        "manifest": output_dir/"ugr16_recurrent_march_screen_manifest.json",
    }

    atomic_csv(fit_runtime, paths["fold_runtime"])
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(horizon_metrics, paths["horizon_metrics"])
    atomic_csv(candidate_summary, paths["candidate_summary"])
    atomic_text(report, paths["report"])

    output_records = {
        k: {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
        for k, p in paths.items() if k != "manifest"
    }
    manifest = {
        "campaign_id": CAMPAIGN_ID, "parent_campaign_id": PARENT_CAMPAIGN_ID,
        "status": "PASS",
        "runner": {"version": RUNNER_VERSION, "sha256": runner_sha},
        "protocol": {"name": PROTOCOL_NAME, "sha256": PROTOCOL_SHA},
        "governance": {
            "march_role": "screening", "april_used_for_training_or_scoring": False,
            "june_present": False, "tuning_outside_grid": False,
            "canonical_seed": CANONICAL_SEED, "representatives_selected": False,
            "top2_per_family_selected": True,
        },
        "inputs": {
            "march": {"sha256": MARCH_SHA, "rows": MARCH_ROWS},
            "april": {"sha256": APRIL_SHA, "rows": APRIL_ROWS, "used_for_training_or_scoring": False},
        },
        "software": software, "grid": GRID,
        "training": {
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "loss": "L1Loss",
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "min_delta": MIN_DELTA, "gradient_clipping": True,
            "max_grad_norm": MAX_GRAD_NORM, "dataloader_shuffle": False,
            "hard_fit_seconds": HARD_FIT_SECONDS, "outer_splits": N_SPLITS,
            "outer_gap": OUTER_GAP, "inner_validation_fraction": INNER_VAL_FRACTION,
            "inner_gap": INNER_GAP,
        },
        "selection": {
            "score": "mean_h(MAE_model_h / MAE_persistence_h)",
            "advance": "top-2 per family", "exact_tie_tolerance": 1e-12,
            "exact_tie_break": "config_id ascending",
        },
        "validation": {
            "fit_count_216": len(fit_runtime) == 216,
            "all_fit_status_pass": bool(fit_runtime["status"].eq("PASS").all()),
            "candidate_rows_18": len(candidate_summary) == 18,
            "top2_rows_6": int(candidate_summary["selected_top2"].sum()) == 6,
            "june_absent_after_run": not bool(find_forbidden_june(bundle_root)),
            "march_hash_unchanged": sha256(bundle_root/"inputs"/MARCH_NAME) == MARCH_SHA,
            "april_hash_unchanged": sha256(bundle_root/"inputs"/APRIL_NAME) == APRIL_SHA,
            "protocol_hash_unchanged": sha256(bundle_root/"governance"/PROTOCOL_NAME) == PROTOCOL_SHA,
        },
        "campaign_seconds": float(time.perf_counter()-started),
        "outputs": output_records,
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError("Final validation failed")
    atomic_json(manifest, paths["manifest"])

    zip_path = output_dir/"ugr16_recurrent_march_screen_outputs_v1.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output_dir.iterdir()):
            if p.is_file() and p != zip_path:
                zf.write(p, arcname=p.name)

    print(report)
    print("VALIDATIONS")
    for k, v in manifest["validation"].items():
        print(f"{k:48s}: {'PASS' if v else 'FAIL'}")
    print(f"Outputs ZIP: {zip_path}")
    print(f"Outputs ZIP SHA-256: {sha256(zip_path)}")
    print(f"{CAMPAIGN_ID}: PASS")
    print("RNN*/LSTM*/GRU*/RECURRENT*: NOT SELECTED IN MARCH")
    print("RECURRENT-APRIL-SELECTION-001: NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
