#!/usr/bin/env python3
"""
Análisis exploratorio reproducible de las series temporales UGR'16.

El script analiza las resoluciones preparadas de 1, 5 y 10 minutos sin
modificar los datos de entrada. La resolución de 5 minutos se utiliza como
serie principal; 1 minuto y 10 minutos se emplean como análisis de
sensibilidad.

Salidas:
- Informe TXT y manifiesto JSON.
- Tablas CSV de estadísticos, autocorrelación, perfiles temporales y picos.
- Figuras PNG listas para revisión y posterior integración en la memoria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_1MIN = Path(
    "data/processed/ugr16/march_week3_prepared_1min.parquet"
)
DEFAULT_INPUT_5MIN = Path(
    "data/processed/ugr16/march_week3_prepared_5min.parquet"
)
DEFAULT_INPUT_10MIN = Path(
    "data/processed/ugr16/march_week3_prepared_10min.parquet"
)
DEFAULT_FIGURES_DIR = Path("results/figures/ugr16_eda")
DEFAULT_METRICS_DIR = Path("results/metrics")
DEFAULT_PREFIX = "ugr16_march_week3_eda"
DEFAULT_CAPTURE_NAME = "UGR'16 March Week #3"

REQUIRED_COLUMNS = {
    "timestamp",
    "flows_total",
    "packets_total",
    "bytes_total",
    "bitrate_bps",
    "bytes_background",
    "bytes_blacklist",
    "bytes_anomaly",
    "flows_background",
    "flows_blacklist",
    "flows_anomaly",
    "tcp_flows",
    "udp_flows",
    "icmp_flows",
    "other_protocol_flows",
    "tcp_bytes",
    "udp_bytes",
    "icmp_bytes",
    "other_protocol_bytes",
}

DESCRIPTIVE_COLUMNS = [
    "flows_total",
    "packets_total",
    "bytes_total",
    "bitrate_bps",
    "flow_rate_fps",
    "packet_rate_pps",
    "mean_flow_duration",
    "max_flow_bytes",
    "max_flow_packets",
    "bytes_background",
    "bytes_blacklist",
    "bytes_anomaly",
]

CORRELATION_COLUMNS = [
    "bitrate_bps",
    "flow_rate_fps",
    "packet_rate_pps",
    "mean_flow_duration",
    "max_flow_bytes",
    "max_flow_packets",
    "tcp_bytes",
    "udp_bytes",
    "icmp_bytes",
    "bytes_blacklist",
    "bytes_anomaly",
]


@dataclass
class SeriesMetadata:
    frequency: str
    path: str
    sha256: str
    rows: int
    columns: int
    start: str
    end: str
    expected_delta_seconds: float
    duplicate_timestamps: int
    missing_intervals: int
    zero_flow_intervals: int
    total_flows: int
    total_packets: int
    total_bytes: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio reproducible de UGR'16."
    )

    parser.add_argument(
        "--input-1min",
        type=Path,
        default=DEFAULT_INPUT_1MIN,
        help="Serie preparada de 1 minuto.",
    )
    parser.add_argument(
        "--input-5min",
        type=Path,
        default=DEFAULT_INPUT_5MIN,
        help="Serie preparada de 5 minutos.",
    )
    parser.add_argument(
        "--input-10min",
        type=Path,
        default=DEFAULT_INPUT_10MIN,
        help="Serie preparada de 10 minutos.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directorio de salida para figuras.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directorio de salida para tablas, informe y manifiesto.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Prefijo común de los archivos generados.",
    )
    parser.add_argument(
        "--capture-name",
        type=str,
        default=DEFAULT_CAPTURE_NAME,
        help=(
            "Nombre legible de la captura para títulos, informe "
            "y manifiesto."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="bitrate_bps",
        help="Variable principal del análisis.",
    )
    parser.add_argument(
        "--acf-max-lags",
        type=int,
        default=288,
        help="Máximo número de retardos para la ACF de 5 minutos.",
    )
    parser.add_argument(
        "--top-peaks",
        type=int,
        default=20,
        help="Número máximo de picos robustos que se conservarán.",
    )
    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=6.0,
        help="Umbral del z-score robusto para candidatos a pico.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Resolución de las figuras PNG.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas existentes.",
    )

    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)

    return digest.hexdigest()


def frequency_to_seconds(frequency: str) -> float:
    return float(pd.Timedelta(frequency).total_seconds())


def atomic_write_text(text: str, output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output_path)


def atomic_write_json(payload: dict[str, Any], output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    os.replace(temporary, output_path)


def atomic_write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def save_figure(output_path: Path, dpi: int) -> None:
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    plt.tight_layout()
    plt.savefig(temporary, dpi=dpi, bbox_inches="tight")
    plt.close()
    os.replace(temporary, output_path)


def load_and_validate_series(
    path: Path,
    frequency: str,
) -> tuple[pd.DataFrame, SeriesMetadata]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra la serie: {path}")

    dataframe = pd.read_parquet(path)

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {path}: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    duplicated = int(dataframe["timestamp"].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"La serie {frequency} contiene {duplicated} timestamps duplicados."
        )

    if dataframe.empty:
        raise ValueError(f"La serie {frequency} está vacía.")

    expected_delta = pd.Timedelta(frequency)
    differences = dataframe["timestamp"].diff().dropna()
    invalid_differences = differences[~differences.eq(expected_delta)]

    if not invalid_differences.empty:
        raise ValueError(
            f"La serie {frequency} no es continua. "
            f"Se detectaron {len(invalid_differences)} saltos."
        )

    full_index = pd.date_range(
        start=dataframe["timestamp"].min(),
        end=dataframe["timestamp"].max(),
        freq=frequency,
    )
    missing_intervals = int(len(full_index) - len(dataframe))

    if missing_intervals != 0:
        raise ValueError(
            f"La serie {frequency} tiene {missing_intervals} intervalos ausentes."
        )

    metadata = SeriesMetadata(
        frequency=frequency,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        start=dataframe["timestamp"].min().isoformat(),
        end=dataframe["timestamp"].max().isoformat(),
        expected_delta_seconds=frequency_to_seconds(frequency),
        duplicate_timestamps=duplicated,
        missing_intervals=missing_intervals,
        zero_flow_intervals=int((dataframe["flows_total"] == 0).sum()),
        total_flows=int(dataframe["flows_total"].sum()),
        total_packets=int(dataframe["packets_total"].sum()),
        total_bytes=int(dataframe["bytes_total"].sum()),
    )

    return dataframe, metadata


def verify_common_totals(metadata: list[SeriesMetadata]) -> None:
    totals = {
        (
            item.total_flows,
            item.total_packets,
            item.total_bytes,
        )
        for item in metadata
    }

    if len(totals) != 1:
        raise ValueError(
            "Las resoluciones no contienen los mismos totales globales."
        )


def descriptive_statistics(
    series_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for frequency, dataframe in series_map.items():
        available_columns = [
            column
            for column in DESCRIPTIVE_COLUMNS
            if column in dataframe.columns
        ]

        description = (
            dataframe[available_columns]
            .describe(
                percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            )
            .transpose()
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        description.insert(0, "frequency", frequency)
        frames.append(description)

    return pd.concat(frames, ignore_index=True)


def build_label_share_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    total_flows = float(dataframe["flows_total"].sum())
    total_bytes = float(dataframe["bytes_total"].sum())

    rows = []

    for label in ["background", "blacklist", "anomaly"]:
        flows = int(dataframe[f"flows_{label}"].sum())
        bytes_value = int(dataframe[f"bytes_{label}"].sum())

        rows.append(
            {
                "category": label,
                "flows": flows,
                "flows_share_pct": (
                    100.0 * flows / total_flows if total_flows else math.nan
                ),
                "bytes": bytes_value,
                "bytes_share_pct": (
                    100.0 * bytes_value / total_bytes if total_bytes else math.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_protocol_share_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    total_flows = float(dataframe["flows_total"].sum())
    total_bytes = float(dataframe["bytes_total"].sum())

    rows = []

    mapping = {
        "TCP": ("tcp_flows", "tcp_bytes"),
        "UDP": ("udp_flows", "udp_bytes"),
        "ICMP": ("icmp_flows", "icmp_bytes"),
        "Other": ("other_protocol_flows", "other_protocol_bytes"),
    }

    for protocol, (flows_column, bytes_column) in mapping.items():
        flows = int(dataframe[flows_column].sum())
        bytes_value = int(dataframe[bytes_column].sum())

        rows.append(
            {
                "protocol": protocol,
                "flows": flows,
                "flows_share_pct": (
                    100.0 * flows / total_flows if total_flows else math.nan
                ),
                "bytes": bytes_value,
                "bytes_share_pct": (
                    100.0 * bytes_value / total_bytes if total_bytes else math.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_autocorrelation(
    values: pd.Series,
    max_lags: int,
) -> pd.DataFrame:
    clean = values.astype("float64").dropna().reset_index(drop=True)

    if len(clean) < 3:
        raise ValueError("No hay observaciones suficientes para calcular la ACF.")

    effective_max = min(max_lags, len(clean) - 2)
    rows = []

    for lag in range(effective_max + 1):
        autocorrelation = 1.0 if lag == 0 else clean.autocorr(lag=lag)
        rows.append(
            {
                "lag": lag,
                "autocorrelation": float(autocorrelation),
                "lag_minutes": lag * 5,
                "lag_hours": lag * 5 / 60.0,
            }
        )

    return pd.DataFrame(rows)


def compute_robust_peak_candidates(
    dataframe: pd.DataFrame,
    target: str,
    threshold: float,
    top_peaks: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    values = dataframe[target].astype("float64")
    median = float(values.median())
    mad = float((values - median).abs().median())

    if mad > 0:
        robust_z = 0.6744897501960817 * (values - median) / mad
        method = "MAD"
    else:
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        if iqr <= 0:
            robust_z = pd.Series(
                np.zeros(len(values), dtype="float64"),
                index=values.index,
            )
            method = "constant_series"
        else:
            robust_z = (values - median) / (iqr / 1.349)
            method = "IQR"

    result = dataframe[
        [
            "timestamp",
            target,
            "flows_total",
            "packets_total",
            "bytes_background",
            "bytes_blacklist",
            "bytes_anomaly",
        ]
    ].copy()

    result["robust_z"] = robust_z.astype("float64")
    result["blacklist_bytes_share_pct"] = (
        100.0
        * result["bytes_blacklist"]
        / dataframe["bytes_total"].replace(0, np.nan)
    )
    result["anomaly_bytes_share_pct"] = (
        100.0
        * result["bytes_anomaly"]
        / dataframe["bytes_total"].replace(0, np.nan)
    )

    candidates = (
        result.loc[result["robust_z"] >= threshold]
        .sort_values(["robust_z", target], ascending=False)
        .head(top_peaks)
        .reset_index(drop=True)
    )

    diagnostics = {
        "median": median,
        "mad": mad,
        "threshold": float(threshold),
        "method": method,
        "candidate_count_before_limit": int(
            (result["robust_z"] >= threshold).sum()
        ),
        "candidate_count_saved": int(len(candidates)),
    }

    return candidates, diagnostics


def build_hourly_profile(
    dataframe: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    prepared = dataframe[["timestamp", target]].copy()
    prepared["hour"] = prepared["timestamp"].dt.hour

    profile = (
        prepared.groupby("hour")[target]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    profile["coefficient_of_variation"] = (
        profile["std"] / profile["mean"].replace(0, np.nan)
    )
    return profile


def build_daily_summary(
    dataframe: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    prepared = dataframe[
        [
            "timestamp",
            target,
            "flows_total",
            "packets_total",
            "bytes_total",
            "bytes_background",
            "bytes_blacklist",
            "bytes_anomaly",
        ]
    ].copy()

    prepared["date"] = prepared["timestamp"].dt.date.astype(str)

    summary = (
        prepared.groupby("date")
        .agg(
            intervals=("timestamp", "count"),
            target_mean=(target, "mean"),
            target_median=(target, "median"),
            target_std=(target, "std"),
            target_max=(target, "max"),
            flows_total=("flows_total", "sum"),
            packets_total=("packets_total", "sum"),
            bytes_total=("bytes_total", "sum"),
            bytes_background=("bytes_background", "sum"),
            bytes_blacklist=("bytes_blacklist", "sum"),
            bytes_anomaly=("bytes_anomaly", "sum"),
        )
        .reset_index()
    )

    return summary


def build_correlation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in CORRELATION_COLUMNS
        if column in dataframe.columns
    ]

    matrix = dataframe[columns].corr(method="pearson")
    matrix.index.name = "variable"
    return matrix.reset_index()


def plot_bitrate_5min(
    dataframe: pd.DataFrame,
    output_path: Path,
    dpi: int,
    capture_name: str,
) -> None:
    seconds = 300.0
    background_bitrate = dataframe["bytes_background"] * 8.0 / seconds

    plt.figure(figsize=(13, 5.5))
    plt.plot(
        dataframe["timestamp"],
        dataframe["bitrate_bps"] / 1e6,
        label="Tráfico total",
        linewidth=1.0,
    )
    plt.plot(
        dataframe["timestamp"],
        background_bitrate / 1e6,
        label="Tráfico background",
        linewidth=1.0,
    )
    plt.xlabel("Tiempo")
    plt.ylabel("Tasa equivalente (Mbit/s)")
    plt.title(f"{capture_name} — tráfico agregado cada 5 minutos")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.gcf().autofmt_xdate()
    save_figure(output_path, dpi)


def plot_bitrate_1min(
    dataframe: pd.DataFrame,
    output_path: Path,
    dpi: int,
    capture_name: str,
) -> None:
    plt.figure(figsize=(13, 5.5))
    plt.plot(
        dataframe["timestamp"],
        dataframe["bitrate_bps"] / 1e6,
        linewidth=0.8,
    )
    plt.xlabel("Tiempo")
    plt.ylabel("Tasa equivalente (Mbit/s)")
    plt.title(f"{capture_name} — tráfico agregado cada minuto")
    plt.grid(True, alpha=0.25)
    plt.gcf().autofmt_xdate()
    save_figure(output_path, dpi)


def plot_label_shares(
    table: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    positions = np.arange(len(table))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    plt.bar(
        positions - width / 2,
        table["flows_share_pct"],
        width,
        label="Porcentaje de flujos",
    )
    plt.bar(
        positions + width / 2,
        table["bytes_share_pct"],
        width,
        label="Porcentaje de bytes",
    )
    plt.xticks(positions, table["category"])
    plt.ylabel("Porcentaje (%)")
    plt.title("Distribución del tráfico por etiqueta — resolución de 5 minutos")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(output_path, dpi)


def plot_protocol_shares(
    table: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    positions = np.arange(len(table))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    plt.bar(
        positions - width / 2,
        table["flows_share_pct"],
        width,
        label="Porcentaje de flujos",
    )
    plt.bar(
        positions + width / 2,
        table["bytes_share_pct"],
        width,
        label="Porcentaje de bytes",
    )
    plt.xticks(positions, table["protocol"])
    plt.ylabel("Porcentaje (%)")
    plt.title("Distribución del tráfico por protocolo — resolución de 5 minutos")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save_figure(output_path, dpi)


def plot_autocorrelation(
    table: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.vlines(
        table["lag"],
        0.0,
        table["autocorrelation"],
        linewidth=0.8,
    )
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Retardo (intervalos de 5 minutos)")
    plt.ylabel("Autocorrelación")
    plt.title("Autocorrelación del tráfico agregado a 5 minutos")
    plt.grid(True, alpha=0.25)
    save_figure(output_path, dpi)


def plot_hourly_profile(
    table: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        table["hour"],
        table["mean"] / 1e6,
        marker="o",
        label="Media",
    )
    plt.plot(
        table["hour"],
        table["median"] / 1e6,
        marker="o",
        label="Mediana",
    )
    plt.xticks(range(24))
    plt.xlabel("Hora del día")
    plt.ylabel("Tasa equivalente (Mbit/s)")
    plt.title("Perfil horario exploratorio — resolución de 5 minutos")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(output_path, dpi)


def plot_peak_candidates(
    dataframe: pd.DataFrame,
    peaks: pd.DataFrame,
    target: str,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(13, 5.5))
    plt.plot(
        dataframe["timestamp"],
        dataframe[target] / 1e6,
        linewidth=0.9,
        label="Serie de 5 minutos",
    )

    if not peaks.empty:
        plt.scatter(
            peaks["timestamp"],
            peaks[target] / 1e6,
            s=35,
            label="Candidatos a pico",
            zorder=3,
        )

    plt.xlabel("Tiempo")
    plt.ylabel("Tasa equivalente (Mbit/s)")
    plt.title("Candidatos a pico mediante z-score robusto")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.gcf().autofmt_xdate()
    save_figure(output_path, dpi)


def create_output_paths(
    figures_dir: Path,
    metrics_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    return {
        "descriptive_csv": metrics_dir / f"{prefix}_descriptive_statistics.csv",
        "label_shares_csv": metrics_dir / f"{prefix}_label_shares_5min.csv",
        "protocol_shares_csv": metrics_dir / f"{prefix}_protocol_shares_5min.csv",
        "acf_csv": metrics_dir / f"{prefix}_autocorrelation_5min.csv",
        "peaks_csv": metrics_dir / f"{prefix}_peak_candidates_5min.csv",
        "hourly_profile_csv": metrics_dir / f"{prefix}_hourly_profile_5min.csv",
        "daily_summary_csv": metrics_dir / f"{prefix}_daily_summary_5min.csv",
        "correlation_csv": metrics_dir / f"{prefix}_correlation_5min.csv",
        "manifest_json": metrics_dir / f"{prefix}_manifest.json",
        "report_txt": metrics_dir / f"{prefix}_report.txt",
        "figure_01": figures_dir / f"{prefix}_01_bitrate_5min.png",
        "figure_02": figures_dir / f"{prefix}_02_bitrate_1min.png",
        "figure_03": figures_dir / f"{prefix}_03_label_shares_5min.png",
        "figure_04": figures_dir / f"{prefix}_04_protocol_shares_5min.png",
        "figure_05": figures_dir / f"{prefix}_05_autocorrelation_5min.png",
        "figure_06": figures_dir / f"{prefix}_06_hourly_profile_5min.png",
        "figure_07": figures_dir / f"{prefix}_07_peak_candidates_5min.png",
    }


def validate_outputs(
    output_paths: dict[str, Path],
    overwrite: bool,
) -> None:
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = [
            str(path)
            for path in output_paths.values()
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite o cambia el prefijo:\n"
                + "\n".join(existing)
            )


def selected_acf_values(acf_table: pd.DataFrame) -> dict[str, float | None]:
    selected = {
        "lag_1_5min": 1,
        "lag_12_1h": 12,
        "lag_72_6h": 72,
        "lag_144_12h": 144,
        "lag_288_24h": 288,
    }

    result: dict[str, float | None] = {}

    for name, lag in selected.items():
        row = acf_table.loc[acf_table["lag"] == lag]
        result[name] = (
            float(row["autocorrelation"].iloc[0])
            if not row.empty
            else None
        )

    return result


def build_report(
    metadata: list[SeriesMetadata],
    capture_name: str,
    target: str,
    label_shares: pd.DataFrame,
    protocol_shares: pd.DataFrame,
    acf_selected: dict[str, float | None],
    peak_diagnostics: dict[str, Any],
    peaks: pd.DataFrame,
    output_paths: dict[str, Path],
) -> str:
    metadata_lines = []

    for item in metadata:
        metadata_lines.extend(
            [
                f"{item.frequency:>5}: {item.rows:,} filas | "
                f"{item.start} → {item.end}",
                f"      SHA-256: {item.sha256}",
            ]
        )

    label_lines = [
        (
            f"{row.category:>10}: "
            f"{int(row.flows):,} flujos "
            f"({row.flows_share_pct:.6f} %) | "
            f"{int(row.bytes):,} bytes "
            f"({row.bytes_share_pct:.6f} %)"
        )
        for row in label_shares.itertuples(index=False)
    ]

    protocol_lines = [
        (
            f"{row.protocol:>5}: "
            f"{int(row.flows):,} flujos "
            f"({row.flows_share_pct:.6f} %) | "
            f"{int(row.bytes):,} bytes "
            f"({row.bytes_share_pct:.6f} %)"
        )
        for row in protocol_shares.itertuples(index=False)
    ]

    acf_lines = [
        f"{name}: {value:.6f}" if value is not None else f"{name}: N/D"
        for name, value in acf_selected.items()
    ]

    output_lines = [
        f"{name}: {path.resolve()}"
        for name, path in output_paths.items()
    ]

    primary_metadata = next(
        item for item in metadata if item.frequency == "5min"
    )
    coverage_hours = (
        primary_metadata.rows
        * primary_metadata.expected_delta_seconds
        / 3600.0
    )
    coverage_days = coverage_hours / 24.0

    if coverage_hours < 72.0:
        coverage_note = (
            f"- La captura cubre {coverage_hours:.2f} horas. "
            "El perfil horario y la autocorrelación estacional son "
            "exploratorios y no constituyen evidencia concluyente de "
            "estacionalidad diaria o semanal."
        )
    elif coverage_days < 14.0:
        coverage_note = (
            f"- La captura cubre {coverage_hours:.2f} horas "
            f"({coverage_days:.2f} días). Permite describir el perfil "
            "intradía con mayor estabilidad que una captura corta, pero "
            "una única semana no basta para concluir estacionalidad "
            "semanal ni estabilidad entre semanas."
        )
    else:
        coverage_note = (
            f"- La captura cubre {coverage_days:.2f} días. "
            "Los patrones temporales deben confirmarse mediante "
            "validación fuera de muestra y comparación entre segmentos."
        )

    lines = [
        "=" * 78,
        "ANÁLISIS EXPLORATORIO DE SERIES TEMPORALES UGR'16",
        "=" * 78,
        f"Captura: {capture_name}",
        f"Variable principal: {target}",
        "",
        "SERIES VALIDADAS",
        "-" * 78,
        *metadata_lines,
        "",
        "DISTRIBUCIÓN POR ETIQUETAS — 5 MINUTOS",
        "-" * 78,
        *label_lines,
        "",
        "DISTRIBUCIÓN POR PROTOCOLOS — 5 MINUTOS",
        "-" * 78,
        *protocol_lines,
        "",
        "AUTOCORRELACIÓN SELECCIONADA — 5 MINUTOS",
        "-" * 78,
        *acf_lines,
        "",
        "CANDIDATOS A PICO",
        "-" * 78,
        f"Método de escala robusta: {peak_diagnostics['method']}",
        f"Mediana de {target}: {peak_diagnostics['median']:.6f}",
        f"MAD de {target}: {peak_diagnostics['mad']:.6f}",
        f"Umbral z robusto: {peak_diagnostics['threshold']:.3f}",
        (
            "Candidatos antes del límite: "
            f"{peak_diagnostics['candidate_count_before_limit']:,}"
        ),
        f"Candidatos guardados: {len(peaks):,}",
        "",
        "LIMITACIONES DE INTERPRETACIÓN",
        "-" * 78,
        coverage_note,
        (
            "- Los candidatos a pico no se eliminan ni se clasifican como "
            "anomalías automáticamente; solo se marcan para inspección."
        ),
        (
            "- bitrate_bps es una tasa equivalente calculada con los bytes "
            "atribuidos al timestamp del flujo, no una lectura instantánea "
            "de una interfaz física."
        ),
        (
            "- Las columnas *_anomaly contienen las etiquetas agrupadas "
            "como anomalía durante la preparación de la serie."
        ),
        "",
        "SALIDAS",
        "-" * 78,
        *output_lines,
        "",
        "Análisis exploratorio terminado correctamente.",
    ]

    return "\n".join(lines) + "\n"


def main() -> int:
    arguments = parse_arguments()

    try:
        if arguments.acf_max_lags <= 0:
            raise ValueError("--acf-max-lags debe ser mayor que cero.")
        if arguments.top_peaks <= 0:
            raise ValueError("--top-peaks debe ser mayor que cero.")
        if arguments.robust_z_threshold <= 0:
            raise ValueError("--robust-z-threshold debe ser mayor que cero.")
        if arguments.dpi <= 0:
            raise ValueError("--dpi debe ser mayor que cero.")
        arguments.capture_name = arguments.capture_name.strip()
        if not arguments.capture_name:
            raise ValueError("--capture-name no puede estar vacío.")

        output_paths = create_output_paths(
            arguments.figures_dir,
            arguments.metrics_dir,
            arguments.prefix,
        )
        validate_outputs(output_paths, arguments.overwrite)

        series_map: dict[str, pd.DataFrame] = {}
        metadata: list[SeriesMetadata] = []

        for frequency, path in [
            ("1min", arguments.input_1min),
            ("5min", arguments.input_5min),
            ("10min", arguments.input_10min),
        ]:
            dataframe, item_metadata = load_and_validate_series(
                path,
                frequency,
            )
            series_map[frequency] = dataframe
            metadata.append(item_metadata)

        verify_common_totals(metadata)

        for frequency, dataframe in series_map.items():
            if arguments.target not in dataframe.columns:
                raise ValueError(
                    f"La variable {arguments.target!r} no existe "
                    f"en la serie {frequency}."
                )

        descriptive = descriptive_statistics(series_map)
        label_shares = build_label_share_table(series_map["5min"])
        protocol_shares = build_protocol_share_table(series_map["5min"])
        acf_table = compute_autocorrelation(
            series_map["5min"][arguments.target],
            arguments.acf_max_lags,
        )
        peaks, peak_diagnostics = compute_robust_peak_candidates(
            series_map["5min"],
            arguments.target,
            arguments.robust_z_threshold,
            arguments.top_peaks,
        )
        hourly_profile = build_hourly_profile(
            series_map["5min"],
            arguments.target,
        )
        daily_summary = build_daily_summary(
            series_map["5min"],
            arguments.target,
        )
        correlation = build_correlation_table(series_map["5min"])

        atomic_write_csv(descriptive, output_paths["descriptive_csv"])
        atomic_write_csv(label_shares, output_paths["label_shares_csv"])
        atomic_write_csv(
            protocol_shares,
            output_paths["protocol_shares_csv"],
        )
        atomic_write_csv(acf_table, output_paths["acf_csv"])
        atomic_write_csv(peaks, output_paths["peaks_csv"])
        atomic_write_csv(
            hourly_profile,
            output_paths["hourly_profile_csv"],
        )
        atomic_write_csv(
            daily_summary,
            output_paths["daily_summary_csv"],
        )
        atomic_write_csv(correlation, output_paths["correlation_csv"])

        plot_bitrate_5min(
            series_map["5min"],
            output_paths["figure_01"],
            arguments.dpi,
            arguments.capture_name,
        )
        plot_bitrate_1min(
            series_map["1min"],
            output_paths["figure_02"],
            arguments.dpi,
            arguments.capture_name,
        )
        plot_label_shares(
            label_shares,
            output_paths["figure_03"],
            arguments.dpi,
        )
        plot_protocol_shares(
            protocol_shares,
            output_paths["figure_04"],
            arguments.dpi,
        )
        plot_autocorrelation(
            acf_table,
            output_paths["figure_05"],
            arguments.dpi,
        )
        plot_hourly_profile(
            hourly_profile,
            output_paths["figure_06"],
            arguments.dpi,
        )
        plot_peak_candidates(
            series_map["5min"],
            peaks,
            arguments.target,
            output_paths["figure_07"],
            arguments.dpi,
        )

        acf_selected = selected_acf_values(acf_table)

        manifest = {
            "inputs": [asdict(item) for item in metadata],
            "analysis": {
                "capture_name": arguments.capture_name,
                "primary_frequency": "5min",
                "secondary_frequencies": ["1min", "10min"],
                "target": arguments.target,
                "acf_max_lags": int(arguments.acf_max_lags),
                "robust_z_threshold": float(
                    arguments.robust_z_threshold
                ),
                "top_peaks": int(arguments.top_peaks),
                "coverage_hours": float(
                    len(series_map["5min"]) * 5.0 / 60.0
                ),
                "acf_selected": acf_selected,
                "peak_diagnostics": peak_diagnostics,
            },
            "outputs": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in output_paths.items()
                if name not in {"manifest_json", "report_txt"}
                and path.exists()
            },
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
            },
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        report = build_report(
            metadata=metadata,
            capture_name=arguments.capture_name,
            target=arguments.target,
            label_shares=label_shares,
            protocol_shares=protocol_shares,
            acf_selected=acf_selected,
            peak_diagnostics=peak_diagnostics,
            peaks=peaks,
            output_paths=output_paths,
        )

        atomic_write_text(report, output_paths["report_txt"])

        manifest["outputs"]["report_txt"] = {
            "path": str(output_paths["report_txt"].resolve()),
            "sha256": sha256_file(output_paths["report_txt"]),
        }

        atomic_write_json(manifest, output_paths["manifest_json"])

        print(report)
        print(
            f"Manifiesto: {output_paths['manifest_json'].resolve()}"
        )

    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
