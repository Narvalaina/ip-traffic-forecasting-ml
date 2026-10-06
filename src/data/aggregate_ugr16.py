#!/usr/bin/env python3
"""
Agregación reproducible del dataset UGR'16 desde un TAR.GZ.

El script procesa el CSV interno por streaming y por bloques, sin extraer
el fichero completo al disco ni cargarlo entero en memoria. Genera una serie
temporal agregada con resolución configurable (1 minuto por defecto).

Entradas predeterminadas
------------------------
Archivo:
    data/raw/ugr16/calibration/march_week3_csv.tar.gz

Miembro interno:
    uniq/march.week3.csv.uniqblacklistremoved

Salidas predeterminadas
-----------------------
Serie Parquet:
    data/interim/ugr16/march_week3_1min.parquet

Serie CSV:
    data/interim/ugr16/march_week3_1min.csv

Resumen JSON:
    results/metrics/ugr16_march_week3_aggregation_summary.json

Informe de texto:
    results/metrics/ugr16_march_week3_aggregation_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


ALL_COLUMNS = [
    "timestamp",
    "duration",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flags",
    "forwarding_status",
    "src_tos",
    "packets",
    "bytes",
    "label",
]

USE_COLUMNS = [
    "timestamp",
    "duration",
    "protocol",
    "packets",
    "bytes",
    "label",
]

DEFAULT_ARCHIVE = Path(
    "data/raw/ugr16/calibration/march_week3_csv.tar.gz"
)
DEFAULT_MEMBER = "uniq/march.week3.csv.uniqblacklistremoved"
DEFAULT_PARQUET = Path(
    "data/interim/ugr16/march_week3_1min.parquet"
)
DEFAULT_CSV = Path(
    "data/interim/ugr16/march_week3_1min.csv"
)
DEFAULT_SUMMARY = Path(
    "results/metrics/ugr16_march_week3_aggregation_summary.json"
)
DEFAULT_REPORT = Path(
    "results/metrics/ugr16_march_week3_aggregation_report.txt"
)


@dataclass
class ProcessingSummary:
    archive: str
    member: str
    frequency: str
    chunk_size: int
    max_rows: int | None
    rows_read: int
    rows_valid: int
    rows_discarded: int
    invalid_timestamps: int
    invalid_packets: int
    invalid_bytes: int
    invalid_durations: int
    negative_packets: int
    negative_bytes: int
    negative_durations: int
    first_timestamp: str | None
    last_timestamp: str | None
    intervals_generated: int
    empty_intervals_inserted: int
    total_flows: int
    total_packets: int
    total_bytes: int
    label_counts: dict[str, int]
    protocol_counts: dict[str, int]
    processing_seconds: float
    rows_per_second: float
    output_parquet: str
    output_csv: str
    completed_utc: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agrega UGR'16 por streaming, sin extraer el CSV completo."
        )
    )

    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Ruta al archivo TAR.GZ.",
    )
    parser.add_argument(
        "--member",
        type=str,
        default=DEFAULT_MEMBER,
        help="Nombre del CSV dentro del TAR.GZ.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="1min",
        help="Frecuencia de agregación de pandas. Valor recomendado: 1min.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Número de filas por bloque. Valor conservador: 250000.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Máximo total de filas para una ejecución de prueba. "
            "Por defecto procesa todo el archivo."
        ),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Ruta del Parquet agregado.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta del CSV agregado.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Ruta del resumen JSON.",
    )
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=DEFAULT_REPORT,
        help="Ruta del informe de texto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir salidas existentes.",
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.archive.is_file():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {arguments.archive}"
        )

    if arguments.chunksize <= 0:
        raise ValueError("--chunksize debe ser mayor que cero.")

    if arguments.max_rows is not None and arguments.max_rows <= 0:
        raise ValueError("--max-rows debe ser mayor que cero.")

    try:
        seconds = pd.Timedelta(arguments.frequency).total_seconds()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Frecuencia no válida: {arguments.frequency}"
        ) from error

    if seconds <= 0:
        raise ValueError("--frequency debe representar un intervalo positivo.")

    output_paths = [
        arguments.output_parquet,
        arguments.output_csv,
        arguments.summary_json,
        arguments.report_txt,
    ]

    if not arguments.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            formatted = "\n".join(existing)
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite o cambia las rutas:\n"
                f"{formatted}"
            )


def ensure_parent_directories(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def normalize_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items())
    }


def create_empty_aggregate() -> pd.DataFrame:
    columns = [
        "flows_total",
        "packets_total",
        "bytes_total",
        "duration_sum",
        "max_flow_bytes",
        "max_flow_packets",
        "flows_background",
        "packets_background",
        "bytes_background",
        "flows_blacklist",
        "packets_blacklist",
        "bytes_blacklist",
        "flows_other_label",
        "packets_other_label",
        "bytes_other_label",
        "tcp_flows",
        "tcp_packets",
        "tcp_bytes",
        "udp_flows",
        "udp_packets",
        "udp_bytes",
        "icmp_flows",
        "icmp_packets",
        "icmp_bytes",
        "other_protocol_flows",
        "other_protocol_packets",
        "other_protocol_bytes",
    ]

    dataframe = pd.DataFrame(columns=columns)
    dataframe.index = pd.DatetimeIndex([], name="timestamp")
    return dataframe


def read_chunks(
    archive_path: Path,
    member_name: str,
    chunk_size: int,
    max_rows: int | None,
):
    tar_file = tarfile.open(archive_path, mode="r:gz")

    try:
        try:
            member = tar_file.getmember(member_name)
        except KeyError as error:
            available = "\n".join(tar_file.getnames())
            raise KeyError(
                "No se encuentra el miembro solicitado.\n"
                f"Solicitado: {member_name}\n"
                f"Disponibles:\n{available}"
            ) from error

        member_stream = tar_file.extractfile(member)

        if member_stream is None:
            raise RuntimeError(
                f"No se pudo abrir el miembro interno: {member_name}"
            )

        reader = pd.read_csv(
            member_stream,
            header=None,
            names=ALL_COLUMNS,
            usecols=USE_COLUMNS,
            dtype={
                "timestamp": "string",
                "duration": "string",
                "protocol": "string",
                "packets": "string",
                "bytes": "string",
                "label": "string",
            },
            chunksize=chunk_size,
            nrows=max_rows,
            skip_blank_lines=True,
            on_bad_lines="warn",
            low_memory=False,
        )

        yield from reader

    finally:
        tar_file.close()


def clean_chunk(
    chunk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "rows_read": int(len(chunk)),
        "invalid_timestamps": 0,
        "invalid_packets": 0,
        "invalid_bytes": 0,
        "invalid_durations": 0,
        "negative_packets": 0,
        "negative_bytes": 0,
        "negative_durations": 0,
        "rows_discarded": 0,
    }

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    chunk["duration"] = pd.to_numeric(
        chunk["duration"],
        errors="coerce",
    )
    chunk["packets"] = pd.to_numeric(
        chunk["packets"],
        errors="coerce",
    )
    chunk["bytes"] = pd.to_numeric(
        chunk["bytes"],
        errors="coerce",
    )

    chunk["protocol"] = (
        chunk["protocol"]
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )
    chunk["label"] = (
        chunk["label"]
        .fillna("unknown")
        .str.strip()
        .str.lower()
    )

    stats["invalid_timestamps"] = int(
        chunk["timestamp"].isna().sum()
    )
    stats["invalid_packets"] = int(
        chunk["packets"].isna().sum()
    )
    stats["invalid_bytes"] = int(
        chunk["bytes"].isna().sum()
    )
    stats["invalid_durations"] = int(
        chunk["duration"].isna().sum()
    )

    stats["negative_packets"] = int(
        (chunk["packets"] < 0).fillna(False).sum()
    )
    stats["negative_bytes"] = int(
        (chunk["bytes"] < 0).fillna(False).sum()
    )
    stats["negative_durations"] = int(
        (chunk["duration"] < 0).fillna(False).sum()
    )

    valid_mask = (
        chunk["timestamp"].notna()
        & chunk["packets"].notna()
        & chunk["bytes"].notna()
        & (chunk["packets"] >= 0)
        & (chunk["bytes"] >= 0)
    )

    cleaned = chunk.loc[valid_mask].copy()
    stats["rows_discarded"] = int(len(chunk) - len(cleaned))

    cleaned["duration"] = (
        cleaned["duration"]
        .fillna(0.0)
        .clip(lower=0.0)
        .astype("float64")
    )
    cleaned["packets"] = cleaned["packets"].astype("int64")
    cleaned["bytes"] = cleaned["bytes"].astype("int64")

    return cleaned, stats


def add_indicator_columns(
    chunk: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    chunk["time_bucket"] = chunk["timestamp"].dt.floor(frequency)
    chunk["flows_total"] = 1

    background = chunk["label"].eq("background")
    blacklist = chunk["label"].eq("blacklist")
    other_label = ~(background | blacklist)

    tcp = chunk["protocol"].eq("TCP")
    udp = chunk["protocol"].eq("UDP")
    icmp = chunk["protocol"].eq("ICMP")
    other_protocol = ~(tcp | udp | icmp)

    label_masks = {
        "background": background,
        "blacklist": blacklist,
        "other_label": other_label,
    }

    protocol_masks = {
        "tcp": tcp,
        "udp": udp,
        "icmp": icmp,
        "other_protocol": other_protocol,
    }

    for prefix, mask in label_masks.items():
        chunk[f"flows_{prefix}"] = mask.astype("int64")
        chunk[f"packets_{prefix}"] = (
            chunk["packets"].where(mask, 0).astype("int64")
        )
        chunk[f"bytes_{prefix}"] = (
            chunk["bytes"].where(mask, 0).astype("int64")
        )

    for prefix, mask in protocol_masks.items():
        chunk[f"{prefix}_flows"] = mask.astype("int64")
        chunk[f"{prefix}_packets"] = (
            chunk["packets"].where(mask, 0).astype("int64")
        )
        chunk[f"{prefix}_bytes"] = (
            chunk["bytes"].where(mask, 0).astype("int64")
        )

    return chunk


def aggregate_chunk(
    chunk: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    prepared = add_indicator_columns(chunk, frequency)

    sum_columns = [
        "flows_total",
        "packets",
        "bytes",
        "duration",
        "flows_background",
        "packets_background",
        "bytes_background",
        "flows_blacklist",
        "packets_blacklist",
        "bytes_blacklist",
        "flows_other_label",
        "packets_other_label",
        "bytes_other_label",
        "tcp_flows",
        "tcp_packets",
        "tcp_bytes",
        "udp_flows",
        "udp_packets",
        "udp_bytes",
        "icmp_flows",
        "icmp_packets",
        "icmp_bytes",
        "other_protocol_flows",
        "other_protocol_packets",
        "other_protocol_bytes",
    ]

    grouped_sum = (
        prepared.groupby("time_bucket", sort=True)[sum_columns]
        .sum()
        .rename(
            columns={
                "packets": "packets_total",
                "bytes": "bytes_total",
                "duration": "duration_sum",
            }
        )
    )

    grouped_max = (
        prepared.groupby("time_bucket", sort=True)[
            ["bytes", "packets"]
        ]
        .max()
        .rename(
            columns={
                "bytes": "max_flow_bytes",
                "packets": "max_flow_packets",
            }
        )
    )

    aggregated = grouped_sum.join(grouped_max, how="outer")
    aggregated.index.name = "timestamp"

    expected_order = create_empty_aggregate().columns
    return aggregated.reindex(columns=expected_order)


def combine_aggregates(
    accumulated: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    if accumulated.empty:
        return current.copy()

    sum_columns = [
        column
        for column in accumulated.columns
        if column not in {"max_flow_bytes", "max_flow_packets"}
    ]
    max_columns = ["max_flow_bytes", "max_flow_packets"]

    combined_sum = accumulated[sum_columns].add(
        current[sum_columns],
        fill_value=0,
    )

    combined_max = pd.concat(
        [
            accumulated[max_columns],
            current[max_columns],
        ],
        axis=0,
    ).groupby(level=0).max()

    combined = combined_sum.join(combined_max, how="outer")
    combined.index.name = "timestamp"
    return combined.sort_index()


def finalize_series(
    aggregate: pd.DataFrame,
    frequency: str,
) -> tuple[pd.DataFrame, int]:
    if aggregate.empty:
        raise RuntimeError(
            "No se generaron intervalos. Revisa los datos de entrada."
        )

    full_index = pd.date_range(
        start=aggregate.index.min(),
        end=aggregate.index.max(),
        freq=frequency,
        name="timestamp",
    )

    original_intervals = len(aggregate)
    aggregate = aggregate.reindex(full_index, fill_value=0)
    empty_intervals = len(aggregate) - original_intervals

    integer_columns = [
        column
        for column in aggregate.columns
        if column != "duration_sum"
    ]

    for column in integer_columns:
        aggregate[column] = (
            aggregate[column]
            .fillna(0)
            .round()
            .astype("int64")
        )

    aggregate["duration_sum"] = (
        aggregate["duration_sum"]
        .fillna(0.0)
        .astype("float64")
    )

    aggregate["mean_flow_duration"] = (
        aggregate["duration_sum"]
        .div(aggregate["flows_total"].replace(0, pd.NA))
        .fillna(0.0)
        .astype("float64")
    )

    interval_seconds = pd.Timedelta(frequency).total_seconds()

    aggregate["bitrate_bps"] = (
        aggregate["bytes_total"] * 8.0 / interval_seconds
    )
    aggregate["packet_rate_pps"] = (
        aggregate["packets_total"] / interval_seconds
    )
    aggregate["flow_rate_fps"] = (
        aggregate["flows_total"] / interval_seconds
    )

    output_order = [
        "flows_total",
        "packets_total",
        "bytes_total",
        "bitrate_bps",
        "packet_rate_pps",
        "flow_rate_fps",
        "mean_flow_duration",
        "max_flow_bytes",
        "max_flow_packets",
        "flows_background",
        "packets_background",
        "bytes_background",
        "flows_blacklist",
        "packets_blacklist",
        "bytes_blacklist",
        "flows_other_label",
        "packets_other_label",
        "bytes_other_label",
        "tcp_flows",
        "tcp_packets",
        "tcp_bytes",
        "udp_flows",
        "udp_packets",
        "udp_bytes",
        "icmp_flows",
        "icmp_packets",
        "icmp_bytes",
        "other_protocol_flows",
        "other_protocol_packets",
        "other_protocol_bytes",
        "duration_sum",
    ]

    return aggregate[output_order], empty_intervals


def validate_reconciliation(
    aggregate: pd.DataFrame,
    valid_rows: int,
) -> None:
    total_flows = int(aggregate["flows_total"].sum())

    if total_flows != valid_rows:
        raise RuntimeError(
            "Fallo de reconciliación: flows_total no coincide con "
            f"las filas válidas ({total_flows} != {valid_rows})."
        )

    label_flows = int(
        aggregate[
            [
                "flows_background",
                "flows_blacklist",
                "flows_other_label",
            ]
        ]
        .sum()
        .sum()
    )

    if label_flows != total_flows:
        raise RuntimeError(
            "Fallo de reconciliación en las etiquetas: "
            f"{label_flows} != {total_flows}."
        )

    protocol_flows = int(
        aggregate[
            [
                "tcp_flows",
                "udp_flows",
                "icmp_flows",
                "other_protocol_flows",
            ]
        ]
        .sum()
        .sum()
    )

    if protocol_flows != total_flows:
        raise RuntimeError(
            "Fallo de reconciliación en los protocolos: "
            f"{protocol_flows} != {total_flows}."
        )


def write_atomic_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    temporary = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    dataframe.reset_index().to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
    )
    os.replace(temporary, output_path)


def write_atomic_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    temporary = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    dataframe.reset_index().to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    os.replace(temporary, output_path)


def write_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    temporary = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    os.replace(temporary, output_path)


def build_report(
    summary: ProcessingSummary,
    aggregate: pd.DataFrame,
) -> str:
    label_lines = "\n".join(
        f"  - {label}: {count:,}"
        for label, count in summary.label_counts.items()
    )
    protocol_lines = "\n".join(
        f"  - {protocol}: {count:,}"
        for protocol, count in summary.protocol_counts.items()
    )

    lines = [
        "=" * 76,
        "AGREGACIÓN DEL DATASET UGR'16",
        "=" * 76,
        f"Archivo:                    {summary.archive}",
        f"Miembro interno:            {summary.member}",
        f"Frecuencia:                 {summary.frequency}",
        f"Tamaño de bloque:           {summary.chunk_size:,}",
        f"Límite de filas:            {summary.max_rows}",
        "",
        "PROCESAMIENTO",
        "-" * 76,
        f"Filas leídas:               {summary.rows_read:,}",
        f"Filas válidas:              {summary.rows_valid:,}",
        f"Filas descartadas:          {summary.rows_discarded:,}",
        f"Timestamps inválidos:       {summary.invalid_timestamps:,}",
        f"Paquetes inválidos:         {summary.invalid_packets:,}",
        f"Bytes inválidos:            {summary.invalid_bytes:,}",
        f"Duraciones inválidas:       {summary.invalid_durations:,}",
        f"Paquetes negativos:         {summary.negative_packets:,}",
        f"Bytes negativos:            {summary.negative_bytes:,}",
        f"Duraciones negativas:       {summary.negative_durations:,}",
        "",
        "SERIE TEMPORAL",
        "-" * 76,
        f"Inicio:                     {summary.first_timestamp}",
        f"Fin:                        {summary.last_timestamp}",
        f"Intervalos generados:       {summary.intervals_generated:,}",
        f"Intervalos vacíos insertados:{summary.empty_intervals_inserted:,}",
        f"Flujos totales:             {summary.total_flows:,}",
        f"Paquetes totales:           {summary.total_packets:,}",
        f"Bytes totales:              {summary.total_bytes:,}",
        "",
        "ETIQUETAS",
        "-" * 76,
        label_lines or "  Sin datos",
        "",
        "PROTOCOLOS",
        "-" * 76,
        protocol_lines or "  Sin datos",
        "",
        "RENDIMIENTO",
        "-" * 76,
        f"Tiempo de procesamiento:    {summary.processing_seconds:.2f} s",
        f"Rendimiento:                {summary.rows_per_second:,.2f} filas/s",
        "",
        "SALIDAS",
        "-" * 76,
        f"Parquet:                    {summary.output_parquet}",
        f"CSV:                        {summary.output_csv}",
        "",
        "PRIMEROS CINCO INTERVALOS",
        "-" * 76,
        aggregate.head(5).reset_index().to_string(index=False),
        "",
        "ÚLTIMOS CINCO INTERVALOS",
        "-" * 76,
        aggregate.tail(5).reset_index().to_string(index=False),
        "",
        "ESTADÍSTICAS PRINCIPALES",
        "-" * 76,
        aggregate[
            [
                "flows_total",
                "packets_total",
                "bytes_total",
                "bitrate_bps",
            ]
        ]
        .describe(percentiles=[0.50, 0.90, 0.99])
        .transpose()
        .to_string(),
        "",
        "Agregación terminada correctamente.",
    ]

    return "\n".join(lines) + "\n"


def main() -> int:
    arguments = parse_arguments()

    try:
        validate_arguments(arguments)

        output_paths = [
            arguments.output_parquet,
            arguments.output_csv,
            arguments.summary_json,
            arguments.report_txt,
        ]
        ensure_parent_directories(output_paths)

        accumulated = create_empty_aggregate()
        label_counts: Counter[str] = Counter()
        protocol_counts: Counter[str] = Counter()

        totals = Counter(
            {
                "rows_read": 0,
                "rows_valid": 0,
                "rows_discarded": 0,
                "invalid_timestamps": 0,
                "invalid_packets": 0,
                "invalid_bytes": 0,
                "invalid_durations": 0,
                "negative_packets": 0,
                "negative_bytes": 0,
                "negative_durations": 0,
            }
        )

        start_time = time.perf_counter()

        chunks = read_chunks(
            archive_path=arguments.archive,
            member_name=arguments.member,
            chunk_size=arguments.chunksize,
            max_rows=arguments.max_rows,
        )

        progress = tqdm(
            chunks,
            desc="Procesando UGR'16",
            unit="bloque",
            dynamic_ncols=True,
        )

        for chunk in progress:
            cleaned, cleaning_stats = clean_chunk(chunk)

            totals.update(cleaning_stats)
            totals["rows_valid"] += int(len(cleaned))

            if cleaned.empty:
                continue

            label_counts.update(
                cleaned["label"]
                .value_counts(dropna=False)
                .to_dict()
            )
            protocol_counts.update(
                cleaned["protocol"]
                .value_counts(dropna=False)
                .to_dict()
            )

            current = aggregate_chunk(
                cleaned,
                frequency=arguments.frequency,
            )
            accumulated = combine_aggregates(
                accumulated,
                current,
            )

            progress.set_postfix(
                filas=f"{totals['rows_read']:,}",
                validas=f"{totals['rows_valid']:,}",
                intervalos=f"{len(accumulated):,}",
            )

        aggregate, empty_intervals = finalize_series(
            accumulated,
            frequency=arguments.frequency,
        )

        validate_reconciliation(
            aggregate,
            valid_rows=int(totals["rows_valid"]),
        )

        processing_seconds = time.perf_counter() - start_time
        rows_per_second = (
            totals["rows_read"] / processing_seconds
            if processing_seconds > 0
            else 0.0
        )

        write_atomic_parquet(
            aggregate,
            arguments.output_parquet,
        )
        write_atomic_csv(
            aggregate,
            arguments.output_csv,
        )

        summary = ProcessingSummary(
            archive=str(arguments.archive.resolve()),
            member=arguments.member,
            frequency=arguments.frequency,
            chunk_size=arguments.chunksize,
            max_rows=arguments.max_rows,
            rows_read=int(totals["rows_read"]),
            rows_valid=int(totals["rows_valid"]),
            rows_discarded=int(totals["rows_discarded"]),
            invalid_timestamps=int(totals["invalid_timestamps"]),
            invalid_packets=int(totals["invalid_packets"]),
            invalid_bytes=int(totals["invalid_bytes"]),
            invalid_durations=int(totals["invalid_durations"]),
            negative_packets=int(totals["negative_packets"]),
            negative_bytes=int(totals["negative_bytes"]),
            negative_durations=int(totals["negative_durations"]),
            first_timestamp=(
                aggregate.index.min().isoformat()
                if not aggregate.empty
                else None
            ),
            last_timestamp=(
                aggregate.index.max().isoformat()
                if not aggregate.empty
                else None
            ),
            intervals_generated=int(len(aggregate)),
            empty_intervals_inserted=int(empty_intervals),
            total_flows=int(aggregate["flows_total"].sum()),
            total_packets=int(aggregate["packets_total"].sum()),
            total_bytes=int(aggregate["bytes_total"].sum()),
            label_counts=normalize_counter(label_counts),
            protocol_counts=normalize_counter(protocol_counts),
            processing_seconds=float(processing_seconds),
            rows_per_second=float(rows_per_second),
            output_parquet=str(arguments.output_parquet.resolve()),
            output_csv=str(arguments.output_csv.resolve()),
            completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        )

        summary_payload = asdict(summary)
        write_json(
            summary_payload,
            arguments.summary_json,
        )

        report = build_report(summary, aggregate)
        temporary_report = arguments.report_txt.with_name(
            f".{arguments.report_txt.name}.tmp"
        )
        temporary_report.write_text(
            report,
            encoding="utf-8",
        )
        os.replace(temporary_report, arguments.report_txt)

        print()
        print(report)
        print(
            f"Resumen JSON: {arguments.summary_json.resolve()}"
        )
        print(
            f"Informe TXT:  {arguments.report_txt.resolve()}"
        )

    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        tarfile.TarError,
        pd.errors.ParserError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
