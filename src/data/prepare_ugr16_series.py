#!/usr/bin/env python3
"""
Prepara las series temporales UGR'16 para modelado.

- Valida la serie agregada de 1 minuto.
- Elimina los dos intervalos extremos parciales.
- Alinea una ventana común de horas completas.
- Renombra *_other_label como *_anomaly.
- Genera resoluciones de 1, 5, 10, 15 y 60 minutos.
- Guarda Parquet, CSV, manifiesto JSON e informe TXT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/interim/ugr16/march_week3_1min.parquet")
DEFAULT_SUMMARY = Path(
    "results/metrics/ugr16_march_week3_aggregation_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/ugr16")
DEFAULT_MANIFEST = Path(
    "results/metrics/ugr16_march_week3_preparation_manifest.json"
)
DEFAULT_REPORT = Path(
    "results/metrics/ugr16_march_week3_preparation_report.txt"
)
DEFAULT_PREFIX = "march_week3_prepared"
DEFAULT_FREQUENCIES = ["1min", "5min", "10min", "15min", "1h"]

LABEL_GROUPS = ["background", "blacklist", "anomaly"]
PROTOCOL_GROUPS = ["tcp", "udp", "icmp", "other_protocol"]

SUM_COLUMNS = ["flows_total", "packets_total", "bytes_total"]
for group in LABEL_GROUPS:
    SUM_COLUMNS += [
        f"flows_{group}",
        f"packets_{group}",
        f"bytes_{group}",
    ]
for group in PROTOCOL_GROUPS:
    SUM_COLUMNS += [
        f"{group}_flows",
        f"{group}_packets",
        f"{group}_bytes",
    ]
SUM_COLUMNS += ["duration_sum"]

MAX_COLUMNS = ["max_flow_bytes", "max_flow_packets"]
DERIVED_COLUMNS = [
    "bitrate_bps",
    "packet_rate_pps",
    "flow_rate_fps",
    "mean_flow_duration",
]
OUTPUT_COLUMNS = [
    "timestamp",
    "flows_total",
    "packets_total",
    "bytes_total",
    "bitrate_bps",
    "packet_rate_pps",
    "flow_rate_fps",
    "mean_flow_duration",
    "max_flow_bytes",
    "max_flow_packets",
]
for group in LABEL_GROUPS:
    OUTPUT_COLUMNS += [
        f"flows_{group}",
        f"packets_{group}",
        f"bytes_{group}",
    ]
for group in PROTOCOL_GROUPS:
    OUTPUT_COLUMNS += [
        f"{group}_flows",
        f"{group}_packets",
        f"{group}_bytes",
    ]
OUTPUT_COLUMNS += ["duration_sum"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara series UGR'16 alineadas para modelado."
    )
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--aggregation-summary", type=Path, default=DEFAULT_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--frequencies", nargs="+", default=DEFAULT_FREQUENCIES
    )
    parser.add_argument("--alignment-frequency", default="1h")
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-txt", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seconds(frequency: str) -> float:
    try:
        value = pd.Timedelta(frequency).total_seconds()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Frecuencia no válida: {frequency}") from error
    if value <= 0:
        raise ValueError(f"Frecuencia no positiva: {frequency}")
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_csv(path: Path, dataframe: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON no contiene un objeto: {path}")
    return payload


def validate_paths(args: argparse.Namespace) -> None:
    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"No existe: {args.input_parquet}")
    if not args.aggregation_summary.is_file():
        raise FileNotFoundError(f"No existe: {args.aggregation_summary}")

    if len(set(args.frequencies)) != len(args.frequencies):
        raise ValueError("--frequencies contiene valores duplicados.")

    alignment_seconds = seconds(args.alignment_frequency)
    for frequency in args.frequencies:
        if not (alignment_seconds / seconds(frequency)).is_integer():
            raise ValueError(
                f"{args.alignment_frequency} no es múltiplo de {frequency}."
            )

    outputs = [args.manifest_json, args.report_txt]
    for frequency in args.frequencies:
        outputs += [
            args.output_dir / f"{args.prefix}_{frequency}.parquet",
            args.output_dir / f"{args.prefix}_{frequency}.csv",
        ]

    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite:\n"
                + "\n".join(existing)
            )


def rename_anomalies(
    dataframe: pd.DataFrame,
    summary: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    label_counts = summary.get("label_counts")
    if not isinstance(label_counts, dict):
        raise ValueError("El resumen no contiene label_counts.")

    anomaly_labels = sorted(
        str(label)
        for label in label_counts
        if str(label) not in {"background", "blacklist"}
    )
    invalid = [
        label
        for label in anomaly_labels
        if not label.startswith("anomaly-")
    ]
    if invalid:
        raise ValueError(
            "Hay etiquetas no compatibles con anomaly: "
            + ", ".join(invalid)
        )

    rename_map = {
        "flows_other_label": "flows_anomaly",
        "packets_other_label": "packets_anomaly",
        "bytes_other_label": "bytes_anomaly",
    }
    missing = [column for column in rename_map if column not in dataframe]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))

    return dataframe.rename(columns=rename_map), anomaly_labels


def validate_schema(dataframe: pd.DataFrame) -> None:
    required = (
        {"timestamp"}
        | set(SUM_COLUMNS)
        | set(MAX_COLUMNS)
        | set(DERIVED_COLUMNS)
    )
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )
    if dataframe.empty:
        raise ValueError("La serie está vacía.")
    if dataframe["timestamp"].isna().any():
        raise ValueError("Hay timestamps nulos.")
    if dataframe["timestamp"].duplicated().any():
        raise ValueError("Hay timestamps duplicados.")
    if not dataframe["timestamp"].is_monotonic_increasing:
        raise ValueError("Los timestamps no están ordenados.")


def validate_regular(
    dataframe: pd.DataFrame,
    expected: pd.Timedelta,
) -> None:
    differences = dataframe["timestamp"].diff().dropna()
    if differences.empty:
        raise ValueError("Se necesitan al menos dos observaciones.")
    if not bool(differences.eq(expected).all()):
        raise ValueError(f"La serie no es regular a {expected}.")


def validate_reconciliation(dataframe: pd.DataFrame) -> None:
    checks: dict[str, pd.Series] = {}

    for metric, total in [
        ("flows", "flows_total"),
        ("packets", "packets_total"),
        ("bytes", "bytes_total"),
    ]:
        label_columns = [f"{metric}_{group}" for group in LABEL_GROUPS]
        protocol_columns = [
            f"{group}_{metric}" for group in PROTOCOL_GROUPS
        ]
        checks[f"{metric}/label"] = (
            dataframe[label_columns].sum(axis=1) == dataframe[total]
        )
        checks[f"{metric}/protocol"] = (
            dataframe[protocol_columns].sum(axis=1) == dataframe[total]
        )

    failures = {
        name: int((~mask).sum())
        for name, mask in checks.items()
        if not bool(mask.all())
    }
    if failures:
        detail = ", ".join(
            f"{name}={count}" for name, count in failures.items()
        )
        raise RuntimeError(f"Fallo de reconciliación: {detail}")


def common_window(
    dataframe: pd.DataFrame,
    alignment_frequency: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(dataframe) < 3:
        raise ValueError("No hay filas suficientes para retirar extremos.")

    first = dataframe.iloc[0]
    last = dataframe.iloc[-1]

    # El primer y el último minuto son parciales en esta captura.
    trimmed = dataframe.iloc[1:-1].reset_index(drop=True)

    aligned_start = trimmed["timestamp"].iloc[0].ceil(
        alignment_frequency
    )
    aligned_end_exclusive = (
        trimmed["timestamp"].iloc[-1] + pd.Timedelta("1min")
    ).floor(alignment_frequency)

    if aligned_start >= aligned_end_exclusive:
        raise ValueError("La ventana temporal alineada está vacía.")

    mask = (
        (trimmed["timestamp"] >= aligned_start)
        & (trimmed["timestamp"] < aligned_end_exclusive)
    )
    aligned = trimmed.loc[mask].reset_index(drop=True)

    expected_rows = int(
        (aligned_end_exclusive - aligned_start)
        / pd.Timedelta("1min")
    )
    if len(aligned) != expected_rows:
        raise RuntimeError(
            f"Minutos alineados: {len(aligned)} != {expected_rows}."
        )

    info = {
        "input_rows": int(len(dataframe)),
        "input_start": first["timestamp"].isoformat(),
        "input_end": last["timestamp"].isoformat(),
        "removed_first_timestamp": first["timestamp"].isoformat(),
        "removed_first_flows": int(first["flows_total"]),
        "removed_last_timestamp": last["timestamp"].isoformat(),
        "removed_last_flows": int(last["flows_total"]),
        "aligned_start": aligned_start.isoformat(),
        "aligned_end_exclusive": aligned_end_exclusive.isoformat(),
        "alignment_rows_removed_after_edges": int(
            len(trimmed) - len(aligned)
        ),
        "aligned_1min_rows": int(len(aligned)),
    }
    return aligned, info


def aggregate(
    base: pd.DataFrame,
    frequency: str,
) -> tuple[pd.DataFrame, int]:
    interval_seconds = seconds(frequency)
    source_intervals = int(interval_seconds / 60.0)

    grouped = (
        base.set_index("timestamp")
        .resample(
            frequency,
            origin="start_day",
            label="left",
            closed="left",
        )
    )

    counts = grouped.size()
    incomplete = int(counts.ne(source_intervals).sum())
    if incomplete:
        raise RuntimeError(
            f"{frequency}: {incomplete} bins incompletos."
        )

    output = (
        grouped[SUM_COLUMNS]
        .sum()
        .join(grouped[MAX_COLUMNS].max())
    )

    integer_columns = [
        column
        for column in SUM_COLUMNS + MAX_COLUMNS
        if column != "duration_sum"
    ]
    output[integer_columns] = output[integer_columns].astype("int64")
    output["duration_sum"] = output["duration_sum"].astype("float64")

    output["mean_flow_duration"] = (
        output["duration_sum"]
        .div(output["flows_total"].replace(0, pd.NA))
        .fillna(0.0)
        .astype("float64")
    )
    output["bitrate_bps"] = (
        output["bytes_total"] * 8.0 / interval_seconds
    )
    output["packet_rate_pps"] = (
        output["packets_total"] / interval_seconds
    )
    output["flow_rate_fps"] = (
        output["flows_total"] / interval_seconds
    )

    output = output.reset_index()[OUTPUT_COLUMNS]
    validate_regular(output, pd.Timedelta(frequency))
    validate_reconciliation(output)
    return output, source_intervals


def build_report(
    input_path: Path,
    input_hash: str,
    anomaly_labels: list[str],
    window: dict[str, Any],
    outputs: list[dict[str, Any]],
) -> str:
    lines = [
        "=" * 76,
        "PREPARACIÓN DE SERIES TEMPORALES UGR'16",
        "=" * 76,
        f"Entrada:                   {input_path.resolve()}",
        f"SHA-256 entrada:           {input_hash}",
        "",
        "VENTANA TEMPORAL",
        "-" * 76,
        f"Filas de entrada:          {window['input_rows']:,}",
        f"Inicio original:           {window['input_start']}",
        f"Fin original:              {window['input_end']}",
        (
            "Primer minuto eliminado:  "
            f"{window['removed_first_timestamp']} "
            f"({window['removed_first_flows']:,} flujos)"
        ),
        (
            "Último minuto eliminado:  "
            f"{window['removed_last_timestamp']} "
            f"({window['removed_last_flows']:,} flujos)"
        ),
        f"Inicio común:              {window['aligned_start']}",
        (
            "Fin común exclusivo:      "
            f"{window['aligned_end_exclusive']}"
        ),
        (
            "Minutos conservados:      "
            f"{window['aligned_1min_rows']:,}"
        ),
        "",
        "ETIQUETAS AGRUPADAS COMO ANOMALÍA",
        "-" * 76,
        ", ".join(anomaly_labels) or "Ninguna",
        "",
        "SERIES GENERADAS",
        "-" * 76,
    ]

    for item in outputs:
        lines += [
            (
                f"{item['frequency']:>5}: {item['rows']:,} filas | "
                f"{item['start']} → {item['end']} | "
                f"{item['total_flows']:,} flujos"
            ),
            f"      Parquet: {item['parquet_path']}",
            f"      CSV:     {item['csv_path']}",
        ]

    lines += [
        "",
        "VALIDACIONES",
        "-" * 76,
        "PASS — esquema y continuidad temporal.",
        "PASS — extremos parciales eliminados.",
        "PASS — cobertura temporal común.",
        "PASS — bins completos.",
        "PASS — reconciliación por etiquetas y protocolos.",
        "PASS — totales idénticos entre resoluciones.",
        "",
        "Preparación terminada correctamente.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    try:
        validate_paths(args)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_txt.parent.mkdir(parents=True, exist_ok=True)

        summary = read_json(args.aggregation_summary)
        dataframe = pd.read_parquet(args.input_parquet)
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="raise",
        )
        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        dataframe, anomaly_labels = rename_anomalies(
            dataframe,
            summary,
        )
        validate_schema(dataframe)
        validate_regular(dataframe, pd.Timedelta("1min"))
        validate_reconciliation(dataframe)

        input_hash = sha256(args.input_parquet)

        base, window = common_window(
            dataframe,
            args.alignment_frequency,
        )
        validate_regular(base, pd.Timedelta("1min"))
        validate_reconciliation(base)

        output_metadata: list[dict[str, Any]] = []
        reference_totals: tuple[int, int, int] | None = None

        for frequency in args.frequencies:
            prepared, source_intervals = aggregate(base, frequency)

            totals = (
                int(prepared["flows_total"].sum()),
                int(prepared["packets_total"].sum()),
                int(prepared["bytes_total"].sum()),
            )
            if reference_totals is None:
                reference_totals = totals
            elif totals != reference_totals:
                raise RuntimeError(
                    f"{frequency}: totales diferentes a la referencia."
                )

            parquet_path = (
                args.output_dir
                / f"{args.prefix}_{frequency}.parquet"
            )
            csv_path = (
                args.output_dir
                / f"{args.prefix}_{frequency}.csv"
            )
            atomic_parquet(parquet_path, prepared)
            atomic_csv(csv_path, prepared)

            output_metadata.append(
                {
                    "frequency": frequency,
                    "interval_seconds": seconds(frequency),
                    "source_intervals_per_row": source_intervals,
                    "rows": int(len(prepared)),
                    "start": prepared["timestamp"].iloc[0].isoformat(),
                    "end": prepared["timestamp"].iloc[-1].isoformat(),
                    "coverage_end_exclusive": (
                        prepared["timestamp"].iloc[-1]
                        + pd.Timedelta(frequency)
                    ).isoformat(),
                    "total_flows": totals[0],
                    "total_packets": totals[1],
                    "total_bytes": totals[2],
                    "parquet_path": str(parquet_path.resolve()),
                    "parquet_sha256": sha256(parquet_path),
                    "csv_path": str(csv_path.resolve()),
                    "csv_sha256": sha256(csv_path),
                }
            )

        report = build_report(
            args.input_parquet,
            input_hash,
            anomaly_labels,
            window,
            output_metadata,
        )

        manifest = {
            "input": {
                "parquet_path": str(args.input_parquet.resolve()),
                "parquet_sha256": input_hash,
                "aggregation_summary_path": str(
                    args.aggregation_summary.resolve()
                ),
            },
            "preparation": {
                "alignment_frequency": args.alignment_frequency,
                "frequencies": args.frequencies,
                "anomaly_labels": anomaly_labels,
                "window": window,
            },
            "outputs": output_metadata,
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        atomic_json(args.manifest_json, manifest)
        atomic_text(args.report_txt, report)

        print(report)
        print(f"Manifiesto: {args.manifest_json.resolve()}")
        print(f"Informe:    {args.report_txt.resolve()}")

    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
