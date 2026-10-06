#!/usr/bin/env python3
"""
Corrige la serie agregada de UGR'16 April Week #3 tras la auditoría completa
del esquema raw.

La auditoría confirmó exactamente dos líneas mal formadas:
- una línea de 17 campos que fue incluida por el agregador como un flujo TCP
  con label "0";
- una línea de 20 campos que ya fue descartada por packets no numérico.

Este script:
1. conserva intactos los artefactos originales;
2. elimina del minuto 2016-04-11 22:55:00 la contribución de la línea de
   17 campos incluida erróneamente;
3. actualiza las métricas derivadas;
4. crea una serie limpia en Parquet y CSV;
5. genera un resumen JSON limpio, un manifiesto y un informe;
6. valida continuidad y reconciliación por etiquetas y protocolos.
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


DEFAULT_INPUT_PARQUET = Path(
    "data/interim/ugr16/april_week3_1min.parquet"
)
DEFAULT_INPUT_SUMMARY = Path(
    "results/metrics/ugr16_april_week3_aggregation_summary.json"
)
DEFAULT_SCHEMA_AUDIT = Path(
    "results/metrics/ugr16_april_week3_full_schema_audit.txt"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "data/interim/ugr16/april_week3_1min_clean.parquet"
)
DEFAULT_OUTPUT_CSV = Path(
    "data/interim/ugr16/april_week3_1min_clean.csv"
)
DEFAULT_OUTPUT_SUMMARY = Path(
    "results/metrics/ugr16_april_week3_aggregation_clean_summary.json"
)
DEFAULT_MANIFEST = Path(
    "results/metrics/ugr16_april_week3_aggregation_clean_manifest.json"
)
DEFAULT_REPORT = Path(
    "results/metrics/ugr16_april_week3_aggregation_clean_report.txt"
)

AFFECTED_TIMESTAMP = pd.Timestamp("2016-04-11 22:55:00")

# Contribución que el agregador incluyó a partir de la línea raw de 17 campos.
CORRECTION = {
    "flows_total": 1,
    "packets_total": 4,
    "bytes_total": 1030,
    "duration_sum": 4.284,
    "flows_other_label": 1,
    "packets_other_label": 4,
    "bytes_other_label": 1030,
    "tcp_flows": 1,
    "tcp_packets": 4,
    "tcp_bytes": 1030,
}

EXPECTED_RAW_ROWS = 1_107_385_572
EXPECTED_STRUCTURALLY_VALID_ROWS = 1_107_385_570
EXPECTED_MALFORMED_ROWS = 2
EXPECTED_ORIGINAL_VALID_ROWS = 1_107_385_571
EXPECTED_ORIGINAL_DISCARDED_ROWS = 1
EXPECTED_ORIGINAL_LABEL_ZERO = 1
EXPECTED_ORIGINAL_TCP = 657_380_040


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrige la agregación de UGR'16 April Week #3 después de "
            "la auditoría estructural completa."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=DEFAULT_INPUT_PARQUET,
    )
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=DEFAULT_INPUT_SUMMARY,
    )
    parser.add_argument(
        "--schema-audit",
        type=Path,
        default=DEFAULT_SCHEMA_AUDIT,
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON no contiene un objeto: {path}")
    return payload


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


def atomic_csv(path: Path, dataframe: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def validate_paths(args: argparse.Namespace) -> None:
    inputs = [
        args.input_parquet,
        args.input_summary,
        args.schema_audit,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"No existe: {path}")

    outputs = [
        args.output_parquet,
        args.output_csv,
        args.output_summary,
        args.manifest_json,
        args.report_txt,
    ]
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Ya existen salidas. Usa --overwrite:\n"
                + "\n".join(existing)
            )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)


def validate_audit(audit_text: str) -> None:
    required_fragments = [
        "Filas de datos:                 1107385572",
        "Filas con exactamente 13 campos:1107385570",
        "Filas con esquema incorrecto:   2",
        "13 campos: 1107385570",
        "17 campos: 1",
        "20 campos: 1",
        "Paquetes no numéricos:          1",
        "[0]: 1",
        "[53]: 1",
    ]
    missing = [
        fragment for fragment in required_fragments
        if fragment not in audit_text
    ]
    if missing:
        raise ValueError(
            "La auditoría no contiene los resultados esperados:\n"
            + "\n".join(missing)
        )


def validate_original_summary(summary: dict[str, Any]) -> None:
    expected = {
        "rows_read": EXPECTED_RAW_ROWS,
        "rows_valid": EXPECTED_ORIGINAL_VALID_ROWS,
        "rows_discarded": EXPECTED_ORIGINAL_DISCARDED_ROWS,
        "invalid_packets": 1,
        "total_flows": EXPECTED_ORIGINAL_VALID_ROWS,
        "total_packets": 25_072_915_147,
        "total_bytes": 18_337_250_119_833,
    }
    failures = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if failures:
        detail = "\n".join(
            f"{key}: obtenido={obtained!r}, esperado={expected_value!r}"
            for key, (obtained, expected_value) in failures.items()
        )
        raise ValueError(
            "El resumen original no coincide con la campaña auditada:\n"
            + detail
        )

    labels = summary.get("label_counts")
    protocols = summary.get("protocol_counts")
    if not isinstance(labels, dict) or not isinstance(protocols, dict):
        raise ValueError("Faltan label_counts o protocol_counts.")

    if labels.get("0") != EXPECTED_ORIGINAL_LABEL_ZERO:
        raise ValueError("No se encontró exactamente una etiqueta '0'.")
    if protocols.get("TCP") != EXPECTED_ORIGINAL_TCP:
        raise ValueError("El conteo TCP original no coincide.")


def validate_series(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise ValueError("La serie está vacía.")

    required = {
        "timestamp",
        "flows_total",
        "packets_total",
        "bytes_total",
        "duration_sum",
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
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))

    if dataframe["timestamp"].isna().any():
        raise ValueError("Hay timestamps nulos.")
    if dataframe["timestamp"].duplicated().any():
        raise ValueError("Hay timestamps duplicados.")
    if not dataframe["timestamp"].is_monotonic_increasing:
        raise ValueError("Los timestamps no están ordenados.")

    differences = dataframe["timestamp"].diff().dropna()
    if not bool(differences.eq(pd.Timedelta("1min")).all()):
        raise ValueError("La serie no mantiene una cadencia regular de 1 minuto.")


def validate_reconciliation(dataframe: pd.DataFrame) -> None:
    label_checks = {
        "flows": (
            dataframe[
                [
                    "flows_background",
                    "flows_blacklist",
                    "flows_other_label",
                ]
            ].sum(axis=1)
            == dataframe["flows_total"]
        ),
        "packets": (
            dataframe[
                [
                    "packets_background",
                    "packets_blacklist",
                    "packets_other_label",
                ]
            ].sum(axis=1)
            == dataframe["packets_total"]
        ),
        "bytes": (
            dataframe[
                [
                    "bytes_background",
                    "bytes_blacklist",
                    "bytes_other_label",
                ]
            ].sum(axis=1)
            == dataframe["bytes_total"]
        ),
    }
    protocol_checks = {
        "flows": (
            dataframe[
                [
                    "tcp_flows",
                    "udp_flows",
                    "icmp_flows",
                    "other_protocol_flows",
                ]
            ].sum(axis=1)
            == dataframe["flows_total"]
        ),
        "packets": (
            dataframe[
                [
                    "tcp_packets",
                    "udp_packets",
                    "icmp_packets",
                    "other_protocol_packets",
                ]
            ].sum(axis=1)
            == dataframe["packets_total"]
        ),
        "bytes": (
            dataframe[
                [
                    "tcp_bytes",
                    "udp_bytes",
                    "icmp_bytes",
                    "other_protocol_bytes",
                ]
            ].sum(axis=1)
            == dataframe["bytes_total"]
        ),
    }

    failures: list[str] = []
    for metric, mask in label_checks.items():
        if not bool(mask.all()):
            failures.append(
                f"{metric}/label={int((~mask).sum())}"
            )
    for metric, mask in protocol_checks.items():
        if not bool(mask.all()):
            failures.append(
                f"{metric}/protocol={int((~mask).sum())}"
            )
    if failures:
        raise RuntimeError(
            "Fallo de reconciliación: " + ", ".join(failures)
        )


def apply_correction(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    corrected = dataframe.copy()

    mask = corrected["timestamp"].eq(AFFECTED_TIMESTAMP)
    if int(mask.sum()) != 1:
        raise ValueError(
            "Debe existir exactamente un intervalo afectado; "
            f"encontrados={int(mask.sum())}."
        )

    index = corrected.index[mask][0]
    before = corrected.loc[index].to_dict()

    expected_before = {
        "flows_total": 107_789,
        "packets_total": 2_783_674,
        "bytes_total": 2_040_841_358,
        "duration_sum": 475_236.9,
        "flows_other_label": 5_952,
        "packets_other_label": 88_112,
        "bytes_other_label": 11_057_280,
        "tcp_flows": 71_827,
        "tcp_packets": 2_361_973,
        "tcp_bytes": 1_862_430_263,
        "max_flow_bytes": 219_320_172,
        "max_flow_packets": 146_627,
    }
    for column, expected_value in expected_before.items():
        obtained = before[column]
        if isinstance(expected_value, float):
            if abs(float(obtained) - expected_value) > 1e-9:
                raise ValueError(
                    f"{column}: obtenido={obtained}, "
                    f"esperado={expected_value}."
                )
        elif int(obtained) != expected_value:
            raise ValueError(
                f"{column}: obtenido={obtained}, "
                f"esperado={expected_value}."
            )

    for column, amount in CORRECTION.items():
        corrected.at[index, column] = corrected.at[index, column] - amount

    corrected.at[index, "bitrate_bps"] = (
        corrected.at[index, "bytes_total"] * 8.0 / 60.0
    )
    corrected.at[index, "packet_rate_pps"] = (
        corrected.at[index, "packets_total"] / 60.0
    )
    corrected.at[index, "flow_rate_fps"] = (
        corrected.at[index, "flows_total"] / 60.0
    )
    corrected.at[index, "mean_flow_duration"] = (
        corrected.at[index, "duration_sum"]
        / corrected.at[index, "flows_total"]
    )

    integer_columns = [
        column for column in corrected.columns
        if column != "timestamp"
        and column != "duration_sum"
        and not column.endswith("_bps")
        and not column.endswith("_pps")
        and not column.endswith("_fps")
        and column != "mean_flow_duration"
    ]
    for column in integer_columns:
        if pd.api.types.is_integer_dtype(dataframe[column].dtype):
            corrected[column] = corrected[column].astype("int64")

    after = corrected.loc[index].to_dict()
    return corrected, before, after


def build_clean_summary(
    original: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    clean = dict(original)

    clean["rows_valid"] = EXPECTED_STRUCTURALLY_VALID_ROWS
    clean["rows_discarded"] = EXPECTED_MALFORMED_ROWS
    clean["total_flows"] = original["total_flows"] - 1
    clean["total_packets"] = original["total_packets"] - 4
    clean["total_bytes"] = original["total_bytes"] - 1030

    labels = dict(original["label_counts"])
    if labels.get("0") != 1:
        raise ValueError("No se puede retirar la etiqueta '0'.")
    del labels["0"]
    clean["label_counts"] = labels

    protocols = dict(original["protocol_counts"])
    protocols["TCP"] = protocols["TCP"] - 1
    clean["protocol_counts"] = protocols

    clean["output_parquet"] = str(args.output_parquet.resolve())
    clean["output_csv"] = str(args.output_csv.resolve())

    clean["structural_quality_control"] = {
        "schema_fields_expected": 13,
        "raw_rows_read": EXPECTED_RAW_ROWS,
        "structurally_valid_rows": EXPECTED_STRUCTURALLY_VALID_ROWS,
        "malformed_rows": EXPECTED_MALFORMED_ROWS,
        "malformed_field_count_distribution": {
            "17": 1,
            "20": 1,
        },
        "blank_physical_lines": 1,
        "malformed_percentage": (
            100.0 * EXPECTED_MALFORMED_ROWS / EXPECTED_RAW_ROWS
        ),
        "included_malformed_row_removed": {
            "data_row": 146_924_782,
            "physical_line": 146_924_783,
            "timestamp": "2016-04-11T22:55:15",
            "aggregated_interval": AFFECTED_TIMESTAMP.isoformat(),
            "field_count": 17,
            "interpreted_protocol": "TCP",
            "interpreted_packets": 4,
            "interpreted_bytes": 1030,
            "interpreted_label": "0",
        },
        "already_discarded_malformed_row": {
            "data_row": 597_637_693,
            "physical_line": 597_637_694,
            "timestamp": "2016-04-14T16:30:18",
            "field_count": 20,
            "reason": "packets no numérico por concatenación de registros",
        },
        "policy": (
            "Excluir líneas raw cuyo número de campos sea distinto de 13; "
            "no reconstruir registros mediante imputaciones no verificables."
        ),
    }
    return clean


def validate_clean_totals(
    dataframe: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    expected = {
        "total_flows": 1_107_385_570,
        "total_packets": 25_072_915_143,
        "total_bytes": 18_337_250_118_803,
    }
    obtained = {
        "total_flows": int(dataframe["flows_total"].sum()),
        "total_packets": int(dataframe["packets_total"].sum()),
        "total_bytes": int(dataframe["bytes_total"].sum()),
    }
    for key, expected_value in expected.items():
        if obtained[key] != expected_value:
            raise RuntimeError(
                f"{key}: obtenido={obtained[key]}, "
                f"esperado={expected_value}."
            )
        if summary[key] != expected_value:
            raise RuntimeError(
                f"{key} en JSON: obtenido={summary[key]}, "
                f"esperado={expected_value}."
            )

    if sum(summary["label_counts"].values()) != expected["total_flows"]:
        raise RuntimeError("label_counts no reconcilia con total_flows.")
    if sum(summary["protocol_counts"].values()) != expected["total_flows"]:
        raise RuntimeError("protocol_counts no reconcilia con total_flows.")
    if "0" in summary["label_counts"]:
        raise RuntimeError("La etiqueta espuria '0' sigue presente.")
    if summary["protocol_counts"]["TCP"] != 657_380_039:
        raise RuntimeError("El conteo TCP limpio no coincide.")


def scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    args = parse_args()

    try:
        validate_paths(args)

        audit_text = args.schema_audit.read_text(encoding="utf-8")
        validate_audit(audit_text)

        original_summary = read_json(args.input_summary)
        validate_original_summary(original_summary)

        dataframe = pd.read_parquet(args.input_parquet)
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="raise",
        )
        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        validate_series(dataframe)
        validate_reconciliation(dataframe)

        corrected, before, after = apply_correction(dataframe)
        validate_series(corrected)
        validate_reconciliation(corrected)

        clean_summary = build_clean_summary(original_summary, args)
        validate_clean_totals(corrected, clean_summary)

        atomic_parquet(args.output_parquet, corrected)
        atomic_csv(args.output_csv, corrected)
        atomic_json(args.output_summary, clean_summary)

        input_hashes = {
            "input_parquet": sha256(args.input_parquet),
            "input_summary": sha256(args.input_summary),
            "schema_audit": sha256(args.schema_audit),
        }
        output_hashes = {
            "output_parquet": sha256(args.output_parquet),
            "output_csv": sha256(args.output_csv),
            "output_summary": sha256(args.output_summary),
        }

        manifest = {
            "campaign": "UGR16-APRIL-AGG-CLEAN-001",
            "status": "PASS",
            "inputs": {
                "aggregation_parquet": {
                    "path": str(args.input_parquet.resolve()),
                    "sha256": input_hashes["input_parquet"],
                },
                "aggregation_summary": {
                    "path": str(args.input_summary.resolve()),
                    "sha256": input_hashes["input_summary"],
                },
                "schema_audit": {
                    "path": str(args.schema_audit.resolve()),
                    "sha256": input_hashes["schema_audit"],
                },
            },
            "correction": {
                "timestamp": AFFECTED_TIMESTAMP.isoformat(),
                "subtract": CORRECTION,
                "before": {
                    key: scalar(value) for key, value in before.items()
                },
                "after": {
                    key: scalar(value) for key, value in after.items()
                },
            },
            "validation": {
                "full_raw_schema_audit_completed": True,
                "exactly_two_malformed_rows": True,
                "only_one_malformed_row_was_aggregated": True,
                "time_series_regular": True,
                "timestamps_unique": True,
                "label_reconciliation": True,
                "protocol_reconciliation": True,
                "global_totals_reconciled": True,
                "spurious_label_zero_removed": True,
                "original_artifacts_preserved": True,
            },
            "outputs": {
                "clean_parquet": {
                    "path": str(args.output_parquet.resolve()),
                    "sha256": output_hashes["output_parquet"],
                    "bytes": args.output_parquet.stat().st_size,
                },
                "clean_csv": {
                    "path": str(args.output_csv.resolve()),
                    "sha256": output_hashes["output_csv"],
                    "bytes": args.output_csv.stat().st_size,
                },
                "clean_summary": {
                    "path": str(args.output_summary.resolve()),
                    "sha256": output_hashes["output_summary"],
                    "bytes": args.output_summary.stat().st_size,
                },
            },
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }
        atomic_json(args.manifest_json, manifest)

        report_lines = [
            "=" * 80,
            "CORRECCIÓN DE LA AGREGACIÓN UGR'16 — APRIL WEEK #3",
            "=" * 80,
            "Campaña:                 UGR16-APRIL-AGG-CLEAN-001",
            "Estado:                  PASS",
            "",
            "AUDITORÍA RAW",
            "-" * 80,
            f"Filas de datos:          {EXPECTED_RAW_ROWS:,}",
            (
                "Filas con 13 campos:   "
                f"{EXPECTED_STRUCTURALLY_VALID_ROWS:,}"
            ),
            f"Filas mal formadas:      {EXPECTED_MALFORMED_ROWS:,}",
            "Distribución:            1 fila con 17 campos; 1 con 20",
            "",
            "CORRECCIÓN APLICADA",
            "-" * 80,
            f"Intervalo:               {AFFECTED_TIMESTAMP}",
            "Flujos retirados:        1",
            "Paquetes retirados:      4",
            "Bytes retirados:         1,030",
            "Duración retirada:       4.284 s",
            "Grupo de etiqueta:       other_label",
            "Protocolo:               TCP",
            "Etiqueta espuria:        0",
            "",
            "TOTALES LIMPIOS",
            "-" * 80,
            f"Flujos:                  {clean_summary['total_flows']:,}",
            f"Paquetes:                {clean_summary['total_packets']:,}",
            f"Bytes:                   {clean_summary['total_bytes']:,}",
            "",
            "VALIDACIONES",
            "-" * 80,
            "PASS — auditoría completa del esquema raw.",
            "PASS — exactamente dos líneas mal formadas identificadas.",
            "PASS — contribución espuria retirada del minuto afectado.",
            "PASS — continuidad y unicidad temporal.",
            "PASS — reconciliación por etiquetas.",
            "PASS — reconciliación por protocolos.",
            "PASS — etiqueta espuria 0 eliminada.",
            "PASS — artefactos originales preservados.",
            "",
            "SALIDAS",
            "-" * 80,
            f"Parquet:   {args.output_parquet.resolve()}",
            f"CSV:       {args.output_csv.resolve()}",
            f"Resumen:   {args.output_summary.resolve()}",
            f"Manifiesto:{args.manifest_json.resolve()}",
            "",
            "Corrección terminada correctamente.",
        ]
        atomic_text(args.report_txt, "\n".join(report_lines) + "\n")

        print("\n".join(report_lines))
        print(f"Informe:   {args.report_txt.resolve()}")
        print(f"Manifiesto:{args.manifest_json.resolve()}")
        return 0

    except Exception as error:
        print(
            f"ERROR: {error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
