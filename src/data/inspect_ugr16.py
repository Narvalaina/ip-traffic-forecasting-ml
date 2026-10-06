#!/usr/bin/env python3
"""
Inspección controlada del dataset UGR'16.

Lee una muestra de registros directamente desde el archivo TAR.GZ,
sin extraer el CSV completo al disco ni cargar todo el dataset en memoria.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import pandas as pd


COLUMNS = [
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

DTYPES = {
    "duration": "float64",
    "src_ip": "string",
    "dst_ip": "string",
    "src_port": "Int64",
    "dst_port": "Int64",
    "protocol": "string",
    "flags": "string",
    "forwarding_status": "Int64",
    "src_tos": "Int64",
    "packets": "Int64",
    "bytes": "Int64",
    "label": "string",
}

DEFAULT_ARCHIVE = Path(
    "data/raw/ugr16/calibration/march_week3_csv.tar.gz"
)

DEFAULT_MEMBER = "uniq/march.week3.csv.uniqblacklistremoved"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona una muestra del CSV UGR'16 almacenado "
            "dentro de un archivo TAR.GZ."
        )
    )

    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Ruta al archivo TAR.GZ de UGR'16.",
    )

    parser.add_argument(
        "--member",
        type=str,
        default=DEFAULT_MEMBER,
        help="Nombre del fichero CSV dentro del TAR.GZ.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Número máximo de registros que se leerán.",
    )

    return parser.parse_args()


def load_sample(
    archive_path: Path,
    member_name: str,
    number_of_rows: int,
) -> pd.DataFrame:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {archive_path}"
        )

    if number_of_rows <= 0:
        raise ValueError("--rows debe ser un número entero mayor que cero.")

    with tarfile.open(archive_path, mode="r:gz") as tar_file:
        try:
            member = tar_file.getmember(member_name)
        except KeyError as error:
            available_members = "\n".join(tar_file.getnames())
            raise KeyError(
                "No se encuentra el miembro solicitado dentro del TAR.GZ.\n"
                f"Miembro solicitado: {member_name}\n"
                f"Miembros disponibles:\n{available_members}"
            ) from error

        member_stream = tar_file.extractfile(member)

        if member_stream is None:
            raise RuntimeError(
                f"No se pudo abrir el fichero interno: {member_name}"
            )

        dataframe = pd.read_csv(
            member_stream,
            header=None,
            names=COLUMNS,
            dtype=DTYPES,
            nrows=number_of_rows,
            skip_blank_lines=True,
            on_bad_lines="warn",
            low_memory=False,
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    return dataframe


def print_report(
    dataframe: pd.DataFrame,
    archive_path: Path,
    member_name: str,
) -> None:
    print("=" * 72)
    print("INSPECCIÓN CONTROLADA DEL DATASET UGR'16")
    print("=" * 72)

    print(f"Archivo:          {archive_path.resolve()}")
    print(f"Miembro interno:  {member_name}")
    print(f"Filas leídas:     {len(dataframe):,}")
    print(f"Columnas:         {len(dataframe.columns)}")
    print(
        "Memoria muestra:  "
        f"{dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MiB"
    )

    print("\nCOLUMNAS Y TIPOS")
    print("-" * 72)
    print(dataframe.dtypes.to_string())

    print("\nPRIMEROS CINCO REGISTROS")
    print("-" * 72)
    print(dataframe.head(5).to_string(index=False))

    print("\nVALORES NULOS")
    print("-" * 72)
    print(dataframe.isna().sum().to_string())

    print("\nRANGO TEMPORAL DE LA MUESTRA")
    print("-" * 72)
    print(f"Inicio: {dataframe['timestamp'].min()}")
    print(f"Fin:    {dataframe['timestamp'].max()}")

    print("\nPROTOCOLOS")
    print("-" * 72)
    print(
        dataframe["protocol"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nETIQUETAS")
    print("-" * 72)
    print(
        dataframe["label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nESTADÍSTICAS NUMÉRICAS")
    print("-" * 72)

    numeric_columns = [
        "duration",
        "src_port",
        "dst_port",
        "forwarding_status",
        "src_tos",
        "packets",
        "bytes",
    ]

    print(
        dataframe[numeric_columns]
        .describe(percentiles=[0.50, 0.90, 0.99])
        .transpose()
        .to_string()
    )

    print("\nCONTROL DE CALIDAD")
    print("-" * 72)
    print(f"Filas duplicadas: {int(dataframe.duplicated().sum()):,}")
    print(
        "Timestamps inválidos: "
        f"{int(dataframe['timestamp'].isna().sum()):,}"
    )
    print(
        "Paquetes negativos: "
        f"{int((dataframe['packets'] < 0).sum()):,}"
    )
    print(
        "Bytes negativos: "
        f"{int((dataframe['bytes'] < 0).sum()):,}"
    )

    print("\nInspección terminada correctamente.")


def main() -> int:
    arguments = parse_arguments()

    try:
        dataframe = load_sample(
            archive_path=arguments.archive,
            member_name=arguments.member,
            number_of_rows=arguments.rows,
        )

        print_report(
            dataframe=dataframe,
            archive_path=arguments.archive,
            member_name=arguments.member,
        )

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        tarfile.TarError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())