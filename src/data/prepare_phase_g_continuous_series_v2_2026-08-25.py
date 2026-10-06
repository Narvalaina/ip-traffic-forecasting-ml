#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT = Path.home() / "TFM_IP_Traffic_Forecasting"

FINAL_AUDIT_MANIFEST = (
    PROJECT
    / "results/metrics/phase_g/aggregation/final_audit/v1_2026-08-24/"
      "phase_g_aggregation_final_audit_v1_2026-08-24_manifest.json"
)
FINAL_AUDIT_MANIFEST_SHA = "6a612f2b1063387e2752367e23832717baec2d2b6ea065d331fd1bda2a8a6532"

FAILED_V1_RUNNER = PROJECT / "src/data/prepare_phase_g_continuous_series_v1_2026-08-25.py"
FAILED_V1_RUNNER_SHA = "684b256d0ec90c82748d7d4407db5fa58d1de869d2c27faa236ae7f992ef5a3e"

FAILED_V1_CONSOLE = (
    PROJECT
    / "results/metrics/phase_g/phase_g_continuity_series_preparation_console_v1_2026-08-25.txt"
)
FAILED_V1_CONSOLE_SHA = "9bb92f3f0b021eb53171e2b40f1e348f831aeb8c73a6611c0eaf4aa7c71c4c3e"

GOV071 = (
    PROJECT / "docs/project_governance/"
    "071_phase_g_aggregation_final_closure_continuity_series_preparation_authorization_2026-08-25.md"
)
GOV071_SHA = "c30d65fae44bdb1ad110c878ae62b699f2fa32b124d20c32603c6a3e3ad3b34b"

GOV072 = (
    PROJECT / "docs/project_governance/"
    "072_phase_g_continuity_boundary_incident_method_repair_authorization_2026-08-25.md"
)

OUTPUT_DIR = PROJECT / "data/processed/ugr16/phase_g"
METRICS_DIR = PROJECT / "results/metrics/phase_g/continuity_series_preparation/v2_2026-08-25"

KEYS = [
    "april_week5",
    "may_week1",
    "may_week2",
    "may_week3",
    "may_week4",
    "may_week5",
    "may_week6",
    "june_week1",
    "june_week2",
    "june_week3",
    "june_week4",
]

LABEL_RAW = ["background", "blacklist", "other_label"]
LABEL_FINAL = ["background", "blacklist", "anomaly"]
PROTOCOL_GROUPS = ["tcp", "udp", "icmp", "other_protocol"]

SUM_COLUMNS_RAW = ["flows_total", "packets_total", "bytes_total"]
for group in LABEL_RAW:
    SUM_COLUMNS_RAW += [
        f"flows_{group}",
        f"packets_{group}",
        f"bytes_{group}",
    ]
for group in PROTOCOL_GROUPS:
    SUM_COLUMNS_RAW += [
        f"{group}_flows",
        f"{group}_packets",
        f"{group}_bytes",
    ]
SUM_COLUMNS_RAW += ["duration_sum"]

MAX_COLUMNS = ["max_flow_bytes", "max_flow_packets"]

SUM_COLUMNS_FINAL = ["flows_total", "packets_total", "bytes_total"]
for group in LABEL_FINAL:
    SUM_COLUMNS_FINAL += [
        f"flows_{group}",
        f"packets_{group}",
        f"bytes_{group}",
    ]
for group in PROTOCOL_GROUPS:
    SUM_COLUMNS_FINAL += [
        f"{group}_flows",
        f"{group}_packets",
        f"{group}_bytes",
    ]
SUM_COLUMNS_FINAL += ["duration_sum"]

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
for group in LABEL_FINAL:
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


class StopError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise StopError(f"JSON no es objeto: {path}")
    return obj


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise StopError(f"Colisión de salida: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    os.replace(tmp, path)


checks: list[dict[str, Any]] = []


def check(name: str, condition: bool, detail: Any = "") -> None:
    ok = bool(condition)
    checks.append({"name": name, "pass": ok, "detail": str(detail)})
    print(f"{name:<108}: {'PASS' if ok else 'FAIL'} | {detail}")
    if not ok:
        raise StopError(f"{name}: {detail}")


def validate_regular(df: pd.DataFrame, delta: pd.Timedelta, name: str) -> None:
    diffs = df["timestamp"].diff().dropna()
    check(f"{name}: >= 2 filas", len(df) >= 2, len(df))
    bad = diffs[diffs != delta]
    check(
        f"{name}: cadencia exacta {delta}",
        bad.empty,
        "PASS" if bad.empty else f"bad={len(bad)} first={bad.head(10).to_dict()}",
    )


def recompute_derived(df: pd.DataFrame, interval_seconds: float) -> pd.DataFrame:
    out = df.copy()
    out["mean_flow_duration"] = (
        out["duration_sum"]
        .div(out["flows_total"].replace(0, pd.NA))
        .fillna(0.0)
        .astype("float64")
    )
    out["bitrate_bps"] = out["bytes_total"] * 8.0 / interval_seconds
    out["packet_rate_pps"] = out["packets_total"] / interval_seconds
    out["flow_rate_fps"] = out["flows_total"] / interval_seconds
    return out


def validate_reconciliation(df: pd.DataFrame, label_groups: list[str], name: str) -> None:
    for metric, total in [
        ("flows", "flows_total"),
        ("packets", "packets_total"),
        ("bytes", "bytes_total"),
    ]:
        label_cols = [f"{metric}_{g}" for g in label_groups]
        protocol_cols = [f"{g}_{metric}" for g in PROTOCOL_GROUPS]
        check(
            f"{name}: reconciliación {metric}/label",
            bool(df[label_cols].sum(axis=1).eq(df[total]).all()),
        )
        check(
            f"{name}: reconciliación {metric}/protocol",
            bool(df[protocol_cols].sum(axis=1).eq(df[total]).all()),
        )


def coalesce_boundary_minutes(stacked: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    counts = stacked.groupby("timestamp", sort=True).size()
    duplicate_ts = counts[counts > 1].index

    boundary_records: list[dict[str, Any]] = []
    key_index = {key: i for i, key in enumerate(KEYS)}

    for ts in duplicate_ts:
        rows = stacked.loc[stacked["timestamp"] == ts]
        keys = list(rows["_fragment_key"])
        unique_keys = list(dict.fromkeys(keys))
        check(
            f"Boundary {ts}: multiplicidad exactamente 2",
            len(rows) == 2,
            keys,
        )
        check(
            f"Boundary {ts}: dos fragmentos distintos",
            len(unique_keys) == 2,
            unique_keys,
        )
        a, b = unique_keys
        check(
            f"Boundary {ts}: fragmentos cronológicamente adyacentes",
            abs(key_index[a] - key_index[b]) == 1,
            unique_keys,
        )
        boundary_records.append({
            "timestamp": pd.Timestamp(ts).isoformat(),
            "fragment_keys": unique_keys,
            "rows_coalesced": 2,
        })

    grouped = stacked.groupby("timestamp", sort=True)
    sums = grouped[SUM_COLUMNS_RAW].sum()
    maxs = grouped[MAX_COLUMNS].max()
    merged = sums.join(maxs).reset_index()

    integer_columns = [
        c for c in SUM_COLUMNS_RAW + MAX_COLUMNS
        if c != "duration_sum"
    ]
    merged[integer_columns] = merged[integer_columns].astype("int64")
    merged["duration_sum"] = merged["duration_sum"].astype("float64")
    merged = recompute_derived(merged, 60.0)

    return merged, boundary_records


def aggregate_5min(base: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        base.set_index("timestamp")
        .resample("5min", origin="start_day", label="left", closed="left")
    )
    counts = grouped.size()
    incomplete = counts[counts != 5]
    check(
        "5min: todos los bins contienen 5 minutos",
        incomplete.empty,
        "PASS" if incomplete.empty else incomplete.head(20).to_dict(),
    )

    output = grouped[SUM_COLUMNS_FINAL].sum().join(grouped[MAX_COLUMNS].max())

    integer_columns = [
        c for c in SUM_COLUMNS_FINAL + MAX_COLUMNS
        if c != "duration_sum"
    ]
    output[integer_columns] = output[integer_columns].astype("int64")
    output["duration_sum"] = output["duration_sum"].astype("float64")
    output = output.reset_index()
    output = recompute_derived(output, 300.0)
    output = output[OUTPUT_COLUMNS]
    validate_regular(output, pd.Timedelta("5min"), "Serie global 5min")
    validate_reconciliation(output, LABEL_FINAL, "Serie global 5min")
    return output


def save_pair(stem: str, df: pd.DataFrame) -> dict[str, Any]:
    parquet = OUTPUT_DIR / f"{stem}.parquet"
    csv = OUTPUT_DIR / f"{stem}.csv"
    atomic_parquet(parquet, df)
    atomic_csv(csv, df)
    return {
        "rows": int(len(df)),
        "start": df["timestamp"].iloc[0].isoformat(),
        "end": df["timestamp"].iloc[-1].isoformat(),
        "coverage_end_exclusive": (
            df["timestamp"].iloc[-1] + pd.Timedelta("5min")
        ).isoformat(),
        "parquet_path": str(parquet),
        "parquet_sha256": sha256(parquet),
        "csv_path": str(csv),
        "csv_sha256": sha256(csv),
    }


def main() -> int:
    print("=" * 136)
    print("PHASE G — CONTINUIDAD Y PREPARACIÓN DE SERIES V2")
    print("=" * 136)
    print("Reparación metodológica: conserva extremos internos y coalesce únicamente minutos de frontera duplicados.")
    print("No lee raws TAR.GZ. No ejecuta modelos.")
    print()

    check("Gobernanza 071 existe", GOV071.is_file(), GOV071)
    check("Gobernanza 071 SHA exacto", sha256(GOV071) == GOV071_SHA, sha256(GOV071))
    check("Gobernanza 072 existe", GOV072.is_file(), GOV072)
    gov072_text = GOV072.read_text(encoding="utf-8")
    check(
        "072 autoriza v2",
        "PHASE-G-CONTINUITY-SERIES-PREPARATION-V2 = AUTHORIZED FOR EXECUTION" in gov072_text,
    )

    check("Runner v1 fallido preservado", FAILED_V1_RUNNER.is_file(), FAILED_V1_RUNNER)
    check("Runner v1 fallido SHA exacto", sha256(FAILED_V1_RUNNER) == FAILED_V1_RUNNER_SHA, sha256(FAILED_V1_RUNNER))
    check("Consola v1 fallida preservada", FAILED_V1_CONSOLE.is_file(), FAILED_V1_CONSOLE)
    check("Consola v1 fallida SHA exacto", sha256(FAILED_V1_CONSOLE) == FAILED_V1_CONSOLE_SHA, sha256(FAILED_V1_CONSOLE))

    check("Audit final agregación existe", FINAL_AUDIT_MANIFEST.is_file(), FINAL_AUDIT_MANIFEST)
    check(
        "Audit final agregación SHA exacto",
        sha256(FINAL_AUDIT_MANIFEST) == FINAL_AUDIT_MANIFEST_SHA,
        sha256(FINAL_AUDIT_MANIFEST),
    )
    final_audit = read_json(FINAL_AUDIT_MANIFEST)
    check("Audit final agregación PASS", final_audit.get("status") == "PASS", final_audit.get("status"))
    check(
        "Audit final agregación 11/11",
        bool(final_audit.get("validation", {}).get("fragments_11_of_11")),
    )

    check("Output dir ausente", not OUTPUT_DIR.exists(), OUTPUT_DIR)
    check("Metrics dir v2 ausente", not METRICS_DIR.exists(), METRICS_DIR)

    audit_fragments = {x["key"]: x for x in final_audit.get("fragments", [])}
    check("11 keys exactas", list(audit_fragments) == KEYS, list(audit_fragments))

    fragment_frames: list[pd.DataFrame] = []
    fragment_ranges: list[dict[str, Any]] = []

    print()
    print("LECTURA Y VALIDACIÓN POR FRAGMENTO")
    print("-" * 136)

    for key in KEYS:
        entry = audit_fragments[key]
        audit_path = Path(entry["audit_path"])
        check(f"{key}: audit existe", audit_path.is_file(), audit_path)
        check(f"{key}: audit SHA exacto", sha256(audit_path) == entry["audit_sha256"], entry["audit_sha256"])
        audit = read_json(audit_path)
        pmeta = audit["outputs"]["parquet"]
        parquet = Path(pmeta["path"])
        check(f"{key}: parquet existe", parquet.is_file(), parquet)
        check(f"{key}: parquet bytes exactos", parquet.stat().st_size == int(pmeta["bytes"]), parquet.stat().st_size)
        check(f"{key}: parquet SHA exacto", sha256(parquet) == pmeta["sha256"], pmeta["sha256"])

        df = pd.read_parquet(parquet).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        check(f"{key}: no vacío", not df.empty, len(df))
        check(f"{key}: timestamps únicos internos", not df["timestamp"].duplicated().any())
        check(f"{key}: timestamps ordenados", df["timestamp"].is_monotonic_increasing)
        validate_regular(df, pd.Timedelta("1min"), key)

        missing = sorted((set(SUM_COLUMNS_RAW) | set(MAX_COLUMNS)) - set(df.columns))
        check(f"{key}: esquema raw completo", not missing, missing)

        fragment_ranges.append({
            "key": key,
            "rows": int(len(df)),
            "start": df["timestamp"].iloc[0].isoformat(),
            "end": df["timestamp"].iloc[-1].isoformat(),
            "parquet_path": str(parquet),
            "parquet_sha256": pmeta["sha256"],
        })

        df["_fragment_key"] = key
        fragment_frames.append(df[["timestamp"] + SUM_COLUMNS_RAW + MAX_COLUMNS + ["_fragment_key"]])

    print()
    print("TOPOLOGÍA DE FRONTERAS")
    print("-" * 136)

    for left, right in zip(fragment_ranges[:-1], fragment_ranges[1:]):
        left_end = pd.Timestamp(left["end"])
        right_start = pd.Timestamp(right["start"])
        delta = right_start - left_end
        print(f"{left['key']} -> {right['key']}: left_end={left_end} right_start={right_start} delta={delta}")

    stacked = pd.concat(fragment_frames, ignore_index=True)
    stacked = stacked.sort_values(["timestamp", "_fragment_key"], kind="mergesort").reset_index(drop=True)

    duplicate_count = int(stacked["timestamp"].duplicated(keep=False).sum())
    duplicate_timestamp_count = int(stacked["timestamp"].duplicated().sum())
    print(f"Filas que participan en timestamps duplicados: {duplicate_count}")
    print(f"Timestamps duplicados adicionales: {duplicate_timestamp_count}")

    merged, boundary_records = coalesce_boundary_minutes(stacked)

    check("Serie fusionada sin timestamps duplicados", not merged["timestamp"].duplicated().any())
    validate_regular(merged, pd.Timedelta("1min"), "Serie fusionada 1min")
    validate_reconciliation(merged, LABEL_RAW, "Serie fusionada 1min raw")

    # Solo se eliminan los extremos globales parciales, mediante alineación a 5 min.
    global_start = merged["timestamp"].iloc[0].ceil("5min")
    global_end_exclusive = (
        merged["timestamp"].iloc[-1] + pd.Timedelta("1min")
    ).floor("5min")

    check("Ventana global 5min no vacía", global_start < global_end_exclusive, f"{global_start}->{global_end_exclusive}")

    aligned = merged.loc[
        (merged["timestamp"] >= global_start)
        & (merged["timestamp"] < global_end_exclusive)
    ].reset_index(drop=True)

    expected_1min_rows = int((global_end_exclusive - global_start) / pd.Timedelta("1min"))
    check("Ventana global 1min completa", len(aligned) == expected_1min_rows, f"{len(aligned)}/{expected_1min_rows}")
    validate_regular(aligned, pd.Timedelta("1min"), "Ventana global alineada 1min")

    aligned = aligned.rename(columns={
        "flows_other_label": "flows_anomaly",
        "packets_other_label": "packets_anomaly",
        "bytes_other_label": "bytes_anomaly",
    })
    aligned = aligned[OUTPUT_COLUMNS]
    validate_reconciliation(aligned, LABEL_FINAL, "Ventana global alineada 1min")

    prepared_5min = aggregate_5min(aligned)

    print()
    print("FRONTERA TRAIN / TEST DERIVADA DE JUNE WEEK #4")
    print("-" * 136)

    ranges = {x["key"]: x for x in fragment_ranges}
    j3_end = pd.Timestamp(ranges["june_week3"]["end"])
    j4_start = pd.Timestamp(ranges["june_week4"]["start"])
    j4_end = pd.Timestamp(ranges["june_week4"]["end"])

    # Primer bin 5min completamente posterior al inicio parcial de JuneW4.
    test_start = j4_start.ceil("5min")
    # Último extremo exclusivo 5min completamente cubierto por JuneW4.
    test_end_exclusive = (j4_end + pd.Timedelta("1min")).floor("5min")

    check("Test start está en serie 5min", bool((prepared_5min["timestamp"] == test_start).any()), test_start)
    check("Test end > test start", test_end_exclusive > test_start, f"{test_start}->{test_end_exclusive}")
    check(
        "JuneW3 alcanza la vecindad inmediata de la frontera",
        j3_end >= test_start - pd.Timedelta("5min"),
        f"j3_end={j3_end} test_start={test_start}",
    )

    # No se permite que training use bins cuyo timestamp pertenezca a JuneW4.
    train_end = test_start

    windows: dict[str, Any] = {}
    for weeks in (1, 2, 4, 8):
        start = train_end - pd.Timedelta(weeks=weeks)
        subset = prepared_5min.loc[
            (prepared_5min["timestamp"] >= start)
            & (prepared_5min["timestamp"] < train_end)
        ].reset_index(drop=True)
        expected_rows = weeks * 7 * 24 * 12
        check(f"Train {weeks}w: filas exactas", len(subset) == expected_rows, f"{len(subset)}/{expected_rows}")
        validate_regular(subset, pd.Timedelta("5min"), f"Train {weeks}w")
        windows[f"train_{weeks}w"] = {
            "rows": int(len(subset)),
            "start": subset["timestamp"].iloc[0].isoformat(),
            "end": subset["timestamp"].iloc[-1].isoformat(),
            "end_exclusive": train_end.isoformat(),
        }

    test = prepared_5min.loc[
        (prepared_5min["timestamp"] >= test_start)
        & (prepared_5min["timestamp"] < test_end_exclusive)
    ].reset_index(drop=True)
    expected_test_rows = int((test_end_exclusive - test_start) / pd.Timedelta("5min"))
    check("Test JuneW4: filas exactas", len(test) == expected_test_rows, f"{len(test)}/{expected_test_rows}")
    validate_regular(test, pd.Timedelta("5min"), "Test JuneW4")

    check(
        "Historia 8w disponible sin huecos",
        pd.Timestamp(windows["train_8w"]["start"]) >= prepared_5min["timestamp"].iloc[0],
        f"{windows['train_8w']['start']} >= {prepared_5min['timestamp'].iloc[0]}",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    METRICS_DIR.mkdir(parents=True, exist_ok=False)

    outputs: dict[str, Any] = {}

    def write_pair(stem: str, frame: pd.DataFrame) -> dict[str, Any]:
        parquet = OUTPUT_DIR / f"{stem}.parquet"
        csv = OUTPUT_DIR / f"{stem}.csv"
        atomic_parquet(parquet, frame)
        atomic_csv(csv, frame)
        return {
            "rows": int(len(frame)),
            "start": frame["timestamp"].iloc[0].isoformat(),
            "end": frame["timestamp"].iloc[-1].isoformat(),
            "parquet_path": str(parquet),
            "parquet_sha256": sha256(parquet),
            "csv_path": str(csv),
            "csv_sha256": sha256(csv),
        }

    outputs["continuous_1min"] = write_pair("phase_g_continuous_prepared_1min", aligned)
    outputs["continuous_5min"] = write_pair("phase_g_continuous_prepared_5min", prepared_5min)

    for weeks in (1, 2, 4, 8):
        start = train_end - pd.Timedelta(weeks=weeks)
        subset = prepared_5min.loc[
            (prepared_5min["timestamp"] >= start)
            & (prepared_5min["timestamp"] < train_end)
        ].reset_index(drop=True)
        outputs[f"train_{weeks}w"] = write_pair(f"phase_g_train_{weeks}w_5min", subset)

    outputs["test_june_week4"] = write_pair("phase_g_test_june_week4_5min", test)

    manifest = {
        "campaign_id": "PHASE-G-CONTINUITY-SERIES-PREPARATION-V2",
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "aggregation_final_audit_manifest": str(FINAL_AUDIT_MANIFEST),
            "aggregation_final_audit_manifest_sha256": FINAL_AUDIT_MANIFEST_SHA,
            "governance_071": str(GOV071),
            "governance_071_sha256": GOV071_SHA,
            "failed_v1_runner": str(FAILED_V1_RUNNER),
            "failed_v1_runner_sha256": FAILED_V1_RUNNER_SHA,
            "failed_v1_console": str(FAILED_V1_CONSOLE),
            "failed_v1_console_sha256": FAILED_V1_CONSOLE_SHA,
            "failure_class": "INCORRECT_PER_FRAGMENT_EDGE_TRIMMING_CREATED_INTERNAL_DISCONTINUITIES",
        },
        "method_repair": {
            "old_rule_superseded": "drop first/last 1min from every fragment before concatenation",
            "new_rule": (
                "retain internal fragment edges; concatenate all audited 1min aggregates; "
                "coalesce only duplicate timestamps across adjacent fragment boundaries using "
                "SUM for additive columns and MAX for max columns; recompute derived rates; "
                "align only global outer coverage to complete 5min bins"
            ),
            "duplicate_boundary_records": boundary_records,
            "duplicate_timestamp_count": duplicate_timestamp_count,
        },
        "fragment_ranges": fragment_ranges,
        "continuous_window": {
            "start": global_start.isoformat(),
            "end_exclusive": global_end_exclusive.isoformat(),
            "rows_1min": int(len(aligned)),
            "rows_5min": int(len(prepared_5min)),
        },
        "test_boundary": {
            "june_week3_end": j3_end.isoformat(),
            "june_week4_start": j4_start.isoformat(),
            "june_week4_end": j4_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end_exclusive": test_end_exclusive.isoformat(),
        },
        "windows": windows | {
            "test_june_week4": {
                "rows": int(len(test)),
                "start": test["timestamp"].iloc[0].isoformat(),
                "end": test["timestamp"].iloc[-1].isoformat(),
                "end_exclusive": test_end_exclusive.isoformat(),
            }
        },
        "outputs": outputs,
        "checks_pass": len(checks),
        "checks_total": len(checks),
        "validation": {
            "all_fragment_hashes_reverified": True,
            "internal_boundary_duplicates_coalesced_only_if_adjacent": True,
            "global_1min_continuity_exact": True,
            "global_5min_bins_complete": True,
            "train_1_2_4_8_weeks_exact": True,
            "test_window_exact": True,
            "model_training": False,
            "data_scaling_exploratory_run": False,
            "phase_a_to_f_modified": False,
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }

    manifest_path = METRICS_DIR / "phase_g_continuity_series_preparation_v2_2026-08-25_manifest.json"
    report_path = METRICS_DIR / "phase_g_continuity_series_preparation_v2_2026-08-25_report.txt"

    atomic_json(manifest_path, manifest)

    report = "\n".join([
        "=" * 136,
        "PHASE G — CONTINUIDAD Y PREPARACIÓN DE SERIES V2",
        "=" * 136,
        "Status: PASS",
        f"Checks PASS/TOTAL: {len(checks)}/{len(checks)}",
        f"Boundary duplicate timestamps coalesced: {duplicate_timestamp_count}",
        f"Continuous 1min: {global_start.isoformat()} -> {global_end_exclusive.isoformat()} [exclusive]",
        f"Continuous 1min rows: {len(aligned)}",
        f"Continuous 5min rows: {len(prepared_5min)}",
        f"Test: {test_start.isoformat()} -> {test_end_exclusive.isoformat()} [exclusive]",
        f"Train 1w rows: {windows['train_1w']['rows']}",
        f"Train 2w rows: {windows['train_2w']['rows']}",
        f"Train 4w rows: {windows['train_4w']['rows']}",
        f"Train 8w rows: {windows['train_8w']['rows']}",
        f"Test rows: {len(test)}",
        "Model training: 0",
        "DATA-SCALING-EXPLORATORY-001: NOT RUN",
        "PHASE A-F: FROZEN",
        "",
        "PHASE-G-CONTINUITY-SERIES-PREPARATION-V2: PASS",
        "",
    ])
    atomic_text(report_path, report)

    print()
    print("=" * 136)
    print("FINAL SUMMARY")
    print("=" * 136)
    print(f"Checks PASS/TOTAL: {len(checks)}/{len(checks)}")
    print(f"Boundary duplicate timestamps coalesced: {duplicate_timestamp_count}")
    print(f"Continuous 1min: {global_start.isoformat()} -> {global_end_exclusive.isoformat()} [exclusive]")
    print(f"Test: {test_start.isoformat()} -> {test_end_exclusive.isoformat()} [exclusive]")
    print(f"Train rows 1/2/4/8w: {windows['train_1w']['rows']}/{windows['train_2w']['rows']}/{windows['train_4w']['rows']}/{windows['train_8w']['rows']}")
    print(f"Test rows: {len(test)}")
    print("Model training: 0")
    print("DATA-SCALING-EXPLORATORY-001: NOT RUN")
    print("PHASE-G-CONTINUITY-SERIES-PREPARATION-V2: PASS")
    print()
    print(manifest_path)
    print(f"SHA-256 = {sha256(manifest_path)}")
    print(report_path)
    print(f"SHA-256 = {sha256(report_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StopError, FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print()
        print("=" * 136)
        print("PHASE-G-CONTINUITY-SERIES-PREPARATION-V2: FAIL / STOP")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print("No ejecutar modelos.")
        raise SystemExit(1)
