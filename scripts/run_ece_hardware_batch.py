#!/usr/bin/env python3
"""Run DA4ML and Vivado for the 20 selected ECE-aware JEDI models.

Each configuration uses the checkpoint with the lowest validation ECE. The
already completed non-permutation-invariant F=3, N=8 smoke result is reused.
The script updates CSV and Markdown summary tables after every model.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "hardware_ece_da4ml"
LOG_ROOT = REPO / "logs/hardware_ece_da4ml/batch"
STATUS_PATH = LOG_ROOT / "status.tsv"
SELECTION_PATH = LOG_ROOT / "selected_models.tsv"
CSV_PATH = RESULT_ROOT / "ece_hardware_results.csv"
MD_PATH = RESULT_ROOT / "ece_hardware_results.md"
PYTHON = Path(sys.executable)
CLOCK_PERIOD_NS = 2.0
VERILOG_SAMPLES = 1024


@dataclass(frozen=True)
class Case:
    kind: str
    features: int
    particles: int
    config: Path
    calibration: Path
    checkpoint: Path
    calibration_row: dict
    output: Path
    reuse_existing: bool = False

    @property
    def name(self) -> str:
        return f"{self.kind}-f{self.features}-n{self.particles}"

    @property
    def epoch(self) -> int:
        match = re.search(r"epoch=(\d+)", self.checkpoint.name)
        if not match:
            raise ValueError(f"Cannot parse epoch from {self.checkpoint.name}")
        return int(match.group(1))

    @property
    def rtl_dir(self) -> Path:
        return self.output / "da4ml_verilog_prjs" / f"epoch={self.epoch}"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validation_ece(checkpoint_name: str) -> float:
    match = re.search(r"val_ece=([0-9.]+)", checkpoint_name)
    if not match:
        raise ValueError(f"Cannot parse validation ECE from {checkpoint_name}")
    return float(match.group(1))


def build_cases() -> list[Case]:
    cases: list[Case] = []
    for particles in (8, 16, 32, 64, 128):
        for features in (3, 16):
            for kind in ("nonperminv", "perminv"):
                config = REPO / f"configs/ece-grid-{kind}-n{particles}-f{features}.yaml"
                if kind == "perminv" and features == 3 and particles == 16:
                    calibration = (
                        REPO
                        / "calibration_results/diagnostic-n16-f3-perminv-lambda2-beta0-lr3e5.json"
                    )
                else:
                    calibration = (
                        REPO
                        / f"calibration_results/ece-grid-{kind}"
                        / f"ece-grid-{kind}-n{particles}-f{features}.json"
                    )

                rows = json.loads(calibration.read_text())
                selected = min(rows, key=lambda row: validation_ece(row["checkpoint_name"]))
                checkpoint = REPO / selected["checkpoint"]

                reuse = kind == "nonperminv" and features == 3 and particles == 8
                output = (
                    REPO / "hardware_smoke_da4ml/nonperminv/f3/n8"
                    if reuse
                    else RESULT_ROOT / kind / f"f{features}" / f"n{particles}"
                )
                cases.append(
                    Case(
                        kind=kind,
                        features=features,
                        particles=particles,
                        config=config,
                        calibration=calibration,
                        checkpoint=checkpoint,
                        calibration_row=selected,
                        output=output,
                        reuse_existing=reuse,
                    )
                )
    return cases


def load_baselines() -> dict[tuple[str, int, int], tuple[float, float]]:
    baselines: dict[tuple[str, int, int], tuple[float, float]] = {}
    with (REPO / "calibration_results/ece_nll.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            kind = "perminv" if "perminv" in row["config"] else "nonperminv"
            features = 3 if "-f3" in row["config"] else 16
            particles = int(row["n_constituents"])
            baselines[(kind, features, particles)] = (
                float(row["acc"]),
                float(row["ece"]),
            )

    fixed = REPO / "calibration_results/baseline_n128_f3_perminv_fixed.csv"
    with fixed.open(newline="") as handle:
        row = next(csv.DictReader(handle))
        baselines[("perminv", 3, 128)] = (float(row["acc"]), float(row["ece"]))
    return baselines


def parse_summary_value(report: Path, label: str) -> int | None:
    if not report.is_file():
        return None
    pattern = re.compile(
        rf"^\|\s*{re.escape(label)}\s*\|\s*([0-9,]+)\s*\|", re.MULTILINE
    )
    match = pattern.search(report.read_text(errors="replace"))
    return int(match.group(1).replace(",", "")) if match else None


def parse_wns(report: Path) -> float | None:
    if not report.is_file():
        return None
    lines = report.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[index + 1 : index + 6]:
            stripped = candidate.strip()
            if not stripped or not any(char.isdigit() for char in stripped):
                continue
            token = stripped.split()[0]
            try:
                return float(token)
            except ValueError:
                continue
    return None


def parse_binder(case: Case) -> tuple[int | None, int | None]:
    binder = case.rtl_dir / "jet_classifier_large_wrapper_binder.cc"
    if not binder.is_file():
        return None, None
    text = binder.read_text(errors="replace")
    ii_match = re.search(r"static const size_t II = (\d+);", text)
    latency_match = re.search(r"static const size_t latency = (\d+);", text)
    return (
        int(ii_match.group(1)) if ii_match else None,
        int(latency_match.group(1)) if latency_match else None,
    )


def parse_rtl_accuracy(case: Case) -> float | None:
    test_acc = case.output / "test_acc.json"
    if not test_acc.is_file():
        return None
    row = json.loads(test_acc.read_text()).get(case.checkpoint.name, {})
    value = row.get("verilog_acc")
    return float(value) if value is not None and float(value) >= 0 else None


def hardware_row(case: Case, baselines: dict) -> dict[str, object]:
    key = (case.kind, case.features, case.particles)
    hgq_acc, hgq_ece = baselines[key]
    ece_acc = float(case.calibration_row["acc"])
    ece_ece = float(case.calibration_row["ece"])

    report_root = case.rtl_dir / "output_jet_classifier_large/reports"
    timing = report_root / "jet_classifier_large_post_route_timing.rpt"
    util = report_root / "jet_classifier_large_post_route_util.rpt"
    wns = parse_wns(timing)
    ii, latency_cycles = parse_binder(case)
    period = CLOCK_PERIOD_NS - wns if wns is not None else None
    fmax = 1000.0 / period if period is not None and period > 0 else None
    latency_ns = (
        latency_cycles * period
        if latency_cycles is not None and period is not None
        else None
    )
    complete = timing.is_file() and util.is_file() and latency_cycles is not None

    return {
        "model": (
            "Permutation-Invariant"
            if case.kind == "perminv"
            else "Non-Permutation-Invariant"
        ),
        "features": case.features,
        "particles": case.particles,
        "epoch": case.epoch,
        "hgq_accuracy": hgq_acc,
        "ece_accuracy": ece_acc,
        "delta_accuracy": ece_acc - hgq_acc,
        "hgq_ece": hgq_ece,
        "ece_ece": ece_ece,
        "delta_ece": ece_ece - hgq_ece,
        "rtl_accuracy_1024": parse_rtl_accuracy(case),
        "setup_wns_ns": wns,
        "period_approx_ns": period,
        "latency_cycles": latency_cycles,
        "latency_ns": latency_ns,
        "ii": ii,
        "fmax_mhz": fmax,
        "lut": parse_summary_value(util, "CLB LUTs"),
        "ff": parse_summary_value(util, "CLB Registers"),
        "dsp": parse_summary_value(util, "DSPs"),
        "bram": parse_summary_value(util, "Block RAM Tile"),
        "uram": parse_summary_value(util, "URAM"),
        "status": "completed" if complete else "pending",
        "checkpoint": str(case.checkpoint.relative_to(REPO)),
        "output": str(case.output.relative_to(REPO)),
    }


def format_optional(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_tables(cases: list[Case], baselines: dict) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [hardware_row(case, baselines) for case in cases]
    fieldnames = list(rows[0])
    temporary = CSV_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, CSV_PATH)

    lines = [
        "# ECE-aware hardware results",
        "",
        (
            "| Model | F | N | Epoch | Accuracy: HGQ → ECE-aware | ΔAcc (pp) | "
            "ECE: HGQ → ECE-aware | ΔECE | RTL Acc. (1024) | WNS (ns) | "
            "Latency (cycles) | Latency (ns) | LUT | FF | DSP | BRAM | URAM | "
            "II | Fmax (MHz) |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in rows:
        accuracy = (
            f"{100 * row['hgq_accuracy']:.3f}% → {100 * row['ece_accuracy']:.3f}%"
        )
        ece = f"{row['hgq_ece']:.5f} → {row['ece_ece']:.5f}"
        model = (
            "Perm-Inv"
            if row["model"] == "Permutation-Invariant"
            else "Non-Perm-Inv"
        )
        values = [
            model,
            str(row["features"]),
            str(row["particles"]),
            str(row["epoch"]),
            accuracy,
            f"{100 * row['delta_accuracy']:+.3f}",
            ece,
            f"{row['delta_ece']:+.5f}",
            (
                f"{100 * row['rtl_accuracy_1024']:.3f}%"
                if row["rtl_accuracy_1024"] is not None
                else ""
            ),
            format_optional(row["setup_wns_ns"], 3),
            format_optional(row["latency_cycles"], 0),
            format_optional(row["latency_ns"], 3),
            format_optional(row["lut"], 0),
            format_optional(row["ff"], 0),
            format_optional(row["dsp"], 0),
            format_optional(row["bram"], 0),
            format_optional(row["uram"], 0),
            format_optional(row["ii"], 0),
            format_optional(row["fmax_mhz"], 3),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            (
                "Approximate period is calculated as 2.000 ns - setup WNS; "
                "latency in ns is latency_cycles × approximate_period."
            ),
            "",
        ]
    )
    temporary_md = MD_PATH.with_suffix(".md.tmp")
    temporary_md.write_text("\n".join(lines))
    os.replace(temporary_md, MD_PATH)


def append_status(case: Case, phase: str, exit_code: int, elapsed: float) -> None:
    with STATUS_PATH.open("a") as handle:
        handle.write(
            f"{now()}\t{case.name}\t{phase}\t{exit_code}\t{elapsed:.1f}\t"
            f"{case.output}\n"
        )


def run_logged(
    command: list[str], cwd: Path, log_path: Path, timeout_text: str
) -> tuple[int, float]:
    wrapped = ["timeout", "--signal=TERM", "--kill-after=10s", timeout_text, *command]
    start = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"\n[{now()}] COMMAND {' '.join(wrapped)}\n")
        log.flush()
        result = subprocess.run(
            wrapped,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
        elapsed = time.monotonic() - start
        log.write(f"[{now()}] EXIT {result.returncode} ELAPSED {elapsed:.1f}s\n")
    return result.returncode, elapsed


def case_complete(case: Case) -> bool:
    reports = case.rtl_dir / "output_jet_classifier_large/reports"
    return (
        (case.output / "test_acc.json").is_file()
        and (case.rtl_dir / "jet_classifier_large_wrapper_binder.cc").is_file()
        and (reports / "jet_classifier_large_post_route_timing.rpt").is_file()
        and (reports / "jet_classifier_large_post_route_util.rpt").is_file()
    )


def write_selection(cases: list[Case]) -> None:
    with SELECTION_PATH.open("w") as handle:
        handle.write(
            "model\tfeatures\tparticles\tepoch\tconfig\tcheckpoint\tcalibration\toutput\n"
        )
        for case in cases:
            handle.write(
                f"{case.kind}\t{case.features}\t{case.particles}\t{case.epoch}\t"
                f"{case.config}\t{case.checkpoint}\t{case.calibration}\t"
                f"{case.output}\n"
            )


def main() -> int:
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TMPDIR", str(REPO / ".tmp-hgq-ece"))
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.environ["PATH"] = f"{PYTHON.parent}:{os.environ.get('PATH', '')}"
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    baselines = load_baselines()
    write_selection(cases)
    STATUS_PATH.write_text(
        "timestamp\tcase\tphase\texit_code\telapsed_seconds\toutput\n"
    )
    write_tables(cases, baselines)

    print(f"[{now()}] ECE_HARDWARE_BATCH_START cases={len(cases)}", flush=True)
    failed = 0
    for index, case in enumerate(cases, start=1):
        print(f"[{now()}] CASE {index}/{len(cases)} START {case.name}", flush=True)
        if case_complete(case):
            print(
                f"[{now()}] CASE {case.name} REUSE completed output={case.output}",
                flush=True,
            )
            append_status(case, "reuse", 0, 0.0)
            write_tables(cases, baselines)
            continue

        case.output.mkdir(parents=True, exist_ok=True)
        da4ml_log = LOG_ROOT / f"{case.name}.da4ml.log"
        vivado_log = LOG_ROOT / f"{case.name}.vivado.log"
        command = [
            str(PYTHON),
            str(REPO / "scripts/run_selected_ece_hardware.py"),
            "--config",
            str(case.config),
            "--checkpoint",
            str(case.checkpoint),
            "--output",
            str(case.output),
            "--verilog-samples",
            str(VERILOG_SAMPLES),
        ]
        exit_code, elapsed = run_logged(command, REPO, da4ml_log, "6h")
        append_status(case, "da4ml", exit_code, elapsed)
        if exit_code != 0:
            failed += 1
            print(
                f"[{now()}] CASE {case.name} DA4ML_FAILED exit={exit_code}",
                flush=True,
            )
            write_tables(cases, baselines)
            continue

        build_tcl = case.rtl_dir / "build_prj.tcl"
        if not build_tcl.is_file():
            failed += 1
            append_status(case, "missing_build_tcl", 2, 0.0)
            print(f"[{now()}] CASE {case.name} MISSING {build_tcl}", flush=True)
            write_tables(cases, baselines)
            continue

        command = [
            "vivado",
            "-mode",
            "batch",
            "-source",
            "build_prj.tcl",
            "-log",
            "vivado_batch.log",
            "-journal",
            "vivado_batch.jou",
        ]
        exit_code, elapsed = run_logged(command, case.rtl_dir, vivado_log, "12h")
        append_status(case, "vivado", exit_code, elapsed)
        if exit_code != 0 or not case_complete(case):
            failed += 1
            print(
                f"[{now()}] CASE {case.name} VIVADO_FAILED exit={exit_code}",
                flush=True,
            )
        else:
            print(f"[{now()}] CASE {case.name} COMPLETE", flush=True)
        write_tables(cases, baselines)

    print(f"[{now()}] ECE_HARDWARE_BATCH_END failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
