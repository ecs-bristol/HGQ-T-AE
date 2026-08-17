#!/usr/bin/env python3
"""Run pending ECE-aware Vivado implementations with five independent workers.

Existing RTL is submitted first. Missing RTL is generated with VerilogModel.write()
only; Verilator compilation and RTL simulation are intentionally skipped.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_ece_hardware_batch import (
    LOG_ROOT,
    PYTHON,
    REPO,
    build_cases,
    case_complete,
    load_baselines,
    now,
    run_logged,
    write_selection,
    write_tables,
)

MAX_VIVADO_WORKERS = 5
STATUS_PATH = LOG_ROOT / "status-vivado-only-parallel5.tsv"
STATUS_LOCK = threading.Lock()
TABLE_LOCK = threading.Lock()


def append_status(case, phase: str, exit_code: int, elapsed: float) -> None:
    with STATUS_LOCK:
        with STATUS_PATH.open("a") as handle:
            handle.write(
                f"{now()}\t{case.name}\t{phase}\t{exit_code}\t{elapsed:.1f}\t"
                f"{case.output}\n"
            )


def refresh_tables(cases, baselines) -> None:
    with TABLE_LOCK:
        write_tables(cases, baselines)


def rtl_ready(case) -> bool:
    required = [
        case.rtl_dir / "build_prj.tcl",
        case.rtl_dir / "jet_classifier_large.v",
        case.rtl_dir / "jet_classifier_large_wrapper.v",
        case.rtl_dir / "jet_classifier_large_wrapper_binder.cc",
        case.rtl_dir / "jet_classifier_large.xdc",
    ]
    return all(path.is_file() for path in required)


def run_vivado(case, cases, baselines):
    log_path = LOG_ROOT / f"{case.name}.vivado.log"
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
    print(f"[{now()}] VIVADO_START {case.name}", flush=True)
    exit_code, elapsed = run_logged(command, case.rtl_dir, log_path, "12h")
    complete = exit_code == 0 and case_complete(case)
    effective_code = exit_code if exit_code != 0 else (0 if complete else 2)
    append_status(case, "vivado", effective_code, elapsed)
    refresh_tables(cases, baselines)
    state = "COMPLETE" if complete else "FAILED"
    print(
        f"[{now()}] VIVADO_{state} {case.name} "
        f"exit={effective_code} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return complete


def submit_vivado(executor, futures, case, cases, baselines) -> None:
    future = executor.submit(run_vivado, case, cases, baselines)
    futures[future] = case
    print(
        f"[{now()}] VIVADO_QUEUED {case.name} queued_total={len(futures)}",
        flush=True,
    )


def main() -> int:
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TMPDIR", str(REPO / ".tmp-hgq-ece"))
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.environ["PATH"] = f"{PYTHON.parent}:{os.environ.get('PATH', '')}"

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    baselines = load_baselines()
    write_selection(cases)
    STATUS_PATH.write_text(
        "timestamp\tcase\tphase\texit_code\telapsed_seconds\toutput\n"
    )
    refresh_tables(cases, baselines)

    completed = [case for case in cases if case_complete(case)]
    pending = [case for case in cases if not case_complete(case)]
    existing_rtl = [case for case in pending if rtl_ready(case)]
    missing_rtl = [case for case in pending if not rtl_ready(case)]
    print(
        f"[{now()}] VIVADO_ONLY_BATCH_START completed={len(completed)} "
        f"pending={len(pending)} existing_rtl={len(existing_rtl)} "
        f"missing_rtl={len(missing_rtl)} vivado_workers={MAX_VIVADO_WORKERS}",
        flush=True,
    )

    for case in completed:
        append_status(case, "reuse", 0, 0.0)

    futures = {}
    failed_generation = 0
    executor = ThreadPoolExecutor(
        max_workers=MAX_VIVADO_WORKERS,
        thread_name_prefix="vivado",
    )
    try:
        # Submit all already-generated RTL first so five implementations start now.
        for case in existing_rtl:
            append_status(case, "reuse_rtl", 0, 0.0)
            submit_vivado(executor, futures, case, cases, baselines)

        # Generate only the missing RTL. This command calls write(), never compile().
        for case in missing_rtl:
            print(f"[{now()}] RTL_GENERATE_START {case.name}", flush=True)
            log_path = LOG_ROOT / f"{case.name}.rtlgen.log"
            command = [
                str(PYTHON),
                str(REPO / "scripts/generate_selected_ece_rtl.py"),
                "--config",
                str(case.config),
                "--checkpoint",
                str(case.checkpoint),
                "--output",
                str(case.output),
            ]
            exit_code, elapsed = run_logged(command, REPO, log_path, "2h")
            ready = exit_code == 0 and rtl_ready(case)
            effective_code = exit_code if exit_code != 0 else (0 if ready else 2)
            append_status(case, "rtl_generate_no_verilator", effective_code, elapsed)
            refresh_tables(cases, baselines)
            if not ready:
                failed_generation += 1
                print(
                    f"[{now()}] RTL_GENERATE_FAILED {case.name} "
                    f"exit={effective_code} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                continue
            print(
                f"[{now()}] RTL_GENERATE_COMPLETE {case.name} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            submit_vivado(executor, futures, case, cases, baselines)

        failed_vivado = 0
        for future in as_completed(futures):
            case = futures[future]
            try:
                if not future.result():
                    failed_vivado += 1
            except Exception as exc:
                failed_vivado += 1
                append_status(case, "vivado_exception", 3, 0.0)
                print(
                    f"[{now()}] VIVADO_EXCEPTION {case.name} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    refresh_tables(cases, baselines)
    failed = failed_generation + failed_vivado
    print(
        f"[{now()}] VIVADO_ONLY_BATCH_END submitted={len(futures)} "
        f"failed_generation={failed_generation} failed_vivado={failed_vivado}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
