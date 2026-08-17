#!/usr/bin/env python3
"""Generate DA4ML Verilog/RTL for one selected checkpoint without Verilator."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import keras
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from da4ml.codegen import VerilogModel
from da4ml.converter.hgq2.parser import trace_model
from da4ml.trace import HWConfig, comb_trace
from src.model import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_selected_model(config_path: Path, checkpoint_path: Path):
    try:
        model = keras.models.load_model(checkpoint_path, compile=False)
        print("Loaded full-model checkpoint", flush=True)
        return model
    except Exception as full_model_error:
        print("Full-model load failed; trying weight-only checkpoint", flush=True)
        conf = OmegaConf.load(config_path)
        model = get_model(conf)
        try:
            model.load_weights(checkpoint_path)
        except Exception:
            raise full_model_error
        return model


def update_test_acc(output_path: Path, checkpoint_path: Path) -> None:
    path = output_path / "test_acc.json"
    if path.is_file():
        results = json.loads(path.read_text())
    else:
        results = {}
    row = results.setdefault(checkpoint_path.name, {})
    acc_match = re.search(r"(?:^|-)acc=([0-9.]+)%", checkpoint_path.name)
    ebops_match = re.search(r"(?:^|-)EBOPs=([0-9.]+)", checkpoint_path.name)
    if acc_match:
        row.setdefault("acc", float(acc_match.group(1)) / 100.0)
    if ebops_match:
        row.setdefault("ebops", float(ebops_match.group(1)))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results))
    temporary.replace(path)


def normalise_vivado_ooc_script(rtl_path: Path) -> None:
    """Match the original JEDI-linear OOC implementation flow.

    The DA4ML simulation wrapper exposes a very wide uniform port and must not
    be the implementation top. Vivado implementation uses the DA core directly.
    """
    path = rtl_path / "build_prj.tcl"
    source_lines = path.read_text().splitlines()
    output_lines = []
    for line in source_lines:
        if line == 'set top_module "${project_name}_wrapper"':
            output_lines.append('set top_module "${project_name}"')
        elif line == 'read_verilog "${project_name}_wrapper.v"':
            continue
        elif "flatten_hierarchy" in line:
            line = line.replace("flatten_hierarchy rebuilt", "flatten_hierarchy full")
            if line.rstrip().endswith(chr(92)):
                line = line.rstrip()[:-1].rstrip()
            output_lines.append(line)
        elif "directive AlternateRoutability" in line:
            continue
        elif line == "opt_design -directive ExploreSequentialArea":
            continue
        elif line == "place_design -directive AltSpreadLogic_high -fanout_opt":
            output_lines.append("place_design -directive SSI_HighUtilSLRs -fanout_opt")
        else:
            output_lines.append(line)
    text = "\n".join(output_lines) + "\n"
    required = [
        'set top_module "${project_name}"',
        'read_xdc "${project_name}.xdc" -mode out_of_context',
        'synth_design -top $top_module -mode out_of_context',
        'place_design -directive SSI_HighUtilSLRs',
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"OOC Tcl normalization incomplete: {missing}")
    temporary = path.with_suffix(".tcl.tmp")
    temporary.write_text(text)
    temporary.replace(path)

def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_path.mkdir(parents=True, exist_ok=True)
    model = load_selected_model(config_path, checkpoint_path)

    epoch_match = re.search(r"epoch=(\d+)", checkpoint_path.name)
    if not epoch_match:
        raise ValueError(f"Cannot parse epoch from {checkpoint_path.name}")
    epoch = epoch_match.group(1)
    rtl_path = output_path / "da4ml_verilog_prjs" / f"epoch={epoch}"

    print(f"Tracing model and writing RTL to {rtl_path}", flush=True)
    inp, out = trace_model(
        model,
        solver_options={"hard_dc": 2},
        hwconf=HWConfig(1, -1, -1),
    )
    solution = comb_trace(inp, out)
    verilog_model = VerilogModel(
        solution,
        prj_name="jet_classifier_large",
        path=rtl_path,
        part_name="xcvu13p-flga2577-2-e",
        clock_period=2,
        clock_uncertainty=0.0,
        latency_cutoff=2,
    )
    verilog_model.write()
    normalise_vivado_ooc_script(rtl_path)
    update_test_acc(output_path, checkpoint_path)

    required = [
        rtl_path / "build_prj.tcl",
        rtl_path / "jet_classifier_large.v",
        rtl_path / "jet_classifier_large_wrapper.v",
        rtl_path / "jet_classifier_large_wrapper_binder.cc",
        rtl_path / "jet_classifier_large.xdc",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"RTL generation incomplete; missing: {missing}")
    print("RTL_GENERATION_COMPLETE (Verilator skipped)", flush=True)


if __name__ == "__main__":
    main()
