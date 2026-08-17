#!/usr/bin/env python3
"""Run the existing JEDI-linear test and DA4ML flow for one selected checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import keras
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader import get_data
from src.model import get_model
from src.syn_test import syn_test_verilog
from src.test import test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verilog-samples",
        type=int,
        default=None,
        help="Number of test samples for RTL simulation; default is the full test set.",
    )
    return parser.parse_args()


def prepare_selected_model(
    model,
    checkpoint_path: Path,
    output_path: Path,
    x_train,
    x_test,
    y_test,
) -> None:
    """Prepare the selected model, accepting weight-only or full-model checkpoints."""
    try:
        test(model, output_path, x_train, x_test, y_test)
        return
    except ValueError as load_weights_error:
        print("Weight-only load failed; trying the checkpoint as a full Keras model")
        try:
            model = keras.models.load_model(checkpoint_path, compile=False)
        except Exception:
            raise load_weights_error

    pred = model.predict(x_test, batch_size=16384, verbose=0)
    acc = float(np.mean(np.argmax(pred, axis=1) == np.asarray(y_test).ravel()))
    ebops = sum(
        float(layer.ebops) for layer in model.layers if hasattr(layer, "ebops")
    )
    model_dir = output_path / "models"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / checkpoint_path.name)
    with (output_path / "test_acc.json").open("w") as handle:
        json.dump({checkpoint_path.name: {"acc": acc, "ebops": ebops}}, handle)
    print(f"Test accuracy: {acc:.5%} @ {ebops:.0f} EBOPs (full-model checkpoint)")


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
    checkpoint_dir = output_path / "ckpts"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_link = checkpoint_dir / checkpoint_path.name
    if checkpoint_link.exists() or checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != checkpoint_path:
            raise RuntimeError(f"Unexpected existing checkpoint link: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(checkpoint_path)

    manifest_path = output_path / "hardware_manifest.json"
    manifest = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "verilog_samples": args.verilog_samples,
        "part": "xcvu13p-flga2577-2-e",
        "clock_period_ns": 2.0,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError(f"Existing manifest does not match: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    conf = OmegaConf.load(config_path)
    conf.save_path = str(output_path)

    random.seed(int(conf.seed))
    np.random.seed(int(conf.seed))
    print(f"Loading data for n={conf.n_constituents}, pt_eta_phi={conf.pt_eta_phi}")
    x_train, x_test, y_train, y_test = get_data(
        Path(conf.datapath),
        int(conf.n_constituents),
        ptetaphi=bool(conf.pt_eta_phi),
    )

    print("Creating HGQ model")
    model_hgq = get_model(conf)

    print(f"Testing selected checkpoint: {checkpoint_path.name}")
    prepare_selected_model(
        model_hgq, checkpoint_path, output_path, x_train, x_test, y_test
    )

    print("Generating and simulating DA4ML RTL")
    syn_test_verilog(
        output_path,
        x_test,
        y_test,
        N=args.verilog_samples,
    )
    print("SELECTED_ECE_HARDWARE_COMPLETE")


if __name__ == "__main__":
    main()
