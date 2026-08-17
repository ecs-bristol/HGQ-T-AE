from __future__ import annotations

import argparse
import csv
import gc
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import keras
import numpy as np
import yaml
from hgq.utils import trace_minmax

from src.model import get_model


def _coerce(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _coerce(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if any(c in stripped for c in ".eE") else int(stripped)
        except ValueError:
            return stripped
    return value


def load_config(path: Path):
    with path.open() as f:
        data = yaml.safe_load(f)
    data["save_path"] = str(data["save_path"]).strip()
    data["config_path"] = str(path)
    return _coerce(data)


def load_data(data_path: Path, n_constituents: int, pt_eta_phi: bool):
    feature_idx = [5, 8, 11] if pt_eta_phi else slice(None)
    with h5py.File(data_path / "150c-train.h5") as f:
        x_train = np.array(f["feature"][:, :n_constituents, feature_idx]).astype(
            np.float32
        )
    with h5py.File(data_path / "150c-test.h5") as f:
        x_test = np.array(f["feature"][:, :n_constituents, feature_idx]).astype(
            np.float32
        )
        y_test = np.array(f["label"]).astype(np.int64)

    scale = np.std(x_train, axis=(0, 1), keepdims=True)
    shift = np.mean(x_train, axis=(0, 1), keepdims=True)
    x_train = (x_train - shift) / scale
    x_test = (x_test - shift) / scale
    return x_train, x_test, y_test


def softmax(logits: np.ndarray):
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, n_bins: int):
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = pred == labels

    bin_ids = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    bins = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(np.sum(mask))
        if count == 0:
            bins.append(
                {
                    "bin": bin_id,
                    "count": 0,
                    "accuracy": None,
                    "confidence": None,
                    "gap": None,
                }
            )
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        gap = abs(acc_bin - conf_bin)
        ece += count / labels.shape[0] * gap
        bins.append(
            {
                "bin": bin_id,
                "count": count,
                "accuracy": acc_bin,
                "confidence": conf_bin,
                "gap": gap,
            }
        )

    return {
        "acc": float(np.mean(correct)),
        "ece": float(ece),
        "bins": bins,
    }


def evaluate_config(conf, x_train, x_test, y_test, n_bins: int, batch_size: int):
    save_path = Path(conf.save_path)
    ckpts = sorted((save_path / "ckpts").glob("*.keras"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {save_path / 'ckpts'}")

    rows = []
    for ckpt in ckpts:
        model = get_model(conf)
        load_mode = "strict"
        try:
            model.load_weights(ckpt)
        except ValueError:
            model_path = save_path / "models" / ckpt.name
            model_candidates = (
                (model_path, "saved_model"),
                (ckpt, "checkpoint_model"),
            )
            for candidate, candidate_mode in model_candidates:
                if not candidate.exists():
                    continue
                try:
                    model = keras.models.load_model(candidate, compile=False)
                except Exception:
                    continue
                load_mode = candidate_mode
                break
            else:
                load_mode = "skip_mismatch"
                model = get_model(conf)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.load_weights(ckpt, skip_mismatch=True)
        if load_mode not in {"saved_model", "checkpoint_model"}:
            trace_minmax(model, x_train, batch_size=batch_size)
        logits = model.predict(x_test, batch_size=batch_size, verbose=0)
        metrics = calibration_metrics(logits, y_test, n_bins=n_bins)
        ebops = sum(float(layer.ebops) for layer in model.layers if hasattr(layer, "ebops"))
        rows.append(
            {
                "config": conf.config_path,
                "save_path": conf.save_path,
                "checkpoint": str(ckpt),
                "checkpoint_name": ckpt.name,
                "n_constituents": int(conf.n_constituents),
                "pt_eta_phi": bool(conf.pt_eta_phi),
                "model_class": str(conf.model_class),
                "load_mode": load_mode,
                "ebops": ebops,
                "n_bins": n_bins,
                **metrics,
            }
        )
    return rows


def write_outputs(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(rows, f, indent=2)

    csv_path = output.with_suffix(".csv")
    fieldnames = [
        "config",
        "save_path",
        "checkpoint_name",
        "n_constituents",
        "pt_eta_phi",
        "model_class",
        "load_mode",
        "ebops",
        "n_bins",
        "acc",
        "ece",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--data-path", default="dataset")
    parser.add_argument("--output", default="calibration_results/ece.json")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16384)
    return parser.parse_args()


def main():
    args = parse_args()
    config_paths = (
        [Path(p) for p in args.configs]
        if args.configs
        else sorted(Path("configs").glob("*.yaml"))
    )
    configs = [load_config(path) for path in config_paths]
    configs.sort(key=lambda c: (bool(c.pt_eta_phi), int(c.n_constituents), str(c.model_class)))

    rows = []
    for key in sorted({(int(c.n_constituents), bool(c.pt_eta_phi)) for c in configs}):
        n_constituents, pt_eta_phi = key
        print(f"Loading data: n={n_constituents}, pt_eta_phi={pt_eta_phi}", flush=True)
        x_train, x_test, y_test = load_data(Path(args.data_path), n_constituents, pt_eta_phi)
        for conf in [
            c
            for c in configs
            if int(c.n_constituents) == n_constituents
            and bool(c.pt_eta_phi) == pt_eta_phi
        ]:
            print(f"Evaluating {conf.config_path}", flush=True)
            config_rows = evaluate_config(
                conf, x_train, x_test, y_test, args.bins, args.batch_size
            )
            rows.extend(config_rows)
            for row in config_rows:
                print(
                    f"{row['checkpoint_name']}: "
                    f"acc={row['acc']:.6f}, ece={row['ece']:.6f}, "
                    f"ebops={row['ebops']:.0f}",
                    flush=True,
                )
            keras.backend.clear_session()
            gc.collect()
        del x_train, x_test, y_test
        gc.collect()

    write_outputs(rows, Path(args.output))
    print(f"Wrote {args.output} and {Path(args.output).with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
