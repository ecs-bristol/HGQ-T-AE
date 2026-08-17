#!/usr/bin/env python3
"""Four-way counterfactual test of HGQ/Trust-HGQ weights and bit maps."""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import keras
from evaluate_calibration import calibration_metrics
from src.model import get_model  # noqa: F401; registers HGQ custom classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-results",
        default="hardware_ece_da4ml/ece_hardware_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/trust_hgq_bitwidth/counterfactual",
    )
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def selected_row(path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["model"] == "Permutation-Invariant"
                and int(row["features"]) == 3
                and int(row["particles"]) == 16
            ):
                return row
    raise RuntimeError("Missing Permutation-Invariant F=3 N=16 result")


def baseline_path():
    directory = (
        ROOT / "official_models" / "3-feature-perminv"
        / "jet_classifier_large_16" / "models"
    )
    paths = sorted(directory.glob("*.keras"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one HGQ baseline in {directory}, found {len(paths)}")
    return paths[0]


def archive_entries(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        data = {info.filename: archive.read(info.filename) for info in infos}
    return infos, data


def quantizer_groups(h5):
    groups = {}

    def visitor(name, obj):
        if (
            isinstance(obj, h5py.Group)
            and name.startswith("layers/")
            and name.endswith("/quantizer/vars")
            and {"0", "1", "2"}.issubset(obj.keys())
        ):
            groups[name] = obj

    h5.visititems(visitor)
    return groups


def precision_variable(path):
    parts = path.split("/")
    roles = [part for part in parts if part in {"iq", "oq", "kq", "bq"}]
    if len(roles) != 1:
        raise RuntimeError(f"Cannot identify quantizer role from {path}")
    return "1" if roles[0] in {"kq", "bq"} else "2"


def build_hybrid(receiver, donor, output):
    infos, receiver_data = archive_entries(receiver)
    _, donor_data = archive_entries(donor)
    rbuffer = io.BytesIO(receiver_data["model.weights.h5"])
    dbuffer = io.BytesIO(donor_data["model.weights.h5"])
    transferred = []
    with h5py.File(rbuffer, "r+") as rh5, h5py.File(dbuffer, "r") as dh5:
        receiver_groups = quantizer_groups(rh5)
        donor_groups = quantizer_groups(dh5)
        if set(receiver_groups) != set(donor_groups):
            raise RuntimeError("Receiver and donor quantizer paths differ")
        for name in sorted(receiver_groups):
            variable = precision_variable(name)
            receiver_array = receiver_groups[name][variable]
            donor_array = np.asarray(donor_groups[name][variable])
            if receiver_array.shape != donor_array.shape:
                raise RuntimeError(f"Precision-map shape mismatch at {name}")
            receiver_array[...] = donor_array
            transferred.append(
                {
                    "quantizer_path": name,
                    "variable": "b" if variable == "1" else "f",
                    "shape": list(donor_array.shape),
                }
            )
        rh5.flush()
    receiver_data["model.weights.h5"] = rbuffer.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".keras.tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info in infos:
            archive.writestr(info, receiver_data[info.filename])
    temporary.replace(output)
    return transferred


def sum_stored_ebops(model):
    return int(sum(float(layer.ebops) for layer in model.layers if hasattr(layer, "ebops")))


def evaluate(label, weights_source, bit_source, model_path, x_test, y_test, bins, batch_size):
    print(f"LOAD {label}: {model_path}", flush=True)
    model = keras.models.load_model(model_path, compile=False)
    print(f"PREDICT {label}", flush=True)
    logits = model.predict(x_test, batch_size=batch_size, verbose=0)
    metrics = calibration_metrics(logits, y_test, n_bins=bins)
    row = {
        "label": label,
        "weights_source": weights_source,
        "bit_source": bit_source,
        "model_path": str(model_path),
        "accuracy": metrics["acc"],
        "ece": metrics["ece"],
        "stored_receiver_ebops": sum_stored_ebops(model),
    }
    print(
        f"RESULT {label}: accuracy={row['accuracy']:.8f} "
        f"ece={row['ece']:.8f} stored_receiver_ebops={row['stored_receiver_ebops']}",
        flush=True,
    )
    del logits, model
    gc.collect()
    keras.backend.clear_session()
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_test_data_chunked(data_path, n_constituents=16, chunk_size=8192):
    """Avoid HDF5 point selection, which is fragile on the remote NAS."""
    feature_idx = np.asarray([5, 8, 11])
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    with h5py.File(data_path / "150c-train.h5") as handle:
        dataset = handle["feature"]
        for start in range(0, dataset.shape[0], chunk_size):
            stop = min(start + chunk_size, dataset.shape[0])
            block = np.asarray(
                dataset[start:stop, :n_constituents, :], dtype=np.float32
            )[:, :, feature_idx]
            total += np.sum(block, axis=(0, 1), dtype=np.float64)
            total_sq += np.sum(block * block, axis=(0, 1), dtype=np.float64)
            count += block.shape[0] * block.shape[1]
    mean = total / count
    scale = np.sqrt(np.maximum(total_sq / count - mean * mean, 0))

    with h5py.File(data_path / "150c-test.h5") as handle:
        dataset = handle["feature"]
        x_test = np.empty(
            (dataset.shape[0], n_constituents, len(feature_idx)), dtype=np.float32
        )
        for start in range(0, dataset.shape[0], chunk_size):
            stop = min(start + chunk_size, dataset.shape[0])
            block = np.asarray(
                dataset[start:stop, :n_constituents, :], dtype=np.float32
            )
            x_test[start:stop] = block[:, :, feature_idx]
        y_test = np.asarray(handle["label"], dtype=np.int64)
    x_test = (x_test - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    return np.asarray(x_test, dtype=np.float32), y_test


def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = selected_row((ROOT / args.hardware_results).resolve())
    hgq = baseline_path().resolve()
    trust = (ROOT / result["checkpoint"]).resolve()
    if not trust.is_file():
        raise FileNotFoundError(trust)

    hybrid_dir = output_dir / "models"
    hgq_weights_trust_bits = hybrid_dir / "hgq_weights_trust_bits.keras"
    trust_weights_hgq_bits = hybrid_dir / "trust_weights_hgq_bits.keras"
    transfer_ht = build_hybrid(hgq, trust, hgq_weights_trust_bits)
    transfer_th = build_hybrid(trust, hgq, trust_weights_hgq_bits)
    manifest = {
        "definition": (
            "Only trainable precision variables are exchanged: b for KBI "
            "weight/bias quantizers and f for KIF activation quantizers. "
            "Weights and receiver integer-range variables are unchanged."
        ),
        "hgq": str(hgq),
        "trust_hgq": str(trust),
        "hgq_weights_trust_bits": str(hgq_weights_trust_bits),
        "trust_weights_hgq_bits": str(trust_weights_hgq_bits),
        "transferred_quantizers": len(transfer_ht),
        "transfers": transfer_ht,
    }
    (output_dir / "counterfactual_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"PREPARED {len(transfer_ht)} quantizer maps in each direction", flush=True)
    if args.prepare_only:
        print("PREPARE_ONLY_COMPLETE", flush=True)
        return

    x_test, y_test = load_test_data_chunked(ROOT / "dataset")
    cases = [
        ("H_weights_H_bits", "HGQ", "HGQ", hgq),
        ("H_weights_T_bits", "HGQ", "Trust-HGQ", hgq_weights_trust_bits),
        ("T_weights_H_bits", "Trust-HGQ", "HGQ", trust_weights_hgq_bits),
        ("T_weights_T_bits", "Trust-HGQ", "Trust-HGQ", trust),
    ]
    rows = [
        evaluate(label, weights, bits, path, x_test, y_test, args.bins, args.batch_size)
        for label, weights, bits, path in cases
    ]
    by_label = {row["label"]: row for row in rows}
    effects = {
        "trust_map_effect_on_hgq_weights": {
            "delta_accuracy": (
                by_label["H_weights_T_bits"]["accuracy"]
                - by_label["H_weights_H_bits"]["accuracy"]
            ),
            "delta_ece": (
                by_label["H_weights_T_bits"]["ece"]
                - by_label["H_weights_H_bits"]["ece"]
            ),
        },
        "trust_map_effect_on_trust_weights": {
            "delta_accuracy": (
                by_label["T_weights_T_bits"]["accuracy"]
                - by_label["T_weights_H_bits"]["accuracy"]
            ),
            "delta_ece": (
                by_label["T_weights_T_bits"]["ece"]
                - by_label["T_weights_H_bits"]["ece"]
            ),
        },
        "reported_reference": {
            "hgq_accuracy": float(result["hgq_accuracy"]),
            "hgq_ece": float(result["hgq_ece"]),
            "trust_accuracy": float(result["ece_accuracy"]),
            "trust_ece": float(result["ece_ece"]),
        },
    }
    write_csv(output_dir / "counterfactual_results.csv", rows)
    (output_dir / "counterfactual_results.json").write_text(
        json.dumps({"rows": rows, "effects": effects}, indent=2) + "\n"
    )
    print(f"COUNTERFACTUAL_COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
