#!/usr/bin/env python3
"""Compare exact deployed HGQ and Trust-HGQ bit-width allocations.

The Keras v3 archives are read directly. This avoids retracing or otherwise
mutating quantizer state, and therefore analyzes the same integer formats and
per-layer EBOP counters used for the reported models.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAYER_LABEL = {
    "q_einsum_dense_batchnorm": "Embedding",
    "q_sum": "Global pool",
    "q_einsum_dense_batchnorm_1": "Self message",
    "q_einsum_dense_batchnorm_2": "Global message",
    "q_add": "Message merge",
    "q_einsum_dense_batchnorm_3": "Update",
    "q_sum_1": "Readout pool",
    "q_einsum_dense_batchnorm_4": "FC64",
    "q_einsum_dense_batchnorm_5": "FC32",
    "q_einsum_dense_batchnorm_6": "FC16",
    "q_einsum_dense_batchnorm_7": "Logits",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-results",
        default="hardware_ece_da4ml/ece_hardware_results.csv",
    )
    parser.add_argument("--output-dir", default="analysis/trust_hgq_bitwidth")
    parser.add_argument("--representative-features", type=int, default=3)
    parser.add_argument("--representative-particles", type=int, default=16)
    parser.add_argument("--features", type=int, nargs="*", default=[3, 16])
    parser.add_argument("--particles", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    return parser.parse_args()


def read_selection(path, features, particles):
    with path.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        f, n = int(row["features"]), int(row["particles"])
        if row["model"] == "Permutation-Invariant" and f in features and n in particles:
            row = dict(row)
            row["features"], row["particles"] = f, n
            rows.append(row)
    rows.sort(key=lambda row: (row["features"], row["particles"]))
    expected = len(features) * len(particles)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} models, found {len(rows)}")
    return rows


def baseline_model(features, particles):
    directory = (
        ROOT / "official_models" / f"{features}-feature-perminv"
        / f"jet_classifier_large_{particles}" / "models"
    )
    paths = sorted(directory.glob("*.keras"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one deployed HGQ model in {directory}, found {len(paths)}")
    return paths[0]


def rounded(array):
    return np.rint(array).astype(np.int64)


def q_category(role):
    return {
        "kq": "weight", "bq": "bias", "iq": "activation", "oq": "activation"
    }[role]


class Archive:
    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))
            weights = archive.read("model.weights.h5")
        self.order = {
            layer["config"]["name"]: index
            for index, layer in enumerate(config["config"]["layers"])
            if layer["class_name"] != "InputLayer"
        }
        self.buffer = io.BytesIO(weights)
        self.h5 = h5py.File(self.buffer, "r")

    def close(self):
        self.h5.close()
        self.buffer.close()

    def quantizers(self):
        result = {}

        def visitor(name, obj):
            if not isinstance(obj, h5py.Group):
                return
            if not name.startswith("layers/") or not name.endswith("/quantizer/vars"):
                return
            if not {"0", "1", "2"}.issubset(obj.keys()):
                return
            parts = name.split("/")
            layer = parts[1]
            roles = [part for part in parts[2:] if part in {"iq", "oq", "kq", "bq"}]
            if len(roles) != 1:
                raise RuntimeError(f"Cannot identify quantizer role from {name}")
            role = roles[0]
            qtype = "kbi" if role in {"kq", "bq"} else "kif"
            k = rounded(np.asarray(obj["0"]))
            first, second = rounded(np.asarray(obj["1"])), rounded(np.asarray(obj["2"]))
            if qtype == "kbi":
                b, i = first, second
                f = b - i
                allocated = b
            else:
                i, f = first, second
                b = i + f
                allocated = f
            result[name] = {
                "layer": layer,
                "logical_layer": LAYER_LABEL.get(layer, layer),
                "layer_order": self.order[layer],
                "role": role,
                "category": q_category(role),
                "qtype": qtype,
                "shape": tuple(b.shape),
                "k": k,
                "i": i,
                "f": f,
                "allocated": allocated,
                "effective": np.maximum(0, b),
                "total_format": np.maximum(0, k + i + f),
            }

        self.h5.visititems(visitor)
        return result

    def ebops(self):
        result = {}
        for name, layer in self.h5["layers"].items():
            if "vars" not in layer:
                continue
            candidates = []
            for dataset in layer["vars"].values():
                value = np.asarray(dataset)
                if value.shape != ():
                    continue
                scalar = float(value)
                if scalar >= 1 and math.isclose(scalar, round(scalar), abs_tol=1e-6):
                    candidates.append(int(round(scalar)))
            if candidates:
                result[name] = max(candidates)
        return result


def paired_archives(hgq_path, trust_path):
    ha, ta = Archive(hgq_path), Archive(trust_path)
    try:
        hgq, trust = ha.quantizers(), ta.quantizers()
        if set(hgq) != set(trust):
            raise RuntimeError(
                f"Quantizer mismatch; only HGQ={sorted(set(hgq)-set(trust))}, "
                f"only Trust-HGQ={sorted(set(trust)-set(hgq))}"
            )
        for name in hgq:
            if hgq[name]["shape"] != trust[name]["shape"]:
                raise RuntimeError(
                    f"Shape mismatch at {name}: {hgq[name]['shape']} vs {trust[name]['shape']}"
                )
        return hgq, trust, ha.ebops(), ta.ebops()
    finally:
        ha.close()
        ta.close()


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write empty output {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(rows, category=None):
    chosen = [row for row in rows if category is None or row["category"] == category]
    delta = np.asarray([row["delta_effective_bits"] for row in chosen], dtype=float)
    hgq = np.asarray([row["hgq_effective_bits"] for row in chosen], dtype=float)
    trust = np.asarray([row["trust_effective_bits"] for row in chosen], dtype=float)
    return {
        "count": len(chosen),
        "fraction_increased": float(np.mean(delta > 0)),
        "fraction_unchanged": float(np.mean(delta == 0)),
        "fraction_decreased": float(np.mean(delta < 0)),
        "mean_hgq_bits": float(np.mean(hgq)),
        "mean_trust_bits": float(np.mean(trust)),
        "mean_delta_bits": float(np.mean(delta)),
        "mean_abs_delta_bits": float(np.mean(np.abs(delta))),
    }


def transition_panel(fig, axis, rows, category, title):
    chosen = [row for row in rows if row["category"] == category]
    hgq = np.asarray([row["hgq_effective_bits"] for row in chosen], dtype=int)
    trust = np.asarray([row["trust_effective_bits"] for row in chosen], dtype=int)
    low, high = int(min(hgq.min(), trust.min())), int(max(hgq.max(), trust.max()))
    ticks = np.arange(low, high + 1)
    matrix = np.zeros((len(ticks), len(ticks)))
    for hbit, tbit in zip(hgq, trust):
        matrix[tbit - low, hbit - low] += 1
    matrix *= 100 / matrix.sum()
    nonzero = matrix[matrix > 0]
    image = axis.imshow(
        matrix, origin="lower", cmap="Blues", aspect="equal",
        norm=LogNorm(vmin=max(float(nonzero.min()), 0.01), vmax=float(nonzero.max())),
    )
    axis.plot(
        [-0.5, len(ticks)-0.5], [-0.5, len(ticks)-0.5],
        color="black", lw=1, alpha=0.6,
    )
    axis.set_xticks(range(len(ticks)), labels=ticks)
    axis.set_yticks(range(len(ticks)), labels=ticks)
    axis.set_xlabel("HGQ effective bit-width")
    axis.set_ylabel("Trust-HGQ effective bit-width")
    axis.set_title(title)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] >= 0.5:
                color = "white" if matrix[y, x] > 0.35 * matrix.max() else "black"
                axis.text(
                    x, y, f"{matrix[y,x]:.1f}",
                    ha="center", va="center", fontsize=7, color=color,
                )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.72, pad=0.02)
    colorbar.set_label("Allocation units (%)")


def make_figure(bit_rows, layer_rows, features, particles, output_dir):
    bits = [
        row for row in bit_rows
        if row["features"] == features and row["particles"] == particles
    ]
    layers = [
        row for row in layer_rows
        if row["features"] == features and row["particles"] == particles
    ]
    if not bits or not layers:
        raise RuntimeError("Representative model is missing")

    fig = plt.figure(figsize=(15.2, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.55))
    transition_panel(fig, fig.add_subplot(grid[:, 0]), bits, "weight", "(a) Weights migration")
    transition_panel(
        fig, fig.add_subplot(grid[:, 1]), bits, "activation", "(b) Activations migration"
    )

    ordered = sorted({(row["layer_order"], row["logical_layer"]) for row in layers})
    orders, names = [x[0] for x in ordered], [x[1] for x in ordered]
    lookup = {(row["layer_order"], row["category"]): row for row in layers}
    delta = np.full((2, len(orders)), np.nan)
    for x, order in enumerate(orders):
        for category, y in {"weight": 0, "activation": 1}.items():
            row = lookup.get((order, category))
            if row:
                delta[y, x] = row["mean_delta_bits"]

    axis = fig.add_subplot(grid[0, 2])
    limit = max(float(np.abs(delta[np.isfinite(delta)]).max()), 0.25)
    image = axis.imshow(
        delta, cmap="RdBu_r", aspect="auto",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    axis.set_xticks(range(len(names)), labels=names, rotation=40, ha="right")
    axis.set_yticks([0, 1], labels=["Weights", "Activations"])
    axis.set_title("(c) Layer-wise mean bit-width change")
    for y in range(delta.shape[0]):
        for x in range(delta.shape[1]):
            if np.isfinite(delta[y, x]):
                axis.text(x, y, f"{delta[y,x]:+.2f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8, pad=0.02)
    colorbar.set_label("Trust-HGQ minus HGQ (bits)")

    shares = {}
    for row in layers:
        shares.setdefault(
            row["layer_order"], (row["hgq_ebops_share"], row["trust_ebops_share"])
        )
    share_delta = [100 * (shares[o][1] - shares[o][0]) for o in orders]
    axis = fig.add_subplot(grid[1, 2])
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in share_delta]
    axis.bar(range(len(names)), share_delta, color=colors)
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(range(len(names)), labels=names, rotation=40, ha="right")
    axis.set_ylabel("Change in EBOP share (percentage points)")
    axis.set_title("(d) Layer-wise precision-cost redistribution")

    fig.suptitle(
        f"Precision migration induced by Trust-HGQ (Permutation-Invariant, F={features}, N={particles})",
        fontsize=14,
    )
    stem = output_dir / f"trust_hgq_bitwidth_migration_f{features}_n{particles}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, rows):
    lines = [
        "# Trust-HGQ heterogeneous bit-width allocation",
        "",
        "Descriptive results from the exact deployed archives (one seed).",
        "",
        "| F | N | Accuracy: HGQ -> Trust-HGQ | ECE: HGQ -> Trust-HGQ | EBOP: HGQ -> Trust-HGQ | dEBOP | Bits up / same / down | Allocation distance |",
        "|---:|---:|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['features']} | {row['particles']} | "
            f"{row['hgq_accuracy']:.3%} -> {row['trust_accuracy']:.3%} | "
            f"{row['hgq_ece']:.5f} -> {row['trust_ece']:.5f} | "
            f"{row['hgq_ebops']:,} -> {row['trust_ebops']:,} | "
            f"{row['delta_ebops_pct']:+.2f}% | "
            f"{row['fraction_increased']:.2%} / {row['fraction_unchanged']:.2%} / "
            f"{row['fraction_decreased']:.2%} | {row['allocation_distance']:.4f} |"
        )
    lines += [
        "",
        "Allocation distance is total-variation distance between normalized per-layer EBOP shares.",
        "Category-specific migration rates are in allocation_summary.csv.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = read_selection(
        (ROOT / args.hardware_results).resolve(),
        set(args.features),
        set(args.particles),
    )
    bit_rows, layer_rows, summaries = [], [], []
    transitions = Counter()
    metadata = {"representative": {
        "features": args.representative_features,
        "particles": args.representative_particles,
    }, "models": []}

    for result in selected:
        features, particles = result["features"], result["particles"]
        hgq_path = baseline_model(features, particles)
        trust_path = (ROOT / result["checkpoint"]).resolve()
        if not trust_path.is_file():
            raise FileNotFoundError(trust_path)
        print(f"ANALYZE F={features} N={particles}", flush=True)
        print(f"  HGQ:       {hgq_path.relative_to(ROOT)}", flush=True)
        print(f"  Trust-HGQ: {trust_path.relative_to(ROOT)}", flush=True)
        hgq, trust, hebops, tebops = paired_archives(hgq_path, trust_path)

        model_rows = []
        for key in sorted(hgq, key=lambda name: (hgq[name]["layer_order"], name)):
            hq, tq = hgq[key], trust[key]
            arrays = [
                hq["k"], hq["i"], hq["f"], hq["allocated"], hq["effective"], hq["total_format"],
                tq["k"], tq["i"], tq["f"], tq["allocated"], tq["effective"], tq["total_format"],
            ]
            for index, values in enumerate(zip(*(array.ravel() for array in arrays))):
                hk, hi, hf, ha, he, ht, tk, ti, tf, ta, te, tt = map(int, values)
                row = {
                    "features": features, "particles": particles,
                    "layer_order": hq["layer_order"], "layer": hq["layer"],
                    "logical_layer": hq["logical_layer"], "category": hq["category"],
                    "role": hq["role"], "quantizer_path": key, "flat_index": index,
                    "q_type": hq["qtype"],
                    "hgq_k": hk, "hgq_i": hi, "hgq_f": hf,
                    "hgq_allocated_bits": ha, "hgq_effective_bits": he,
                    "hgq_total_format_bits": ht,
                    "trust_k": tk, "trust_i": ti, "trust_f": tf,
                    "trust_allocated_bits": ta, "trust_effective_bits": te,
                    "trust_total_format_bits": tt,
                    "delta_allocated_bits": ta-ha, "delta_effective_bits": te-he,
                    "delta_total_format_bits": tt-ht,
                }
                bit_rows.append(row)
                model_rows.append(row)
                transitions[(features, particles, hq["category"], he, te)] += 1

        htotal, ttotal = sum(hebops.values()), sum(tebops.values())
        layer_names = sorted(
            set(hebops) | set(tebops),
            key=lambda name: min(r["layer_order"] for r in model_rows if r["layer"] == name),
        )
        hshares, tshares = [], []
        for layer in layer_names:
            rows = [r for r in model_rows if r["layer"] == layer]
            order, label = rows[0]["layer_order"], rows[0]["logical_layer"]
            hshare, tshare = hebops.get(layer, 0)/htotal, tebops.get(layer, 0)/ttotal
            hshares.append(hshare)
            tshares.append(tshare)
            for category in ("weight", "activation", "bias"):
                selected_rows = [r for r in rows if r["category"] == category]
                if selected_rows:
                    layer_rows.append({
                        "features": features, "particles": particles,
                        "layer_order": order, "layer": layer,
                        "logical_layer": label, "category": category,
                        **stats(selected_rows),
                        "hgq_layer_ebops": hebops.get(layer, 0),
                        "trust_layer_ebops": tebops.get(layer, 0),
                        "hgq_ebops_share": hshare, "trust_ebops_share": tshare,
                        "delta_ebops_share": tshare-hshare,
                    })

        overall, weight, activation = (
            stats(model_rows), stats(model_rows, "weight"), stats(model_rows, "activation")
        )
        summary = {
            "features": features, "particles": particles,
            "hgq_accuracy": float(result["hgq_accuracy"]),
            "trust_accuracy": float(result["ece_accuracy"]),
            "delta_accuracy": float(result["delta_accuracy"]),
            "hgq_ece": float(result["hgq_ece"]),
            "trust_ece": float(result["ece_ece"]),
            "delta_ece": float(result["delta_ece"]),
            "hgq_ebops": htotal, "trust_ebops": ttotal,
            "delta_ebops_pct": 100*(ttotal/htotal-1),
            "allocation_distance": 0.5*float(
                np.sum(np.abs(np.asarray(tshares)-np.asarray(hshares)))
            ),
            **overall,
            "weight_fraction_increased": weight["fraction_increased"],
            "weight_fraction_unchanged": weight["fraction_unchanged"],
            "weight_fraction_decreased": weight["fraction_decreased"],
            "weight_mean_delta_bits": weight["mean_delta_bits"],
            "activation_fraction_increased": activation["fraction_increased"],
            "activation_fraction_unchanged": activation["fraction_unchanged"],
            "activation_fraction_decreased": activation["fraction_decreased"],
            "activation_mean_delta_bits": activation["mean_delta_bits"],
        }
        summaries.append(summary)
        metadata["models"].append({
            "features": features, "particles": particles,
            "hgq": str(hgq_path), "trust_hgq": str(trust_path),
            "quantizers": len(hgq),
        })
        print(
            f"  EBOPs {htotal} -> {ttotal}; allocation distance="
            f"{summary['allocation_distance']:.4f}; bits up/same/down="
            f"{overall['fraction_increased']:.2%}/"
            f"{overall['fraction_unchanged']:.2%}/"
            f"{overall['fraction_decreased']:.2%}",
            flush=True,
        )

    transition_rows = []
    for (features, particles, category, hbit, tbit), count in sorted(transitions.items()):
        total = sum(
            value for (f, n, c, _, _), value in transitions.items()
            if (f, n, c) == (features, particles, category)
        )
        transition_rows.append({
            "features": features, "particles": particles, "category": category,
            "hgq_effective_bits": hbit, "trust_effective_bits": tbit,
            "count": count, "fraction": count/total,
        })

    write_csv(output_dir/"bitwidth_long.csv", bit_rows)
    write_csv(output_dir/"layer_summary.csv", layer_rows)
    write_csv(output_dir/"allocation_summary.csv", summaries)
    write_csv(output_dir/"bitwidth_transitions.csv", transition_rows)
    (output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    write_markdown(output_dir/"allocation_summary.md", summaries)
    make_figure(
        bit_rows, layer_rows, args.representative_features,
        args.representative_particles, output_dir,
    )
    print(f"COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
