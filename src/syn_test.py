from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from da4ml.converter.hgq2.parser import trace_model
from da4ml.trace import comb_trace, HWConfig
from da4ml.codegen import VerilogModel

import keras


def syn_test_hls(save_path: Path, X, Y, N=None, softmax=False):
    from hls4ml.converters import convert_from_keras_model

    X = np.ascontiguousarray(X).astype(np.float32)
    Y = np.ascontiguousarray(Y)
    N = N or X.shape[0]
    print(f"Running inference on {N} samples")
    pbar = tqdm(list(save_path.glob("models/*.keras")))
    (save_path / "hls4ml_prjs").mkdir(exist_ok=True, parents=True)
    with open(save_path / "test_acc.json", "r") as f:
        results = json.load(f)
    for ckpt in pbar:
        model: keras.Model = keras.models.load_model(ckpt)  # type: ignore

        hls_prj_path = save_path / f"hls4ml_prjs/{ckpt.stem.split('-')[0]}"

        model_hls = convert_from_keras_model(
            model,
            hls_config={
                "Model": {
                    "Precision": "ap_fixed<1,0>",
                    "ReuseFactor": 1,
                    "Strategy": "distributed_arithmetic",
                },
            },
            output_dir=str(hls_prj_path),
            project_name="jet_classifier_large",
            part="xcvu13p-flga2577-2-e",
            clock_period=5,
            io_type="io_parallel",
            backend="vitis",
        )

        model_hls.compile()

        pred_keras = model.predict(X[:N], verbose=0, batch_size=16384)  # type: ignore
        pred_hls = model_hls.predict(X[:N])
        hls_acc = np.mean(np.argmax(pred_hls, axis=1) == np.array(Y[:N]).ravel())
        results[ckpt.name]["hls_acc"] = hls_acc
        da_kernel_cost = sum(
            g.attributes.get("da_kernel_cost", 0) for g in model_hls.graph.values()
        )
        results[ckpt.name]["da_kernel_cost"] = da_kernel_cost

        ndiff = np.sum(np.any(pred_hls - pred_keras != 0, axis=1))

        with open(save_path / "test_acc.json", "w") as f:
            json.dump(results, f)

        if ndiff > 0:
            print(f"{ndiff} out of {N} samples differ for {ckpt.name}")


def syn_test_verilog(save_path: Path, X, Y, N=None, softmax=False):
    X, Y = np.ascontiguousarray(X).astype(np.float32), np.ascontiguousarray(Y)
    N = N or X.shape[0]
    print(f"Running inference on {N} samples")
    pbar = tqdm(list(save_path.glob("models/*.keras")))
    (save_path / "da4ml_verilog_prjs").mkdir(exist_ok=True)
    with open(save_path / "test_acc.json", "r") as f:
        results = json.load(f)
    for ckpt in pbar:
        model: keras.Model = keras.models.load_model(ckpt)  # type: ignore
        verilog_prj_path = save_path / f"da4ml_verilog_prjs/{ckpt.stem.split('-')[0]}"

        inp, out = trace_model(
            model, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1)
        )
        solution = comb_trace(inp, out)
        verilog_model = VerilogModel(
            solution,
            prj_name="jet_classifier_large",
            path=verilog_prj_path,
            part_name="xcvu13p-flga2577-2-e",
            clock_period=2,
            clock_uncertainty=0.0,
            latency_cutoff=2,
        )
        verilog_model.compile()  # Verilator and OpenMP binding
        try:
            verilog_model._load_lib()
        except Exception as e:
            print(f"Error loading Verilog model: {e}")
            results[ckpt.name]["verilog_acc"] = float(-1)
            results[ckpt.name]["verilog_ndiff"] = -1
            results[ckpt.name]["da_est_FF"] = -1
            results[ckpt.name]["da_est_LUT"] = -1

            with open(save_path / "test_acc.json", "w") as f:
                json.dump(results, f)

            continue

        pred_verilog = verilog_model.predict(X[:N])

        sim_acc = np.mean(np.argmax(pred_verilog, axis=1) == Y[:N].ravel())
        results[ckpt.name]["verilog_acc"] = float(sim_acc)
        results[ckpt.name]["da_est_FF"] = round(verilog_model._pipe.reg_bits)  # type: ignore
        results[ckpt.name]["da_est_LUT"] = round(verilog_model._pipe.cost)  # type: ignore

        with open(save_path / "test_acc.json", "w") as f:
            json.dump(results, f)

        pbar.set_description(f"RTL Acc: {sim_acc:.5%}")
