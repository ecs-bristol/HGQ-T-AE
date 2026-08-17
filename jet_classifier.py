import sys
from pathlib import Path

sys.path.append("../")

import omegaconf
import argparse
import random
import mplhep as hep
import numpy as np


from src.dataloader import get_data
from src.model import get_model
from src.train import train_hgq
from src.test import test
from src.syn_test import syn_test_verilog

hep.style.use(hep.style.CMS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--run", "-r", nargs="*", type=str, default=["all"])
    parser.add_argument("--ckpt", "-k", type=str, default=None)
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)

    print("Setting seed...")
    random.seed(conf.seed)
    np.random.seed(conf.seed)

    data_path = Path(conf.datapath)

    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data(
        data_path, conf.n_constituents, ptetaphi=conf.pt_eta_phi
    )
    print("Creating models...")

    model_hgq = get_model(conf)
    model_hgq.summary()

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    if "all" in args.run or "train" in args.run:
        print("Phase: train_hgq")
        model_hgq, _ = train_hgq(
            model_hgq,
            X_train,
            y_train,
            X_test,
            y_test,
            conf,
        )

    if "all" in args.run or "test" in args.run:
        print("Phase: test")
        test(model_hgq, Path(conf.save_path), X_train, X_test, y_test)
        bops_computed = True

    # if 'hls' in args.run or 'all' in args.run:
    #     print('Phase: syn - hls4ml')
    #     syn_test_hls(Path(conf.save_path), X_test, y_test, N=None)

    if "verilog" in args.run or "all" in args.run:
        print("Phase: syn - da4ml - verilog")
        syn_test_verilog(Path(conf.save_path), X_test, y_test, N=None)
