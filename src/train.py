from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from hgq.utils.sugar import BetaScheduler, ParetoFront, PBar, PieceWiseSchedule

from hgq.utils.sugar import FreeEBOPs, Dataset
import keras
import pickle as pkl


def _has_conf_key(conf, key):
    if conf is None:
        return False
    if isinstance(conf, dict):
        return key in conf
    if hasattr(conf, "__contains__"):
        try:
            return key in conf
        except TypeError:
            pass
    return hasattr(conf, key)


def _get_conf_value(conf, key, default=None):
    if conf is None:
        return default
    if hasattr(conf, "get"):
        return conf.get(key, default)
    return getattr(conf, key, default)


def train_hgq(model: keras.Model, X, Y, Xs, Ys, conf):
    save_path = Path(conf.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    init_weights = _get_conf_value(conf.train, "init_weights", None)
    if init_weights:
        print(f"Loading initial weights from {init_weights}")
        try:
            model.load_weights(init_weights)
        except ValueError as exc:
            message = str(exc)
            known_legacy_quantizer_mismatch = (
                "FixedPointQuantizerKIF" in message
                and "i_decay_speed" in message
                and "received 0 variables" in message
            )
            if not known_legacy_quantizer_mismatch:
                raise
            weights_path = Path(init_weights)
            model_path = (
                weights_path.parent.parent / "models" / weights_path.name
            )
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Compatible full model not found: {model_path}"
                ) from exc
            print(f"Loading compatible full model from {model_path}")
            model = keras.models.load_model(model_path, compile=False)

    pred = model.predict(Xs, batch_size=2048, verbose=0)  # type: ignore
    hgq_acc_1 = np.mean(np.argmax(pred, axis=1) == np.array(Ys).ravel())
    print(f"pre-training HGQ accuracy: {hgq_acc_1:.2%}")

    with open(save_path / "pretrain_acc.txt", "w") as f:
        f.write(f"pre-training HGQ accuracy: {hgq_acc_1:.2%}\n")

    print("Compiling model & registering callbacks...")
    opt = keras.optimizers.Adam()
    trust_conf = _get_conf_value(conf, "trust", None)
    trust_enabled = (
        _has_conf_key(conf, "trust")
        and trust_conf is not None
        and bool(_get_conf_value(trust_conf, "enabled", True))
    )
    ece_bins = int(_get_conf_value(trust_conf, "ece_bins", 15))
    ece_weight = float(_get_conf_value(trust_conf, "ece_weight", 1.0))
    if trust_enabled:
        from hgq.losses import SparseCategoricalCrossentropyWithSoftECE
        from hgq.metrics import SparseCategoricalECE

        loss = SparseCategoricalCrossentropyWithSoftECE(
            from_logits=True,
            n_bins=ece_bins,
            ece_weight=ece_weight,
        )
        metrics = ["accuracy", SparseCategoricalECE(from_logits=True, n_bins=ece_bins)]
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)  # type: ignore

    assert conf.train.cdr_args["t_mul"] == 1
    first_decay_steps = conf.train.cdr_args["first_decay_steps"]
    initial_learning_rate = conf.train.cdr_args["initial_learning_rate"]
    t_mul = conf.train.cdr_args["t_mul"]
    m_mul = conf.train.cdr_args["m_mul"]
    alpha = conf.train.cdr_args["alpha"]
    alpha_steps = conf.train.cdr_args["alpha_steps"]

    def cosine_decay_restarts(global_step):
        from math import cos, pi

        n_cycle = 1
        cycle_step = global_step
        cycle_len = first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1

        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        lr = alpha + 0.5 * (initial_learning_rate - alpha) * (
            1 + cos(pi * cycle_t)
        ) * m_mul ** max(n_cycle - 1, 0)
        return lr

    scheduler = keras.callbacks.LearningRateScheduler(cosine_decay_restarts)

    if trust_enabled:
        pbar = PBar(
            metric="loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - ece: {ece:.4f}/{val_ece:.4f} - lr:{learning_rate:.2e} - beta: {beta:.2e}"
        )
    else:
        pbar = PBar(
            metric="loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr:{learning_rate:.2e} - beta: {beta:.2e}"
        )

    terminate_on_nan = keras.callbacks.TerminateOnNaN()

    if trust_enabled:
        pareto_metrics = ["val_accuracy", "ebops", "val_ece"]
        pareto_sides = [1, -1, -1]
        fname_format = (
            "epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}"
            "-val_ece={val_ece:.5f}-EBOPs={ebops}.keras"
        )
    else:
        pareto_metrics = ["val_accuracy", "ebops"]
        pareto_sides = [1, -1]
        fname_format = "epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-EBOPs={ebops}.keras"
    if trust_enabled:
        pareto_min_accuracy = float(
            _get_conf_value(trust_conf, "pareto_min_accuracy", 0.5)
        )
    else:
        pareto_min_accuracy = 0.5

    save = ParetoFront(
        path=save_path / "ckpts",
        fname_format=fname_format,
        metrics=pareto_metrics,
        enable_if=lambda x: x["val_accuracy"] > pareto_min_accuracy,
        sides=pareto_sides,
    )

    ebops = FreeEBOPs()
    beta_sched = BetaScheduler(PieceWiseSchedule(conf.beta.intervals))

    callbacks = [scheduler, beta_sched, ebops, save, pbar, terminate_on_nan]
    early_stopping_conf = _get_conf_value(conf.train, "early_stopping", None)
    early_stopping_enabled = bool(
        _get_conf_value(early_stopping_conf, "enabled", False)
    )
    if early_stopping_enabled:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=str(_get_conf_value(early_stopping_conf, "monitor", "val_loss")),
            mode=str(_get_conf_value(early_stopping_conf, "mode", "min")),
            patience=int(_get_conf_value(early_stopping_conf, "patience", 15)),
            min_delta=float(_get_conf_value(early_stopping_conf, "min_delta", 0.0)),
            restore_best_weights=bool(
                _get_conf_value(early_stopping_conf, "restore_best_weights", False)
            ),
            verbose=1,
        )
        callbacks.append(early_stopping)
        print(
            "Early stopping enabled: "
            f"monitor={early_stopping.monitor}, mode={early_stopping.mode}, "
            f"patience={early_stopping.patience}, min_delta={early_stopping.min_delta}"
        )

    batch_size = conf.train.bsz

    val_split = 0.1
    val_size = int(len(X) * val_split)
    X, Y = X.astype(np.float16), Y.astype(np.int32)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    dataset_train = Dataset(
        X_train, Y_train, batch_size=batch_size, drop_last=True, device="gpu:0"
    )
    dataset_val = Dataset(
        X_val, Y_val, batch_size=batch_size, drop_last=True, device="gpu:0"
    )

    model.fit(
        dataset_train,
        epochs=conf.train.epochs,
        validation_data=dataset_val,
        callbacks=callbacks,
        verbose=0,
    )  # type: ignore
    history = model.history.history  # type: ignore
    with open(save_path / "history.pkl", "wb") as f:
        f.write(pkl.dumps(history))

    model.save(save_path / "last.h5")

    return model, history
