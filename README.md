# Reproduction Instructions

This artifact extends the open-source JEDI-linear repository under the Apache
License, Version 2.0. The JEDI-linear components originate from that repository,
their original copyright attribution is retained in `NOTICE`, and the software
remains licensed under Apache License, Version 2.0, as provided in `LICENSE`.

Install environment

```bash
conda env create -f environment.yml
conda activate jedi-linear
```

For trustworthiness-aware HGQ training, use the environment file that installs
the ECE-enabled HGQ2 branch:

```bash
conda env create -f environment-trust.yml
conda activate jedi-linear-trust
```

This file pins Python 3.11 and the tested HGQ2, da4ml, Keras, and JAX
revisions used for the Trust-HGQ results.

### Download and prepare dataset

```bash
bash prepare_dataset.sh
```


### Obtaining the models

The models shown in the tables in the papers are included in `official_models.tar.gz`.
You can extract them with:

```bash
tar -xvf official_models.tar.gz
```

If you want to retrain the models, you can do so with the following commands.
This will launch a whole Pareto scan, so many models will be saved.

```bash
KERAS_BACKEND=jax python jet_classifier.py -c configs/<config_file> -r train
```

By default, the original HGQ training objective is used: sparse categorical
cross-entropy for accuracy, the existing beta schedule for EBOPs, and a
two-objective Pareto front over validation accuracy and EBOPs. To add
trustworthiness-aware training, include a `trust` section in the YAML config.
If the section is present, ECE is added to the loss, metrics, checkpoint names,
and Pareto-front tracking:

```yaml
trust:
  enabled: true
  ece_bins: 15
  ece_weight: 5.0
  pareto_min_accuracy: 0.68
```

Omitting the `trust` section keeps the original training behavior. Setting
`trust.enabled: false` also disables the trustworthiness path, which is useful
when comparing configurations without editing the rest of the YAML.

The example trust-aware configuration for the `n=16`, 3-feature,
non-permutation-invariant GNN model is:

```bash
KERAS_BACKEND=jax python jet_classifier.py \
  -c configs/sweep-n16-f3-ece.yaml \
  -r train
```

The 20 Trust-HGQ configurations used in the paper cover both model families,
two feature counts, and five particle counts. Run each family from an activated
`jedi-linear-trust` environment, for example:

```bash
bash scripts/run_ece_grid_perminv.sh 0 configs/ece-grid-perminv-*.yaml
bash scripts/run_ece_grid_nonperminv.sh 1 configs/ece-grid-nonperminv-*.yaml
```

The first argument selects the GPU. The scripts derive the repository root
from their own locations, use `python` from the active environment, and write
temporary files below the repository. Set `PYTHON` or `HGQ_ECE_TMPDIR` to
override those defaults:

```bash
PYTHON=/path/to/python HGQ_ECE_TMPDIR=/path/to/tmp bash scripts/run_ece_grid_perminv.sh 0 configs/ece-grid-perminv-*.yaml
```

### Evaluation on test set, convert to Verilog

The outputs are already included in the `official_models.tar.gz`, but you can validate them with:

```bash
KERAS_BACKEND=jax python jet_classifier.py -c configs/<config_file> -r test verilog
```

The Verilator may require a newer C++ compiler. We tested our code with g++ 15.1.1.

### Evaluation of calibration metrics

`evaluate_calibration.py` evaluates trained HGQ Keras models on the test set
and reports accuracy and expected calibration
error (ECE). It does not run `da4ml` or generate Verilog. The script uses the
same dataset normalization as `src/dataloader.py`: it reads
`dataset/150c-train.h5` to compute the feature mean/std and evaluates on
`dataset/150c-test.h5`.

Before running it, prepare the dataset and extract the official models:

```bash
bash prepare_dataset.sh
tar -xvf official_models.tar.gz
```

For example, to evaluate the `n=16`, 3-feature, non-permutation-invariant HGQ
model:

```bash
KERAS_BACKEND=jax python evaluate_calibration.py \
  --configs configs/sweep-n16-f3.yaml \
  --output calibration_results/n16_f3_non_perminv.json \
  --bins 15
```

This writes both JSON and CSV outputs:

```text
calibration_results/n16_f3_non_perminv.json
calibration_results/n16_f3_non_perminv.csv
```

The CSV contains one row per checkpoint with fields including `acc`, `ece`,
`ebops`, and `load_mode`. ECE is computed from the softmax confidence using
15 equal-width confidence bins by default.

To evaluate every config in `configs/`, omit `--configs`:

```bash
KERAS_BACKEND=jax python evaluate_calibration.py \
  --output calibration_results/ece.json \
  --bins 15
```

### Synthesis

Due to size consideration, we removed the Vivado project files, but only included the gererated reports.

```bash
cd <output_directory>/<model_directory>/da4ml_verilog_prjs/<verilog_project>
vivado -mode batch -source build_prj.tcl

# Starting v0.5.x, the da4ml generated projects layout changed, and the synthesis script is now named build_vivado_prj.tcl
# vivado -mode batch -source build_vivado_prj.tcl
```

### Generate json report

Using the included `load_summary.py` script in the tarball, you can generate the json report from all the synthesis results:

```bash
cd <output_directory>
for p in *-feature*; do
    for N in 8 16 32 64 128; do
        name=$(basename $p)
        for f in $p/*$N; do python3 load_summary.py -e $f/test_acc.json $f/da4ml_verilog_prjs/* -o summary/$N-particle-$name.json; done
    done
done
```
