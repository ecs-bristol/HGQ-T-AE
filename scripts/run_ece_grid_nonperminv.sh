#!/usr/bin/env bash
set -u

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 GPU_ID CONFIG [CONFIG ...]" >&2
    exit 2
fi

gpu_id="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root="$(cd -- "${script_dir}/.." >/dev/null 2>&1 && pwd)"
python_bin="${PYTHON:-python}"
tmp_dir="${HGQ_ECE_TMPDIR:-${root}/.tmp-hgq-ece}"
log_dir="logs/ece-grid-nonperminv"
result_dir="calibration_results/ece-grid-nonperminv"

cd "$root" || exit 2
mkdir -p "$tmp_dir" "$log_dir" "$result_dir"
chmod 700 "$tmp_dir"

for config in "$@"; do
    stem="${config##*/}"
    stem="${stem%.yaml}"
    log_path="$log_dir/$stem.log"
    output_path="$result_dir/$stem.json"

    if [ -e "$output_path" ]; then
        echo "Refusing to overwrite existing result: $output_path" >&2
        exit 3
    fi

    echo "START $(date -Is) GPU=$gpu_id CONFIG=$config" | tee "$log_path"
    timeout --signal=TERM --kill-after=10s 30m env \
        TMPDIR="$tmp_dir" \
        PYTHONUNBUFFERED=1 \
        KERAS_BACKEND=jax \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$python_bin" jet_classifier.py -c "$config" -r train \
        >> "$log_path" 2>&1
    train_status="$?"
    echo "TRAIN_EXIT=$train_status" | tee -a "$log_path"

    if [ "$train_status" -ne 0 ]; then
        echo "ABORTING_WORKER_AFTER_TRAIN_FAILURE GPU=$gpu_id CONFIG=$config" | tee -a "$log_path"
        exit "$train_status"
    fi

    timeout --signal=TERM --kill-after=10s 2h env \
        TMPDIR="$tmp_dir" \
        PYTHONUNBUFFERED=1 \
        KERAS_BACKEND=jax \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$python_bin" evaluate_calibration.py \
        --configs "$config" \
        --output "$output_path" \
        --bins 15 \
        >> "$log_path" 2>&1
    eval_status="$?"
    echo "EVAL_EXIT=$eval_status" | tee -a "$log_path"

    if [ "$eval_status" -ne 0 ]; then
        echo "ABORTING_WORKER_AFTER_EVAL_FAILURE GPU=$gpu_id CONFIG=$config" | tee -a "$log_path"
        exit "$eval_status"
    fi

    echo "DONE $(date -Is) GPU=$gpu_id CONFIG=$config" | tee -a "$log_path"
done

echo "WORKER_COMPLETE $(date -Is) GPU=$gpu_id"
