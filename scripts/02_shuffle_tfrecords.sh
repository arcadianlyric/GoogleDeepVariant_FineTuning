#!/usr/bin/env bash
# Step 2: Shuffle and split TFRecords into train/tune datasets.

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

OUTPUT_DIR="${PROJECT_DIR}/output/tfrecords_shuffled"
SAMPLE="${TRAIN_SAMPLE}"

log_step "Shuffling TFRecords (${SAMPLE})"

EXAMPLE_DIR="${PROJECT_DIR}/output/examples/${SAMPLE}"
EXAMPLE_INFO="${EXAMPLE_DIR}/examples.tfrecord.gz.example_info.json"

if ! find "${EXAMPLE_DIR}" -name "*.tfrecord.gz*" ! -name "*.json" | grep -q .; then
  echo "ERROR: TFRecords not found. Run 01_make_examples.sh first." >&2
  exit 1
fi

[[ -f "${EXAMPLE_INFO}" ]] || {
  echo "ERROR: Missing example_info.json: ${EXAMPLE_INFO}" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}/train" "${OUTPUT_DIR}/tune" "${PROJECT_DIR}/output/intermediate"

cat > "${PROJECT_DIR}/output/intermediate/shuffle_split.py" <<'PY'
import glob
import os
import random
import shutil
import sys

import tensorflow as tf

sample = os.environ["TRAIN_SAMPLE"]
seed = int(os.environ.get("SHUFFLE_SEED", "42"))
train_fraction = float(os.environ.get("TRAIN_FRACTION", "0.9"))

root = "/work"
example_dir = f"{root}/output/examples/{sample}"
out_dir = f"{root}/output/tfrecords_shuffled"

example_files = sorted(
    f for f in glob.glob(f"{example_dir}/examples.tfrecord.gz*")
    if not f.endswith(".json")
)
if not example_files:
    raise SystemExit(f"No TFRecords found in {example_dir}")

records = list(tf.data.TFRecordDataset(example_files, compression_type="GZIP"))
random.seed(seed)
random.shuffle(records)

split = int(len(records) * train_fraction)
train_records = records[:split]
tune_records = records[split:]

print(f"Total examples: {len(records):,}")
print(f"Train examples: {len(train_records):,}")
print(f"Tune examples : {len(tune_records):,}")

train_dir = f"{out_dir}/train"
tune_dir = f"{out_dir}/tune"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(tune_dir, exist_ok=True)

opts = tf.io.TFRecordOptions(compression_type="GZIP")
for path, recs in [
    (f"{train_dir}/train.tfrecord.gz", train_records),
    (f"{tune_dir}/tune.tfrecord.gz", tune_records),
]:
    with tf.io.TFRecordWriter(path, options=opts) as writer:
        for rec in recs:
            writer.write(rec.numpy())
    print(f"Wrote {path}")

for directory in [train_dir, tune_dir]:
    shutil.copy2(f"{example_dir}/examples.tfrecord.gz.example_info.json", f"{directory}/example_info.json")

with open(f"{train_dir}/train_dataset.pbtxt", "w") as fh:
    fh.write(f'name: "{sample}_train"\n')
    fh.write(f'tfrecord_path: "{train_dir}/train.tfrecord.gz"\n')
    fh.write(f'num_examples: {len(train_records)}\n')

with open(f"{tune_dir}/tune_dataset.pbtxt", "w") as fh:
    fh.write(f'name: "{sample}_tune"\n')
    fh.write(f'tfrecord_path: "{tune_dir}/tune.tfrecord.gz"\n')
    fh.write(f'num_examples: {len(tune_records)}\n')
PY

run_dv_docker \
  env TRAIN_SAMPLE="${SAMPLE}" SHUFFLE_SEED="${SHUFFLE_SEED:-42}" TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}" \
  python3 /work/output/intermediate/shuffle_split.py

log "Shuffled TFRecords written to ${OUTPUT_DIR}"
