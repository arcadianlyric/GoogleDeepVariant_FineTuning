#!/usr/bin/env bash
# Step 3: Fine-tune DeepVariant model from pretrained WGS checkpoint.
# CPU implementation uses DeepVariant 1.6.1 Keras training API:
#   /opt/deepvariant/bin/train --config config.py --experiment_dir ... --limit ...

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

CONFIG=$(config_file)
BATCH_SIZE=$(yaml_val "$CONFIG" "training.batch_size")
LR=$(yaml_val "$CONFIG" "training.learning_rate")
NUM_STEPS=$(yaml_val "$CONFIG" "training.num_training_steps")
INIT_CKPT=$(yaml_val "$CONFIG" "training.init_checkpoint")

TRAIN_DIR="${PROJECT_DIR}/output/training"
SHUFFLED_DIR="${PROJECT_DIR}/output/tfrecords_shuffled"
PRETRAINED_DIR="${TRAIN_DIR}/pretrained"
DEFAULT_CKPT="${PRETRAINED_DIR}/deepvariant.wgs.ckpt"

log_step "Training: fine-tune from WGS checkpoint"
log "batch_size=${BATCH_SIZE}, lr=${LR}, steps=${NUM_STEPS}"

mkdir -p "${PRETRAINED_DIR}"

if [[ -n "$INIT_CKPT" ]]; then
  if [[ "$INIT_CKPT" = /* ]]; then
    CKPT_HOST="${INIT_CKPT}"
    CKPT_CONTAINER="${INIT_CKPT}"
  else
    CKPT_HOST="${PROJECT_DIR}/${INIT_CKPT}"
    CKPT_CONTAINER="$(work_path "$INIT_CKPT")"
  fi
else
  CKPT_HOST="${DEFAULT_CKPT}"
  CKPT_CONTAINER="/work/output/training/pretrained/deepvariant.wgs.ckpt"
  if [[ ! -f "${CKPT_HOST}.index" ]]; then
    if ! command -v gsutil >/dev/null 2>&1; then
      echo "ERROR: pretrained checkpoint missing and gsutil is not available." >&2
      echo "Install gsutil or set training.init_checkpoint in config/config.yaml." >&2
      exit 1
    fi
    log "Downloading DeepVariant 1.6.1 WGS checkpoint..."
    gsutil -m cp -r "gs://deepvariant/models/DeepVariant/1.6.1/checkpoints/wgs/*" "${PRETRAINED_DIR}/"
  fi
fi

[[ -f "${CKPT_HOST}.index" ]] || {
  echo "ERROR: checkpoint index not found: ${CKPT_HOST}.index" >&2
  exit 1
}

[[ -f "${SHUFFLED_DIR}/train/train_dataset.pbtxt" ]] || {
  echo "ERROR: missing train dataset config. Run 02_shuffle_tfrecords.sh first." >&2
  exit 1
}
[[ -f "${SHUFFLED_DIR}/tune/tune_dataset.pbtxt" ]] || {
  echo "ERROR: missing tune dataset config. Run 02_shuffle_tfrecords.sh first." >&2
  exit 1
}

CONFIG_PY="${TRAIN_DIR}/config.py"
cat > "${CONFIG_PY}" <<PY
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.train_dataset_pbtxt = "/work/output/tfrecords_shuffled/train/train_dataset.pbtxt"
    config.tune_dataset_pbtxt = "/work/output/tfrecords_shuffled/tune/tune_dataset.pbtxt"

    config.model_type = "inception_v3"
    config.init_checkpoint = "${CKPT_CONTAINER}"
    config.init_backbone_with_imagenet = False
    config.weight_decay = 0.00004
    config.backbone_dropout_rate = 0.2

    config.batch_size = ${BATCH_SIZE}
    config.num_epochs = 1
    config.learning_rate = ${LR}
    config.learning_rate_num_epochs_per_decay = 2.0
    config.learning_rate_decay_rate = 0.94
    config.warmup_steps = 0
    config.label_smoothing = 0.0001

    config.optimizer = "rmsprop"
    config.rho = 0.9
    config.momentum = 0.9
    config.epsilon = 1.0

    config.prefetch_buffer_bytes = 16 * 1024 * 1024
    config.input_read_threads = 4
    config.shuffle_buffer_elements = 1000

    config.log_every_steps = 10
    config.tune_every_steps = 50
    config.best_checkpoint_metric = "tune/f1_weighted"
    config.early_stopping_patience = 0
    config.num_validation_examples = 0

    return config
PY

log "Training config: ${CONFIG_PY}"

run_dv_docker \
  /opt/deepvariant/bin/train \
  --config="/work/output/training/config.py" \
  --experiment_dir="/work/output/training" \
  --limit="${NUM_STEPS}"

log "Training completed. Checkpoints in ${TRAIN_DIR}"
