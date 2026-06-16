#!/usr/bin/env bash
# Shared utility functions for DV fine-tuning pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config. config/.env is optional; scripts have Linux CPU
# defaults so a fresh checkout can be configured mostly through config.yaml.
load_env() {
  local env_file="${PROJECT_DIR}/config/.env"
  export PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
  if [[ -f "$env_file" ]]; then
    set -a
    source "$env_file"
    set +a
  fi

  export DV_IMAGE="${DV_IMAGE:-google/deepvariant:1.6.1}"
  export NUM_SHARDS="${NUM_SHARDS:-$(nproc)}"
  export SHM_SIZE="${SHM_SIZE:-12gb}"
  export TRAIN_SAMPLE="${TRAIN_SAMPLE:-pe150}"
  if [[ "${REF_FASTA:-}" == *.gz && -f "${REF_FASTA%.gz}" ]]; then
    export REF_FASTA="${REF_FASTA%.gz}"
  fi
}

# Parse YAML value using Python.
# Usage: yaml_val config.yaml "training.batch_size"
yaml_val() {
  local file="$1" key="$2"
  python3 - "$file" "$key" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as fh:
    data = yaml.safe_load(fh)

value = data
for part in sys.argv[2].split("."):
    value = value[part]

if isinstance(value, bool):
    print(str(value).lower())
elif value is None:
    print("")
else:
    print(value)
PY
}

# Get config.yaml path
config_file() {
  echo "${PROJECT_DIR}/config/config.yaml"
}

# Ensure output directories exist
ensure_output_dirs() {
  mkdir -p "${PROJECT_DIR}/output/examples/pe150"
  mkdir -p "${PROJECT_DIR}/output/examples/se600"
  mkdir -p "${PROJECT_DIR}/output/tfrecords_shuffled"
  mkdir -p "${PROJECT_DIR}/output/training"
  mkdir -p "${PROJECT_DIR}/output/call_variants"
  mkdir -p "${PROJECT_DIR}/output/evaluation"
  mkdir -p "${PROJECT_DIR}/output/intermediate"
}

# Convert a project-relative host path to its in-container path.
work_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    case "$path" in
      "${PROJECT_DIR}"/*) echo "/work/${path#"${PROJECT_DIR}/"}" ;;
      *) echo "$path" ;;
    esac
  else
    echo "/work/$path"
  fi
}

sample_bam() {
  local sample="$1"
  yaml_val "$(config_file)" "samples.${sample}.bam"
}

truth_vcf() {
  yaml_val "$(config_file)" "truth.vcf"
}

confident_bed() {
  yaml_val "$(config_file)" "truth.confident_bed"
}

# Run DeepVariant Docker in CPU mode with standard Linux server mounts.
run_dv_docker() {
  local extra_args=("$@")
  docker run \
    --rm \
    -u "$(id -u):$(id -g)" \
    -v "${PROJECT_DIR}":/work \
    -v "${PROJECT_DIR}/data":/input \
    -v "${PROJECT_DIR}/ref":/ref \
    -v "${PROJECT_DIR}/output":/output \
    -v "$(dirname "${REF_FASTA}")":/reference \
    -w /work \
    --shm-size "${SHM_SIZE}" \
    "${DV_IMAGE}" \
    "${extra_args[@]}"
}

# Check BAM index exists, create if missing
ensure_bam_index() {
  local bam="$1"
  if [[ ! -f "${bam}.bai" ]] && [[ ! -f "${bam%.bam}.bai" ]]; then
    echo "Creating index for ${bam}..."
    samtools index "$bam"
  fi
}

# Log with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_step() {
  echo ""
  echo "============================================"
  log "STEP: $*"
  echo "============================================"
}
