#!/usr/bin/env bash
# Main workflow controller for DeepVariant fine-tuning pipeline
#
# Usage:
#   ./scripts/run_pipeline.sh              # Run all steps
#   ./scripts/run_pipeline.sh 3            # Run from step 3 onwards
#   ./scripts/run_pipeline.sh 1 2          # Run steps 1 and 2 only

source "$(dirname "$0")/utils.sh"
load_env

STEPS=(
  "00_prep_ref.sh"
  "01_make_examples.sh"
  "02_shuffle_tfrecords.sh"
  "03_train.sh"
  "04_call_variants.sh"
  "05_evaluate.sh"
)

run_step() {
  local step_script="${SCRIPT_DIR}/$1"
  log_step "Running $1"
  bash "$step_script"
  if [[ $? -ne 0 ]]; then
    log "FAILED: $1"
    exit 1
  fi
  log "DONE: $1"
}

# Parse arguments
if [[ $# -eq 0 ]]; then
  # Run all steps
  for step in "${STEPS[@]}"; do
    run_step "$step"
  done
elif [[ $# -eq 1 ]]; then
  # Run from step N onwards
  start_idx=$(( $1 - 1 ))
  for (( i=start_idx; i<${#STEPS[@]}; i++ )); do
    run_step "${STEPS[$i]}"
  done
else
  # Run specific steps
  for step_num in "$@"; do
    idx=$(( step_num - 1 ))
    run_step "${STEPS[$idx]}"
  done
fi

log_step "Pipeline complete"
