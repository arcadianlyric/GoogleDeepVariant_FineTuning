#!/usr/bin/env bash
# Step 5: Evaluate variant calls against GIAB truth using hap.py
# Compares fine-tuned model output against HG002 benchmark

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

CONFIG=$(config_file)
EVAL_REGION=$(yaml_val "$CONFIG" "eval_region")
HAPPY_IMAGE=$(yaml_val "$CONFIG" "evaluation.happy_image")
REF_BASENAME=$(basename "${REF_FASTA}")

CALL_VCF="${PROJECT_DIR}/output/call_variants/output.vcf.gz"
TRUTH_REL="$(truth_vcf)"
TRUTH_VCF="${PROJECT_DIR}/${TRUTH_REL}"
if [[ ! -f "$TRUTH_VCF" && -f "${TRUTH_VCF}.gz" ]]; then
  TRUTH_VCF="${TRUTH_VCF}.gz"
  TRUTH_REL="${TRUTH_REL}.gz"
fi
TRUTH_CONTAINER="$(work_path "$TRUTH_REL")"
CONFIDENT_REL="$(confident_bed)"
OUTPUT_DIR="${PROJECT_DIR}/output/evaluation"

if [[ ! -f "$CALL_VCF" ]]; then
  echo "ERROR: ${CALL_VCF} not found. Run 04_call_variants.sh first." >&2
  exit 1
fi

log_step "Evaluating with hap.py (${EVAL_REGION})"

docker run \
  --rm \
  -u "$(id -u):$(id -g)" \
  -v "${PROJECT_DIR}":/work \
  -v "${PROJECT_DIR}/output":/output \
  -v "${PROJECT_DIR}/ref":/ref \
  -v "$(dirname "${REF_FASTA}")":/reference \
  -w /work \
  "${HAPPY_IMAGE}" \
  /opt/hap.py/bin/hap.py \
  "${TRUTH_CONTAINER}" \
  "/output/call_variants/output.vcf.gz" \
  -f "$(work_path "${CONFIDENT_REL}")" \
  -r "/reference/${REF_BASENAME}" \
  -o "/output/evaluation/happy_results" \
  --engine=vcfeval \
  --threads="${NUM_SHARDS}" \
  -l "${EVAL_REGION}"

log "Evaluation complete."
log "Results: ${OUTPUT_DIR}/happy_results.summary.csv"

# Print summary
echo ""
echo "=== hap.py Summary ==="
if [[ -f "${OUTPUT_DIR}/happy_results.summary.csv" ]]; then
  column -t -s',' "${OUTPUT_DIR}/happy_results.summary.csv"
else
  echo "Summary file not found. Check logs for errors."
fi
