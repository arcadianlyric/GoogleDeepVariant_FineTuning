#!/usr/bin/env bash
# Step 4: Run call_variants + postprocess_variants using the fine-tuned model
# Evaluates on a held-out region (default: chr20)

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

CONFIG=$(config_file)
EVAL_REGION=$(yaml_val "$CONFIG" "eval_region")
REF_BASENAME=$(basename "${REF_FASTA}")
SAMPLE="${TRAIN_SAMPLE}"
BAM_REL="$(sample_bam "$SAMPLE")"

# Use the latest checkpoint
CKPT_DIR="${PROJECT_DIR}/output/training/checkpoints"
LATEST_CKPT=$(ls -t "${CKPT_DIR}"/ckpt-*.index 2>/dev/null | head -1 | sed 's/.index$//')

if [[ -z "$LATEST_CKPT" ]]; then
  echo "ERROR: No checkpoint found in ${CKPT_DIR}. Run 03_train.sh first." >&2
  exit 1
fi
CKPT_CONTAINER="/work/output/training/checkpoints/$(basename "$LATEST_CKPT")"

log_step "call_variants on ${EVAL_REGION} with fine-tuned model"
log "Sample: ${SAMPLE}"
log "BAM: ${BAM_REL}"
log "Checkpoint: ${CKPT_CONTAINER}"

if [[ "$(yaml_val "$(config_file)" "samples.${SAMPLE}.channel_insert_size")" == "true" ]]; then
  CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref,insert_size"
else
  CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref"
fi
SE600_FLAGS=()
if [[ "$SAMPLE" == "se600" ]]; then
  SE600_FLAGS=(--norealign_reads --min_mapping_quality=0)
fi

log "make_examples (eval): ${SAMPLE}"
run_dv_docker \
  /opt/deepvariant/bin/make_examples \
  --mode calling \
  --ref "/reference/${REF_BASENAME}" \
  --reads "$(work_path "$BAM_REL")" \
  --examples "/output/call_variants/eval_${SAMPLE}.tfrecord.gz" \
  --regions "${EVAL_REGION}" \
  --channel_list="${CHANNEL_LIST}" \
  --keep_supplementary_alignments \
  --normalize_reads \
  "${SE600_FLAGS[@]}" \
  --task 0

# call_variants
log "Running call_variants..."

run_dv_docker \
  /opt/deepvariant/bin/call_variants \
  --outfile "/output/call_variants/call_variants_output.tfrecord.gz" \
  --examples "/output/call_variants/eval_${SAMPLE}.tfrecord.gz" \
  --checkpoint "${CKPT_CONTAINER}"

# postprocess_variants
log "Running postprocess_variants..."

run_dv_docker \
  /opt/deepvariant/bin/postprocess_variants \
  --ref "/reference/${REF_BASENAME}" \
  --infile "/output/call_variants/call_variants_output.tfrecord.gz" \
  --outfile "/output/call_variants/output.vcf.gz" \
  --gvcf_outfile "/output/call_variants/output.g.vcf.gz"

log "VCF output: output/call_variants/output.vcf.gz"
