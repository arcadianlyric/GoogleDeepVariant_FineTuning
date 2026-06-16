#!/usr/bin/env bash
# Step 1: Generate TFRecord examples from one BAM.
# Default TRAIN_SAMPLE=pe150. Set TRAIN_SAMPLE=se600 to use the SE600 BAM.

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

REF_BASENAME=$(basename "${REF_FASTA}")
REGION=$(yaml_val "$(config_file)" "train_region")
SAMPLE="${TRAIN_SAMPLE}"
BAM_REL="$(sample_bam "$SAMPLE")"
TRUTH_REL="$(truth_vcf)"
CONF_BED_REL="$(confident_bed)"
EXAMPLES="/output/examples/${SAMPLE}/examples.tfrecord.gz"
if [[ ! -f "${PROJECT_DIR}/${TRUTH_REL}" && -f "${PROJECT_DIR}/${TRUTH_REL}.gz" ]]; then
  TRUTH_REL="${TRUTH_REL}.gz"
fi

if [[ "$(yaml_val "$(config_file)" "samples.${SAMPLE}.channel_insert_size")" == "true" ]]; then
  CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref,insert_size"
else
  CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref"
fi

log_step "make_examples: ${SAMPLE} (${REGION})"
log "BAM: ${BAM_REL}"
log "Truth: ${TRUTH_REL}"
log "Confident BED: ${CONF_BED_REL}"
log "Channels: ${CHANNEL_LIST}"

SE600_FLAGS=()
if [[ "$SAMPLE" == "se600" ]]; then
  SE600_FLAGS=(--norealign_reads --min_mapping_quality=0)
fi

run_dv_docker \
  /opt/deepvariant/bin/make_examples \
  --mode training \
  --ref "/reference/${REF_BASENAME}" \
  --reads "$(work_path "$BAM_REL")" \
  --truth_variants "$(work_path "$TRUTH_REL")" \
  --confident_regions "$(work_path "$CONF_BED_REL")" \
  --examples "${EXAMPLES}" \
  --regions "${REGION}" \
  --channel_list="${CHANNEL_LIST}" \
  --keep_supplementary_alignments \
  --normalize_reads \
  --max_reads_per_partition 600 \
  "${SE600_FLAGS[@]}" \
  --task 0

log "make_examples completed for ${SAMPLE}."
