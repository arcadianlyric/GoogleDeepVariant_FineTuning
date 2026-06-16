#!/usr/bin/env bash
# Step 0: Prepare reference files
# - Decompress FASTA if gzipped (DeepVariant requires uncompressed .fa)
# - Create .fai index if missing
# - Check BAM indexes exist

source "$(dirname "$0")/utils.sh"
load_env
ensure_output_dirs

log_step "Preparing reference files"

# Decompress FASTA if needed
if [[ "$REF_FASTA" == *.gz ]]; then
  UNZIPPED="${REF_FASTA%.gz}"
  if [[ ! -f "$UNZIPPED" ]]; then
    log "Decompressing ${REF_FASTA}..."
    run_dv_docker bash -lc "gunzip -k /reference/$(basename "$REF_FASTA")"
  else
    log "Uncompressed FASTA already exists: ${UNZIPPED}"
  fi
  # Update REF_FASTA to point to uncompressed file
  REF_FASTA="$UNZIPPED"
  log "NOTE: Update REF_FASTA in config/.env to: ${REF_FASTA}"
fi

# Create .fai index if missing
if [[ ! -f "${REF_FASTA}.fai" ]]; then
  log "Creating FASTA index..."
  run_dv_docker samtools faidx "/reference/$(basename "$REF_FASTA")"
fi

# Check BAM indexes
TRAIN_BAM="$(sample_bam "$TRAIN_SAMPLE")"
TRAIN_BAM_HOST="${PROJECT_DIR}/${TRAIN_BAM}"
if [[ ! -f "${TRAIN_BAM_HOST}.bai" ]] && [[ ! -f "${TRAIN_BAM_HOST%.bam}.bai" ]]; then
  log "Creating BAM index for ${TRAIN_BAM_HOST}..."
  run_dv_docker samtools index "$(work_path "$TRAIN_BAM")"
fi

TRUTH_VCF="$(truth_vcf)"
TRUTH_VCF_HOST="${PROJECT_DIR}/${TRUTH_VCF}"
if [[ "${TRUTH_VCF_HOST}" != *.gz && -f "${TRUTH_VCF_HOST}" && ! -f "${TRUTH_VCF_HOST}.gz" ]]; then
  log "Compressing and indexing truth VCF..."
  run_dv_docker bash -lc "bgzip -c $(work_path "$TRUTH_VCF") > $(work_path "$TRUTH_VCF").gz && tabix -p vcf $(work_path "$TRUTH_VCF").gz"
fi

# Verify all required files
log "Checking required files..."
MISSING=0
CONFIDENT_BED_HOST="${PROJECT_DIR}/$(confident_bed)"
if [[ "${TRUTH_VCF_HOST}" != *.gz && -f "${TRUTH_VCF_HOST}.gz" ]]; then
  TRUTH_VCF_HOST="${TRUTH_VCF_HOST}.gz"
fi
for f in "$REF_FASTA" "${REF_FASTA}.fai" "$TRAIN_BAM_HOST" "$TRUTH_VCF_HOST" "$CONFIDENT_BED_HOST"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    MISSING=1
  fi
done

if [[ $MISSING -eq 1 ]]; then
  echo "ERROR: Missing required files. See above." >&2
  exit 1
fi

log "All reference files ready."
