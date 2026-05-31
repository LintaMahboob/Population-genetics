#!/usr/bin/env bash
# ============================================================================
# Script 01: Quality Control and IBS Distance Computation Pipeline
# ----------------------------------------------------------------------------
# Author      : Linta Mahboob
# Institution : Moscow Institute of Physics and Technology (MIPT)
# Supervisor  : Prof. Dr. Oleg Balanovsky, Vavilov Institute of General Genetics
# Date        : 2021 (analysis); 2025 (documented and reproduced)
# ----------------------------------------------------------------------------
# Description :
#   This script reproduces the full quality control (QC) pipeline applied to
#   the raw genotype dataset of 2,500 South, West, and Central Asian individuals.
#   It uses SAMtools/BCFtools for file conversion and merging, and PLINK v1.9
#   for QC filtering and IBS distance matrix computation.
#
# Input files required:
#   - Raw VCF/BCF or PLINK binary files (.bed/.bim/.fam) per cohort
#   - A merge list file: merge_list.txt
#
# Output files:
#   - final_dataset.bed / .bim / .fam     (QC-passed genotype data)
#   - final_dataset.mdist                 (pairwise IBS distance matrix)
#   - final_dataset.mdist.id              (sample IDs matching matrix rows)
#   - final_dataset_QC_summary.log        (QC statistics)
#
# Software dependencies:
#   - SAMtools >= 1.12   (https://samtools.github.io)
#   - BCFtools >= 1.12   (https://samtools.github.io/bcftools)
#   - PLINK v1.9         (https://www.cog-genomics.org/plink)
#
# Usage:
#   chmod +x 01_quality_control_pipeline.sh
#   ./01_quality_control_pipeline.sh
# ============================================================================

set -euo pipefail   # Exit on error, undefined variable, or pipe failure
IFS=$'\n\t'         # Safer word splitting

# ── Directory structure ──────────────────────────────────────────────────────
RAW_DIR="./raw_data"          # Directory containing per-cohort raw files
WORK_DIR="./working"          # Intermediate files
OUT_DIR="./output"            # Final outputs
LOG_DIR="./logs"              # Log files

mkdir -p "$RAW_DIR" "$WORK_DIR" "$OUT_DIR" "$LOG_DIR"

LOGFILE="${LOG_DIR}/QC_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1   # Tee all output to log

echo "============================================================"
echo " Quality Control Pipeline — South/West/Central Asian Dataset"
echo " Started: $(date)"
echo "============================================================"

# ── Parameters ───────────────────────────────────────────────────────────────
# These thresholds were applied in the original analysis
GENO_THRESH=0.05        # Exclude SNPs with >5% missing genotype rate
MIND_THRESH=0.10        # Exclude samples with >10% missing genotype rate
MAF_THRESH=0.01         # Exclude SNPs with minor allele frequency < 1%
HWE_THRESH=1e-6         # Hardy-Weinberg equilibrium p-value threshold
THREADS=4               # Number of CPU threads for PLINK

PREFIX_MERGED="${WORK_DIR}/merged_raw"
PREFIX_QC="${OUT_DIR}/final_dataset"

echo ""
echo "── STEP 1: Convert VCF files to PLINK binary format ──────────────────"
# For each cohort VCF file, convert to PLINK binary format
# Adjust the glob pattern to match your actual file naming convention
for vcf_file in "${RAW_DIR}"/*.vcf.gz; do
    cohort=$(basename "$vcf_file" .vcf.gz)
    echo "  Converting: ${cohort}"

    plink \
        --vcf "$vcf_file" \
        --make-bed \
        --out "${WORK_DIR}/${cohort}" \
        --double-id \
        --allow-extra-chr \
        --threads "$THREADS" \
        2>> "${LOG_DIR}/${cohort}_convert.log"

    echo "    Done: ${cohort}.bed/.bim/.fam"
done

echo ""
echo "── STEP 2: Merge all cohorts into one dataset ─────────────────────────"
# Build merge list (all cohorts except the first, which is the base)
# Format: one line per cohort: path/to/file.bed path/to/file.bim path/to/file.fam

MERGE_LIST="${WORK_DIR}/merge_list.txt"
> "$MERGE_LIST"   # Empty the file

first_cohort=""
for bed_file in "${WORK_DIR}"/*.bed; do
    cohort_prefix="${bed_file%.bed}"
    if [ -z "$first_cohort" ]; then
        first_cohort="$cohort_prefix"
    else
        echo "${cohort_prefix}.bed ${cohort_prefix}.bim ${cohort_prefix}.fam" >> "$MERGE_LIST"
    fi
done

echo "  Base dataset : ${first_cohort}"
echo "  Merge list   : ${MERGE_LIST}"
echo "  Cohorts to merge: $(wc -l < "$MERGE_LIST")"

plink \
    --bfile "$first_cohort" \
    --merge-list "$MERGE_LIST" \
    --make-bed \
    --out "$PREFIX_MERGED" \
    --allow-no-sex \
    --threads "$THREADS"

echo "  Merged dataset: $(wc -l < "${PREFIX_MERGED}.fam") individuals, $(wc -l < "${PREFIX_MERGED}.bim") SNPs"

echo ""
echo "── STEP 3: Resolve merge conflicts (missnp / strand flips) ───────────"
# If PLINK reports missnp issues, flip and retry
if [ -f "${PREFIX_MERGED}-merge.missnp" ]; then
    echo "  Missnp file detected — flipping strands and retrying merge..."

    # Flip SNPs in all secondary cohort files
    while IFS= read -r line; do
        cohort_prefix=$(echo "$line" | awk '{print $1}' | sed 's/.bed//')
        plink \
            --bfile "$cohort_prefix" \
            --flip "${PREFIX_MERGED}-merge.missnp" \
            --make-bed \
            --out "${cohort_prefix}_flipped" \
            --threads "$THREADS"
        sed -i "s|${cohort_prefix}|${cohort_prefix}_flipped|" "$MERGE_LIST"
    done < "$MERGE_LIST"

    # Retry merge
    plink \
        --bfile "$first_cohort" \
        --merge-list "$MERGE_LIST" \
        --make-bed \
        --out "${PREFIX_MERGED}_final" \
        --allow-no-sex \
        --threads "$THREADS"

    PREFIX_MERGED="${PREFIX_MERGED}_final"
fi

echo ""
echo "── STEP 4: Quality Control Filtering ──────────────────────────────────"
echo "  Parameters applied:"
echo "    --geno  ${GENO_THRESH}   (SNP missingness)"
echo "    --mind  ${MIND_THRESH}   (sample missingness)"
echo "    --maf   ${MAF_THRESH}    (minor allele frequency)"
echo "    --hwe   ${HWE_THRESH}    (Hardy-Weinberg equilibrium)"

# Step 4a: Remove duplicate variant positions
plink \
    --bfile "$PREFIX_MERGED" \
    --list-duplicate-vars ids-only suppress-first \
    --out "${WORK_DIR}/duplicates" \
    --threads "$THREADS"

STEP4A="${WORK_DIR}/step4a_nodup"
plink \
    --bfile "$PREFIX_MERGED" \
    --exclude "${WORK_DIR}/duplicates.dupvar" \
    --make-bed \
    --out "$STEP4A" \
    --threads "$THREADS"

echo "  After removing duplicate SNPs: $(wc -l < "${STEP4A}.bim") SNPs"

# Step 4b: Apply all main QC filters
plink \
    --bfile "$STEP4A" \
    --geno "$GENO_THRESH" \
    --mind "$MIND_THRESH" \
    --maf "$MAF_THRESH" \
    --hwe "$HWE_THRESH" \
    --make-bed \
    --out "$PREFIX_QC" \
    --allow-no-sex \
    --threads "$THREADS"

N_INDIV=$(wc -l < "${PREFIX_QC}.fam")
N_SNPS=$(wc -l < "${PREFIX_QC}.bim")
echo ""
echo "  ✓ QC complete:"
echo "    Individuals retained : ${N_INDIV}"
echo "    SNPs retained        : ${N_SNPS}"

echo ""
echo "── STEP 5: Compute pairwise IBS distance matrix ───────────────────────"
# PLINK computes identity-by-state (IBS) distances for all pairs of individuals
# Output: .mdist (N x N distance matrix) and .mdist.id (sample ID list)
#
# --distance ibs : compute IBS (not genomic relationship matrix)
# The diagonal of the matrix is 0 (identity with self)

plink \
    --bfile "$PREFIX_QC" \
    --distance ibs \
    --out "$PREFIX_QC" \
    --allow-no-sex \
    --threads "$THREADS"

echo "  ✓ IBS distance matrix: ${PREFIX_QC}.mdist"
echo "    Dimensions: ${N_INDIV} x ${N_INDIV}"

echo ""
echo "── STEP 6: Generate QC summary report ─────────────────────────────────"
BEFORE_N=2500
AFTER_N=$N_INDIV
REMOVED=$((BEFORE_N - AFTER_N))

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║       QUALITY CONTROL SUMMARY REPORT     ║"
echo "  ╠══════════════════════════════════════════╣"
echo "  ║  Individuals before QC : ${BEFORE_N}              ║"
echo "  ║  Individuals after QC  : ${AFTER_N}              ║"
echo "  ║  Individuals removed   : ${REMOVED}                ║"
echo "  ║  Final SNP count       : ${N_SNPS}         ║"
echo "  ║  Geno threshold        : ${GENO_THRESH}             ║"
echo "  ║  Mind threshold        : ${MIND_THRESH}             ║"
echo "  ║  MAF threshold         : ${MAF_THRESH}            ║"
echo "  ║  HWE p-value cutoff    : ${HWE_THRESH}          ║"
echo "  ╚══════════════════════════════════════════╝"

echo ""
echo "── PIPELINE COMPLETE ───────────────────────────────────────────────────"
echo "  Finished: $(date)"
echo "  Output files:"
echo "    ${PREFIX_QC}.bed"
echo "    ${PREFIX_QC}.bim"
echo "    ${PREFIX_QC}.fam"
echo "    ${PREFIX_QC}.mdist"
echo "    ${PREFIX_QC}.mdist.id"
echo "    ${LOGFILE}"
echo "============================================================"
