#!/usr/bin/env bash
# =============================================================================
# ncu_profile.sh — Generate an Nsight Compute .ncu-rep report for any binary
#
# Usage:
#   ./tools/ncu_profile.sh <binary> [output_base] [-- binary_args...]
#
# Examples:
#   # Profile all kernels, output to reports/elementwise.ncu-rep
#   ./tools/ncu_profile.sh ./build/elementwise/elementwise
#
#   # Profile a specific kernel (-k 2) and custom output name
#   ./tools/ncu_profile.sh ./build/elementwise/elementwise my_report -- --kernel 2
#
#   # Profile the transpose binary with a 2048x2048 matrix
#   ./tools/ncu_profile.sh ./build/transpose/transpose transpose_2k -- --M 2048 --N 2048
#
# The .ncu-rep file can be opened with Nsight Compute (ncu-ui) for analysis.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <binary> [output_base] [-- binary_args...]"
    exit 1
fi

BINARY="$1"
shift

# Second positional arg is optional output base name
OUTPUT_BASE=""
if [[ $# -gt 0 && "$1" != "--" ]]; then
    OUTPUT_BASE="$1"
    shift
fi

# Remaining args after '--' are passed to the binary
BINARY_ARGS=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
    BINARY_ARGS=("$@")
fi

# Default output name: derived from binary name
if [[ -z "$OUTPUT_BASE" ]]; then
    OUTPUT_BASE="$(basename "${BINARY}")"
fi

# Output directory
REPORT_DIR="$(dirname "$0")/../ncu-reports"
mkdir -p "${REPORT_DIR}"
OUTPUT_FILE="${REPORT_DIR}/${OUTPUT_BASE}"

# --------------------------------------------------------------------------
# Run ncu
# --------------------------------------------------------------------------
echo "============================================================"
echo "  Binary : ${BINARY}"
echo "  Args   : ${BINARY_ARGS[*]:-<none>}"
echo "  Report : ${OUTPUT_FILE}.ncu-rep"
echo "============================================================"

ncu \
    --set full \
    --export "${OUTPUT_FILE}" \
    --force-overwrite \
    --target-processes all \
    "${BINARY}" "${BINARY_ARGS[@]:-}"

echo ""
echo "Done.  Open '${OUTPUT_FILE}.ncu-rep' in Nsight Compute (ncu-ui)."
