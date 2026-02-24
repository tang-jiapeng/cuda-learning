#!/usr/bin/env bash
# =============================================================================
# ncu_profile_kernel.sh — Deep per-kernel profiling (single kernel, rich metrics)
#
# Compared to ncu_profile.sh (which uses --set full for all kernels), this
# script profiles ONE specific kernel index with a hand-picked metric set that
# is valuable when optimising memory-bound CUDA kernels.
#
# Usage:
#   ./tools/ncu_profile_kernel.sh <binary> <kernel_index> [output_base] [-- extra_args...]
#
# Examples:
#   # Profile kernel 3 of elementwise
#   ./tools/ncu_profile_kernel.sh ./build/elementwise/elementwise 3
#
#   # Profile kernel 5 of reduce with 32M elements
#   ./tools/ncu_profile_kernel.sh ./build/reduce/reduce 5 reduce_k5 -- --N 33554432
#
#   # Profile kernel 4 of transpose (shared-memory version)
#   ./tools/ncu_profile_kernel.sh ./build/transpose/transpose 4 transpose_shared
# =============================================================================

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <binary> <kernel_index> [output_base] [-- extra_args...]"
    exit 1
fi

BINARY="$1"
KERNEL_IDX="$2"
shift 2

OUTPUT_BASE=""
if [[ $# -gt 0 && "$1" != "--" ]]; then
    OUTPUT_BASE="$1"
    shift
fi

EXTRA_ARGS=()
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

if [[ -z "$OUTPUT_BASE" ]]; then
    OUTPUT_BASE="$(basename "${BINARY}")_k${KERNEL_IDX}"
fi

REPORT_DIR="$(dirname "$0")/../ncu-reports"
mkdir -p "${REPORT_DIR}"
OUTPUT_FILE="${REPORT_DIR}/${OUTPUT_BASE}"

# --------------------------------------------------------------------------
# Metric groups for memory-bound kernel analysis
#
#   Memory throughput, L2/L1 cache hit rates, warp occupancy, SM utilisation,
#   memory access pattern (replay), bank conflicts, instruction statistics.
# --------------------------------------------------------------------------
METRICS=(
    # Memory throughput
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second"
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second"
    "dram__bytes_read.sum.per_second"
    "dram__bytes_write.sum.per_second"

    # Achieved occupancy & SM utilisation
    "sm__warps_active.avg.pct_of_peak_sustained_active"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"

    # Cache efficiency
    "l1tex__t_sector_hit_rate.pct"
    "lts__t_sector_hit_rate.pct"

    # Memory replay (uncoalesced access indicator)
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"

    # Shared memory bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"

    # Warp stall reasons
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_membar_per_warp_active.pct"
)

# Build comma-separated metric list
METRIC_STR="$(printf '%s,' "${METRICS[@]}")"
METRIC_STR="${METRIC_STR%,}"  # strip trailing comma

echo "============================================================"
echo "  Binary : ${BINARY}"
echo "  Kernel : ${KERNEL_IDX}"
echo "  Args   : ${EXTRA_ARGS[*]:-<none>}"
echo "  Report : ${OUTPUT_FILE}.ncu-rep"
echo "============================================================"

ncu \
    --metrics "${METRIC_STR}" \
    --export "${OUTPUT_FILE}" \
    --force-overwrite \
    --target-processes all \
    "${BINARY}" --kernel "${KERNEL_IDX}" "${EXTRA_ARGS[@]:-}"

echo ""
echo "Done.  Open '${OUTPUT_FILE}.ncu-rep' in Nsight Compute (ncu-ui)."
