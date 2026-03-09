#!/usr/bin/env bash
# ============================================================
# Qwen3-8B experiments across all agent types
# Only GPU 6 and 7 available → run in batches of 2
#
# GPU 6 has ~73GB free → gpu_memory_utilization=0.85
# GPU 7 has ~81GB free → gpu_memory_utilization=0.95
#
# Ctrl+C will kill all child processes.
#
# Usage:
#   bash run_qwen3_8b.sh [splits]
#   default splits = human
# ============================================================

SPLITS=${1:-human}
LLM=Qwen3-8B

LOG_DIR=logs/qwen3_8b_${SPLITS}
mkdir -p ${LOG_DIR}

# ---- Trap: Ctrl+C kills all child processes ----
CHILD_PIDS=()
cleanup() {
    echo ""
    echo "[SIGINT] Killing all child processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "=============================="
echo "Running Qwen3-8B on splits: ${SPLITS}"
echo "Log dir: ${LOG_DIR}"
echo "Available GPUs: 6 (mem=0.85), 7 (mem=0.95)"
echo "=============================="

# ==================== Batch 1 ====================
echo ""
echo "[Batch 1] LLM-modulo (GPU6) + ReAct (GPU7)"
echo "----------------------------------------------"

VLLM_GPU_MEMORY_UTILIZATION=0.85 python run_exp.py \
    --splits ${SPLITS} \
    --agent LLM-modulo \
    --llm ${LLM} \
    --refine_steps 10 \
    --oracle_translation \
    --gpu 6 \
    > ${LOG_DIR}/llm_modulo.log 2>&1 &
PID1=$!
CHILD_PIDS+=($PID1)
echo "  LLM-modulo  PID=${PID1}"

VLLM_GPU_MEMORY_UTILIZATION=0.95 python run_exp.py \
    --splits ${SPLITS} \
    --agent ReAct \
    --llm ${LLM} \
    --gpu 7 \
    > ${LOG_DIR}/react.log 2>&1 &
PID2=$!
CHILD_PIDS+=($PID2)
echo "  ReAct       PID=${PID2}"

wait ${PID1}; EC1=$?
wait ${PID2}; EC2=$?
echo "[Batch 1] Done. LLM-modulo=${EC1}, ReAct=${EC2}"

# ==================== Batch 2 ====================
echo ""
echo "[Batch 2] ReAct0 (GPU6) + Act (GPU7)"
echo "----------------------------------------------"

VLLM_GPU_MEMORY_UTILIZATION=0.85 python run_exp.py \
    --splits ${SPLITS} \
    --agent ReAct0 \
    --llm ${LLM} \
    --gpu 6 \
    > ${LOG_DIR}/react0.log 2>&1 &
PID3=$!
CHILD_PIDS+=($PID3)
echo "  ReAct0      PID=${PID3}"

VLLM_GPU_MEMORY_UTILIZATION=0.95 python run_exp.py \
    --splits ${SPLITS} \
    --agent Act \
    --llm ${LLM} \
    --gpu 7 \
    > ${LOG_DIR}/act.log 2>&1 &
PID4=$!
CHILD_PIDS+=($PID4)
echo "  Act         PID=${PID4}"

wait ${PID3}; EC3=$?
wait ${PID4}; EC4=$?
echo "[Batch 2] Done. ReAct0=${EC3}, Act=${EC4}"

# ==================== Batch 3 ====================
echo ""
echo "[Batch 3] LLMNeSy (GPU6) + RuleNeSy (CPU)"
echo "----------------------------------------------"

VLLM_GPU_MEMORY_UTILIZATION=0.85 python run_exp.py \
    --splits ${SPLITS} \
    --agent LLMNeSy \
    --llm ${LLM} \
    --oracle_translation \
    --gpu 6 \
    > ${LOG_DIR}/llmnesy.log 2>&1 &
PID5=$!
CHILD_PIDS+=($PID5)
echo "  LLMNeSy     PID=${PID5}"

CUDA_VISIBLE_DEVICES="" python run_exp.py \
    --splits ${SPLITS} \
    --agent RuleNeSy \
    --llm rule \
    --oracle_translation \
    > ${LOG_DIR}/rulenesy.log 2>&1 &
PID6=$!
CHILD_PIDS+=($PID6)
echo "  RuleNeSy    PID=${PID6}"

wait ${PID5}; EC5=$?
wait ${PID6}; EC6=$?
echo "[Batch 3] Done. LLMNeSy=${EC5}, RuleNeSy=${EC6}"

# ==================== Summary ====================
echo ""
echo "=============================="
echo "All experiments completed!"
echo "  LLM-modulo : ${EC1}"
echo "  ReAct      : ${EC2}"
echo "  ReAct0     : ${EC3}"
echo "  Act        : ${EC4}"
echo "  LLMNeSy    : ${EC5}"
echo "  RuleNeSy   : ${EC6}"
echo ""
echo "Logs in: ${LOG_DIR}/"
echo "=============================="
