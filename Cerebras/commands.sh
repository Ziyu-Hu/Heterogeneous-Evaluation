#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="configs/params_llama2_7b.yaml"   # 原始配置（不修改）
OUT_ROOT="pp_bs4"                              # cszoo fit 里传给 --model_dir 的相对目录
LOG_DIR="logs"

# 你要检查/删除 checkpoint 的“真实绝对路径根”
ABS_PP_ROOT="/home/ziyu/R_2.6.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama/pp"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

for L in $(seq 2 2 32); do
  RUN_CFG="$(mktemp --suffix=".yaml")"

  # 替换第一个 num_hidden_layers:
  sed -E "0,/^([[:space:]]*num_hidden_layers:)[[:space:]]*[0-9]+/s//\1 ${L}/" \
    "$BASE_CFG" > "$RUN_CFG"

  MODEL_DIR_REL="${OUT_ROOT}/model_dir_llama2_7b_${L}layer"
  LOG_FILE="${LOG_DIR}/llama2_7b_${L}layer.log"

  echo "===== Running ${L} layers ====="
  echo "Config: $RUN_CFG"
  echo "Model dir: $MODEL_DIR_REL"
  echo "Log: $LOG_FILE"

  # 运行训练
  cszoo fit "$RUN_CFG" \
    --job_labels "name=llama2_7b_${L}layer" \
    --model_dir "$MODEL_DIR_REL" |& tee "$LOG_FILE"

  rm -f "$RUN_CFG"
  prev=$((L-2))
  # 训练结束后检查 checkpoint_200.mdl：存在则删除
  CKPT="${ABS_PP_ROOT}/model_dir_llama2_7b_${prev}layer/checkpoint_100.mdl"
  if [[ -f "$CKPT" ]]; then
    echo "[CLEANUP] Found checkpoint: $CKPT"
    rm -f "$CKPT"
    echo "[CLEANUP] Deleted."
  else
    echo "[CLEANUP] No checkpoint_200.mdl at: $CKPT"
  fi
done
