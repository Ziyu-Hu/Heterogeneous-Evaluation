#!/bin/bash
set -euo pipefail

ORIG_SCRIPT="/grand/sbi-fair/ziyu1/llama7B_infer.sh"
SWEEP_DIR="/grand/sbi-fair/ziyu1/decode_log"
TMP_JOB_DIR="${SWEEP_DIR}/tmp_jobs"

mkdir -p "$SWEEP_DIR"
mkdir -p "$TMP_JOB_DIR"

for max_req in 1 2 3 4; do
  for prompt_len in 128 256 512 1024; do
    job_script="${TMP_JOB_DIR}/llama7B_req${max_req}_prompt${prompt_len}.sh"

    cp "$ORIG_SCRIPT" "$job_script"

    # 1) 替换 inference-max-requests
    sed -i "s/--inference-max-requests 1 \\\\/--inference-max-requests ${max_req} \\\\/" "$job_script"

    # 2) 替换 num-tokens-to-prompt
    sed -i "s/--num-tokens-to-prompt 128 128 \\\\/--num-tokens-to-prompt ${prompt_len} ${prompt_len} \\\\/" "$job_script"

    # 3) 替换脚本内部日志目录
    sed -i "s|LOGDIR=/grand/sbi-fair/ziyu1/logs|LOGDIR=${SWEEP_DIR}|g" "$job_script"

    chmod +x "$job_script"

    echo "Submitting: max_req=${max_req}, prompt_len=${prompt_len}"
    qsub "$job_script"
  done
done