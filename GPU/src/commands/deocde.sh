#!/bin/bash
set -euo pipefail

ORIG_SCRIPT="/grand/sbi-fair/ziyu1/llama70B_infer.sh"
LOG_DIR="/grand/sbi-fair/ziyu1/decode_log"
TMP_JOB_DIR="${LOG_DIR}/tmp_jobs"

mkdir -p "$LOG_DIR"
mkdir -p "$TMP_JOB_DIR"

for max_req in 1; do
  for prompt_len in 4096; do
    job_script="${TMP_JOB_DIR}/llama70B_req${max_req}_prompt${prompt_len}.sh"

    cp "$ORIG_SCRIPT" "$job_script"

    # 替换参数
    sed -i "s/--inference-max-requests 1 \\\\/--inference-max-requests ${max_req} \\\\/" "$job_script"
    sed -i "s/--num-tokens-to-prompt 128 128 \\\\/--num-tokens-to-prompt ${prompt_len} ${prompt_len} \\\\/" "$job_script"

    # 替换日志目录
    sed -i "s|LOGDIR=/grand/sbi-fair/ziyu1/logs|LOGDIR=${LOG_DIR}|g" "$job_script"

    # 让输出日志名带参数，避免看不出是哪组
    sed -i "s|LOG_OUT=\"\${LOGDIR}/llama30B.\${PBS_JOBID}.out\"|LOG_OUT=\"\${LOGDIR}/llama70B_req${max_req}_prompt${prompt_len}.\${PBS_JOBID:-interactive}.out\"|g" "$job_script"
    sed -i "s|LOG_ERR=\"\${LOGDIR}/llama30B.\${PBS_JOBID}.err\"|LOG_ERR=\"\${LOGDIR}/llama70B_req${max_req}_prompt${prompt_len}.\${PBS_JOBID:-interactive}.err\"|g" "$job_script"

    chmod +x "$job_script"

    echo "=================================================="
    echo "Running: max_req=${max_req}, prompt_len=${prompt_len}"
    echo "Script : $job_script"
    echo "=================================================="

    bash "$job_script"
  done
done