#!/usr/bin/env bash
set -euo pipefail

BASE="/grand/sbi-fair/ziyu1/llama70B_infer.sh"
TMPDIR="/grand/sbi-fair/ziyu1/tmp_sweep_scripts"
LOGDIR="/grand/sbi-fair/ziyu1/logs_sweep"
mkdir -p "$TMPDIR" "$LOGDIR"

SQ_LIST=(512 1024)
BS_LIST=(1 2 4 8)
# 更细：mapfile -t SQ_LIST < <(seq 128 64 1024)

for sq in "${SQ_LIST[@]}"; do
  for bs in "${BS_LIST[@]}"; do
    tmp="${TMPDIR}/llama70B_sq${sq}_bs${bs}.sh"

    # 我们用 3 个文件：stdout、stderr、合并后的总 log
    out_prefix="${LOGDIR}/log70B_sq${sq}_bs${bs}"
    out="${out_prefix}.txt"
    out_stdout="${out_prefix}.out"
    out_stderr="${out_prefix}.err"

    cp "$BASE" "$tmp"

    # 1) sweep 参数
    sed -i -E "s/(--inference-max-requests)[[:space:]]+[0-9]+/\1 ${bs}/g" "$tmp"
    sed -i -E "s/(--num-tokens-to-prompt)[[:space:]]+[0-9]+[[:space:]]+[0-9]+/\1 ${sq} ${sq}/g" "$tmp"

    # 2) 强制脚本内部把 LOG_OUT/LOG_ERR 写到我们指定的文件
    sed -i -E "s|^LOG_OUT=.*$|LOG_OUT=\"${out_stdout}\"|g" "$tmp"
    sed -i -E "s|^LOG_ERR=.*$|LOG_ERR=\"${out_stderr}\"|g" "$tmp"

    chmod +x "$tmp"

    echo "[RUN] sq=${sq} bs=${bs}"
    bash "$tmp"

    # 3) 合并 stdout+stderr 到单个 txt（保留原始两个文件，方便你排查）
    {
      echo "===== META ====="
      echo "sq=${sq} bs=${bs}"
      echo "tmp_script=${tmp}"
      echo "stdout_file=${out_stdout}"
      echo "stderr_file=${out_stderr}"
      echo "===== STDOUT ====="
      [[ -f "${out_stdout}" ]] && cat "${out_stdout}" || echo "[WARN] stdout file missing"
      echo
      echo "===== STDERR ====="
      [[ -f "${out_stderr}" ]] && cat "${out_stderr}" || echo "[WARN] stderr file missing"
    } > "${out}"

    echo "[DONE] -> ${out}"
  done
done