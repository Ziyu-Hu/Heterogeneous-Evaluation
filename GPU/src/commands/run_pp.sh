# Sweep batch sizes
MICRO_BATCH_SIZES=(1 2 4 8 16 32)
GLOBAL_BATCH_SIZES=(1024)

LOGDIR=/grand/sbi-fair/ziyu1/logs
mkdir -p "$LOGDIR"

for MBS in "${MICRO_BATCH_SIZES[@]}"; do
  for GBS in "${GLOBAL_BATCH_SIZES[@]}"; do

    # Skip invalid cases: global batch should be divisible by micro_batch * data_parallel_size.
    # Here TP=4, PP=1, so DP = TOTAL_GPUS / (TP * PP)
    TP=4
    PP=1
    DP=$(( TOTAL_GPUS / (TP * PP) ))
    DENOM=$(( MBS * DP ))

    if (( GBS % DENOM != 0 )); then
      echo "[SKIP] micro_batch=$MBS global_batch=$GBS not divisible by MBS*DP=$DENOM"
      continue
    fi

    echo "[RUN] micro_batch=$MBS global_batch=$GBS"

    LOG_OUT="${LOGDIR}/llama13B_mbs${MBS}_gbs${GBS}.${PBS_JOBID}.out"
    LOG_ERR="${LOGDIR}/llama13B_mbs${MBS}_gbs${GBS}.${PBS_JOBID}.err"

    mpiexec -hostfile "$PBS_NODEFILE" -n "$NODES" -ppn 1 \
      apptainer exec --fakeroot --nv $BIND "$CONTAINER" bash -lc "
set -euo pipefail

export LD_PRELOAD=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda
export PATH=\$CUDA_HOME/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin:\$PATH
export PYTHONPATH=/grand/sbi-fair/ziyu1/Megatron-LM:\$PYTHONPATH

source /usr/local/venv/bin/activate
cd /grand/sbi-fair/ziyu1/Megatron-LM

NODE_RANK=\${PMI_RANK:-0}
RDZV_ID=${PBS_JOBID}_mbs${MBS}_gbs${GBS}

torchrun \
  --nproc_per_node=${PPN} \
  --nnodes=${NODES} \
  --node_rank=\${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${RDZV_HOST}:${RDZV_PORT} \
  --rdzv_id=\${RDZV_ID} \
  pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size 1 \
    --num-layers 3 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --swiglu \
    --disable-bias-linear \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 100 \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --no-masked-softmax-fusion \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --weight-decay 1.0e-1 \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --split 99990,8,2 \
    --vocab-size 32000 \
    --dataloader-type cyclic \
    --save-interval 200000 \
    --log-interval 1 \
    --timing-log-level 2 \
    --timing-log-option minmax \
    --eval-interval 320000 \
    --eval-iters 1 \
    --num-workers 8 \
    --log-throughput \
    --no-save-optim \
    --no-save-rng \
    --group-query-attention \
    --num-query-groups 40 \
    --no-gradient-accumulation-fusion \
    --distributed-backend nccl \
    --distributed-timeout-minutes 120 \
    --overlap-grad-reduce \
    --attention-backend unfused
" 1> "$LOG_OUT" 2> "$LOG_ERR"

  done
done