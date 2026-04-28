#!/bin/sh
#PBS -l select=8:system=polaris
#PBS -q preemptable
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -A sbi-fair
##PBS -j oe
##PBS -o /grand/sbi-fair/ziyu1/logs
# === Works but not perfectly ===


set -euo pipefail

MEGATRON_DIR=/grand/sbi-fair/ziyu1/Megatron-LM
cd "$MEGATRON_DIR"
export PBS_O_WORKDIR="$(realpath .)"

# make sure dataset is in data/

# LD_PRELOAD= \
unset LD_PRELOAD; export LD_PRELOAD=


ml use /soft/modulefiles
ml spack-pe-base/0.8.1
ml use /soft/spack/testing/0.8.1/modulefiles
ml apptainer/main
ml load e2fsprogs

# Avoid OSError: AF_UNIX path too long
export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
export TORCH_COMPILE_DISABLE=1
export BASE_SCRATCH_DIR=/local/scratch/
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR
export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR


# Proxy setup for internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Environment variables for MPI
export ADDITIONAL_PATH=/opt/cray/pe/pals/1.2.12/lib
module load cray-mpich-abi

# For NCCL
module load libfabric

# Set MPI ranks
export NODES=$(wc -l < "$PBS_NODEFILE")
GPUS_PER_NODE=4
export TOTAL_GPUS=$(( NODES * GPUS_PER_NODE ))
export PPN=1
export PROCS=$((NODES * PPN))
echo "NUM_OF_NODES=${NODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, TOTAL_GPUS=${TOTAL_GPUS}, RANKS_PER_NODE=${PPN}, TOTAL_NUM_RANKS=${PROCS}"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # RuntimeError: Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1


# ---- NCCL / OFI envs ----
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} # WARN if you don't want the noise
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=0

# Toggle AWS OFI NCCL plugin ON (1) / OFF (0)
PLUGIN=${PLUGIN:-0}

if [[ "$PLUGIN" == "1" ]]; then
  echo "[INFO] Enabling AWS OFI NCCL Plugin"
    unset NCCL_CROSS_NIC NCCL_COLLNET_ENABLE \
        NCCL_SOCKET_IFNAME NCCL_NSOCKS_PERTHREAD NCCL_SOCKET_NTHREADS

    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}  
    export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws-libfabric-1.22.0/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:$LD_LIBRARY_PATH # for libfabric
    export LD_LIBRARY_PATH=/eagle/lc-mpi/bharadhwaj/containers/cxi-shim:$LD_LIBRARY_PATH # for libcxi.so.1
    export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

    # echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    # Don't forget to re-export the LD path below inside the container
    export NCCL_NET="AWS Libfabric" # Use OFI if using some other version of libfabric
    export NCCL_OFI_LOG_LEVEL=${NCCL_OFI_LOG_LEVEL:-INFO} # TRACE if having issues
    export FI_PROVIDER=${FI_PROVIDER:-cxi}
    export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export NCCL_COLLNET_ENABLE=0

    # export NCCL_NET_GDR_LEVEL=PHB
    # export NCCL_CROSS_NIC=1
    # export NCCL_COLLNET_ENABLE=1
    # export FI_CXI_DISABLE_HOST_REGISTER=1
    # export FI_MR_CACHE_MONITOR=userfaultfd
    # export FI_CXI_DEFAULT_CQ_SIZE=131072
    # # export NCCL_SOCKET_IFNAME="hsn,bond0"
    # export NCCL_SOCKET_IFNAME=hsn,ib0,ib1
    # export NCCL_NSOCKS_PERTHREAD=4
    # export NCCL_SOCKET_NTHREADS=2
else
  echo "[INFO] Plugin disabled (fallback)."
  unset NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET
  unset FI_PROVIDER FI_CXI_DISABLE_HOST_REGISTER FI_MR_CACHE_MONITOR FI_CXI_DEFAULT_CQ_SIZE
fi

PROFILE=${PROFILE:-0}
if [[ "$PROFILE" == "1" ]]; then
    echo "[INFO] Profiling enabled"
    export NSYS_MPI_STORE_TEAMS_PER_RANK=1
    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin:$PATH
else
    echo "[INFO] Profiling not enabled"
fi

# Do not force NVTE attention mode via env; let args control it
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN


# ---- Config ----
export RDZV_HOST=$(head -n1 "$PBS_NODEFILE")
export RDZV_PORT=${RDZV_PORT:-29400}
CONTAINER="/grand/sbi-fair/ziyu1/Container/devel-cudnn.sif"
# Bind host paths into the container. Include /grand so absolute paths work inside.
BIND="-B /opt -B /var/run/palsd -B /soft -B /etc/alternatives -B /lus -B /eagle -B /grand"


echo "[INFO] Using container: $CONTAINER"
echo "[INFO] Env preview:"
echo "  NCCL_DEBUG=$NCCL_DEBUG  PLUGIN=$PLUGIN  PPN=$PPN   "


# ---- Preprocess Data ---- (don't have to do it every time)
# -- MAX seq_len is 1024. DO NOT EXCEED IT bec gpt2 tokenizer
# -- set --max-position-embeddings the same
#   # 1) Fetch TinyStories with HF datasets and write JSONL
#   # 2) Megatron-LM preprocessing
# PREPROCESS_DATA=${PREPROCESS_DATA:-0}
# # Align all these  to the leftmost indent. Do not tab anything
# if [[ "$PREPROCESS_DATA" == "1" ]]; then
# echo "Preprocessing Data..."
# apptainer exec --fakeroot --nv $BIND "$CONTAINER" bash -lc "
# set -euo pipefail
# cd /path/to/Megatron-LM
# mkdir -p data
# python3 - <<'PY'
# from datasets import load_dataset
# ds = load_dataset('roneneldan/TinyStories', split='train')
# if 'text' not in ds.column_names:
#     if 'content' in ds.column_names:
#         ds = ds.rename_column('content', 'text')
#     else:
#         raise SystemExit(f\"Expected 'text' or 'content' in columns: {ds.column_names}\")
# out = 'data/corpus.jsonl'
# ds.to_json(out, lines=True, orient='records', force_ascii=False)
# print(f'Wrote {out} with {len(ds)} rows')
# PY
# python3 tools/preprocess_data.py \
#   --input data/corpus.jsonl \
#   --tokenizer-type HuggingFaceTokenizer \
#   --tokenizer-model gpt2 \
#   --output-prefix data/corpus_tokenized \
#   --append-eod \
#   --workers 4
# "
# else


# Launch: MPICH places processes; torch uses NCCL for comms

# Ensure a tiny local dataset exists to avoid network downloads
if [[ ! -f "$MEGATRON_DIR/data/corpus_tokenized_text_document.bin" ]]; then
  echo "[INFO] Preparing tiny sample dataset under $MEGATRON_DIR/data"
  apptainer exec --fakeroot --nv $BIND "$CONTAINER" bash -lc '
    set -euo pipefail
    source /usr/local/venv/bin/activate
    cd /grand/sbi-fair/ziyu1/Megatron-LM
    export PYTHONPATH=$PWD:$PYTHONPATH
    mkdir -p data
    python3 - <<"PY"
import json, os
out = os.path.join("data", "corpus.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for i in range(1000):
        rec = {"text": f"This is a tiny training sample number {i}."}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Wrote {out}")
PY
    PYTHONPATH=$PWD:$PYTHONPATH python3 tools/preprocess_data.py \
      --input data/corpus.jsonl \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model gpt2 \
      --output-prefix data/corpus_tokenized \
      --append-eod \
      --workers 2
  '
else
  echo "[INFO] Reusing existing dataset at $MEGATRON_DIR/data"
fi

# Tried:
# mpiexec nsys apptainer torchrun script - fails at gpu-metrics=all
# mpiexec apptainer nsys torchrun script - appears to work. There is a runtime error but that happens after the .qdrep is generated
#    check report # 6585909 and the 2 before it
# mpiexec apptainer torchrun nsys script - works. There is a rendezvous error when more than 2 nodes are run but everything looks perfect

LOGDIR=/grand/sbi-fair/ziyu1/logs
mkdir -p "$LOGDIR"

LOG_OUT="${LOGDIR}/llama30B.${PBS_JOBID}.out"
LOG_ERR="${LOGDIR}/llama30B.${PBS_JOBID}.err"

OVERLAY_DIR="$HOME/overlays"
OVERLAY_IMG="$OVERLAY_DIR/py_overlay.img"

# 只用 1 个进程初始化 overlay + 安装 simpy（避免并发写坏）
mpiexec -hostfile "$PBS_NODEFILE" -n 1 -ppn 1 bash -lc "
set -euo pipefail
mkdir -p '$OVERLAY_DIR'

# overlay 不存在就创建
if [ ! -f '$OVERLAY_IMG' ]; then
  apptainer overlay create --size 4096 '$OVERLAY_IMG'
fi

# 进容器检查 simpy；没有才安装（不升级 pip）
apptainer exec --fakeroot --nv \
  --overlay '$OVERLAY_IMG' \
  $BIND '$CONTAINER' bash -lc '
    set -euo pipefail
    source /usr/local/venv/bin/activate
    python -c \"import simpy; print(\\\"[INFO] simpy already present\\\", simpy.__version__)\" \
      || (python -m pip install --no-cache-dir simpy && python -c \"import simpy; print(\\\"[INFO] simpy installed\\\", simpy.__version__)\" )
  '
"


mpiexec -hostfile "$PBS_NODEFILE" -n "$NODES" -ppn 1 \
  apptainer exec --fakeroot --nv \
  --overlay "$OVERLAY_IMG"  \
  $BIND "$CONTAINER" bash -lc '
set -euo pipefail



# ====== 容器里环境 ======
export LD_PRELOAD=
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda
export PATH=$CUDA_HOME/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin:$PATH
export PYTHONPATH=/grand/sbi-fair/ziyu1/Megatron-LM:$PYTHONPATH

source /usr/local/venv/bin/activate

cd /grand/sbi-fair/ziyu1/Megatron-LM/examples/inference/gpt

# ====== 关键：告诉 torchrun 这是第几个 node ======
# 对 cray-mpich 来说一般有 PMI_RANK，如果不放心可以 echo 看一下
NODE_RANK=${PMI_RANK:-0}

echo "[INFO] Inside container: PMI_RANK=$PMI_RANK, NODE_RANK=$NODE_RANK"

# 可选：显式设一个 rdzv_id，保证所有节点一致且不会跟别的 job 撞
RDZV_ID=${PBS_JOBID:-megatron_test_run}
# PROMPTS="What is the capital of France?"
# TOKENS_TO_GENERATE=4
# MAX_BATCH_SIZE=2

torchrun \
  --nproc_per_node=${PPN} \
  --nnodes=${NODES} \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${RDZV_HOST}:${RDZV_PORT} \
  --rdzv_id=${RDZV_ID} \
  gpt_static_inference.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    \
    --max-position-embeddings 4096 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 32 \
    --no-gradient-accumulation-fusion\
    \
    --max-position-embeddings 12000 \
    --position-embedding-type rope \
    --no-position-embedding \
    \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --bf16 \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /grand/sbi-fair/ziyu1/llama2/tokenizer.model \
    --vocab-size 32000 \
    --seq-length 10000 \
    --attention-backend unfused \
    --num-tokens-to-generate 0 \
    --log-throughput \
    --incoming-requests-per-sec 1000 \
    --incoming-requests-duration 1 \
    --inference-max-requests 4 \
    --num-tokens-to-prompt 2048 2048 \
    --inference-max-seq-length 10000 \
' 1> "$LOG_OUT" 2> "$LOG_ERR"


COREDIR="/grand/sbi-fair/ziyu1/Megatron-LM"
echo "[INFO] Cleaning core* files in $COREDIR"
find "$COREDIR" -maxdepth 1 -type f -name 'core*' -print -delete
echo "[MPIEXEC] JOB ENDED"
