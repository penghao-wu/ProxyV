!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export WANDB_RESUME="allow"

export WANDB_API_KEY=""
export WANDB_PROJECT="proxyv"

LLM_VERSION="lmsys/vicuna-7b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH="TODO"
IMAGE_FOLDER="TODO"

NUM_GPUS=8
NNODES=1
RANK=${RANK:-0} # srun env node rank
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}

WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num
echo "nnodes=${WORLD_SIZE}, node_rank=${RANK}"

############### Pretrain ################

PROMPT_VERSION=v1

BASE_RUN_NAME="proxyv_vicuna7b_pretrain_baseline"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

LOG_DIR=checkpoints/projectors/${BASE_RUN_NAME}/logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/node${RANK}_${T}.log

BATCH_SIZE=512  # 1 = 8
PER_DEVICE_BATCH_SIZE=16
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NUM_GPUS / WORLD_SIZE))

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${WORLD_SIZE}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    ./llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --max_num_image_crops 1 \
    --per_crop_token_len 576 \
    --proxyv False \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    2>&1 | tee ${LOG_FILE}