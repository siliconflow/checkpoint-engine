#!/bin/bash

## requrires one arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 gpu_ids [port:8000]"
    exit 1
fi

## input gpu ids and port number from command line
GPU_IDS=$1
if [ "$#" -ge 2 ]; then
    PORT=$2
else
    PORT=29501
fi
# split gpu ids by comma
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARRAY[@]}

echo "Using GPU_IDS: $GPU_IDS port: $PORT"
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARRAY[@]}
TP_SIZE=$NUM_GPUS      # tensor并行度

#### Ethernet
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
IB_HCAS=(mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7 mlx5_8)
# export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
# to export specific IB_HCA according to the number of GPUs, we can use the following code
export NCCL_IB_HCA=$(IFS=,; echo "${IB_HCAS[*]:0:$NUM_GPUS}")
echo "Using NCCL_IB_HCA: $NCCL_IB_HCA"

checkpoint_path="/nvme_data/hf_models/Qwen/Qwen3-8B-int8"
meta_path="output/Qwen3-8B-int8_ckpt_meta.pkl"
num_layers=36
# checkpoint_path="/nvme_data/hf_models/deepseek-ai/DeepSeek-R1-0528-block-fp8"
# meta_path="output/DeepSeek-R1-0528-block-fp8_ckpt_meta.pkl"
# num_layers=62

export PYTHONPATH=/nvme_data/chenxiaotao/repositories/checkpoint_engine:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --nproc-per-node ${TP_SIZE} \
    --master_port ${PORT} \
    tests/test_buffer_granularity.py \
    --checkpoint-path $checkpoint_path \
    --num-layers $num_layers \
    --save-metas-file $meta_path \
    --sleep-time 3600

