export OMP_NUM_THREADS=8
export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

export TORCH_HOME=/home/yharim/Source/YottaCap/data/

export PYTHONPATH=$(pwd)

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run \
	--nproc_per_node=$NUM_GPUS_PER_NODE \
	--nnodes=$NUM_NODES \
	--node_rank $NODE_RANK \
	yottacap/train.py