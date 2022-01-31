#!/bin/bash
#SBATCH --job-name eval_tmanet
#SBATCH -o %j.log
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1

CONFIG_FILE=$1
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
WORK_DIR="./work_dirs/${CONFIG}"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULT_FILE="${WORK_DIR}/result.pkl"

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\nconfig file: ${CONFIG}\n"

# evaluation
echo -e "\nEvaluating ${WORK_DIR}\n"

# Initialize the module command
source /etc/profile

# Load modules
module load anaconda/2020a
module load mpi/openmpi-4.0
module load cuda/10.1
module load nccl/2.5.6-cuda10.1

# These flags tell MPI how to set up communication
export MPI_FLAGS="--tag-output --bind-to socket -map-by core -mca btl ^openib -mca pml ob1 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

# Set some environment variables needed by torch.distributed 
export MASTER_ADDR=$(hostname -s)
# Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

export NCCL_DEBUG=INFO

# Do not use the torch.distributed.launch utility. Use mpirun as shown below
# to launch your code. The file torch_test.py has additional setup code needed to the
# distributed training capability 
mpirun ${MPI_FLAGS} python \
  supercloud_test.py \
  ${CONFIG_FILE} \
  ${CHECKPOINT} \
  --work-dir $WORK_DIR \
  --eval mIoU  \
  --tmpdir $TMPDIR \

