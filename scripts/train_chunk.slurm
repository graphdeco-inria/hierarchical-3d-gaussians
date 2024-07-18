#!/bin/bash

#SBATCH -A hzb@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00


SOURCE_CHUNK=$1
OUTPUT_DIR=$2
ENV=$3
CHUNK_NAME=$4
HIERARCHY_CREATOR_EXE=$5
IMAGES_DIR=$6
DEPTHS_DIR=$7

if [ -z "$8" ]; then
    echo "No masks provided."
    MASKS_ARG=""
else
    MASKS_ARG="$8"
    echo "masks provided: $MASKS_ARGS"
fi

TRAINED_CHUNK=${OUTPUT_DIR}"/trained_chunks/"${CHUNK_NAME}

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV}

# Train the chunk
python -u train_single.py --save_iterations -1 -i ${IMAGES_DIR} -d ${DEPTHS_DIR} --scaffold_file ${OUTPUT_DIR}/scaffold/point_cloud/iteration_30000 --disable_viewer --skybox_locked ${MASKS_ARG} -s ${SOURCE_CHUNK} --model_path ${TRAINED_CHUNK} --bounds_file ${SOURCE_CHUNK}

# Generate a hierarchy within the chunk
${HIERARCHY_CREATOR_EXE} ${TRAINED_CHUNK}/point_cloud/iteration_30000/point_cloud.ply ${SOURCE_CHUNK} ${TRAINED_CHUNK} ${OUTPUT_DIR}/scaffold/point_cloud/iteration_30000

python -u train_post.py --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i ${IMAGES_DIR} --scaffold_file ${OUTPUT_DIR}/scaffold/point_cloud/iteration_30000 ${MASKS_ARG} -s ${SOURCE_CHUNK} --disable_viewer --model_path ${TRAINED_CHUNK} --hierarchy ${TRAINED_CHUNK}/hierarchy.hier

echo "CHUNK " ${CHUNK_NAME} " FULLY TRAINED."
