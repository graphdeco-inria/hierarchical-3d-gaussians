#!/bin/bash

#SBATCH -A hzb@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00

HIERARCHY_MERGER_EXE=$1
TRAINED_CHUNKS=$2
NB=$3
CHUNKS_COLMAP=$4
OUTPUT=$5

# Remove the first two arguments, leaving only the array elements
shift 5

# The remaining arguments are the array elements
array_elements=("$@")

echo "CHUNKS TO BE MERGED " ${CHUNK_LIST}
${HIERARCHY_MERGER_EXE} ${TRAINED_CHUNKS} ${NB} ${CHUNKS_COLMAP} ${OUTPUT} "${array_elements[@]}"
