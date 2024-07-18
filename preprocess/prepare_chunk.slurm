#!/bin/bash

## YOU SHOULD PUT YOUR SLURM PARAMETERS HERE (GPU, ACCOUNT, ETC ...)

#SBATCH -A hzb@a100
#SBATCH -C a100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00

RAW_CHUNK=$1
OUT_CHUNK=$2
IMAGES_DIR=$3
PREPROCESS_DIR=$5
N_NEIGHB=200


BUNDLE_ADJ_CHUNK=${RAW_CHUNK}/bundle_adjustment
echo "JOB FOR CHUNK" ${RAW_CHUNK}
source $WORK/miniconda3/etc/profile.d/conda.sh

module load cpuarch/amd
module load colmap
conda activate 3dgs_single

## Generate the chunk's colmap in an intermediate folder
mkdir ${BUNDLE_ADJ_CHUNK}
mkdir ${BUNDLE_ADJ_CHUNK}/sparse

python ${PREPROCESS_DIR}/fill_database.py --in_dir ${RAW_CHUNK}/sparse/0 --database_path ${BUNDLE_ADJ_CHUNK}/database.db

python ${PREPROCESS_DIR}/make_colmap_custom_matcher_distance.py --base_dir ${RAW_CHUNK}/sparse/0 --n_neighbours ${N_NEIGHB}
cp ${RAW_CHUNK}/sparse/0/matching_${N_NEIGHB}.txt ${BUNDLE_ADJ_CHUNK}/matching_${N_NEIGHB}.txt

colmap image_undistorter --image_path ${IMAGES_DIR} --input_path ${RAW_CHUNK}/sparse/0 --output_path ${BUNDLE_ADJ_CHUNK} --output_type COLMAP
colmap feature_extractor --database_path ${BUNDLE_ADJ_CHUNK}/database.db --image_path ${BUNDLE_ADJ_CHUNK}/images --ImageReader.existing_camera_id 1
colmap matches_importer --database_path ${BUNDLE_ADJ_CHUNK}/database.db --match_list_path ${BUNDLE_ADJ_CHUNK}/matching_${N_NEIGHB}.txt
# colmap exhaustive_matcher --database_path ${BUNDLE_ADJ_CHUNK}/database.db

mkdir ${BUNDLE_ADJ_CHUNK}/sparse/o ${BUNDLE_ADJ_CHUNK}/sparse/t ${BUNDLE_ADJ_CHUNK}/sparse/b ${BUNDLE_ADJ_CHUNK}/sparse/t2 ${BUNDLE_ADJ_CHUNK}/sparse/0
cp ${RAW_CHUNK}/sparse/0/images.bin ${RAW_CHUNK}/sparse/0/cameras.bin ${BUNDLE_ADJ_CHUNK}/sparse/o/

touch ${BUNDLE_ADJ_CHUNK}/sparse/o/points3D.bin

## 2 Rounds of triangulation + bundle adjustment
colmap point_triangulator --Mapper.ba_global_function_tolerance 0.000001 --Mapper.ba_global_max_num_iterations 30 --Mapper.ba_global_max_refinements 3 --database_path ${BUNDLE_ADJ_CHUNK}/database.db --image_path ${BUNDLE_ADJ_CHUNK}/images --input_path ${BUNDLE_ADJ_CHUNK}/sparse/o --output_path ${BUNDLE_ADJ_CHUNK}/sparse/t
colmap bundle_adjuster --BundleAdjustment.refine_extra_params 0 --BundleAdjustment.function_tolerance 0.000001 --BundleAdjustment.max_linear_solver_iterations 100 --BundleAdjustment.max_num_iterations 50 --BundleAdjustment.refine_focal_length 0 --input_path ${BUNDLE_ADJ_CHUNK}/sparse/t --output_path ${BUNDLE_ADJ_CHUNK}/sparse/b

colmap point_triangulator --Mapper.ba_global_function_tolerance 0.000001 --Mapper.ba_global_max_num_iterations 30 --Mapper.ba_global_max_refinements 3 --database_path ${BUNDLE_ADJ_CHUNK}/database.db --image_path ${BUNDLE_ADJ_CHUNK}/images --input_path ${BUNDLE_ADJ_CHUNK}/sparse/b --output_path ${BUNDLE_ADJ_CHUNK}/sparse/t2
colmap bundle_adjuster --BundleAdjustment.refine_extra_params 0 --BundleAdjustment.function_tolerance 0.000001 --BundleAdjustment.max_linear_solver_iterations 100 --BundleAdjustment.max_num_iterations 50 --BundleAdjustment.refine_focal_length 0 --input_path ${BUNDLE_ADJ_CHUNK}/sparse/t2 --output_path ${BUNDLE_ADJ_CHUNK}/sparse/0

## Correct shifts that might have happened when bundle adjusting
python ${PREPROCESS_DIR}/transform_colmap.py --in_dir ${RAW_CHUNK} --new_colmap_dir ${BUNDLE_ADJ_CHUNK} --out_dir ${OUT_CHUNK}

echo ${OUT_CHUNK} " DONE."
