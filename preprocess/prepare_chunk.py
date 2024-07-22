#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys, shutil
import subprocess
import argparse
import time, platform
from read_write_model import write_points3D_binary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_chunk', type=str, help='Input raw chunk', required=True)
    parser.add_argument('--out_chunk', type=str, help='Output chunk', required=True)
    parser.add_argument('--images_dir', type=str, help='Images directory', required=True)
    parser.add_argument('--skip_bundle_adjustment', action="store_true", default=False)
    args = parser.parse_args()

    matching_nb = 50 if args.skip_bundle_adjustment else 200
    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    bundle_adj_chunk = os.path.join(args.raw_chunk, "bundle_adjustment")

    if not os.path.exists(bundle_adj_chunk):
        os.makedirs(os.path.join(bundle_adj_chunk, "sparse"))

    # First, create a new colmap database for each chunk, it is filled with the raw chunk colmap
    gen_db_attr = [
        "python", "preprocess/fill_database.py",
        "--in_dir", os.path.join(args.raw_chunk, "sparse", "0"), 
        "--database_path", os.path.join(bundle_adj_chunk, "database.db")
    ]
    try:
        subprocess.run(gen_db_attr, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing gen_database: {e}")
        sys.exit(1)

    ## A custom matching file is generated for the chunk, this one is based on distance 
    make_colmap_custom_matcher_args = [
        "python", "preprocess/make_colmap_custom_matcher_distance.py",
        "--base_dir", os.path.join(args.raw_chunk, "sparse", "0"), 
        "--n_neighbours", f"{matching_nb}"
    ]
    try:
        subprocess.run(make_colmap_custom_matcher_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing custom matcher distance: {e}")
        sys.exit(1)
    
    shutil.copy(os.path.join(args.raw_chunk, "sparse", "0", f"matching_{matching_nb}.txt"), os.path.join(bundle_adj_chunk, f"matching_{matching_nb}.txt"))

    ## Extracting the subset of images corresponding to that chunk
    print(f"undistorting to chunk {bundle_adj_chunk}...")
    colmap_image_undistorter_args = [
        colmap_exe, "image_undistorter",
        "--image_path", f"{args.images_dir}",
        "--input_path", f"{args.raw_chunk}/sparse/0", 
        "--output_path", f"{bundle_adj_chunk}",
        "--output_type", "COLMAP"
        ]
    try:
        subprocess.run(colmap_image_undistorter_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    print("extracting features...")
    colmap_feature_extractor_args = [
        colmap_exe, "feature_extractor",
        "--database_path", f"{bundle_adj_chunk}/database.db",
        "--image_path", f"{bundle_adj_chunk}/images",
        "--ImageReader.existing_camera_id", "1",
        ]
    
    try:
        subprocess.run(colmap_feature_extractor_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)

    print("feature matching...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", f"{bundle_adj_chunk}/database.db",
        "--match_list_path", f"{bundle_adj_chunk}/matching_{matching_nb}.txt"
        ]
    try:
        subprocess.run(colmap_matches_importer_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    os.makedirs(os.path.join(bundle_adj_chunk, "sparse", "o"))
    os.makedirs(os.path.join(bundle_adj_chunk, "sparse", "t"))
    os.makedirs(os.path.join(bundle_adj_chunk, "sparse", "b"))
    os.makedirs(os.path.join(bundle_adj_chunk, "sparse", "t2"))
    os.makedirs(os.path.join(bundle_adj_chunk, "sparse", "0"))
    
    shutil.copy(os.path.join(args.raw_chunk, "sparse", "0", "images.bin"), os.path.join(bundle_adj_chunk, "sparse", "o", "images.bin"))
    shutil.copy(os.path.join(args.raw_chunk, "sparse", "0", "cameras.bin"), os.path.join(bundle_adj_chunk, "sparse", "o", "cameras.bin"))
    
    # points3D.bin shouldnt be completely empty (must have 1 BYTE)
    write_points3D_binary({}, os.path.join(bundle_adj_chunk, "sparse", "o", "points3D.bin")) 

    if args.skip_bundle_adjustment:
        subprocess.run([colmap_exe, "point_triangulator",
            "--Mapper.ba_global_max_num_iterations", "5",
            "--Mapper.ba_global_max_refinements", "1", 
            "--database_path", f"{bundle_adj_chunk}/database.db",
            "--image_path", f"{bundle_adj_chunk}/images",
            "--input_path", f"{bundle_adj_chunk}/sparse/o",
            "--output_path", f"{bundle_adj_chunk}/sparse/0",
            ], check=True)
    else:
        colmap_point_triangulator_args = [
            colmap_exe, "point_triangulator",
            "--Mapper.ba_global_function_tolerance", "0.000001",
            "--Mapper.ba_global_max_num_iterations", "30",
            "--Mapper.ba_global_max_refinements", "3",
            ]

        colmap_bundle_adjuster_args = [
            colmap_exe, "bundle_adjuster",
            "--BundleAdjustment.refine_extra_params", "0",
            "--BundleAdjustment.function_tolerance", "0.000001",
            "--BundleAdjustment.max_linear_solver_iterations", "100",
            "--BundleAdjustment.max_num_iterations", "50", 
            "--BundleAdjustment.refine_focal_length", "0"
            ]
        # 2 rounds of triangulation + bundle adjustment
        try:
            subprocess.run(colmap_point_triangulator_args + [
                "--database_path", f"{bundle_adj_chunk}/database.db",
                "--image_path", f"{bundle_adj_chunk}/images",
                "--input_path", f"{bundle_adj_chunk}/sparse/o",
                "--output_path", f"{bundle_adj_chunk}/sparse/t",
                ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap_point_triangulator_args: {e}")
            sys.exit(1)

        try:
            subprocess.run(colmap_bundle_adjuster_args + [
                "--input_path", f"{bundle_adj_chunk}/sparse/t",
                "--output_path", f"{bundle_adj_chunk}/sparse/b",
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap_bundle_adjuster_args: {e}")
            sys.exit(1)

        try:
            subprocess.run(colmap_point_triangulator_args + [
                "--database_path", f"{bundle_adj_chunk}/database.db",
                "--image_path", f"{bundle_adj_chunk}/images",
                "--input_path", f"{bundle_adj_chunk}/sparse/b",
                "--output_path", f"{bundle_adj_chunk}/sparse/t2",
                ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap_point_triangulator_args: {e}")
            sys.exit(1)

        try:
            subprocess.run(colmap_bundle_adjuster_args + [
                "--input_path", f"{bundle_adj_chunk}/sparse/t2",
                "--output_path", f"{bundle_adj_chunk}/sparse/0",
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap_bundle_adjuster_args: {e}")
            sys.exit(1)

    transform_colmap_args = [
        "python", "preprocess/transform_colmap.py",
        "--in_dir", args.raw_chunk,
        "--new_colmap_dir", bundle_adj_chunk,
        "--out_dir", args.out_chunk
    ]

    ## Correct slight shifts that might have happened during bundle adjustments
    try:
        subprocess.run(transform_colmap_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing transform_colmap_args: {e}")
        sys.exit(1)
    