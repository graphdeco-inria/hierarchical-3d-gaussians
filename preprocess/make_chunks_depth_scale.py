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

import sys, os
import subprocess
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks_dir', required=True)
    parser.add_argument('--depths_dir', required=True)
    args = parser.parse_args()

    chunk_names = os.listdir(args.chunks_dir)
    for chunk_name in chunk_names:

        ## Generate depth_params.json file for each chunks as each chunk now has its own colmap
        make_depth_scale_args = [
            "python", "preprocess/make_depth_scale.py",
            "--base_dir", os.path.join(args.chunks_dir, chunk_name),
            "--depths_dir", args.depths_dir, 
        ]
        try:
            subprocess.run(make_depth_scale_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing make_depth_scale: {e}")
            sys.exit(1)