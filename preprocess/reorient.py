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

import numpy as np
import torch
import argparse
import ast
import os, time
from read_write_model import *

def rotate_camera(qvec, tvec, rot_matrix, upscale):
    # Assuming cameras have 'T' (translation) field

    R = qvec2rotmat(qvec)
    T = np.array(tvec)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = np.copy(C2W[:3, 3])
    cam_rot_orig = np.copy(C2W[:3, :3])
    cam_center = np.matmul(cam_center, rot_matrix)
    cam_rot = np.linalg.inv(rot_matrix) @ cam_rot_orig
    C2W[:3, 3] = upscale * cam_center
    C2W[:3, :3] = cam_rot
    Rt = np.linalg.inv(C2W)
    new_pos = Rt[:3, 3]
    new_rot = rotmat2qvec(Rt[:3, :3])

    # R_test = qvec2rotmat(new_rots[-1])
    # T_test = np.array(new_poss[-1])
    # Rttest = np.zeros((4, 4))
    # Rttest[:3, :3] = R_test
    # Rttest[:3, 3] = T_test
    # Rttest[3, 3] = 1.0
    # C2Wtest = np.linalg.inv(Rttest) 

    return new_pos, new_rot

# Function to compute the cross product of two 3D vectors
def cross_product(v1, v2):
    return torch.cross(v1, v2)

# Function to normalize a 3D vector
def normalize_vector(v):
    norm = torch.norm(v, p=2)  # Calculate the Euclidean norm (L2 norm)
    return v / norm
    
def parse_vector(s):
    try:
        result = ast.literal_eval(s)
        if isinstance(result, tuple) and len(result) == 3:
            return result
        else:
            raise ValueError("Invalid vector format. Must be a 3-element tuple.")
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid vector format. Example: (1.0, 2.0, 3.0)")

def main():

    parser = argparse.ArgumentParser(description='Example script with command-line arguments.')
    
    # Add command-line argument(s)
    parser.add_argument('--input_path', type=str, help='Path to input colmap dir',  required=True)
    parser.add_argument('--output_path', type=str, help='Path to output colmap dir',  required=True)
    parser.add_argument('--upscale', type=float, help='Upscaling factor',  required=True)
    
    # Add command-line arguments for two 3D vectors
    parser.add_argument('--up', type=parse_vector, help='Up 3D vector in the format (x, y, z)', required=True)
    parser.add_argument('--right', type=parse_vector, help='Right 3D vector in the format (x, y, z)', required = True)
    parser.add_argument('--input_format', type=str, help='specify which file format to use when processing colmap files (txt or bin)', choices=['bin','txt'], default='bin')

    # Parse the command-line arguments
    args = parser.parse_args()

    global_start = time.time()

    # Access the parsed vectors
    vector1 = args.up
    vector2 = args.right

    # Your main logic goes here
    print("Up Vector:", vector1)
    print("Right Vector:", vector2)

    ext = args.input_format
    
    #print(float(vector1[0]),float(vector1[1]),float(vector1[2]))
    
    up = torch.Tensor(vector1).double()
    right = torch.Tensor(vector2).double()

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments    
    os.makedirs(args.output_path, exist_ok=True)

    # Your main logic goes here
    print("Input path:", args.input_path)
    print("Output path:", args.output_path)
    print(f"processing format: .{ext}")

    #up = torch.Tensor([0, -1, -0.2]).double()
    #right = torch.Tensor([-0.5, 0, -0.5]).double()

    up = normalize_vector(up)
    right = normalize_vector(right)
    
    forward = cross_product(up, right)
    forward = normalize_vector(forward)
    right = cross_product(forward, up)
    right = normalize_vector(right)

    # Stack the target axes as columns to form the rotation matrix
    rotation_matrix = torch.stack([right, forward, up], dim=1)

    # Read colmap cameras, images and points
    start_time = time.time()
    cameras, images_metas_in, points3d_in = read_model(args.input_path, ext=f".{ext}")
    end_time = time.time()
    print(f"{len(images_metas_in)} images read in {end_time - start_time} seconds.")

    positions = []
    print("Doing points")
    for key in points3d_in: 
        positions.append(points3d_in[key].xyz)
    
    positions = torch.from_numpy(np.array(positions))
    
    # Perform the rotation by matrix multiplication
    rotated_points = args.upscale * torch.matmul(positions, rotation_matrix)

    points3d_out = {}
    for key, rotated in zip(points3d_in, rotated_points):
        point3d_in = points3d_in[key]
        points3d_out[key] = Point3D(
            id=point3d_in.id,
            xyz=rotated,
            rgb=point3d_in.rgb,
            error=point3d_in.error,
            image_ids=point3d_in.image_ids,
            point2D_idxs=point3d_in.point2D_idxs,
        )

    print("Doing images")
    images_metas_out = {} 
    for key in images_metas_in: 
        image_meta_in = images_metas_in[key]
        new_pos, new_rot = rotate_camera(image_meta_in.qvec, image_meta_in.tvec, rotation_matrix.double().numpy(), args.upscale)
        
        images_metas_out[key] = Image(
            id=image_meta_in.id,
            qvec=new_rot,
            tvec=new_pos,
            camera_id=image_meta_in.camera_id,
            name=image_meta_in.name,
            xys=image_meta_in.xys,
            point3D_ids=image_meta_in.point3D_ids,
        )

    write_model(cameras, images_metas_out, points3d_out, args.output_path, f".{ext}")

    global_end = time.time()

    print(f"reorient script took {global_end - global_start} seconds ({ext} file processed).")

if __name__ == "__main__":
    main()