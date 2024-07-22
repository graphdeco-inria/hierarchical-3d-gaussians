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
import argparse
from read_write_model import *
import torch
import argparse
import os, time
from scipy import spatial

def fit_plane_least_squares(points):
    # Augment the point cloud with a column of ones
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]

    # Solve the least squares problem A * [a, b, c].T = B to get the plane equation z = a*x + b*y + c
    coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Plane coefficients: z = a*x + b*y + c
    a, b, c = coefficients
    
    # The normal vector is [a, b, -1]
    normal_vector = np.array([a, b, -1])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector
    
    # An in-plane vector can be any vector orthogonal to the normal. One simple choice is:
    in_plane_vector = np.cross(normal_vector, np.array([0, 0, 1]))
    if np.linalg.norm(in_plane_vector) == 0:
        in_plane_vector = np.cross(normal_vector, np.array([0, 1, 0]))
    in_plane_vector /= np.linalg.norm(in_plane_vector)  # Normalize the in-plane vector
    
    return normal_vector, in_plane_vector, np.mean(points, axis=0)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automatically reorient colmap')
    
    # Add command-line argument(s)
    parser.add_argument('--input_path', type=str, help='Path to input colmap dir',  required=True)
    parser.add_argument('--output_path', type=str, help='Path to output colmap dir',  required=True)
    parser.add_argument('--upscale', type=float, help='Upscaling factor',  default=0)
    parser.add_argument('--target_med_dist', default=20)
    parser.add_argument('--model_type', type=str, help='Specify which file format to use when processing colmap files (txt or bin)', choices=['bin','txt'], default='bin')

    args = parser.parse_args()


    # Read colmap cameras, images and points
    start_time = time.time()
    cameras, images_metas_in, points3d_in = read_model(args.input_path, ext=f".{args.model_type}")
    end_time = time.time()
    print(f"{len(images_metas_in)} images read in {end_time - start_time} seconds.")

    if args.upscale != 0:
        upscale = args.upscale
        print("manual upscale")
    else:    
        # compute upscale factor
        median_distances = []
        for key in images_metas_in:
            image_meta = images_metas_in[key]
            cam_center = -qvec2rotmat(image_meta.qvec).astype(np.float32).T @ image_meta.tvec.astype(np.float32)
            
            median_distances.extend([
                np.linalg.norm(points3d_in[pt_idx].xyz - cam_center) for pt_idx in image_meta.point3D_ids if pt_idx != -1
            ])

        median_distance = np.median(np.array(median_distances))
        upscale = (args.target_med_dist / median_distance)


    cam_centers = np.array([
        -qvec2rotmat(images_metas_in[key].qvec).T @ images_metas_in[key].tvec
        for key in images_metas_in
    ])

    up, _, _ = fit_plane_least_squares(cam_centers)

    # two cameras which are fruthest apart will occur as vertices of the convex hull
    candidates = cam_centers[spatial.ConvexHull(cam_centers).vertices]

    # get distances between each pair of cameras
    dist_mat = spatial.distance_matrix(candidates, candidates)

    # get indices of cameras that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    right = candidates[i] - candidates[j]
    right /= np.linalg.norm(right)
    
    up = torch.from_numpy(up).double()
    right = torch.from_numpy(right).double()

    forward = torch.cross(up, right)
    forward /= torch.norm(forward, p=2)

    right = torch.cross(forward, up)
    right /= torch.norm(right, p=2)

    # Stack the target axes as columns to form the rotation matrix
    rotation_matrix = torch.stack([right, forward, up], dim=1)


    positions = []
    print("Doing points")
    for key in points3d_in: 
        positions.append(points3d_in[key].xyz)
    
    positions = torch.from_numpy(np.array(positions))
    
    # Perform the rotation by matrix multiplication
    rotated_points = upscale * torch.matmul(positions, rotation_matrix)



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
        new_pos, new_rot = rotate_camera(image_meta_in.qvec, image_meta_in.tvec, rotation_matrix.double().numpy(), upscale)
        
        images_metas_out[key] = Image(
            id=image_meta_in.id,
            qvec=new_rot,
            tvec=new_pos,
            camera_id=image_meta_in.camera_id,
            name=image_meta_in.name,
            xys=image_meta_in.xys,
            point3D_ids=image_meta_in.point3D_ids,
        )

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    write_model(cameras, images_metas_out, points3d_out, args.output_path, f".{args.model_type}")

    global_end = time.time()

    print(f"reorient script took {global_end - start_time} seconds ({args.model_type} file processed).")
