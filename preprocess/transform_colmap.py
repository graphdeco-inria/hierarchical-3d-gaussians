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
import os
import shutil
import random
import torch
from read_write_model import *

Transform = collections.namedtuple(
    "Transform", ["t0", "t1", "s0", "s1", "R"]
)

def procrustes_analysis(X0,X1): # [N,3]
    """
    From https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/803291bd0ee91c7c13fb5cc42195383c5ade7d15/camera.py#L278
    """
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = Transform(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def get_nb_pts(image_metas):
    n_pts = 0
    for key in image_metas:
        pts_idx = image_metas[key].point3D_ids
        if(len(pts_idx) > 5):
            n_pts = max(n_pts, np.max(pts_idx))

    return n_pts + 1

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--new_colmap_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    old_images_metas = read_images_binary(f"{args.in_dir}/sparse/0/images.bin")
    new_images_metas = read_images_binary(f"{args.new_colmap_dir}/sparse/0/images.bin")
    n_pts = get_nb_pts(new_images_metas)
    
    old_keys = old_images_metas.keys()
    old_keys_dict = {old_images_metas[key].name: key for key in old_keys}
    new_old_key_mapping = {key: old_keys_dict[new_images_metas[key].name] for key in new_images_metas}

    old_cam_centers = np.array([
        -qvec2rotmat(old_images_metas[new_old_key_mapping[key]].qvec).astype(np.float32).T @ old_images_metas[new_old_key_mapping[key]].tvec.astype(np.float32) 
        for key in new_images_metas
    ])
    new_cam_centers = np.array([
        -qvec2rotmat(new_images_metas[key].qvec).astype(np.float32).T @ new_images_metas[key].tvec.astype(np.float32) 
        for key in new_images_metas
    ])

    dists = np.linalg.norm(old_cam_centers - new_cam_centers, axis=-1)
    valid_cams = dists <= (np.median(dists) * 5) + 1e-8

    old_cam_centers_torch = torch.from_numpy(old_cam_centers)
    new_cam_centers_torch = torch.from_numpy(new_cam_centers)

    old_cam_centers_trimmed = old_cam_centers[valid_cams]
    new_cam_centers_trimmed = new_cam_centers[valid_cams]
    old_cam_centers_torch_trimmed = torch.from_numpy(old_cam_centers_trimmed)
    new_cam_centers_torch_trimmed = torch.from_numpy(new_cam_centers_trimmed)

    sim3 = procrustes_analysis(old_cam_centers_torch_trimmed, new_cam_centers_torch_trimmed)
    center_aligned = (new_cam_centers_torch-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    points3d = read_points3D_binary(f"{args.new_colmap_dir}/sparse/0/points3D.bin")

    xyzs = np.zeros([n_pts, 3], np.float32)
    errors = np.zeros([n_pts], np.float32) + 9e9
    indices = np.zeros([n_pts], np.int64)
    n_images = np.zeros([n_pts], np.int64)
    colors = np.zeros([n_pts, 3], np.float32)

    idx = 0    
    for key in points3d:
        xyzs[idx] = points3d[key].xyz
        indices[idx] = points3d[key].id
        errors[idx] = points3d[key].error
        colors[idx] = points3d[key].rgb
        n_images[idx] = len(points3d[key].image_ids)
        idx +=1

    mask = errors < 1.5
    mask *= n_images > 3

    xyzsC, colorsC, errorsC, indicesC, n_imagesC = xyzs[mask], colors[mask], errors[mask], indices[mask], n_images[mask]

    points3dC_aligned = ((torch.from_numpy(xyzsC)-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0).numpy()
    
    R = torch.from_numpy(np.array([
        qvec2rotmat(new_images_metas[key].qvec).astype(np.float32)
        for key in new_images_metas
    ]))
    R_aligned = R@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]

    with open(f"{args.in_dir}/center.txt", 'r') as f:
        center = np.array(tuple(map(float, f.readline().strip().split())))
    with open(f"{args.in_dir}/extent.txt", 'r') as f:
        extent = np.array(tuple(map(float, f.readline().strip().split())))

    corner_min = center - 1.1 * extent / 2
    corner_max = center + 1.1 * extent / 2

    out_colmap = f"{args.out_dir}/sparse/0"
    os.makedirs(out_colmap, exist_ok=True)
    
    mask = np.all(points3dC_aligned < corner_max, axis=-1) * np.all(points3dC_aligned > corner_min, axis=-1)
    new_points3d = points3dC_aligned#[mask]
    new_colors = np.clip(colorsC, 0, 255).astype(np.uint8)
    new_indices = indicesC#[mask]
    new_errors = errorsC#[mask]

    images_metas_out = {}
    for key, R, t, valid_cam in zip(new_images_metas, R_aligned.numpy(), t_aligned.numpy(), valid_cams):
        if valid_cam:
            image_meta = new_images_metas[key]

            images_metas_out[key] = Image(
                id = key,
                qvec = rotmat2qvec(R),
                tvec = t,
                camera_id = image_meta.camera_id,
                name = image_meta.name,
                xys = image_meta.xys,
                point3D_ids = image_meta.point3D_ids,
            )

    write_images_binary(images_metas_out, f"{out_colmap}/images.bin")

    points_out = {
        new_indices[idx] : Point3D(
                id=indicesC[idx],
                xyz= points3dC_aligned[idx],
                rgb=new_colors[idx],
                error=errorsC[idx],
                image_ids=np.array([]),
                point2D_idxs=np.array([])
            )
        for idx in range(len(points3dC_aligned))
    }

    write_points3D_binary(points_out, f"{out_colmap}/points3D.bin")    

    shutil.copy(f"{args.new_colmap_dir}/sparse/0/cameras.bin", f"{out_colmap}/cameras.bin")
    shutil.copy(f"{args.in_dir}/center.txt", f"{args.out_dir}/center.txt")
    shutil.copy(f"{args.in_dir}/extent.txt", f"{args.out_dir}/extent.txt")

    print(0)