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

import os, argparse
import numpy as np
from read_write_model import Image, read_images_binary, read_images_text, write_images_binary, write_images_text, qvec2rotmat
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help="path to colmap images file (.txt or .bin)", required="true")
    parser.add_argument('--mult_min_dist', type=float, help="points at distance > [mult_min_dist]*median_dist_neighbors are removed. (default 10)", default=10)
    parser.add_argument('--model_type', default="bin")

    print("cleaning colmap images.bin file: removing useless data.")

    args = parser.parse_args()

    ext = args.model_type
    images_file = os.path.join(args.base_dir, f"images.{ext}")

    images_metas = {}
    if ext == "txt":
        images_metas = read_images_text(images_file)
    elif ext == "bin":
        images_metas = read_images_binary(images_file)


    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).T @ images_metas[key].tvec 
        for key in images_metas
    ])
    cam_nbrs = NearestNeighbors(n_neighbors=2).fit(cam_centers)
    centers_std = cam_centers.std(axis=0).mean()
    
    second_min_distances = []
    for key, cam_center in zip(images_metas, cam_centers):
        distances, indices = cam_nbrs.kneighbors(cam_center[None])
        second_min_distances.append(distances[0, -1])

        
    med_dist = np.median(second_min_distances)
    filtered_images = {}

    for key, second_min_distance in zip(images_metas, second_min_distances):
        image_meta = images_metas[key]

        if len(image_meta.point3D_ids) > 0 and second_min_distance <= args.mult_min_dist * med_dist:
            valid_pts_mask = image_meta.point3D_ids >= 0
            if valid_pts_mask.sum() > 0:
                filtered_images[key] = Image(
                        id=image_meta.id,
                        qvec=image_meta.qvec,
                        tvec=image_meta.tvec,
                        camera_id=image_meta.camera_id,
                        name=image_meta.name,
                        xys=image_meta.xys[valid_pts_mask],
                        point3D_ids=image_meta.point3D_ids[valid_pts_mask],
                    )

            
            # filtered_images[key].valid_point3D_ids = filtered_images[key].point3D_ids[filtered_images[key].point3D_ids >= 0]
            # filtered_images[key].valid_xys = filtered_images[key].xys[filtered_images[key].point3D_ids >= 0]
            
            # if(len(filtered_images[key].valid_point3D_ids) != len(filtered_images[key].point3D_ids)):
            #     print("reducing size ...")

    print(f"{len(images_metas)} images before; {len(filtered_images)} images after")

    # rename old images.bin/txt as images_heavy
    if os.path.exists(f"images_heavy.{ext}"):
        os.remove(f"images_heavy.{ext}")
    os.rename(images_file, os.path.join(args.base_dir, f"images_heavy.{ext}"))

    if ext == "txt":
        write_images_text(filtered_images, images_file)
    elif ext == "bin":
        write_images_binary(filtered_images, images_file)

    print(0)