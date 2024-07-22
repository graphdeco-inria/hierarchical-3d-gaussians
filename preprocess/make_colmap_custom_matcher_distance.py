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
from sklearn.neighbors import NearestNeighbors
from read_write_model import read_images_binary

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_images_metas(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images_metas = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                idx = int(elems[0])
                images_metas[idx] = {
                    "camera_id": int(elems[8]),
                    "name":elems[9],
                    "qvec": np.array(tuple(map(float, elems[1:5]))),
                    "tvec": np.array(tuple(map(float, elems[5:8]))),
                    }
                elems = fid.readline().split()

    return images_metas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--n_neighbours', default=100, type=int)
    args = parser.parse_args()

    images_metas = read_images_binary(f"{args.base_dir}/images.bin")
    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32) 
        for key in images_metas
    ])
    n_neighbours = min(args.n_neighbours, len(cam_centers))
    cam_nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(cam_centers)

    def get_matches(key, cam_center):
        _, indices = cam_nbrs.kneighbors(cam_center[None])
        matches = ""
        keys = list(images_metas.keys())
        for idx in indices[0, 1:]:
            matches += f"{images_metas[key].name} {images_metas[keys[idx]].name}\n" 
        return matches

    matches = [get_matches(key, cam_center) for key, cam_center in zip(images_metas, cam_centers)]
    # matches = Parallel(n_jobs=-1, backend="threading")(
    #     delayed(get_matches)(key, cam_center) for key, cam_center in zip(images_metas, cam_centers)
    # )

    matches_str = []
    for match in matches:
        matches_str.append(match)

    with open(f"{args.base_dir}/matching_{args.n_neighbours}.txt", "w") as f:
        f.write(''.join(matches_str))