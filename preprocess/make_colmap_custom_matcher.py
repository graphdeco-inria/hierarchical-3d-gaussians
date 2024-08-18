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

import os
import numpy as np
from joblib import delayed, Parallel
import argparse
from exif import Image
from sklearn.neighbors import NearestNeighbors

#TODO: clean it
def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def image_coordinates(image_name):
    with open(os.path.join(args.image_path, image_name), 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = [
                decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                decimal_coords(img.gps_longitude, img.gps_longitude_ref)
            ]
            return coords
        except AttributeError:
            return None
    else:
        return None    
    
def get_matches(img_name, cam_center, cam_nbrs, img_names_gps):
    _, indices = cam_nbrs.kneighbors(cam_center[None])
    matches = ""
    for idx in indices[0, 1:]:
        matches += f"{img_name} {img_names_gps[idx]}\n" 
    return matches

def find_images_names(root_dir):
    image_files_by_subdir = []

    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):

        # Filter for image files (you can add more extensions if needed), sort images
        image_files = sorted([f for f in filenames if f.lower().endswith(('.png', '.jpg', '.JPG', '.PNG'))])

        # If there are image files in the current directory, add them to the list
        if image_files:
            image_files_by_subdir.append({
                'dir': os.path.basename(dirpath) if dirpath != root_dir else "",
                'images': image_files
            })

    return image_files_by_subdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_seq_matches_per_view', default=0, type=int)
    parser.add_argument('--n_quad_matches_per_view', default=10, type=int)
    parser.add_argument('--n_loop_closure_match_per_view', default=5, type=int)
    parser.add_argument('--loop_matches', default=[], type=int) 
    parser.add_argument('--n_gps_neighbours', default=25, type=int)
    args = parser.parse_args()


    loop_matches = np.array(args.loop_matches, dtype=np.int64).reshape(-1, 2)

    loop_rel_matches = np.arange(0, args.n_loop_closure_match_per_view)
    loop_rel_matches = 2**loop_rel_matches
    loop_rel_matches = np.concatenate([-loop_rel_matches[::-1], np.array([0]), loop_rel_matches])

    image_files_organised = find_images_names(args.image_path)

    cam_folder_list = []
    cam_folder_list = os.listdir(f"{args.image_path}")


    matches_str = []
    def add_match(cam_id, matched_cam_id, current_image_file, matched_frame_id):
        # REMOVE AFTER
        # if (cam_folder_list[cam_id + matched_cam_id] == "backleft") and (not cam_folder_list[cam_id] == "backleft") and (matched_frame_id >= 785):
        #     matched_frame_id -= 647
        # if (not cam_folder_list[cam_id + matched_cam_id] == "backleft") and (cam_folder_list[cam_id] == "backleft") and (matched_frame_id >= 785):
        #     matched_frame_id += 647

        if matched_frame_id < len(matched_cam['images']):
            matched_image_file = matched_cam['images'][matched_frame_id]
            matches_str.append(f"{cam_folder_list[cam_id]}/{current_image_file} {cam_folder_list[cam_id + matched_cam_id]}/{matched_image_file}\n")


    for cam_id, current_cam in enumerate(image_files_organised):
        for matched_cam_id, matched_cam in enumerate(image_files_organised[cam_id:]):
            for current_image_id, current_image_file in enumerate(current_cam['images']):
                for frame_step in range(args.n_seq_matches_per_view):
                    matched_frame_id = current_image_id + frame_step
                    add_match(cam_id, matched_cam_id, current_image_file, matched_frame_id)

                for match_id in range(args.n_quad_matches_per_view):
                    frame_step = args.n_seq_matches_per_view + int(2**match_id) - 1
                    matched_frame_id = current_image_id + frame_step
                    add_match(cam_id, matched_cam_id, current_image_file, matched_frame_id)

            ## Loop closure
            for loop_match in loop_matches:
                for current_loop_rel_match in loop_rel_matches:
                    current_image_id = (loop_match[0] + current_loop_rel_match) 
                    if current_image_id < len(current_cam['images']):
                        current_image_file = current_cam['images'][current_image_id]
                        for matched_loop_rel_match in loop_rel_matches:
                            matched_frame_id = (loop_match[1] + matched_loop_rel_match)
                            add_match(cam_id, matched_cam_id, current_image_file, matched_frame_id)


    ## Add GPS matches
    if args.n_gps_neighbours > 0:
        all_img_names = []
        for ind, cam in enumerate(image_files_organised):
            all_img_names += [os.path.join(cam['dir'], img_name) for img_name in cam['images']]

        all_cam_centers = [image_coordinates(img_name) for img_name in all_img_names]
        # all_cam_centers = Parallel(n_jobs=-1, backend="threading")(
        #     delayed(image_coordinates)(img_name) for img_name in all_img_names
        # )
        img_names_gps = [img_name for img_name, cam_center in zip(all_img_names, all_cam_centers) if cam_center is not None]
        cam_centers_gps =  [cam_center for cam_center in all_cam_centers if cam_center is not None]
        cam_centers = np.array(cam_centers_gps)
        cam_nbrs = NearestNeighbors(n_neighbors=args.n_gps_neighbours).fit(cam_centers) if cam_centers.size else []

        matches_str += [get_matches(img_name, cam_center, cam_nbrs, img_names_gps) for img_name, cam_center in zip(img_names_gps, cam_centers)]


    ## Remove duplicate matches
    intermediate_out_matches = list(dict.fromkeys(matches_str))
    reciproc_matches = [f"{match.split(' ')[1][:-1]} {match.split(' ')[0]}\n" for match in intermediate_out_matches]
    reciproc_matches_dict = dict.fromkeys(reciproc_matches)
    out_matches = [
        match for match in intermediate_out_matches
        if not match in reciproc_matches_dict
        ]

    # with open(f"{args.image_path}/TEST_new_{args.n_seq_matches_per_view}_{args.n_quad_matches_per_view}_{args.n_loop_closure_match_per_view}_{args.n_gps_neighbours}.txt", "w") as f:
    with open(args.output_path, "w") as f:
        f.write(''.join(out_matches))

    print(0)