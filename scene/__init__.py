#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON, CameraDataset
from utils.system_utils import mkdir_p

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], create_from_hier=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.alpha_masks, args.depths, args.eval, args.train_test_exp)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Making Training Dataset")
            self.train_cameras[resolution_scale] = CameraDataset(scene_info.train_cameras, args, resolution_scale, False)

            print("Making Test Dataset")
            self.test_cameras[resolution_scale] = CameraDataset(scene_info.test_cameras, args, resolution_scale, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif args.pretrained:
            self.gaussians.create_from_pt(args.pretrained, self.cameras_extent)
        elif create_from_hier:
            self.gaussians.create_from_hier(args.hierarchy, self.cameras_extent, args.scaffold_file)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 
                                           scene_info.train_cameras,
                                           self.cameras_extent, 
                                           args.skybox_num,
                                           args.scaffold_file,
                                           args.bounds_file,
                                           args.skybox_locked)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        mkdir_p(point_cloud_path)
        if self.gaussians.nodes is not None:
            self.gaussians.save_hier()
        else:
            with open(os.path.join(point_cloud_path, "pc_info.txt"), "w") as f:
                f.write(str(self.gaussians.skybox_points))
            if self.gaussians._xyz.size(0) > 8_000_000:
                self.gaussians.save_pt(point_cloud_path)
            else:
                self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

            exposure_dict = {
                image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.gaussians.exposure_mapping
            }

            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
