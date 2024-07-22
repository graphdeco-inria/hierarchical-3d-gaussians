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

import math
import os
import torch
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_post
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips

from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

def direct_collate(x):
    return x

@torch.no_grad()
def render_set(args, scene, pipe, out_dir, tau, eval):
    render_path = out_dir

    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

    psnr_test = 0.0
    ssims = 0.0
    lpipss = 0.0

    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()

    for viewpoint in tqdm(cameras):
        viewpoint=viewpoint
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()

        tanfovx = math.tan(viewpoint.FoVx * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

        to_render = expand_to_size(
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            threshold,
            viewpoint.camera_center,
            torch.zeros((3)),
            render_indices,
            parent_indices,
            nodes_for_render_indices)
        
        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices,
            threshold,
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            viewpoint.camera_center.cpu(),
            torch.zeros((3)),
            interpolation_weights,
            num_siblings
        )

        image = torch.clamp(render_post(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            render_indices=indices,
            parent_indices = parent_indices,
            interpolation_weights = interpolation_weights,
            num_node_kids = num_siblings, 
            use_trained_exp=args.train_test_exp
            )["render"], 0.0, 1.0)

        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        alpha_mask = viewpoint.alpha_mask.cuda()

        if args.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]

        try:
            torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
        except:
            os.makedirs(os.path.dirname(os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
            torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
        if eval:
            image *= alpha_mask
            gt_image *= alpha_mask
            psnr_test += psnr(image, gt_image).mean().double()
            ssims += ssim(image, gt_image).mean().double()
            lpipss += lpips(image, gt_image, net_type='vgg').mean().double()

        torch.cuda.empty_cache()
    if eval and len(scene.getTestCameras()) > 0:
        psnr_test /= len(scene.getTestCameras())
        ssims /= len(scene.getTestCameras())
        lpipss /= len(scene.getTestCameras())
        print(f"tau: {tau}, PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0, 3.0, 6.0, 15.0])
    args = parser.parse_args(sys.argv[1:])
    
    print("Rendering " + args.model_path)

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)

    for tau in args.taus:
        render_set(args, scene, pipe, os.path.join(args.out_dir, f"render_{tau}"), tau, args.eval)

