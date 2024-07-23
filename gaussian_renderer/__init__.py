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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import _C
import numpy as np

def render(
        viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, indices = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        do_depth=True,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    rendered_image = rendered_image.clamp(0, 1)

    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth" : depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter.nonzero().flatten().long(),
            "radii": radii[subfilter]}


def render_post(
        viewpoint_camera, 
        pc : GaussianModel, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        render_indices = torch.Tensor([]).int(),
        parent_indices = torch.Tensor([]).int(),
        interpolation_weights = torch.Tensor([]).float(),
        num_node_kids = torch.Tensor([]).int(),
        interp_python = True,
        use_trained_exp = False):
    """
    Render the scene from a hierarchy.  
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
        
    if render_indices.size(0) != 0:
        render_inds = render_indices.long()
        if interp_python:
            num_entries = render_indices.size(0)

            interps = interpolation_weights[:num_entries].unsqueeze(1)
            interps_inv = (1 - interpolation_weights[:num_entries]).unsqueeze(1)
            parent_inds = parent_indices[:num_entries].long()

            means3D_base = (interps * means3D[render_inds] + interps_inv * means3D[parent_inds]).contiguous()
            scales_base = (interps * scales[render_inds] + interps_inv * scales[parent_inds]).contiguous()
            shs_base = (interps.unsqueeze(2) * shs[render_inds] + interps_inv.unsqueeze(2) * shs[parent_inds]).contiguous()
            
            parents = rotations[parent_inds]
            rots = rotations[render_inds]
            dots = torch.bmm(rots.unsqueeze(1), parents.unsqueeze(2)).flatten()
            parents[dots < 0] *= -1
            rotations_base = ((interps * rots) + interps_inv * parents).contiguous()
            
            opacity_base = (interps * opacity[render_inds] + interps_inv * opacity[parent_inds]).contiguous()

            if pc.skybox_points == 0:
                skybox_inds = torch.Tensor([]).long()
            else:
                skybox_inds = torch.range(pc._xyz.size(0) - pc.skybox_points, pc._xyz.size(0)-1, device="cuda").long()

            means3D = torch.cat((means3D_base, means3D[skybox_inds])).contiguous()  
            shs = torch.cat((shs_base, shs[skybox_inds])).contiguous() 
            opacity = torch.cat((opacity_base, opacity[skybox_inds])).contiguous()  
            rotations = torch.cat((rotations_base, rotations[skybox_inds])).contiguous()    
            means2D = means2D[:(num_entries + pc.skybox_points)].contiguous()     
            scales = torch.cat((scales_base, scales[skybox_inds])).contiguous()  

            interpolation_weights = interpolation_weights.clone().detach()
            interpolation_weights[num_entries:num_entries+pc.skybox_points] = 1.0 
            num_node_kids[num_entries:num_entries+pc.skybox_points] = 1 
        
        else:
            means3D = means3D[render_inds].contiguous()
            means2D = means2D[render_inds].contiguous()
            shs = shs[render_inds].contiguous()
            opacity = opacity[render_inds].contiguous()
            scales = scales[render_inds].contiguous()
            rotations = rotations[render_inds].contiguous() 

        render_indices = torch.Tensor([]).int()
        parent_indices = torch.Tensor([]).int()
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_kids,
        do_depth=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if use_trained_exp and pc.pretrained_exposures:
        try:
            exposure = pc.pretrained_exposures[viewpoint_camera.image_name]
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
        except Exception as e:
            print(f"Exposures should be optimized in single. Missing exposure for image {viewpoint_camera.image_name}")
    rendered_image = rendered_image.clamp(0, 1)
    
    vis_filter = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[vis_filter]}

def render_coarse(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, zfar=0.0, override_color = None, indices = None):
    """
    Render the scene for the coarse optimization. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True,
        do_depth=False,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    
    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[subfilter]}
