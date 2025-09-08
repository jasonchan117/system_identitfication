#
# Copyright (C) 2023, Inria
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
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], skip_train=False, skip_val=False, skip_test=False, pcd = None, cam_info = None):
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
        
        self.train_cameras = {} if cam_info is None else cam_info.get("train_cams")
        self.test_cameras = {} if cam_info is None else cam_info.get("test_cams")
        read_cam = True if cam_info is None else False

        self.init_cameras = {} if cam_info is None else cam_info.get("train_cams")
        self.val_cameras = {}


        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "all_data.json")):
            print("Found all_data.json file, assuming PacNeRF dataset!")
            scene_info = sceneLoadTypeCallbacks["PacNeRF"](args.source_path, args.config_path, args.white_background, load_fix_pcd=False, read_cam=True)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, '.png',
                                                           skip_train, skip_val, skip_test)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            # json_cams = []
            # camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            # if scene_info.train_cameras:
            #     camlist.extend(scene_info.train_cameras)
            # for id, cam in enumerate(camlist):
            #     json_cams.append(camera_to_JSON(id, cam))
            # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file)

        if shuffle:
            if not skip_train:
                random.shuffle(scene_info.train_cameras) # Multi-res consistent random shuffling
            if not skip_test:
                random.shuffle(scene_info.test_cameras) # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if not read_cam:
                continue
            if not skip_train:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                                args)
                self.init_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.init_cameras, resolution_scale,
                                                                            args)
            if not skip_val:
                print("Loading Val Cameras")
                self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale,
                                                                            args)
            if not skip_test:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:


            if pcd is not None:
                self.gaussians.create_from_pcd(pcd, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getInitCameras(self, scale=1.0):
        return self.init_cameras[scale]

    def getValCameras(self, scale=1.0):
        return self.val_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]



    def overwrite_alphas(self, pipeline, dataset: ModelParams, deform: DeformModel, static_threshold):
        from gaussian_renderer import render
        import einops
        import copy
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        xyz_canonical = self.gaussians.get_xyz.detach()
        with torch.no_grad():
            deform_code = deform.code_field(xyz_canonical)

            dxyz, _, _ = deform.step(xyz_canonical, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
            sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
            sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz_canonical.shape[0])
            static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
            for i in range(75):
                dxyz_t, _, _ = deform.step(xyz_canonical, sampled_time[:, i:i + 1], deform_code, 1 / 30)
                static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
            motion_mask = ~static_mask
        def overwrite(cam_dict):
            for scale, cam_list in cam_dict.items():
                from gaussian_renderer import render
                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                xyz_canonical = self.gaussians.get_xyz.detach()




                deform_code = deform.code_field(xyz_canonical)
                d_xyz = torch.zeros_like(self.gaussians.get_xyz)
                d_rotation = torch.zeros_like(self.gaussians.get_rotation)
                d_scaling = torch.zeros_like(self.gaussians.get_scaling)
                t0 = torch.zeros_like(xyz_canonical[..., :1])
                d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
                    xyz_canonical[static_mask], t0[static_mask], deform_code[static_mask]
                )
                for view in cam_list:
                    fid = view.fid
                    time_input = fid.unsqueeze(0).expand(xyz_canonical.shape[0], -1)
                    # time_input = deform.deform.expand_time(fid)
                    # d_values = deform.step(xyz_canonical.detach(), time_input, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask)
                    d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
                    xyz_canonical[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / 30
                    )
                    # d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            
                    # time_input = fid.unsqueeze(0).expand(1, -1)
                    # d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
                    # results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
                    results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                    # results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                    # alpha = results["alpha"]
                    # view.gt_alpha_mask = alpha.to(view.data_device)
            return copy.deepcopy(cam_dict)
        train_cams = overwrite(self.train_cameras)
        test_cams = overwrite(self.test_cameras)
        cameras_extent = copy.deepcopy(self.cameras_extent)
        return train_cams, test_cams, cameras_extent