# coding=utf-8

import torch
import taichi as ti
import torch.nn as nn
import time, os, json
import numpy as np
from tqdm import tqdm, trange
import einops
from argparse import ArgumentParser
from gaussian_renderer import render
from scene import Scene, DeformModel, GaussianModel
from utils.general_utils import safe_state
from render import track_points_with_given_velocity
# from simulator import MPMSimulator, Estimator, Estimator_exp
from simulator import ExpSimulator,  Estimator_exp
# from utils.system_utils import check_gs_model, draw_curve, write_particles
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import sys
from utils.general_utils import build_scaling_rotation
image_scale = 1.0
torch.autograd.set_detect_anomaly(True)

def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, static_threshold=0.01, fps=60):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")


    with torch.no_grad():
        xyz = gaussians.get_xyz
        # deform_code = gaussians.get_deform_code
        deform_code = deform.code_field(xyz)
        # gate = deform.deform.get_gate(deform_code)

        dxyz, _, _ = deform.step(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
        sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
        sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
        static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
        for i in range(75):
            dxyz_t, _, _ = deform.step(xyz, sampled_time[:, i:i + 1], deform_code, 1 / fps)
            static_mask &= ((dxyz_t - dxyz).norm(dim=1) < static_threshold)
        motion_mask = ~static_mask

    xyz = gaussians.get_xyz
    # deform_code = gaussians.get_deform_code
    deform_code = deform.code_field(xyz)

    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
    d_xyz = torch.zeros_like(gaussians.get_xyz)
    d_rotation = torch.zeros_like(gaussians.get_rotation)
    d_scaling = torch.zeros_like(gaussians.get_scaling)
    t0 = torch.zeros_like(xyz[..., :1])

    d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        xyz[static_mask], t0[static_mask], deform_code[static_mask]
    )
    #Sorting views
    vels = []
    position = []


    views.sort(key = lambda v:v.fid.item()) 
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:

            view.load2device()

        fid = view.fid

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step(
            xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / fps
        )
        
        '''
        def get_vel_field(self, xyz, time_emb, deform_code, dt = 1/60, max_time=0.75):
        '''
  
        vel = deform.get_vel_field( xyz[motion_mask], d_xyz[motion_mask], time_input[motion_mask], deform_code[motion_mask], 1 / 60)

        position.append(xyz[motion_mask] + d_xyz[motion_mask])

        vels.append(vel)

    # vels = torch.stack(vels, dim = 0).cpu().numpy() # 14, n, 3
    
    # position = torch.stack(position, dim = 0).cpu().numpy()
    return position, vels
def prepare_gt(dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args, static_threshold=0.01, fps=30):
    
    def build_rotation(q):
        # norm = torch.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])

        # q = q / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R


    def build_scaling_rotation(s, r):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        R = build_rotation(r)

        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]

        L = R @ L
        return L
    def count_singular_matrices(mats, threshold=0):

        dets = torch.linalg.det(mats)
        singular_mask = (dets.abs() == threshold)

        count = singular_mask.sum().item()
        return count, singular_mask



    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[image_scale])
    deform = DeformModel(max_time=dataset.max_time, light=dataset.light, physics_code=dataset.physics_code)
    # print(">>", dataset.K, dataset.deform_type,dataset.is_blender, dataset.skinning, dataset.hyper_dim, dataset.node_num, dataset.pred_opacity, dataset.pred_color, dataset.use_hash, dataset.hash_time, dataset.d_rot_as_res, dataset.local_frame, dataset.progressive_brand_time, dataset.max_d_scale)
    deform.load_weights(dataset.model_path)
    # print('---', deform.deform.skinning, deform.deform.hyper_dim, deform.deform.nodes.shape, deform.deform.with_node_weight, deform.deform.d_rot_as_res)
    # sys.exit(0)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gts, vels = render_set(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, static_threshold, fps=fps)
    
 
    opacitiy = gaussians.get_opacity.squeeze()
    grid_size = phys_args.density_grid_size

    opacity_threshold = phys_args.opacity_threshold
    with torch.no_grad():

            
        train_cams, test_cams, cameras_extent = scene.overwrite_alphas(pipeline, dataset, deform, static_threshold)
        cam_info = {
            "train_cams": train_cams,
            "test_cams": test_cams,
            "cameras_extent": cameras_extent,
        }
        vol = gts[0]
        curr_grid_size = 0.0075
        vol_surface = torch.arange(vol.shape[0]).to(vol.device).to(torch.int64)


    return gts, vol, opacitiy, torch.tensor([curr_grid_size]), vol_surface, cam_info, vels


def forward(estimator: Estimator_exp, img_backward=True):
    dt = estimator.simulator.dt_ori[None]
    # while True:
    for idx in tqdm(range(estimator.max_f - 1), desc="Forwarding"):

        if idx == 0:
            estimator.initialize()
            estimator.simulator.set_dt(dt)
        else:
            pass
            # estimator.update_volume(idx)
        x = estimator.forward(idx, img_backward)
        '''
        if not estimator.succeed():
            dt /= 2
            print('cfl condition dissatisfy, shrink dt {}, step cnt {}'.format(dt, estimator.simulator.n_substeps[None] * 2))
        else:
        '''
    #     break

def backward(estimator: Estimator_exp):

    max_f = estimator.max_f - 1
    pbar = trange(max_f)
    pbar.set_description(f"[Backward]")
    
    estimator.loss.grad[None] = 1
    estimator.clear_grads()

    for ri in pbar:
        # i = max_f - 1 - ri
        i = ri
        # if i > 0:
            # estimator.backward(i)
        # else:
        mu_grad, lam_grad, \
        yield_stress_grad, viscosity_grad, \
        friction_alpha_grad, cohesion_grad, rho_grad = estimator.backward(i)
        estimator.init_rhos.backward(retain_graph=True, gradient=rho_grad)
        estimator.init_mu.backward(retain_graph=True, gradient=mu_grad)
        estimator.init_lam.backward(retain_graph=True, gradient=lam_grad)
        estimator.yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
        estimator.plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
        estimator.friction_alpha.backward(retain_graph=True, gradient=friction_alpha_grad)
        estimator.cohesion.backward(gradient=cohesion_grad)

def forward_n_backward(estimator, ite):
    dt = estimator.simulator.dt_ori[None]

    loss = 0.
    
    estimator.init_F_C()

    # estimator.initialize()
    for idx in range(estimator.max_f):
        
        estimator.initialize()
        estimator.zero_grad()
        # estimator.loss[None] = 0.0
        
        
        estimator.simulator.set_dt(dt)
        
        estimator.forward(idx)

        loss += estimator.simulator.loss[None]
        estimator.simulator.loss.grad[None] = 1
        estimator.clear_grads()
        # estimator.loss.grad[None] = 1
        
   
        mu_grad, lam_grad, \
        yield_stress_grad, viscosity_grad, \
        friction_alpha_grad, cohesion_grad, rho_grad, velocity_grad, position_grad = estimator.backward(idx)
        print("Loss:", estimator.simulator.loss[None])
        print("E:", 10 **estimator.E, "Nu:",  estimator.get_nu())
        # print(mu_grad, lam_grad)
        nan_mask = torch.isnan(mu_grad)
        nan_indices = torch.nonzero(nan_mask, as_tuple=False)
        # print(f"NaN found at mu at indices: {nan_indices.squeeze().tolist()}")
        mu_grad[nan_indices.squeeze()] = 0
        nan_mask = torch.isnan(lam_grad)
        nan_indices = torch.nonzero(nan_mask, as_tuple=False)
        # print(f"NaN found at lam at indices: {nan_indices.squeeze().tolist()}")
        lam_grad[nan_indices.squeeze()] = 0
        
        # print(torch.isnan(mu_grad).sum(), torch.isnan(lam_grad).sum())
        # print(torch.isnan(estimator.init_mu).any(), torch.isnan(estimator.init_lam).any())
        # estimator.init_rhos.backward(retain_graph=True, gradient=rho_grad)
        estimator.init_mu.backward(retain_graph=True, gradient=mu_grad)


        # estimator.vels.backward(retain_graph = True, gradient = velocity_grad)
        # estimator.gts.backward(retain_graph = True, gradient = position_grad)


        estimator.init_lam.backward(retain_graph=True, gradient=lam_grad)
        
        estimator.yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
        estimator.plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
        estimator.friction_alpha.backward(retain_graph=True, gradient=friction_alpha_grad)
        estimator.cohesion.backward(gradient=cohesion_grad)
        
        estimator.step(ite)
        # sys.exit(0)
    return loss

def train(estimator: Estimator_exp, phys_args, max_f=None):
    losses = []
    estimated_params = []
    
    if estimator.stage[None] == Estimator_exp.velocity_stage:
        iter_cnt = phys_args.vel_iter_cnt
    elif estimator.stage[None] == Estimator_exp.physical_params_stage:
        iter_cnt = phys_args.iter_cnt

    if max_f is not None:
        estimator.max_f = max_f


    for i in range(150):
        # 1. record current params
        d = {}
        param_groups = estimator.get_optimizer().param_groups
        report_msg = ''
        report_msg += f'iter {i}'
        # report_msg += f'\nvelocity: {estimator.init_vel.cpu().detach().tolist()}'
        for params in param_groups:
            name = params['name']
            p = params['params'][0].detach().cpu()
            if name == 'Poisson ratio':
                p = estimator.get_nu().detach().cpu()
                report_msg += f'\n{name}: {p}'
            elif name in ['Youngs modulus', 'Yield stress', 'plastic viscosity', 'shear modulus', 'bulk modulus']:
                p = 10**p
                report_msg += f'\n{name}: {p}'
            #TODO: optimization
            if name != 'velocity' and name != 'position':
                d.update({name: p.item()})
            else:
                d.update({name: p})
        print(report_msg)
        estimated_params.append(d)
        
        # 2. forward, backward, and update
        L = forward_n_backward(estimator, i)
        print("Momentum Loss:", L)
        # track_points_with_given_velocity(estimator.gts.data.cpu().numpy(), estimator.vels.data.cpu().numpy(), 20,  save_dir="analysis/opt_pos_and_vels", gif_name = "trajectory_with_velocity_"+ str(i) +".gif", indx = i)

        losses.append(L)
        # 3. record loss and save best params
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        # print("Best params: ", best_params, 'in {} iteration'.format(min_idx))
        print("Min loss: {}".format(losses[min_idx]))

    return losses, estimated_params

def export_result(dataset, phys_args, estimator: Estimator_exp, losses, estimated_params, config_id):
    save_attr = ['mpm_iter_cnt', 'rho', 'voxel_size', 'gravity', 'bc', 'fps', 'density_grid_size']
    pred = dict()
    pred['config_id'] = config_id
    v = estimator.init_vel.detach().cpu().numpy().tolist()
    pred['vel'] = v
    min_idx = losses.index(min(losses))
    best_params = estimated_params[min_idx]
    # best_params = estimated_params[-1]
    mat_params = dict()
    m = phys_args.material
    mat_params['material'] = m
    if (m == ExpSimulator.von_mises and estimator.simulator.non_newtonian == 1) or \
        m == ExpSimulator.viscous_fluid:
        # non_newtonian & newtonian
        mu = best_params['shear modulus']
        kappa = best_params['bulk modulus']
        mat_params['mu'] = mu
        mat_params['kappa'] = kappa
    else:
        # elasticity, drucker_prager, plasticine
        if 'Youngs modulus' in best_params and 'Poisson ratio' in best_params:
            E = best_params['Youngs modulus']
            nu = best_params['Poisson ratio']
        else:
            E = float((10 ** estimator.E).detach().cpu().numpy())
            nu = float((estimator.get_nu()).detach().cpu().numpy())
        mat_params['E'] = E
        mat_params['nu'] = nu
    
    if m == ExpSimulator.drucker_prager:
        mat_params['friction_alpha'] = best_params['friction angle']

    if m == ExpSimulator.von_mises:
        ys = best_params['Yield stress']
        mat_params['yield_stress'] = ys
        if estimator.simulator.non_newtonian == 1:
            eta = best_params['plastic viscosity']
            mat_params['plastic_viscosity'] = eta
    
    pred['mat_params'] = mat_params
    for attr in save_attr:
        pred[attr] = getattr(phys_args, attr)
    
    with open(os.path.join(dataset.model_path, f'{config_id}-pred.json'), 'w') as f:
        json.dump(pred, f, indent=4)
def assign_gs_to_pcd(xyz, xyz_opacity, dataset, opt, pipe, cam_info, grid_size=0.12, scene=None):
    # TODO remove useless params
    from scene.gaussian_model import BasicPointCloud
    from utils.sh_utils import SH2RGB
    xyz = xyz.cpu().detach().numpy()
    num_pts= xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    gaussians = GaussianModel(dataset.sh_degree)
    if scene is None:
        scene = Scene(dataset, gaussians, resolution_scales=[1.0], pcd=pcd, cam_info=cam_info)
    xyz_opacity = xyz_opacity.reshape(-1, 1)
    scene.gaussians._opacity = torch.nn.Parameter(scene.gaussians.inverse_opacity_activation(torch.clamp(xyz_opacity, max=1-1e-4)).requires_grad_(True))
    scales = torch.ones_like(xyz_opacity) * grid_size / 32 * 0.5
    scene.gaussians._scaling = torch.nn.Parameter(scene.gaussians.scaling_inverse_activation(scales).requires_grad_(True))
    return scene
if __name__ == "__main__":
    # Set up command line argument parser
    start_time = time.time()

    parser = ArgumentParser(description="Physical parameter estimation")
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--config_file", default='config/torus.json', type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--static_threshold", default=0.01, type=float)
    parser.add_argument('--fps', type=int, default=30)
    gs_args, phys_args = get_combined_args(parser)
    # for key, value in vars(gs_args).items():
    #     print(f"{key}: {value}")
    # sys.exit(0)
    config_id = phys_args.id
    print(phys_args)
    safe_state(gs_args.quiet)

    # 1. train def gs
    dataset = model.extract(gs_args)
    '''
    dict_items([('sh_degree', 3), ('source_path', '/media/lang/My Passport/Python Project/gic/data/pacnerf/torus'), ('model_path', 'output/pacnerf/torus'), ('config_path', 'config/pacnerf/torus.json'), ('images', 'images'), ('resolution', -1), ('white_background', False), ('data_device', 'cuda'), ('num_frame', -1), ('eval', True), ('load2gpu_on_the_fly', False), ('is_blender', True), ('is_6dof', False), ('x_multires', 10), ('t_multires', 6), ('timenet', True), ('time_out', 30), ('num_basis', 10), ('num_coeff_set_per_basis', 1), ('channel_mb', 256), ('depth_mb', 8), ('channel_cn', 256), ('depth_cn', 2), ('softmax', False), ('num_attribute', 4), ('res_scale', 1), ('dxyz_scale', 1.0)])
    '''
    # print(vars(op.extract(gs_args)).items())
    '''
    dict_items([('iterations', 40000), ('warm_up', 3000), ('position_lr_init', 0.00016), ('position_lr_final', 1.6e-06), ('position_lr_delay_mult', 0.01), ('position_lr_max_steps', 30000), ('deform_lr_max_steps', 40000), ('feature_lr', 0.0025), ('opacity_lr', 0.1), ('scaling_lr', 0.001), ('rotation_lr', 0.001), ('percent_dense', 0.01), ('lambda_dssim', 0.2), ('densification_interval', 100), ('opacity_reset_interval', 3000), ('densify_from_iter', 500), ('densify_until_iter', 15000), ('densify_grad_threshold', 0.00015), ('reg_rigid', False), ('reg_scale', True), ('reg_tgs', False), ('reg_alpha', True), ('num_knn', 20), ('tgs_bound', 0.01), ('tgs_max_screen_size', 3), ('tgs_densify_until_iter', 30000)])
    '''

    torch.cuda.empty_cache()

    # 2. estimate velocity
    # gts, vol, opacitiy, torch.tensor([curr_grid_size]), vol_surface, cam_info
    gts, vol, vol_densities, grid_size, volume_surface, cam_info, vels = prepare_gt(model.extract(gs_args), gs_args.iteration, pipeline.extract(gs_args), phys_args)


    
    torch.cuda.empty_cache()
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    ti._logging.set_logging_level(ti._logging.ERROR)
    print('gt point count: {}'.format(gts[0].shape[0]))

    scene = assign_gs_to_pcd(vol, vol_densities, dataset, op.extract(gs_args), 
                                    pipeline.extract(gs_args), 
                                    cam_info, phys_args.density_grid_size)
    estimator = Estimator_exp(phys_args, 'float32', gts, vels,  surface_index=volume_surface, init_vol=vol, dynamic_scene=None, 
                        image_scale=image_scale, pipeline=pipeline.extract(gs_args), image_op=op.extract(gs_args))
    estimator.set_scene(scene)
    max_f = len(gts) - 1
    estimator.set_stage(Estimator_exp.physical_params_stage)
    losses, e_s = train(estimator, phys_args, max_f)

    print(phys_args)
    # print(estimator.init_vel)
    print(e_s)
    print(config_id)
    export_result(dataset, phys_args, estimator, losses, e_s, config_id)
    print("consume time {}".format(time.time() - start_time))
    