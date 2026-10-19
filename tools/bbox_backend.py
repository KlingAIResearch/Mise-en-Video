import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mev_matplotlib_cache")

import json
import cv2
from glob import glob

from PIL import Image

import torch
import torch.nn.functional as F

from utils.vis_utils import load_depth_to_point_cloud, SMPLVisualizer, visualize_depth_as_point_cloud
from utils.train_utils import rotmat_to_aa, rebuild_intrinsics

from utils.align_utils import get_scene_camera, convert_to_full_img_cam, \
    get_scene_intr, ransac_fit_offset_parallel, adjust_smpl_vertices, compute_oriented_bbox, \
    get_body_forward_direction, write_oabb_to_ply, get_scene_depth_cuda, get_scene_part_mask_cuda

from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)
from pytorch3d.structures import Meshes



class UnlitShader(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        texels = meshes.sample_textures(fragments)
        pixel_colors = texels[:, :, :, 0, :]
        mask = fragments.pix_to_face[:, :, :, 0] >= 0
        alpha = mask.float().unsqueeze(-1)
        rgba_image = torch.cat([pixel_colors, alpha], dim=-1)
        rgba_image[~mask] = 0.0
        return rgba_image


def align_human_to_scene(video_base, save_vis=False, keyframe_interval=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    json_info_path = os.path.join(video_base, "info.json") 
    if not os.path.isfile(json_info_path):
        print(f"Missing video metadata: {json_info_path}")
        return False
    with open(json_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    match_info_path = os.path.join(video_base, "matched_id.json")
    if not os.path.isfile(match_info_path):
        print(f"Missing identity mapping: {match_info_path}")
        return False
    with open(match_info_path, 'r', encoding='utf-8') as f:
        match_info = json.load(f)
    match_info = extract_shot_ids(match_info)

    num_shot = int(data['shot_num_from_caption'])
    visualizer = SMPLVisualizer(model_type="smpl", device=device)

    for shot_idx in range(num_shot):
        shot_key = f"shot_{shot_idx}"
        shot_dir = os.path.join(video_base, shot_key)

        if not os.path.isdir(shot_dir): continue

        scene_info_path = os.path.join(shot_dir, "pi3_all_results.npz")
        track_info_path = os.path.join(shot_dir, "tracks.npy")
        hmr_dir = os.path.join(shot_dir, "camerahmr", "camerahmr_json")

        if not (os.path.exists(scene_info_path) and os.path.exists(track_info_path) and os.path.exists(hmr_dir)):
            print(f"[shot_{shot_idx}] missing alignment inputs; skipping")
            continue

        scene_info = get_scene_camera(scene_info_path)
        track_info = np.load(track_info_path, allow_pickle=True).item()
        valid_shot_ids = [str(x) for x in match_info.get(shot_key, [])]
        
        hmr_list = sorted(glob(os.path.join(hmr_dir, "*.json")))
        if not hmr_list: continue

        frame_map = {}
        all_frames = []
        for p in hmr_list:
            fid = int((p.split('/')[-1]).split('.')[0])
            frame_map[fid] = p
            all_frames.append(fid)
        all_frames.sort()

        scene_intr_raw = get_scene_intr(scene_info)
        human_bbox_path = os.path.join(shot_dir, "human_bbox.npz")
        final_shot_human_bbox = {}

        # Run expensive depth alignment on keyframes, then interpolate its two
        # scalar parameters for the remaining frames.
        keyframe_ids = [
            all_frames[index]
            for index in range(0, len(all_frames), keyframe_interval)
        ]
        if all_frames[-1] not in keyframe_ids:
            keyframe_ids.append(all_frames[-1])
        keyframe_ids = sorted(list(set(keyframe_ids)))
        keyframe_files = [frame_map[fid] for fid in keyframe_ids]


        print(f"[shot_{shot_idx}] aligning {len(keyframe_ids)} keyframes")
        for i, hmr_json_path in enumerate(keyframe_files):
            try:
                with open(hmr_json_path, 'r', encoding='utf-8') as f:
                    hmr_data = json.load(f)

                if valid_shot_ids:
                    keys_to_remove = [k for k in list(hmr_data.keys()) if str(k) not in valid_shot_ids]
                    for k in keys_to_remove:
                        del hmr_data[k]

                if not hmr_data:
                    continue

                fid = int((hmr_json_path.split('/')[-1]).split('.')[0])

                bbox = get_human_bbox(
                        hmr_data=hmr_data,
                        scene_info=scene_info,
                        match_info=match_info,
                        track_info=track_info,
                        frame_idx=fid,
                        shot_dir=shot_dir,
                        save_vis=save_vis,
                        Visualizer=visualizer,
                        device=device,
                        precomputed_params=None,
                        scene_intr_raw=scene_intr_raw[fid]
                )
                final_shot_human_bbox[fid] = bbox

            except Exception as e:
                print(f"Error processing keyframe {hmr_json_path}: {e}")

        inter_frame_ids = [fid for fid in all_frames if fid not in final_shot_human_bbox]
        
        if inter_frame_ids:
            print(f"[shot_{shot_idx}] interpolating {len(inter_frame_ids)} remaining frames")
            for fid in inter_frame_ids:
                prev_k = max([k for k in keyframe_ids if k < fid], default=None)
                next_k = min([k for k in keyframe_ids if k > fid], default=None)
                
                params_for_frame = {}
                
                if prev_k is not None and next_k is not None:
                    res_prev = final_shot_human_bbox.get(prev_k, {})
                    res_next = final_shot_human_bbox.get(next_k, {})
                    
                    t = (fid - prev_k) / float(next_k - prev_k)
                    
                    all_hids = set(res_prev.keys()) | set(res_next.keys())
                    
                    for hid in all_hids:
                        if valid_shot_ids and str(hid) not in valid_shot_ids:
                            continue

                        if hid in res_prev and hid in res_next:
                            alpha = res_prev[hid]['alpha'] * (1-t) + res_next[hid]['alpha'] * t
                            gamma = res_prev[hid]['gamma'] * (1-t) + res_next[hid]['gamma'] * t
                            params_for_frame[hid] = {'alpha': alpha, 'gamma': gamma}
                        elif hid in res_prev:
                            params_for_frame[hid] = {'alpha': res_prev[hid]['alpha'], 'gamma': res_prev[hid]['gamma']}
                        elif hid in res_next:
                            params_for_frame[hid] = {'alpha': res_next[hid]['alpha'], 'gamma': res_next[hid]['gamma']}
                
                elif prev_k is not None:
                    res_prev = final_shot_human_bbox.get(prev_k, {})
                    for hid, val in res_prev.items():
                        if valid_shot_ids and str(hid) not in valid_shot_ids: continue
                        params_for_frame[hid] = {'alpha': val['alpha'], 'gamma': val['gamma']}
                elif next_k is not None:
                    res_next = final_shot_human_bbox.get(next_k, {})
                    for hid, val in res_next.items():
                        if valid_shot_ids and str(hid) not in valid_shot_ids: continue
                        params_for_frame[hid] = {'alpha': val['alpha'], 'gamma': val['gamma']}

                hmr_json_path = frame_map[fid]
                try:
                    with open(hmr_json_path, 'r', encoding='utf-8') as f:
                        hmr_data = json.load(f)
                    
                    if valid_shot_ids:
                        keys_to_remove = [k for k in hmr_data.keys() if str(k) not in valid_shot_ids]
                        for k in keys_to_remove:
                            del hmr_data[k]

                    if hmr_data:
                        bbox = get_human_bbox(
                            hmr_data=hmr_data,
                            scene_info=scene_info,
                            match_info=match_info,
                            track_info=track_info,
                            frame_idx=fid,
                            shot_dir=shot_dir,
                            save_vis=save_vis,
                            Visualizer=visualizer,
                            device=device,
                            precomputed_params=params_for_frame,
                            scene_intr_raw=scene_intr_raw[fid]
                        )
                        final_shot_human_bbox[fid] = bbox
                
                except Exception as e:
                    print(f"Error processing frame {fid}: {e}")

        if final_shot_human_bbox:
            sorted_bbox = {k: final_shot_human_bbox[k] for k in sorted(final_shot_human_bbox)}
            np.savez(human_bbox_path, shot_human_bbox=sorted_bbox)
            print(f"[Shot {shot_idx}] Saved {len(sorted_bbox)} frames to {human_bbox_path}")
        else:
            print(f"[shot_{shot_idx}] no valid human bbox to save")

    return True


def get_human_bbox(hmr_data, scene_info, match_info, track_info, frame_idx, shot_dir, save_vis=False,
                    max_ransac_iters=2000, ransac_threshold=0.0005, device='cuda',
                    erosion_kernel_size=5, erosion_iterations=10, Visualizer=None,
                    precomputed_params=None, scene_intr_raw=None):

    if Visualizer is None:
        Visualizer = SMPLVisualizer(model_type="smpl", device=device)
    if scene_intr_raw is None:
        scene_intr_raw = get_scene_intr(scene_info)[frame_idx]

    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    hmr_ids = hmr_data.keys()
    hmr_align_param = {}

    shot_name = shot_dir.split('/')[-1]
    valid_id = match_info[shot_name]

    if len(scene_intr_raw.shape) == 3:
        scene_intr_raw = scene_intr_raw[frame_idx]

    # Intermediate frames reuse interpolated alignment parameters and do not need a
    # dense scene-depth reconstruction. Load it lazily only for RANSAC or debug PLY.
    scene_depth = None

    if save_vis:
        scene_depth, _, _, _ = get_scene_depth_cuda(scene_info, f_id=frame_idx)
        scene_depth = scene_depth.to(device)
        os.makedirs(os.path.join(shot_dir, 'depth_ply'), exist_ok=True)
        os.makedirs(os.path.join(shot_dir, 'oabb_ply'), exist_ok=True)

        scene_h, scene_w = scene_info['points'][frame_idx].shape[-3:-1]

        rgb_path = os.path.join(shot_dir, "images", f"{(frame_idx):04d}.jpg")
        rgb_raw = Image.open(rgb_path).convert("RGB")
        rgb_raw = torch.from_numpy(np.array(rgb_raw)) / 255.0
        rgb = F.interpolate(rgb_raw.unsqueeze(0).permute(0, 3, 1, 2), size=(scene_h, scene_w), mode='nearest')
        rgb = rgb.squeeze(0).permute(1, 2, 0).to(device)

        depth_ply_path = os.path.join(shot_dir, 'depth_ply', f'depth_{frame_idx:04d}.ply')
        visualize_depth_as_point_cloud(scene_depth, scene_intr_raw, depth_ply_path, rgb, dump=True)

    for hmr_id in hmr_ids:

        if not hmr_id in valid_id:
            continue

        use_fast_mode = False
        fixed_alpha = 1.0
        fixed_gamma = 0.0
        
        # JSON IDs are strings, while cached interpolation keys may be integers.
        if precomputed_params is not None:
            if hmr_id in precomputed_params:
                val = precomputed_params[hmr_id]
                fixed_alpha = val['alpha']
                fixed_gamma = val['gamma']
                use_fast_mode = True
            elif int(hmr_id) in precomputed_params:
                val = precomputed_params[int(hmr_id)]
                fixed_alpha = val['alpha']
                fixed_gamma = val['gamma']
                use_fast_mode = True

        try:
            smpl_params = hmr_data[hmr_id]
            out_cam = torch.tensor(smpl_params["out_cam"]).unsqueeze(0).to(device)
            global_orient = torch.tensor(smpl_params["smpl_params"]["global_orient"]).to(device)
            body_pose = torch.tensor(smpl_params["smpl_params"]["body_pose"]).to(device)
            betas = torch.tensor(smpl_params["smpl_params"]["betas"]).to(device)
            box_center = torch.tensor(smpl_params["box_center"]).to(device)
            box_size = torch.tensor(smpl_params["box_size"]).max() * 200.0
            box_size = box_size.to(device)
            cam_int = torch.tensor(smpl_params["cam_int"]).to(device)
            img_size = torch.tensor(smpl_params["img_size"]).to(device)
            focal_length = cam_int[0, 0]
            
            trans = convert_to_full_img_cam(out_cam, box_size, box_center, img_size[1], img_size[0], focal_length)
            poses = torch.cat([global_orient, body_pose], dim=1)
            poses = rotmat_to_aa(poses.reshape(-1, 3, 3)).reshape(1, -1)

            hmr_intr = torch.tensor([
                [focal_length, 0, img_size[1] / 2],
                [0, focal_length, img_size[0] / 2],
                [0, 0, 1]
            ]).to(device)

            if use_fast_mode:
                scene_h, scene_w = scene_info['points'][frame_idx].shape[-3:-1] 
                
                scene_intr = rebuild_intrinsics(torch.tensor(scene_intr_raw).unsqueeze(0), (scene_h, scene_w)).squeeze(0)
                scale = img_size[1] / scene_w
                scene_intr_resized = torch.tensor([
                    [scene_intr[0, 0] * scale, 0, scene_intr[0, 2] * scale],
                    [0, scene_intr[1, 1] * scale, scene_intr[1, 2] * scale],
                    [0, 0, 1]
                ]).to(device)
                
                alpha = fixed_alpha
                gamma = fixed_gamma
                adjusted_trans = trans.clone()
                adjusted_trans[:, 2] = adjusted_trans[:, 2] * scene_intr_resized[0, 0] / hmr_intr[0, 0]

            else:
                if scene_depth is None:
                    scene_depth, _, _, _ = get_scene_depth_cuda(scene_info, f_id=frame_idx)
                    scene_depth = scene_depth.to(device)

                scene_part_mask = get_scene_part_mask_cuda(
                    scene_info, track_info, f_id=frame_idx, h_id=int(hmr_id),
                )
                scene_h, scene_w = scene_part_mask.shape
                scene_part_mask = cv2.erode(scene_part_mask, kernel, iterations=erosion_iterations)

                scene_intr = rebuild_intrinsics(torch.tensor(scene_intr_raw).unsqueeze(0), (scene_h, scene_w)).squeeze(0)
                scale = img_size[1] / scene_w
                scene_intr_resized = torch.tensor([
                    [scene_intr[0, 0] * scale, 0, scene_intr[0, 2] * scale],
                    [0, scene_intr[1, 1] * scale, scene_intr[1, 2] * scale],
                    [0, 0, 1]
                ]).to(device)

                adjusted_trans = trans.clone()
                adjusted_trans[:, 2] = adjusted_trans[:, 2] * scene_intr_resized[0, 0] / hmr_intr[0, 0]

                vertices, joints = Visualizer.get_vertices(poses, betas, adjusted_trans, "neutral")

                smpl_depth, smpl_mask = Visualizer.get_smpl_depth(vertices, scene_intr)
                smpl_mask = torch.logical_and(torch.tensor(scene_part_mask > 0).to(device), smpl_mask)
                
                flattened_scene_depth = scene_depth[smpl_mask].view(-1)
                flattened_smpl_depth = smpl_depth[smpl_mask].view(-1)

                # Point-cloud height ratios estimate the scene-to-SMPL scale.
                smpl_point_cloud = load_depth_to_point_cloud(smpl_depth, scene_intr, mask=smpl_mask.cpu().numpy())
                human_point_cloud = load_depth_to_point_cloud(scene_depth, scene_intr, mask=smpl_mask.cpu().numpy())

                if smpl_point_cloud.shape[0] == 0 or human_point_cloud.shape[0] == 0:
                    continue

                smpl_vertices_np = smpl_point_cloud
                bbox_edges_smpl = smpl_vertices_np.max(axis=0).values - smpl_vertices_np.min(axis=0).values
                smpl_y_axis_length = float(bbox_edges_smpl[1])

                human_vertices_np = human_point_cloud
                bbox_edges_human = human_vertices_np.max(axis=0).values - human_vertices_np.min(axis=0).values
                human_y_axis_length = float(bbox_edges_human[1])

                alpha = human_y_axis_length / smpl_y_axis_length
                gamma = ransac_fit_offset_parallel(flattened_smpl_depth, flattened_scene_depth, alpha, max_iters=max_ransac_iters, threshold=ransac_threshold, device=device)[0]

            vertices, joints = Visualizer.get_vertices(poses, betas, adjusted_trans, "neutral") 
            adjusted_vertices = adjust_smpl_vertices(vertices, scene_intr_resized, alpha, gamma)

            forward, axes = get_body_forward_direction(poses[0], Visualizer.models['neutral'], method="upper_joints")
            oabb = compute_oriented_bbox(adjusted_vertices, axes)
            
            hmr_align_param[hmr_id] = {
                'pose': poses.detach().to('cpu'),
                'betas': betas.detach().to('cpu'),
                'adjusted_trans': adjusted_trans.detach().to('cpu'),
                'alpha': alpha,
                'gamma': gamma,
                'oabb': oabb.detach().to('cpu') if isinstance(oabb, torch.Tensor) else oabb,
                'scene_intr': scene_intr_resized.detach().to('cpu'),
                'width': int(img_size[0]),
                'height': int(img_size[1])
            }

            if save_vis:
                oabb_ply_path = os.path.join(shot_dir, 'oabb_ply', f'{frame_idx:04d}_{hmr_id}.ply')
                write_oabb_to_ply(oabb, save_path=oabb_ply_path)

        except Exception as e:
            print(f"[ERROR] frame={frame_idx} hmr_id={hmr_id}: {e}")
            continue 

    return hmr_align_param


def extract_shot_ids(input_dict):
    """Convert identity groups into ``shot_N -> local IDs`` lookup tables."""
    shot_id_dict = {}

    if isinstance(input_dict, list):
        if len(input_dict) == 0:
            raise ValueError("Identity mapping list is empty")
        input_dict = input_dict[0]
        
    if not isinstance(input_dict, dict):
        raise TypeError(f"Expected an identity mapping, got {type(input_dict).__name__}")

    for shot_id_list in input_dict.values():
        for shot_id_str in shot_id_list:
            try:
                shot_part, id_part = shot_id_str.split('-')
                pure_id = id_part.split('_')[1]

                if shot_part not in shot_id_dict:
                    shot_id_dict[shot_part] = []
                shot_id_dict[shot_part].append(pure_id)
            except Exception as e:
                print(f"Error processing shot id {shot_id_str}: {e}")
    
    return shot_id_dict


def render_oabb_colormap_camera_coord_batched(
    oabb_list,
    K_org_list,
    img_size=(480, 640),
    device="cuda"
):
    B = len(oabb_list)

    K = torch.zeros((B, 4, 4), device=device)
    for i, K_org in enumerate(K_org_list):
        K[i, :3, :3] = K_org.to(device)
        K[i, 2, 2] = 0
        K[i, 2, -1] = 1
        K[i, -1, 2] = 1

    cameras = PerspectiveCameras(
        device=device,
        K=K,
        R=torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1),  # [B, 3, 3]
        T=torch.zeros(3, device=device).unsqueeze(0).repeat(B, 1),   # [B, 3]
        image_size=torch.tensor([img_size], device=device).repeat(B, 1),
        in_ndc=False
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1,
                cull_backfaces=True
            ),
            cameras=cameras
        ),
        shader=UnlitShader()
    )

    mesh = create_cube_mesh_batched(oabb_list, device=device)
    images = renderer(mesh) 
    images = torch.flip(images[..., :4], [1, 2])  

    return images

def create_cube_mesh_batched(verts_list, device='cuda'):
    """Create one heterogeneous PyTorch3D mesh per video frame."""
    all_verts_list = []
    all_faces_list = []
    all_textures_list = []

    verts_base = torch.tensor([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1], 
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]  
    ], dtype=torch.float32, device=device)

    face_base = torch.tensor([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], 
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], 
        [0, 4, 7], [0, 7, 3], [1, 6, 5], [1, 2, 6]  
    ], dtype=torch.int64, device=device)

    for frame_verts in verts_list:
        frame_verts = frame_verts.to(device)
        N_i = frame_verts.shape[0]
        frame_verts_flat = frame_verts.view(-1, 3)
        all_verts_list.append(frame_verts_flat)
        
        offsets = (torch.arange(N_i, device=device) * 8).view(-1, 1, 1)
        frame_faces = face_base.unsqueeze(0) + offsets
        frame_faces_flat = frame_faces.view(-1, 3)
        all_faces_list.append(frame_faces_flat)
        
        nocs_colors = verts_base.repeat(N_i, 1) * 0.5 + 0.5
        all_textures_list.append(nocs_colors)

    mesh = Meshes(
        verts=all_verts_list,
        faces=all_faces_list,
        textures=TexturesVertex(verts_features=all_textures_list)
    )
    return mesh
