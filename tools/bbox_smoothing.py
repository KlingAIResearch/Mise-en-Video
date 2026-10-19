import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from typing import Any, Dict, List
import numpy as np
import torch
import json
from pathlib import Path

from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from utils.vis_utils import SMPLVisualizer

import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(rotation_6d_to_matrix(rotation))


def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    if weight is None:
        x = torch.arange(kernel_size, dtype=torch.float32, device=data.device)
        x = x - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
    else:
        kernel = weight
    padded = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(padded, kernel)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMB_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23]
IGNORE_JOINT_INDICES = [16, 17, 18, 19, 20, 21, 22, 23]
EPS = 1e-8
MODE_NEAREST = "reflect"

LOCAL_BOX_VERTS = torch.tensor([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
    [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
    [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
], dtype=torch.float32)

with (Path(__file__).resolve().parents[1] / "smpl_vert_segmentation.json").open("r") as f:
    seg = json.load(f)

UPPER_PARTS = ["head", "neck", 'spine', 'spine1', 'spine2', "leftShoulder", "rightShoulder", "leftArm", "rightArm", "leftHand", "rightHand", 'leftForeArm', 'rightForeArm', 'leftHandIndex1', 'rightHandIndex1']
LOWER_PARTS = ["hips", 'leftUpLeg', 'rightUpLeg', 'leftLeg', 'rightLeg', 'leftFoot', 'rightFoot', 'leftToeBase', 'rightToeBase']
UPPER_PARTS = np.concatenate([seg[p] for p in UPPER_PARTS])
LOWER_PARTS = np.concatenate([seg[p] for p in LOWER_PARTS])


def smooth_smpl_pose_seq(pose_seq: np.ndarray, sigma: float = 2.0, fps: int = 25) -> np.ndarray:
    """Smooth an SMPL axis-angle sequence in continuous 6D rotation space."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = torch.from_numpy(pose_seq).float().to(device)  # (T, 72)
    T = pose.shape[0]

    poses = pose.view(T, 24, 3)
    poses_6d = axis_angle_to_rotation_6d(poses) # (T,24,6)


    poses_6d = poses_6d.view(T, 1, -1).permute(2, 1, 0)  # [D, 1, T]
    poses_6d = gaussian_filter_1d(
        poses_6d, kernel_size=9, sigma=max(sigma * fps / 30, EPS)
    )
    poses_6d = poses_6d.permute(2, 1, 0).view(T, 24, 6)
    poses_smooth = rotation_6d_to_axis_angle(poses_6d).view(T, 72)

    return poses_smooth.cpu().numpy()


def adjust_smpl_vertices(V_smpl, intr, a, b):
    """Align camera-space SMPL vertices with ``z_scene = a * z_smpl + b``."""
    if not isinstance(V_smpl, torch.Tensor):
        V_smpl = torch.tensor(V_smpl, dtype=torch.float32)
    device = V_smpl.device
    dtype = V_smpl.dtype

    if not isinstance(intr, torch.Tensor):
        intr = torch.tensor(intr, dtype=dtype, device=device)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    a = torch.tensor(a, dtype=dtype, device=device)
    b = torch.tensor(b, dtype=dtype, device=device)

    z_smpl = V_smpl[:, 2].clone()
    valid_mask = z_smpl > 1e-6
    z_smpl_valid = z_smpl[valid_mask]
    V_valid = V_smpl[valid_mask]

    if not torch.any(valid_mask):
        return V_smpl.clone()

    u = (V_valid[:, 0] * fx) / z_smpl_valid + cx
    v = (V_valid[:, 1] * fy) / z_smpl_valid + cy

    z_scene = a * z_smpl_valid + b
    z_scene = torch.clamp(z_scene, min=1e-6)

    x_scene = (u - cx) * z_scene / fx
    y_scene = (v - cy) * z_scene / fy
    V_aligned_valid = torch.stack([x_scene, y_scene, z_scene], dim=1)

    V_smpl_aligned = V_smpl.clone()
    V_smpl_aligned[valid_mask] = V_aligned_valid

    return V_smpl_aligned


def build_axes_seq_from_global_orient_with_gravity(
    pose_seq_np: np.ndarray,
    joints_seq: torch.Tensor,
    alpha_up: float = 0.15,
    eps: float = 1e-8,
    use_ema_up: bool = True,
):
    """Build stable body axes from torso gravity and pelvis orientation.

    The pelvis forward vector is projected onto the plane orthogonal to the
    torso up vector, removing pitch and roll while preserving heading.
    """
    device = joints_seq.device
    T = joints_seq.shape[0]

    up_raw = joints_seq[:, 12] - joints_seq[:, 3]

    if use_ema_up:
        up_s = torch.zeros_like(up_raw)
        u = up_raw[0] / (torch.norm(up_raw[0]) + eps)
        up_s[0] = u
        for t in range(1, T):
            x = up_raw[t] / (torch.norm(up_raw[t]) + eps)
            if torch.dot(x, u) < 0:
                x = -x
            u = (1 - alpha_up) * u + alpha_up * x
            u = u / (torch.norm(u) + eps)
            up_s[t] = u
    else:
        up_s = up_raw / (torch.norm(up_raw, dim=-1, keepdim=True) + eps)

    model_forward = torch.tensor([0.0, 0.0, 1.0], device=device)
    axes_list = []
    prev_f = None

    for t in range(T):
        aa = np.asarray(pose_seq_np[t][:3], dtype=np.float64)
        rot_mat = torch.tensor(R.from_rotvec(aa).as_matrix(), dtype=torch.float32, device=device)

        pelvis_f = rot_mat @ model_forward
        pelvis_f = pelvis_f / (torch.norm(pelvis_f) + eps)

        up = up_s[t]

        f = pelvis_f - torch.dot(pelvis_f, up) * up
        fn = torch.norm(f)

        if fn < 1e-3:
            f = prev_f if prev_f is not None else model_forward.clone()
            f = f / (torch.norm(f) + eps)
        else:
            f = f / (fn + eps)

        if prev_f is not None and torch.dot(f, prev_f) < 0:
            f = -f

        r = torch.cross(up, f, dim=0)
        r = r / (torch.norm(r) + eps)

        axes_list.append({
            "right": r.detach().cpu(),
            "up": up.detach().cpu(),
            "forward": f.detach().cpu(),
        })
        prev_f = f

    return axes_list


def compute_oriented_bbox_debug(vertices, axes, core_mask):
    """Fit a rigid upper-body OABB and extend it down to the lower body."""
    device = vertices.device
    upper_idx = torch.from_numpy(UPPER_PARTS).to(device)
    lower_idx = torch.from_numpy(LOWER_PARTS).to(device)

    upper_vertices = vertices[upper_idx[core_mask[upper_idx]]]
    lower_vertices = vertices[lower_idx[core_mask[lower_idx]]]

    centroid = torch.mean(upper_vertices, dim=0)
    vertices_centered = upper_vertices - centroid
    
    R_mat = torch.stack([axes["right"], axes["up"], axes["forward"]], dim=1).to(device)
    vertices_body = torch.matmul(vertices_centered, R_mat)
    
    min_vals = torch.min(vertices_body, dim=0)[0]
    max_vals = torch.max(vertices_body, dim=0)[0]

    lower_centered = lower_vertices - centroid
    lower_body = torch.matmul(lower_centered, R_mat)
    lower_min_up = torch.min(lower_body[:, 1])
    extend_length = min_vals[1] - lower_min_up

    min_vals_extended = min_vals.clone()
    min_vals_extended[1] = min_vals[1] - extend_length
    size_extended = max_vals - min_vals_extended

    body_vertices = torch.stack([
        torch.stack([min_vals_extended[0], min_vals_extended[1], min_vals_extended[2]]),
        torch.stack([max_vals[0], min_vals_extended[1], min_vals_extended[2]]),
        torch.stack([max_vals[0], max_vals[1], min_vals_extended[2]]),
        torch.stack([min_vals_extended[0], max_vals[1], min_vals_extended[2]]),
        torch.stack([min_vals_extended[0], min_vals_extended[1], max_vals[2]]),
        torch.stack([max_vals[0], min_vals_extended[1], max_vals[2]]),
        torch.stack([max_vals[0], max_vals[1], max_vals[2]]),
        torch.stack([min_vals_extended[0], max_vals[1], max_vals[2]]),
    ], dim=0)
    
    world_vertices = torch.matmul(body_vertices, R_mat.T) + centroid

    bbox_center_local = (min_vals_extended + max_vals) / 2.0
    bbox_center_world = torch.matmul(bbox_center_local, R_mat.T) + centroid
    
    oabb = {
        "center": bbox_center_world.detach().to('cpu'),
        "size": size_extended.detach().to('cpu'),
        "vertices": world_vertices.detach().to('cpu'),
        "forward": axes["forward"].detach().to('cpu'),
        "right": axes["right"].detach().to('cpu'),
        "up": axes["up"].detach().to('cpu'),
    }
    return oabb


def build_smoothed_oabb(center: torch.Tensor, size: torch.Tensor, axes: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Rebuild an OABB from its smoothed center, size, and axes."""
    device = center.device
    local_verts = LOCAL_BOX_VERTS.to(device) * size
    R_mat = torch.stack([axes["right"], axes["up"], axes["forward"]], dim=1).to(device)
    world_verts = torch.matmul(local_verts, R_mat.T) + center
    return {
        "center": center.cpu(),
        "size": size.cpu(),
        "vertices": world_verts.cpu(),
        "forward": axes["forward"].cpu(),
        "right": axes["right"].cpu(),
        "up": axes["up"].cpu()
    }


def smooth_axes_seq_quat(axes_list: List[Dict[str, torch.Tensor]], sigma: float = 2.0) -> List[Dict[str, torch.Tensor]]:
    """Smooth body axes in quaternion space and project them back to SO(3)."""
    T = len(axes_list)
    if T == 0:
        return []

    mats = np.zeros((T, 3, 3), dtype=np.float64)
    for i, ax in enumerate(axes_list):
        r = ax["right"].cpu().detach().numpy()
        u = ax["up"].cpu().detach().numpy()
        f = ax["forward"].cpu().detach().numpy()
        mats[i, 0] = r / (np.linalg.norm(r) + EPS)
        mats[i, 1] = u / (np.linalg.norm(u) + EPS)
        mats[i, 2] = f / (np.linalg.norm(f) + EPS)

    def proj_so3(M: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(M)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    Rrows = np.zeros_like(mats)
    Rrows[0] = proj_so3(mats[0])
    for i in range(1, T):
        Ri = proj_so3(mats[i])
        prev = Rrows[i-1]
        if np.trace(prev.T @ (-Ri)) > np.trace(prev.T @ Ri):
            Ri = -Ri
        Rrows[i] = Ri

    rots = R.from_matrix(np.transpose(Rrows, (0, 2, 1)))
    q = rots.as_quat()
    for i in range(1, T):
        if np.dot(q[i], q[i-1]) < 0:
            q[i] *= -1

    q_s = gaussian_filter1d(q, sigma=sigma, axis=0, mode=MODE_NEAREST)
    q_s /= (np.linalg.norm(q_s, axis=1, keepdims=True) + EPS)
    Rcols_s = R.from_quat(q_s).as_matrix()
    Rrows_s = np.transpose(Rcols_s, (0, 2, 1))

    out = []
    for i in range(T):
        Ri = proj_so3(Rrows_s[i])
        out.append({
            "right": torch.tensor(Ri[0], dtype=torch.float32),
            "up": torch.tensor(Ri[1], dtype=torch.float32),
            "forward": torch.tensor(Ri[2], dtype=torch.float32),
        })
    return out


def smooth_intr_seq(
    intr_seq: np.ndarray,
    sigma: float = 2.0,
    mode: str = "nearest",
) -> np.ndarray:
    """Smooth focal lengths and principal points over time."""
    if intr_seq.ndim != 3 or intr_seq.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T, 3, 3), got {intr_seq.shape}")
    
    T_frames = intr_seq.shape[0]
    if T_frames == 0:
        return intr_seq.copy()

    fx = intr_seq[:, 0, 0]
    fy = intr_seq[:, 1, 1]
    cx = intr_seq[:, 0, 2]
    cy = intr_seq[:, 1, 2]

    fx_s = gaussian_filter1d(fx, sigma=sigma, mode=mode)
    fy_s = gaussian_filter1d(fy, sigma=sigma, mode=mode)
    cx_s = gaussian_filter1d(cx, sigma=sigma, mode=mode)
    cy_s = gaussian_filter1d(cy, sigma=sigma, mode=mode)

    intr_smooth = np.zeros_like(intr_seq, dtype=np.float32)
    intr_smooth[:, 0, 0] = fx_s
    intr_smooth[:, 1, 1] = fy_s
    intr_smooth[:, 0, 2] = cx_s
    intr_smooth[:, 1, 2] = cy_s
    intr_smooth[:, 2, 2] = 1.0

    return intr_smooth


def smooth_shot_bboxes(
    video_base: str,
    sigma: float = 2.0,
    sigma_intr: float = 2.0,
    save_name: str = "human_bbox_smoothed.npz"
) -> bool:
    
    visualizer = SMPLVisualizer(model_type="smpl", device=DEVICE)
    
    json_info_path = os.path.join(video_base, "info.json") 
    if not os.path.isfile(json_info_path):
        print(f"{json_info_path} does not exist!")
        return False
        
    with open(json_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_shot = int(data['shot_num_from_caption'])

    smpl_model = visualizer.models['neutral']
    lbs_weights = smpl_model.lbs_weights
    closest_joints = torch.argmax(lbs_weights, dim=1)
    core_mask = ~torch.isin(closest_joints, torch.tensor(IGNORE_JOINT_INDICES, device=DEVICE))


    for shot_idx in range(num_shot):
        shot_dir = os.path.join(video_base, f"shot_{shot_idx}")
        bbox_path = os.path.join(shot_dir, "human_bbox.npz")
        if not os.path.exists(bbox_path):
            continue

        try:
            bbox_data = np.load(bbox_path, allow_pickle=True)
            shot_human_bbox = bbox_data["shot_human_bbox"].item()
        except Exception as e:
            print(f"[WARN] Load {bbox_path} failed: {e}")
            continue
            
        sorted_fids = sorted(shot_human_bbox.keys())
        if not sorted_fids:
            continue
        
        # Collect one trajectory per tracked subject.
        id_pose_trajectories = {}
        id_betas_trajectories = {}
        id_intr_trajectories = {}
        id_joints_trajectories = {}


        for f_id in sorted_fids:
            for hmr_id, data_dict in shot_human_bbox[f_id].items():
                if hmr_id not in id_pose_trajectories:
                    id_pose_trajectories[hmr_id] = []
                    id_betas_trajectories[hmr_id] = []
                    id_intr_trajectories[hmr_id] = []
                    id_joints_trajectories[hmr_id] = []
                
                pose = data_dict.get('pose', data_dict.get('poses_standing'))
                if torch.is_tensor(pose):
                    pose = pose.cpu().numpy()
                pose = pose.flatten()
                standing_pose = pose.copy()
                if len(standing_pose) == 72:
                    for j_idx in LIMB_JOINTS:
                        standing_pose[j_idx*3 : (j_idx+1)*3] = 0.0
                
                id_pose_trajectories[hmr_id].append(standing_pose)

                beta = data_dict['betas']
                if torch.is_tensor(beta): beta = beta.cpu().numpy()
                id_betas_trajectories[hmr_id].append(beta.flatten())

                intr = data_dict['scene_intr']
                if torch.is_tensor(intr):
                    intr = intr.cpu().numpy()
                id_intr_trajectories[hmr_id].append(np.array(intr))

                pose_t = torch.tensor(standing_pose, dtype=torch.float32, device=DEVICE)
                beta_t = torch.tensor(beta.flatten(), dtype=torch.float32, device=DEVICE)
                trans_t = torch.as_tensor(data_dict["adjusted_trans"], dtype=torch.float32, device=DEVICE)
                _, joints = visualizer.get_vertices(pose_t, beta_t, trans_t, "neutral")
                id_joints_trajectories[hmr_id].append(joints.squeeze().cpu())

        # Smooth pose, shape, intrinsics, and orientation independently.
        for hmr_id in id_pose_trajectories:
            pose_seq = np.stack(id_pose_trajectories[hmr_id], axis=0)
            id_pose_trajectories[hmr_id] = smooth_smpl_pose_seq(pose_seq, sigma=sigma)

        id_mean_betas = {}
        for hmr_id in id_betas_trajectories:
            mean_b = np.mean(np.stack(id_betas_trajectories[hmr_id], axis=0), axis=0)
            id_mean_betas[hmr_id] = torch.tensor(mean_b, dtype=torch.float32)

        id_smoothed_intr = {}
        for hmr_id in id_intr_trajectories:
            intr_seq = np.stack(id_intr_trajectories[hmr_id], axis=0)
            intr_smooth = smooth_intr_seq(intr_seq, sigma=sigma_intr, mode=MODE_NEAREST)
            id_smoothed_intr[hmr_id] = torch.from_numpy(intr_smooth)

        id_axes_smooth: Dict[str, List[Dict]] = {}
        for hmr_id in id_pose_trajectories:
            pose_seq_np = id_pose_trajectories[hmr_id]
            joints_seq = torch.stack(id_joints_trajectories[hmr_id], dim=0).to(DEVICE)
            axes_list = build_axes_seq_from_global_orient_with_gravity(pose_seq_np, joints_seq)
            id_axes_smooth[hmr_id] = axes_list

        # Recompute OABBs from smoothed rigid-body poses.
        id_collect = {}
        hmr_id_frame_idx = {k:0 for k in id_pose_trajectories.keys()}
        
        for f_id in sorted_fids:
            for hmr_id, data_dict in shot_human_bbox[f_id].items():

                idx = hmr_id_frame_idx[hmr_id]
                hmr_id_frame_idx[hmr_id] += 1
                
                pose_t = torch.tensor(id_pose_trajectories[hmr_id][idx], dtype=torch.float32)
                beta_t = id_mean_betas[hmr_id]
                trans_t = data_dict['adjusted_trans']
                vertices, _ = visualizer.get_vertices(pose_t, beta_t, trans_t, "neutral")
                scene_intr_resized = data_dict['scene_intr']
                adj_vertices = adjust_smpl_vertices(vertices, scene_intr_resized, data_dict['alpha'], data_dict['gamma'])

                axes = id_axes_smooth[hmr_id][idx]
                oabb = compute_oriented_bbox_debug(adj_vertices, axes, core_mask)

                if hmr_id not in id_collect:
                    id_collect[hmr_id] = {"axes": [], "centers": [], "sizes": [], "f_ids": []}
                id_collect[hmr_id]["f_ids"].append(f_id)
                id_collect[hmr_id]["axes"].append(axes)
                id_collect[hmr_id]["centers"].append(oabb["center"].numpy())
                id_collect[hmr_id]["sizes"].append(oabb["size"].numpy())

                data_dict["pose"] = pose_t.cpu()
                data_dict["poses_standing"] = pose_t.cpu()
                data_dict["scene_intr"] = id_smoothed_intr[hmr_id][idx].cpu()

                
        # Apply a final temporal pass to box center, size, and axes.
        for hmr_id, traj in id_collect.items():
            smoothed_axes = smooth_axes_seq_quat(traj["axes"], sigma=sigma)
            smoothed_centers = gaussian_filter1d(np.array(traj["centers"]), sigma=sigma, axis=0, mode=MODE_NEAREST)
            smoothed_sizes = gaussian_filter1d(np.array(traj["sizes"]), sigma=sigma, axis=0, mode=MODE_NEAREST)

            for i, f_id in enumerate(traj["f_ids"]):
                data_dict = shot_human_bbox[f_id][hmr_id]

                data_dict["oabb"] = build_smoothed_oabb(
                    torch.tensor(smoothed_centers[i]),
                    torch.tensor(smoothed_sizes[i]),
                    smoothed_axes[i]
                )

        smoothed_bbox_path = os.path.join(shot_dir, save_name)
        np.savez(smoothed_bbox_path, shot_human_bbox=shot_human_bbox)
        print(f"[shot_{shot_idx}] saved smoothed bboxes to {smoothed_bbox_path}")

    return True
