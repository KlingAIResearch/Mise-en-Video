from typing import Tuple

import cv2
import time
from functools import partial

import numpy as np

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from pycocotools import mask as masktool


def numpy_to_tensor(obj, device=None, dtype=None):
    """Recursively convert NumPy values to tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype) if device is not None or dtype is not None else obj
    if isinstance(obj, np.ndarray):
        result = torch.from_numpy(obj.astype(bool) if obj.dtype == np.bool_ else obj)
        return result.to(device=device, dtype=dtype) if device is not None or dtype is not None else result
    if isinstance(obj, dict):
        return {key: numpy_to_tensor(value, device, dtype) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_tensor(value, device, dtype) for value in obj)
    return obj


def ransac_fit_offset_parallel(X, Y, alpha, max_iters=1000, threshold=0.001, confidence=0.99, 
                                min_inlier_ratio=0.1, device='cuda', max_memory_batch=None):
    """Fit the offset in ``Y = alpha * X + offset`` with batched RANSAC."""
    if isinstance(X, torch.Tensor):
        X = X.to(device, dtype=torch.float32)
    else:
        X = torch.tensor(X, dtype=torch.float32, device=device)
    
    if isinstance(Y, torch.Tensor):
        Y = Y.to(device, dtype=torch.float32)
    else:
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
        
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
    threshold = torch.tensor(threshold, dtype=torch.float32, device=device)
    
    n_points = X.numel()
    if n_points == 0:
        return 0.0, {"inliers": 0, "time": 0.0}

    start_time = time.time()

    # Offset hypotheses can be sampled directly from this residual vector.
    diffs = Y - alpha * X
    
    # Remove extreme residuals without discarding a small valid population.
    std_diff = torch.std(diffs)
    if std_diff > 1e-8:
        mean_diff = torch.mean(diffs)
        z_scores = torch.abs((diffs - mean_diff) / std_diff)
        valid_mask = z_scores < 4.0
        
        if valid_mask.sum() > max(3, int(n_points * min_inlier_ratio)):
            diffs_filtered = diffs[valid_mask]
        else:
            diffs_filtered = diffs
    else:
        diffs_filtered = diffs
    
    n_filtered = diffs_filtered.numel()
    
    estimated_ratio = max(min_inlier_ratio, 0.3)
    try:
        needed_iters = int(np.log(1 - confidence) / np.log(1 - estimated_ratio**5)) + 1
    except:
        needed_iters = max_iters
        
    num_hypotheses = min(max_iters, max(needed_iters, 50))
    if n_filtered < 6:
        num_hypotheses = max_iters

    # Generate all five-point median hypotheses in one GPU operation.
    sample_indices = torch.randint(0, n_filtered, (num_hypotheses, 5), device=device)
    sampled_diffs = diffs_filtered[sample_indices]
    sorted_samples, _ = torch.sort(sampled_diffs, dim=1)
    candidates_b = sorted_samples[:, 2] 
    
    best_inliers_count = -1
    total_elements = n_points * num_hypotheses
    
    if max_memory_batch is None and total_elements < 5e7:
        residuals = torch.abs(diffs.unsqueeze(0) - candidates_b.unsqueeze(1))
        inlier_counts = (residuals < threshold).sum(dim=1)
        best_inliers_count, best_idx = torch.max(inlier_counts, dim=0)
        best_offset = candidates_b[best_idx]
        
    else:
        # Bound the temporary residual matrix on memory-constrained devices.
        chunk_size = 1000
        best_offset = 0.0
        
        for i in range(0, num_hypotheses, chunk_size):
            end = min(i + chunk_size, num_hypotheses)
            curr_candidates = candidates_b[i:end]
            residuals = torch.abs(diffs.unsqueeze(0) - curr_candidates.unsqueeze(1))
            inlier_counts = (residuals < threshold).sum(dim=1)
            
            curr_max, curr_argmax = torch.max(inlier_counts, dim=0)
            
            if curr_max > best_inliers_count:
                best_inliers_count = curr_max
                best_offset = curr_candidates[curr_argmax]

    end_time = time.time()

    stats = {
        "iterations": num_hypotheses,
        "inliers": best_inliers_count.item(),
        "inlier_ratio": best_inliers_count.item() / n_points,
        "time": end_time - start_time,
        "device": device
    }

    return best_offset.item(), stats


def recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor = None,
    focal: torch.Tensor = None,
    downsample_size: Tuple[int, int] = (64, 64),
):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height**2 + width**2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(
        width, height, dtype=points.dtype, device=points.device
    )  # (H, W, 2)

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(
            uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = (
            points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        )
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(
                uv_lr_i_np, points_lr_i_np
            )
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(
        optim_shift, device=points.device, dtype=points.dtype
    ).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(
            optim_focal, device=points.device, dtype=points.dtype
        ).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def get_scene_extr(scene_info, f_id):
    R, t = scene_info["camera_poses"][f_id][:3, :3], scene_info["camera_poses"][f_id][:3, 3]
    return R, t


def get_scene_intr(scene_info, f_id=None):
    scene_info = numpy_to_tensor(scene_info)
    points = scene_info["local_points"]  # Shape: (B, N, H, W, 3)
    masks = torch.sigmoid(scene_info["conf"][..., 0]) > 0.1  # Shape: (B, N, H, W)
    focal, shift = recover_focal_shift(points, masks, downsample_size=(64, 64))
    original_height, original_width = points.shape[-3:-1]
    aspect_ratio = original_width / original_height
    fx = (focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * original_width) # N
    fy = (focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * original_height) # N
    cx = original_width // 2 
    cy = original_height // 2 

    N = focal.shape[0]

    K = np.zeros((N, 3, 3))
    K[:, 0, 0] = fx.numpy()
    K[:, 1, 1] = fy.numpy()
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1

    if f_id is None:
        return K
    else:
        return K[f_id]


def get_scene_camera(cam_file):
    with np.load(cam_file) as payload:
        return {
            "points": payload["points"][0],
            "local_points": payload["local_points"][0],
            "conf": payload["conf"][0],
            "camera_poses": payload["camera_poses"][0],
        }


def convert_to_full_img_cam(pred_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    tz = 2. * focal_length / (bbox_height * s)
    cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
    trans = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return trans


def adjust_smpl_vertices(V_smpl, intr, a, b):
    """Align camera-space SMPL vertices with ``z_scene = a * z_smpl + b``."""
    if not isinstance(V_smpl, torch.Tensor):
        V_smpl = torch.tensor(V_smpl, dtype=torch.float32)
    device = V_smpl.device
    dtype = V_smpl.dtype

    if not isinstance(intr, torch.Tensor):
        intr = torch.tensor(intr, dtype=dtype, device=device)
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]

    a = torch.tensor(a, dtype=dtype, device=device)
    b = torch.tensor(b, dtype=dtype, device=device)

    z_smpl = V_smpl[:, 2].clone()
    valid_mask = z_smpl > 1e-6
    z_smpl_valid = z_smpl[valid_mask]
    V_valid = V_smpl[valid_mask]

    if not torch.any(valid_mask):
        return V_smpl.clone()

    # Preserve image coordinates while replacing each vertex depth.
    u = (V_valid[:, 0] * fx) / z_smpl_valid + cx
    v = (V_valid[:, 1] * fy) / z_smpl_valid + cy

    z_scene = torch.clamp(a * z_smpl_valid + b, min=1e-6)

    x_scene = (u - cx) * z_scene / fx
    y_scene = (v - cy) * z_scene / fy
    V_aligned_valid = torch.stack([x_scene, y_scene, z_scene], dim=1)

    V_smpl_aligned = V_smpl.clone()
    V_smpl_aligned[valid_mask] = V_aligned_valid

    return V_smpl_aligned


def write_oabb_to_ply(oabb, c2w=None, save_path="oabb_bbox.ply", color=(255, 0, 0)):
    """Write an OABB as an ASCII PLY containing eight vertices and 12 edges."""
    bbox_vertices = oabb["vertices"].cpu().numpy()
    num_vertices = len(bbox_vertices)

    if c2w is not None:
        R_cw_C, t_cw_C = c2w
        
        if isinstance(R_cw_C, np.ndarray):
            R = R_cw_C.copy()
        else:
            R = R_cw_C.cpu().numpy()
        if isinstance(t_cw_C, np.ndarray):
            t = t_cw_C.copy()
        else:
            t = t_cw_C.cpu().numpy()
        
        assert R.shape == (3, 3), f"Expected rotation shape (3, 3), got {R.shape}"
        assert t.shape in ((3,), (3, 1)), f"Expected translation shape (3,) or (3, 1), got {t.shape}"
        t = t.reshape(3,)
        bbox_vertices = (R @ bbox_vertices.T).T + t
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]
    num_edges = len(edges)
    
    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {num_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        
        for v in bbox_vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        
        for (v1, v2) in edges:
            f.write(f"{v1} {v2}\n")


def compute_oriented_bbox(vertices, axes):
    """Compute a body-oriented box from SMPL vertices and orthonormal axes."""
    device = vertices.device
    centroid = torch.mean(vertices, dim=0)
    vertices_centered = vertices - centroid
    
    R = torch.stack([axes["right"], axes["up"], axes["forward"]], dim=1)
    vertices_body = torch.matmul(vertices_centered, R)
    min_vals = torch.min(vertices_body, dim=0)[0].clone()
    max_vals = torch.max(vertices_body, dim=0)[0].clone()

    height = max_vals[1] - min_vals[1]

    # Limit width so arm poses do not dominate the body box.
    max_allowed_width = height * 0.5        
    current_width = max_vals[0] - min_vals[0]
    if current_width > max_allowed_width:
        x_center = (max_vals[0] + min_vals[0]) / 2.0
        min_vals[0] = x_center - (max_allowed_width / 2.0)
        max_vals[0] = x_center + (max_allowed_width / 2.0)

    # Estimate head width from the top eighth of the body.
    head_threshold_y = max_vals[1] - (height / 8.0)
    head_vertices = vertices_body[vertices_body[:, 1] > head_threshold_y]
    
    if len(head_vertices) > 0:
        head_width = torch.max(head_vertices[:, 0]) - torch.min(head_vertices[:, 0])
    else:
        head_width = height / 10.0

    # Use head width as a robust upper bound for box depth.
    max_allowed_depth = head_width
    current_depth = max_vals[2] - min_vals[2]
    
    if current_depth > max_allowed_depth:
        z_center = (max_vals[2] + min_vals[2]) / 2.0
        min_vals[2] = z_center - (max_allowed_depth / 2.0)
        max_vals[2] = z_center + (max_allowed_depth / 2.0)
    size = max_vals - min_vals
    
    body_vertices = torch.tensor([
        [min_vals[0], min_vals[1], min_vals[2]],
        [max_vals[0], min_vals[1], min_vals[2]],
        [max_vals[0], max_vals[1], min_vals[2]],
        [min_vals[0], max_vals[1], min_vals[2]],
        [min_vals[0], min_vals[1], max_vals[2]],
        [max_vals[0], min_vals[1], max_vals[2]],
        [max_vals[0], max_vals[1], max_vals[2]],
        [min_vals[0], max_vals[1], max_vals[2]],
    ], device=device)
    
    world_vertices = torch.matmul(body_vertices, R.T) + centroid
    oabb = {
        "center": centroid.detach().to('cpu'),
        "size": size.detach().to('cpu'),
        "vertices": world_vertices.detach().to('cpu'),
        "forward": axes["forward"].detach().to('cpu'),
        "right": axes["right"].detach().to('cpu'),
        "up": axes["up"].detach().to('cpu')
    }
    return oabb


def get_body_forward_direction(pose, smpl_model=None, method="upper_joints"):
    """Return orthonormal body axes using joints, root pose, or vertex PCA."""
    device = pose.device
    
    if method == "upper_joints":
        assert smpl_model is not None, "method='upper_joints' requires smpl_model"
        output = smpl_model(
            betas=torch.zeros(1, 10, device=device),
            global_orient=pose[:3].unsqueeze(0),
            body_pose=pose[3:].unsqueeze(0)
        )
        joints = output.joints[0]

        # SMPL joints 16/17 are the shoulders; 3/12 span the torso.
        right = - joints[17] + joints[16]
        right = right / torch.norm(right)
        temp_up = - joints[12] + joints[3]
        temp_up = temp_up / torch.norm(temp_up)
        forward = torch.cross(temp_up, right)
        forward = forward / torch.norm(forward)
        up = torch.cross(forward, right)
        up = up / torch.norm(up)

    elif method == "pose":
        assert smpl_model is not None, "method='pose' requires smpl_model"
        global_orient = pose[:3].cpu().numpy()
        rot = R.from_rotvec(global_orient)
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32, device=device)
        
        model_forward = torch.tensor([0.0, 0.0, 1.0], device=device)
        forward = torch.matmul(rot_mat, model_forward)
        forward = forward / torch.norm(forward)
        
        model_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        model_right = torch.tensor([1.0, 0.0, 0.0], device=device)
        up = torch.matmul(rot_mat, model_up)
        up = up / torch.norm(up)
        right = torch.matmul(rot_mat, model_right)
        right = right / torch.norm(right)
        
    elif method == "pca":
        assert smpl_model is not None, "method='pca' requires smpl_model"
        vertices = smpl_model(
            betas=torch.zeros(1, 10, device=device),
            global_orient=pose[:3].unsqueeze(0),
            body_pose=pose[3:].unsqueeze(0)
        ).vertices[0]
        centroid = torch.mean(vertices, dim=0)
        vertices_centered = vertices - centroid
        
        cov = torch.matmul(vertices_centered.T, vertices_centered) / (vertices.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eig(cov)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        
        sorted_idx = torch.argsort(eigvals, descending=True)
        up = eigvecs[:, sorted_idx[0]]
        right = eigvecs[:, sorted_idx[1]]
        
        forward = torch.cross(right, up)
        forward = forward / torch.norm(forward)
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        
    else:
        raise ValueError("method must be 'upper_joints', 'pose' or 'pca'")
    
    axes = {
        "right": right / torch.norm(right),
        "up": up / torch.norm(up),
        "forward": forward / torch.norm(forward)
    }
    return axes["forward"], axes


def get_scene_depth_cuda(scene_info, f_id=0):
    R_cw_C, t_cw_C = get_scene_extr(scene_info, f_id=f_id)
    depth_map = torch.tensor(scene_info['local_points'][f_id][:, :, -1])

    return depth_map, scene_info['points'][f_id], R_cw_C, t_cw_C

def get_scene_part_mask_cuda(scene_info, track_1, f_id=0, h_id=1):

    scene_1 = scene_info['points'][f_id].reshape(-1,3)
    original_height, original_width = scene_info['points'][f_id].shape[-3:-1]
    img_height, img_width = (original_height, original_width)
    
    human_info = track_1[h_id][f_id]
    human_mask = masktool.decode(human_info['rle'])
    human_mask = cv2.resize(human_mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    return human_mask
