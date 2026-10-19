"""SMPL geometry and point-cloud helpers required by bbox alignment."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import trimesh

from .smpl_renderer import Renderer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_MODELS = REPO_ROOT / "3rdparty" / "CameraHMR" / "data" / "models"


def resolve_body_model_path(model_folder, model_type):
    """Resolve an SMPL-family model file from this repository or an override."""
    configured = model_folder or os.getenv("MEV_BODY_MODELS") or DEFAULT_BODY_MODELS
    base = Path(configured).expanduser().resolve()
    model_type = model_type.lower()
    if model_type == "smpl":
        marker = "SMPL_NEUTRAL.pkl"
        candidates = (base, base / "SMPL", base / "smpl")
    elif model_type == "smplx":
        marker = "SMPLX_NEUTRAL.npz"
        candidates = (base, base / "SMPLX", base / "smplx", base / "smplx" / "models")
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    for candidate in candidates:
        model_file = candidate / marker
        if model_file.is_file():
            return model_file
    searched = "\n  - ".join(str(candidate / marker) for candidate in candidates)
    raise FileNotFoundError(
        f"Cannot find {marker}. Searched:\n  - {searched}\n"
        "Set MEV_BODY_MODELS to the directory containing SMPL/ or SMPLX/."
    )


class SMPLVisualizer:
    """Load a body model and expose only operations needed by bbox alignment."""

    def __init__(self, model_folder=None, model_type="smpl", device="cpu", dtype=torch.float32):
        import smplx

        self.model_type = model_type.lower()
        self.device = torch.device(device)
        self.dtype = dtype
        model_path = resolve_body_model_path(model_folder, self.model_type)
        options = {
            "model_type": self.model_type,
            "gender": "neutral",
            "num_betas": 10 if self.model_type == "smpl" else 11,
        }
        if self.model_type == "smpl":
            options["ext"] = "pkl"
        else:
            options.update(ext="npz", flat_hand_mean=True, use_pca=False)
        self.models = {
            "neutral": smplx.create(str(model_path), **options).to(
                self.device, dtype=self.dtype
            )
        }

    def get_vertices(self, poses, betas, trans, gender="neutral"):
        if gender not in self.models:
            raise ValueError(f"Body model for gender {gender!r} is not loaded")
        poses = torch.as_tensor(poses, device=self.device, dtype=self.dtype)
        betas = torch.as_tensor(betas, device=self.device, dtype=self.dtype)
        trans = torch.as_tensor(trans, device=self.device, dtype=self.dtype)
        if poses.ndim == 1:
            poses = poses.unsqueeze(0)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        if trans.ndim == 1:
            trans = trans.unsqueeze(0)

        kwargs = {
            "betas": betas,
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3:72] if self.model_type == "smpl" else poses[:, 3:66],
            "transl": trans,
        }
        if self.model_type == "smplx":
            kwargs.update(
                jaw_pose=poses[:, 66:69],
                leye_pose=poses[:, 69:72],
                reye_pose=poses[:, 72:75],
                left_hand_pose=poses[:, 75:120],
                right_hand_pose=poses[:, 120:165],
            )
        output = self.models[gender](**kwargs)
        return output.vertices[0], output.joints[0]

    def get_smpl_depth(self, vertices, intrinsics, extrinsics=None, width=None, height=None):
        """Rasterize SMPL depth and foreground mask in source image space."""
        vertices = torch.as_tensor(vertices, device=self.device, dtype=self.dtype)
        intrinsics = torch.as_tensor(intrinsics, device=self.device, dtype=self.dtype)
        if extrinsics is None:
            extrinsics = torch.eye(4, device=self.device, dtype=self.dtype)
        else:
            extrinsics = torch.as_tensor(
                extrinsics, device=self.device, dtype=self.dtype
            )
        width = int(width or intrinsics[0, 2] * 2)
        height = int(height or intrinsics[1, 2] * 2)
        renderer = Renderer(
            width,
            height,
            device=self.device,
            faces=self.models["neutral"].faces,
            K=intrinsics,
        )
        renderer.create_camera(R=extrinsics[:3, :3], T=extrinsics[:3, 3])
        return renderer.render_depth_only(vertices)


def _depth_points(depth, intrinsic, c2w=None, mask=None):
    depth = torch.as_tensor(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got {tuple(depth.shape)}")
    intrinsic = torch.as_tensor(intrinsic, device=depth.device, dtype=depth.dtype)
    height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    points = torch.stack(
        (
            (x - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        dim=-1,
    ).reshape(-1, 3)
    if c2w is not None:
        c2w = torch.as_tensor(c2w, device=depth.device, dtype=depth.dtype)
        homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
        points = (c2w @ homogeneous.T).T[:, :3]

    valid = torch.isfinite(points).all(dim=-1) & (points[:, 2] > 0.01)
    if mask is not None:
        mask = torch.as_tensor(mask, device=depth.device)
        if mask.shape[:2] != (height, width):
            mask = F.interpolate(
                mask.float()[None, None], size=(height, width), mode="nearest"
            )[0, 0]
        valid &= mask.reshape(-1).bool()
    return points, valid


def load_depth_to_point_cloud(depth, intrinsic, c2w=None, mask=None):
    """Return valid camera/world-space points on the depth tensor device."""
    points, valid = _depth_points(depth, intrinsic, c2w=c2w, mask=mask)
    return points[valid]


def visualize_depth_as_point_cloud(
    depth, intrinsic, save_path, rgb=None, c2w=None, mask=None, dump=False
):
    """Build a point cloud and optionally export it for save-vis debugging."""
    points, valid = _depth_points(depth, intrinsic, c2w=c2w, mask=mask)
    valid_points = points[valid].detach().cpu().numpy()

    colors = None
    if rgb is not None:
        colors = torch.as_tensor(rgb, device=points.device)
        if colors.ndim == 3 and colors.shape[0] == 3:
            colors = colors.permute(1, 2, 0)
        colors = colors.reshape(-1, 3)[valid]
        if colors.numel() and colors.max() <= 1.01:
            colors = colors * 255
        colors = colors.clamp(0, 255).byte().cpu().numpy()

    cloud = trimesh.points.PointCloud(valid_points, colors=colors)
    if dump:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cloud.export(path)
    return cloud
