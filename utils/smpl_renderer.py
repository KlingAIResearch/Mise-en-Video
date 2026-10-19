"""Minimal PyTorch3D depth renderer used by SMPL alignment."""

import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


def overlay_depth_onto_background(
    depth_map, mask, bbox, background_shape, background_value=0.0
):
    """Place a cropped depth map and mask on full-resolution tensors."""
    device = depth_map.device
    dtype = depth_map.dtype
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=device, dtype=torch.float32)

    h_bg, w_bg = background_shape
    out_depth = torch.full((h_bg, w_bg), background_value, device=device, dtype=dtype)
    out_mask = torch.zeros((h_bg, w_bg), device=device, dtype=mask.dtype)

    if bbox.dim() > 1:
        bbox = bbox[0]

    bbox = bbox.to(torch.int32)
    left = torch.clamp(bbox[0], 0, w_bg)
    top = torch.clamp(bbox[1], 0, h_bg)
    right = torch.clamp(bbox[2], left + 1, w_bg)
    bottom = torch.clamp(bbox[3], top + 1, h_bg)
    expected_height = int((bottom - top).item())
    expected_width = int((right - left).item())

    if (depth_map.shape[0] != expected_height or depth_map.shape[1] != expected_width):
        raise ValueError(
            f"Depth map shape {depth_map.shape[:2]} does not match "
            f"crop {(expected_height, expected_width)}"
        )

    if (mask.shape[0] != expected_height or mask.shape[1] != expected_width):
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match "
            f"crop {(expected_height, expected_width)}"
        )
    
    left_int = left.item()
    top_int = top.item()
    right_int = right.item()
    bottom_int = bottom.item()
    out_depth[top_int:bottom_int, left_int:right_int] = depth_map
    out_mask[top_int:bottom_int, left_int:right_int] = mask

    return out_depth, out_mask


def update_intrinsics_from_bbox(intrinsics, bbox):
    """Adjust intrinsics and image sizes for cropped rendering."""
    device, dtype = intrinsics.device, intrinsics.dtype

    K = torch.zeros((intrinsics.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = intrinsics.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(points, intrinsics, rotation=None, translation=None):
    """Project camera-space points and mark invalid results with -999."""
    if rotation is not None:
        points = torch.matmul(rotation, points.transpose(1, 2)).transpose(1, 2)
    if translation is not None:
        points = points + translation.transpose(1, 2)

    z_coords = points[..., 2:]
    valid_z_mask = (torch.abs(z_coords) >= 1e-5) & torch.isfinite(z_coords)
    z_coords_safe = torch.where(valid_z_mask, z_coords, torch.sign(z_coords) * 1e-5 + 1e-5)

    x3d_safe = points.clone()
    x3d_safe[..., 2:] = z_coords_safe
    x2d = torch.div(x3d_safe, x3d_safe[..., 2:])
    x2d = torch.matmul(intrinsics, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    
    final_valid_mask = valid_z_mask.squeeze(-1) & torch.isfinite(x2d).all(dim=-1, keepdim=True).squeeze(-1)
    x2d_masked = torch.where(final_valid_mask.unsqueeze(-1), x2d, torch.tensor(-999.0, device=x2d.device, dtype=x2d.dtype))
    
    return x2d_masked


def compute_bbox_from_points(points, img_w, img_h, scale_factor=1.2):
    """Compute a padded image-space box, falling back to the full image."""
    def full_image_bbox():
        return torch.tensor(
            [[0, 0, img_w, img_h]], dtype=torch.float32, device=points.device
        )

    if points.numel() == 0:
        return full_image_bbox()
    
    if len(points.shape) == 3:
        points_flat = points.reshape(-1, points.shape[-1])
    elif len(points.shape) == 2:
        points_flat = points
    else:
        return full_image_bbox()
    
    valid_mask = torch.isfinite(points_flat).all(dim=-1)
    if not valid_mask.any():
        return full_image_bbox()
    
    valid_points = points_flat[valid_mask]
    img_w_tensor = torch.tensor(img_w, dtype=valid_points.dtype, device=valid_points.device)
    img_h_tensor = torch.tensor(img_h, dtype=valid_points.dtype, device=valid_points.device)
    
    left = torch.clamp(valid_points[:, 0].min(), min=0, max=img_w_tensor)
    right = torch.clamp(valid_points[:, 0].max(), min=0, max=img_w_tensor)
    top = torch.clamp(valid_points[:, 1].min(), min=0, max=img_h_tensor)
    bottom = torch.clamp(valid_points[:, 1].max(), min=0, max=img_h_tensor)
    
    if left >= right:
        left = torch.clamp(left - 10, min=0, max=img_w_tensor)
        right = torch.clamp(left + 20, min=1, max=img_w_tensor)
    if top >= bottom:
        top = torch.clamp(top - 10, min=0, max=img_h_tensor)
        bottom = torch.clamp(top + 20, min=1, max=img_h_tensor)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    img_w_tensor = torch.tensor(img_w, dtype=cx.dtype, device=cx.device)
    img_h_tensor = torch.tensor(img_h, dtype=cy.dtype, device=cy.device)
    scale_tensor = torch.tensor(scale_factor, dtype=cx.dtype, device=cx.device)

    new_left = torch.clamp(cx - width / 2 * scale_tensor, min=0, max=img_w_tensor - 1)
    new_right = torch.clamp(cx + width / 2 * scale_tensor, min=1, max=img_w_tensor)
    new_top = torch.clamp(cy - height / 2 * scale_tensor, min=0, max=img_h_tensor - 1)
    new_bottom = torch.clamp(cy + height / 2 * scale_tensor, min=1, max=img_h_tensor)
    
    if new_left >= new_right:
        new_left = torch.tensor(0, dtype=new_left.dtype, device=new_left.device)
        new_right = torch.tensor(max(1, min(img_w, 100)), dtype=new_right.dtype, device=new_right.device)
    if new_top >= new_bottom:
        new_top = torch.tensor(0, dtype=new_top.dtype, device=new_top.device)
        new_bottom = torch.tensor(max(1, min(img_h, 100)), dtype=new_bottom.dtype, device=new_bottom.device)

    bbox = torch.stack((new_left.detach(), new_top.detach(), new_right.detach(), new_bottom.detach())).int().float()
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)
    
    return bbox



class Renderer:
    """Crop-aware SMPL depth rasterizer."""

    def __init__(self, width, height, device="cuda", faces=None, K=None, bin_size=0):
        if K is None:
            raise ValueError("K is required")
        self.width = int(width)
        self.height = int(height)
        self.device = torch.device(device)
        self.bin_size = bin_size
        if faces is None:
            raise ValueError("SMPL faces are required")
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces.astype(np.int64))
        self.faces = faces.unsqueeze(0).to(self.device)
        self.K = K.float().reshape(1, 3, 3).to(self.device)
        self.R = torch.eye(3, device=self.device).unsqueeze(0)
        self.T = torch.zeros((1, 3), device=self.device)
        self.reset_bbox()

    def _create_camera(self):
        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False,
        )

    def _create_rasterizer(self):
        return MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=self.image_sizes[0],
                blur_radius=1e-5,
                bin_size=self.bin_size,
            )
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().reshape(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().reshape(1, 3).to(self.device)
        self.cameras = self._create_camera()
        return self.cameras

    def _set_bbox(self, bbox):
        self.bboxes = bbox.float().to(self.device)
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self._create_camera()
        self.rasterizer = self._create_rasterizer()

    def update_bbox(self, vertices, scale=1.2):
        projected = perspective_projection(
            vertices.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1)
        )
        self._set_bbox(
            compute_bbox_from_points(projected, self.width, self.height, scale)
        )

    def reset_bbox(self):
        self._set_bbox(
            torch.tensor(
                [[0, 0, self.width, self.height]],
                dtype=self.K.dtype,
                device=self.device,
            )
        )

    def render_depth_only(self, vertices, vertex_interval=50):
        self.update_bbox(vertices[::vertex_interval])
        vertices = vertices.unsqueeze(0)
        texture = TexturesVertex(
            verts_features=torch.ones(
                (1, vertices.shape[1], 3),
                device=vertices.device,
                dtype=vertices.dtype,
            )
        )
        mesh = Meshes(verts=vertices, faces=self.faces, textures=texture)
        fragments = self.rasterizer(mesh, cameras=self.cameras)
        depth = torch.flip(fragments.zbuf[0, ..., 0], [0, 1])
        mask = torch.flip(fragments.pix_to_face[0, ..., 0] >= 0, [0, 1])
        depth = torch.where(depth < 0, torch.zeros_like(depth), depth)
        depth, mask = overlay_depth_onto_background(
            depth, mask, self.bboxes, (self.height, self.width)
        )
        self.reset_bbox()
        return depth, mask
