"""MeV adapter for batched CameraHMR inference on tracked people.

The upstream CameraHMR checkout is intentionally left unmodified.  This module
adds the tracking-aware batching and JSON output needed by the MeV pipeline by
building on CameraHMR's public estimator and internal preprocessing helpers.
"""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mev_matplotlib_cache")

import cv2
import numpy as np
import smplx
import torch
import trimesh
from torchvision.transforms import Normalize

# run_camerahmr.py adds the upstream CameraHMR checkout to sys.path before this
# module is imported.
from core.constants import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, NUM_BETAS
from core.datasets.utils import (
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)
from core.utils.renderer_pyrd import Renderer
from mesh_estimator import HumanMeshEstimator, resize_image


class TrackedPersonDataset(torch.utils.data.Dataset):
    """Create CameraHMR crops for tracked people collected across many frames."""

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples
        self.image_size = IMAGE_SIZE
        self.mean = 255.0 * np.asarray(IMAGE_MEAN)
        self.std = 255.0 * np.asarray(IMAGE_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = sample["image"]
        center = sample["box_center"]
        scale = sample["box_scale"]
        bbox_size = expand_to_aspect_ratio(
            scale * 200.0, target_aspect_ratio=None
        ).max()

        patch, _ = generate_image_patch_cv2(
            image,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            self.image_size,
            self.image_size,
            False,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        image_tensor = convert_cvimg_to_tensor(patch[:, :, ::-1])
        for channel in range(min(image.shape[2], 3)):
            image_tensor[channel] = (
                image_tensor[channel] - self.mean[channel]
            ) / self.std[channel]

        return {
            "img": image_tensor,
            "personid": index,
            "imgname": str(sample["image_path"]),
            "box_center": center.astype(np.float32),
            "box_size": np.asarray(bbox_size, dtype=np.float32),
            "img_size": np.asarray(image.shape[:2], dtype=np.float32),
            "cam_int": sample["cam_int"].astype(np.float32),
        }


def _to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


class BatchHumanMeshEstimator(HumanMeshEstimator):
    """Run CameraHMR in batches using boxes from the tracking stage."""

    def __init__(self, model_type: str = "smpl"):
        # Do not call HumanMeshEstimator.__init__: the upstream initializer also
        # creates a ViTDet detector, while this adapter consumes tracking boxes.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = self.init_model()
        self.cam_model = self.init_cam_model().to(self.device)

        if model_type == "smpl":
            from core.constants import SMPL_MODEL_PATH

            self.body_model = smplx.SMPLLayer(
                model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS
            ).to(self.device)
        elif model_type == "smplx":
            from core.constants import NUM_BETAS_SMPLX, SMPLX_MODEL_DIR

            self.body_model = smplx.SMPLXLayer(
                model_path=SMPLX_MODEL_DIR, num_betas=NUM_BETAS_SMPLX
            ).to(self.device)
        else:
            raise ValueError("model_type must be 'smpl' or 'smplx'")

        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def get_cam_intrinsics(self, image: np.ndarray) -> np.ndarray:
        """Estimate intrinsics while keeping the FoV model on our CUDA device."""
        image_height, image_width = image.shape[:2]
        _, resized = resize_image(image, IMAGE_SIZE)
        resized = np.transpose(resized.astype("float32"), (2, 0, 1)) / 255.0
        tensor = self.normalize_img(torch.from_numpy(resized).float()).to(self.device)
        with torch.inference_mode():
            estimated_fov, _ = self.cam_model(tensor.unsqueeze(0))
        vertical_fov = estimated_fov[0, 1]
        focal_length = (
            image_height / (2.0 * torch.tan(vertical_fov / 2.0))
        ).item()
        return np.asarray(
            [
                [focal_length, 0.0, image_width / 2.0],
                [0.0, focal_length, image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def collect_samples(
        self, image_paths: list[str], tracking: dict[Any, dict[int, dict[str, Any]]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect model inputs and matching output metadata."""
        samples: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []

        for image_index, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                raise OSError(f"Cannot read CameraHMR input image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_name = Path(image_path).stem
            frame_id = int(frame_name) if frame_name.isdigit() else image_index
            cam_int = self.get_cam_intrinsics(image)

            for human_id, frames in tracking.items():
                record = frames.get(frame_id)
                if not record or not record.get("det"):
                    continue
                seg_box = record.get("seg_box")
                if seg_box is None:
                    continue
                box = np.asarray(seg_box, dtype=np.float32).reshape(4)
                box_scale = (box[2:4] - box[0:2]) / 200.0
                box_center = (box[2:4] + box[0:2]) / 2.0
                samples.append(
                    {
                        "image": image,
                        "image_path": image_path,
                        "box_center": box_center,
                        "box_scale": box_scale,
                        "cam_int": cam_int,
                    }
                )
                metadata.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "image_index": image_index,
                        "image_extension": Path(image_path).suffix,
                        "human_id": human_id,
                        "box_center": box_center.reshape(1, 2),
                        "box_scale": box_scale.reshape(1, 2),
                        "cam_int": cam_int,
                    }
                )
        return samples, metadata

    def infer_batches(
        self, samples: list[dict[str, Any]], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        if not samples:
            return []
        worker_count = max(0, min(8, (os.cpu_count() or 1) // 12))
        loader = torch.utils.data.DataLoader(
            TrackedPersonDataset(samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=self.device.type == "cuda",
        )

        results: list[dict[str, Any]] = []
        for batch in loader:
            device_batch = self._recursive_to(batch, self.device)
            with torch.inference_mode():
                smpl_params, out_cam, focal_length = self.model(device_batch)
            for index in range(out_cam.shape[0]):
                results.append(
                    {
                        "smpl_params": {
                            key: value[index : index + 1].detach().cpu()
                            for key, value in smpl_params.items()
                        },
                        "out_cam": out_cam[index].detach().cpu(),
                        "focal_length": focal_length[index].detach().cpu(),
                        "img_size": batch["img_size"][index].detach().cpu(),
                        "cam_int": batch["cam_int"][index].detach().cpu(),
                        "box_size": batch["box_size"][index].detach().cpu(),
                        "box_center": batch["box_center"][index].detach().cpu(),
                    }
                )
        return results

    def save_results(
        self,
        image_paths: list[str],
        results: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        output_folder: str,
        save_vis: bool,
    ) -> None:
        output = Path(output_folder)
        json_folder = output / "camerahmr_json"
        output.mkdir(parents=True, exist_ok=True)
        json_folder.mkdir(parents=True, exist_ok=True)

        # Write one JSON file per input frame, including frames with no people.
        frame_results: dict[str, dict[str, Any]] = {
            Path(path).stem: {} for path in image_paths
        }
        for result, meta in zip(results, metadata):
            frame_name = meta["frame_name"]
            human_id = meta["human_id"]
            frame_results[frame_name][human_id] = {
                "smpl_params": result["smpl_params"],
                "out_cam": result["out_cam"],
                "box_center": meta["box_center"],
                "box_size": meta["box_scale"],
                "cam_int": meta["cam_int"],
                "img_size": result["img_size"],
            }
            if save_vis:
                self._save_visualization(result, meta, output)

        for frame_name, payload in frame_results.items():
            path = json_folder / f"{frame_name}.json"
            path.write_text(
                json.dumps(_to_json_value(payload)), encoding="utf-8"
            )

    def _save_visualization(
        self, result: dict[str, Any], meta: dict[str, Any], output: Path
    ) -> None:
        smpl_params = {
            key: value.to(self.device) for key, value in result["smpl_params"].items()
        }
        out_cam = result["out_cam"].unsqueeze(0).to(self.device)
        batch = {
            "img_size": result["img_size"].unsqueeze(0).to(self.device),
            "cam_int": result["cam_int"].unsqueeze(0).to(self.device),
            "box_size": result["box_size"].unsqueeze(0).to(self.device),
            "box_center": result["box_center"].unsqueeze(0).to(self.device),
        }
        vertices, _, camera_translation = self.get_output_mesh(
            smpl_params, out_cam, batch
        )
        frame_name = meta["frame_name"]
        human_id = meta["human_id"]
        image_index = meta["image_index"]
        mesh_path = output / f"{frame_name}_{image_index:06d}_{human_id}.obj"
        mesh = trimesh.Trimesh(
            vertices[0].detach().cpu().numpy(), self.body_model.faces, process=False
        )
        mesh.export(mesh_path)

        image = cv2.imread(meta["image_path"])
        if image is None:
            raise OSError(f"Cannot read visualization image: {meta['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        focal_length = float(result["focal_length"].item())
        renderer = Renderer(
            focal_length=focal_length,
            img_w=image_width,
            img_h=image_height,
            faces=self.body_model.faces,
            same_mesh_color=True,
        )
        rendered = renderer.render_front_view(
            (vertices + camera_translation.unsqueeze(1)).detach().cpu().numpy(),
            bg_img_rgb=image.copy(),
        )
        renderer.delete()
        overlay_path = output / (
            f"{frame_name}_{image_index:06d}_{human_id}{meta['image_extension']}"
        )
        rendered_bgr = cv2.cvtColor(
            rendered.astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        if not cv2.imwrite(str(overlay_path), rendered_bgr):
            raise OSError(f"Failed to write CameraHMR overlay: {overlay_path}")

    @classmethod
    def _recursive_to(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: cls._recursive_to(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._recursive_to(item, device) for item in value]
        return value

    def run_on_images(
        self,
        image_folder: str,
        output_folder: str,
        save_vis: bool,
        tracking: dict[Any, dict[int, dict[str, Any]]],
    ) -> None:
        extensions = (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        )
        image_paths = sorted(
            path
            for extension in extensions
            for path in glob(os.path.join(image_folder, extension))
        )
        samples, metadata = self.collect_samples(image_paths, tracking)
        print(f"Collected {len(samples)} tracked people from {len(image_paths)} frames")
        results = self.infer_batches(samples)
        self.save_results(
            image_paths, results, metadata, output_folder, save_vis=save_vis
        )
        print(f"Saved CameraHMR JSON for {len(image_paths)} frames")
