import os
from typing import *

import torch
import supervision as sv

import numpy as np

import cv2
from PIL import Image
from pycocotools import mask as masktool

# detectron2
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf

class DefaultPredictor_Lazy:
    """Create a simple end-to-end predictor with the given config that runs on single device for a
    single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            test dataset name in the config.


    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: omegaconf dict object.
        """
        # new LazyConfig
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
        if isinstance(test_dataset, (list, tuple)):
            test_dataset = test_dataset[0]

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

        mapper = instantiate(cfg.dataloader.test.mapper)
        self.aug = mapper.augmentations
        self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        

def arrange_boxes(boxes, mode='size', min_size=None):
    """ Helper to re-order boxes """
    # Left2right priority
    if mode == 'left2right':
        cx = (boxes[:,2] - boxes[:,0]) / 2 + boxes[:,0]
        boxes = boxes[np.argsort(cx)]
    # size priority
    elif mode == 'size':
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        area = w*h
        boxes = boxes[np.argsort(area)[::-1]]
    # confidence priority
    elif mode == 'conf':  
        conf = boxes[:,4]
        boxes = boxes[np.argsort(conf)[::-1]]
    # filter boxes by size
    if min_size is not None:
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        valid = np.stack([w, h]).max(axis=0) > min_size
        boxes = boxes[valid]

    return boxes


def detect_sam2(detector, video_predictor, imgfiles, thresh=0.5, min_size=None, save_vos=True):
    """ A simple pipeline for human detection, segmentation, and tracking. Mainly as input for TRAM.
    Detection: ViTDet.
    Segmentation & Tracking: SAM.
    """

    # First frame
    first_frame_path = imgfiles[0]
    img_cv2 = cv2.imread(first_frame_path)
    image = np.array(Image.open(first_frame_path).convert("RGB"))
    SAVE_TRACKING_RESULTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(first_frame_path)), "sam2"
    )

    ### --- Detection ---
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > thresh)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            confs = det_instances.scores[valid_idx].cpu().numpy()

            boxes = np.hstack([boxes, confs[:, None]])
            boxes = arrange_boxes(boxes, mode="size", min_size=min_size)

    ### --- SAM ---
    inference_state = video_predictor.init_state(
        video_path=os.path.dirname(first_frame_path)
    )
    ann_frame_idx = 0
    input_boxes = [box[:4] for box in boxes]
    confidences = [box[-1] for box in boxes]

    if len(input_boxes) == 0:
        return None

    ### --- Tracking ---
    for object_id, (conf, box) in enumerate(zip(confidences, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )

    # Propagate the video predictor to get the segmentation results for each frame
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    CONF_TO_OBJECTS = {i: obj for i, obj in enumerate(confidences, start=1)}
    # {i: obj for i, obj in enumerate(confidences, start=1)}

    if save_vos and not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(SAVE_TRACKING_RESULTS_DIR)

    masks_ = []
    xyxys_ = []
    anns_ = []
    obj_ids_ = []

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(imgfiles[frame_idx])

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )

        xyxys_.append(sv.mask_to_xyxy(masks))
        masks_.append(masks)
        anns_.append(frame_idx)
        obj_ids_.append(object_ids)

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[str(CONF_TO_OBJECTS[i]) for i in object_ids])
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
        if save_vos:
            cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

    # Merge tracking info
    tracks = {}
    dynamic_masks = []
    for f_index in range(len(imgfiles)):

        # if has obj
        if f_index in video_segments.keys():
            masks = masks_[f_index]
            xyxys = xyxys_[f_index]
            obj_ids = obj_ids_[f_index]

            for idx, subj_idx in enumerate(obj_ids):

                if abs(xyxys[idx][0] - xyxys[idx][2]) < 5 or abs(xyxys[idx][1] - xyxys[idx][3]) < 5:
                    subj = {
                        "id": subj_idx,
                        "frame": f_index,
                        "rle": masktool.encode(np.asfortranarray(masks[idx] > 0)),
                        "det": False,
                        "det_box": [np.hstack([xyxys[idx], CONF_TO_OBJECTS[subj_idx]])],
                        "seg_box": xyxys[idx],
                    }
                else:
                    subj = {
                        "id": subj_idx,
                        "frame": f_index,
                        "rle": masktool.encode(np.asfortranarray(masks[idx] > 0)),
                        "det": True,
                        "det_box": [np.hstack([xyxys[idx], CONF_TO_OBJECTS[subj_idx]])],
                        "seg_box": xyxys[idx],
                    }

                if subj_idx in tracks:
                    tracks[subj_idx][f_index] = subj
                else:
                    tracks[subj_idx] = {f_index: subj}

            dynamic_masks.append(masks.sum(axis=0))
        else:
            dynamic_masks.append(np.zeros_like(masks_[f_index][0]))

    tracks = np.array(tracks, dtype=object)
    # masks_ = np.array([masktool.encode(np.asfortranarray(mask > 0)) for mask in dynamic_masks], dtype=object)
    # boxes_ = np.array(xyxys_, dtype=object)

    return tracks
