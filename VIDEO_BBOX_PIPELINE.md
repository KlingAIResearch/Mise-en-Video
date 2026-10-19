# Video to Human 3D BBox Pipeline

This repository converts multi-shot video into scene geometry, tracked subjects,
SMPL humans, smoothed 3D bounding boxes, rendered MP4s, and Gemini captions.

## Input

Place one complete video and its pre-split shots in `example/video/`:

```text
example/video/
├── example.mp4
├── example-shot0.mp4
└── example-shot1.mp4
```

Shot names must end with `-shotN.mp4`, and IDs must be contiguous from zero.

## Environment

All stages run in the unified `mev` conda environment. The entry scripts do not
enforce an environment name; dependency import errors are reported directly.

For a fresh machine, install the Python environment first:

```bash
conda create -n mev python=3.10 -y
conda activate mev
./install.sh
```

Then follow the Pi3, CameraHMR, and SAM2 upstream links in [`README.md`](README.md)
to prepare their source trees and checkpoints. Tested source revisions are recorded
in `3rdparty/LOCK.json`; model weights are local assets and are not committed.

## Run the pipeline

For bbox videos with consistent cross-shot identities, set the Gemini API key and
run the pipeline. The `match` stage runs before bbox alignment and rendering, and
`bbox` is the terminal pipeline stage:

```bash
conda activate mev
export GEMINI_API_KEY='YOUR_KEY'
./tools/run_pipeline.py
```

To run geometry and bbox without Gemini, stop at CameraHMR and invoke bbox directly.
This uses fallback per-shot identities for subjects that do not already have a
`matched_id.json` mapping:

```bash
./tools/run_pipeline.py --to-stage camerahmr
./tools/run_bbox_video.py
```

The identity-match request goes directly to the Google Gemini API. Text captions
are optional and run separately after the bbox pipeline.

Run only a range of stages:

```bash
./tools/run_pipeline.py --from-stage tracking --to-stage bbox
```

Available stages are:

```text
depth -> tracking -> camerahmr -> match -> bbox
```

Existing complete artifacts are reused. Add `--overwrite` to rerun depth,
tracking, and CameraHMR. Add `--save-vis` only when debug images, meshes, or point
clouds are needed.

## Run individual stages

```bash
conda activate mev

./tools/run_pi3_depth.py
./tools/run_tracking.py
./tools/run_camerahmr.py
./tools/run_gemini_captions.py --stage match
./tools/run_bbox_video.py
./tools/run_gemini_captions.py --stage caption
```

Useful partial operations:

```bash
# Build reference images from an existing tracks.npy
./tools/run_tracking.py --refs-only

# Render existing bbox files without alignment or smoothing
./tools/run_bbox_video.py --render-only

# Regenerate raw and smoothed bbox data
./tools/run_bbox_video.py --sigma 4.0 --keyframe-interval 4

# Run one caption stage
./tools/run_gemini_captions.py --stage global
```

All commands accept `--video-dir` and `--results-dir`. Their defaults are
`example/video/` and `example/results/`.

## Output

The main output tree is:

```text
example/results/
├── info.json
├── matched_id.json
├── bbox.mp4
├── bbox_overlay.mp4
├── captions/
├── ref_imgs/
├── shot_0/
│   ├── pi3_all_results.npz
│   ├── tracks.npy
│   ├── camerahmr/
│   ├── human_bbox.npz
│   ├── human_bbox_smoothed.npz
│   ├── human_bbox_render.npz
│   ├── bbox_nocs.mp4
│   └── bbox_overlay.mp4
└── shot_1/
```

The bbox artifacts have distinct roles:

- `human_bbox.npz` stores raw per-frame OABBs from CameraHMR-to-Pi3 alignment.
- `human_bbox_smoothed.npz` stores the temporally smoothed pose, camera, and OABB
  result.
- `human_bbox_render.npz` is generated immediately before rendering. It keeps
  smoothed records and fills any missing frame or subject from the raw file. This
  prevents smoothing gaps from reducing the number of visible subject IDs.

`bbox.mp4` uses a black background and contains no ID labels.
`bbox_overlay.mp4` overlays the boxes on the source video and displays `new_id_*`
labels from `matched_id.json`.

## Captions

Set the API key outside the repository:

```bash
export GEMINI_API_KEY='YOUR_KEY'
./tools/run_gemini_captions.py --stage caption
```

Gemini stages are `global`, `shot`, `match`, and `subject`. The aggregate `caption`
stage runs `global`, `shot`, and `subject` without changing `matched_id.json`; this is
what the pipeline runs after bbox. Their JSON files are written under
`example/results/captions/`.
