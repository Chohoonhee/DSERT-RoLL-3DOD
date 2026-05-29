# DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR

**CVPR 2026** · 3D object detection track

[**Hoonhee Cho**](https://hoonhee.cho)\* · [**Jae-Young Kang**](#)\* · [**Yuhwan Jeong**](https://jeongyh98.github.io/)\* · Yunseo Yang · Wonyoung Lee · Youngho Kim · **Kuk-Jin Yoon**

Visual Intelligence Lab, KAIST · \* equal contribution

🌐 **[Project page](https://jeongyh98.github.io/dsert-roll)**

---

This repository contains the **3D object detection** code, configs, and
pre-trained checkpoint for our modality-adaptive fusion baseline
(R+E+T+4R+L) on the DSERT-RoLL dataset. It implements the framework
described in Section 4 of the paper and reproduces the "Ours" row of
Table 4 (and Table 3 ablations).

The framework fuses **LiDAR + 4D Radar + stereo RGB + Thermal + Event**
into a unified voxel-centric feature space with a confidence-gated
camera-3D fusion module and a LoGoHead-based second-stage refinement,
built on top of the [LoGoNet](https://github.com/PJLab-ADG/LoGoNet)
codebase.

### Citation

```bibtex
@inproceedings{cho2026dsertroll,
  title     = {DSERT-RoLL: Robust Multi-Modal Perception for Diverse
               Driving Conditions with Stereo Event-RGB-Thermal Cameras,
               4D Radar, and Dual-LiDAR},
  author    = {Cho, Hoonhee and Kang, Jae-Young and Jeong, Yuhwan and
               Yang, Yunseo and Lee, Wonyoung and Kim, Youngho and
               Yoon, Kuk-Jin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Table of Contents
1. [Repository Layout](#repository-layout)
2. [Installation](#installation) (verified end-to-end with the exact pinned
   versions below — please follow in order)
3. [Dataset Setup](#dataset-setup)
4. [Pre-trained Checkpoint](#pre-trained-checkpoint)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Reproducing the Reported Numbers](#reproducing-the-reported-numbers)
8. [Troubleshooting](#troubleshooting)
9. [Code Notes](#code-notes)

---

## Repository Layout

```
DSERT_release/
├── README.md
├── requirements.txt
├── setup_py.sh                              # Builds CUDA extensions
├── utils/                                   # al3d_utils package
│   ├── al3d_utils/
│   └── setup.py
└── detection/
    ├── setup.py
    ├── al3d_det/
    │   ├── datasets/
    │   │   ├── dataset.py                   # DatasetTemplate base class
    │   │   ├── augmentor/                   # Data augmentation
    │   │   ├── processor/                   # Voxelization, etc.
    │   │   └── dsert/                       # DSERT dataset + Waymo-protocol eval
    │   │       ├── dsert_dataset.py         # DSERTTrainingDataset, DSERTInferenceDataset
    │   │       └── dsert_eval_detection.py  # Waymo eval metric estimator
    │   ├── models/                          # CenterPoint MM, LoGoHead, Swin, ...
    │   └── utils/
    ├── tools/
    │   ├── test.py
    │   ├── train_mm.py
    │   ├── eval_utils.py
    │   ├── cfgs/
    │   │   ├── det_model_cfgs/dsert/ours.yaml
    │   │   └── det_dataset_cfgs/dsert_mm_lidar_radar_thermal_event.yaml
    │   └── scripts/
    │       ├── dist_test.sh
    │       └── dist_train_mm.sh
    ├── data/                                # Symlink target (gitignored)
    │   └── dsert-roll -> /path/to/dsert_roll/
    └── checkpoints/                         # Pre-trained weights (gitignored)
        └── checkpoint_epoch_20.pth
```

---

## Installation

**Tested configuration** (all installs verified from a clean conda env):

| Component | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| GPU driver | CUDA 11.1 toolkit (`nvcc`) + matching driver |
| Python | **3.6.13** (3.6 is required by mmdet 2.11.0 + spconv-cu111 wheels) |
| PyTorch | 1.10.0+cu111 |
| spconv | 2.1.25 (`spconv-cu111`) |
| mmcv-full | **1.4.0** (mmdet 2.11.0 is incompatible with newer mmcv) |
| mmdet | 2.11.0 (vendored under `detection/al3d_det/models/image_modules/swin_model/`) |
| timm | **< 0.6** (newer timm pulls `safetensors` which needs Rust to build on py3.6) |
| opencv-python / -headless | **4.6.0.66** (latest pre-built wheel for py3.6) |

The full installation takes ~30 min on a wired connection (most of the time
is `torch-scatter` CUDA compilation).

### 1. Conda environment + PyTorch (CUDA 11.1)

```bash
conda create --name dsert python=3.6 -y
conda activate dsert

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. System / sparse-conv / Waymo eval dependencies

```bash
conda install cmake -y
pip install spconv-cu111
pip install waymo-open-dataset-tf-2-0-0
```

### 3. Project Python dependencies

**Install opencv as a pre-built wheel first** — otherwise pip tries to compile
opencv from source (which fails on py3.6, and leaves all other packages
half-installed):

```bash
# from the repo root
pip install 'opencv-python==4.6.0.66' 'opencv-python-headless==4.6.0.66'
pip install -r requirements.txt
pip install pickle5 pycocotools
```

### 4. mmdet (vendored Swin Transformer backbone)

Don't run `pip install -r requirements.txt` inside `swin_model/` — it pulls
`opencv-python-headless` (source build) and `safetensors` (Rust build) which
both fail on py3.6. Install the required subset manually:

```bash
cd detection/al3d_det/models/image_modules/swin_model

# Build/runtime deps (skip optional + tests)
pip install 'timm<0.6' cython==0.29.33 matplotlib numpy six terminaltables

# mmcv-full must be 1.4.0 (mmdet 2.11.0 hard-asserts mmcv<=1.4.0)
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# Now build mmdet itself
python setup.py develop
cd -
```

### 5. Build CUDA extensions

A helper script chains the three `setup.py develop` calls:

```bash
bash setup_py.sh
```

Equivalent step-by-step (run from the repo root):

```bash
cd utils && python setup.py develop && cd -                              # al3d_utils ops
cd detection && python setup.py develop && cd -                          # al3d_det core
cd detection/al3d_det/models/ops && python setup.py develop && cd -      # DCN op
```

### 6. Verify the install

```bash
python -c "
from al3d_det.datasets.dsert.dsert_dataset import DSERTTrainingDataset
from al3d_det.models import build_network
print('Install OK')
"
```

You should see `Install OK`. If it errors, see [Troubleshooting](#troubleshooting).

> **CUDA note** — `spconv-cu111`, `mmcv-full` and the local extensions all
> need a CUDA 11.1 toolkit (`nvcc` + `CUDA_HOME`) at build time. For a
> different CUDA, switch the PyTorch wheel, `spconv-cu1XX`, and the
> `mmcv-full` index URL to matching versions.

---

## Dataset Setup

The DSERT dataset is not shipped in this repo. After obtaining the dataset,
symlink it under `detection/data/`:

```bash
ln -s /absolute/path/to/dsert_roll detection/data/dsert-roll
```

### Expected directory layout

```
dsert_roll/
├── ImageSets/
│   └── final/
│       ├── train.txt          # one "<weather>/<sequence>" per line
│       └── val.txt
└── processed_data/
    ├── Clear/
    │   └── <yyyy_mm_dd_hh_mm_ss>/
    │       ├── label.pkl
    │       ├── LIDAR_LIVOX_Tilted/0000.npy, ...
    │       ├── LIDAR_OUSTER_Tilted_90_degree/...
    │       ├── RADAR_Tilted/0000.npy, ...
    │       ├── rectified_crop_RGB_L/0000.jpg, ...
    │       ├── rectified_crop_RGB_R/...
    │       ├── rectified_THERMAL_L/0000.png, ...
    │       ├── rectified_THERMAL_R/...
    │       ├── rectified_EVENT_L/0000.npz, ...
    │       └── VOXEL_L/0000.npz, ...   # event voxel grids
    ├── Fog/        <yyyy_mm_dd_hh_mm_ss>/...
    ├── Light_Rain/ <yyyy_mm_dd_hh_mm_ss>/...
    ├── Heavy_Rain/ <yyyy_mm_dd_hh_mm_ss>/...
    ├── Light_Snow/ <yyyy_mm_dd_hh_mm_ss>/...
    └── Heavy_Snow/ <yyyy_mm_dd_hh_mm_ss>/...
```

### Weather and light classes

The dataset is split by **weather × light** condition; both labels live in
`label.pkl[meta]`:

| `meta['weather']` | Folder | Sequences with `label.pkl` | `meta['light']` | Sequences |
|---|---|---:|---|---:|
| `Clear`      | `Clear/`       | 104 | `Normal`      | 78 |
| `Fog`        | `Fog/`         |  32 | `Low_Light`   | 62 |
| `Light_Rain` | `Light_Rain/`  |  17 | `Over_Expose` | 32 |
| `Heavy_Rain` | `Heavy_Rain/`  |  19 | `HDR`         | 21 |
| `Light_Snow` | `Light_Snow/`  |  14 | | |
| `Heavy_Snow` | `Heavy_Snow/`  |   7 | | |
| **Total** |  | **193** | | **193** |

> **Note on Clear** — earlier dataset releases had the `Clear` weather class
> split across two folders (`Normal/` and `Night/`), with the day/night
> distinction encoded only in `meta['light']`. This release merges those
> two folders into a single `Clear/` directory, and every `label.pkl`
> already stores `weather='Clear'`. If you start from a pre-merge dump,
> use `_merge_clear.py` at the dataset root to consolidate; the
> corresponding `_merge_clear_rollback.py` undoes it. Code expects the
> merged layout.

### `label.pkl` structure

Each sequence's `label.pkl` is a dict with two top-level keys:

```python
data = pickle.load(open('Clear/2025_04_01_15_33_39/label.pkl', 'rb'))

data['meta'] = {
    'weather':     'Clear',
    'light':       'Normal',
    'sequence_len': 71,
    'calibration': {            # per-sensor intrinsics / extrinsics
        'RGB_L':      {'intrinsic': (3,3), 'shape': (W,H)},
        'RGB_R':      {'intrinsic': (3,3), 'extrinsic': (4,4), 'shape': (W,H)},
        'Thermal_L':  {'intrinsic': (3,3), 'extrinsic': (4,4), 'shape': (W,H)},
        'Thermal_R':  {'intrinsic': (3,3), 'extrinsic': (4,4), 'shape': (W,H)},
        'Event_L':    {'intrinsic': (3,3), 'extrinsic': (4,4), 'shape': (W,H)},
        'Event_R':    {'intrinsic': (3,3), 'extrinsic': (4,4), 'shape': (W,H)},
        'Livox':      {<cam>: (4,4) extrinsic},
        'Ouster':     {<cam>: (4,4) extrinsic},
        'Radar':      {<cam>: (4,4) extrinsic},
        'IMU':        {<cam>: (4,4) extrinsic},
    },
}

data['info'] = [
    {
        'time_stamp':    1744668663233690368,
        'sample_idx':    0,
        'frame_idx':     '0000',
        'sequence_name': 'Clear/2025_04_01_15_33_39',
        'sensor': {
            'rgb_left_path':     'Clear/.../rectified_crop_RGB_L/0000.jpg',
            'rgb_right_path':    'Clear/.../rectified_crop_RGB_R/0000.jpg',
            'thermal_left_path': 'Clear/.../rectified_THERMAL_L/0000.png',
            'thermal_right_path':'Clear/.../rectified_THERMAL_R/0000.png',
            'event_left_path':   'Clear/.../rectified_EVENT_L/0000.npz',
            'event_right_path':  'Clear/.../rectified_EVENT_R/0000.npz',
            'livox_path':        'Clear/.../LIDAR_LIVOX_Tilted/0000.npy',
            'ouster_path':       'Clear/.../LIDAR_OUSTER_Tilted_90_degree/0000.npy',
            'radar_path':        'Clear/.../RADAR_Tilted/0000.npy',
        },
        'pose':  (4, 4) ndarray,
        'annos': {
            'name':            (N,)  str,
            'obj_ids':         (N,)  int,
            'dimensions':      (N,3) float,    # l, w, h
            'location':        (N,3) float,    # x, y, z in Livox frame
            'heading_angles':  (N,)  float,
            'gt_boxes_livox':  (N,9) float,    # x,y,z,l,w,h,heading,vx,vy
            'gt_boxes_rgb':    (N,3) float,    # projected
            'gt_boxes_event':  (N,3) float,
            'gt_boxes_thermal':(N,3) float,
        },
    },
    ...
]
```

The loader normalizes the per-frame info by injecting `meta` keys into each
info dict ([dsert_dataset.py:132](detection/al3d_det/datasets/dsert/dsert_dataset.py#L132)).

### Split files

`ImageSets/final/{train,val}.txt` contains `<weather>/<sequence_name>`
per line, matching the folder structure under `processed_data/`. Example:

```
Clear/2025_04_01_15_33_39
Clear/2025_04_01_15_45_22
Fog/2025_04_15_07_10_51
Heavy_Snow/2025_02_12_04_42_55
...
```

Default split sizes (190 sequences total, ~5637 val frames):

| Split | Clear | Fog | L.Rain | H.Rain | L.Snow | H.Snow | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| train.txt | 72 | 23 | 12 | 12 | 8 | 5 | 132 |
| val.txt   | 30 |  9 |  5 |  6 | 6 | 2 | 58  |

(A handful of sequences exist on disk but are intentionally not in either
split — held out / corrupted captures.)

---

## Pre-trained Checkpoint

`checkpoint_epoch_20.pth` (~1.4 GB) is not tracked in git. Download it from
the project release page (or your shared storage) and place it at:

```
detection/checkpoints/checkpoint_epoch_20.pth
```

Verify with a md5sum or file size (~1.4 GB) before evaluation.

---

## Training

Run from `detection/tools/`:

```bash
cd detection/tools
bash scripts/dist_train_mm.sh 0,1 2 \
    --cfg_file cfgs/det_model_cfgs/dsert/ours.yaml \
    --extra_tag ours \
    --workers 4 \
    --find_unused_parameters \
    --max_ckpt_save_num 10
```

### Script arguments

```
bash scripts/dist_train_mm.sh <CUDA_VISIBLE_DEVICES> <NUM_GPUS> <python args...>
```

- `0,1` — comma-separated GPU IDs (mapped to local ranks 0..N-1)
- `2`   — number of processes (must match the GPU count)

### Common `train_mm.py` arguments

| Flag | Purpose |
|---|---|
| `--cfg_file` (required) | Model config (the dataset cfg is loaded via `_BASE_CONFIG_`) |
| `--extra_tag` | Run tag; appears in `output/.../<cfg>/<extra_tag>/` |
| `--workers` | DataLoader workers per process |
| `--find_unused_parameters` | DDP option; required for the LoGoHead second stage |
| `--max_ckpt_save_num N` | Keep at most N most-recent checkpoints |
| `--epochs N` | Override total epochs (config default: 20) |
| `--batch_size N` | Override per-GPU batch size (config default: 8) |
| `--pretrained_model PATH` | Load weights into the first stage before training |

### What a healthy run looks like

After the data loader builds (~30-60 s on first run, including pkl loads),
you should see progress like:

```
2026-05-29 13:09:41   INFO  *********** Start training det_model_cfgs/dsert/ours(ours) ***********
epochs:   0%|          | 0/20 [00:00<?, ?it/s]
train:   1%|▏         | 10/1003 [00:25<41:24,  2.50s/it, total_it=10]
... loss=6.85, lr=0.000108 ...
... loss=3.99, lr=0.000109 ...
```

On 2× Quadro RTX 8000 (48 GB), expect **~2.5 s/iter** at the default
`BATCH_SIZE_PER_GPU: 8` (≈ 1003 iters/epoch × 20 epochs ≈ 14 h total).
Both GPUs should sit near 100% utilization and ~40 GB VRAM each.

Outputs land under
`detection/output/det_model_cfgs/dsert/ours/<extra_tag>/`:

```
ckpt/                  # checkpoint_epoch_*.pth
log_train_<ts>.txt     # full training log
tensorboard/           # TensorBoard event files
```

---

## Evaluation

```bash
cd detection/tools
bash scripts/dist_test.sh 0,1 2 \
    --cfg_file cfgs/det_model_cfgs/dsert/ours.yaml \
    --batch_size 4 \
    --ckpt ../checkpoints/checkpoint_epoch_20.pth \
    --extra_tag eval
```

The output is reported in four blocks (via the Waymo Open Dataset official
`compute_ap` API):

```
==== Overall ====
[overall] N=5637
  OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP: 0.8709
  OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH: 0.8626
  OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP: 0.3709
  OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH: 0.2344
  OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP: 0.5862
  OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH: 0.5659

==== Per Weather ====
[weather=Clear]      N=3248   ... AP / APH per class ...
[weather=Fog]        N=608
[weather=Light_Rain] N=576
[weather=Heavy_Rain] N=496
[weather=Light_Snow] N=469
[weather=Heavy_Snow] N=240

==== Per Light ====
[light=Normal]      N=1802
[light=Low_Light]   N=2029
[light=Over_Expose] N=956
[light=HDR]         N=850
```

Results are also written to
`detection/output/det_model_cfgs/dsert/ours/<extra_tag>/eval/epoch_<E>/val/`:

| File | Contents |
|---|---|
| `result.pkl` | Per-frame predictions (boxes, scores, labels) |
| `log_eval_<ts>.txt` | Full eval log (the AP blocks above) |
| `final_result/data/*.bin` | Waymo protobuf format |

Inference throughput on 2× RTX 8000: **~197 ms / frame** (≈5 fps).

---

## Reproducing the Reported Numbers

With the released `checkpoint_epoch_20.pth`, the eval command above
should reproduce these numbers (Vehicle AP shown):

| Slice | N | Vehicle AP | Vehicle APH |
|---|---:|---:|---:|
| **Overall**       | 5637 | **0.8709** | 0.8626 |
| Clear             | 3248 | 0.9030 | 0.8950 |
| Fog               |  608 | 0.7142 | 0.7085 |
| Light_Rain        |  576 | 0.9510 | 0.9390 |
| Heavy_Rain        |  496 | 0.8026 | 0.7957 |
| Light_Snow        |  469 | 0.8559 | 0.8445 |
| Heavy_Snow        |  240 | 0.7294 | 0.7121 |
| ─── per-light ─── | | | |
| Normal            | 1802 | 0.8293 | 0.8210 |
| Low_Light         | 2029 | 0.9265 | 0.9171 |
| Over_Expose       |  956 | 0.8547 | 0.8462 |
| HDR               |  850 | 0.8633 | 0.8569 |

Small numerical deltas (±0.001 AP) are expected due to non-determinism in
distributed all-gather ordering. If you see large deltas (> 0.01), check
that you are loading the released `epoch_20.pth` and not an in-training
snapshot.

---

## Troubleshooting

Issues we hit during a clean install, and how to resolve each one:

| Symptom | Root cause | Fix |
|---|---|---|
| `ERROR: Could not build wheels for opencv-python` | py3.6 source build of opencv fails | `pip install 'opencv-python==4.6.0.66' 'opencv-python-headless==4.6.0.66'` BEFORE `requirements.txt` |
| `ERROR: Could not build wheels for safetensors` | New `timm` (≥0.6) pulls `safetensors` which needs Rust | `pip install 'timm<0.6'` |
| `AssertionError: MMCV==1.7.2 is used but incompatible` | mmdet 2.11.0 requires `mmcv>=1.2.4,<=1.4.0` | `pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html` |
| `ModuleNotFoundError: No module named 'pycocotools'` | Missing dep | `pip install pycocotools` |
| `ValueError: unsupported pickle protocol: 5` | py3.6 stdlib `pickle` only supports up to protocol 4 | `pip install pickle5` (the loader uses `import pickle5 as pickle`) |
| `ModuleNotFoundError: No module named 'al3d_det.datasets.kitti'` | Stale import in old `data_augmentor.py` | Already removed in this release; pull latest if you see this |
| `AttributeError: ... 'max_distance'` in `dataset.py` | `DSERTTrainingDataset.__init__` doesn't set `max_distance` | Already fixed; `self.max_distance = cfg.get('MAX_DIST', 80.0)` is required |
| `torchaudio-...+rocm4.1` warning | pip auto-picks ROCm wheel for torchaudio when only torchaudio==0.10.0 (no `+cu111` variant) is published | Harmless — the project doesn't import torchaudio |
| Training crashes in DataLoader worker with little context | Set `--workers 0` to surface the real traceback synchronously, fix, then restore workers |
| GPU 0/1 OOM | Reduce `BATCH_SIZE_PER_GPU` in `ours.yaml` (default 8) or use fewer cameras in the dataset cfg |
| `tensorflow-gpu` import warnings | TF 2.0 is only used for the Waymo eval metric, not training | Ignore |

If `Install OK` from step 6 succeeds but training fails before the first
iter, run with `--workers 0` to see the real traceback (worker subprocess
errors are often swallowed by torch's DataLoader).

---

## Code Notes

- **Main dataset class**: `DSERTTrainingDataset` in
  [detection/al3d_det/datasets/dsert/dsert_dataset.py](detection/al3d_det/datasets/dsert/dsert_dataset.py).
  Reads `processed_data/<seq>/label.pkl`, joins per-frame `info` with the
  shared `meta` (weather/light/calibration), and yields multi-modal tensors.
- **Evaluation**: [detection/al3d_det/datasets/dsert/dsert_eval_detection.py](detection/al3d_det/datasets/dsert/dsert_eval_detection.py)
  wraps `waymo_open_dataset.metrics.python.detection_metrics`. The
  per-slice grouping (overall / weather / light / per-sequence) is in
  `DSERTTrainingDataset.evaluation()`.
- **Model**: the cfg `MODEL.NAME: CenterPointMM` selects a multi-modal
  CenterPoint with three image backbones (RGB, Thermal, Event), a
  per-modality FPN, LiDAR + Radar sparse 3D backbones, BEV fusion, and a
  LoGoHead second stage (point-voxel ROI + cross-attention image fusion).
- **Why mmdet 2.11.0?** It's the version vendored under
  `swin_model/`; this avoids depending on the upstream `mmdetection` repo
  and pins the Swin Transformer implementation against a known mmcv API.
- **CUDA extensions** that get built locally:
  - `al3d_utils`: `iou3d_nms`, `roiaware_pool3d`, `roipoint_pool3d`,
    `pointnet2_stack/_batch`, `dcn/deform_conv`
  - `al3d_det`: `MultiScaleDeformableAttention`
- **Dataset merge tooling** (lives at the dataset root):
  - `_merge_clear.py` — folds `Normal/` + `Night/` into `Clear/`
    (already applied in this release)
  - `_merge_clear_rollback.py` — undoes the merge using the recorded
    backup mapping (`_merge_clear_backup.txt`)
