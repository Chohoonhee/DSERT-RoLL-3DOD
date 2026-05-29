# DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions with Stereo Event-RGB-Thermal Cameras, 4D Radar, and Dual-LiDAR

**CVPR 2026** · 3D object detection

[**Hoonhee Cho**](https://chohoonhee.github.io/hoonheecho/)\* · [**Jae-Young Kang**](https://mickeykang16.github.io/)\* · [**Yuhwan Jeong**](https://jeongyh98.github.io/)\* · Yunseo Yang · Wonyoung Lee · Youngho Kim · [**Kuk-Jin Yoon**](https://vi.kaist.ac.kr/)

[Visual Intelligence Lab](https://vi.kaist.ac.kr/), KAIST · \* equal contribution

🌐 **[Project page](https://jeongyh98.github.io/dsert-roll)**

---

This repository contains the **3D object detection** code, configs, and
pre-trained checkpoint for our modality-adaptive fusion baseline
(R+E+T+4R+L) on the DSERT-RoLL dataset. The framework fuses
**LiDAR + 4D Radar + RGB + Thermal + Event** into a unified
voxel-centric feature space with a confidence-gated camera–3D fusion
module and a LoGoHead-based second-stage refinement.

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

## Installation

Tested on Ubuntu 20.04/22.04, CUDA 11.1, **Python 3.6**, PyTorch 1.10.

### 1. Conda environment + PyTorch

```bash
conda create --name dsert python=3.6 -y
conda activate dsert

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. System / sparse-conv / Waymo eval

```bash
conda install cmake -y
pip install spconv-cu111
pip install waymo-open-dataset-tf-2-0-0
```

### 3. Project Python dependencies

```bash
pip install -r requirements.txt
```

### 4. mmdet (vendored Swin Transformer backbone)

```bash
cd detection/al3d_det/models/image_modules/swin_model
pip install 'timm<0.6' cython==0.29.33 matplotlib numpy six terminaltables
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
python setup.py develop
cd -
```

### 5. Build CUDA extensions

```bash
bash setup_py.sh
```

This compiles `al3d_utils` ops, `al3d_det`, and the Deformable-attention
DCN op. On success the helper script prints `Install OK`.

---

## Dataset

Download the DSERT-RoLL dataset from the
[project page](https://jeongyh98.github.io/dsert-roll) and symlink it
under `detection/data/`:

```bash
ln -s /absolute/path/to/dsert_roll detection/data/dsert-roll
```

The dataset ships with this layout:

```
dsert_roll/
├── ImageSets/final/{train,val}.txt    # split files (sequence per line)
└── processed_data/
    ├── Clear/<sequence>/{label.pkl, livox/, radar/, RGB_L/, ...}
    ├── Fog/<sequence>/...
    ├── Light_Rain/<sequence>/...
    ├── Heavy_Rain/<sequence>/...
    ├── Light_Snow/<sequence>/...
    └── Heavy_Snow/<sequence>/...
```

**Annotations.** Each sequence's `label.pkl` is a dict with
`meta` (calibration, `weather`, `light`) and per-frame `info`
(timestamps, sensor file paths, pose, and 3D `annos`). See
[`dsert_dataset.py`](detection/al3d_det/datasets/dsert/dsert_dataset.py)
for the exact schema.

---

## Pre-trained Checkpoint

The pre-trained checkpoint (~1.4 GB) is hosted on Hugging Face:
🤗 **[HoonheeCho/DSERT-RoLL-3DOD](https://huggingface.co/HoonheeCho/DSERT-RoLL-3DOD)**

```bash
mkdir -p detection/checkpoints
# Option 1: direct download
wget -O detection/checkpoints/checkpoint_epoch_20.pth \
    https://huggingface.co/HoonheeCho/DSERT-RoLL-3DOD/resolve/main/checkpoint_epoch_20.pth

# Option 2: via huggingface_hub
pip install -U huggingface_hub
huggingface-cli download HoonheeCho/DSERT-RoLL-3DOD checkpoint_epoch_20.pth \
    --local-dir detection/checkpoints --local-dir-use-symlinks False
```

---

## Training

```bash
cd detection/tools
bash scripts/dist_train_mm.sh 0,1 2 \
    --cfg_file cfgs/det_model_cfgs/dsert/ours.yaml \
    --extra_tag ours \
    --workers 4 \
    --find_unused_parameters \
    --max_ckpt_save_num 10
```

- `0,1` — `CUDA_VISIBLE_DEVICES`
- `2`   — number of GPUs

Outputs land under `detection/output/det_model_cfgs/dsert/ours/<extra_tag>/`.

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

Evaluation reports **Overall**, **Per Weather**, and **Per Light** AP
using the Waymo Open Dataset official `compute_ap` API. Results and
the per-frame `result.pkl` are written under
`detection/output/det_model_cfgs/dsert/ours/<extra_tag>/eval/`.

---

## Acknowledgement

This codebase builds on
[LoGoNet](https://github.com/PJLab-ADG/LoGoNet) and the vendored
[Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
backbone.
