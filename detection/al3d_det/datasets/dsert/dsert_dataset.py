import os
import pickle5 as pickle
import copy

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from al3d_utils import common_utils

from al3d_det.datasets.dataset import DatasetTemplate
from al3d_det.datasets.augmentor.data_augmentor import DataAugmentor
from al3d_det.datasets.augmentor.test_time_augmentor import TestTimeAugmentor


def project_livox_to_rgb_l(points_livox, K, T_livox_to_cam, image_shape_wh):
    """
    points_livox: (N,3) in Livox frame
    K: (3,3) intrinsic
    T_livox_to_cam: (4,4) extrinsic (Livox -> Camera)
    image_shape_wh: (W,H)
    returns:
      uv: (N,2) projected pixels
      z_cam: (N,) camera-depth
    """
    points_livox = np.asarray(points_livox, dtype=np.float64)
    if points_livox.ndim == 1:
        points_livox = points_livox[None, :]

    N = points_livox.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.concatenate([points_livox, ones], axis=1)

    pts_cam_h = (T_livox_to_cam @ pts_h.T).T
    x, y, z = pts_cam_h[:, 0], pts_cam_h[:, 1], pts_cam_h[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = np.stack([u, v], axis=1)

    return uv, z


def make_in_image_mask(new_info, cam_name="RGB_L", lidar_name="Livox"):
    """
    Returns:
      mask: (N,) True if projected point is inside the image & in front of camera
      uv:   (N,2) projected pixels
    """
    K = np.asarray(new_info["calibration"][cam_name]["intrinsic"], dtype=np.float64)
    W, H = map(float, new_info["calibration"][cam_name]["shape"])
    T = np.asarray(new_info["calibration"][lidar_name][cam_name], dtype=np.float64)

    uv, z = project_livox_to_rgb_l(new_info['annos']['location'], K, T, (W, H))

    front = z > 0
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    mask = front & inside
    return mask, uv


class DSERTInferenceDataset(DatasetTemplate):
    """
    The Dataset class for Inference on DSERT
    """
    def __init__(self, dataset_cfg, class_names, data_infos, point_list, training=False, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, logger)
        self.data_infos = data_infos
        self.point_list = point_list
        self.init_infos()

    def init_infos(self):
        self.infos = self.data_infos

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            infos.append(self.infos[i])
            points.append(self.point_list[i])
        return infos, points


class DSERTTrainingDataset(DatasetTemplate):
    """
    The Dataset class for Training / Evaluation on DSERT (from File System)
    """

    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.max_distance = self.dataset_cfg.get('MAX_DIST', 80.0)
        self.data_path = self.root_path + '/' + dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = os.path.join(self.root_path, 'ImageSets/final', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.init_infos()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names,
            training=self.training, root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets/final', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.init_infos()

    def init_infos(self):
        self.logger.info('Loading DSERT dataset')
        dsert_infos = []
        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = os.path.join(self.data_path, sequence_name, 'label.pkl')

            if not os.path.exists(info_path):
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

            new_infos = []
            for i in range(len(infos['info'])):
                new_info = copy.deepcopy(infos['info'][i])
                new_info.update(infos['meta'])
                new_infos.append(new_info)
            dsert_infos.extend(new_infos)

        self.infos.extend(dsert_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for DSERT dataset: %d' % (len(dsert_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_infos.append(self.infos[k])
            self.infos = sampled_infos
            self.logger.info('Total sampled samples for DSERT dataset: %d' % len(self.infos))

    def get_infos_and_points(self, idx_list):
        infos, points, points_radar = [], [], []
        for i in idx_list:
            lidar_path = self.infos[i]['sensor']["livox_path"]
            radar_path = self.infos[i]['sensor']["radar_path"]
            lidar_path = os.path.join(self.data_path, lidar_path)
            radar_path = os.path.join(self.data_path, radar_path)

            p_load = np.load(lidar_path)
            current_point = np.stack([p_load['x'], p_load['y'], p_load['z'], p_load['intensity']], -1)

            r_load = np.load(radar_path)
            ones = np.ones((r_load.shape[0]))

            radar2thermal = self.infos[i]['calibration']['Radar']['Thermal_L']
            thermal2lidar = self.inverse_T(self.infos[i]['calibration']['Livox']['Thermal_L'])
            radar2lidar = thermal2lidar @ radar2thermal

            if r_load.shape[0] > 0:
                new_radar_points = (radar2lidar @ np.stack([r_load['x'], r_load['y'], r_load['z'], ones], -1).T).T
                new_r_x, new_r_y, new_r_z = new_radar_points[:, 0], new_radar_points[:, 1], new_radar_points[:, 2]
                radar_point = np.stack([new_r_x, new_r_y, new_r_z, r_load['doppler']], -1)
            else:
                radar_point = np.array([[50.0, 0.0, 0.0, 0.0]])

            # Append a sentinel point to guarantee non-empty tensors downstream
            if len(current_point) > 0:
                current_point = np.concatenate([current_point, np.array([[10.0, 0.0, 0.0, 0.0]])], 0)
            else:
                current_point = np.array([[10.0, 0.0, 0.0, 0.0]])

            if len(radar_point) > 0:
                radar_point = np.concatenate([radar_point, np.array([[10.0, 0.0, 0.0, 0.0]])], 0)
            else:
                radar_point = np.array([[10.0, 0.0, 0.0, 0.0]])

            infos.append(self.infos[i])
            points.append(current_point)
            points_radar.append(radar_point)

        return infos, points, points_radar

    def inverse_T(self, T):
        assert T.shape == (4, 4)
        R = T[:3, :3]
        R_inv = np.linalg.inv(R)
        t = T[:-1, -1]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:-1, -1] = -R_inv @ t
        return T_inv

    def get_images_and_params(self, current_idx, idx_list):
        imgs_dict = {
            'images': {},
            'extrinsic': {},
            'intrinsic': {},
            'image_shape': {}
        }

        cam_defs = [
            ("camera_0", "rgb_left_path",     "RGB_L",     "color"),
            ("camera_1", "thermal_left_path", "Thermal_L", "thermal"),
            ("camera_2", "event_left_path",   "Event_L",   "event"),
        ]

        for i in idx_list:
            if not self.load_multi_images:
                if i != current_idx:
                    continue

            img_infos = dict()
            for j, (cam_name, sensor_key, calib_key, mode) in enumerate(cam_defs):
                rel_path = self.infos[i]["sensor"].get(sensor_key, None)
                if rel_path is None:
                    continue
                if cam_name not in self.image_scale.keys():
                    continue

                img_path = os.path.join(self.data_path, rel_path)

                if mode == "color":
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    image = image.astype(np.float32) / 255.0
                elif mode == "thermal":
                    timg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if timg is None:
                        continue
                    if timg.dtype == np.uint16:
                        image = (timg.astype(np.float32) / 65535.0)
                    else:
                        image = timg.astype(np.float32) / 255.0
                else:  # event
                    image = np.load(img_path.replace('rectified_EVENT_L', 'VOXEL_L'))['voxel']

                if self.image_scale[cam_name] != 1:
                    if mode in ["color", "thermal"]:
                        new_w = int(image.shape[1] * self.image_scale[cam_name])
                        new_h = int(image.shape[0] * self.image_scale[cam_name])
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        new_w = int(image.shape[2] * self.image_scale[cam_name])
                        new_h = int(image.shape[1] * self.image_scale[cam_name])
                        voxel = torch.from_numpy(image)
                        voxel = voxel.unsqueeze(0)
                        voxel = F.interpolate(voxel, size=(new_h, new_w))
                        image = voxel.squeeze(0).numpy()
                        image = image.transpose(1, 2, 0)
                    img_infos[f"image_shape_{j}"] = (new_h, new_w)
                else:
                    img_infos[f"image_shape_{j}"] = (image.shape[0], image.shape[1])

                if cam_name not in imgs_dict["images"]:
                    imgs_dict["images"][cam_name] = []
                imgs_dict["images"][cam_name].append(image)

            for j, (cam_name, sensor_key, calib_key, mode) in enumerate(cam_defs):
                if cam_name not in imgs_dict["images"]:
                    continue

                ext = self.infos[i]["calibration"]["Livox"][calib_key]
                imgs_dict["extrinsic"][cam_name] = ext

                intr = self.infos[i]["calibration"][calib_key]["intrinsic"]
                imgs_dict["intrinsic"][cam_name] = intr

                imgs_dict["image_shape"][cam_name] = img_infos.get(f"image_shape_{j}", None)

        return imgs_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        """
        Overall / per-weather / per-light / per-sequence evaluation
        using Waymo Open Dataset's official detection metrics API.
        """
        from collections import defaultdict

        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        EXCLUDED_SIGN_NAMES = {'sign', 'traffic_sign', 'traffic-sign', 'traffic sign'}
        filtered_class_names = [
            c for c in class_names
            if c.strip().lower() not in EXCLUDED_SIGN_NAMES
        ]

        WEATHER_ORDER = ['Clear', 'Fog', 'Light_Rain', 'Heavy_Rain', 'Light_Snow', 'Heavy_Snow']
        LIGHT_ORDER = ['Normal', 'Low_Light', 'Over_Expose', 'HDR']

        def _ordered_keys(keys, preferred):
            s = set(keys)
            head = [k for k in preferred if k in s]
            tail = sorted(k for k in s - set(preferred))
            return head + tail

        def get_seq_name(info):
            for k in ('sequence', 'seq', 'segment_name', 'sequence_name', 'log_id'):
                if k in info and info[k] is not None:
                    s = str(info[k]).strip()
                    if s:
                        return s
            if 'frame_id' in info:
                return f"seq:unknown:{info['frame_id']}"
            return "seq:unknown"

        def _to_float(v):
            if isinstance(v, (list, tuple, np.ndarray)):
                return float(np.asarray(v).reshape(-1)[0])
            try:
                return float(v)
            except Exception:
                return v

        def _keep_l1_drop_l2_and_sign(ap_dict):
            out = {}
            for k, v in ap_dict.items():
                ku = k.upper()
                if 'SIGN' in ku:
                    continue
                if 'LEVEL_2' in ku:
                    continue
                if ('LEVEL_1' not in ku) and ('LEVEL_2' not in ku):
                    arr = np.asarray(v).reshape(-1)
                    if arr.size == 2:
                        out[f'{k}_LEVEL_1'] = _to_float(arr[0])
                    else:
                        out[k] = _to_float(v)
                else:
                    out[k] = _to_float(v)
            return out

        def dsert_eval(eval_det_annos, eval_gt_annos):
            from .dsert_eval_detection import WaymoDetectionMetricsEstimator
            estimator = WaymoDetectionMetricsEstimator()
            ap_dict = estimator.waymo_evaluation(
                eval_det_annos, eval_gt_annos,
                class_name=filtered_class_names,
                distance_thresh=70.0,
                fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False),
                fov_flag=self.dataset_cfg.get('EVAL_FOV_FLAG', False),
            )
            ap_dict = _keep_l1_drop_l2_and_sign(ap_dict)
            return ap_dict

        def run_eval_on_indices(indices):
            eval_det = [copy.deepcopy(det_annos[i]) for i in indices]
            eval_gt = [copy.deepcopy(self.infos[i]['annos']) for i in indices]
            return dsert_eval(eval_det, eval_gt)

        # ---------- Overall ----------
        overall_dict = dsert_eval(
            copy.deepcopy(det_annos),
            [copy.deepcopy(info['annos']) for info in self.infos]
        )
        overall_N = len(self.infos)

        # ---------- Group indices ----------
        weather_to_idx = defaultdict(list)
        light_to_idx = defaultdict(list)
        for i, info in enumerate(self.infos):
            w = info.get('weather', None)
            l = info.get('light', None)
            if w is not None:
                weather_to_idx[w].append(i)
            if l is not None:
                light_to_idx[l].append(i)

        # ---------- Per weather ----------
        per_weather = {}
        for w in _ordered_keys(weather_to_idx.keys(), WEATHER_ORDER):
            per_weather[w] = run_eval_on_indices(weather_to_idx[w])

        # ---------- Per light ----------
        per_light = {}
        for l in _ordered_keys(light_to_idx.keys(), LIGHT_ORDER):
            per_light[l] = run_eval_on_indices(light_to_idx[l])

        # ---------- Per-sequence (selected weathers) ----------
        target_seq_weathers = kwargs.get(
            'seq_weathers',
            ['Clear', 'Fog', 'Heavy_Snow', 'Light_Snow', 'Heavy_Rain', 'Light_Rain']
        )

        per_weather_seq_indices = {}
        for w, idxs in weather_to_idx.items():
            if w not in target_seq_weathers:
                continue
            bucket = defaultdict(list)
            for i in idxs:
                seq_name = get_seq_name(self.infos[i])
                bucket[seq_name].append(i)
            per_weather_seq_indices[w] = dict(bucket)

        per_sequence_by_weather = {}
        for w in _ordered_keys(per_weather_seq_indices.keys(), WEATHER_ORDER):
            seq_map = per_weather_seq_indices[w]
            per_sequence_by_weather[w] = {}
            for seq_name in sorted(seq_map.keys()):
                idxs = seq_map[seq_name]
                if not idxs:
                    continue
                per_sequence_by_weather[w][seq_name] = run_eval_on_indices(idxs)

        def format_block(title, metrics_dict, N, indent=0):
            pad = ' ' * indent
            lines = [f'{pad}[{title}] N={N}']
            for k in sorted(metrics_dict.keys()):
                v = metrics_dict[k]
                try:
                    lines.append(f'{pad}{k}: {float(v):.4f}')
                except Exception:
                    lines.append(f'{pad}{k}: {v}')
            return '\n'.join(lines)

        per_weather_seq_counts = {}
        for w, seq_map in per_weather_seq_indices.items():
            per_weather_seq_counts[w] = {seq: len(idxs) for seq, idxs in seq_map.items()}

        parts = []
        parts.append('==== Overall ====')
        parts.append(format_block('overall', overall_dict, overall_N))

        parts.append('\n==== Per Weather ====')
        if len(per_weather) == 0:
            parts.append('[weather] N=0 (no weather keys)')
        else:
            for w in _ordered_keys(per_weather.keys(), WEATHER_ORDER):
                parts.append(format_block(f'weather={w}', per_weather[w], len(weather_to_idx[w])))

        parts.append('\n==== Per Light ====')
        if len(per_light) == 0:
            parts.append('[light] N=0 (no light keys)')
        else:
            for l in _ordered_keys(per_light.keys(), LIGHT_ORDER):
                parts.append(format_block(f'light={l}', per_light[l], len(light_to_idx[l])))

        ap_result_str = '\n'.join(parts) + '\n'

        merged = {
            'overall': overall_dict,
            'per_weather': per_weather,
            'per_light': per_light,
            'per_sequence_by_weather': per_sequence_by_weather,
            'counts': {
                'overall': overall_N,
                'per_weather': {k: len(v) for k, v in weather_to_idx.items()},
                'per_light': {k: len(v) for k, v in light_to_idx.items()},
                'per_sequence_by_weather': per_weather_seq_counts,
            }
        }
        return ap_result_str, merged
