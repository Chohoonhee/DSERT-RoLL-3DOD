from functools import partial

import numpy as np
import torch
from skimage import transform
import cumm.tensorview as tv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

from al3d_utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        # 🔽 기존 단일 제너레이터 대신 모달리티별 dict 사용
        self.voxel_generators = {}
        self.voxel_coords_generators = {}

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)
    def _normalize_points(self, pts, expC):
        # DataContainer unwrap
        if hasattr(pts, 'data'):
            pts = pts.data
        # torch -> numpy
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        pts = np.asarray(pts, dtype=np.float32)

        # 채널 정합
        if pts.shape[1] != expC:
            if pts.shape[1] > expC:
                pts = pts[:, :expC]
            else:
                raise ValueError(f'points C={pts.shape[1]} but expected {expC}')
        return pts

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            # 🔽 기본 'points' 말고도 원하는 키로 동작 가능
            self._mpr_points_key = getattr(config, 'POINTS_KEY', 'points')
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        pts_key = getattr(config, 'POINTS_KEY', 'points')
        if data_dict.get(pts_key, None) is not None:
            mask = common_utils.mask_points_by_range(data_dict[pts_key], self.point_cloud_range)
            data_dict[pts_key] = data_dict[pts_key][mask]

        # gt_boxes는 라이다 기준일 때만 보통 쓰므로 그대로 유지
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            self._shuf_points_key = getattr(config, 'POINTS_KEY', 'points')
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            pts_key = getattr(config, 'POINTS_KEY', 'points')
            points = data_dict[pts_key]
            shuffle_idx = np.random.permutation(points.shape[0])
            data_dict[pts_key] = points[shuffle_idx]
        return data_dict


    def generate_voxel_coords(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.generate_voxel_coords, config=config)

        pts_key  = getattr(config, 'POINTS_KEY', 'points')
        out_key  = getattr(config, 'VOXEL_COORDS_KEY', 'voxel_coords_downscale')
        expC     = getattr(config, 'NUM_FEATURES', self.num_point_features)

        # 제너레이터 키(모달리티/해상도별로 따로 캐시)
        vg_key = f'coords|{pts_key}|{tuple(config.VOXEL_SIZE)}|{expC}|{self.mode}'
        if vg_key not in self.voxel_coords_generators:
            self.voxel_coords_generators[vg_key] = VoxelGenerator(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=expC,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        pts = self._normalize_points(data_dict[pts_key], expC)
        _, tv_coordinates, _ = self.voxel_coords_generators[vg_key].point_to_voxel(tv.from_numpy(pts))
        data_dict[out_key] = tv_coordinates.numpy()
        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, config=config)

        pts_key  = getattr(config, 'POINTS_KEY', 'points')
        vox_key  = getattr(config, 'VOXELS_KEY', 'voxels')
        coor_key = getattr(config, 'VOXEL_COORDS_KEY', 'voxel_coords')
        nump_key = getattr(config, 'VOXEL_NUM_POINTS_KEY', 'voxel_num_points')
        expC     = getattr(config, 'NUM_FEATURES', self.num_point_features)
        use_lead_xyz = getattr(config, 'USE_LEAD_XYZ', True)

        vg_key = f'vox|{pts_key}|{tuple(config.VOXEL_SIZE)}|{expC}|{self.mode}'
        if vg_key not in self.voxel_generators:
            self.voxel_generators[vg_key] = VoxelGenerator(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=expC,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        pts = self._normalize_points(data_dict[pts_key], expC)
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        tv_voxels, tv_coordinates, tv_num_points = self.voxel_generators[vg_key].point_to_voxel(tv.from_numpy(pts))
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()

        if not use_lead_xyz:
            voxels = voxels[..., 3:]  # xyz 제거

        data_dict[vox_key]  = voxels
        data_dict[coor_key] = coordinates
        data_dict[nump_key] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict
    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    def pad_image(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.pad_image, config=config)

        w_h_lists = config.TARGET_SHAPE 
        
        
        if 'images' in data_dict:
            for cam, w_h_list in zip(data_dict['images'].keys(), w_h_lists):
                max_w, max_h  = w_h_list
                cur_size = data_dict['images'][cam][0].shape
                
                pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=cur_size[0])
                pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=cur_size[1])
                pad_width = (pad_h, pad_w, (0, 0))
                pad_value = 0
                # import pdb; pdb.set_trace()
                img_pad = [np.pad(img, pad_width=pad_width, mode='constant', constant_values=pad_value)
                           for img in data_dict['images'][cam]]
                data_dict['images'][cam] = img_pad
                # w, h, c = img_pad[0].shape
                # data_dict['image_shape'][cam] = [max_h, max_w]

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
