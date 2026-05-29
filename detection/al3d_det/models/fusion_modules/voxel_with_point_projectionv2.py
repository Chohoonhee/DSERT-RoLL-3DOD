import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_fusion import DeformTransLayer
from .point_to_image_projectionv2 import Point2ImageProjectionV2
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock1D



class CameraChannelAttention(nn.Module):
    """
    입력 x: (N, K, Cmid)
    카메라 축(K)에 대해 SE-style 채널 어텐션을 수행하여 (K,) 게이트를 산출하고
    각 카메라 분기(feature)에 스칼라를 곱해줍니다.
    """
    def __init__(self, num_cams: int, reduction: int = 4, normalize: bool = False):
        super().__init__()
        hidden = max(1, num_cams // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(num_cams, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_cams, bias=True),
            nn.Sigmoid()
        )
        self.normalize = normalize

    def forward(self, x: torch.Tensor):
        # x: (N, K, Cmid)
        assert x.dim() == 3, f"Expected (N, K, Cmid), got {x.shape}"
        # N과 Cmid를 평균 풀링 → (K,)
        s = x.mean(dim=(0, 2))                   # (K,)
        w = self.mlp(s)                          # (K,), 0~1
        if self.normalize:
            w = w / (w.sum() + 1e-6) * w.numel() # 선택: 평균 1로 정규화(선호시)
        x = x * w.view(1, -1, 1)                 # 브로드캐스팅
        return x
    
    
class VoxelWithPointProjectionV2(nn.Module):
    def __init__(self, 
                fuse_mode, 
                interpolate, 
                voxel_size, 
                pc_range, 
                image_list, 
                image_scale=[1.0], 
                depth_thres=0, 
                mid_channels = 16,
                double_flip=False, 
                dropout_ratio=0,
                layer_channel=None,
                activate_out=True,
                fuse_out=False):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_projector = Point2ImageProjectionV2(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        self.mid_channels = mid_channels
        self.dropout_ratio = dropout_ratio
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)
        elif self.fuse_mode == 'crossattention_deform':
            self.pts_key_proj = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            self.pts_transform = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            self.fuse_blocks = DeformTransLayer(d_model=self.mid_channels, \
                    n_levels=1, n_heads=4, n_points=4)
            if self.fuse_out:
                self.fuse_conv = nn.Sequential(
                    nn.Linear(self.mid_channels + len(self.image_list)*self.mid_channels, self.mid_channels),
                    # For pts the BN is initialized differently by default
                    # TODO: check whether this is necessary
                    nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU())
                
                self.num_cams = len(self.image_list)
                self.cam_attn = CameraChannelAttention(num_cams=self.num_cams, reduction=4, normalize=False)

    def fusion_back(self, voxel_feat, layer_name):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        fuse_feat = torch.zeros(voxel_feat.shape).to(voxel_feat.device)
        concat_feat = torch.cat([fuse_feat.permute(1,0).contiguous(), voxel_feat.permute(1,0).contiguous()], dim=0)
        voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
        voxel_feat = voxel_feat.permute(1,0).contiguous()
        return voxel_feat


    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if self.fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = voxel_feat + fuse_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0).contiguous()) / 2
        elif self.fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0).contiguous()], dim=0)
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'crossattention':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]].permute(1,0).contiguous()
            voxel_feat = self.fuse_blocks(fuse_feat.unsqueeze(0), voxel_feat.unsqueeze(0))
        else:
            raise NotImplementedError
        
        return voxel_feat
    def fusion_withdeform(self, img_pre_fuse, voxel_feat):
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feat)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([pts_pre_fuse, img_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    # def forward(self, batch_dict, point_features, point_coords, layer_name=None, img_conv_func=None):
    #     """
    #     Generates voxel features via 3D transformation and sampling
    #     Args:
    #         batch_dict:
    #             voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
    #             lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
    #             cam_to_img: (B, 3, 4), Camera projection matrix
    #             image_shape: (B, 2), Image shape [H, W]
    #         encoded_voxel: (N, C), Sparse Voxel featuress
    #     Returns:
    #         batch_dict:
    #             voxel_features: (B, C, Z, Y, X), Image voxel features
    #         voxel_features: (N, C), Sparse Image voxel features
    #     """
    #     voxel_fusefeatlist = []
    #     final_img_voxels = point_features.new_zeros((point_features.shape[0], self.mid_channels))
    #     pts_feats_org = self.pts_key_proj(point_features)
    #     for cam_key in self.image_list:
    #         # Generate sampling grid for frustum volume
    #         projection_dict = self.point_projector(voxel_coords=point_coords.float(),
    #                                                image_scale=self.image_scale[cam_key],
    #                                                batch_dict=batch_dict, 
    #                                                cam_key=cam_key)
    #         batch_size = len(batch_dict['image_shape'][cam_key])
    #         if not self.training and self.double_flip:
    #             tta_ops = batch_dict["tta_ops"]
    #             tta_num = len(tta_ops)
    #             batch_size = batch_size * tta_num
    #         voxel_featlist = []
    #         bs = 1 
    #         for _idx in range(batch_size): #(len(batch_dict['image_shape'][cam_key])):
    #             _idx_key = _idx//tta_num if self.double_flip else _idx
    #             image_feat = batch_dict['image_features'][layer_name+'_feat2d'][cam_key][_idx_key]
    #             if img_conv_func:
    #                 image_feat = img_conv_func(image_feat.unsqueeze(0))[0]
    #             raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
    #             feat_shape = image_feat.shape[-2:]
    #             if self.image_interp:
    #                 image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2], mode='bilinear', align_corners=False)[0]
    #             index_mask = point_coords[:,0]==_idx
    #             voxel_feat = pts_feats_org[index_mask]
    #             image_grid = projection_dict['image_grid'][_idx]
    #             voxel_grid = projection_dict['batch_voxel'][_idx]
    #             point_mask = projection_dict['point_mask'][_idx]
    #             image_depth = projection_dict['image_depths'][_idx]
    #             img_voxels = final_img_voxels[index_mask]
    #             voxel_mask = point_mask[:len(voxel_feat)]               
    #             if self.training and 'overlap_mask' in batch_dict.keys():
    #                 overlap_mask = batch_dict['overlap_mask'][_idx]
    #                 is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
    #                 if 'depth_mask' in batch_dict.keys():
    #                     depth_mask = batch_dict['depth_mask'][_idx]
    #                     depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
    #                     is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
    #                     is_overlap = is_overlap & (~is_inrange)

    #                 image_grid = image_grid[~is_overlap]
    #                 voxel_grid = voxel_grid[~is_overlap]
    #                 point_mask = point_mask[~is_overlap]
    #                 voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
    #             if not self.image_interp:
    #                 image_grid = image_grid.float()
    #                 image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
    #                 image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
    #                 image_grid = image_grid.long()
    #             if image_grid[point_mask].shape[0]>1:
    #                 image_feat = image_feat.unsqueeze(0)
    #                 _, channel_num, h, w = image_feat.shape
    #                 flatten_img_feat = image_feat.permute(0, 2, 3, 1).reshape(1, h * w, channel_num)
    #                 ref_points = image_grid[point_mask].float()
    #                 ref_points[:, 0] /= feat_shape[1]
    #                 ref_points[:, 1] /= feat_shape[0]
    #                 ref_points = ref_points.reshape(bs, -1, 1, 2)
    #                 N, Len_in, _ = flatten_img_feat.shape
    #                 pts_feats = voxel_feat[voxel_mask].reshape(bs, -1, self.mid_channels)
    #                 level_spatial_shapes = pts_feats.new_tensor([(h, w)], dtype=torch.long)
    #                 level_start_index = pts_feats.new_tensor([0], dtype=torch.long)
    #                 img_voxels[voxel_mask] = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index).squeeze(0)
    #             final_img_voxels[index_mask] = img_voxels
    #     final_voxelimg_feat = self.fusion_withdeform(final_img_voxels, point_features)
    #     # import pdb; pdb.set_trace()
    #     return final_voxelimg_feat
        

    def forward(self, batch_dict, point_features, point_coords, layer_name=None, img_conv_func=None):
        """
        카메라별 (N, Cmid) img_voxel을 만든 뒤 채널 concat → (선택)축소/평균 → deform 융합
        """
        N = point_features.shape[0]
        Cmid = self.mid_channels
        K = len(self.image_list)

        pts_feats_org = self.pts_key_proj(point_features)

        cam_voxels_list = []  # 각 카메라별 (N, Cmid)

        for cam_key in self.image_list:
            # 카메라별 결과 버퍼
            img_voxels_cam = point_features.new_zeros((N, Cmid))

            # 투영/그리드 생성
            projection_dict = self.point_projector(
                voxel_coords=point_coords.float(),
                image_scale=self.image_scale[cam_key],
                batch_dict=batch_dict,
                cam_key=cam_key
            )

            batch_size = len(batch_dict['image_shape'][cam_key])
            tta_num = 1
            if (not self.training) and self.double_flip:
                tta_ops = batch_dict["tta_ops"]
                tta_num = len(tta_ops)
                batch_size = batch_size * tta_num

            bs = 1
            for _idx in range(batch_size):
                _idx_key = (_idx // tta_num) if ((not self.training) and self.double_flip) else _idx

                image_feat = batch_dict['image_features'][layer_name + '_feat2d'][cam_key][_idx_key]
                if img_conv_func:
                    image_feat = img_conv_func(image_feat.unsqueeze(0))[0]

                raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
                feat_shape = image_feat.shape[-2:]

                if self.image_interp:
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2],
                                            mode='bilinear', align_corners=False)[0]

                index_mask = (point_coords[:, 0] == _idx)
                voxel_feat = pts_feats_org[index_mask]  # (n_i, Cmid)

                image_grid = projection_dict['image_grid'][_idx]
                voxel_grid = projection_dict['batch_voxel'][_idx]
                point_mask = projection_dict['point_mask'][_idx]          # (m,)
                image_depth = projection_dict['image_depths'][_idx]

                # 현재 배치 인덱스 subset만 편집
                img_voxels_slice = img_voxels_cam[index_mask]             # (n_i, Cmid)

                voxel_mask = point_mask[:len(voxel_feat)]
                if self.training and ('overlap_mask' in batch_dict):
                    overlap_mask = batch_dict['overlap_mask'][_idx]
                    is_overlap = overlap_mask[image_grid[:, 1], image_grid[:, 0]].bool()
                    if 'depth_mask' in batch_dict:
                        depth_mask = batch_dict['depth_mask'][_idx]
                        depth_range = depth_mask[image_grid[:, 1], image_grid[:, 0]]
                        is_inrange = (image_depth > depth_range[:, 0]) & (image_depth < depth_range[:, 1])
                        is_overlap = is_overlap & (~is_inrange)

                    image_grid = image_grid[~is_overlap]
                    voxel_grid = voxel_grid[~is_overlap]
                    point_mask = point_mask[~is_overlap]
                    voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])

                if not self.image_interp:
                    image_grid = image_grid.float()
                    image_grid[:, 0] *= (feat_shape[1] / raw_shape[1])
                    image_grid[:, 1] *= (feat_shape[0] / raw_shape[0])
                    image_grid = image_grid.long()

                if image_grid[point_mask].shape[0] > 1:
                    image_feat_b = image_feat.unsqueeze(0)                 # (1, C, H, W)
                    _, channel_num, h, w = image_feat_b.shape
                    flatten_img_feat = image_feat_b.permute(0, 2, 3, 1).reshape(1, h * w, channel_num)

                    ref_points = image_grid[point_mask].float()
                    ref_points[:, 0] /= feat_shape[1]
                    ref_points[:, 1] /= feat_shape[0]
                    ref_points = ref_points.reshape(bs, -1, 1, 2)          # (1, L, 1, 2)

                    pts_feats = voxel_feat[voxel_mask].reshape(bs, -1, Cmid)
                    level_spatial_shapes = pts_feats.new_tensor([(h, w)], dtype=torch.long)
                    level_start_index = pts_feats.new_tensor([0], dtype=torch.long)

                    fused = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat,
                                            level_spatial_shapes, level_start_index).squeeze(0)  # (L, Cmid)

                    # slice에 반영
                    img_voxels_slice[voxel_mask] = fused

                # 카메라별 전체 버퍼에 반영
                img_voxels_cam[index_mask] = img_voxels_slice

            # 이 카메라 결과 보관
            cam_voxels_list.append(img_voxels_cam)  # (N, Cmid)
        # ----- 카메라별 (N, Cmid) → concat (N, K*Cmid) -----
        fused_img_voxels = torch.stack(cam_voxels_list, dim=1)  # (N, K*Cmid)
        cam_voxels = self.cam_attn(fused_img_voxels)
        fused_img_voxels = cam_voxels.reshape(cam_voxels.shape[0], -1)

        # 최종 deform 융합
        final_voxelimg_feat = self.fusion_withdeform(fused_img_voxels, point_features)
        return final_voxelimg_feat


