from functools import partial

import spconv.pytorch as spconv
import torch.nn as nn


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

# VoxelResBackBone8x
class Backbone3D_align(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        channels = getattr(model_cfg, "CHANNELS", [16, 32, 64, 128])
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(channels[3], channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(channels[3]),
            nn.ReLU(),
        )
        self.num_point_features = channels[3]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


# VoxelResBackBone8x
class Backbone3D_align_radar(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        channels = getattr(model_cfg, "CHANNELS", [16, 32, 64, 128])
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(channels[3], channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(channels[3]),
            nn.ReLU(),
        )
        self.num_point_features = channels[3]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features_radar'], batch_dict['voxel_coords_radar']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        batch_dict.update({
            'encoded_spconv_tensor_radar': out,
            'encoded_spconv_tensor_stride_radar': 8
        })
        batch_dict.update({
            'multi_scale_3d_features_radar': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides_radar': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict



# VoxelResBackBone8x
class Backbone3D_align_radar_only(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        channels = getattr(model_cfg, "CHANNELS", [16, 32, 64, 128])
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(channels[3], channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(channels[3]),
            nn.ReLU(),
        )
        self.num_point_features = channels[3]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features_radar'], batch_dict['voxel_coords_radar']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        batch_dict.update({
            'encoded_spconv_tensor_radar': out,
            'encoded_spconv_tensor_stride_radar': 8
        })
        batch_dict.update({
            'multi_scale_3d_features_radar': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides_radar': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict

import torch
import torch.nn as nn
import spconv.pytorch as spconv

def _hash_idx(idx, Z, Y, X):
    b, z, y, x = idx.unbind(dim=1)
    return (((b * Z + z) * Y + y) * X + x).long()

@torch.no_grad()
def _union_gather(x1: spconv.SparseConvTensor,
                  x2: spconv.SparseConvTensor,
                  fill: float = 0.0):
    """좌표 합집합 인덱스와, 합집합에 정렬된 f1,f2(제로패딩) 리턴"""
    assert x1.spatial_shape == x2.spatial_shape and x1.batch_size == x2.batch_size
    Z, Y, X = map(int, x1.spatial_shape)
    dev = x1.features.device
    dt  = x1.features.dtype

    idx1, idx2 = x1.indices.int(), x2.indices.int()
    h1, h2 = _hash_idx(idx1, Z, Y, X), _hash_idx(idx2, Z, Y, X)

    all_h = torch.cat([h1, h2], 0)
    uniq_h, inv = torch.unique(all_h, sorted=False, return_inverse=True)
    n1 = h1.numel()
    inv1, inv2 = inv[:n1], inv[n1:]

    U = uniq_h.numel()
    C1, C2 = x1.features.shape[1], x2.features.shape[1]

    f1 = torch.full((U, C1), fill, device=dev, dtype=dt)
    f2 = torch.full((U, C2), fill, device=dev, dtype=dt)
    f1.index_add_(0, inv1, x1.features)
    f2.index_add_(0, inv2, x2.features)

    # 해시 -> 좌표, 정렬
    b = uniq_h // (Z*Y*X); rem = uniq_h % (Z*Y*X)
    z = rem // (Y*X);       rem = rem % (Y*X)
    y = rem // X;           x = rem % X
    merged_idx = torch.stack([b, z, y, x], dim=1).int()
    order = (b * (Z*Y*X) + z*(Y*X) + y*X + x)
    perm = torch.argsort(order)

    return merged_idx[perm], f1[perm], f2[perm]

class _PerLevelProjector(nn.Module):
    """키(스케일)별로 2C->C 선형투영을 lazy하게 만들어 캐시."""
    def __init__(self):
        super().__init__()
        self.fc = nn.ModuleDict()

    def forward(self, key: str, feats: torch.Tensor, out_c: int):
        in_c = feats.shape[1]
        if key not in self.fc:
            self.fc[key] = nn.Linear(in_c, out_c, bias=False)
        return self.fc[key](feats)

class LidarRadarUnion(nn.Module):
    """
    LiDAR / Radar sparse 텐서를 좌표 합집합 기준으로 융합.
    - FUSE_OP: 'cat' | 'sum'  (cat 시 2C->C 투영)
    - FUSE_KEYS: 융합할 스케일 키 리스트
    """
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.fuse_op   = getattr(model_cfg, 'FUSE_OP', 'cat')  # 'cat' or 'sum'
        self.fill      = float(getattr(model_cfg, 'FILL_VALUE', 0.0))
        # 어떤 스케일들을 융합할지 (백본에서 쓰는 키들)
        self.fuse_keys = getattr(model_cfg, 'FUSE_KEYS',
                                 ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'out'])
        # cat일 때 2C->C 투영기
        # self.projector = _PerLevelProjector() if self.fuse_op == 'cat' else None
        # channels = [16, 32, 64, 128, 128]
        # channels = self.model_cfg.c
        channels = getattr(model_cfg, "CHANNELS", [16, 32, 64, 128, 128])
        self.projector_0 = nn.Linear(channels[0]*2, channels[0], bias=False)
        self.projector_1 = nn.Linear(channels[1]*2, channels[1], bias=False)
        self.projector_2 = nn.Linear(channels[2]*2, channels[2], bias=False)
        self.projector_3 = nn.Linear(channels[3]*2, channels[3], bias=False)
        self.projector_4 = nn.Linear(channels[4]*2, channels[4], bias=False)
        self.projector_list = [self.projector_0, self.projector_1, self.projector_2,
                            self.projector_3, self.projector_4]

    def _fuse_pair(self, a: spconv.SparseConvTensor,
                         b: spconv.SparseConvTensor,
                         key: str,
                         index: int) -> spconv.SparseConvTensor:
        # dtype/device 정렬
        if a.features.dtype != b.features.dtype:
            b = b.replace_feature(b.features.to(a.features.dtype))
        if a.features.device != b.features.device:
            b = b.replace_feature(b.features.to(a.features.device))

        idx, f1, f2 = _union_gather(a, b, fill=self.fill)
        if self.fuse_op == 'sum':
            fused = f1
            fused.add_(f2)  # sum
        elif self.fuse_op == 'cat':
            fused = torch.cat([f1, f2], dim=1)
            # 투영으로 채널 복원(기본: LiDAR 채널로)
            # for projection in self.projector_list:
            fused = self.projector_list[index](fused)
        else:
            raise ValueError(f"Unknown FUSE_OP: {self.fuse_op}")

        return spconv.SparseConvTensor(
            features=fused, indices=idx,
            spatial_shape=a.spatial_shape, batch_size=a.batch_size
        )

    def forward(self, batch_dict):
        # 필수 입력 체크
        assert 'multi_scale_3d_features' in batch_dict and 'multi_scale_3d_features_radar' in batch_dict, \
            "multi_scale_3d_features(_radar)가 필요합니다."
        ms_l = batch_dict['multi_scale_3d_features']
        ms_r = batch_dict['multi_scale_3d_features_radar']

        fused_ms = {}
        # 스케일별 융합
        index = 0
        for k in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
            if k in self.fuse_keys:
                fused_ms[k] = self._fuse_pair(ms_l[k], ms_r[k], key=k, index= index)
            else:
                # 융합 안 하면 LiDAR 그대로
                fused_ms[k] = ms_l[k]
            index += 1

        # 최종 out도 융합
        out_l = batch_dict['encoded_spconv_tensor']
        out_r = batch_dict['encoded_spconv_tensor_radar']
        fused_out = self._fuse_pair(out_l, out_r, key='out', index=index) if 'out' in self.fuse_keys else out_l

        # 배치딕트 업데이트 (fused를 기본으로 사용하도록 교체)
        batch_dict.update({
            'encoded_spconv_tensor': fused_out,
            'encoded_spconv_tensor_stride': batch_dict.get('encoded_spconv_tensor_stride', 8),
            'multi_scale_3d_features': fused_ms,
            'multi_scale_3d_strides': batch_dict.get('multi_scale_3d_strides', {
                'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8
            })
        })
        # 다운스트림이 기존 키만 참조해도 동작하도록 기본 키를 fused로 덮어쓰기(원본 유지 원하면 이 두 줄은 빼도 됨)
        batch_dict['encoded_spconv_tensor'] = fused_out
        batch_dict['multi_scale_3d_features'] = fused_ms
        
        return batch_dict



class LidarRadarUnionPass(nn.Module):
    """
    LiDAR / Radar sparse 텐서를 좌표 합집합 기준으로 융합.
    - FUSE_OP: 'cat' | 'sum'  (cat 시 2C->C 투영)
    - FUSE_KEYS: 융합할 스케일 키 리스트
    """
    def __init__(self, model_cfg, **kwargs):
        super().__init__()


    def forward(self, batch_dict):
        
        return batch_dict






# VoxelResBackBone8x_2xchannels
class Backbone3D_align2X(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        channels = getattr(model_cfg, "CHANNELS", [32, 64, 128, 256])
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(channels[2], channels[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(channels[3], channels[3], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(channels[3]),
            nn.ReLU(),
        )
        self.num_point_features = channels[3]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict