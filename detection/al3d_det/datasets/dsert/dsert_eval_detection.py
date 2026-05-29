import argparse
import pickle

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

try:
    # 신형
    from waymo_open_dataset.protos import label_pb2 as _label_mod
except ImportError:
    try:
        # 구형(최상위 경로)
        from waymo_open_dataset import label_pb2 as _label_mod
    except ImportError:
        _label_mod = None  # 정말 구버전이면 아래에서 우회 처리


tf.get_logger().setLevel('INFO')

DEFAULT_IOU_THRESH = {
    'unknown': 0.0,      # Waymo config 상 첫 항목(unknown) 보통 0.0 유지
    'Vehicle': 0.50,
    'Pedestrian': 0.30,
    'Truck': 0.50,
    'Bike': 0.30,        # 리포에 따라 'Cyclist'를 쓰면 아래 매핑에서 처리
}

ALIASES = {
    'Cyclist': 'Bike',
    'cyclist': 'Bike',
    'bike': 'Bike',
    'truck': 'Truck',
    'vehicle': 'Vehicle',
    'pedestrian': 'Pedestrian',
    'unknown': 'unknown',
}

def parse_iou_overrides(s: str):
    """
    'Vehicle=0.6,Pedestrian=0.55,Cyclist=0.55' 형태 문자열을 dict로 파싱
    """
    if not s:
        return {}
    out = {}
    for item in s.split(','):
        if not item.strip():
            continue
        k, v = item.split('=')
        k = k.strip()
        v = float(v.strip())
        # alias 정규화
        k_std = ALIASES.get(k, k)
        out[k_std] = v
    return out

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class WaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Truck', 'Bike']
    def __init__(self, iou_overrides=None):
        super().__init__()
        # 사용자가 준 override를 보관
        self.iou_overrides = iou_overrides or {}

    def generate_waymo_type_results(self, infos, class_names, is_gt=False, fake_gt_infos=True):
        def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
            """
            Args:
                boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

            Returns:
                boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            """
            w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
            boxes3d_lidar[:, 2] += h[:, 0] / 2
            return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
        for frame_index, info in enumerate(infos):
            if is_gt:
                            # 관심 클래스만 필터링
                box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)

                # difficulty를 항상 "박스 수" 길이의 numpy 배열로 보정
                num_all_boxes = len(info['name'])
                if 'difficulty' not in info or info['difficulty'] is None:
                    # 값이 없으면 모두 0으로 초기화
                    info['difficulty'] = np.zeros(num_all_boxes, dtype=np.int32)
                else:
                    info['difficulty'] = np.asarray(info['difficulty'])
                    if info['difficulty'].ndim == 0:
                        # 스칼라였다면 전 박스에 동일값으로 채우기
                        info['difficulty'] = np.full(num_all_boxes, int(info['difficulty']), dtype=np.int32)
                    else:
                        info['difficulty'] = info['difficulty'].astype(np.int32, copy=False)

                # 음수 방지 (옵션)
                info['difficulty'] = np.clip(info['difficulty'], 0, None)

                # difficulty 규칙 적용: 0 -> 1, 1 -> 2 (의도하신 단계 상승으로 보임)
                zero_mask = (info['difficulty'] == 0)
                one_mask  = (info['difficulty'] == 1)
                info['difficulty'][zero_mask] = 1
                info['difficulty'][one_mask]  = 2

                # (옵션) num_points_in_gt 사용할 때는 여기에 추가 마스크 결합
                # if 'num_points_in_gt' in info:
                #     nonzero_mask = info['num_points_in_gt'] > 0
                #     box_mask = box_mask & nonzero_mask

                num_boxes = int(box_mask.sum())
                box_name = info['name'][box_mask]

                difficulty.append(info['difficulty'][box_mask])
                score.append(np.ones(num_boxes, dtype=np.float32))

                if fake_gt_infos:
                    info['gt_boxes_livox'] = boxes3d_kitti_fakelidar_to_lidar(info['gt_boxes_livox'])

                boxes3d.append(info['gt_boxes_livox'][box_mask][:, :7])
            else:
                num_boxes = len(info['boxes_lidar'])
                difficulty.append([0] * num_boxes)
                score.append(info['score'])
                boxes3d.append(np.array(info['boxes_lidar'][:, :7]))
                box_name = info['name']

            obj_type += [self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        box_type: TYPE_3D
        """
        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)
        text_format.Merge(config_text, config)

        # 1) 라벨 enum 나열 (버전 호환)
        label_names_in_order = []
        if _label_mod is not None and hasattr(_label_mod, "Label") and hasattr(_label_mod.Label, "Type"):
            # protobuf enum 안전한 나열 방법: 0부터 Name()이 실패할 때까지 시도
            i = 0
            while True:
                try:
                    name = _label_mod.Label.Type.Name(i)
                except Exception:
                    break
                label_names_in_order.append(name)
                i += 1
        else:
            # 정말로 enum을 못 찾는 희귀 케이스: 가장 흔한 10개로 대체
            # (객체 타입 개수가 다르면 다시 위의 import가 되게 패키지 업데이트 권장)
            label_names_in_order = [
                "TYPE_UNKNOWN",
                "TYPE_VEHICLE",
                "TYPE_PEDESTRIAN",
                "TYPE_SIGN",
                "TYPE_CYCLIST",
                "TYPE_MOTORCYCLIST",
                "TYPE_BICYCLIST",
                "TYPE_BUS",
                "TYPE_TRAILER",
                "TYPE_TRUCK",
            ]

        # 2) 기본 IoU 규칙
        def default_thr_for(label_name: str) -> float:
            n = label_name.upper()
            if "UNKNOWN" in n:
                return 0.0
            if any(k in n for k in ["VEHICLE", "TRUCK", "BUS", "TRAILER"]):
                # return 0.70
                return 0.50
            if "PEDESTRIAN" in n:
                # return 0.50
                return 0.30
            if any(k in n for k in ["CYCLIST", "BICYCLIST", "MOTORCYCLIST", "BIKE"]):
                # return 0.50
                return 0.30
            if "SIGN" in n:
                return 0.50
            return 0.50

        # 3) (옵션) 사용자 override 반영
        def normalize_alias(name: str) -> str:
            # 'Bike'/'Cyclist' 혼용 대응
            if name.lower() == "cyclist":
                return "Bike"
            return name

        overrides = getattr(self, "iou_overrides", {}) or {}

        # 4) 라벨 개수만큼 임계치 push
        for name in label_names_in_order:
            thr = default_thr_for(name)
            if overrides:
                # exact name 우선
                if name in overrides:
                    thr = float(overrides[name])
                else:
                    alias = normalize_alias(name)
                    if alias in overrides:
                        thr = float(overrides[alias])
            config.iou_thresholds.append(thr)

        return config

    def build_graph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
        self,
        sess,
        graph,
        metrics,
        prediction_frame_id,
        prediction_bbox,
        prediction_type,
        prediction_score,
        prediction_overlap_nlz,
        ground_truth_frame_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        # mask = np.linalg.norm(boxes_3d[:, 0:3], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def mask_by_fov(self, boxes_3d, *args):
        root = np.sqrt(np.square(boxes_3d[:, 0]) + np.square(boxes_3d[:, 1]))
        sin_a, cos_a = boxes_3d[:, 0] / root, -boxes_3d[:, 1] / root
        mask = sin_a < -np.sin(np.pi/180*35) & np.abs(cos_a) < np.cos(np.pi/180*35)
        mask = (1 - mask) > 0
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])
        
        return tuple(ret_ans)

    def waymo_evaluation(self, prediction_infos, gt_infos, class_name, distance_thresh=100, fake_gt_infos=True, fov_flag=False):
        print('Start the waymo evaluation...')
        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        tf.compat.v1.disable_eager_execution()
        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = self.generate_waymo_type_results(
            prediction_infos, class_name, is_gt=False
        )
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = self.generate_waymo_type_results(
            gt_infos, class_name, is_gt=True, fake_gt_infos=fake_gt_infos
        )

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_distance(
            distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
        )
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = self.mask_by_distance(
            distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty
        )

        if fov_flag:
            pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_fov(
                pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
            )
            gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = self.mask_by_fov(
                gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty
            )

        print('Number: (pd, %d) VS. (gt, %d)' % (len(pd_boxes3d), len(gt_boxes3d)))
        print('Level 1: %d, Level2: %d)' % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum()))

        if pd_score.max() > 1:
            # assert pd_score.max() <= 1.0, 'Waymo evaluation only supports normalized scores'
            pd_score = 1 / (1 + np.exp(-pd_score))
            print('Warning: Waymo evaluation only supports normalized scores')

        graph = tf.Graph()
        metrics = self.build_graph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess, graph, metrics, pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz,
                gt_frameid, gt_boxes3d, gt_type, gt_difficulty,
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)
        return aps


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=5, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results...')
    eval = WaymoDetectionMetricsEstimator()

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        gt_infos_dst.append(cur_info)

    waymo_AP = eval.waymo_evaluation(
        pred_infos, gt_infos_dst, class_name=args.class_names, distance_thresh=1000, fake_gt_infos=True
    )

    print(waymo_AP)


if __name__ == '__main__':
    main()
