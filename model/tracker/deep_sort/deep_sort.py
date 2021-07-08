import numpy as np
import torch
from model.tracker.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from model.tracker.deep_sort.sort.preprocessing import non_max_suppression
from model.tracker.deep_sort.sort.detection import Detection
from model.tracker.deep_sort.sort.tracker import Tracker


class DeepSort(object):
    def __init__(self, extractor, max_dist=0.2):
        # detector 로 도출된 결과에 대한 신뢰 임계값, 신뢰점수 0.3 이하를 필터링
        self.min_confidence = 0.3

        # Non-maximum suppression 임계값, 1로 set 한다는 것은 suppression 하지 않겠다는 뜻
        self.nms_max_overlap = 1.0

        # 이미지를 임베딩해 feature 를 추출할 extractor 정의
        self.extractor = extractor

        # max_dist (변화 임계점) 보다 크면 무시, cascade matching 에서 사용
        max_cosine_distance = max_dist
        # 각각의 범주에 대한 최대 샘플 개수, 해당 숫자를 넘어가게 되면, 오래된 것부터 삭제
        nn_budget = 100
        # 첫번째 파라미터는 'cosine' 혹은 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, img, boxes, classes, scores):
        features = self._get_features(img, boxes)
        # 관찰값 생성
        self.height, self.width = img.shape[:2]

        boxes = boxes.to('cpu')
        scores = scores.to('cpu')
        classes = classes.to('cpu')

        # 원본 이미지에서, bounding box 를 잘라옴
        bbox_tlwh = self._xyxy_to_tlwh(boxes)

        bbox_tlwh = bbox_tlwh.numpy()
        confidences = scores.numpy()
        classes = classes.numpy()
        features = features.numpy()

        # generate detections
        # min_confidence 보다 작은 대상 필터링 및 탐지 개체 목록 구성
        # Detection : graph 형태의 bbox 결과 저장고
        # 필요 : 1. bbox(tlwh 형태), 2. 해당 신뢰도, 3. 해당 임베딩
        detections = [Detection(bbox_tlwh[i], classes[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # Use non-maximum suppression
        # 디폴트 값인 nms_thres=1 이 적용된다면, 이 부분은 작동하지 않음
        # 즉, non-maximum suppression 적용하지 않음
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores) #  return pick
        detections = [detections[i] for i in indices]

        # update tracker
        # tracker 가 예측 결과를 제공한 다음 kalman 필터 작동을 위해 detection 안으로 들어간다.
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        # 결과를 저장하고 시각화
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        bbox_tlwh = torch.zeros(bbox_xyxy.shape, dtype=torch.float32)
        # bbox_tlwh[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2.0
        # bbox_tlwh[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2.0

        bbox_tlwh[:, 0] = bbox_xyxy[:, 0]
        bbox_tlwh[:, 1] = bbox_xyxy[:, 1]
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

        return bbox_tlwh

    def _get_features(self, img, boxes):
        features = self.extractor.feature_extraction(img, boxes)
        return features
