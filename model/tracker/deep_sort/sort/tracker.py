# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is thet multi-arget tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """
    # 칼만필터를 호출하여 트랙의 새 상태 예측 + 프레임 초기화 담당
    # Tracker 클래스의 update 나 predict 메소드가 호츨되면,
    # 각각의 Track 에서 자체의 update 또는 predict 메소드가 호출됨

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        # 호출되면, 하위 파라미터들은 모두 디폴트
        self.metric = metric
        # metric 은 distance 를 계산하는데 사용되는 클래스 (코사인 거리, 마할라노비스 거리)
        self.max_iou_distance = max_iou_distance
        # Maximum iou, iou matches 를 사용할 때 이용
        self.max_age = max_age
        # cascade maching 의 cascade_depth 매개변수
        self.n_init = n_init
        # n_init은 트랙의 상태를 confirmed 로 설정하기 위해 필요한 update 시간

        self.kf = kalman_filter.KalmanFilter() # 칼만필터
        self.tracks = [] # track 들을 저장
        self._next_id = 1 # 다음 할당될 track id

    def predict(self):
        # track 횡단과 예측 생성
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        # 1. matching 결과용
        for track_idx, detection_idx in matches:
            # 관측값에 따른 track 업데이트
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 2. matching 되지 않음 tracker 를 위한 'mark_missed' 마크 호출
        # 잘못 maching 된 track , 보류중인경우 삭제, 업데이트 시간이 너무 긴 경우 삭제
        # max age 는 수명, 디폴트 70 frame
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 3. matching 되지 않은 관측과 잘못 matching 된 관측, 초기화
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # 최근 track 리스트를 가져오며, confirmed 와 Tentative 로 마크된 track 저장
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # confirmed 된 상태의 모든 track id 가져옴
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features # tracks list 와 features list 결합
            # 각각의 feature 에 상응하는 track 가져옴
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 거리 측정에서의 feature set update
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # match 를 위한 메인 함수, match 된 부분과 안된 부분을 찾는다
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # Function : track 과 detection 사이의 거리를 계산하기 위해 사용된다, cost function
            # KM algorithm 사용전에 사용해야함
            # transfer:
            # cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # 1. nearest neighbor 을 통한 코사인거리 cost matrix 계산
            cost_matrix = self.metric.distance(features, targets)
            # 2. 새로운 상태에 대한 마할라노비스 거리 계산
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 여러 트랙의 상태를 나눔 (confirmed <-> unconfirmed)
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # matched track, unmatched track, unmatched detection 을 얻기위해 cascading matching 을 수행한다

        '''
        !!!!!!!!!!
            Cascade matching
        !!!!!!!!!!
        '''

        # Associate confirmed tracks using appearance features.
        # gated_metric -> cosine distance
        # 특정 trajectories 에 대한 cascade matching 만
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # 상태가 확인되지 않은 모든 트랙과 당장 일치하지 않은 트랙을 iou_track_candidates 에 결합
        # IoU matching 수행
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1   # 일치하지 않음
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1   # 오랫동안 일치하지 않음
        ]

        '''
        !!!!!!!!!!
            IoU matching
            IoU matching is performed on the targets that have not been successfully matched in the cascade matching
        !!!!!!!!!!
        '''

        # cascade matching 에서 min_cost_matching 이 코어로 사용되지만,
        # 여기서 사용되는 metric 은 위와 상이하다
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # 두 부분의 match 에 대한 결과물을 합친다
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        class_id = detection.class_id
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, class_id,
            detection.feature,))
        self._next_id += 1
