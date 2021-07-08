# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # 쌍의 거리 제곱을 계산하는데 사용됨
    # a NxM 은 N개의 객체를 나타내며, 각 객체는 비교를 위한 임베딩으로 M 값을 가진다
    # b LxM 은 L개의 객체를 나타내며, 각 객체는 비교를 위한 임베딩으로 M 값을 가진다
    # NxL 행렬이 반환되는데, 예를들어 dist[i][j] 는 a[i] 와 b[j] 사이의 거리 제곱 합을 나타낸다

    a, b = np.asarray(a), np.asarray(b) # 데이터 복사본 복사
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1) # 각각의 임베딩값에 제곱을 구한다
    # sum(N) + sum(L) -2 x [NxM]x[MxL] = [NxL]
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        # 코사인 유사성을 유클리드 거리와 유사한 코사인 거리로 변환할 필요가 있다
        # np.linalg.norm 는 norm 형태의 vector를 반환하는 연산자로,
        # 디폴트는 벡터의 유클리드 거리를 찾는것과 같은 L2 norm 이다
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        # 각각의 대상에 대해 가장 가까운 거리 반환

        if metric == "euclidean":
            # 유클리디안 거리를 이용한 최근접이웃
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            # 코사인 유사도를 이용한 최근접이웃
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        # cascade matching 을 위한 함수 호출
        self.budget = budget
        # budget, feature 의 숫자를 조절
        self.samples = {}
        # sample 은 {id -> feature list} 구조의 사전

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        # 호출 : feature 집합 업데이트를 위한 모듈 부분을 호출한다, tracker.update
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            # 상응하는 target 에 대해 새로운 feature 를 추출하고, feature collection 을 업데이트 한다
            # Target id : feature list
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
            # 각 범주의 최대 목표 수인 budget 설정, 초과할 경우 직접적으로 무시

        # activated targets 을 필터링링
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        # 호출 : matching 단계에서 distance 를 gated_metric 으로 캡슐화,
        # appearance information (re-id 를 통해 얻어진 depth feature)
        #  +
        # Motion information (마할라노비스 거리는 두 분포의 유사성을 측정하는데 사용)
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
