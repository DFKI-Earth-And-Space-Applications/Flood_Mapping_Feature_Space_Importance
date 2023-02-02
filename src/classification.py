import json
import pickle
import sys
from typing import Optional, Union, cast, List

import lightgbm
import numba
import numpy as np
import sklearn.base
import faiss

import utils


# the Thresholding implementations are almost completely copied from Landuyt et al.
# Code at https://github.com/llanduyt/floodOBIA/blob/f85227655738d97bccc9b7cca078094d80f8829c/Modules/TiledThresholdingFunctions.py

class PerImageThresholdingClassifier(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, use_tiled: bool = True, tile_dim: int = 256, bin_count: Optional[Union[int, str]] = None,
                 n_final=5,
                 percentile: float = 95., histogram_merge_threshold: float = 5., min_tile_dim: int = 16,
                 force_merge_on_failure: bool = False, print_warning=True) -> None:
        super().__init__()
        assert 0 < percentile < 100
        self.min_tile_dim = min_tile_dim
        self.force_merge_on_failure: bool = force_merge_on_failure
        self.use_tiled: bool = use_tiled
        self.tile_dim: int = tile_dim
        self.bin_count: Union[int, str] = 'tile_dim' if bin_count is None else bin_count
        self.n_final: int = n_final
        self.percentile: float = percentile
        self.histogram_merge_threshold: float = histogram_merge_threshold
        self.print_warning = print_warning
        self.threshold = None

    def compute_threshold(self, image: np.ndarray, bin_count: Optional[Union[int, str]]) -> np.ndarray:
        raise NotImplementedError

    def tile_vars(self, image, tile_dim=(256, 256), bin_count: Optional[Union[int, str]] = 200,
                  incomplete_tile_warning=True):
        """ Calculate tile variables

        Inputs:
        image: nd array
            Array of pixel values.
        tile_dim: list (default=[200, 200])
            Dimension of tiles.
        hand_matrix: ndarray or None (default=None)
            Array of HAND values.
        incomplete_tile_warning: bool (default=True)
            Whether to give a warning when incomplete tiles are encountered.
        Outputs:
        tile_ki: nd array
            Array of KI thresholds.
        tile_o: nd array
            Array of Otsu thresholds.
        average: nd array
            Array of tile averages.
        stdev: nd array
            Array of tiled st. devs.
        hand: nd array
            Array of tile mean HAND values.
        """
        tile_rows, tile_cols = tile_dim
        if type(bin_count) is str and bin_count == 'tile_dim':
            bin_count = min(tile_rows, tile_cols)
        nrt = np.ceil(image.shape[0] / tile_rows).astype('int')
        nct = np.ceil(image.shape[1] / tile_cols).astype('int')
        tile_threshold = np.full([nrt, nct], np.nan)
        stdev = np.full([nrt, nct], np.nan)
        average = np.full([nrt, nct], np.nan)
        for r in np.arange(0, image.shape[0], tile_rows):
            tile_rindex = np.floor(r / tile_rows).astype('int')
            for c in np.arange(0, image.shape[1], tile_cols):
                tile_cindex = np.floor(c / tile_cols).astype('int')
                tile = image[r:min(r + tile_rows, image.shape[0]), c:min(c + tile_cols, image.shape[1])]
                if np.sum(np.isnan(tile)) <= 0.1 * np.size(tile):
                    tile_threshold[tile_rindex, tile_cindex] = self.compute_threshold(tile, bin_count=bin_count)
                    tr, tc = tile.shape
                    mu1 = np.nanmean(tile[0:tr // 2, 0:tc // 2])
                    mu2 = np.nanmean(tile[0:tr // 2, tc // 2:])
                    mu3 = np.nanmean(tile[tr // 2:, 0:tc // 2])
                    mu4 = np.nanmean(tile[tr // 2:, tc // 2:])
                    stdev[tile_rindex, tile_cindex] = np.nanstd([mu1, mu2, mu3, mu4])
                    average[tile_rindex, tile_cindex] = np.nanmean(tile)
                elif incomplete_tile_warning:
                    print("Tile ({0:.0f}, {1:.0f}) is incomplete.".format(tile_rindex, tile_cindex), file=sys.stderr)
        return tile_threshold, average, stdev, None  # Modify this line to include other selection methods!

    def tiled_thresholding(self, image, tile_dim=(256, 256), bin_count: Optional[Union[int, str]] = 200, n_final=5,
                           percentile: float = 95., histogram_merge_threshold: float = 5.,
                           print_warning=True):
        """ Apply tiled thresholding

        Inputs:
        image: nd array
            Array of pixel values.
        selection: str (default='Martinis')
            Method for tile selection. Currently only option is 'Martinis'.
        t_method: list (default=['KI', 'Otsu'])
            List of thresholds to calculate. Should contain one or both of "KI", "Otsu".
        tile_dim: list (default=[200, 200])
            Dimension of tiles.
        n_final: int (default=5)
            Number of tiles to select.
        incomplete_tile_warning: bool (default=True)
            Whether to give a warning when incomplete tiles are encountered.
        Outputs:
        a threshold computed in a tiled way
        """
        tile_dim = np.array(tile_dim)
        # Tile properties
        tile_threshold, average, stdev, hand = self.tile_vars(image, tile_dim=tile_dim, bin_count=bin_count,
                                                              incomplete_tile_warning=print_warning)
        # Tile selection
        q = np.nanpercentile(stdev, percentile)
        stdev[average > np.nanmean(average)] = np.nan
        i_r, i_c = np.nonzero(stdev > q)  # select tiles with stdev > 95-percentile
        # differ from Landuyt paper in that we don't allow tiles with less than 4 elements...
        i = 0
        while len(i_r) == 0 and np.all(tile_dim > self.min_tile_dim):
            tile_dim = tile_dim // 2
            i += 1
            if print_warning:
                print(f'Tile dimensions have been halved {i}-time' + ('' if i == 1 else 's'), file=sys.stderr)
            tile_threshold, average, stdev, hand = self.tile_vars(image, tile_dim=tile_dim, bin_count=bin_count,
                                                                  incomplete_tile_warning=print_warning)
            q = np.percentile(stdev, percentile)
            i_r, i_c = np.nonzero(stdev > q)
        force_merge = False
        # this also means we have to consider all tiles as eligible for selection
        if len(i_r) == 0:
            if print_warning:
                print(f'Tiled thresholding terminated after {i}-iteration(s) but could not find appropriate tiles',
                      file=sys.stderr)
            force_merge = self.force_merge_on_failure
            i_r = np.arange(0, stdev.shape[0])
            i_c = np.arange(0, stdev.shape[0])
        sorted_indices = np.argsort(stdev[i_r, i_c])[::-1]
        i_r = i_r[sorted_indices]
        i_c = i_c[sorted_indices]
        if i_r.size > n_final:
            tile_selection = (i_r[:n_final], i_c[:n_final])
        else:
            tile_selection = (i_r, i_c)
        # Threshold and quality indicator
        res_threshold = np.mean(tile_threshold[tile_selection])
        if force_merge or np.std(tile_threshold[tile_selection]) > histogram_merge_threshold:
            if print_warning:
                print('Histogram merge necessary for ' + str(type(self)) + '.', file=sys.stderr)
            pixel_selection = np.empty(0)
            for tile_rindex, tile_cindex in zip(tile_selection[0], tile_selection[1]):
                tile = image[tile_rindex * tile_dim[0]:min((tile_rindex + 1) * tile_dim[0], image.shape[0]),
                       tile_cindex * tile_dim[1]:min((tile_cindex + 1) * tile_dim[1], image.shape[1])]
                pixel_selection = np.append(pixel_selection, tile.ravel())
                del tile
            res_threshold = self.compute_threshold(pixel_selection, bin_count=bin_count)
        return res_threshold

    def do_thresholding(self, data: np.ndarray) -> np.ndarray:
        if self.use_tiled:
            return self.tiled_thresholding(data, tile_dim=(self.tile_dim, self.tile_dim),
                                           bin_count=self.bin_count, n_final=self.n_final,
                                           percentile=self.percentile,
                                           histogram_merge_threshold=self.histogram_merge_threshold,
                                           print_warning=self.print_warning)
        else:
            bin_count = self.tile_dim if self.bin_count == 'tile_dim' else self.bin_count
            return self.compute_threshold(data, bin_count=bin_count)

    def fit(self, data: np.ndarray):
        data = np.squeeze(data)
        if data.ndim < 2 or 3 < data.ndim:
            raise ValueError('Cannot handle data with less than 2 or more than 3 dimensions!!')
        elif data.ndim == 3 and data.shape[2] != 1:
            raise ValueError('Found 3-dimensional array, for which the shape (H, W, C) could be assumed. However'
                             'as the channel order may vary between fit and predict, this is not supported. Instead'
                             'fit once per channel and ensure channel consistency via meta.channel_names!')
            # if self.print_warning:
            #     print('Found 3-dimensional array: assuming shape (H, W, C)')
            # self.threshold = np.squeeze(np.array([self.do_thresholding(data[:, i]) for i in range(data.shape[2])]))
        else:
            assert data.ndim != 3 or data.shape[2] != 1  # np.squeeze should prevent this...
            # it remains a 2-dimensional array, for which calculating a threshold is straight forward
            self.threshold = self.do_thresholding(data)

    def predict(self, data: np.ndarray):
        prev_shape = data.shape
        data = np.squeeze(data)
        if self.threshold is None:
            raise RuntimeError('Cannot predict without a defined threshold!!!')
        if len(data.shape) > 2:
            raise ValueError('Expected data with at most two dimensions! (H, W) or (N)/(N, 1) are '
                             'valid shapes. However given shape is ' + str(data.shape) + '!')
        data = data.reshape(-1)
        res = np.zeros(data.shape, dtype=np.uint8)
        res[data < self.threshold] = 1
        return res.reshape(prev_shape)


class KIPerImageThresholdingClassifier(PerImageThresholdingClassifier):
    def compute_threshold(self, image: np.ndarray, bin_count: Optional[Union[int, str]]) -> np.ndarray:
        """Select threshold according to Kittler & Illingworth

            Inputs:
            image: nd array
                Array of pixel values.
            bin_count: int(default=200)
                Number of bins to construct histogram.
            Outputs:
            t: float
                Threshold value
            """
        # Histogram
        h, bin_edges = np.histogram(image[~np.isnan(image)], bins=bin_count, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        g = np.arange(bin_edges[0] + bin_width / 2.0, bin_edges[-1], bin_width)
        g_pos = g - np.min(g)
        g01 = g_pos / np.max(g_pos)

        # Cost function and threshold
        c = np.cumsum(h)
        m = np.cumsum(h * g01)
        s = np.cumsum(h * g01 ** 2)
        cb = c[-1] - c
        mb = m[-1] - m
        sb = s[-1] - s
        c[c == 0] = 1e-9
        cb[cb == 0] = 1e-9
        var_f = s / c - (m / c) ** 2
        if np.any(var_f < 0):
            var_f[var_f < 0] = 0
        sigma_f = np.sqrt(var_f)
        var_b = sb / cb - (mb / cb) ** 2
        if np.any(var_b < 0):
            var_b[var_b < 0] = 0
        sigma_b = np.sqrt(var_b)
        p = c / c[-1]
        sigma_f[sigma_f == 0] = 1e-9
        sigma_b[sigma_b == 0] = 1e-9
        # originally this was
        # j = p * np.log(sigma_f) + (1 - p) * np.log(sigma_b) - p * np.log(p) - (1 - p) * np.log(1 - p + 1e-9)
        # however, what if p is zero in the np.log(p) term ? => add a small constant
        j = p * np.log(sigma_f) + (1 - p) * np.log(sigma_b) - p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
        j[~np.isfinite(j)] = np.nan
        if np.all(np.isnan(j)):
            return g[g.shape[0]//2]
        idx = np.nanargmin(j)
        t = g[idx]
        # Return
        return t


class OtsuPerImageThresholdingClassifier(PerImageThresholdingClassifier):
    def compute_threshold(self, image: np.ndarray, bin_count: Optional[Union[int, str]]) -> np.ndarray:
        """Select threshold according to Otsu

            Inputs:
            image: nd array
                Array of pixel values.
            accuracy: int(default=200)
                Number of bins to construct histogram.
            plot_j: bool (default=False)
                Whether to plot the cost function
            Outputs:
            t: float
                Threshold value
            """
        # Histogram
        h, bin_edges = np.histogram(image[~np.isnan(image)], bins=bin_count, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        g = np.arange(bin_edges[0] + bin_width / 2.0, bin_edges[-1], bin_width)
        # Between class variance and threshold
        w1 = np.cumsum(h)
        w2 = w1[-1] - w1
        w2[w2 == 0] = 1e-9
        gh = np.cumsum(g * h)
        mu1 = gh / w1
        mu2 = (gh[-1] - gh) / w2
        var_between = w1 * w2 * (mu1 - mu2) ** 2
        idx = np.nanargmax(var_between)
        t = g[idx]
        # Return
        return t


class BitwiseAndChannelCombiner(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, reduction_axis: int = 1):
        self.reduction_axis = reduction_axis

    def fit(self, data: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        if data.ndim <= 1:
            print(
                f'Bitwise and of an array with only one channel (shape is {str(data.shape)}) will just be the input.' +
                ' This is likely not intentional!', file=sys.stderr)
            return data
        data = utils.move_channel_to_position(data, 0)
        res = np.bitwise_and.reduce(data, axis=self.reduction_axis)
        return res


class BitwiseOrChannelCombiner(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, reduction_axis: int = 1):
        self.reduction_axis = reduction_axis

    def fit(self, data: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        if data.ndim <= 1:
            print(
                f'Bitwise or of an array with only one channel (shape is {str(data.shape)}) will just be the input.' +
                ' This is likely not intentional!', file=sys.stderr)
            return data
        data = utils.move_channel_to_position(data, 0)
        res = np.bitwise_or.reduce(data, axis=self.reduction_axis)
        return res

_METRICS_MAP = {
    'L2': 'L2',
    'cosine': 'cosine',
    'euclidean': 'L2'
}
class FaissKNNClassifier(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, k: int, use_gpu: bool, metric: str = 'L2', weights: Optional[str] = None,
                 use_index: bool = False, nprobe: int = 64, pred_batch_size: int = 16384):
        assert weights is None or weights in ['uniform','sklearn', 'direct']
        assert metric in _METRICS_MAP
        if weights == 'uniform':
            weights = None
        self.k = k
        self.use_gpu = use_gpu
        self.index: Optional[faiss.Index] = None
        self.labels: Optional[np.ndarray] = None
        self.offset: Optional[int] = None
        self.weights = weights
        self.metric = _METRICS_MAP[metric]
        self.use_index = use_index
        self.nprobe = nprobe
        self.pred_batch_size = pred_batch_size

    def fit(self, data: np.ndarray, labels: np.ndarray):
        assert data.shape[0] == labels.shape[0], f'Labels and data should have the same number of elements. ' \
                                                 f'Got data{data.shape} and labels{labels.shape}!'
        data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        if self.use_gpu:
            gpu_resource = faiss.StandardGpuResources()

            if self.use_index:
                k = min(int(4*np.round(np.sqrt(data.shape[0]))), data.shape[0]//30)
                if k > 2048:
                    print(f'There is so much data, that it was attempted to use more than 2048 as k ({k}). '
                          'This is not supported by faiss on the gpu, therefore k will be set to 2048.')
                    k = 2048
                #self.index = faiss.index_factory(data.shape[1], f'IVF{k},Flat')
                self.index = faiss.GpuIndexIVFFlat(gpu_resource, data.shape[1], k,  faiss.METRIC_INNER_PRODUCT)#faiss.index_cpu_to_gpu(gpu_resource, 0, self.index)
                if self.nprobe > 2048:
                    print(f'Attempted to use nprobe > 2048 ({self.nprobe}). '
                          'This is not supported by faiss on the gpu, therefore nprobe will be set to 2048.')
                    self.index.nprobe = 2048
                else:
                    self.index.nprobe = self.nprobe
            else:
                if self.metric == 'cosine':
                    faiss.normalize_L2(data)
                    self.index = faiss.GpuIndexFlatIP(gpu_resource, data.shape[1])
                else:
                    self.index = faiss.GpuIndexFlatL2(gpu_resource, data.shape[1])
        else:
            if self.use_index:
                self.index = faiss.IndexHNSWFlat(data.shape[1])
            else:
                self.index = faiss.IndexFlatL2(data.shape[1])
        if self.use_index:
            self.index.train(data)
        self.index.add(data)
        #labels = labels.astype(np.int)
        self.offset = np.min(labels)
        self.labels = labels - self.offset

    @staticmethod
    @utils.njit(parallel = True)
    def _unweighted_counts(neighbors:np.ndarray, labels:np.ndarray, offset: int) -> np.ndarray:
        # This orients itself after https://machinelearningapplied.com/fast-gpu-based-nearest-neighbors-with-faiss/
        res = np.empty(neighbors.shape[0], dtype=labels.dtype)
        for i in numba.prange(neighbors.shape[0]):
            relevant_labels = np.empty(neighbors.shape[1], dtype=labels.dtype)
            for j, neighbor in enumerate(neighbors[i]):
                relevant_labels[j] = labels[neighbor]
            counts = np.bincount(relevant_labels)
            res[i] = np.argmax(counts) + offset
        return res

    @staticmethod
    @utils.njit(parallel=True)
    def _weighted_counts(neighbors: np.ndarray, distances: np.ndarray, root:bool, labels: np.ndarray, offset: int) -> np.ndarray:
        # This orients itself after https://machinelearningapplied.com/fast-gpu-based-nearest-neighbors-with-faiss/
        res = np.empty(neighbors.shape[0], dtype=labels.dtype)
        for i in numba.prange(neighbors.shape[0]):
            relevant_labels = np.empty(neighbors.shape[1], dtype=labels.dtype)
            for j, neighbor in enumerate(neighbors[i]):
                relevant_labels[j] = labels[neighbor]
            weighted_distances = 1/np.sqrt(distances[i]) if root else 1/distances[i]
            inf_mask = np.isinf(weighted_distances)
            if np.any(inf_mask):
                weighted_distances[inf_mask] = 1.0
                weighted_distances[~inf_mask] = 0.0
            counts = np.bincount(relevant_labels, weights=weighted_distances)
            res[i] = np.argmax(counts) + offset
        return res

    @staticmethod
    @utils.njit(parallel=True)
    def _direct_weighted_counts(neighbors: np.ndarray, distances: np.ndarray, root:bool, labels: np.ndarray, offset: int) -> np.ndarray:
        # This orients itself after https://machinelearningapplied.com/fast-gpu-based-nearest-neighbors-with-faiss/
        res = np.empty(neighbors.shape[0], dtype=labels.dtype)
        for i in numba.prange(neighbors.shape[0]):
            relevant_labels = np.empty(neighbors.shape[1], dtype=labels.dtype)
            for j, neighbor in enumerate(neighbors[i]):
                relevant_labels[j] = labels[neighbor]
            weighted_distances = -np.sqrt(distances[i]) if root else -distances[i]
            if np.any(np.isinf(weighted_distances)):
                weighted_distances = np.full_like(weighted_distances, 1.0)
            counts = np.bincount(relevant_labels, weights=weighted_distances)
            res[i] = np.argmax(counts) + offset
        return res

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.index is None or self.labels is None:
            raise RuntimeError('Classifier needs to be fit before prediction!')
        data = data.astype(np.float32)
        is_cosine = self.metric == 'cosine'
        if is_cosine:
            data = faiss.normalize_L2(data)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        total_neighbors = np.empty((data.shape[0], self.k), dtype=np.int32)
        total_distances = np.empty((data.shape[0], self.k), dtype=np.float32)
        for i in range(0, data.shape[0], self.pred_batch_size):
            max_idx = min(i+self.pred_batch_size, data.shape[0])
            slice = data[i:max_idx]
            distances, neighbors = self.index.search(slice, self.k)
            total_neighbors[i:max_idx] = neighbors
            if total_distances is not None:
                total_distances[i:max_idx] = distances
        if self.weights == 'sklearn':
            return self._weighted_counts(total_neighbors, total_distances, is_cosine,self.labels, self.offset)
        elif self.weights == 'direct':
            return self._direct_weighted_counts(total_neighbors, total_distances, is_cosine,  self.labels, self.offset)
        return self._unweighted_counts(total_neighbors, self.labels, self.offset)
        # res = []
        # for i in range(data.shape[0]):
        #     relevant_labels = np.take(self.labels, neighbors[i], axis=0)
        #     unique, counts = np.unique(relevant_labels, return_counts=True)
        #     res.append(unique[np.argmax(counts)])
        # return np.array(res)

class SavableLGBMClassifier():
    def __init__(self, delegate: Union[lightgbm.LGBMClassifier, str]):
        self.delegate = delegate
        self.params = None

    def save(self, f_name) -> List[str]:
        with open(f_name+'.pkl', 'wb') as fd:
            pickle.dump(self.delegate, fd)
        booster = cast(lightgbm.Booster, self.delegate.booster_)
        booster.save_model(f_name+'.txt')
        return [f_name+'.pkl', f_name+'.txt']

    def __call__(self, *args, **kwargs):
        if isinstance(self.delegate, str):
            with open(self.delegate+'.pkl', 'rb') as fd:
                res: lightgbm.LGBMClassifier = pickle.load(fd)
            res.set_params(**kwargs)
            #res._Booster = lightgbm.Booster(params=kwargs, model_file=self.delegate+'.txt')
            self.delegate = res
        self.params = kwargs
        return self

    def fit(self, *args, **kwargs):
        # kwargs.update(self.params)
        return self.delegate.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        kwargs.update(self.params)
        return self.delegate.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        kwargs.update(self.params)
        return self.delegate.predict_proba(*args, **kwargs)
