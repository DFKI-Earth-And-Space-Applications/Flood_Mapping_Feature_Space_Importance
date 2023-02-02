import argparse
import os
import time
from typing import Union, List

import numpy as np
import pandas as pd
import torch

from classification import BitwiseAndChannelCombiner, FaissKNNClassifier
from metrics import *
from pre_process import _StatsExtractor, _ZonalStatsExtractor
from utils import trace4d, rgb_to_hsv_np, rgb_to_hsv, rgb_to_hsv_torch


def test_stats_extractor():
    # Expected output:
    # [[[0.5   0. - 1.   10.]
    #   [1.5 - 0.5   1.  100.]]
    # [[0.5   0. - 1.   10.]
    # [1.5 - 0.5
    # 1.
    # 100.]]]
    # [[0 1 0 1]
    #  [1 0 1 1]]
    # [[0]
    #  [1]][[-2.5000000e-01  1.2500000e+00  7.5000000e-01  2.5000000e-01
    #        5.6250000e-01  6.2500000e-02]
    # [5.0000000e+00
    # 4.9750000e+01
    # 5.0000000e+00
    # 5.0250000e+01
    # 2.5000000e+01
    # 2.5250625e+03]]
    # [[0]
    #  [1]][[0.00000000e+00 - 5.00000000e-01  0.00000000e+00  0.00000000e+00
    #        0.00000000e+00  0.00000000e+00]
    # [3.16666667e+00
    # 3.41666667e+01
    # 4.87054640e+00
    # 4.65516440e+01
    # 2.37222222e+01
    # 2.16705556e+03]]
    intensity = np.array(
        [[[0.5, 1.5], [0, -0.5], [-1, 1], [10, 100]], [[0.5, 1.5], [0, -0.5], [-1, 1], [10, 100]]]).transpose((0, 2, 1))
    label = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    print(intensity)
    print(label)
    intensity_dataset = ArrayInMemoryDataset(intensity)
    label_dataset = ArrayInMemoryDataset(label)
    label_wrapper = _StatsExtractor(label_dataset, intensity_dataset, ['label'])
    intensity_wrapper = _StatsExtractor(label_dataset, intensity_dataset, ['mean_intensity', 'std', 'var'])  #
    print(label_wrapper[0], intensity_wrapper[0])
    print(label_wrapper[1], intensity_wrapper[1])


def test_bitwise_and():
    # Expected output
    # [[[1 1]
    #   [0 1]
    #  [1 0]
    # [0
    # 0]]
    # [[1 1]
    #  [0 0]
    # [1
    # 1]
    # [0 0]]
    # [[0 0]
    #  [0 0]
    # [0
    # 0]
    # [0 0]]
    # [[1 1]
    #  [1 1]
    # [1
    # 1]
    # [0 1]]]
    # [[1 0 0 0]
    #  [1 0 1 0]
    # [0 0 0 0]
    # [1 1 1 0]]
    data = np.array([[[1, 1], [0, 1], [1, 0], [0, 0]], [[1, 1], [0, 0], [1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0],
                                                                                          [0, 0]],
                     [[1, 1], [1, 1], [1, 1], [0, 1]]])
    classifier = BitwiseAndChannelCombiner()
    print(data)
    print(classifier.predict(data))


def trace_withaxis01_is_identical_to_no_arg():
    array = np.random.random((4, 4, 4, 4))
    print(array.shape, array)
    ta = np.abs(np.trace(array, axis1=0, axis2=1))
    print(ta.shape, ta)
    ta = np.abs(np.trace(array))
    print(ta.shape, ta)
    ta = np.abs(trace4d(array))
    print(ta.shape, ta)


def hsv_test(test_amount=(256, 512, 512)):
    test_amount = test_amount if isinstance(test_amount, tuple) else tuple(test_amount)
    red: np.ndarray = np.random.random(test_amount).astype(np.float32).flatten()
    green: np.ndarray = np.random.random(test_amount).astype(np.float32).flatten()
    blue: np.ndarray = np.random.random(test_amount).astype(np.float32).flatten()
    _ = rgb_to_hsv(np.array([0.2, 0.1], dtype=np.float32),
                   np.array([0.1, 0.5], dtype=np.float32),
                   np.array([0.8, 0.0], dtype=np.float32))
    t = time.time()
    h, s, v = rgb_to_hsv(red, green, blue)
    t = time.time() - t
    print(f'Numba took {t:.3f}s.')
    t1 = time.time()
    h_np, s_np, v_np = rgb_to_hsv_np(red, green, blue)
    t1 = time.time() - t1
    print(f'Numpy took {t1:.3f}s.')
    red_t, green_t, blue_t = tuple(torch.from_numpy(color_array) for color_array in (red, green, blue))
    t2 = time.time()
    h_t, s_t, v_t = rgb_to_hsv_torch(red_t, green_t, blue_t)
    t2 = time.time() - t2
    print(f'Torch took {t2:.3f}s.')
    h_t, s_t, v_t = tuple(t.numpy() for t in (h_t, s_t, v_t))
    h[h == 1.0] = 0.0
    # h_np[h_np == 1.0] = 0.0
    # h_t[h_t == 1.0] = 0.0
    print('red  :', red)
    print('green:', green)
    print('blue :', blue)
    print('hue       (numba):', h)
    print('hue       (numpy):', h_np)
    print('hue       (torch):', h_t)
    print('saturation(numba):', s)
    print('saturation(numpy):', s_np)
    print('saturation(torch):', s_t)
    print('value     (numba):', v)
    print('value     (numpy):', v_np)
    print('value     (torch):', v_t)
    nnbh = ~np.isclose(h_np, h, atol=0.0001, rtol=0.0)
    nth = ~np.isclose(h_np, h_t, atol=0.0001, rtol=0.0)
    nbth = ~np.isclose(h, h_t, atol=0.0001, rtol=0.0)
    print('numpy-numba-hue mismatch:', h[nnbh], h_np[nnbh], np.argwhere(nnbh))
    print('numpy-torch-hue mismatch:', h_np[nth], h_t[nth], np.argwhere(nth))
    print('numba-torch-hue mismatch:', h[nbth], h_t[nbth], np.argwhere(nbth))
    nnbs = ~np.isclose(s_np, s)
    nts = ~np.isclose(s_np, s_t)
    nbts = ~np.isclose(s, s_t)
    print('numpy-numba-sat mismatch:', s[nnbs], s_np[nnbs], np.argwhere(nnbs))
    print('numpy-torch-sat mismatch:', s_np[nts], s_t[nts], np.argwhere(nts))
    print('numba-torch-sat mismatch:', s[nbts], s_t[nbts], np.argwhere(nbts))
    nnbv = ~np.isclose(v_np, v)
    ntv = ~np.isclose(v_np, v_t)
    nbtv = ~np.isclose(v, v_t)
    print('numpy-numba-val mismatch:', v[nnbv], v_np[nnbv], np.argwhere(nnbv))
    print('numpy-torch-val mismatch:', v_np[ntv], v_t[ntv], np.argwhere(ntv))
    print('numba-torch-val mismatch:', v[nbtv], v_t[nbtv], np.argwhere(nbtv))
    assert not np.any(nnbh) and not np.any(nth) and not np.any(nbth)
    assert not np.any(nnbs) and not np.any(nts) and not np.any(nbts)
    assert not np.any(nnbv) and not np.any(ntv) and not np.any(nbtv)


@utils.njit(parallel=False, fastmath=True)
def merge_neighbouring(n_pred_labels: int, allowed_labels: int, pred_ints: np.ndarray):
    shape = pred_ints.shape
    flattened_array = pred_ints.flatten()
    while n_pred_labels > allowed_labels:
        pos = np.random.randint(0, min(shape[1:]), size=(2,))
        label = flattened_array[pos[0] * shape[1] + pos[1]]
        found = False
        found_x = pos[0]
        found_y = pos[1]
        i = 1
        while not found:
            if found_x + i < shape[1] and flattened_array[(found_x + i) * shape[1] + found_y] != label:
                found = True
                found_x += i
            elif 0 <= found_x - i and flattened_array[(found_x - i) * shape[1] + found_y] != label:
                found = True
                found_x -= i
            elif found_y + i < shape[2] and flattened_array[found_x * shape[1] + (found_y + 1)] != label:
                found = True
                found_y += i
            elif 0 <= found_y - i and flattened_array[found_x * shape[1] + (found_y - 1)] != label:
                found = True
                found_y -= i
            else:
                i += 1
        flattened_array[flattened_array == label] = flattened_array[found_x * shape[1] + found_y]
        n_pred_labels -= 1
    return flattened_array.reshape(shape)

class RunHookStub(RunHook):

    def __init__(self):
        super().__init__(None, None, None)

    def report(self, value, step):
        print(f'Reporting step {step} with value {str(value)}')

    def should_prune(self):
        return False

    def get_artifact_file_name(self, name: str) -> str:
        return path.abspath(name)

    def open_artifact_file(self, name: str, method: str) -> ContextManager:
        return open(path.abspath(name), method)

    def log_metric(self, metric: str, value: Any, step=None):
        print(f'Logging metric {metric} with value {str(value)} and step {step}.')


def contingency_matrix_merge_test(test_amount=(86, 512, 512), min_label=0, max_label=9, num_contained_trials=9,
                                  num_merge_trials=10):
    def timed_append(pred_ints: Union[np.ndarray, torch.Tensor],
                     label_ints: Union[np.ndarray, torch.Tensor],
                     comp: ContingencyMatrixComputation,
                     ls: List[float],
                     clear: bool):
        t = time.time()
        res = comp(step=None, final=False, **{KEY_PREDICTIONS: pred_ints, KEY_LABELS: label_ints})
        t = time.time() - t
        ls.append(t)
        if clear:
            comp.contingency_matrix = None
        return res

    def assert_contingency_eq(comp_np: dict, comp_t: dict):
        np_contingency, np_unique_pred, np_unique_labels = comp_np[KEY_CONTINGENCY_MATRIX], comp_np[
            KEY_CONTINGENCY_UNIQUE_PREDICTIONS], comp_np[KEY_CONTINGENCY_UNIQUE_LABELS]
        torch_contingency, torch_unique_pred, torch_unique_labels = comp_t[KEY_CONTINGENCY_MATRIX], comp_t[
            KEY_CONTINGENCY_UNIQUE_PREDICTIONS], comp_t[KEY_CONTINGENCY_UNIQUE_LABELS]
        assert np.all(np_contingency == torch_contingency.numpy()), f'{np_contingency} != {torch_contingency.numpy()}'
        assert np.all(np_unique_pred == torch_unique_pred.numpy()), f'{np_unique_pred} != {torch_unique_pred.numpy()}'
        assert np.all(
            np_unique_labels == torch_unique_labels.numpy()), f'{np_unique_labels} != {torch_unique_labels.numpy()}'
        assert np_contingency.sum() == torch_contingency.numpy().sum(), f'{np_contingency.sum()} != {torch_contingency.numpy().sum()}'

    comp_np = ContingencyMatrixComputation('np_contingency').set_hook(RunHookStub())
    comp_t = ContingencyMatrixComputation('torch_contingency').set_hook(RunHookStub())
    label_range = max_label - min_label
    t_np_merge = []
    t_t_merge = []
    t_np_contained = []
    t_t_contained = []
    t_start = time.time()
    for i in range(num_merge_trials):
        print(f'{time.time() - t_start:.3f}:Executing Merge Trial {i + 1}')
        add = 1 if i > 0 else 0
        min_label_used = min_label + i * label_range + add
        max_label_used = max_label + i * label_range + add + 1  # plus 1 due to the spec of randint
        pred_ints = np.random.randint(min_label_used, max_label_used, test_amount)
        label_ints = np.random.randint(min_label_used, max_label_used, test_amount)
        np_res = timed_append(pred_ints, label_ints, comp_np, t_np_merge, True)
        torch_res = timed_append(torch.from_numpy(pred_ints), torch.from_numpy(label_ints), comp_t, t_t_merge, True)
        assert_contingency_eq(np_res, torch_res)
        for j in range(num_contained_trials):
            print(f'{time.time() - t_start:.3f}:\tExecuting contained trial {j + 1}')
            pred_ints = np.random.randint(min_label_used, max_label_used, test_amount)
            label_ints = np.random.randint(min_label_used, max_label_used, test_amount)
            np_res = timed_append(pred_ints, label_ints, comp_np, t_np_contained, j == num_contained_trials - 1)
            torch_res = timed_append(torch.from_numpy(pred_ints), torch.from_numpy(label_ints), comp_t, t_t_contained,
                                     j == num_contained_trials - 1)
            assert_contingency_eq(np_res, torch_res)
    print('Trials finished.')
    #    print('Unique-Labels     :', comp_np.contingency_matrix[1])
    #    print('Unique-Predictions:', comp_np.contingency_matrix[2])
    #    print('Contingency       :', comp_np.contingency_matrix[0])
    print(f'Statistics:')
    merge = pd.DataFrame({'Numpy-Non-Merge': t_np_merge, 'Torch-Non-Merge': t_t_merge})
    contained = pd.DataFrame({'Numpy-Merge': t_np_contained, 'Torch-Merge': t_t_contained})
    print(merge.describe())
    print(contained.describe())
    print('Raw-Measurments:')
    print(merge)
    print(contained)
    # Statistics:
    #        Numpy-Non-Merge  Torch-Non-Merge
    # count        10.000000        10.000000
    # mean          2.394893         2.483783
    # std           0.161688         0.181301
    # min           2.043315         2.013924
    # 25%           2.309710         2.455881
    # 50%           2.438316         2.532095
    # 75%           2.485289         2.570989
    # max           2.619220         2.677048
    #        Numpy-Merge  Torch-Merge
    # count    90.000000    90.000000
    # mean      2.483770     2.579755
    # std       0.264269     0.247469
    # min       1.951875     1.975325
    # 25%       2.376832     2.460076
    # 50%       2.451799     2.533410
    # 75%       2.502520     2.577723
    # max       3.935250     3.588336


def test_metrics_on_data(test_labels: np.ndarray, test_predictions: np.ndarray):
    tested_contingency_metrics = DistributedMetricComputation([
        AccuracyComputation('accuracy'),
        PrecisionComputation('precision'),
        RecallComputation('recall'),
        FScoreComputation('f1_score', 1.0)
    ])
    matrix_comp = ContingencyMatrixComputation('contingency', tested_contingency_metrics)
    matrix_comp.forward(test_labels, test_predictions)


def test_metrics():
    def generate_data(*class_counts):
        ls = []
        for i, class_count in enumerate(class_counts):
            ls.extend([i] * class_count)
        return np.array(ls)

    homogenity_truth = generate_data(5, 2, 1, 1, 1, 4)
    homogenity_u1 = generate_data(4, 3, 7)
    homogenity_u2 = generate_data(4, 1, 2, 7)
    completeness_truth = generate_data(5, 2, 1, 1, 1, 4)
    completeness_u1 = generate_data(4, 3, 7)
    completeness_u2 = generate_data(4, 1, 2, 7)


def silhouette_speed_test():
    def ref_compute(predictions, data_points, up):
        predictions, data_points = sklearn_numpy_prepare(predictions, data_points)
        centroids = np.empty((up.shape[0], data_points.shape[1]))
        intra_dists = np.empty(up.shape[0])
        center = np.mean(data_points, axis=0)
        for i, p in enumerate(up):
            masked_points = data_points[predictions == p]
            centroids[i] = np.mean(masked_points, axis=0)
            intra_dists[i] = np.sum((masked_points - centroids[i]) ** 2)
        return centroids, intra_dists, center, data_points.shape[0]

    def reference_impl(kwargs: Dict[str, Any]):
        assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
        predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
        if KEY_CONTINGENCY_UNIQUE_PREDICTIONS in kwargs:
            up = kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
        else:
            up = np.unique(predictions)
        centroids, intra_dists, center, num_samples = ref_compute(predictions, data_points, up)
        return {'center': center, 'num_samples': num_samples, 'intra_dists': intra_dists, 'centroids': centroids}

    def check_res(ref_score_dict, score_dict):
        for k in ref_score_dict.keys():
            assert k in score_dict, f'Assumed {k} to be present in the result!'
            assert np.isclose(ref_score_dict[k], score_dict[k]).all(), \
                f'Values for {k} are not close: {ref_score_dict[k]} != {score_dict[k]}'

    rand_points = np.random.random((86, 12, 512, 512))
    rand_clusters = np.random.randint(0, 10, (86, 512, 512, 1))
    timings = []
    reference_timings = []
    scorer = ClusterSpatialDistributionComputation('test')
    print('Starting test sequence.')
    for i, points, clusters in zip(range(86), rand_points, rand_clusters):
        unique = np.unique(clusters)
        d = {KEY_INPUT: points,
             KEY_PREDICTIONS: clusters,
             KEY_CONTINGENCY_UNIQUE_PREDICTIONS: unique}
        scorer._compute_numpy(d, final=True)
        t = time.time()
        score_dict = scorer._compute_numpy(d, final=True)
        timings.append(time.time() - t)
        t = time.time()
        ref_score_dict = reference_impl(d)
        reference_timings.append(time.time() - t)
        print(f'Image run {i} took {timings[-1]:.3f}s for scorer and {reference_timings[-1]:.3f}s for ref-impl..')
        # score_dict = {k: utils.recursive_simplify(v) for k, v in score_dict.items()}
        # ref_score_dict = {k: utils.recursive_simplify(v) for k, v in ref_score_dict.items()}
        # print(json.dumps(score_dict, indent=4))
        check_res(ref_score_dict, score_dict)
    df = pd.DataFrame({'Numba-Timings': timings, 'Numpy-Timings': reference_timings})
    print(df)
    print(df.describe())
    print('Running total...')
    unique = np.unique(rand_clusters)
    d = {KEY_INPUT: rand_points, KEY_PREDICTIONS: rand_clusters, KEY_CONTINGENCY_UNIQUE_PREDICTIONS: unique}
    t = time.time()
    score_dict = scorer._compute_numpy(d, final=True)
    t1 = time.time() - t
    t = time.time()
    ref_score_dict = reference_impl(d)
    t2 = time.time() - t
    print(f'Total took {t1:.3f}s for Numba and {t2:.3f}s for Numpy.')
    #        Numba-Timings  Numpy-Timings
    # count      86.000000      86.000000
    # mean        0.035191       0.116522
    # std         0.014848       0.013938
    # min         0.026530       0.095255
    # 25%         0.030235       0.108297
    # 50%         0.032136       0.113856
    # 75%         0.035685       0.123175
    # max         0.132600       0.208844
    # Running total...
    # Total took 5.260s for Numba and 6.233s for Numpy.


def test_index_alignment():
    df = pd.DataFrame(list(range(101, 0, -1)), columns=['range'],
                      index=pd.MultiIndex.from_arrays([np.zeros(101, dtype=np.int), np.arange(201, 302)],
                                                      names=['id', 'num']))
    print(df['range'])
    single_row = df.loc[0]
    print(single_row)
    range_idx = single_row.index
    print(range_idx)
    print(range_idx.get_loc(206))
    print(single_row.index[range_idx.get_loc(206)])


def test_locate():
    df = pd.DataFrame(
        {'data': np.random.randint(0, 100, 200), 'choice': np.random.choice(['COMPLETE', 'FAILED'], 200, p=[0.9, 0.1])})
    print(df)
    print(df.loc[df['choice'] == 'COMPLETE'])
    df.loc[:, 'test'] = 1
    print(df)

def test_faiss_knn():
    import sklearn.neighbors as neighbours
    np.random.seed(42)
    index_data = np.random.random((57_472_450, 13))
    index_labels = np.random.randint(0, 2, (index_data.shape[0]))
    query_data = np.random.random((20_297_809, 13))
    print('Fitting Faiss')
    t = time.time()
    classifier = FaissKNNClassifier(5, True, weights='uniform', use_index=True)
    classifier.fit(index_data, index_labels)
    t1 = time.time() - t
    print('Fitting sklearn')
    t = time.time()
    ref_classifier = neighbours.KNeighborsClassifier(5, algorithm='brute', weights='uniform')
    ref_classifier.fit(index_data, index_labels)
    t2 = time.time() - t
    print(f'Fitting took {t1:.3f}s for Faiss and {t2:.3f}s for sklearn.')
    print(f'Predicting with Faiss.')
    t = time.time()
    pred_faiss = classifier.predict(query_data)
    t1 = time.time() - t
    print(f'Predicting with sklearn.')
    t = time.time()
    pred_sklearn = ref_classifier.predict(query_data)
    t2 = time.time() - t
    print(f'Predicting took {t1:.3f}s for Faiss and {t2:.3f}s for sklearn.')
    print(f'Accuracy is {np.count_nonzero(pred_faiss == pred_sklearn)/pred_faiss.shape[0]}')
    assert np.all(pred_faiss == pred_sklearn)

def test_sen1floods11_metrics():
    def computeIOU(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255)#.cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        intersection = torch.sum(output * target)
        union = torch.sum(target) + torch.sum(output) - intersection
        iou = (intersection + .0000001) / (union + .0000001)

        if iou != iou:
            print("failed, replacing with 0")
            iou = torch.tensor(0).float()

        return iou

    # => Accuracy ignores all no-data pixels including not calculating the
    def computeAccuracy(output, target):
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255)#.cuda()
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        correct = torch.sum(output.eq(target))

        return correct.float() / len(target)

    def compare_total_with_average(random_labels_1: torch.Tensor, ground_truth: torch.Tensor):
        test_iou_0 = computeIOU(random_labels_1, ground_truth)
        test_acc_0 = computeAccuracy(random_labels_1, ground_truth)
        test_iou_1 = sum(computeIOU(random_labels_1[i].reshape((1,)+ random_labels_1[i].shape),
                                    ground_truth[i].reshape((1,) + ground_truth[i].shape))
                         for i in range(random_labels_1.shape[0])) / random_labels_1.shape[0]
        test_acc_1 = sum(computeAccuracy(random_labels_1[i].reshape((1,)+ random_labels_1[i].shape),
                                         ground_truth[i].reshape((1,) + ground_truth[i].shape))
                         for i in range(random_labels_1.shape[0])) / random_labels_1.shape[0]
        print('IOU\'s', test_iou_0, test_iou_1)
        print('Acc\'s', test_acc_0, test_acc_1)
        return test_iou_0, test_acc_0, test_iou_1, test_acc_1

    def batch_size_cat(source: torch.Tensor, num_batches: int) -> torch.Tensor:
        to_cat = [torch.cat([source[i * num_batches + j].reshape((1,) + source.shape[1:]) for j in range(num_batches)], dim=0)
                  for i in range(source.shape[0] // num_batches)]
        return torch.cat(to_cat, dim=2)

    def batch_wise_tests(N: int,  random_labels_1: torch.Tensor, random_labels_2: torch.Tensor):
        for num_batches in [12, 6, 4]:
            assert N % num_batches == 0
            print(f'Testing with {num_batches} batches')
            cat_rand_1 = batch_size_cat(random_labels_1, num_batches)
            cat_rand_2 = batch_size_cat(random_labels_2, num_batches)
            compare_total_with_average(cat_rand_1, cat_rand_2)

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    N = 84 # define a number of images
    H, W = 256, 256 # define height and width
    # simulate the NN outputting a label, in order to make that simple we just draw that once and then cat the binary
    # additive inverse
    random_labels_1 = torch.randint(0, 2, (N,1,H,W), dtype=torch.int32)
    random_labels_1 = torch.cat((random_labels_1, 1-random_labels_1), dim=1)
    # for this to be realistic we need a high propability of dry land and a low propability of flood
    # in particular this will require two 0s and one 1 in order to be 1 => p(1) = 0.5^3=0.125
    random_labels_2 = torch.div(torch.argmax(torch.randint(0, 2, (N,3,H,W), dtype=torch.int32), dim=1), 2,
                                rounding_mode='floor')
    # we also need some no-data cases, again I'll use the same probility of 0.125 of something being no-data
    random_labels_2_mod = torch.div(torch.argmax(torch.randint(0, 2, (N,3,H,W), dtype=torch.int32), dim=1), 2,
                                    rounding_mode='floor')
    random_labels_2[random_labels_2_mod > 0] = 255
    compare_total_with_average(random_labels_1, random_labels_2)
    print('now let\'s go crazy and force some images to be no-data')
    to_null = torch.randint(low=0, high=N, size=(4,))
    random_labels_2[to_null] = 255
    compare_total_with_average(random_labels_1, random_labels_2)
    print('As we can see accuracy cannot be calculated if we do that imagewise and there is also a slight difference in IOU...')
    print('well let\'s do it batchwise then!')
    # in order to stick with the same compare code, concatenation happens on the height dimension.
    # This doesn't make a difference as can be verified with the above code - it is flattened anyway
    batch_wise_tests(N, random_labels_1, random_labels_2)
    print('Hey, it works :). But what if we shuffle it around?')
    for _ in range(4):
        perm = torch.randperm(N)
        random_labels_1 = random_labels_1[perm]
        random_labels_2 = random_labels_2[perm]
        print('Permuted data')
        batch_wise_tests(N, random_labels_1, random_labels_2)

def test_sen1floods11_metrics_reproduction():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    N = 89  # define a number of images
    H, W = 512, 512  # define height and width
    # simulate the NN outputting a label, in order to make that simple we just draw that once and then cat the binary
    # additive inverse
    predictions = torch.randint(0, 2, (N, 1, H, W), dtype=torch.int32, device='cuda')
    predictions = torch.cat((predictions, 1 - predictions), dim=1)
    # for this to be realistic we need a high propability of dry land and a low propability of flood
    # in particular this will require two 0s and one 1 in order to be 1 => p(1) = 0.5^3=0.125
    labels = torch.div(torch.argmax(torch.randint(0, 2, (N, 3, H, W), dtype=torch.int32, device='cuda'), dim=1), 2,
                                rounding_mode='floor')
    # we also need some no-data cases, again I'll use the same probility of 0.125 of something being no-data
    random_labels_2_mod = torch.div(torch.argmax(torch.randint(0, 2, (N, 3, H, W), dtype=torch.int32, device='cuda'), dim=1), 2,
                                    rounding_mode='floor')

    labels[random_labels_2_mod > 0] = 255
    predictions = torch.argmax(predictions, dim=1)
    results = []
    batch_sizes = [2,4,8,16]
    direct_batchwise_calculation(labels, predictions, 1000, 16, 42,
                                 exclude_labels=(255,))
    for batch_size in batch_sizes:
        np.random.default_rng(42)
        print(f'Testing batch size {batch_size}.')
        t = time.time()
        calc = direct_batchwise_calculation(predictions, labels, 1000, batch_size, 42,
                                            exclude_labels=(255,))
        t = time.time() -t
        calc['time'] = t
        results.append(calc)
    results = pd.concat(results, axis=0, ignore_index=True)
    results = results.rename(index={cur:batch_size_to_use for cur, batch_size_to_use in zip(results.index, batch_sizes)})
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results)
        print(results.describe())


def test_contingency_matrices():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--contingency_name',dest='contingency_name', default='final.contingency.pkl')
    parser.add_argument('--contingency_key',dest='contingency_key', default=KEY_CONTINGENCY_MATRIX)
    args = parser.parse_args()
    folder = args.folder
    key = args.contingency_key
    contingency_name = args.contingency_name
    matrices = []
    for sub_folder in os.listdir(folder):
        sub_folder = path.join(folder, sub_folder)
        if not path.isdir(sub_folder):
            print(sub_folder, 'is not a directory, skipping.')
        elif not path.exists(path.join(sub_folder, contingency_name)):
            print(sub_folder, 'does not contain a contingency matrix, skipping')
        else:
            with open(path.join(sub_folder, contingency_name), 'rb') as fd:
                matrices.append(pickle.load(fd))
    matrices = [{int(k): {k1: np.array(v1) for k1, v1 in v[0].items()} for k, v in m.items()} for m in matrices]
    mapping = [{id_val: np.any(cm_map[key] != matrices[0][id_val][key]) for id_val, cm_map in m.items()} for m in matrices]
    if any(map(lambda m: any(m.values()), mapping)):
        print(f'Found {len([v for m in mapping for v in m.values() if v])} '
              f'unequal matrices out of {len([v for m in mapping for v in m.values()])}.')
    else:
        print('Found equal matrices')

def comp_contingencies():
    parser = argparse.ArgumentParser()
    parser.add_argument('con1')
    parser.add_argument('con2')
    args = parser.parse_args()
    with open(args.con1, 'rb') as fd:
        con1 = pickle.load(fd)
    with open(args.con2, 'rb') as fd:
        con2 = pickle.load(fd)
    assert con1 == con2, f'{con1} != {con2}'

def test_ncrand_index():
    ground_truth = np.array([0] * 20 + [1] * 2 + [2] * 4)
    u_1 = np.array([0] * 21 + [1] + [2] * 4)
    u_2 = np.array([0] * 18 + [1] * 4 + [2] * 4)
    seq = SequenceMetricComputation([
        ContingencyMatrixComputation('contingency'),
        ForceNumpy(),
        MutualInfoScore('mutual_information'),
        Homogenity('homogenity'),
        Completeness('Completeness'),
        VMeasure('v_measure', 1.0),
        ClusterFMeasure('cluster_f', 1.0),
        PairConfusionMatrix([
            RandScore('rand_index'),
            AdjustedRandScore('ari')
        ]),
        NormalizedClassSizePairConfusionMatrix([
            RandScore('ncri')
        ], execute_parallel=False),
        AccuracyComputation('accuracy'),
        IOUComputation('iou'),
        MaximumAgreementComputation([
            AccuracyComputation('max_agree_accuracy'),
            IOUComputation('max_agree_iou')
        ])
    ])
    res1 = seq.forward(**{KEY_LABELS:ground_truth, KEY_PREDICTIONS:u_1})
    res2 = seq.forward(**{KEY_LABELS:ground_truth, KEY_PREDICTIONS:u_2})
    print('Results for U1',res1)
    print('Results for U2',res2)

def test_contingency_merge():
    labels = torch.tensor([-1, 0, 1])
    predictions = torch.tensor([0, 1])
    comp = ContingencyMatrixComputation('test')
    total_contingency = torch.zeros((3, 2), dtype=torch.int32)
    for i in range(4):
        rand_contingency = torch.randint(0, 256*256, (3, 2))
        total_contingency += rand_contingency
        comp.merge_contingency_torch(None, None, rand_contingency, labels, predictions)
    assert torch.all(comp.contingency_matrix[0] == total_contingency)

def test_gbdt_labels():
    import data_source as ds
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--split', dest='split', default=ds.SPLIT_VALIDATION)
    args = parser.parse_args()
    data_source = ds.Sen1Floods11DataDingsDatasource(args.folder, type=ds.TYPE_GBDT_WEAK_LABEL, split=args.split,
                                                     as_array=True)
    dataset, meta = data_source(None, None)
    data = cast(ArrayInMemoryDataset, dataset).data
    print(meta)
    print(data.shape)
    print(data)

if __name__ == '__main__':
    # ref = np.array([[0, 0, 0, 0, 1, 0]])
    # index = np.array([[[0, 2,5,4,1,1,0,0], [1, 2,3,0,3,5,4,5], [0, 2,5,4,1,1,0,0], [0, 2,5,4,1,1,0,0], [1, 2,3,0,3,5,4,5], [1, 2,3,0,3,5,4,5], [0, 2,5,4,1,1,0,0], [0, 2,5,4,1,1,0,0] ]])
    # print(ref.shape)
    # print(index.shape)
    # print([(ref_entry,index_entry) for ref_entry, index_entry in zip(ref, index)])
    # res = np.array([ref_entry[index_entry] for ref_entry, index_entry in zip(ref, index)])
    # print(res.shape, res)
    # trace_withaxis01_is_identical_to_no_arg()
    # contingency_matrix_comp_test()
    # hsv_test()
    # contingency_matrix_merge_test()
    # silhouette_speed_test()
    # df = pd.DataFrame({'pre': np.random.randint(0, 100, 200), 'post': np.random.randint(0, 100, 200)})
    #test_faiss_kmeans()
    # test_sen1floods11_metrics()
    #test_faiss_knn()
    # df = df.append(second)
    # print(df)
    # tens = torch.tensor([[[list(range(6)) for _ in range(6)]]], dtype=torch.int32)
    # print(tens.shape, tens)
    # dataset = ArrayInMemoryDataset(tens)
    # tens: torch.Tensor = ArrayInMemoryDataset(SlidingWindowExtractorDataset(dataset, 3, 1, 1, 1)).data
    # print(tens.shape, tens)
    # values, indices = torch.mode(tens, 1)
    # print(values.shape, values)
    # print(indices.shape, indices)
    # test_sen1floods11_metrics_reproduction()
    # import  pickle as pkl
    # with open('F:\\Daten\\Briefe\\Kevin\\Experimente\\results\\simple_discriminant_analysis_final_test_42\\4\\valid.region.contingency.pkl', 'rb') as fd:
    #     print(pkl.load(fd))
    # test_ncrand_index()
    # test_crop_extract()
    # test_contingency_merge()
    # comp_contingencies()
    # test_gbdt_labels()
    tens = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    print(tens)
    print(tens[3:4])
