import math

from typing import List, Tuple, Iterable

import numba
import numpy as np
import pandas as pd
import skimage.measure
import scipy.optimize as sp_opt
import torch
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

from multipipeline import *

KEY_PREDICTIONS = 'predictions'
KEY_LABELS = 'labels'
KEY_INPUT = 'input'
DO_TYPE_CHECK = True

# TODO cleanup and remove hook. This is no longer necessary as log-metric doesn't work anyway
class MetricComputation(JSONSerializable):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.hook: Optional[RunHook] = None

    def set_hook(self, hook: Optional[RunHook]) -> 'MetricComputation':
        self.hook = hook
        return self

    def append_to_name(self, appendix: str) -> 'MetricComputation':
        self.name += '.' + appendix
        return self

    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        relevant_vals = [val for val in kwargs.values() if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray)]
        if DO_TYPE_CHECK:
            assert not relevant_vals or not any(map(lambda a: type(a) != type(relevant_vals[0]), relevant_vals)), \
                f'Types should be homogeneous, but got {[type(a) for a in relevant_vals]}!'
        if relevant_vals and any(map(lambda arg: isinstance(arg, torch.Tensor), relevant_vals)):
            kwargs = {k: (v.detach().cpu() if v is not None and isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            val = self._compute_torch(kwargs, final)
        else:
            val = self._compute_numpy(kwargs, final)
        kwargs[self.name] = val
        if final:
            self.finish_computation(kwargs, val, step, final)
        return kwargs

    def finish_computation(self, kwargs: Dict[str, Any],
                           value: Any,
                           step: Optional[int] = None,
                           final: bool = True):
        if self.hook is not None:
            self.hook.log_metric(self.name, value, step)

    def all_names(self) -> List[str]:
        return [self.name]

    def clone(self):
        return self.__class__(self.name)


class DistributedMetricComputation(MetricComputation):
    def __init__(self, sub_modules: List[MetricComputation]):
        super().__init__('undefined')
        self.sub_modules: List[MetricComputation] = []
        self.allow_extract: bool = True
        for sub_module in sub_modules:
            if isinstance(sub_module, DistributedMetricComputation) and sub_module.allow_extract:
                self.sub_modules.extend(sub_module.sub_modules)
            else:
                self.sub_modules.append(sub_module)

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = super().get_params_as_dict()
        del res['allow_extract']
        return res

    def set_hook(self, hook: Optional[RunHook]) -> 'DistributedMetricComputation':
        super().set_hook(hook)
        for sub_module in self.sub_modules:
            sub_module.set_hook(hook)
        return self

    def append_to_name(self, appendix: str) -> 'DistributedMetricComputation':
        super().append_to_name(appendix)
        for sub_module in self.sub_modules:
            sub_module.append_to_name(appendix)
        return self

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        res_kwargs = kwargs.copy()
        for sub_module in self.sub_modules:
            res_kwargs.update(sub_module.forward(step, final, **kwargs))
        return res_kwargs

    def finish_computation(self, kwargs: Dict[str, Any], value: Union[torch.Tensor, np.ndarray],
                           step: Optional[int] = None,
                           final: bool = True):
        pass

    def clone(self):
        return self.__class__([mod.clone() for mod in self.sub_modules])

    def all_names(self) -> List[str]:
        res = []
        for sub_module in self.sub_modules:
            res.extend(sub_module.all_names())
        return res


class SequenceMetricComputation(MetricComputation):
    def __init__(self, sub_modules: List[MetricComputation]):
        super().__init__('undefined')
        self.sub_modules: List[MetricComputation] = []
        self.allow_extract: bool = True
        for sub_module in sub_modules:
            if isinstance(sub_module, SequenceMetricComputation) and sub_module.allow_extract:
                self.sub_modules.extend(sub_module.sub_modules)
            else:
                self.sub_modules.append(sub_module)

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = super().get_params_as_dict()
        del res['allow_extract']
        return res

    def set_hook(self, hook: Optional[RunHook]) -> 'SequenceMetricComputation':
        super().set_hook(hook)
        for sub_module in self.sub_modules:
            sub_module.set_hook(hook)
        return self

    def append_to_name(self, appendix: str) -> 'SequenceMetricComputation':
        super().append_to_name(appendix)
        for sub_module in self.sub_modules:
            sub_module.append_to_name(appendix)
        return self

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        for sub_module in self.sub_modules:
            kwargs = sub_module.forward(step, final, **kwargs)
        return kwargs

    def finish_computation(self, kwargs: Dict[str, Any], value: Union[torch.Tensor, np.ndarray],
                           step: Optional[int] = None,
                           final: bool = True):
        pass

    def clone(self):
        return self.__class__([mod.clone() for mod in self.sub_modules])

    def all_names(self) -> List[str]:
        res = []
        for sub_module in self.sub_modules:
            res.extend(sub_module.all_names())
        return res


KEY_CONTINGENCY_MATRIX = 'key:contingency_matrix'
KEY_CONTINGENCY_UNIQUE_LABELS = 'key:contingency_unique_labels'
KEY_CONTINGENCY_UNIQUE_PREDICTIONS = 'key:contingency_unique_predictions'
KEY_CONTINGENCY_EXPECTED_LABELS = 'key:contingency_expected_labels'
KEY_CONTINGENCY_EXPECTED_PREDICTIONS = 'key:contingency_expected_predictions'

# Tests indicate a runtime of ~5s for the complete validation set
class ContingencyMatrixComputation(MetricComputation):
    def __init__(self, name: str, check_predictions: bool = True, check_labels: bool = True,
                 force_numpy_output: bool = False) -> None:
        super().__init__(name)
        self.contingency_matrix = None
        self.check_predictions = check_predictions
        self.check_labels = check_labels
        self.force_numpy_output = force_numpy_output

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = super().get_params_as_dict()
        del res['contingency_matrix']
        return res

    @staticmethod
    def _compute_contingency_torch(predictions: torch.Tensor, labels: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_predictions = torch.unique(predictions, sorted=True)
        unique_labels = torch.unique(labels, sorted=True)

        matrix = torch.empty((unique_labels.shape[0], unique_predictions.shape[0]), dtype=torch.int32)
        label_masks = [labels == unique_label for unique_label in unique_labels]
        for pi, pred_label in enumerate(unique_predictions):
            pred_mask = predictions == pred_label
            for li, label_mask in enumerate(label_masks):
                matrix[li, pi] = torch.count_nonzero(torch.logical_and(pred_mask, label_mask))
        return matrix, unique_labels, unique_predictions

    @staticmethod
    # @numba.njit(fastmath = False, parallel = True) - actually jitting this makes it slower as far as I can tell
    def _compute_contingency_numpy(predictions: np.ndarray, labels: np.ndarray,
                                   check_labels: bool, check_predictions: bool) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        unique_predictions = np.unique(predictions)
        unique_labels = np.unique(labels)
        if predictions.ndim > 2:
            predictions = predictions.view()
            predictions.shape = tuple(s for s in predictions.shape if s > 1)
        if labels.ndim > 2:
            labels = labels.view()
            labels.shape = tuple(s for s in labels.shape if s > 1)

        if predictions.ndim == 2 and labels.ndim == 2 and unique_predictions.shape[0] * unique_labels.shape[0] >= 100:
            if check_predictions and np.any(unique_predictions <= 0):
                pred_offset = -np.min(unique_predictions) + 1
                used_predictions = predictions + pred_offset
            else:
                pred_offset = 0
                used_predictions = predictions
            if check_labels and np.any(unique_labels <= 0):
                label_offset = -np.min(unique_labels) + 1
                used_labels = labels + label_offset
            else:
                label_offset = 0
                used_labels = labels
            pred_bbs = utils.extract_bbs(skimage.measure.regionprops(used_predictions), True)
            label_bbs = utils.extract_bbs(skimage.measure.regionprops(used_labels), True)
            matrix = np.empty((unique_labels.shape[0], unique_predictions.shape[0]), dtype=np.int32)
            label_masks = [utils.index_by_bounding_box_nonumba(used_labels, label_bb) == (label + label_offset)
                           for label, label_bb in zip(unique_labels, label_bbs)]
            for pi, (pred, pred_bb) in enumerate(zip(unique_predictions, pred_bbs)):
                pred_mask = utils.index_by_bounding_box_nonumba(used_predictions, pred_bb) == (pred + pred_offset)
                for li, (label_mask, label_bb) in enumerate(zip(label_masks, label_bbs)):
                    intersection_bb = utils.bbs_intersection(label_bb, pred_bb)
                    if np.any(intersection_bb[0] >= intersection_bb[1]):
                        matrix[li, pi] = 0
                    else:
                        label_aligned = utils.align_bounding_box(intersection_bb, label_bb)
                        pred_aligned = utils.align_bounding_box(intersection_bb, pred_bb)
                        match = np.logical_and(utils.index_by_bounding_box_nonumba(pred_mask, pred_aligned),
                                               utils.index_by_bounding_box_nonumba(label_mask, label_aligned))
                        matrix[li, pi] = np.count_nonzero(match)
            return matrix, unique_labels, unique_predictions
        else:
            matrix = np.empty((unique_labels.shape[0], unique_predictions.shape[0]), dtype=np.int32)
            label_masks = [labels == label for label in unique_labels]
            for pi, pred in enumerate(unique_predictions):
                pred_mask = predictions == pred
                for li, label_mask in enumerate(label_masks):
                    matrix[li, pi] = np.count_nonzero(np.logical_and(pred_mask, label_mask))
            return matrix, unique_labels, unique_predictions

    def merge_contingency_torch(self, expected_labels: Optional[torch.Tensor],
                                expected_predictions: Optional[torch.Tensor],
                                cm: torch.Tensor, ul: torch.Tensor, up: torch.Tensor):
        if self.contingency_matrix is None:
            self.contingency_matrix = cm, ul, up
            return
        scm, sul, sup = self.contingency_matrix
        assert type(scm) == type(sul) == type(sup)
        if not isinstance(scm, torch.Tensor):
            scm, sul, sup = torch.from_numpy(scm), torch.from_numpy(sul), torch.from_numpy(sup)

        if not torch.all(torch.isin(ul, sul)) or not torch.all(torch.isin(up, sup)):
            new_sul = torch.unique(torch.cat((ul, sul)), sorted=True) if expected_labels is None else expected_labels
            new_sup = torch.unique(torch.cat((up, sup)), sorted=True) if expected_predictions is None else expected_predictions
            new_scm = torch.empty((new_sul.shape[0], new_sup.shape[0]), dtype=torch.int32)
            l_indexes = [(il, torch.nonzero(sul == label), torch.nonzero(ul == label))
                         for il, label in enumerate(new_sul)]
            p_indexes = [(ip, torch.nonzero(sup == prediction), torch.nonzero(up == prediction))
                         for ip, prediction in enumerate(new_sup)]
            for il, sl_index, l_index in l_indexes:
                for ip, sp_index, p_index in p_indexes:
                    new_scm[il, ip] = torch.sum(scm[sl_index, sp_index]) + torch.sum(cm[l_index, p_index])
            self.contingency_matrix = new_scm, new_sul, new_sup
        else:
            l_indexes = [(il, torch.nonzero(sul == label)) for il, label in enumerate(ul)]
            p_indexes = [(ip, torch.nonzero(sup == prediction)) for ip, prediction in enumerate(up)]
            for il, l_index in l_indexes:
                for ip, p_index in p_indexes:
                    scm[l_index, p_index] += cm[il, ip]
            self.contingency_matrix = scm, sul, sup

    def merge_contingency_numpy(self, cm: np.ndarray, ul: np.ndarray, up: np.ndarray):
        if self.contingency_matrix is None:
            self.contingency_matrix = cm, ul, up
            return
        scm, sul, sup = self.contingency_matrix
        assert type(cm) == type(ul) == type(up) == type(scm) == type(sul) == type(sup)

        if not np.all(np.isin(ul, sul)) or not np.all(np.isin(up, sup)):
            new_sul = np.unique(np.concatenate((ul, sul)))
            new_sup = np.unique(np.concatenate((up, sup)))
            new_scm = np.empty((new_sul.shape[0], new_sup.shape[0]), dtype=np.int32)
            l_indexes = [(il, np.nonzero(sul == label), np.nonzero(ul == label))
                         for il, label in enumerate(new_sul)]
            p_indexes = [(ip, np.nonzero(sup == prediction), np.nonzero(up == prediction))
                         for ip, prediction in enumerate(new_sup)]
            for il, sl_index, l_index in l_indexes:
                for ip, sp_index, p_index in p_indexes:
                    new_scm[il, ip] = np.sum(scm[sl_index, sp_index]) + np.sum(cm[l_index, p_index])
            self.contingency_matrix = new_scm, new_sul, new_sup
        else:
            l_indexes = [(il, np.nonzero(sul == label)) for il, label in enumerate(ul)]
            p_indexes = [(ip, np.nonzero(sup == prediction)) for ip, prediction in enumerate(up)]
            for il, l_index in l_indexes:
                for ip, p_index in p_indexes:
                    scm[l_index, p_index] += cm[il, ip]

    def _contingency_to_dict(self) -> Dict[str, Any]:
        if self.force_numpy_output and isinstance(self.contingency_matrix[0], torch.Tensor):
            return {KEY_CONTINGENCY_MATRIX: self.contingency_matrix[0].detach().cpu().numpy(),
                    KEY_CONTINGENCY_UNIQUE_LABELS: self.contingency_matrix[1].detach().cpu().numpy(),
                    KEY_CONTINGENCY_UNIQUE_PREDICTIONS: self.contingency_matrix[2].detach().cpu().numpy()}
        return {KEY_CONTINGENCY_MATRIX: self.contingency_matrix[0],
                KEY_CONTINGENCY_UNIQUE_LABELS: self.contingency_matrix[1],
                KEY_CONTINGENCY_UNIQUE_PREDICTIONS: self.contingency_matrix[2]}

    def _use_numpy(self, predictions_shape: Tuple[int, ...], labels_shape: Tuple[int, ...]) -> bool:
        return sum(map(lambda d: (1 if d > 1 else 0), predictions_shape)) == 2 and \
               sum(map(lambda d: (1 if d > 1 else 0), labels_shape)) == 2 and \
               (self.contingency_matrix is None or not any(
                   map(lambda a: not isinstance(a, torch.Tensor), self.contingency_matrix)))
        # or

    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        assert KEY_PREDICTIONS in kwargs and KEY_LABELS in kwargs
        predictions, labels = kwargs['predictions'], kwargs['labels']
        if not any(map(lambda t: isinstance(t, torch.Tensor), [predictions, labels])) and self._use_numpy(predictions.shape, labels.shape):
            # print('Using numpy')
            v_tup = self._compute_contingency_numpy(predictions=predictions.numpy(), labels=labels.numpy(),
                                                    check_labels=self.check_labels,
                                                    check_predictions=self.check_predictions)
            self.merge_contingency_numpy(*v_tup)
            return self._contingency_to_dict()
        else:
            v_tup = self._compute_contingency_torch(predictions=predictions,
                                                    labels=labels)
            expected_labels = None if KEY_CONTINGENCY_EXPECTED_LABELS not in kwargs else utils.np_as_torch(kwargs[KEY_CONTINGENCY_EXPECTED_LABELS])
            expected_predictions = None if KEY_CONTINGENCY_EXPECTED_PREDICTIONS not in kwargs else utils.np_as_torch(kwargs[KEY_CONTINGENCY_EXPECTED_PREDICTIONS])
            self.merge_contingency_torch(expected_labels, expected_predictions, *v_tup)
            return self._contingency_to_dict()

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        assert KEY_PREDICTIONS in kwargs and KEY_LABELS in kwargs
        predictions, labels = kwargs['predictions'], kwargs['labels']
        if self._use_numpy(predictions.shape, labels.shape):
            v_tup = self._compute_contingency_numpy(predictions=predictions, labels=labels,
                                                    check_labels=self.check_labels,
                                                    check_predictions=self.check_predictions)
            self.merge_contingency_numpy(*v_tup)
            return self._contingency_to_dict()
        else:
            v_tup = self._compute_contingency_torch(predictions=torch.from_numpy(predictions),
                                                    labels=torch.from_numpy(labels))
            expected_labels = None if KEY_CONTINGENCY_EXPECTED_LABELS not in kwargs else utils.np_as_torch(kwargs[KEY_CONTINGENCY_EXPECTED_LABELS])
            expected_predictions = None if KEY_CONTINGENCY_EXPECTED_PREDICTIONS not in kwargs else utils.np_as_torch(kwargs[KEY_CONTINGENCY_EXPECTED_PREDICTIONS])
            self.merge_contingency_torch(expected_labels, expected_predictions, *v_tup)
            return self._contingency_to_dict()

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res.update(res[self.name])
        return res

    def finish_computation(self, kwargs: Dict[str, Any],
                           value: Dict[str, Any],
                           step: Optional[int] = None,
                           final: bool = True):
        super().finish_computation(kwargs,
                                   value,
                                   step,
                                   final)
        if final:
            self.contingency_matrix = None

def normalize_contingency(cm: np.ndarray, ul: np.ndarray, up: np.ndarray, exclude_clouds: bool) \
        -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not exclude_clouds:
        if not np.any(up == -1):
            up = np.concatenate((up, [-1]))
            cm = np.concatenate((cm, np.zeros((cm.shape[0], 1), dtype=np.int32)), axis=1)
        if not np.any(ul == -1):
            ul = np.concatenate((ul, [-1]))
            cm = np.concatenate((cm, np.zeros((1, cm.shape[1]), dtype=np.int32)))
    if not np.any(ul == 0):
        ul = np.concatenate((ul, [0]))
        cm = np.concatenate((cm, np.zeros((1, cm.shape[1]), dtype=np.int32)))
    if not np.any(ul == 1):
        ul = np.concatenate((ul, [1]))
        cm = np.concatenate((cm, np.zeros((1, cm.shape[1]), dtype=np.int32)))
    if not np.any(up == 0):
        up = np.concatenate((up, [0]))
        cm = np.concatenate((cm, np.zeros((cm.shape[0], 1), dtype=np.int32)), axis=1)
    if not np.any(up == 1):
        up = np.concatenate((up, [1]))
        cm = np.concatenate((cm, np.zeros((cm.shape[0], 1), dtype=np.int32)), axis=1)
    if exclude_clouds:
        cm = cm[ul != -1]
        ul = ul[ul != -1]


    return cm, ul, up

class ToSen1Floods11NormalizedContingency(SequenceMetricComputation):
    def __init__(self, sub_modules: Union[MetricComputation, List[MetricComputation]], exclude_clouds: bool = True):
        if isinstance(sub_modules, MetricComputation):
            sub_modules = [sub_modules]
        super().__init__(sub_modules)
        self.allow_extract = False
        self.exclude_clouds = exclude_clouds

    def forward(self, step: Optional[int] = None, final: bool = True, **d) -> Dict[str, any]:
        if not final:
            raise NotImplementedError
        cm = utils.torch_as_np(d[KEY_CONTINGENCY_MATRIX])
        ul = utils.torch_as_np(d[KEY_CONTINGENCY_UNIQUE_LABELS])
        up = utils.torch_as_np(d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS])

        if not self.exclude_clouds or (not np.all(ul == -1) and not cm[ul == -1].sum() == cm.sum()):
            cm, ul, up = normalize_contingency(cm, ul, up, self.exclude_clouds)

            d = d.copy()
            d[KEY_CONTINGENCY_MATRIX] = cm
            d[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
            d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = up
            d = super(ToSen1Floods11NormalizedContingency, self).forward(step, final, **d)

        return d


class ForceNumpy(MetricComputation):
    def __init__(self) -> None:
        super().__init__('Nameless-utility-module!')

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        return {k: utils.torch_as_np(v) for k, v in kwargs.items()}

class TorchNumpyComputable(MetricComputation):
    def _compute(self, kwargs: Dict[str, Any], final: bool) -> Any:
        raise NotImplementedError

    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Any:
        return self._compute(kwargs, final)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        return self._compute(kwargs, final)


def tp_fp_fn(cm: Union[torch.Tensor, np.ndarray], index: int, index2: Optional[int] = None) -> Tuple[int, int, int]:
    if index2 is None:
        index2 = index
    tp = cm[index, index2]
    fp = cm[:, index2].sum() - tp
    fn = cm[index].sum() - tp
    return tp, fp, fn


def get_contingency(kwargs: Dict[str, Any]) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    assert KEY_CONTINGENCY_MATRIX in kwargs, \
        f'Expected keys {str({KEY_CONTINGENCY_MATRIX, KEY_CONTINGENCY_UNIQUE_LABELS})} to be present in {str(kwargs)}'

    return kwargs[KEY_CONTINGENCY_MATRIX], kwargs[KEY_CONTINGENCY_UNIQUE_LABELS]


def get_contingency_predictions(kwargs: Dict[str, Any]) -> Tuple[
    Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    assert KEY_CONTINGENCY_MATRIX in kwargs, \
        f'Expected keys {str({KEY_CONTINGENCY_MATRIX, KEY_CONTINGENCY_UNIQUE_LABELS, KEY_CONTINGENCY_UNIQUE_PREDICTIONS})} to be present in {str(kwargs)}'
    return kwargs[KEY_CONTINGENCY_MATRIX], kwargs[KEY_CONTINGENCY_UNIQUE_LABELS], kwargs[
        KEY_CONTINGENCY_UNIQUE_PREDICTIONS]


def get_contingency_valid(kwargs: Dict[str, Any]) -> Tuple[Union[torch.Tensor, np.ndarray],
                                                           Union[torch.Tensor, np.ndarray],
                                                           Union[torch.Tensor, np.ndarray],
                                                           List[Tuple[int, int, int]]]:
    cm, ul, up = get_contingency_predictions(kwargs)
    first_elements = [(i, l, utils.find_first_index(l, up))
                      for i, l in enumerate(ul)
                      if np.isin(up, l).any()]
    return cm, ul, up, first_elements


KEY_PRECISON = 'key:precision'
class PrecisionComputation(MetricComputation):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = on_zero_divide

    def compute_precision(self, cm, index_label, index_prediction) -> float:
        tp = cm[index_label, index_prediction]
        divisor = cm[:, index_prediction].sum()
        if divisor == 0:
            return self.on_zero_divide
        return float(tp) / divisor

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        if not final:
            raise NotImplementedError
        cm, ul, up, first_elements = get_contingency_valid(kwargs)
        res = {l: self.compute_precision(cm, i, j) for i, l, j in first_elements}
        return res

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res[KEY_PRECISON] = res[self.name]
        return res


KEY_RECALL = 'key:recall'
class RecallComputation(TorchNumpyComputable):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = on_zero_divide

    def compute_recall(self, cm, index_label, index_prediction) -> float:
        tp = cm[index_label, index_prediction]
        divisor = cm[index_label].sum()
        if divisor == 0:
            return self.on_zero_divide
        return float(tp) / divisor

    def _compute(self, kwargs: Dict[str, Any], final: bool) \
            -> Dict[int, float]:
        if not final:
            raise NotImplementedError
        cm, ul, up, first_elements = get_contingency_valid(kwargs)
        res = {l: self.compute_recall(cm, i, j) for i, l, j in first_elements}
        return res

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res[KEY_RECALL] = res[self.name]
        return res


class FScoreComputation(TorchNumpyComputable):
    def __init__(self, name: str, beta: float, on_zero_divide: float = 0) -> None:
        assert beta > 0.0, f'0 or negative beta are not supported by F-Score. Given: {beta}'
        super().__init__(name)
        self.beta = beta ** 2
        self.on_zero_divide = on_zero_divide

    def _compute(self, kwargs: Dict[str, Any], final: bool) \
            -> Dict[int, float]:
        if not final:
            raise NotImplementedError
        precision, recall = kwargs[KEY_PRECISON], kwargs[KEY_RECALL]
        res = {l: (((1 + self.beta) * p * recall[l] / (self.beta * p + recall[l])) if p != 0 and recall[l] != 0 else self.on_zero_divide)
               for l, p in precision.items()}
        return res

    def clone(self):
        return self.__class__(self.name, math.sqrt(self.beta))


class IOUComputation(TorchNumpyComputable):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = on_zero_divide

    def compute_iou(self, cm: Union[torch.Tensor, np.ndarray], label_index: int, prediction_index: int):
        tp, fp, fn = tp_fp_fn(cm, label_index, prediction_index)
        if tp == fp == fn == 0: # avoid division by zero due to tp+fp+fn == 0
            return self.on_zero_divide
        elif tp == 0:
            return 0.0
        return float(tp) / (tp + fp + fn)

    def _compute(self, kwargs: Dict[str, Any], final: bool) \
            -> Dict[int, float]:
        if not final:
            raise NotImplementedError
        cm, ul, up, first_elements = get_contingency_valid(kwargs)
        res = {l: self.compute_iou(cm, i, j) for i, l, j in first_elements}
        return res


class AccuracyComputation(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[int, float]:
        if not final:
            raise NotImplementedError
        cm, ul = get_contingency(kwargs)
        return torch.trace(cm) / torch.sum(cm)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> float:
        if not final:
            raise NotImplementedError
        cm, ul, up, first_elements = get_contingency_valid(kwargs)
        res = [cm[i, j] for i, _, j in first_elements]
        return np.sum(np.array(res)) / np.sum(cm)

class LinearSumAssignment(SequenceMetricComputation):
    def __init__(self, sub_modules: List[MetricComputation]):
        super().__init__(sub_modules)
        self.allow_extract = False

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        if not final:
            raise NotImplementedError
        cm, ul, up = get_contingency_predictions(kwargs)
        match = sp_opt.linear_sum_assignment(cm.sum() - cm.T)
        # see also https://github.com/ZhiyuanDang/NNM/blob/aac3b2cbccac1880810d9ffec886c9271606fe4b/utils/evaluate_utils.py#L128
        match = np.array(list(zip(*match)))
        res_cm = np.zeros((ul.shape[0], ul.shape[0]))
        for p, l in match:
            res_cm[:, ul == l] += cm[:, up == p]

        kwargs[KEY_CONTINGENCY_MATRIX] = res_cm
        kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = ul
        res = super(LinearSumAssignment, self).forward(step, final, **kwargs)
        res[KEY_CONTINGENCY_MATRIX] = cm
        res[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = up
        return res

class MaximumAgreementComputation(SequenceMetricComputation):
    def __init__(self, sub_modules: List[MetricComputation]):
        super().__init__(sub_modules)
        self.allow_extract = False

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        if not final:
            raise NotImplementedError
        cm, ul, up = get_contingency_predictions(kwargs)
        assert ul.shape[0] > 0
        res_cm = np.zeros((ul.shape[0], ul.shape[0]), dtype=np.int32)
        #res_pred_map = {}
        for i, pred in enumerate(up):
            max_index = np.argmax(cm[:, i])
            res_cm[:, max_index] += cm[:, i]
            #res_pred_map[pred] = ul[max_index]

        kwargs[KEY_CONTINGENCY_MATRIX] = res_cm
        kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = ul
        res = super(MaximumAgreementComputation, self).forward(step, final, **kwargs)
        res[KEY_CONTINGENCY_MATRIX] = cm
        res[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = up
        return res


# TODO add pair confusion and derived metrics
KEY_PAIR_CONFUSION = 'pair_confusion_matrix'

class PairConfusionMatrix(SequenceMetricComputation):
    def __init__(self, sub_modules: List[MetricComputation]):
        super().__init__(sub_modules)
        self.allow_extract = False

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        if not final:
            raise NotImplementedError
        cm, _ = get_contingency(kwargs)
        cm = cm.astype(np.longlong)
        # direct copy of the implementation in sklearn.metrics.cluster.pair_confusion_matrix
        sum_squares = (cm ** 2).sum()
        pair_confusion = np.empty((2, 2), dtype=np.longlong)
        n_samples = cm.sum()
        pair_confusion[1, 1] = sum_squares - n_samples
        pair_confusion[0, 1] = cm.dot(np.ravel(cm.sum(axis=0))).sum() - sum_squares
        pair_confusion[1, 0] = cm.transpose().dot(np.ravel(cm.sum(axis=1))).sum() - sum_squares
        pair_confusion[0, 0] = n_samples ** 2 - pair_confusion[0, 1] - pair_confusion[1, 0] - sum_squares
        # print(pair_confusion)
        kwargs[KEY_PAIR_CONFUSION] = pair_confusion
        res = super(PairConfusionMatrix, self).forward(step, final, **kwargs)
        del res[KEY_PAIR_CONFUSION]
        # if np.all(pair_confusion.sum(axis=0) != 0) and np.all(pair_confusion.sum(axis=1) != 0):
        #     pass
        # else:
        #     #print('Skipping Pair Confusion calculations as there is not enough relevant data.')
        #     res = kwargs
        return res

def _perform_ncpc_computation(contingency: np.ndarray) -> np.ndarray:
    # see "A comparison of external clustering evaluation indices in the
    # context of imbalanced data sets" by de Souto et al. This is direct implementation of their Method in
    # numba compiled python
    a_N = 0.0
    d_N = 0.0
    cstar_relevant_classes = 0
    contingency = contingency.astype(np.int64)
    num_objects_per_class = np.empty(contingency.shape[0], dtype=np.float64)
    for clazz in numba.prange(contingency.shape[0]): # cluster is referred to with n_{i.}
        num_objects_per_class[clazz] = contingency[clazz].sum()
        if num_objects_per_class[clazz] > 1:
            cstar_relevant_classes+=1

    for clazz in numba.prange(contingency.shape[0]): # clazz is referred to with n_{.j}
        # this corresponds to the n_{.j} choose 2 in the a_N sum
        # the name originates from the description in the text
        # notice that the 2 that appears in both binomial coefficients and therefore cancels out
        max_positive_agreement = num_objects_per_class[clazz] * (num_objects_per_class[clazz] - 1)
        for cluster in numba.prange(contingency.shape[1]):
            # if contingency[j, i] == 1 , then we would just add 0...
            # if contingency[j, i] < 1, then the binomial coefficient to be evaluated is not defined and I
            # chose to drop it from the calculation
            if contingency[clazz, cluster] > 1: # => num_objects_per_class[clazz] > 1  => no nan values in result
                a_N+= float(contingency[clazz, cluster]) * float(contingency[clazz, cluster] - 1) / max_positive_agreement
            for s in numba.prange(cluster+1, contingency.shape[1]):
                for k in numba.prange(clazz+1, contingency.shape[0]):
                    true_negative_cases = float(contingency[clazz, cluster]) * float(contingency[k, s])
                    # true_negative_cases > 0 =>
                    # => num_objects_per_class[clazz] > 0 and num_objects_per_class[k] > 0 =>
                    # => no nan values in result
                    if true_negative_cases > 0:
                        max_num_true_negative_cases = num_objects_per_class[clazz] * num_objects_per_class[k]
                        d_N += true_negative_cases / max_num_true_negative_cases
    if cstar_relevant_classes > 0:
        a_N/= cstar_relevant_classes
        if cstar_relevant_classes > 1:
            d_N/= (cstar_relevant_classes * (cstar_relevant_classes - 1) / 2.0)
        else:
            # if there is just one class, then you never have object pairs that are in different classes
            # hence by the logic of Sen1Floods11 this should be one
            d_N = 1.0
    else:
        a_N = 1.0
        d_N = 1.0
    b_N = 1.0 - a_N
    c_N = 1.0 - d_N
    return np.array([[a_N, b_N], [c_N, d_N]])

_perform_ncpc_computation_numba = utils.njit(parallel=False)(_perform_ncpc_computation)
_perform_ncpc_computation_numba_parallel = utils.njit(parallel=True)(_perform_ncpc_computation)

class NormalizedClassSizePairConfusionMatrix(SequenceMetricComputation):
    def __init__(self, sub_modules: List[MetricComputation], execute_parallel: bool = True):
        super().__init__(sub_modules)
        self.allow_extract = False
        self._perform_computation = _perform_ncpc_computation_numba_parallel if execute_parallel else _perform_ncpc_computation_numba


    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, Any]:
        if not final:
            raise NotImplementedError
        cm, _ = get_contingency(kwargs)
        # direct copy of the implementation in sklearn.metrics.cluster.pair_confusion_matrix
        pair_confusion = self._perform_computation(cm)
        # print(pair_confusion)
        kwargs[KEY_PAIR_CONFUSION] = pair_confusion
        res = super(NormalizedClassSizePairConfusionMatrix, self).forward(step, final, **kwargs)
        del res[KEY_PAIR_CONFUSION]
        # if np.all(pair_confusion.sum(axis=0) != 0) and np.all(pair_confusion.sum(axis=1) != 0):
        #     kwargs[KEY_PAIR_CONFUSION] = pair_confusion
        #     res = super(NormalizedClassSizePairConfusionMatrix, self).forward(step, final, **kwargs)
        #     del res[KEY_PAIR_CONFUSION]
        # else:
        #     #print('Skipping Pair Confusion calculations as there is not enough relevant data.')
        #     res = kwargs
        return res

def get_pair_confusion(kwargs: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
    assert KEY_PAIR_CONFUSION in kwargs
    return kwargs[KEY_PAIR_CONFUSION]

def entropy_np(occurrences: np.ndarray):
    # Note that occurrences can be calculated from the contingency as depicted by label and prediction entropy
    mask = occurrences > 0
    if not np.any(mask):
        return 0
    pi = occurrences[mask]
    # direct copy of sklearn.metrics.cluster.entropy
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))

class LabelEntropy(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, _ = get_contingency(kwargs)
        return entropy_np(np.ravel(cm.sum(axis=1)))

class PredictionEntropy(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, _ = get_contingency(kwargs)
        return entropy_np(np.ravel(cm.sum(axis=0)))

KEY_MUTUAL_INFORMATION = 'mutual_information'
from sklearn.metrics.cluster import mutual_info_score
class MutualInfoScore(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, _ = get_contingency(kwargs)
        # direct copy of the implementation in sklearn.metrics.cluster.mutual_info_score
        nzx, nzy = np.nonzero(cm)
        nz_val = cm[nzx, nzy]

        contingency_sum = cm.sum()
        pi = np.ravel(cm.sum(axis=1))
        pj = np.ravel(cm.sum(axis=0))
        log_contingency_nm = np.log(nz_val)
        contingency_nm = nz_val / contingency_sum
        # Don't need to calculate the full outer product, just for non-zeroes
        outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
            np.int64, copy=False
        )
        log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
        mi = (
                contingency_nm * (log_contingency_nm - np.log(contingency_sum))
                + contingency_nm * log_outer
        )
        mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
        return np.clip(mi.sum(), 0.0, None)

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res[KEY_MUTUAL_INFORMATION] = res[self.name]
        return res


from sklearn.metrics.cluster import expected_mutual_information, normalized_mutual_info_score
# copy from sklearn.metrics.cluster
def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError(
            "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'"
        )


class AdjustedMutualInfoScore(MetricComputation):
    def __init__(self, name: str, average_method: str = 'arithmetic') -> None:
        super().__init__(name)
        self.average_method = average_method

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        # copy of sklearn.metrics.cluster.adjusted_mutual_info_score
        mi = kwargs[KEY_MUTUAL_INFORMATION]
        cm, _ = get_contingency(kwargs)
        h_true, h_pred = entropy_np(np.ravel(cm.sum(axis=1))), entropy_np(np.ravel(cm.sum(axis=0)))
        normalizer = _generalized_average(h_true, h_pred, self.average_method)
        emi = expected_mutual_information(cm, cm.sum())
        denominator = normalizer - emi
        # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
        # normalizer should always be >= emi, but because of floating-point
        # representation, sometimes emi is slightly larger. Correct this
        # by preserving the sign.
        if denominator < 0:
            denominator = min(denominator, -np.finfo(np.float64).eps)
        else:
            denominator = max(denominator, np.finfo(np.float64).eps)
        ami = (mi - emi) / denominator
        return ami

class NormalizedMutualInfoScore(MetricComputation):
    def __init__(self, name: str, average_method: str = 'arithmetic') -> None:
        super().__init__(name)
        self.average_method = average_method

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        # copy of sklearn.metrics.cluster.adjusted_mutual_info_score
        mi = kwargs[KEY_MUTUAL_INFORMATION]
        cm, _ = get_contingency(kwargs)
        h_true, h_pred = entropy_np(np.ravel(cm.sum(axis=1))), entropy_np(np.ravel(cm.sum(axis=0)))
        normalizer = _generalized_average(h_true, h_pred, self.average_method)
        # Calculate the expected value for the mutual information
        # Calculate entropy for each labeling
        normalizer = max(normalizer, np.finfo(np.float64).eps)
        nmi = mi / normalizer
        return nmi

KEY_HOMOGENITY = 'homogenity'
class Homogenity(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        # copy of sklearn.metrics.cluster.adjusted_mutual_info_score
        mi = kwargs[KEY_MUTUAL_INFORMATION]
        cm, _ = get_contingency(kwargs)
        pred_entropy = entropy_np(np.ravel(cm.sum(axis=1)))
        return mi if pred_entropy == 0 else mi / pred_entropy

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res[KEY_HOMOGENITY] = res[self.name]
        return res


KEY_COMPLETENESS = 'completeness'
class Completeness(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        # copy of sklearn.metrics.cluster.adjusted_mutual_info_score
        mi = kwargs[KEY_MUTUAL_INFORMATION]
        cm, _ = get_contingency(kwargs)
        label_entropy = entropy_np(np.ravel(cm.sum(axis=0)))
        return mi if label_entropy == 0 else mi / label_entropy

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super().forward(step, final, **kwargs)
        res[KEY_COMPLETENESS] = res[self.name]
        return res

class VMeasure(MetricComputation):
    def __init__(self, name: str, beta: float) -> None:
        super().__init__(name)
        self.beta = beta ** 2

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        # copy of sklearn.metrics.cluster.adjusted_mutual_info_score
        #mi = kwargs[KEY_MUTUAL_INFORMATION]
        cm, _ = get_contingency(kwargs)
        homogenity = kwargs[KEY_HOMOGENITY]
        completeness = kwargs[KEY_COMPLETENESS]
        if homogenity == 0.0 or completeness == 0.0:
            return 0.0

        return (1 + self.beta) * homogenity * completeness / (self.beta * homogenity + completeness)

class RandScore(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm: np.ndarray = get_pair_confusion(kwargs)
        # direct copy of the implementation in sklearn.metrics.cluster.rand_score

        numerator = cm.diagonal().sum()
        denominator = cm.sum()

        if numerator == denominator or denominator == 0:
            return 1.0

        return numerator / denominator

from sklearn.metrics.cluster import adjusted_rand_score
class AdjustedRandScore(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm: np.ndarray = get_pair_confusion(kwargs)
        (tn, fp), (fn, tp) = cm
        # convert to python int to avoid overflows
        (tn, fp), (fn, tp) = (int(tn), int(fp)), (int(fn), int(tp))
        # direct copy of the implementation in sklearn.metrics.cluster.rand_score

        # Special cases: empty data or full agreement
        if fn == 0 and fp == 0:
            return 1.0
        divisor = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        if divisor == 0:
            return 1.0

        return 2.0 * (tp * tn - fn * fp) / divisor


class PairPrecisionComputation(MetricComputation):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = float(on_zero_divide)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> float:
        if not final:
            raise NotImplementedError
        cm = get_pair_confusion(kwargs)
        sum = cm[:, 0].sum()
        if sum == 0:
            return self.on_zero_divide
        return cm[0, 0] / sum


class PairRecallComputation(TorchNumpyComputable):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = float(on_zero_divide)

    def _compute(self, kwargs: Dict[str, Any], final: bool) -> float:
        if not final:
            raise NotImplementedError
        cm = get_pair_confusion(kwargs)
        sum = cm[0].sum()
        if sum == 0:
            return self.on_zero_divide
        return cm[0, 0] / sum

class FowlkesMallowsScore(MetricComputation):

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm: np.ndarray = get_pair_confusion(kwargs)
        (tn, fp), (fn, tp) = cm

        if tp == 0: # note that this results in a check for tp + fp ==0 or tp + fn == 0 being unnecessary
            return 0

        return tp * np.sqrt((1.0 / (tp + fp)) * (1.0 / (tp + fn)))

class PairFScore(MetricComputation):
    def __init__(self, name: str, beta: float, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.beta = beta**2
        self.on_zero_divide = float(on_zero_divide)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm: np.ndarray = get_pair_confusion(kwargs)
        (tn, fp), (fn, tp) = cm

        if tp == 0:
            if fp == 0 and fn == 0:
                return self.on_zero_divide
            return 0
        tp *= (1 + self.beta)
        fn *= self.beta
        return tp / (tp + fn + fp)

class PairIOUScore(MetricComputation):
    def __init__(self, name: str, on_zero_divide: int = 1) -> None:
        super().__init__(name)
        self.on_zero_divide = float(on_zero_divide)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm: np.ndarray = get_pair_confusion(kwargs)
        (tn, fp), (fn, tp) = cm

        if tp == 0:
            if fp == 0 and fn == 0:
                return self.on_zero_divide
            return 0
        return tp / (tp + fn + fp)

class LabelDistribution(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, ul = get_contingency(kwargs)
        return {l: cm[ul == l].sum() for l in ul}

class ClusterFMeasure(MetricComputation):
    def __init__(self, name: str, beta: float, on_zero_divide: float = 1.) -> None:
        super().__init__(name)
        self.beta = beta**2
        self.on_zero_divide = on_zero_divide

    @staticmethod
    @utils.njit(parallel = False)
    def _compute_cluster_f(contingency: np.ndarray, beta: float, on_zero_divide: float) -> float:
        f_max = 0.0
        n_samples = contingency.sum()
        for label_idx in range(contingency.shape[0]):
            label_sum = contingency[label_idx].sum()
            max_row_f = 0.0
            for prediction_idx in range(contingency.shape[1]):
                prediction_sum = contingency[:, prediction_idx].sum()
                if prediction_sum == 0 and label_sum == 0:
                    if on_zero_divide > max_row_f:
                        max_row_f = on_zero_divide
                    else:
                        continue
                elif contingency[label_idx, prediction_idx] == 0:
                    continue
                else:
                    # notice the identity: label_sum = fn + tp and row_sum = fp + tp
                    # Thus beta * label_sum + prediction_sum = beta * fn + beta * tp + fp + tp =
                    #  = (1 + beta) * tp + beta * fn + fp = denominator of F-Score
                    f_score = (1+ beta) * contingency[label_idx, prediction_idx] / (beta * label_sum + prediction_sum)
                    if f_score > max_row_f:
                        max_row_f = f_score
                if max_row_f >= 1.0:
                    break
            f_max += max_row_f * label_sum / n_samples
        return f_max

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, ul = get_contingency(kwargs)
        return self._compute_cluster_f(cm, self.beta, self.on_zero_divide)

KEY_BCUBED_PRECISON = 'key:bcubed_precision'
class BCubedPrecision(MetricComputation):

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, ul = get_contingency(kwargs)
        # copy of https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py
        pred_sums: np.ndarray = cm.sum(axis = 0)
        precision = (cm.T ** 2) / (pred_sums * pred_sums)
        return (precision * pred_sums).sum() / pred_sums.sum()

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super(BCubedPrecision, self).forward(step, final, **kwargs)
        res[KEY_BCUBED_PRECISON] = res[self.name]
        return res


KEY_BCUBED_Recall = 'key:bcubed_recall'
class BCubedRecall(MetricComputation):
    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Any:
        cm, ul = get_contingency(kwargs)
        # copy of https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py
        class_sums: np.ndarray = cm.sum(axis = 1)
        recall = (cm **2) / (class_sums * class_sums)
        return (recall * class_sums).sum() / class_sums.sum()

    def forward(self, step: Optional[int] = None, final: bool = True, **kwargs) -> Dict[str, any]:
        res = super(BCubedRecall, self).forward(step, final, **kwargs)
        res[KEY_BCUBED_Recall] = res[self.name]
        return res

class BCubedF(MetricComputation):
    def __init__(self, name: str, beta: float) -> None:
        super().__init__(name)
        self.beta = beta ** 2

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> float:
        bp, br = kwargs[KEY_BCUBED_PRECISON], kwargs[KEY_BCUBED_Recall]
        return (1 + self.beta) * bp * br / (self.beta * bp + br)


# import sklearn.metrics.cluster as cl
# cl.entropy()
def sklearn_numpy_prepare(predictions: np.ndarray, data_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data_points = np.transpose(data_points, (data_points.ndim - 3,) + tuple(i for i in range(data_points.ndim)
                                                                            if i != data_points.ndim - 3))
    data_points = data_points.reshape(data_points.shape[0], -1).T
    predictions = predictions.flatten()
    return predictions, data_points


def sklearn_torch_prepare(predictions: torch.Tensor, data_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    data_points = torch.permute(data_points, (data_points.ndim - 3,) + tuple(i for i in range(data_points.ndim)
                                                                             if i != data_points.ndim - 3))
    data_points = torch.transpose(data_points.reshape(data_points.shape[0], -1), 0, 1)
    predictions = predictions.flatten()
    return predictions, data_points

class ClusterSpatialDistributionComputation(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, np.ndarray]:
        return self._compute_numpy({k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()},
                                   final)

    @staticmethod
    @utils.njit(fastmath=False, parallel=True)
    def _measure_compute_numba(predictions: np.ndarray, data_points: np.ndarray, up: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centroids = np.empty((up.shape[0], data_points.shape[1]))
        centroid_sums = np.empty((up.shape[0], data_points.shape[1]))
        intra_dists = np.empty(up.shape[0])
        for i in numba.prange(up.shape[0]):
            masked_points = data_points[predictions == up[i]]
            centroid_sums[i] = 0
            for j in numba.prange(masked_points.shape[0]):
                centroid_sums[i] += masked_points[j]
            centroids[i] = centroid_sums[i] / masked_points.shape[0]
            intra_dists[i] = np.sum((masked_points - centroids[i]) ** 2)
        center = np.zeros(data_points.shape[1])
        for i in numba.prange(up.shape[0]):
            center += centroid_sums[i]
        center /= data_points.shape[0]
        return centroids, intra_dists, center, data_points.shape[0]

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[str, np.ndarray]:
        assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
        predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
        if KEY_CONTINGENCY_UNIQUE_PREDICTIONS in kwargs:
            up = kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
        else:
            up = np.unique(predictions)
        predictions, data_points = sklearn_numpy_prepare(predictions, data_points)
        centroids, intra_dists, center, num_samples = self._measure_compute_numba(predictions, data_points, up)
        return {'center': center, 'num_samples': num_samples, 'intra_dists': intra_dists, 'centroids': centroids}


# class SSEScore(MetricComputation):
#     def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[int, float]:
#         def cluster_sse(cluster_data) -> float:
#             mean = torch.mean(cluster_data, dim=0)
#             dif = cluster_data - mean
#             return float(torch.sum(dif ** 2))
#
#         assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
#         predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
#         up = torch.unique(predictions, sorted=False)
#         res = {p: cluster_sse(data_points[predictions == p]) for p in up}
#         return res
#
#     def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[int, float]:
#         def cluster_sse(cluster_data) -> float:
#             mean = np.mean(cluster_data, axis=0)
#             dif = cluster_data - mean
#             return float(np.sum(dif ** 2))
#
#         assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
#         predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
#         up = np.unique(predictions)
#         res = {p: cluster_sse(data_points[predictions == p]) for p in up}
#         return res
#
# class SSBScore(MetricComputation):
#     def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> Dict[int, float]:
#         assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
#         predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
#         center = np.mean(data_points, axis=0)
#         up, up_counts = np.unique(predictions, return_counts=True)
#         res = {p: count * (center - np.mean(data_points[predictions == p], axis=0))**2
#                for p, count in zip(up, up_counts)}
#         return res
#
#     def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> Dict[int, float]:
#         assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
#         predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
#         center = np.mean(data_points, axis=0)
#         up, up_counts = np.unique(predictions, return_counts=True)
#         res = {p: count * (center - np.mean(data_points[predictions == p], axis=0))**2
#                for p, count in zip(up, up_counts)}
#         return res



from sklearn.metrics.cluster import davies_bouldin_score
class DaviesBouldinScore(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> float:
        return self._compute_numpy({k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()},
                                   final)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> float:
        assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
        cm, ul, up = get_contingency_predictions(kwargs)
        center, intra_dists, centroids = kwargs['center'], kwargs['intra_dists'], kwargs['centroids']
        predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
        predictions, data_points = sklearn_numpy_prepare(predictions, data_points)
        res = davies_bouldin_score(data_points, predictions.flatten())
        return float(res)

# Computable from SSE and SSB
class CalinskiHarabaszScore(MetricComputation):
    def _compute_torch(self, kwargs: Dict[str, Any], final: bool) -> float:
        return self._compute_numpy({k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()},
                                   final)

    def _compute_numpy(self, kwargs: Dict[str, Any], final: bool) -> float:
        assert KEY_PREDICTIONS in kwargs and KEY_INPUT in kwargs
        predictions, data_points = kwargs[KEY_PREDICTIONS], kwargs[KEY_INPUT]
        res = calinski_harabasz_score(data_points.reshape((-1, data_points.shape[-1])),
                                      predictions.flatten())
        return float(res)
KwargsResult = Tuple[Dict[str, Any], Optional[data.Dataset], Optional[data.Dataset], Optional[data.Dataset]]


class MetricsModule(MultiPipeline):
    def __init__(self, prediction_criterion: Optional[SelectionCriterion] = None,
                 label_criterion: Optional[SelectionCriterion] = None,
                 source_criterion: Optional[SelectionCriterion] = None,
                 per_data_computation: Optional[MetricComputation] = None,
                 per_region_computation: Optional[MetricComputation] = None,
                 total_computation: Optional[MetricComputation] = None,
                 delete_prediction: bool = False,
                 delete_label: bool = False,
                 delete_source: bool = False):
        super().__init__()
        self.prediction_criterion = prediction_criterion
        self.label_criterion = label_criterion
        self.source_criterion = source_criterion
        self.per_data_computation = per_data_computation
        self.per_region_computation = per_region_computation
        self.total_computation = total_computation
        self.delete_prediction = delete_prediction
        self.delete_label = delete_label
        self.delete_source = delete_source

    def _construct_kwargs(self,
                          label_dataset: Optional[data.Dataset],
                          pred_dataset: Optional[data.Dataset],
                          input_dataset: Optional[data.Dataset],
                          filter: Optional[Union[int, Tuple[int, int], List[int]]] = None) -> KwargsResult:
        require_array = filter is None or type(filter) is not int

        def perform_filter(dataset: Union[ArrayInMemoryDataset, ShapelessInMemoryDataset]) -> Union[torch.Tensor, np.ndarray]:
            if filter is None:
                return dataset.data  # utils.move_channel_to_position(dataset.data, dataset.data.ndim-1, 1)
            elif type(filter) is int:
                filtered = dataset.data[filter] # todo fix for landuyt
                # res = utils.move_channel_to_position(filtered, filtered.ndim-1, 0)
                return filtered  # res
            else:
                filtered = dataset.data[filter[0]:filter[1]] if isinstance(filter, tuple) else dataset.data[
                    tuple(filter)]
                return filtered  # utils.move_channel_to_position(filtered, filtered.ndim-1,
                #                               0 if filtered.ndim < dataset.data.ndim else 1)

        kwargs = {}
        if label_dataset is not None:
            label_dataset = ArrayInMemoryDataset(label_dataset) if require_array else ShapelessInMemoryDataset(
                label_dataset)
            kwargs[KEY_LABELS] = perform_filter(label_dataset)
        if pred_dataset is not None:
            pred_dataset = ArrayInMemoryDataset(pred_dataset) if require_array else ShapelessInMemoryDataset(
                pred_dataset)
            kwargs[KEY_PREDICTIONS] = perform_filter(pred_dataset)
        if input_dataset is not None:
            input_dataset = ArrayInMemoryDataset(input_dataset) if require_array else ShapelessInMemoryDataset(
                input_dataset)
            kwargs[KEY_INPUT] = perform_filter(input_dataset)
        return kwargs, label_dataset, pred_dataset, input_dataset

    def _perform_computation(self, summary: Summary, computation: MetricComputation,
                             kwargs: Dict[str, Any], timeit: bool):
        t = time.time()
        try:
            res = computation.forward(None, True, **kwargs)
        except Exception as e:
            self.print(summary, f'Encountered exception {str(type(e))} with args: {str(e.args)}. Skipping!!!',
                       is_err=False)
            print(traceback.format_exc(), file=sys.stderr)
            return None
        else:
            t = time.time() - t
            if timeit:
                self.print(summary, f'Computation completed in {t:.3f}s and produced keys {res.keys()}')
            return res

    def _total_computation(self, summary: Summary, any_meta: Meta,
                           label_dataset: Optional[data.Dataset],
                           pred_dataset: Optional[data.Dataset],
                           input_dataset: Optional[data.Dataset]) \
            -> Tuple[Optional[data.Dataset], Optional[data.Dataset], Optional[data.Dataset]]:
        self.print(summary, f'Constructing kwargs for total-computation.')
        t = time.time()
        kwargs, label_dataset, pred_dataset, input_dataset = self._construct_kwargs(label_dataset, pred_dataset,
                                                                                    input_dataset)
        t = time.time() - t
        self.print(summary,
                   f'Kwargs construction completed in {t:.3f}s. Starting total-computation with input keys {str(kwargs.keys())}.')
        self.total_computation.set_hook(summary.run_hook.set_auto_flush(True, False))
        self._perform_computation(summary, self.total_computation, kwargs, True)
        t = time.time()
        summary.run_hook.flush_metrics()
        self.total_computation.set_hook(None)
        t = time.time() - t
        self.print(summary, f'Finished saving. Took {t:.3f}s.')
        return label_dataset, pred_dataset, input_dataset

    def _per_region_computation(self, summary: Summary, any_meta: Meta,
                                label_dataset: Optional[data.Dataset],
                                pred_dataset: Optional[data.Dataset],
                                input_dataset: Optional[data.Dataset]) \
            -> Tuple[Optional[data.Dataset], Optional[data.Dataset], Optional[data.Dataset]]:
        self.print(summary, f'Grouping entries for per-region metric computation.')
        grouping = [(k, [i for i, info in enumerate(any_meta.per_item_info) if info.region == k])
                    for k in set([info.region for info in any_meta.per_item_info])]
        self.print(summary, f'Found {len(grouping)} different regions.')
        summary.run_hook.set_auto_flush(False, True)
        for region, indices in grouping:
            t = time.time()
            if any(map(lambda i: indices[i] + 1 != indices[i + 1], range(len(indices) - 1))):
                kwargs, label_dataset, pred_dataset, input_dataset = self._construct_kwargs(label_dataset, pred_dataset,
                                                                                            input_dataset,
                                                                                            filter=indices)
                self.print(summary, f'Kwargs computation completed in {time.time() - t:.3f}s. '
                                    f'Performing region computation for non-consecutive region {region} with {len(indices)} elements.')
            else:
                kwargs, label_dataset, pred_dataset, input_dataset = self._construct_kwargs(label_dataset, pred_dataset,
                                                                                            input_dataset, filter=(
                    indices[0], indices[-1]))
                self.print(summary, f'Kwargs computation completed in {time.time() - t:.3f}s. '
                                    f'Performing region computation for consecutive region {region} with {len(indices)} elements.')
            computation = self.per_region_computation.clone() \
                .append_to_name(region) \
                .set_hook(hook=summary.run_hook)
            self._perform_computation(summary, computation, kwargs, True)
        t = time.time()
        summary.run_hook.flush_metrics()
        t = time.time() - t
        self.print(summary, f'Finished saving. Took {t:.3f}s.')
        return label_dataset, pred_dataset, input_dataset

    def _per_data_computation(self, summary: Summary, any_meta: Meta,
                              label_dataset: Optional[data.Dataset],
                              pred_dataset: Optional[data.Dataset],
                              input_dataset: Optional[data.Dataset]) \
            -> Tuple[Optional[data.Dataset], Optional[data.Dataset], Optional[data.Dataset]]:
        self.print(summary, f'Starting per-data-computation.')
        summary.run_hook.set_auto_flush(False, True)
        t = time.time()
        for i in range(len(any_meta.per_item_info)):
            kwargs, label_dataset, pred_dataset, input_dataset = self._construct_kwargs(label_dataset, pred_dataset,
                                                                                        input_dataset, filter=i)
            computation = self.per_data_computation.clone() \
                .append_to_name(str(i)) \
                .set_hook(hook=summary.run_hook)
            self._perform_computation(summary, computation, kwargs, False)
        t = time.time() - t
        t1 = time.time()
        summary.run_hook.flush_metrics()
        t1 = time.time() - t1
        self.print(summary,
                   f'Per-data-computation completed in {t:.3f}s - average {t / len(any_meta.per_item_info):.3f}s. Save took {t1:.3f}s.')
        return label_dataset, pred_dataset, input_dataset

    def __call__(self, summary: Summary) -> Summary:
        def none_or_summary_value(criterion: Optional[SelectionCriterion]) \
                -> Tuple[Optional[data.Dataset], Optional[Meta]]:
            return (None, (None, None)) if criterion is None else summary.by_criterion(criterion, delete=True,
                                                                                       return_name=True)

        def integrate_into_summary(criterion: Optional[SelectionCriterion], delete: bool,
                                   dataset: Optional[data.Dataset], meta: Optional[Meta], name: Optional[str]):
            if not delete and criterion is not None:
                assert dataset is not None and meta is not None and name is not None
                summary.add(dataset, meta, name)

        self.print(summary, 'Extracting available data from summary.')
        label_name, (label_dataset, label_meta) = none_or_summary_value(self.label_criterion)
        pred_name, (pred_dataset, pred_meta) = none_or_summary_value(self.prediction_criterion)
        input_name, (input_dataset, input_meta) = none_or_summary_value(self.source_criterion)
        available_metas = [meta for meta in (label_meta, pred_meta, input_meta) if meta is not None]
        if available_metas:
            if any(map(lambda m: len(m.per_item_info) != len(available_metas[0].per_item_info), available_metas)):
                raise RuntimeError('Specified datasets of varying sizes for Metric calculation. This is not supported!')
            if len(available_metas) > 1:
                for i, info in enumerate(available_metas[0].per_item_info):
                    if any(map(lambda m: m.per_item_info[i] != info, available_metas)):
                        self.print(summary, f'Found differing item info at index {i}. Results may be unpredictable!',
                                   is_err=True)
            self.print(summary, f'Comparing metrics with {len(available_metas)} inputs.')
        else:
            self.print(summary, 'No inputs provided to metric calculation. This should not happen!', is_err=True)

        if self.total_computation is not None:
            label_dataset, pred_dataset, input_dataset = \
                self._total_computation(summary,
                                        next(m for m in (label_meta, pred_meta, input_meta) if m is not None),
                                        label_dataset, pred_dataset, input_dataset)
        if self.per_region_computation is not None:
            label_dataset, pred_dataset, input_dataset = \
                self._per_region_computation(summary,
                                             next(m for m in (label_meta, pred_meta, input_meta) if m is not None),
                                             label_dataset, pred_dataset, input_dataset)
        if self.per_data_computation is not None:
            label_dataset, pred_dataset, input_dataset = \
                self._per_data_computation(summary,
                                           next(m for m in (label_meta, pred_meta, input_meta) if m is not None),
                                           label_dataset, pred_dataset, input_dataset)

        integrate_into_summary(self.label_criterion, self.delete_label, label_dataset, label_meta, label_name)
        integrate_into_summary(self.prediction_criterion, self.delete_prediction, pred_dataset, pred_meta, pred_name)
        integrate_into_summary(self.source_criterion, self.delete_source, input_dataset, input_meta, input_name)

        return summary

def accumulate_contingency(contingencies: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], exclude_clouds: bool,
                           normalize: bool = False) -> Dict[str, np.ndarray]:
    def add_mask_safely(cm: np.ndarray, contingency: np.ndarray, labels: np.ndarray, val: int, index: int):
        mask = labels == val
        if np.any(mask):
            cm[index] = contingency[mask]

    cm = np.zeros(((2 if exclude_clouds else 3), 3 if normalize and not exclude_clouds else (2 if normalize else contingencies[0][2].shape[0])),
                  dtype=np.int64)
    res_labels = np.array([0, 1] if exclude_clouds else [-1, 0, 1])
    for contingency, labels, predictions in contingencies:
        if normalize:
            contingency, labels, predictions = normalize_contingency(contingency, labels, predictions, exclude_clouds)
        if exclude_clouds:
            if np.any(predictions == -1):
                contingency = contingency[:, predictions != -1]
            add_mask_safely(cm, contingency, labels, 0, 0)
            add_mask_safely(cm, contingency, labels, 1, 1)
        else:
            add_mask_safely(cm, contingency, labels, -1, 0)
            add_mask_safely(cm, contingency, labels, 0, 1)
            add_mask_safely(cm, contingency, labels, 1, 2)

    return {KEY_CONTINGENCY_MATRIX: cm,
            KEY_CONTINGENCY_UNIQUE_LABELS: res_labels,
            KEY_CONTINGENCY_UNIQUE_PREDICTIONS: (np.array([-1, 0, 1])  if normalize and not exclude_clouds else (np.array([0, 1] if normalize else contingencies[0][2])))}


@utils.njit()
def _batch_calculation(bin_mat: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Calculate batchwise mean and standard deviation for the given set of confusion matrices. Notice that the result
    is order dependent!

    Calculations are performed with double precision.

    :param bin_mat: A "flattened" binary confusion matrix of shape (N, 4). The first axis indexes the index of the data
                    point (image) for which statistics were calculated wheras the second axis indexes the flattened
                    confusion matrix ([TP, FP, FN, TN]).
    :param batch_size: The size of batches to use for calculating the metrics.
    :return: An array consisting of mean and standard deviations of IOU, Accuracy, Precision, Recall (in this order).
             The first axis indexes the metric, the second axis indexes mean vs. standard deviation.
    """
    num_combined = bin_mat.shape[0] // batch_size
    if bin_mat.shape[0] % batch_size != 0:
        num_combined += 1
    ious = np.empty(num_combined, np.float64)
    accs = np.empty(num_combined, np.float64)
    precision = np.empty(num_combined, np.float64)
    recall = np.empty(num_combined, np.float64)
    f1_score = np.empty(num_combined, np.float64)
    for i in range(num_combined):
        min_idx = i*batch_size
        max_idx = min((i+1) * batch_size, bin_mat.shape[0])
        tp = bin_mat[min_idx:max_idx, 0].sum()
        fp = bin_mat[min_idx:max_idx, 1].sum()
        fn = bin_mat[min_idx:max_idx, 2].sum()
        tn = bin_mat[min_idx:max_idx, 3].sum()
        if tp == fp == fn == tn == 0:
            accs[i] = np.nan
        else:
            accs[i] = float(tp + tn) / float(tp + fp + fn + tn)
        if tp > 0:
            ious[i] = float(tp) / float(tp + fp + fn)
            recall[i] = float(tp) / float(tp + fp)
            precision[i] = float(tp) / float(tp + fn)
        elif fp == 0 and fn == 0:
            ious[i] = 1.0
            recall[i] = 1.0
            precision[i] = 1.0
        elif fp == 0:
            ious[i] = 0.0
            recall[i] = 1.0
            precision[i] = 0.0
        elif fn == 0:
            ious[i] = 0.0
            recall[i] = 0.0
            precision[i] = 1.0
        else:
            ious[i] = 0.0
            recall[i] = 0.0
            precision[i] = 0.0
        if precision[i] == recall[i] == 0:
            f1_score[i] = 1.0
        else:
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return np.array([[np.mean(ious), np.std(ious)],
                     [np.mean(accs), np.std(accs)],
                     [np.mean(precision), np.std(precision)],
                     [np.mean(recall), np.std(recall)],
                     [np.mean(f1_score), np.std(f1_score)]])

@utils.njit(parallel=True)
def _batchwise_computation(contingency_matrices: np.ndarray, n_iter: int,
                           batch_size: int, random_state:int) -> np.ndarray:
    """
    Perform n_iter many batchwise metric calculations on the contingency_matrices and return statistics over
    the calculated mean and standard deviations. Notice that for those statistics, all nan results are ignored
    (these can occur for particularly unlucky orderings where some batches contain no data of interest).

    Batchwise metrics are calculated for the class represented by label index 0 of the contingency matrices.

    Calculations are performed with double precision and iterations are cpu-parallelized.

    :param contingency_matrices: Confusion matrices to use for batchwise calculations. Should have shape (N, L, P) where
                                 N is the number of datapoints, L the number of distinct labels and P the number of
                                 distinct predictions.
    :param n_iter: Number of iterations to test
    :param batch_size: The batch size to use for metric calculations
    :param random_state: A random seed to produce deterministic results (defines the shuffle order)
    :return: Array of shape (4,2,4) where the first axis indexes the metric (IOU, Accuracy, Precision, Recall
             (in this order)), the second axis indexes whether the value represents the batchwise mean over or the
             batchwise standard deviation over iterations and the third axis indexes whether the mean, std., min or max
             is calculated over all performed iterations.
    """
    computed = np.empty((n_iter, 5, 2), dtype=np.float64)
    np.random.seed(random_state)
    binary_matrix = np.empty((contingency_matrices.shape[0], 4), dtype=np.int64)
    for i in numba.prange(contingency_matrices.shape[0]):
        tp = contingency_matrices[i,0,0]
        binary_matrix[i,0] = tp
        binary_matrix[i,1] = contingency_matrices[i,0].sum() - tp
        binary_matrix[i,2] = contingency_matrices[i,:,0].sum() - tp
        binary_matrix[i,3] = np.diag(contingency_matrices[i]).sum() - tp
    for i in numba.prange(n_iter):
        shuffled_mat = np.random.permutation(binary_matrix)
        computed[i] = _batch_calculation(shuffled_mat, batch_size)
    res = np.empty((5,2,4), dtype=np.float64)
    for i in range(5):
        for j in range(2):
            res[i,j,0] = np.nanmean(computed[:,i,j])
            res[i,j,1] = np.nanstd(computed[:,i,j])
            res[i,j,2] = np.nanmin(computed[:,i,j])
            res[i,j,3] = np.nanmax(computed[:,i,j])
    return res

def prepare_contingencies(contingency_matrices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                          exclude_labels: Tuple[int, ...] = (-1,),
                          label_of_interest: int = 1, pred_of_interest: Tuple[int, ...] = (1,)) -> np.ndarray:
    """
    Prepare contingency matrices for extraction of the batchwise metrics, by pushing the predictions of interest and label
    of interest to index 0 and removing any excluded predictions from the contingency matrices. The result may be
    directly passed to _batchwise_computation.

    :param contingency_matrices: A List of contingency matrices together with arrays representing a mapping
                                 index->label/prediction for the individual mappings. Note that the contingency matrices
                                 need not have the same shape or even index alignment. Only the label/prediction mappings
                                 must be consistent.
    :param exclude_labels: Any labels that should not appear in the resulting contingency matrices. For example mask labels...
    :param label_of_interest: This is the label for which batchwise metrics should be computed. Will be moved to index
                              0 for the resulting contingency matrix array.
    :param pred_of_interest: The prediction(-s for an overclustering setting) to be matched to the label_of_interest.
                             These will be merged into the single positive element of the resulting contingency matrices.
    :return: An array of contingency matrices that can be passed to numba. The first axis indexes the data point (this
             corresponds to the index within contingency_matrices), the second axis index the contingency matrix label
             and the third axis indexes the contingency matrix prediction.
    """
    pred_exclude = set(pred_of_interest)
    label_exclude = set((label_of_interest,) + exclude_labels)
    pred_offset = len(pred_of_interest) - 1

    def construct_matched_contingency(cm, labels, predictions, all_labels, all_predictions):
        res_cm = np.zeros((len(all_labels), len(all_predictions)), dtype=np.int64)
        pred_masks = [predictions == prediction for prediction in all_predictions]
        pred_masks = [(pred_mask, np.any(pred_mask)) for pred_mask in pred_masks]
        for i, label in enumerate(all_labels):
            label_mask = labels == label
            if not np.any(label_mask):
                continue
            for j, (pred_mask, any) in enumerate(pred_masks):
                if not any:
                    continue
                if j <= pred_offset:
                    j = 0
                else:
                    j -= pred_offset
                res_cm[i, j] += cm[label_mask, pred_mask].sum()
        return res_cm

    all_unique_labels = set([label
                             for _, labels, _ in contingency_matrices
                             for label in labels])
    all_unique_predictions = set([prediction
                                  for _, _, predictions in contingency_matrices
                                  for prediction in predictions])
    all_labels = [label_of_interest] + list(sorted(filter(lambda l: l not in label_exclude, all_unique_labels)))
    all_predictions = list(sorted(pred_of_interest)) + \
                      list(sorted(filter(lambda p: p not in pred_exclude, all_unique_predictions)))
    matched_contingencies = np.array(
        [construct_matched_contingency(cm, labels, predictions, all_labels, all_predictions)
         for cm, labels, predictions in contingency_matrices])
    return matched_contingencies

def batchwise_metrics(contingency_matrices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], n_iter: int,
                      batch_size: int, random_state:int, exclude_predictions: Tuple[int, ...] = (-1,),
                      label_of_interest: int = 1, pred_of_interest: Tuple[int, ...] = (1,)) \
        -> np.ndarray:
    """
    Given a list of contingency_matrix, label, prediction tuples, compute batchwise metrics

    For param and result descriptions see prepare_contingencies and _batchwise_computation.
    """
    matched_contingencies = prepare_contingencies(contingency_matrices, exclude_predictions, label_of_interest,
                                                  pred_of_interest)
    return _batchwise_computation(matched_contingencies, n_iter, batch_size, random_state)

BM_METRIC_LEVEL = ['iou', 'accuracy', 'precision', 'recall', 'f1_score']
BM_CLEVEL_METRIC = 'metric'
BM_CLEVEL_BATCH_CALC = 'batch_calculation'
BM_CLEVEL_ITER_CALC = 'iter_calculation'
BM_CLEVLS = [BM_CLEVEL_METRIC, BM_CLEVEL_BATCH_CALC, BM_CLEVEL_ITER_CALC]
BM_BATCH_LEVEL = ['Mean of Batches', 'Std. of Batches']
BM_ITERATION_LEVEL = ['Iteration Mean', 'Iteration Std.', 'Iteration Min', 'Iteration Max']
def batchwise_metrics_df(contingency_matrices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], n_iter: int,
                         batch_size: int, random_state:int, exclude_labels: Tuple[int, ...] = (-1,),
                         label_of_interest: int = 1, pred_of_interest: Tuple[int, ...] = (1,)) \
        -> pd.DataFrame:
    """
    Return the results of batchwise_metrics in the form of a dataframe to avoid making errors when attempting to access
    a specific statistic of a specific metric...

    :return: A dataframe indexed by batch_size and with Multindex columns corresponding to the metric, the type of
             batchwise statistic and the type of iteration statistic (in that order). See also the BM_* constants for
             indexing these columns.
    """
    res = batchwise_metrics(contingency_matrices, n_iter, batch_size, random_state, exclude_labels,
                            label_of_interest, pred_of_interest)
    midx = pd.MultiIndex.from_product([BM_METRIC_LEVEL,
                                       BM_BATCH_LEVEL,
                                       BM_ITERATION_LEVEL],
                                      names = BM_CLEVLS)
    return pd.DataFrame(data = res.reshape(1, -1), columns = midx)

def direct_batchwise_calculation(labels: Iterable[torch.Tensor], predictions: Iterable[torch.Tensor], n_iter: int,
                                 batch_size: int, random_state:int, exclude_labels: Tuple[int, ...] = (-1,),
                                 label_of_interest: int = 1, pred_of_interest: Tuple[int, ...] = (1,)) -> pd.DataFrame:
    """
    Calculate batchwise statistics and return them as a dataframe. See also batchwise_metrics_df, prepare_contingencies
    and _batchwise_computation

    :param labels: An iterable of the ground truth tensors, where each entry reflects one datapoint. Notice that a tensor
                   is also an iterable...
    :param predictions: An iterable of the predictions made by the model, where each entry reflects one datapoint.
                        Notice that a tensor is also an iterable...
    :param n_iter: Number of iterations to test
    :param batch_size: The batch size to use for metric calculations
    :param random_state: A random seed to produce deterministic results (defines the shuffle order)
    :param exclude_labels: Any labels that should be ignored for batchwise statistics. For example mask labels...
    :param label_of_interest: This is the label for which batchwise metrics should be computed.
    :param pred_of_interest: The prediction(-s for an overclustering setting) to be matched to the label_of_interest.
                             The intersection of this with the label_of_interest corresponds to true positives. FP, FN
                             and TN are calculated correspondingly.
    :return: A dataframe indexed by batch_size and with a hierarchical column
    index corresponding to metric, type of batchwise statistic, type of iteration statistic.
    """
    assert len(labels) == len(predictions) and not any(map(lambda i: labels[i].shape != predictions[i].shape, range(len(labels))))
    def build_contingency(labels: torch.Tensor, predictions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_labels = torch.unique(labels, sorted=False)
        unique_predictions = torch.unique(predictions, sorted=False)
        pred_masks = [predictions == prediction for prediction in unique_predictions]
        res_array = np.zeros((unique_labels.shape[0], unique_predictions.shape[0]), dtype=np.int64)
        for i, label in enumerate(unique_labels):
            label_mask = labels == label
            for j, pred_mask in enumerate(pred_masks):
                res_array[i,j] = torch.count_nonzero(torch.logical_and(label_mask, pred_mask)).sum().cpu().numpy()
        return res_array, unique_labels.cpu().numpy(), unique_predictions.cpu().numpy()

    contingencies = [build_contingency(l, p) for l, p in zip(labels, predictions)]
    return batchwise_metrics_df(contingencies, n_iter, batch_size, random_state, exclude_labels, label_of_interest,
                                pred_of_interest)