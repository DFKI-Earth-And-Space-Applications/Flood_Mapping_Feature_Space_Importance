import argparse
import numbers
import shutil
import traceback
from contextlib import suppress
from math import ceil

import numpy as np
import time

import skorch.dataset
import torch
from skorch.callbacks import BatchScoring, EpochTimer, PrintLog, PassthroughScoring, LRScheduler, Checkpoint, LoadInitState, TrainEndCheckpoint
from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.utils import is_dataset, noop, Ansi
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF

from utils import list_or_none
from multipipeline import *
from metrics import KEY_LABELS, KEY_PREDICTIONS, KEY_INPUT, KEY_CONTINGENCY_MATRIX, KEY_CONTINGENCY_UNIQUE_LABELS, \
    KEY_CONTINGENCY_EXPECTED_LABELS, KEY_CONTINGENCY_EXPECTED_PREDICTIONS, \
    KEY_CONTINGENCY_UNIQUE_PREDICTIONS, MetricComputation, ContingencyMatrixComputation, \
    SequenceMetricComputation, IOUComputation, ToSen1Floods11NormalizedContingency

from skorch import NeuralNetClassifier, NeuralNet
from skorch.helper import predefined_split

class SerializableEpochTimer(EpochTimer, JSONSerializable):
    pass

class SerializableLRScheduler(LRScheduler, JSONSerializable):
    def get_params_as_dict(self) -> Dict[str, Any]:
        res = self.get_params(deep=True)
        res['policy'] = serialisation.get_type_str(self.policy)
        return res

class SerializablePrintLog(PrintLog, JSONSerializable):
    def __init__(self, keys_ignored=None, sink=print, tablefmt='simple', floatfmt='.4f', stralign='right', highlight_best: bool = True):
        super().__init__(keys_ignored, sink, tablefmt, floatfmt, stralign)
        self.highlight_best = highlight_best

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = self.get_params(deep=True)
        utils.del_if_present(res, 'sink')
        return res

    def format_row(self, row, key, color):
        # largely a copy of the code in PrintLog as of skorch version 0.11.0
        # the only change made is to allow a user to specify highlight_best=False in the constructor to prevent color
        # formatting which can mess up for example the integrated editor in WinSCP that I often use for an intermediate
        # inspection of the results on the cluster
        value = row[key]

        if isinstance(value, bool) or value is None:
            return '+' if value else ''

        if not isinstance(value, numbers.Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if self.highlight_best and (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)


class SerializablePassThroughScoring(PassthroughScoring, JSONSerializable):
    def __init__(self, name, lower_is_better=True, on_train=False):
        super().__init__(name, lower_is_better, on_train)
        self.hook: Optional[RunHook] =None

    def _set_hook(self, hook: Optional[RunHook]):
        self.hook = hook

    def on_epoch_end(self, net, **kwargs):
        super().on_epoch_end(net, **kwargs)
        if self.hook is not None:
            self.hook.log_metric(self.name, net.history[-1, self.name], step=net.history[-1, 'epoch'], flush=False)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X, y, **kwargs)
        if self.hook is not None:
            self.hook.flush_metrics(remove_suffix=True)

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.get_params(deep=True)

class SerializableCheckpoint(Checkpoint, JSONSerializable):
    def __init__(self, monitor: Optional[str]='valid_loss_best', f_params='params.pt', f_optimizer='optimizer.pt',
                 f_criterion='criterion.pt', f_history='history.json', f_pickle=None, fn_prefix='', dirname='',
                 event_name: Optional[str] ='event_cp', sink=noop, load_best=False, **kwargs):
        super().__init__(monitor, f_params, f_optimizer, f_criterion, f_history, f_pickle, fn_prefix, dirname,
                         event_name, sink, load_best, **kwargs)
        self.hook: Optional[RunHook] = None

    def get_params_as_dict(self) -> Dict[str, Any]:
        res =  self.get_params(deep=True)
        utils.del_if_present(res, 'sink')
        utils.del_if_present(res, 'hook')
        return res

    def _set_hook(self, hook: Optional[RunHook]):
        self.hook = hook

    def _save_params(self, f: str, net, f_name, log_name):
        super()._save_params(f, net, f_name, log_name)
        if path.exists(f) and path.isfile(f):
            self.hook.add_artifact(f)

    def save_model(self, net):
        super().save_model(net)
        f_pickle = self.f_pickle
        if f_pickle:
            f_pickle = self._format_target(net, f_pickle, -1)
            self.hook.add_artifact(f_pickle)

    def _format_target(self, net, f, idx):
        if f is None:
            return None
        if isinstance(f, str):
            format_args = {
                'net': net
            }
            try:
                format_args['last_epoch'] = net.history[idx],
                format_args['last_batch'] = net.history[idx, 'batches', -1]
            except IndexError:
                print('Cannot add history to format arguments!', file=sys.stderr)
            f = self.fn_prefix + f.format(**format_args)
            f = path.join(self.dirname, f)
            if not path.exists(f):
                print(f'Cannot load non-existing file "{f}"! Skipping!', file=sys.stderr)
                return None
        return f


class SerializableTrainEndCheckpoint(TrainEndCheckpoint, JSONSerializable):
    def __init__(self,
            f_params='params.pt',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            f_pickle=None,
            fn_prefix='train_end_',
            dirname='',
            sink=noop,
            **kwargs):
        super().__init__(f_params, f_optimizer, f_criterion, f_history, f_pickle, fn_prefix, dirname, sink, **kwargs)
        self.hook: Optional[RunHook] = None
        self.checkpoint_: Optional[SerializableCheckpoint] = None

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = self.get_params(deep=True)
        utils.del_if_present(res, 'hook')
        utils.del_if_present(res, 'sink')
        utils.del_if_present(res, 'checkpoint__sink')
        utils.del_if_present(res, 'checkpoint')
        return res

    def _set_hook(self, hook: Optional[RunHook]):
        self.hook = hook
        if self.checkpoint_ is not None:
            self.checkpoint_._set_hook(hook)

    def initialize(self):
        self.checkpoint_ = SerializableCheckpoint(
            monitor=None,
            fn_prefix=self.fn_prefix,
            dirname=self.dirname,
            event_name=None,
            sink=self.sink,
            **self._f_kwargs()
        )
        self.checkpoint_._set_hook(self.hook)
        self.checkpoint_.initialize()
        return self

class SerializableLoadInitState(LoadInitState, JSONSerializable):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.hook: Optional[RunHook] = None

    def _set_hook(self, hook: Optional[RunHook]):
        self.hook = hook

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        if not self.did_load_:
            cp = self.checkpoint.checkpoint_ if isinstance(self.checkpoint, TrainEndCheckpoint) else self.checkpoint
            for k, v in self.get_params(True).items():
                if k.startswith('checkpoint__f_') and isinstance(v, str):
                    sf = path.join(cp.dirname, v)
                    if path.exists(sf) and path.isfile(sf):
                        af = self.hook.get_artifact_file_name(v)
                        shutil.copy(sf, af)
                        self.hook.add_artifact(af)
                    else:
                        print(f'Unable to copy potential checkpoint file {sf} (key={k}) as it does not exist!', file=sys.stderr)
        super().on_train_begin(net, X, y, **kwargs)

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = self.get_params(deep=True)
        utils.del_if_present(res, 'hook')
        utils.del_if_present(res, 'checkpoint__sink')
        utils.del_if_present(res, 'checkpoint')
        return res

class MetricComputationScoring(BatchScoring, JSONSerializable):
    def __init__(self, comp: MetricComputation, on_train=False, name=None, use_caching: bool = True):
        super().__init__(None, False, on_train, name, use_caching)
        self.computation: MetricComputation = comp

    def get_params(self, deep=True):
        return serialisation.recursive_serializable_extract(self)

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = super().get_params_as_dict()
        utils.del_if_present(res, 'use_caching')
        utils.del_if_present(res, 'X_indexing_')
        utils.del_if_present(res, 'y_indexing_')
        utils.del_if_present(res, 'best_scores')
        utils.del_if_present(res, 'best_score_')
        utils.del_if_present(res, 'keys')
        utils.del_if_present(res, 'observed_keys')
        return res

    def initialize(self):
        self.keys = set(self.computation.all_names())
        self.observed_keys: Set[str]  = set()
        return self

    def _set_hook(self, hook: Optional[RunHook]):
        # if hook is None:
        #     print('Resetting hook')
        # else:
        #     print('Setting hook')
        self.computation.set_hook(hook)
    # pylint: disable=attribute-defined-outside-init,arguments-differ
    def on_train_begin(self, net, X, y, **kwargs):
        self.X_indexing_ = skorch.utils.check_indexing(X)
        self.y_indexing_ = skorch.utils.check_indexing(y)
        self.best_scores = {k: -np.inf for k in self.keys}
        self.best_score_ = -np.inf
        # TODO Test!
        with suppress(ValueError, IndexError, KeyError):
            for k in net.history[-1].items():
                if k.endswith('_best') and any(map(lambda comp_key: k.startswith(comp_key), self.keys)):
                    with suppress(ValueError, IndexError, KeyError):
                        best_name_history = net.history[:, '{}_best'.format(self.name_)]
                        idx_best_reverse = best_name_history[::-1].index(True)
                        idx_best = len(best_name_history) - idx_best_reverse - 1
                        self.best_scores[k[:-5]] = net.history[idx_best, self.name_]

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.observed_keys: Set[str] = set()

    def _scoring(self, net: NeuralNet, X_test, y_test):
        """Resolve scoring and apply it to data. Use cached prediction
        instead of running inference again, if available."""
        cur_epoch = net.history[-1, 'epoch']
        X_test = utils.torch_as_np(X_test)
        y_test = None if y_test is None else utils.torch_as_np(y_test)
        # do optionally cached predict... We know it's only one batch anyway
        y_preds = next(iter(net.forward_iter(X_test)))
        all_results = []
        for X, y, pred in zip(X_test, y_test, y_preds):
            pred = utils.torch_as_np(torch.argmax(pred, dim=0))
            computed = self.computation.forward(final=True,
                                                step=cur_epoch,
                                                **{KEY_INPUT:X,
                                                   KEY_LABELS:y,
                                                   KEY_PREDICTIONS:pred})
            computed = {k: v for k, v in computed.items() if k in self.keys and
                        (isinstance(v, numbers.Number) or (isinstance(v, dict) and not
                            any(map(lambda val: not isinstance(val, numbers.Number), v.values()))))}
            all_results.append(computed)
        results_by_metric = {k: [d[k] for d in all_results if k in d] for k in {k for d in all_results for k in d.keys()}}
        for k, values in results_by_metric.items():
            if isinstance(values[0], numbers.Number):
                net.history.record_batch(k, sum(values) / len(values))
                self.observed_keys.add(k)
            elif isinstance(values[0], dict): # per-class-metrics...
                by_sub_key = {sk: [d[sk] for d in values if sk in d] for sk in {sk1 for d in values for sk1 in d.keys()}}
                for sk, sub_values in by_sub_key.items():
                    jk = k+'_'+str(sk)
                    net.history.record_batch(jk, sum(sub_values) / len(sub_values))
                    self.observed_keys.add(jk)
            else:
                raise AssertionError('According to the previous check it should not be possible that there is any '
                                     'type that doesn\'t match this if-else!')


    def on_batch_end(self, net, batch, training, **kwargs):
        if training != self.on_train:
            return

        X, y = batch
        y_preds = [kwargs['y_pred']]
        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            self._scoring(cached_net, X, y)

    def on_epoch_end(self, net, **kwargs):
        for key in self.observed_keys:
            self.name_ = key
            self.best_score_ = self.best_scores.get(key, -np.inf)
            super(MetricComputationScoring, self).on_epoch_end(net, **kwargs)
            self.best_scores[key] = self.best_score_
        self.computation.hook.flush_metrics(remove_suffix=True)


class AccumulatingMetricScoring(skorch.callbacks.Callback, JSONSerializable):

    def __init__(self, contingency_name: str, scoring: Optional[MetricComputation], repeat_factor: int, on_train: bool,
                 expected_labels: Optional[torch.Tensor] = None, expected_predictions: Optional[torch.Tensor] = None,
                 restart_best: bool = True) \
            -> None:
        super().__init__()
        self.contingency_comp = ContingencyMatrixComputation(contingency_name)
        self.scoring: Optional[MetricComputation] = scoring
        self._repeats = 1
        self._contingencies: List[Dict[str, Any]] = []
        self._results: List[Dict[str, numbers.Number]] = []
        self._keys: Set[str] = set(self.scoring.all_names()) if self.scoring is not None else set()
        self._observed_keys: Set[str] = set()
        self._hook: Optional[RunHook] = None
        self._expected_labels = expected_labels
        self._expected_predictions = expected_predictions
        self.repeat_factor = repeat_factor
        self.on_train = on_train
        self.cached_selectors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.restart_best = restart_best
        self.first_index = None

    def _set_hook(self, hook: Optional[RunHook]):
        self._hook = hook

    def get_params(self, deep=True):
        res = super().get_params(deep)
        res = {k: v for k, v in res.items() if not k.startswith('_')}
        return res

    def on_train_begin(self, net: NeuralNetClassifier, X=None, y=None, **kwargs):
        self.nonlin = net._get_predict_nonlinearity()
        self.first_index = None

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self._repeats = 1
        self._results = []
        self._contingencies = []
        self._observed_keys = set()
        self.first_index = len(net.history)-1 if self.restart_best and self.first_index is None else self.first_index

    def add_result(self, computed: Dict[str, Any], epoch: int):
        if self.scoring is None:
            return

        def insert_result(new_res: Dict[str, numbers.Number], k:str, value):
            new_res[k] = value
            self._observed_keys.add(k)

        scores = self.scoring.forward(final=True, step=epoch, **computed)
        scores = {k: v for k, v in scores.items() if k in self._keys and
                    (isinstance(v, numbers.Number) or (isinstance(v, dict) and not
                    any(map(lambda val: not isinstance(val, numbers.Number), v.values()))))}
        new_res: Dict[str, numbers.Number] = {}
        for k, score in scores.items():
            if isinstance(score, numbers.Number):
                insert_result(new_res, k, score)
            elif isinstance(score, dict):  # per-class-metrics...
                for sk, sub_score in score.items():
                    jk = k + '_' + str(sk)
                    insert_result(new_res, jk, sub_score)
            else:
                raise AssertionError('According to the previous check it should not be possible that there is any '
                                     'type that doesn\'t match this if-else!')
        if new_res: # if it is empty anyway...
            self._results.append(new_res)

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        if self.contingency_comp.contingency_matrix is not None:
            # There is still something to be processed and we need to remove that before the next epoch starts...
            self._contingencies.append(self.contingency_comp._contingency_to_dict())
            self.contingency_comp.contingency_matrix = None
        epoch = net.history[-1, 'epoch']
        for i, d in enumerate(self._contingencies):
            self.add_result(d, epoch)
            if self._hook is not None:
                self._hook.log_metric(self.contingency_comp.name+f'.{i}', d, step=epoch, flush=False)
        self._hook.set_auto_flush(False, False)
        self._contingencies.clear()
        if self._results:
            results_by_metric = {k: np.mean([d[k] for d in self._results if k in d]) for k in self._observed_keys}
            self._results.clear()
            for k, v in results_by_metric.items():
                try:
                    if self.first_index is not None:
                        cur_best = np.max(net.history[self.first_index:, k])
                    else:
                        cur_best = np.max(net.history[:, k])
                # sometimes I hate python... why can't they design an API with a nice check?!?
                except KeyError:
                    cur_best = -np.inf
                net.history.record(k, float(v))
                is_best = bool(v >= cur_best)
                net.history.record(k+'_best', is_best)

    def on_batch_end(self, net: NeuralNetClassifier, batch=None, training=None, **kwargs):
        if self.on_train != training:
            return

        with torch.no_grad():
            X_test, y_test = batch
            epoch = net.history[-1, 'epoch']
            X_test = utils.np_as_torch(X_test)
            y_test = utils.np_as_torch(y_test)
            # do optionally cached predict... We know it's only one batch anyway
            y_pred = kwargs['y_pred']
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            y_pred = utils.np_as_torch(y_pred)
            y_pred = self.nonlin(y_pred)
            if y_pred.shape[-3] > 1:
                y_pred = torch.argmax(y_pred, dim=1)
            else:
                # if possible, avoid re-creating the same tensor over and over again
                # (this might happen on the gpu and thus fragment gpu memory!)
                if self.cached_selectors is None or self.cached_selectors[0].shape != y_pred.shape:
                    self.cached_selectors = torch.zeros_like(y_pred), torch.ones_like(y_pred)
                zeros, ones = self.cached_selectors
                y_pred = torch.where(y_pred < 0.5, zeros, ones)
            if torch.is_floating_point(y_test):
                y_test_new = y_test.long()
                y_test_new[y_test > 0.5] = 1
                y_test = y_test_new
                del y_test_new
            if len(y_test.shape) == 4 and y_test.shape[1] > 1:
                y_test = y_test[:, -1].reshape((y_test.shape[0], 1)+y_test.shape[2:])
            for X, y, pred in zip(X_test.detach().cpu(), y_test.detach().cpu(), y_pred.detach().cpu()):
                is_final = self._repeats >= self.repeat_factor
                res = self.contingency_comp.forward(final=is_final,
                                                         step=epoch,
                                                         **{KEY_INPUT: X,
                                                            KEY_LABELS: y.reshape(pred.shape),
                                                            KEY_PREDICTIONS: pred,
                                                            KEY_CONTINGENCY_EXPECTED_LABELS: self._expected_labels,
                                                            KEY_CONTINGENCY_EXPECTED_PREDICTIONS: self._expected_predictions})
                self._repeats = self._repeats + 1 if self._repeats < self.repeat_factor else 1
                if is_final:
                    self._contingencies.append(res[self.contingency_comp.name])

    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X, y, **kwargs)
        if self._hook is not None:
            self._hook.flush_metrics(remove_suffix=True)
        self.cached_selectors = None


class NonDeterministicCrossEntropy(nn.Module):
    """
    A version of the cross entropy loss that can be used even with torch.use_deterministic_algorithms(True).
    This is done by resetting the use_deterministic_algorithms...
    """
    def __init__(self, weight=None, ignore_index=-1,  reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()
        self.cross_entropy = CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                              reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cur_det = torch.are_deterministic_algorithms_enabled()
        if cur_det:
            torch.use_deterministic_algorithms(False)
        res = self.cross_entropy(input, target)
        if cur_det:
            torch.use_deterministic_algorithms(True)
        return res

class SegmentationClassifier(NeuralNetClassifier):
    def __init__(self, module, *args, criterion=CrossEntropyLoss, train_split=skorch.dataset.ValidSplit(5),
                 classes=None, has_prob_output: bool = False, deferred_load: Optional[Checkpoint] = None, **kwargs):
        super().__init__(module, *args, criterion=criterion, train_split=train_split, classes=classes, **kwargs)
        self._hook = None
        self.has_prob_output = has_prob_output
        self._deferred_load = deferred_load

    def get_default_callbacks(self):
        res = [
            ('epoch_timer', SerializableEpochTimer()),
            ('train_loss', SerializablePassThroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', SerializablePassThroughScoring(
                name='valid_loss',
            )),
            ('print_log', SerializablePrintLog()),
        ]
        return res

    def _set_hook(self, hook: Optional[RunHook]):
        self._hook = hook

    def _apply_hook(self):
        if self._hook is not None:
            for name, callback in self.callbacks_:
                if hasattr(callback, '_set_hook') and callable(getattr(callback, '_set_hook')):
                    callback._set_hook(self._hook)

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        super().on_train_begin(net, X, y, **kwargs)
        self._apply_hook()

    def initialize(self):
        res = super().initialize()
        self._apply_hook()
        if self._deferred_load is not None:
            self.load_params(checkpoint=self._deferred_load)
        return res

    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)
        if y is not None:
            # pylint: disable=attribute-defined-outside-init
            assert self.classes is not None, 'In order to avoid loading labels into memory, the classes parameter has to be set!'
            self.classes_inferred_ = self.classes

    def train_step_single(self, batch, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        step : dict
          A dictionary ``{'loss': loss, 'y_pred': y_pred}``, where the
          float ``loss`` is the result of the loss function and
          ``y_pred`` the prediction generated by the PyTorch module.

        """
        self._set_training(True)
        Xi, yi = batch
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        if loss is not None:
            loss.backward()
        else:
            loss = torch.tensor([0], dtype=torch.float32)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def validation_step(self, batch, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self._set_training(False)
        Xi, yi = batch
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
            if loss is None:
                loss = torch.tensor([0], dtype=torch.float32)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def predict(self, X):
        if self.has_prob_output:
            pred = super(SegmentationClassifier, self).predict_proba(X)
            return np.where(pred < 0.5, np.zeros(pred.shape), np.ones(pred.shape))
        return super().predict(X)

    def __str__(self) -> str:
        return f'{self.__class__.__qualname__}({self.module.__class__.__qualname__}({self.module.get_params_as_dict()}))'


class PrepareTargetDataset(TransformerDataset):


    def __init__(self, wrapped: data.Dataset, force_int: bool = True):
        super().__init__(wrapped)
        self.force_int = force_int

    def _transform(self, data):
        res = data.reshape(data.shape[1:]) if data.shape[0] == 1 else data
        if self.force_int:
            if isinstance(res, torch.Tensor):
                res = res.long()
            else:
                res = res.astype(np.int64)
        return res

class AugmentationDataset(TransformerDataset):
    def __init__(self, wrapped: data.Dataset, transform_fun: Callable):
        super().__init__(wrapped)
        self.transform_fun = transform_fun

    def _transform(self, data):
        return self.transform_fun(data)


class Augmentation(JSONSerializable):
    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return AugmentationDataset(dataset, self._augment)

    def _augment(self, data):
        if isinstance(data, tuple):
            X, y = data
            X, y = self.augment(utils.np_as_torch(X), utils.np_as_torch(y))
            return X, y
        else:
            X, y = self.augment(utils.np_as_torch(data), None)
            return X

    def augment(self, X: Union[torch.Tensor, Dict[str, torch.Tensor]],
                y: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> \
        Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        raise NotImplementedError

class ComposedAugmentation(Augmentation):

    def __init__(self, sub_augments: List[Augmentation]) -> None:
        super().__init__()
        self.sub_augments = sub_augments

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        for augment in self.sub_augments:
            dataset = augment(dataset)
        return dataset

class RandomCrop(Augmentation):
    def __init__(self, crop_dims: Tuple[int, int]):
        super().__init__()
        self.crop_dims = crop_dims

    def augment(self, X: Union[torch.Tensor, Dict[str, torch.Tensor]],
                y: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> \
            Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        p = TT.RandomCrop.get_params(X, self.crop_dims)
        X = TTF.crop(X, *p)
        if y is not None:
            y = TTF.crop(y, *p)
        return X, y

class CropExtractDataset(data.Dataset):

    def __init__(self, wrapped: data.Dataset, crop_size: Tuple[int, int], hfactor: int, vfactor: int):
        self.wrapped = wrapped
        self.crop_size = crop_size
        self.vfactor = vfactor
        self.ncrops = hfactor * vfactor
        self._data_cache: Optional[int, Any] = None

    def do_crop(self, data, h_pos: int, v_pos: int):
        if data is None:
            return None
        data = data[...,
               h_pos * self.crop_size[0]: min((h_pos + 1) * self.crop_size[0], data.shape[-2]),
               v_pos * self.crop_size[1]: min((v_pos + 1) * self.crop_size[1], data.shape[-1])
               ]
        if tuple(data.shape[-2:]) != self.crop_size:
            padding = (0, 0, self.crop_size[1] - data.shape[-1], self.crop_size[0] - data.shape[-2])
            data = TTF.pad(utils.np_as_torch(data), list(padding))
        return data


    def __getitem__(self, item):
        sub_item = item // self.ncrops
        h_pos = (item % self.ncrops) // self.vfactor
        v_pos = (item % self.ncrops) % self.vfactor
        if self._data_cache is None or self._data_cache[0] != sub_item:
            data = self.wrapped[sub_item]
            self._data_cache = sub_item, data
        else:
            data = self._data_cache[1]
        if isinstance(data, tuple):
            return tuple((self.do_crop(d, h_pos, v_pos) if len(d.shape) >= 2 else d) for d in data)
        else:
            return self.do_crop(data, h_pos, v_pos)

    def __len__(self):
        return len(self.wrapped) * self.ncrops


class CropExtract(Augmentation):
    def __init__(self,  crop_size: Tuple[int, int], hfactor: int, vfactor: int):
        self.crop_size = crop_size
        self.hfactor = hfactor
        self.vfactor = vfactor

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return CropExtractDataset(dataset, self.crop_size, self.hfactor, self.vfactor)

class CropCatDataset(data.Dataset):

    def __init__(self, wrapped: data.Dataset, hfactor: int, vfactor: int):
        self.wrapped = wrapped
        self.vfactor = vfactor
        self.hfactor = hfactor
        self.ncrops = hfactor * vfactor

    def __getitem__(self, item):
        sub_item = item * self.ncrops
        data = [self.wrapped[sub_item + i] for i in range(self.ncrops)]
        if isinstance(data[0], torch.Tensor):
            data = torch.cat(tuple(torch.cat(tuple(data[i*self.vfactor + j]for j in range(self.vfactor)), dim=-1)
                                   for i in range(self.hfactor)), dim=-2)
        else:
            data = np.concatenate(tuple(np.concatenate(tuple(data[i*self.vfactor + j] for j in range(self.vfactor)), axis=-1)
                                  for i in range(self.hfactor)), axis=-2)
        return data

    def __len__(self):
        return len(self.wrapped) // self.ncrops

class RevertCropExtract(Augmentation):
    def __init__(self,  hfactor: int, vfactor: int):
        self.hfactor = hfactor
        self.vfactor = vfactor

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        return CropCatDataset(dataset, self.hfactor, self.vfactor)

class RandomHFlip(Augmentation):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def augment(self, X: Union[torch.Tensor, Dict[str, torch.Tensor]],
                y: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> \
            Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        if torch.rand(1, device='cpu')[0] >= self.p:
            X = TTF.hflip(X)
            if y is not None:
                y = TTF.hflip(y)
        return X, y

class RandomVFlip(Augmentation):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def augment(self, X: Union[torch.Tensor, Dict[str, torch.Tensor]],
                y: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> \
            Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        if torch.rand(1, device='cpu')[0] >= self.p:
            X = TTF.vflip(X)
            if y is not None:
                y = TTF.vflip(y)
        return X, y



class SkorchAdaptor(JSONSerializable):
    def __init__(self, skorch_instance: NeuralNet,
                 train_augment: Optional[Augmentation] = None,
                 predict_augment: Optional[Augmentation] = None,
                 post_predict_augment: Optional[Augmentation] = None,
                 force_train_target_int: bool = True,
                 force_valid_target_int: bool = True):
        self.skorch_instance = skorch_instance
        self.train_augment = train_augment
        self.predict_augment = predict_augment
        self.post_predict_augment = post_predict_augment
        self.skorch_instance.initialize()
        self.force_train_target_int = force_train_target_int
        self.force_valid_target_int = force_valid_target_int

    def get_params_as_dict(self) -> Dict[str, Any]:
        def filter_type(val: Union[List[Any], Dict[str, Any], JSONSerializable, int, float, str]) -> Any:
            if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)):
                if isinstance(val, torch.Tensor) and len(val.shape) == 0:
                    return val.item()
                return [filter_type(v) for v in val]
            elif isinstance(val, dict):
                return {k: filter_type(v) for k, v in val.items()}
            elif isinstance(val, (int, float, str, JSONSerializable)):
                return val
            elif isinstance(val, type):
                return val.__qualname__
            else:
                return val.__class__.__qualname__

        # TODO check what is returned here
        res = super(SkorchAdaptor, self).get_params_as_dict()
        res.update(self.skorch_instance.get_params(deep=True))
        del res['skorch_instance']
        assert not any(map(lambda c: not isinstance(c[1], JSONSerializable), self.skorch_instance.callbacks_)), \
            f'All callbacks must be serializable! Not serializable are ' \
            f'{[c for c in self.skorch_instance.callbacks_ if not isinstance(c[1], JSONSerializable)]}'
        res['callbacks'] = dict(self.skorch_instance.callbacks_)
        del res['_kwargs']
        res['lr'] = res.get('optimizer__lr', res['lr'])
        res['optimizer__lr'] = res['lr']
        del res['lr']
        opt_params = serialisation.get_function_arguments(self.skorch_instance.optimizer, {'params'})
        opt_params.update({k[11:]: v for k, v in res.items() if k.startswith('optimizer__') and k[11:] in opt_params})
        res['optimizer'] = serialisation.FixedJsonSerializable(
            serialisation.get_type_str(self.skorch_instance.optimizer),
            opt_params
        )
        crit_params = serialisation.get_function_arguments(self.skorch_instance.criterion, raise_missing=False)
        crit_params.update({k[11:]: v for k, v in res.items() if k.startswith('criterion__') and k[11:] in crit_params})
        res['criterion'] = serialisation.FixedJsonSerializable(
            serialisation.get_type_str(self.skorch_instance.criterion),
            crit_params
        )
        res = {k: v for k, v in res.items()
               if not k.startswith('callbacks__') and not k.startswith('optimizer__')
               and not k.startswith('criterion__')}
        res = filter_type(res)
        return res

    def reset_params(self, params: Dict[str, Any]):
        if params:
            self.skorch_instance.set_params(**params)

    def extract_dataset_only(self, data: Optional[Dict[str, Tuple[data.Dataset, Meta]]]) \
        -> Optional[Union[data.Dataset, Dict[str, data.Dataset]]]:
        if data is not None:
            if len(data) == 1:
                return next(iter(data.values()))[0]
            return {k: d for k, (d, _) in data.items()}
        return None

    def _set_hook(self, hook: Optional[RunHook]) -> 'SkorchAdaptor':
        if hasattr(self.skorch_instance, '_set_hook') and callable(getattr(self.skorch_instance, '_set_hook')):
            self.skorch_instance._set_hook(hook)
        return self

    def fit(self, indent: str, data: Dict[str, Tuple[data.Dataset, Meta]],
            target: Optional[Dict[str, Tuple[data.Dataset, Meta]]] = None,
            validation_data: Optional[Dict[str, Tuple[data.Dataset, Meta]]] = None,
            validation_target: Optional[Dict[str, Tuple[data.Dataset, Meta]]] = None):
        data = self.extract_dataset_only(data)
        target = PrepareTargetDataset(self.extract_dataset_only(target), self.force_train_target_int)
        validation_data = self.extract_dataset_only(validation_data)
        validation_target = self.extract_dataset_only(validation_target)
        params_to_set = {}
        if validation_data is not None:
            if validation_target is not None:
                validation_target = PrepareTargetDataset(validation_target, self.force_valid_target_int)
            sd = skorch.dataset.Dataset(validation_data, validation_target)
            if self.predict_augment is not None:
                sd = self.predict_augment(sd)
            params_to_set['train_split'] = predefined_split(sd)
        if params_to_set:
            self.skorch_instance.set_params(**params_to_set)
        print(indent + f'Training network {self.skorch_instance}.')
        sd = skorch.dataset.Dataset(data, target)
        if self.train_augment is not None:
            sd = self.train_augment(sd)
        t = time.time()
        self.skorch_instance.fit(sd, None)
        t = time.time() - t
        print(indent + f'Training completed successfully for network {self.skorch_instance.module.__class__.__name__} '
                       f'and took {t:.3f}s.')

    def predict(self, indent: str, data: Dict[str, Tuple[data.Dataset, Meta]],
                predict_proba: bool = False) -> data.Dataset:
        data = self.extract_dataset_only(data)
        print(indent + f'Predicting on network {self.skorch_instance}.')
        # This causes problems with predict!!! and is a documented error in skorch
        batch_size_2 = self.skorch_instance.batch_size == 2
        if batch_size_2:
            self.skorch_instance.set_params(batch_size=1)
        sd = skorch.dataset.Dataset(data)
        if self.predict_augment is not None:
            sd = self.predict_augment(sd)
        t = time.time()
        if predict_proba:
            res: np.ndarray = self.skorch_instance.predict_proba(sd)
        else:
            res: np.ndarray = self.skorch_instance.predict(sd)
        t = time.time() - t
        if batch_size_2:
            self.skorch_instance.set_params(batch_size=2)
        print(indent + f'Prediction completed successfully for network {self.skorch_instance.module.__class__.__name__}'
                       f' and took {t:.3f}s.')
        res_data: data.Dataset = ArrayInMemoryDataset(res.reshape((res.shape[0], 1) + res.shape[1:]))
        #np.all(np.concatenate((np.stack((res_data[0], res_data[1])), np.stack((res_data[2], res_data[3]))), axis=0) == result[0])
        if self.post_predict_augment is not None:
            result = self.post_predict_augment(res_data)
        else:
            result = res_data
        return result

    def __str__(self):
        return f'SkorchAdaptor({self.skorch_instance})'

# Generalises sklearn adaptor to accept more than one input (for example dbscan sample_weight)
# however dataset should already have the shape [N, C] and masking is not performed out of the box
# Furthermore channelwise predictions are not supported
# Also this should really be extended to support multiple outputs
class SkorchAdaptorModule(MultiTransformerModule):
    def __init__(self, transformer: SkorchAdaptor,
                 dataset_criteria_map: Iterable[DataInfo],
                 target_criteria_map: Optional[Iterable[DataInfo]] = None,
                 valid_dataset_criteria_map: Optional[Iterable[DataInfo]] = None,
                 valid_target_criteria_map: Optional[Iterable[DataInfo]] = None,
                 prediction_dataset_name: Optional[str] = None,
                 prediction_channel_name: Optional[str] = 'prediction',
                 do_fit: bool = True,
                 do_predict: bool = True,
                 predict_proba: bool = False,
                 reset_params: Optional[Dict[str, Any]] = None
                 ):
        super().__init__()
        self.transformer: SkorchAdaptor = transformer
        self.dataset_criteria_map: List[DataInfo] = list_or_none(dataset_criteria_map)
        self.target_criteria_map: Optional[List[DataInfo]] = list_or_none(target_criteria_map)
        self.valid_dataset_criteria_map: Optional[List[DataInfo]] = list_or_none(valid_dataset_criteria_map)
        self.valid_target_criteria_map: Optional[List[DataInfo]] = list_or_none(valid_target_criteria_map)
        self.prediction_dataset_name = prediction_dataset_name
        self.prediction_channel_name = prediction_channel_name
        self.do_fit = do_fit
        self.do_predict = do_predict
        self.predict_proba = predict_proba
        self.reset_params = reset_params

    def extract_from_summary(self, summary: Summary, criteria_map: Optional[Iterable[DataInfo]]) \
            -> Optional[Dict[str, Tuple[data.Dataset, Meta]]]:
        if criteria_map is not None:
            return {d_info.param_name: summary.by_criterion(d_info.criterion, delete=d_info.delete)
                    for d_info in criteria_map}
        else:
            return None

    def __call__(self, summary: Summary) -> Summary:
        data = self.extract_from_summary(summary, self.dataset_criteria_map)
        assert data is not None, 'As the dataset criteria map should never be None, data should also never have been None!'
        assert len(data) >= 1, 'Some data is required as input to the Neural Network!'
        if self.do_fit:
            target = self.extract_from_summary(summary, self.target_criteria_map)
            valid_data = self.extract_from_summary(summary, self.valid_dataset_criteria_map)
            valid_target = self.extract_from_summary(summary, self.valid_target_criteria_map)
            af = summary.run_hook.auto_flush
            self.print(summary, f'Performing fit on neural network with params {data.keys()} and '
                                f'target={None if target is None else target.keys()} and '
                                f'valid_data={None if valid_data is None else valid_data.keys()} and '
                                f'valid_target={None if valid_target is None else valid_target.keys()}.')
            if self.reset_params is not None:
                self.transformer.reset_params(self.reset_params)
            t = time.time()
            self.transformer._set_hook(summary.run_hook.set_auto_flush(False))\
                .fit(summary.indent + '\t', data,
                     target=target,
                     validation_data=valid_data,
                     validation_target=valid_target)
            t = time.time() - t
            self.print(summary, f'Fit completed in {t:.3f}s, saving computed metrics.')
            summary.run_hook.set_auto_flush(af)
            self.transformer._set_hook(None)
            t = time.time()
            summary.run_hook.flush_metrics(remove_suffix=True)
            self.print(summary, f'Save completed - took {time.time() - t:.3f}s.')
        if self.do_predict:
            self.print(summary, f'Performing predict on Neural Network with params {data.keys()}.')
            t = time.time()
            dataset = self.transformer.predict(summary.indent + '\t', data, self.predict_proba)
            t = time.time() - t
            self.print(summary, f'Predict completed in {t:.3f}s.')
            channel_names = [self.prediction_channel_name]
            meta = next(iter(data.values()))[1]._replace(channel_names=channel_names)
            summary.add(dataset, meta, self.prediction_dataset_name)
        return summary
