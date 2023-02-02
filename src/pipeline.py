import functools as func
import itertools as itert
import os.path as path
import pickle
import sys
import tempfile as tmp
import time
import traceback
from collections import namedtuple, OrderedDict
from typing import List, Tuple, Optional, Iterable, Union, Callable, Any, Dict, ContextManager, Set, Sized, Literal

import numpy as np
import optuna.trial as trial
import sacred.experiment as exp
import torch
import torch.utils.data as data

import utils
from sen1floods11.dataset import MergingDatasetWrapper
from serialisation import JSONSerializable


class ArtifactContextManager(ContextManager):
    def __init__(self, run_obj: exp.Run, f_name: str, method: str):
        self.run_obj = run_obj
        self.f_name = f_name
        self.method = method
        self.f_obj = None

    def __enter__(self):
        assert self.f_obj is None
        self.f_obj = open(self.f_name, self.method)
        return self.f_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.f_obj is not None
        self.f_obj.flush()
        self.f_obj.close()
        self.run_obj.add_artifact(self.f_name)
        self.f_obj = None

class RunFileContext:
    def add_artifact(self, artifact_file, name: str):
        raise NotImplementedError

    def get_execution_dir(self) -> str:
        raise NotImplementedError

    def get_active_dir(self) -> Optional[str]:
        raise NotImplementedError

class RunHook(ContextManager):
    TORCH_GPU_RNG = 'torch_gpu_rng'

    def __init__(self, run_obj: RunFileContext, trial_obj: Optional[trial.BaseTrial], seed: Optional[int] = None,
                           n_jobs: int = -1):
        self.run_obj = run_obj
        # print('WARNING: Replacing sacred Metrics logger with a flushing variant. '
        #       'Metrics will only be sent to the heartbeat-event if flush is called!', file=sys.stderr)
        self.trial_obj = trial_obj
        self.temp_dir: Optional[tmp.TemporaryDirectory] = None
        self.temp_dir_name: Optional[str] = None
        self.numpy_rng = np.random.default_rng(seed=seed)
        self.torch_cpu_rng = torch.Generator(device='cpu')
        if seed is not None:
            self.torch_cpu_rng.manual_seed(seed)
        self._torch_gpu_rng = None
        # Basically this is the result and meant to be queried after the pipeline is finished
        self.recorded_metrics = {}
        self.recorded_metric_prefixes = []
        # An accumulator for the current in-progress-metrics
        self.in_progress_metrics: Dict[str, Dict[int, Any]] = {}
        self.metric_counters: Dict[str, int] = {}
        self.auto_flush = True
        self.remove_suffix = False
        self.n_jobs = n_jobs
        self.parallel_instance = None

    def add_result_metric_prefix(self, name_prefix: str) -> 'RunHook':
        self.recorded_metric_prefixes.append(name_prefix)
        return self

    def report(self, value, step):
        assert self.trial_obj is not None
        self.trial_obj.report(value, step)

    def should_prune(self):
        assert self.trial_obj is not None
        return self.trial_obj.should_prune()

    def get_artifact_file_name(self, name: str) -> str:
        """
        Returns the name for an artifact file. NOTICE THAT THE FILE IS NOT ADDED TO SACRED
        (as it would not be written at the time of adding).
        :param name:
        :return:
        """
        assert self.temp_dir is not None and self.temp_dir_name is not None
        res_path = path.join(self.temp_dir_name, name)
        return res_path

    def open_artifact_file(self, name: str, method: str) -> ContextManager:
        """
        Creates a context manager that will open a new file with the specified name in this RunHook's temporary
        directory. Will be automatically closed together with the RunHook -
        in contrast to get_artifact_file_name this file is added to sacred upon close
        :param name: The name of the file to open
        :param method: The method used for opening. Directly passed to the standard open method.
        :return: A context manager that can be used just as the one returned by open
        """
        return ArtifactContextManager(self.run_obj, self.get_artifact_file_name(name), method)

    def log_metric(self, metric: str, value: Any, step=None, flush: Optional[bool] = None) -> 'RunHook':
        simplified = utils.recursive_simplify(value)
        step = (self.metric_counters.get(metric, -1) + 1 if step is None else step)
        self.metric_counters[metric] = step
        if metric not in self.in_progress_metrics:
            self.in_progress_metrics[metric] = {}
        self.in_progress_metrics[metric][step] = simplified
        if any(map(lambda p: metric.startswith(p), self.recorded_metric_prefixes)):
            if metric not in self.recorded_metrics:
                self.recorded_metrics[metric] = {}
            self.recorded_metrics[metric][step] = value
        flush = self.auto_flush if flush is None else flush
        if flush:
            self.flush_metrics()
        return self

    def set_auto_flush(self, auto_flush: bool, remove_suffix: Optional[bool] = None) -> 'RunHook':
        self.auto_flush = auto_flush
        self.remove_suffix = self.remove_suffix if remove_suffix is None else remove_suffix
        return self

    def _clear_suffixes(self, to_simplify: Iterable[str], remove_suffix: Optional[bool] = None) -> Dict[str, List[Tuple[str, str]]]:
        remove_suffix = self.remove_suffix if remove_suffix is None else remove_suffix
        if not remove_suffix:
            return {to_modify: [(to_modify, to_modify)] for to_modify in to_simplify}
        res = {}
        for to_modify in to_simplify:
            split = to_modify.rsplit('.', maxsplit=1)
            if len(split) > 1:
                res[split[0]] = res.get(split[0], []) + [(to_modify, split[1])]
            else:
                res[to_modify] = [(to_modify, to_modify)]
        return res

    def _write_metric(self, key: str, metric_names: List[Tuple[str, str]]):
        f_name = key + '.pkl'
        current_metric_values: Dict[str, Dict[int, Any]] = {}
        try:
            artifact_file = self.get_artifact_file_name(f_name)
            if path.exists(artifact_file):
                with self.open_artifact_file(f_name, 'rb') as fd:
                    current_metric_values = pickle.load(fd)
        except BaseException:
            print(f'Caught exception whilst attempting to read metric {key} from file {f_name}.')
            print(traceback.format_exc())
        for metric_name, metric_key in metric_names:
            current_val = current_metric_values.get(metric_key, {})
            current_val.update(self.in_progress_metrics[metric_name])
            del self.in_progress_metrics[metric_name]
            current_metric_values[metric_key] = current_val
        with self.open_artifact_file(f_name, 'wb') as fd:
            pickle.dump(current_metric_values, fd, protocol=5)

    def flush_metrics(self, metrics_to_flush: Optional[Union[str, Iterable[str]]] = None,
                      remove_suffix: Optional[bool] = None) -> 'RunHook':
        # print('Flush called')
        if metrics_to_flush is None:
            metrics_to_flush: Iterable[str] = self.in_progress_metrics.keys()
        elif isinstance(metrics_to_flush, str):
            metrics_to_flush: Iterable[str] = [metrics_to_flush]
        metrics_to_flush: Dict[str, List[Tuple[str, str]]] = self._clear_suffixes(metrics_to_flush, remove_suffix)
        for key, metric_names in metrics_to_flush.items():
            self._write_metric(key, metric_names)
        return self

    def add_artifact(self, artifact_file: str, name: Optional[str] = None) -> 'RunHook':
        if self.run_obj is not None:
            self.run_obj.add_artifact(artifact_file, name=name)
        return self

    def __enter__(self) -> 'RunHook':
        assert self.temp_dir is None and self.temp_dir_name is None
        self.temp_dir = tmp.TemporaryDirectory(suffix='_run_outputs')
        self.temp_dir_name = self.temp_dir.__enter__()
        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        res = self.temp_dir.__exit__(__exc_type, __exc_value, __traceback)
        self.temp_dir_name = None
        self.temp_dir = None
        return res


META_KEY_PER_ITEM = 'per_item_meta'
META_KEY_PER_ITEM_ID = 'id'
META_KEY_PER_ITEM_REGION = 'region'
META_TYPE_UNKNOWN = 1
META_TYPE_LABEL = 2
META_TYPE_S1 = 4
META_TYPE_S2 = 8
ALL_META_TYPES: Set[int] = {META_TYPE_UNKNOWN, META_TYPE_LABEL, META_TYPE_S1, META_TYPE_S2}
AMT_TYPE = Literal[1, 2, 4, 8]
PerItemMeta = namedtuple('PerItemMeta', 'region id')
Meta = namedtuple('Meta', 'channel_names per_item_info split type indent exe_start run_hook')


def types_as_name(meta: Meta):
    res = []
    type = meta.type
    if META_TYPE_UNKNOWN & type != 0:
        res.append('UNKNOWN')
    if META_TYPE_LABEL & type != 0:
        res.append('label')
    if META_TYPE_S1 & type != 0:
        res.append('s1')
    if META_TYPE_S2 & type != 0:
        res.append('s2')
    return '_'.join(res)


def check_meta_and_data(dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
    assert dataset is not None, 'Cannot execute without dataset'
    assert meta is not None, 'Cannot execute without meta'
    return dataset, meta


def empty_meta():
    return Meta([], [], '', META_TYPE_UNKNOWN, '', time.time(), None)


def merge_meta(metas: List[Meta], checked: bool = True) -> Optional[Meta]:
    if not metas:
        return None
    elif len(metas) == 1:
        return metas[0]

    res = metas[0]._replace(channel_names=list(itert.chain.from_iterable(map(lambda m: m.channel_names, metas))))
    iterator = itert.zip_longest(*map(lambda m: m.per_item_info, metas))
    if checked:
        for i, t in enumerate(iterator):
            assert not any(map(lambda p: True, filter(lambda p: p != t[-1], t))), \
                f'All per-item-metas should be identical for merge however index {i} consists of {str(t)}.'
    # Because python doesn't have ordered sets I have to do this nonsense with OrderedDicts
    res = res._replace(split='_'.join(OrderedDict(map(lambda m: (m.split, None), metas)).keys()),
                       type=func.reduce(lambda x, y: x | y, map(lambda m: m.type, metas)))
    return res


def short_repr(meta: Optional[Meta]) -> str:
    if meta is None:
        return 'None'
    return f'Meta(channel_names={str(meta.channel_names)}, per_item_info=[' \
           f'{str(meta.per_item_info[0]) + ", ..." if meta.per_item_info else ""}], ' \
           f'split={meta.split}, type={types_as_name(meta)}, indent={meta.indent}, exe_start={meta.exe_start})'


def channel_index(meta: Meta) -> Dict[str, int]:
    return {name: i for i, name in enumerate(meta.channel_names)}


def value_label_index(meta: Meta, label_index: str, value_index: Optional[Tuple[str, ...]]):
    index = channel_index(meta)
    label_index = index[label_index]
    if value_index is not None:
        intensity_index = [index[cn] for cn in value_index]
        ii_names = value_index
    else:
        intensity_index = [val for val in index.values() if val != label_index]
        ii_names = [key for key, val in index.items() if val != label_index]
    return label_index, intensity_index, ii_names


def do_print(self: Any, indent: str, exe_start: float, s: str, is_err: bool = False, prefix: bool = True):
    s = indent + ((type(self).__name__ + f'({(time.time() - exe_start):0{11}.3f}): ') if prefix else '') \
        + s
    if is_err:
        print(s, file=sys.stderr)
    else:
        print(s)


class Pipeline(JSONSerializable):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        raise NotImplementedError

    def print(self, meta: Meta, s: str, is_err: bool = False, prefix: bool = True):
        do_print(self, meta.indent, meta.exe_start, s, is_err, prefix)

    def __repr__(self):
        return f'{type(self).__name__}({str(self.__dict__)}'

    def to_dict(self) -> dict:
        return self.__dict__


class TransformerDataset(data.Dataset, Sized):
    def __init__(self, wrapped: data.Dataset):
        self.wrapped = wrapped

    def _transform(self, data):
        raise NotImplementedError

    def __len__(self):
        return len(self.wrapped)

    def __getitem__(self, item):
        return self._transform(self.wrapped[item])


class TransformerModule(Pipeline):
    pass


class MetaModule(Pipeline):
    pass


class ShapelessInMemoryDataset(data.Dataset):
    def __init__(self, wrapped: Union[torch.Tensor, np.ndarray, data.Dataset, List, Tuple]):
        if isinstance(wrapped, torch.Tensor) or isinstance(wrapped, np.ndarray) or isinstance(wrapped, list) \
                or isinstance(wrapped, tuple):
            self.data = wrapped
        elif isinstance(wrapped, ShapelessInMemoryDataset):
            self.data = wrapped.data
        else:
            self.data = [entry for entry in wrapped]
            del wrapped

    def __getitem__(self, index) -> Union[torch.Tensor, np.ndarray]:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'SimpleInMemoryDataset({repr(self.data)})'

    def __str__(self):
        return f'SimpleInMemoryDataset({str(self.data)})'


class ArrayInMemoryDataset(ShapelessInMemoryDataset):
    def __init__(self, wrapped: Union[torch.Tensor, np.ndarray, data.Dataset]):
        super().__init__(wrapped)
        if isinstance(self.data, list) or isinstance(self.data, tuple):
            if isinstance(self.data[0], torch.Tensor):
                self.data: torch.Tensor = torch.stack(self.data)
            else:
                self.data: np.ndarray = np.stack(self.data)

    def __repr__(self):
        return f'ArrayInMemoryDataset({self.data})'

    def __str__(self):
        return f'ArrayInMemoryDataset({str(self.data)})'


class ShapelessInMemoryModule(Pipeline):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        check_meta_and_data(dataset, meta)
        self.print(meta, 'Loading dataset into memory.')
        t = time.time()
        dataset = ShapelessInMemoryDataset(dataset)
        t = time.time() - t
        self.print(meta, f'Loading dataset into memory completed in {t:.3f}s - average is {t / len(dataset):.3f}s.')
        return dataset, meta


class InMemoryModule(Pipeline):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        check_meta_and_data(dataset, meta)
        self.print(meta, 'Loading dataset into memory.')
        t = time.time()
        dataset = ArrayInMemoryDataset(dataset)
        t = time.time() - t
        self.print(meta, f'Loading dataset into memory completed in {t:.3f}s - average is {t / len(dataset):.3f}s.')
        return dataset, meta


class TorchConversionDataset(TransformerDataset):
    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            return data
        return torch.from_numpy(data)


class TorchConversionModule(MetaModule):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        dataset, meta = check_meta_and_data(dataset, meta)
        if isinstance(dataset, ArrayInMemoryDataset):
            self.print(meta, 'Converting array-dataset to pytorch.')
            dataset.data = dataset.data if isinstance(dataset.data, torch.Tensor) else torch.from_numpy(dataset.data)
        elif isinstance(dataset, ShapelessInMemoryDataset):
            self.print(meta, 'Converting shapeless-list-dataset to pytorch.')
            dataset.data = [(data if isinstance(data, torch.Tensor) else torch.from_numpy(data))
                            for data in dataset.data]
        else:
            self.print(meta, 'Converting dataset to pytorch lazily.')
            dataset = TorchConversionDataset(dataset)
        return dataset, meta


class SequenceModule(MetaModule):
    def __init__(self, sub_modules: List[Pipeline], ignore_none: bool = False):
        super().__init__()
        self.sub_modules = sub_modules
        self.ignore_none = ignore_none

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        meta = empty_meta() if meta is None else meta
        self.print(meta, f'Executing sequence with {len(self.sub_modules)} elements.')
        sub_meta = meta._replace(indent=meta.indent + '\t')
        t = time.time()
        no_output = False
        for module in self.sub_modules:
            output = module(dataset, sub_meta)
            if output is None and self.ignore_none:
                self.print(meta, f'Module did not provide any output. Terminating sequence!')
                no_output = True
                break
            elif output is None:
                raise ValueError(f'Expected output from module {str(module)} but None was given!')
            dataset, sub_meta = output
        t = time.time() - t
        meta = sub_meta._replace(indent=meta.indent)
        self.print(meta, f'Executing sequence took {t:.3f}s.')
        return None if no_output else (dataset, meta)


class Combiner(MergingDatasetWrapper):
    def __init__(self, wrapped: Iterable[data.Dataset], dim: int = 0):
        super().__init__(wrapped)
        self.dim = dim

    def __getitem__(self, index):
        lis = super().__getitem__(index)
        if len(lis) <= 1:
            return lis[0]
        if isinstance(lis[0], torch.Tensor):
            res = torch.cat(lis, dim=self.dim)
        else:
            res = np.concatenate(lis, axis=self.dim)
        return res


def merge_datasets_and_metas(indent: str, output_data: List[data.Dataset], output_meta: List[Meta], num_np: int,
                             num_torch: int, num_total: int, dim: int, check_meta: bool) -> Tuple[data.Dataset, Meta]:
    new_meta = merge_meta(output_meta, checked=check_meta)
    meta = new_meta._replace(indent=indent)
    if len(output_data) == 1:
        dataset = output_data[0]
    elif not output_data:
        raise ValueError('No data given!')
    elif num_np + num_torch == num_total:  # everything in memory
        if num_torch >= num_np:
            output_data = [(od.data if isinstance(od.data, torch.Tensor) else torch.from_numpy(od.data))
                           for od in output_data]
            output_data = torch.cat(output_data, dim + 1)
        else:
            output_data = [(od.data.detach().cpu().numpy() if isinstance(od.data, torch.Tensor) else od.data)
                           for od in output_data]
            output_data = np.concatenate(output_data, dim + 1)
        dataset = ArrayInMemoryDataset(output_data)
    else:
        dataset = Combiner(output_data, dim=dim)
    return dataset, meta


# runs additional pipeline channels "in-parallel"
# afterwards combines them again into one stacked dataset and meta
# optionally the original data can be kept
# This modules allows None dataset and meta inputs!!!
class DistributorModule(MetaModule):
    def __init__(self, distributed_modules: List[Pipeline], keep_source: bool = False, dim: int = 0,
                 check_meta: bool = False, ignore_none: bool = True) -> None:
        super().__init__()
        self.distributed_modules: List[Pipeline] = distributed_modules
        self.keep_source = keep_source
        self.dim = dim
        self.check_meta = check_meta
        self.ignore_none = ignore_none

    @staticmethod
    def _count_npt(od: data.Dataset, num_np: int, num_torch: int) -> Tuple[int, int]:
        if isinstance(od, ArrayInMemoryDataset):
            if isinstance(od.data, torch.Tensor):
                num_torch += 1
            else:
                num_np += 1
        return num_np, num_torch

    def _execute_distributed(self, dataset: data.Dataset, output_data: List[data.Dataset], output_meta: List[Meta],
                             sub_meta: Meta) -> Tuple[int, int]:
        num_np, num_torch = 0, 0
        ssub_meta = sub_meta._replace(indent=sub_meta.indent + '\t')
        for module in self.distributed_modules:
            self.print(sub_meta, '- - - - - - - - - - - - - - - - - -', prefix=False)
            output = module(dataset, ssub_meta)
            if output is None and self.ignore_none:
                self.print(sub_meta, 'Module produced no-output. Ignoring.')
                continue
            elif output is None:
                raise RuntimeError(f'Expected output from module {module} but None was given!')
            od, om = output
            output_data.append(od)
            output_meta.append(om)
            num_np, num_torch = self._count_npt(od, num_np, num_torch)
        if self.distributed_modules:
            self.print(sub_meta, '- - - - - - - - - - - - - - - - - -', prefix=False)
        return num_np, num_torch

    def _create_output_dataset(self, meta: Meta, output_data: List[data.Dataset], output_meta: List[Meta], num_np: int,
                               num_torch: int) -> Tuple[data.Dataset, Meta]:
        return merge_datasets_and_metas(meta.indent, output_data, output_meta, num_np, num_torch,
                                        len(self.distributed_modules), self.dim, self.check_meta)

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        meta = empty_meta() if meta is None else meta
        self.print(meta, f'Executing with {len(self.distributed_modules)} child'
                         f'{"" if len(self.distributed_modules) == 1 else "s"} and keep_source={self.keep_source}.')
        output_data, output_meta = ([dataset], [meta]) if self.keep_source else ([], [])
        sub_meta = meta._replace(indent=meta.indent + '\t')
        t = time.time()
        num_np, num_torch = self._execute_distributed(dataset, output_data, output_meta, sub_meta)
        t = time.time() - t
        self.print(meta, f'Execution of submodules completed in {t:.3f}s. Merging with check_meta={self.check_meta}. ')
        t = time.time()
        res = self._create_output_dataset(meta, output_data, output_meta, num_np, num_torch)
        t = time.time() - t
        self.print(meta, f'Merging completed in {t:.3f}s.')
        return res


class DynamicPerChannelDistributorModule(DistributorModule):
    def __init__(self, module_factory: Callable, keep_source: bool = False, dim: int = 0,
                 check_meta: bool = False):
        super().__init__([], keep_source, dim, check_meta)
        self.module_factory = module_factory

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        meta = empty_meta() if meta is None else meta
        self.print(meta, f'Constructing dynamic modules for channels {str(meta.channel_names)}.')
        self.distributed_modules = [self.module_factory(cn) for cn in meta.channel_names]
        return super().__call__(dataset, meta)


class _ChannelWhitelistDataset(TransformerDataset):
    def __init__(self, wrapped: data.Dataset, retained_channels: List[int]):
        super().__init__(wrapped)
        self.whitelisted = retained_channels

    def _transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        sl = data[self.whitelisted]
        # now copy in order to free the not needed stuff for garbage collection...
        return sl.detach().clone() if isinstance(sl, torch.Tensor) else np.copy(sl)


class WhitelistModule(TransformerModule):
    def __init__(self, retained_channels: Union[str, Iterable[str]]):
        super().__init__()
        retained_channels = [retained_channels] if isinstance(retained_channels, str) else retained_channels
        self.whitelisted = set(retained_channels)

    def get_channel_ids(self, meta: Meta):
        return list(map(lambda t: t[0],
                        filter(lambda n: n[1] in self.whitelisted,
                               enumerate(meta.channel_names))))

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        self.print(meta, f'Whitelisting {str(self.whitelisted)} out of {str(meta.channel_names)}.')
        assert not any(map(lambda cn: cn not in meta.channel_names, self.whitelisted))
        channel_ids = self.get_channel_ids(meta)
        if len(channel_ids) == len(meta.channel_names):
            self.print(meta, 'Returning input dataset, as all channels would be allowed by the whitelist')
            return dataset, meta
        if isinstance(dataset, ArrayInMemoryDataset):
            self.print(meta, f'Slicing ArrayInMemoryDataset.')
            whitelisted_data = dataset.data[:, channel_ids]
            dataset = ArrayInMemoryDataset(
                whitelisted_data.detach().clone() if isinstance(whitelisted_data, torch.Tensor) \
                    else np.copy(whitelisted_data))
        else:
            self.print(meta, f'Creating wrapper.')
            dataset = _ChannelWhitelistDataset(dataset, channel_ids)
        meta = meta._replace(channel_names=list(self.whitelisted))
        self.print(meta, f'Successfully created whitelist for channels {str(self.whitelisted)} retaining id\'s '
                         f'{str(channel_ids)}.')
        return dataset, meta


class BlacklistModule(TransformerModule):
    def __init__(self, removed_channels: Union[str, Iterable[str]]):
        super().__init__()
        removed_channels = [removed_channels] if isinstance(removed_channels, int) else removed_channels
        self.blacklisted = set(removed_channels)

    def get_channel_ids(self, meta: Meta):
        blacklisted = set(filter(lambda n: n[1] not in self.blacklisted,
                                 enumerate(meta.channel_names)))
        return [i for i, _ in blacklisted], [n for _, n in blacklisted]

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        self.print(meta, f'Blacklisting {str(self.blacklisted)} out of {str(meta.channel_names)}.')
        channel_ids, channel_names = self.get_channel_ids(meta)
        if len(channel_ids) == len(meta.channel_names):
            self.print(meta, 'Returning input dataset, as all channels would be allowed by the blacklist')
            return dataset, meta
        if isinstance(dataset, ArrayInMemoryDataset):
            blacklisted_data = dataset.data[:, channel_ids]
            dataset = ArrayInMemoryDataset(
                blacklisted_data.detach().clone() if isinstance(blacklisted_data, torch.Tensor) \
                    else np.copy(blacklisted_data))
        else:
            dataset = _ChannelWhitelistDataset(dataset, channel_ids)
        meta = meta._replace(channel_names=channel_names)
        self.print(meta, f'Successfully created blacklist for channels {str(self.blacklisted)} retaining id\'s '
                         f'{str(channel_ids)}.')
        return dataset, meta


class AdaptorDataset(TransformerDataset):

    def __init__(self, wrapped: data.Dataset, fun: Callable):
        super().__init__(wrapped)
        self.fun = fun

    def __getitem__(self, item):
        return self.fun(self.wrapped[item], item)


class UnsupervisedSklearnAdaptor(JSONSerializable):
    def __init__(self,
                 clazz: Union[str, Tuple[str, Callable]],
                 params: Optional[Dict[str, Any]] = None,
                 per_channel: bool = False,
                 per_data_point: bool = False,
                 image_features: bool = False,
                 clear_on_predict: bool = False,
                 allow_no_fit: bool = False,
                 init_file: Optional[str] = None,
                 save_file: Optional[str] = None):
        class_name, cons_fun = (clazz, utils.get_class_constructor(clazz)) if isinstance(clazz, str) else clazz
        self.class_name = class_name
        self.constructor = (lambda: cons_fun(**params)) if params is not None else cons_fun
        self.per_channel = per_channel
        self.per_data_point = per_data_point
        self.image_features = image_features
        self.fit_model = None
        self.allow_no_fit = allow_no_fit
        self.clear_on_predict: bool = clear_on_predict
        self.params = params
        if init_file is not None and (not path.exists(init_file) or not path.isfile(init_file)):
            print(
                f'WARNING: Init-file "{init_file}" provided for adaptor of class {class_name} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            init_file = None
        self.init_file = init_file
        if self.init_file is not None:
            print(f'Valid init file found at {init_file}. Loading.')
            with open(init_file, 'rb') as fd:
                self.fit_model = pickle.load(fd)
        if init_file is not None and (not path.exists(save_file) or not path.isfile(save_file)):
            print(
                f'WARNING: Save-file "{save_file}" provided for adaptor of class {class_name} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            save_file = None
        self.save_file = save_file

    def get_params_as_dict(self) -> Dict[str, Any]:
        return {
            'class_name': self.class_name,
            'params': self.constructor().get_params(deep=True),
            'per_channel': self.per_channel,
            'per_data_point': self.per_data_point,
            'image_features': self.image_features,
            'allow_no_fit': self.allow_no_fit,
            'clear_on_predict': self.clear_on_predict,
            'init_file': self.init_file,
            'save_file': self.save_file
        }

    @staticmethod
    def _rearrange_channel(data: np.ndarray, per_data_point) -> np.ndarray:
        # swaps the first two dimensions, so that the channel is now the first one...
        if per_data_point:
            return data
        return np.transpose(data, (1, 0) + tuple(range(2, len(data.shape))))

    @staticmethod
    def _to_sample_feature_shape(data: np.ndarray, image_features: bool) -> np.ndarray:
        # remember, we assume that the last two dims are the image dimensions
        if image_features and data.ndim >= 3:
            if data.ndim == 3:
                # if we only have 3 dimensions, then this is simple...
                return data.transpose((1, 2, 0))
            else:
                # if we have to account for more dimensions - it get's a bit trickier
                # assumption: (C, something..., H, W)
                # first ensure that we have (reverse something..., W, H, C )
                data = np.transpose(data, tuple(range(data.ndim - 3, 0, -1)) + (0,))
                # then swap W and H
                data = np.swapaxes(data, axis1=-2, axis2=-3)
                # and now return with all other dimensions being flattened
                return data.reshape((-1,) + data.shape[-3:])
        return data.reshape((data.shape[0], -1)).transpose()

    def _train_per_channel(self, data: np.ndarray, model: Union[Any, Dict[str, Any]], ci: Dict[str, int]):
        data = self._to_sample_feature_shape(data, self.image_features)
        if self.per_channel:
            for name, actual_model in model.items():
                relevant_view = data[..., ci[name]]
                actual_model.fit(relevant_view.reshape(relevant_view.shape + (1,)))
        else:
            model.fit(data)

    def _train_per_data_point(self, data: data.Dataset, ci: Dict[str, int]):
        if self.per_data_point:
            for data_entry, data_model in zip(data, self.fit_model):
                self._train_per_channel(self._rearrange_channel(data_entry, True), data_model, ci)
        else:
            assert isinstance(data, ArrayInMemoryDataset)
            self._train_per_channel(self._rearrange_channel(data.data, False), self.fit_model, ci)

    def _construct_per_channel(self, channel_names: List[str]) -> Union[Any, Dict[str, Any]]:
        if self.per_channel:
            return {cn: self.constructor() for cn in channel_names}
        return self.constructor()

    def _construct(self, dataset: data.Dataset, channel_names: List[str]) -> Union[Any, Dict[str, Any]]:
        return [self._construct_per_channel(channel_names) for _ in range(len(dataset))] \
            if self.per_data_point else self._construct_per_channel(channel_names)

    def _force_loaded_dataset(self, dataset: data.Dataset, indent: str) -> data.Dataset:
        if not self.per_data_point and not isinstance(dataset, ArrayInMemoryDataset) and \
                not isinstance(dataset, ShapelessInMemoryDataset):
            print(indent + 'Found non-in-memory dataset but sklearn model requires in-memory-data. Loading!')
            t = time.time()
            dataset = ShapelessInMemoryDataset(dataset) if self.per_data_point else ArrayInMemoryDataset(dataset)
            t = time.time() - t
            print(indent + f'Loading took {t:.3f}s!')
        return dataset

    def fit(self, dataset: data.Dataset, meta: Meta) -> ArrayInMemoryDataset:
        indent = meta.indent + '\t'
        dataset = self._force_loaded_dataset(dataset, indent)
        channel_names = meta.channel_names
        print(indent + f'Constructing sklearn module(s) with per_channel={self.per_channel} and '
                       f'per_data_point={self.per_data_point} for class {self.class_name}')
        t = time.time()
        self.fit_model = self._construct(dataset, channel_names)
        t = time.time() - t
        print(indent + f'Construction completed successfully for class {self.class_name} and took {t:.3f}s. '
                       f'Training on dataset with {len(dataset)} elements.')
        ci = channel_index(meta)
        t = time.time()
        self._train_per_data_point(dataset, ci)
        t = time.time() - t
        print(indent + f'Training completed successfully for class {self.class_name} and took {t:.3f}s - '
                       f'average {(t / len(dataset)):.3f}s.')
        if self.save_file is not None:
            print(indent + f'Saving trained model to "{self.save_file}".')
            with meta.run_hook.open_artifact_file(self.save_file, 'wb') as fd:
                pickle.dump(self.fit_model, fd)
        return dataset

    def _predict_per_channel(self, data, model, new_ci: Dict[str, int]) -> np.ndarray:
        data = self._to_sample_feature_shape(data, self.image_features)
        if self.per_channel:
            predictions = [np.squeeze(actual_model.predict(data[..., (new_ci[cn],)])) for cn, actual_model in
                           model.items()]
            res = np.stack(predictions)
            # res = res.transpose((res.ndim-1,)+tuple(range(0,res.ndim-1)))
            return res
        else:
            res = model.predict(data)
            if self.image_features:
                return res.reshape((1,)+res.shape)
            return res.reshape((-1, 1)).transpose()

    def _predict_per_data_point(self, data: ArrayInMemoryDataset, new_ci: Dict[str, int]) -> data.Dataset:
        if self.per_data_point:
            combined = [self._predict_per_channel(self._rearrange_channel(data_entry, True), data_model, new_ci)
                        for data_entry, data_model in zip(data, self.fit_model)]
            return ShapelessInMemoryDataset(combined)
        else:
            assert isinstance(data, ArrayInMemoryDataset)
            return ArrayInMemoryDataset(self._predict_per_channel(self._rearrange_channel(data.data, False),
                                                                  self.fit_model,
                                                                  new_ci))

    def predict(self, dataset: data.Dataset, meta: Meta) -> data.Dataset:
        if not self.allow_no_fit and self.fit_model is None:
            raise RuntimeError('Cannot predict with a not yet fit model!')
        elif self.allow_no_fit and self.fit_model is None:
            self.fit_model = self._construct(dataset, meta.channel_names)
        indent = meta.indent + '\t'
        dataset = self._force_loaded_dataset(dataset, indent)
        new_ci = channel_index(meta)
        print(indent + f'Performing prediction on dataset of size {len(dataset)}.')
        t = time.time()
        res = self._predict_per_data_point(dataset, new_ci)
        t = time.time() - t
        print(indent + f'Prediction took {t:.3f}s - average {(t / len(dataset)):.3f}s.')
        if self.clear_on_predict:
            del self.fit_model
            self.fit_model = None
        return res

    def __str__(self):
        return f'UnsupervisedSklearnAdaptor(class_name="{self.class_name}", params={str(self.params)}, ' \
               f'per_channel={self.per_channel}, per_data_point={self.per_data_point}, ' \
               f'clear_on_predict={self.clear_on_predict})'


# Adaptor for sklearn-like Transformers
# transformer: the sklearn adaptor to use
# along the channel dimensions
# channel_res: names for the resulting channels
# reshape: Optionally a tuple of two functions for first reshaping the data into a shape suitable for the Transformer
# and then reshaping it back for storage
# per_data_channel: if the
class UnsupervisedSklearnAdaptorModule(TransformerModule):
    def __init__(self, transformer: UnsupervisedSklearnAdaptor,
                 res_prediction_channel_name: str = 'label',
                 separator: str = '-',
                 do_fit: bool = True,
                 do_predict: bool = True):
        self.transformer = transformer
        self.res_prediction_channel_name = res_prediction_channel_name
        self.do_fit = do_fit
        self.do_predict = do_predict
        self.separator = separator

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        check_meta_and_data(dataset, meta)
        if not (self.do_fit or self.do_predict):
            self.print(meta, 'Neither do_fit nor do_predict were specified. This does not make sense, as this module '
                             'does absolutely nothing in this configuration?!?', is_err=True)
            return dataset, meta
        if self.do_fit:
            self.print(meta, f'Performing fit on sklearn module: {str(self.transformer)}.')
            dataset = self.transformer.fit(dataset, meta)
            self.print(meta, f'Fit completed.')
        if self.do_predict:
            self.print(meta, f'Performing predict on sklearn module: {str(self.transformer)}.')
            dataset = self.transformer.predict(dataset, meta)
            self.print(meta, f'Predict completed.')
            channel_names = [self.res_prediction_channel_name] if len(dataset) > 0 and len(dataset[0]) == 1 else \
                [cn + self.separator + self.res_prediction_channel_name for cn in meta.channel_names]
            meta = meta._replace(channel_names=channel_names)
        return dataset, meta


class TerminationModule(Pipeline):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Optional[Tuple[data.Dataset, Meta]]:
        return None

class ProbabilityToValueDataset(TransformerDataset):
    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            return torch.argmax(data, dim=0)
        else:
            return np.argmax(data, axis=0).reshape((1,) + data.shape[1:])


class ProbabilityToValue(Pipeline):
    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Optional[Tuple[data.Dataset, Meta]]:
        return ProbabilityToValueDataset(dataset), meta

class MaskCopyDataset(TransformerDataset):

    def __init__(self, wrapped: data.Dataset, mask_index: int, mask_value: int = -1):
        super().__init__(wrapped)
        self.mask_index = mask_index
        self.mask_value = mask_value

    def _transform(self, data):
        mask = data[self.mask_index] == self.mask_value
        for i in range(data.shape[0]):
            if i == self.mask_index:
                continue
            data[i, mask] = self.mask_value
        return data

class MaskCopy(Pipeline):
    def __init__(self, mask_index: int = 0, mask_value: int = -1):
        super().__init__()
        self.mask_index = mask_index
        self.mask_value = mask_value

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        self.print(meta, 'Creating mask-copy-wrapper.')
        return MaskCopyDataset(dataset, self.mask_index, self.mask_value), meta


class ThresholdDataset(TransformerDataset):

    def __init__(self, wrapped: data.Dataset, threshold: float):
        super().__init__(wrapped)
        self.threshold = threshold

    def _transform(self, data):
        data = utils.torch_as_np(data)
        res = np.zeros_like(data, dtype=np.int32)
        mask = data > self.threshold
        res[mask] = 1
        return res

class SimpleThresholdModule(Pipeline):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        self.print(meta, 'Creating mask-copy-wrapper.')
        meta = meta._replace(channel_names=['threshold'])
        return ThresholdDataset(dataset, self.threshold), meta