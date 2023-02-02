import sys
import time
import traceback
from typing import cast

import datadings
import datadings.writer
import numpy as np
import torch
from frozendict import frozendict

import serialisation
from dataclasses import dataclass
from pipeline import *


class SelectionCriterion(JSONSerializable):

    def __repr__(self) -> str:
        return f'{type(self)}({str(self.__dict__)})'

    def __str__(self) -> str:
        return self.__repr__()


class IndexSelectionCriterion(SelectionCriterion):
    def __init__(self, index: int):
        assert index >= 0
        self.index = index

    def __eq__(self, o: object) -> bool:
        return isinstance(o, IndexSelectionCriterion) and self.index == o.index

    def __hash__(self) -> int:
        return hash(self.index)


class NameSelectionCriterion(SelectionCriterion):
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, o: object) -> bool:
        return isinstance(o, NameSelectionCriterion) and self.name == o.name

    def __hash__(self) -> int:
        return hash(self.name)


MULTI_EVALUATION_OR = 'or'
MULTI_EVALUATION_AND = 'and'
MULTI_EVALUATION_OR_NOT = 'or_not'
MULTI_EVALUATION_AND_NOT = 'and_not'


class PredicateSelectionCriterion(SelectionCriterion):
    def __init__(self, meta_type: Optional[int] = None, channel_name: Optional[Union[str, Iterable[str]]] = None,
                 split: Optional[str] = None, meta_evaluation: str = MULTI_EVALUATION_AND,
                 channel_evaluation: str = MULTI_EVALUATION_AND):
        assert meta_type is not None or channel_name is not None or split is not None
        self.meta_type: Optional[int] = meta_type
        self.channel_name: Optional[Union[str, Iterable[str]]] = channel_name
        self.split: Optional[str] = split
        self.meta_evaluation: str = meta_evaluation
        self.channel_evaluation: str = channel_evaluation

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PredicateSelectionCriterion) and self.meta_type == o.meta_type and \
               self.channel_name == o.channel_name and self.split == o.split and \
               self.channel_evaluation == o.channel_evaluation and self.meta_evaluation == o.meta_evaluation

    def __hash__(self) -> int:
        return hash((self.meta_type, self.channel_name, self.split, self.meta_evaluation, self.channel_evaluation))


class Summary:
    @staticmethod
    def auto_name(meta: Meta) -> str:
        res = meta.split + '-' + types_as_name(meta)
        lc_cns = set(cn.lower() for cn in meta.channel_names)
        if 'classification' in lc_cns:
            res += '-' + 'classification'
        if 'clustering' in lc_cns:
            res += '-' + 'clustering'
        return res

    @staticmethod
    def from_list_infer_names(ls: List[Tuple[data.Dataset, Meta]]) -> "Summary":
        names: List[str] = [Summary.auto_name(meta) for _, meta in ls]
        if len(set(names)) != len(names):
            raise ValueError('Automatic Name Generation is ambiguous cannot create summary!\n'
                             f'Auto-names are: {str(names)}; Without duplicates: {str(set(names))};\n' +
                             f'Metas are: {str([meta for _, meta in ls])}')
        return Summary.from_list_with_names(ls, names)

    @staticmethod
    def from_list_with_names(ls: List[Tuple[data.Dataset, Meta]], names: List[str]) -> "Summary":
        return Summary(ls, lambda i, t: names[i])

    @staticmethod
    def merge_summaries(summaries: List['Summary']) -> 'Summary':
        if len(summaries) == 1:
            print('Merging a single summary with itself?!?')
            return summaries[0]
        elif len(summaries) < 1:
            raise ValueError('No summary to merge')
        key_sets = [set(s.data.keys()) for s in summaries]
        conflict_sets = list(filter(lambda s: len(s) >= 1,
                                    [s1.intersection(s2) for i, s1 in enumerate(key_sets) for j, s2 in
                                     enumerate(key_sets) if i < j]))
        if not conflict_sets:  # Yay simple to merge!!!
            combined = list(itert.chain.from_iterable([list(s.data.items()) for s in summaries]))
            in_list = [t for _, t in combined]
            return Summary(in_list, lambda i, _: combined[i][0])
        elif len(summaries) == 2:
            metas = [(summaries[0].data[key], summaries[1].data[key]) for key in conflict_sets[0]]
            if not any(map(lambda t: t[0] != t[1], metas)):
                res = summaries[0].copy()
                res.data.update(summaries[1].data)
                res.by_index = list(res.data.values())
                return res
            else:
                raise NotImplementedError('Cannot merge summaries with differing metas!')
        else:
            merged = Summary.merge_summaries(summaries[1:])
            return Summary.merge_summaries([summaries[0], merged])

    def __init__(self, ls: List[Tuple[data.Dataset, Meta]], name_fun: Callable):
        self.indent: str = '\t' * min([len(meta.indent) for _, meta in ls], default=0)
        self.exe_start: float = min([meta.exe_start for _, meta in ls], default=time.time())
        self.run_hook: Optional[RunHook] = None if not ls else next(iter(ls))[1].run_hook
        ls = [(d, meta._replace(indent=self.indent, exe_start=self.exe_start)) for d, meta in ls]
        self.data: Dict[str, Tuple[data.Dataset, Meta]] = {name_fun(i, t): t for i, t in enumerate(ls)}
        self.by_index: List[Tuple[data.Dataset, Meta]] = ls.copy()

    def __getitem__(self, item: Union[int, str, SelectionCriterion]) -> Tuple[data.Dataset, Meta]:
        if isinstance(item, SelectionCriterion):
            return self.by_criterion(item)
        elif type(item) is int:
            return self.by_index[item]
        return self.data[item]

    def __len__(self):
        return len(self.by_index)

    def __bool__(self):
        return len(self.by_index) >= 1

    def __repr__(self):
        return f'Summary({repr(self.data)})'

    def __str__(self):
        return f'Summary({str(self.data)})'

    def __contains__(self, item: Union[int, str, SelectionCriterion]):
        if isinstance(item, SelectionCriterion):
            if isinstance(item, IndexSelectionCriterion):
                return self.__contains__(item.index)
            elif isinstance(item, NameSelectionCriterion):
                return self.__contains__(item.name)
            elif isinstance(item, PredicateSelectionCriterion):
                return len(self.by_predicate(item.meta_type, item.channel_name, item.split,
                                             item.meta_evaluation, item.channel_evaluation)) > 0
            else:
                raise ValueError('Unknown type of selection criterion: ' + str(type(item)))
        elif type(item) is int:
            return 0 <= item < len(self.by_index)
        return item in self.data

    def __setitem__(self, key, value):
        #super().__setitem__(key, value)
        if key == 'indent':
            assert isinstance(value, str)
            self.indent = value
            self.data = {k: (d, meta._replace(indent=self.indent)) for k, (d, meta) in self.data.items()}
            self.by_index = list(self.data.items())
        else:
            raise NotImplementedError

    def set_hook(self, hook: RunHook):
        self.run_hook = hook
        self.data = {n: (d, meta._replace(run_hook=hook)) for n, (d, meta) in self.data.items()}
        self.by_index = list(self.data.items())

    def new_meta(self):
        return self.apply_to_meta(empty_meta())

    def apply_to_meta(self, meta: Meta):
        return meta._replace(indent=self.indent, exe_start=self.exe_start, run_hook=self.run_hook)

    def do_indent(self) -> 'Summary':
        self.indent += '\t'
        return self

    def undo_indent(self) -> 'Summary':
        self.indent = self.indent[:-1]
        return self

    def by_predicate(self, meta_type: Optional[int] = None, channel_name: Optional[Union[str, Iterable[str]]] = None,
                     split: Optional[str] = None, meta_evaluation: str = MULTI_EVALUATION_AND,
                     channel_evaluation: str = MULTI_EVALUATION_AND) -> Tuple[Tuple[data.Dataset, Meta], ...]:
        # holds a list of predicates, for which FALSE (!) means fulfillment
        predicates = []
        if meta_type is not None:
            if meta_evaluation == MULTI_EVALUATION_OR:
                predicates.append(lambda meta: meta_type & meta.type == 0)
            elif meta_evaluation == MULTI_EVALUATION_AND:
                predicates.append(lambda meta: meta_type & meta.type != meta_type)
            elif meta_evaluation == MULTI_EVALUATION_AND_NOT:
                predicates.append(lambda meta: meta_type & meta.type == meta_type)
            elif meta_evaluation == MULTI_EVALUATION_OR_NOT:
                predicates.append(lambda meta: meta_type & meta.type != 0)
            else:
                raise ValueError('Unkown evaluation type ' + meta_evaluation)
        if channel_name is not None:
            channel_name = [channel_name] if isinstance(channel_name, str) else channel_name
            # any of the specified channels is present
            if meta_evaluation == MULTI_EVALUATION_OR:
                predicates.append(lambda meta: not any(map(lambda cn: cn in meta.channel_names, channel_name)))
            # all of the specified channels are present
            elif meta_evaluation == MULTI_EVALUATION_AND:
                predicates.append(lambda meta: any(map(lambda cn: cn not in meta.channel_names, channel_name)))
            # all of the specified channels are not present
            elif meta_evaluation == MULTI_EVALUATION_AND_NOT:
                predicates.append(lambda meta: any(map(lambda cn: cn in meta.channel_names, channel_name)))
            # any of the specified channels is not present
            elif meta_evaluation == MULTI_EVALUATION_OR_NOT:
                predicates.append(lambda meta: not any(map(lambda cn: cn not in meta.channel_names, channel_name)))
            else:
                raise ValueError('Unkown evaluation type ' + meta_evaluation)
        if split is not None:
            predicates.append(lambda meta: meta.split != split)
        if not predicates:
            raise ValueError('Cannot identify dataset without predicate!!!')
        return tuple(t for t in self.by_index if not any(map(lambda p: p(t[1]), predicates)))

    def by_predicate_or_throw(self, meta_type: Optional[int] = None,
                              channel_name: Optional[Union[str, Iterable[str]]] = None,
                              split: Optional[str] = None, meta_evaluation: str = MULTI_EVALUATION_AND,
                              channel_evaluation: str = MULTI_EVALUATION_AND) -> Tuple[data.Dataset, Meta]:
        res = self.by_predicate(meta_type, channel_name, split, meta_evaluation, channel_evaluation)
        if len(res) != 1:
            raise ValueError('Cannot find exactly one dataset for the given specification. Found ' +
                             f'{len(res)}, expected 1. Predicate is {str(meta_type)}, {str(channel_name)}, {str(split)}'
                             + f'. Result is {str(res)}')
        return res[0]

    def by_criterion(self, item: SelectionCriterion, delete: bool = False, return_name: bool = False) \
            -> Union[Tuple[data.Dataset, Meta], Tuple[str, Tuple[data.Dataset, Meta]]]:
        if isinstance(item, IndexSelectionCriterion):
            res = self.__getitem__(item.index)
        elif isinstance(item, NameSelectionCriterion):
            res = self.__getitem__(item.name)
        elif isinstance(item, PredicateSelectionCriterion):
            res = self.by_predicate_or_throw(item.meta_type, item.channel_name, item.split,
                                             item.meta_evaluation, item.channel_evaluation)
        else:
            raise ValueError(f'Unknown type of SelectionCriterion ({type(item)}) with value:' + str(item))
        if delete or return_name:
            matches: List[str] = [name for name, t in self.data.items() if t is res]
            if len(matches) != 1:
                print(self.indent + f'Found {len(matches)} many entries in the name to dataset map for one reference!',
                      file=sys.stderr)
        else:
            matches = []
        if delete:
            for name in matches:
                del self.data[name]
            self.by_index = [t for t in self.by_index if t is not res]
        return (matches[0], res) if return_name else res

    def add(self, dataset: data.Dataset, meta: Meta, name: Optional[str] = None) -> 'Summary':
        if dataset is None or meta is None:
            raise ValueError(f'Cannot add non-existing datasets or metas! Given dataset={str(dataset)} and meta='
                             f'{short_repr(meta)}')
        meta = self.apply_to_meta(meta)
        t = (dataset, meta)
        if name is None:
            name = self.auto_name(meta)
        else:
            assert isinstance(name, str), 'Non string names are not supported!'
        if name in self.data:
            raise ValueError(f'Name {name} is ambiguous!')
        self.by_index.append(t)
        self.data[name] = t
        return self

    def copy(self):
        res = Summary([], lambda a: a)
        res.by_index = self.by_index.copy()
        res.data = dict(res.by_index)
        res.indent = self.indent
        res.exe_start = self.exe_start
        res.run_hook = self.run_hook
        return res


class MultiPipeline(JSONSerializable):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, summary: Summary) -> Summary:
        raise NotImplementedError

    def print(self, summary: Summary, s: str, is_err: bool = False, prefix: bool = True):
        do_print(self, summary.indent, summary.exe_start, s, is_err, prefix)


class MultiMetaModule(MultiPipeline):
    pass


class MultiTransformerDataset(data.Dataset):
    def __init__(self, wrapped: data.Dataset):
        self.wrapped = wrapped

    def __getitem__(self, index) -> Tuple:
        return self._transform(self.wrapped[index], index)

    def __len__(self):
        return len(self.wrapped)

    def _transform(self, data, index):
        raise NotImplementedError


class MultipipelineAdaptorModule(MetaModule):
    def __init__(self, multi_pipeline: Optional[MultiPipeline],
                 dataset_name: Optional[Union[str, List[str]]] = None,
                 selection_criterion: Optional[SelectionCriterion] = None,
                 merge_dim: int = 0,
                 check_meta: bool = False):
        self.multi_pipeline: Optional[MultiPipeline] = multi_pipeline
        self.dataset_name: Optional[List[str]] = [dataset_name] if dataset_name is not None and \
                                                                   isinstance(dataset_name, str) else dataset_name
        self.selection_criterion = selection_criterion
        self.merge_dim = merge_dim
        self.check_meta = check_meta

    def get_data(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> List[Tuple[data.Dataset, Meta]]:
        return [(dataset, meta)]

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> Tuple[data.Dataset, Meta]:
        has_name = self.dataset_name is not None
        data = self.get_data(dataset, meta)
        self.print(meta,
                   'Executing ' + ('no ' if self.multi_pipeline is None else '') + 'MultiPipeline within a Pipeline.' +
                   ' For this the given meta will be wrapped in a summary - with dataset-name ' +
                   (str(self.dataset_name) if has_name else 'being inferred') + '.')
        t = time.time()
        summary = Summary.from_list_with_names(data, self.dataset_name) if has_name \
            else Summary.from_list_infer_names([(dataset, meta)]).do_indent()
        summary = summary if self.multi_pipeline is None else self.multi_pipeline(summary)
        t = time.time() - t
        do_merge = self.selection_criterion is None
        self.print(meta, f'Executed MultiPipeline within a Pipeline in {t:.3f}s. '
                   + ('Performing merge.' if do_merge else f'Selecting according to {str(self.selection_criterion)}'))
        t = time.time()
        if not summary:
            raise RuntimeError('Cannot create result if summary is empty!!!')
        if len(summary) == 1:
            res = summary.by_index[0]
            self.print(meta, 'Only one dataset present in result. Ignoring selection criterion and returning this.')
        elif do_merge:
            datasets, metas = [dataset for dataset, _ in summary.by_index], [meta for _, meta in summary.by_index]
            num_np = sum(
                map(lambda d: (1 if isinstance(d, ArrayInMemoryDataset) and not isinstance(d.data, torch.Tensor)
                               else 0), datasets))
            num_torch = sum(map(lambda d: (1 if isinstance(d, ArrayInMemoryDataset) and isinstance(d.data, torch.Tensor)
                                           else 0), datasets))
            res = merge_datasets_and_metas(meta.indent, datasets, metas, num_np, num_torch, len(datasets),
                                           self.merge_dim,
                                           self.check_meta)
            self.print(meta, f'Merge completed successfully returning result in {time.time() - t:.3f}s.')
        else:
            res = summary[self.selection_criterion]
            self.print(meta, f'Found result with meta {short_repr(res[1])} using criterion ' +
                       f'{str(self.selection_criterion)} in {time.time() - t:.3f}s. Returning.')
        return res


class DistributorMultipipelineAdaptorModule(MultipipelineAdaptorModule):
    def __init__(self, distributed_modules: List[Pipeline],
                 multi_pipe_module: Optional[MultiPipeline],
                 dataset_names: Optional[List[str]] = None,
                 keep_source: bool = False, dim: int = 0,
                 check_meta: bool = False,
                 selection_criterion: Optional[Union[Dict[str, Any], str, int]] = None, ) -> None:
        super(MultipipelineAdaptorModule).__init__(multi_pipe_module, dataset_names, selection_criterion, dim,
                                                   check_meta)
        self.distributed_modules: List[Pipeline] = distributed_modules
        self.keep_source = keep_source

    def get_data(self, dataset: Optional[data.Dataset], meta: Optional[Meta]) -> List[Tuple[data.Dataset, Meta]]:
        self.print(meta, f'Executing {len(self.distributed_modules)} Pipeline modules with keep_source=' +
                   f'{str(self.keep_source)}in a distributed manner in order to generate input for the ' +
                   'MultiPipeline.')
        res = super().get_data(dataset, meta) if self.keep_source else []
        t = time.time()
        for pipeline in self.distributed_modules:
            res.append(pipeline(dataset, meta))
        t = time.time() - t
        self.print(meta, f'Executing {len(self.distributed_modules)} Pipeline modules took {t:.3f}s.')
        return res


class MultiDistributorModule(MultiMetaModule):
    def __init__(self, distributed_modules: List[MultiPipeline],
                 keep_source: bool = False,
                 ignore_none: bool = False) -> None:
        super().__init__()
        self.distributed_modules: List[MultiPipeline] = distributed_modules
        self.keep_source = keep_source
        self.ignore_none = ignore_none

    def _execute_distributed(self, summary: Summary, output_summaries: List[Summary]):
        summary.do_indent()
        for module in self.distributed_modules:
            self.print(summary, '- - - - - - - - - - - - - - - - - -', prefix=False)
            output_summaries.append(module(summary.copy().do_indent()))
        if self.distributed_modules:
            self.print(summary, '- - - - - - - - - - - - - - - - - -', prefix=False)
        summary.undo_indent()

    def __call__(self, summary: Summary) -> Summary:
        self.print(summary, f'Executing with {len(self.distributed_modules)} child'
                            f'{"" if len(self.distributed_modules) == 1 else "s"} and keep_source={self.keep_source}.')
        output_summaries = [summary] if self.keep_source else []
        t = time.time()
        self._execute_distributed(summary, output_summaries)
        t = time.time() - t
        self.print(summary, f'Execution of submodules completed in {t:.3f}s. Merging summaries. ')
        t = time.time()
        if any(map(lambda s: s is None, output_summaries)):
            if self.ignore_none:
                self.print(summary, f'Some modules did not produce an output. Skipping as ignore_none=True.')
                output_summaries = [summary for summary in output_summaries if summary is not None]
            else:
                raise RuntimeError(f'Some modules di not produce output, but ignore_none=False!')
        if len(output_summaries) == 0:
            res = None
        elif len(output_summaries) == 1:
            res = output_summaries[0]
            res.indent = summary.indent
        else:
            res = Summary.merge_summaries(output_summaries)
            res.indent = summary.indent
        t = time.time() - t
        self.print(summary, f'Merging completed in {t:.3f}s.')
        return res


class MultiSequenceModule(MultiMetaModule):
    def __init__(self, sub_modules: List[MultiPipeline], ignore_none: bool = False):
        super().__init__()
        self.sub_modules = sub_modules
        assert sub_modules
        self.ignore_none = ignore_none

    def __call__(self, summary: Summary) -> Summary:
        self.print(summary, f'Executing sequence with {len(self.sub_modules)} elements.')
        summary = summary.do_indent()
        t = time.time()
        new_summary = summary
        for multi_module in self.sub_modules:
            new_summary = multi_module(summary)
            if new_summary is None:
                if self.ignore_none:
                    self.print(summary, f'Got no return from module. As ignore_none=True: terminating sequence!')
                    break
                else:
                    raise RuntimeError(f'Got no return from module but ignore_none=False!')
            summary = new_summary
        t = time.time() - t
        summary = summary.undo_indent()
        self.print(summary, f'Sequence execution took {t:.3f}s.')
        return summary if new_summary == summary else None


class PipelineAdaptorModule(MultiMetaModule):
    def __init__(self, selection_criterion: Optional[SelectionCriterion], pipe_module: Pipeline,
                 dataset_name: Optional[str] = None,
                 keep_source: bool = False):
        super().__init__()
        self.selection_criterion = selection_criterion
        self.pipe_module = pipe_module
        self.dataset_name = dataset_name
        self.delete = not keep_source

    def __call__(self, summary: Summary) -> Summary:
        self.print(summary, f'Preparing to execute PipelineModule {type(self.pipe_module)} in MultiPipeline. '
                            f'Dataset and meta are determined by criterion {self.selection_criterion}.')
        if self.selection_criterion is not None:
            name, (dataset, meta) = summary.by_criterion(self.selection_criterion, delete=self.delete, return_name=True)
        else:
            name, (dataset, meta) = self.dataset_name, (None, summary.new_meta())
        self.print(summary, f'Successfully found dataset with meta {short_repr(meta)} - executing module.')
        res = self.pipe_module(dataset, meta._replace(indent=meta.indent + '\t'))
        if res is None or (isinstance(res, tuple) and any(map(lambda e: e is None, res))):
            self.print(summary, f'Executing pipeline module {type(self.pipe_module)} did not return a result.')
            return summary
        dataset, meta = res
        self.print(summary, f'Successfully finished executing pipeline module {type(self.pipe_module)}, integrating '
                            'result into summary with ' + ('automatic-name.' if self.dataset_name is None else
                                                           f'name {self.dataset_name}.'))
        return summary.add(dataset, meta, self.dataset_name if self.dataset_name is not None else name)


class MultiTransformerModule(MultiPipeline):
    pass


class AssemblerDataset(MultiTransformerDataset):
    def __init__(self, wrapped: data.Dataset, ref_dataset: Union[ArrayInMemoryDataset, ShapelessInMemoryDataset]):
        super().__init__(wrapped)
        self.ref_data: List[Union[np.ndarray, torch.Tensor]] = ref_dataset.data

    def _transform(self, data, index):
        (data, ref_data), _ = utils.match_type(data, self.ref_data[index])
        # TODO verify this is actually correct, even though it right now solves my problems
        if data.shape[0] != ref_data.shape[0]:
            data = data.reshape((ref_data.shape[0],) + data.shape)
        res = np.array([ref_entry[data_entry] for ref_entry, data_entry in zip(ref_data, data)])
        return res


# Merges 2 datasets into one
class AssemblerModule(MultiTransformerModule):
    """
    Takes selection criteria for 2 datsets: values and labels
    The value dataset is assumed to be indexable by the labels
    """

    def __init__(self, value_criterion: SelectionCriterion, label_criterion: SelectionCriterion,
                 res_dataset_name: Optional[str] = None, delete_value: bool = True, delete_label: bool = True):
        super().__init__()
        self.value_criterion = value_criterion
        self.image_criterion = label_criterion
        self.res_name = res_dataset_name
        self.delete_value = delete_value
        self.delete_label = delete_label
        assert (not self.delete_label) or (self.res_name is not None)

    def __call__(self, summary: Summary) -> Summary:
        self.print(summary, f'Merging images specified by {self.image_criterion} and values specified by '
                            f'{self.value_criterion}.')
        name, (image_dataset, image_meta) = summary.by_criterion(self.image_criterion, delete=self.delete_label,
                                                                 return_name=True)
        value_dataset, value_meta = summary.by_criterion(self.value_criterion, delete=self.delete_label)
        assert len(image_dataset) == len(value_dataset)
        if not isinstance(value_dataset, ArrayInMemoryDataset) and not isinstance(value_dataset,
                                                                                  ShapelessInMemoryDataset):
            self.print(summary,
                       'Values are not in memory!!! As a lazy representation is too-slow, loading into memory!',
                       is_err=True)
            value_dataset = ShapelessInMemoryDataset(value_dataset)
        if isinstance(image_dataset, ArrayInMemoryDataset) and isinstance(value_dataset, ArrayInMemoryDataset):
            (data, ref_data), is_torch = utils.match_type(image_dataset.data, value_dataset.data)
            # TODO verify this is correct
            data = utils.move_channel_to_position(ref_data[data], target_position=1)
            res_dataset = ArrayInMemoryDataset(data)
        else:
            res_dataset = AssemblerDataset(image_dataset, value_dataset)

        return summary.add(res_dataset, value_meta, name if self.res_name is None else self.res_name)

class LazyBatchedPredDataset(TransformerDataset):
    def __init__(self, wrapped: data.Dataset, adaptor: 'SupervisedSklearnAdaptor', batch_size: int, ci: Dict[str, int]):
        super().__init__(wrapped)
        self.adaptor = adaptor
        self.batch_size = batch_size
        self.ci = ci
        self.last_batch_index: Optional[int] = None
        self.last_batch: Optional[np.ndarray] = None

    def load_batch(self, batch_index: int) -> np.ndarray:
        lower_bound = batch_index*self.batch_size
        upper_bound = min((batch_index+1)*self.batch_size, len(self.wrapped))
        if isinstance(self.wrapped, ArrayInMemoryDataset):
            data = utils.torch_as_np(self.wrapped.data[lower_bound:upper_bound])
        else:
            data = np.array([utils.torch_as_np(self.wrapped[i]) for i in range(lower_bound, upper_bound)])
        pred = self.adaptor._predict_per_data_point(ArrayInMemoryDataset(data), None, self.ci)
        return pred.data

    def __getitem__(self, item):
        if item < 0 or item >= len(self.wrapped):
            raise IndexError(f'Index {item} is out of range for dataset of length {len(self.wrapped)}!')
        batch_index = item // self.batch_size
        if self.last_batch_index is None or batch_index != self.last_batch_index:
            self.last_batch = self.load_batch(batch_index)
            self.last_batch_index = batch_index
        return self.last_batch[item % self.batch_size]


class SupervisedSklearnAdaptor(JSONSerializable):
    def __init__(self,
                 clazz: Union[str, Tuple[str, Callable]],
                 params: Optional[Dict[str, Any]] = None,
                 per_channel: bool = False,
                 per_data_point: bool = False,
                 predict_proba: bool = False,
                 predict_batch_size: int = -1,
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
        self.predict_batch_size = predict_batch_size
        self.fit_model = None
        self.allow_no_fit = allow_no_fit
        self.clear_on_predict: bool = clear_on_predict
        self.predict_proba = predict_proba
        self.params = params
        assert not self.clear_on_predict or predict_batch_size < 1, \
            'Cannot clear after predict call if lazy-batched-predictions are used'
        if init_file is not None and (not path.exists(init_file) or not path.isfile(init_file)):
            print(
                f'WARNING: Init-file "{init_file}" provided for adaptor of class {clazz} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            init_file = None
        self.init_file = init_file
        if self.init_file is not None and path.exists(init_file) and path.isfile(init_file):
            print(f'Valid init file found at {init_file}. Loading.')
            with open(init_file, 'rb') as fd:
                self.fit_model = pickle.load(fd)
        elif init_file is not None and (not path.exists(init_file) or not path.isfile(init_file)):
            print(
                f'WARNING: Init-file "{save_file}" provided for adaptor of class {clazz} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            self.init_file = None
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
    def _rearrange_channel(data: np.ndarray, per_data_point: bool) -> np.ndarray:
        # swaps the first two dimensions, so that the channel is now the first one...
        if per_data_point:
            return data
        return np.transpose(data, (1, 0) + tuple(range(2, len(data.shape))))

    @staticmethod
    def _rearrange_opt_channel(data: Optional[Union[ArrayInMemoryDataset, np.ndarray]], per_data_point: bool) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, ArrayInMemoryDataset):
            data = data.data
        return SupervisedSklearnAdaptor._rearrange_channel(data, per_data_point)

    @staticmethod
    def _to_sample_feature_shape(data: np.ndarray, image_features: bool) -> np.ndarray:
        # remember, we assume that the last two dims are the image dimensions and that the first one is the channel dim
        if image_features and data.ndim >= 3:
            if data.ndim == 3:
                # if we only have 3 dimensions, then this is simple...
                return np.transpose(data, (1, 2, 0))
            else:
                # if we have to account for more dimensions - it get's a bit trickier
                # assumption: (C, something..., H, W)
                # first ensure that we have (reverse something..., W, H, C )
                data = np.transpose(data, tuple(range(data.ndim - 3, 0, -1)) + (0,))
                # then swap W and H
                data = np.swapaxes(data, axis1=-2, axis2=-3)
                # and now return with all other dimensions being flattened
                return data.reshape((-1,) + data.shape[-3:])
        # First reshape (C, ..., H, W) into (C, ... * H * W)
        # Then transpose into (... * H * W, C), aka the standard sklearn format
        return data.reshape((data.shape[0], -1)).transpose()

    def _train_per_channel(self, data: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray],
                           model: Union[Any, Dict[str, Any]], ci: Dict[str, int]):
        data = self._to_sample_feature_shape(data, self.image_features)
        labels = self._to_sample_feature_shape(labels, self.image_features)
        if mask is not None:
            mask = mask.reshape(-1)
            labels = labels[mask]
        if self.per_channel:
            for name, actual_model in model.items():
                relevant_view = data[..., ci[name]]
                if mask is not None:
                    relevant_view = relevant_view[mask]
                actual_model.fit(relevant_view.reshape(relevant_view.shape + (1,)), labels.ravel())
        else:
            if mask is not None:
                data = data[mask]
            model.fit(data, labels.ravel())

    def _train_per_data_point(self, data: data.Dataset, labels: data.Dataset, mask: Optional[data.Dataset],
                              ci: Dict[str, int]):
        if self.per_data_point:
            mask = [None] * len(data) if mask is None else mask
            for data_entry, label_entry, mask_entry, data_model in zip(data, labels, mask, self.fit_model):
                self._train_per_channel(self._rearrange_channel(data_entry, True),
                                        self._rearrange_channel(label_entry, True),
                                        self._rearrange_opt_channel(mask_entry, True),
                                        data_model, ci)
        else:
            assert isinstance(data, ArrayInMemoryDataset) and isinstance(labels, ArrayInMemoryDataset) and \
                   (mask is None or isinstance(mask, ArrayInMemoryDataset))
            self._train_per_channel(self._rearrange_channel(data.data, False),
                                    self._rearrange_channel(labels.data, False),
                                    self._rearrange_opt_channel(mask, False),
                                    self.fit_model, ci)

    def _construct_per_channel(self, channel_names: List[str]) -> Union[Any, Dict[str, Any]]:
        if self.per_channel:
            return {cn: self.constructor() for cn in channel_names}
        return self.constructor()

    def _construct(self, dataset: data.Dataset, channel_names: List[str]) -> Union[Any, Dict[str, Any]]:
        return [self._construct_per_channel(channel_names) for _ in range(len(dataset))] \
            if self.per_data_point else self._construct_per_channel(channel_names)

    def _force_loaded_dataset(self, dataset: data.Dataset, indent: str, do_print: bool = True) -> ArrayInMemoryDataset:
        # test if both per_data_point and isinstance(dataset, ArrayInMemoryDataset) are false
        # hence: executing not per_data_point requires an ArrayInMemoryDataset
        # (which would not be given with the above condition)
        if not self.per_data_point and not isinstance(dataset, ArrayInMemoryDataset):
            if do_print:
                print(indent + 'Found non-in-memory dataset but sklearn model requires in-memory-data. Loading!')
            t = time.time()
            dataset = ArrayInMemoryDataset(dataset)
            t = time.time() - t
            if do_print:
                print(indent + f'Loading took {t:.3f}s!')
        return dataset

    def fit(self, indent: str, dataset: data.Dataset, meta: Meta, labels: data.Dataset, label_meta: Meta,
            mask: Optional[Tuple[data.Dataset, Meta]]):
        assert len(meta.per_item_info) == len(label_meta.per_item_info) == len(dataset) == len(labels), \
            'Data-dataset and label dataset must have equal size for both actual size and meta size. The following does ' \
            f'not hold: {len(meta.per_item_info)} == {len(label_meta.per_item_info)} == {len(dataset)} == {len(labels)}'
        dataset = self._force_loaded_dataset(dataset, indent)
        labels = self._force_loaded_dataset(labels, indent)
        mask_dataset, mask_meta = (None, meta) if mask is None else (
            self._force_loaded_dataset(mask[0], indent), mask[1])
        channel_names = meta.channel_names
        print(indent + f'Constructing sklearn module(s) with per_channel={self.per_channel} and '
                       f'per_data_point={self.per_data_point} for class {self.class_name}')
        t = time.time()
        self.fit_model = self._construct(dataset, channel_names)
        t = time.time() - t
        print(indent + f'Construction completed successfully for class {self.class_name} and took {t:.3f}s. '
                       f'Training on dataset with {len(dataset)} elements and {"no" if mask is None else ""} mask.')
        ci = channel_index(meta)
        t = time.time()
        self._train_per_data_point(dataset, labels, mask_dataset, ci)
        t = time.time() - t
        print(indent + f'Training completed successfully for class {self.class_name} and took {t:.3f}s - '
                       f'average {(t / len(dataset)):.3f}s.')
        if self.save_file is not None:
            run_hook = cast(RunHook, meta.run_hook)
            print(indent + f'Saving trained model to "{self.save_file}".')
            if hasattr(self.fit_model, 'save') and callable(getattr(self.fit_model, 'save')):
                to_add = self.fit_model.save(run_hook.get_artifact_file_name(self.save_file))
                for f_name in to_add:
                    run_hook.add_artifact(f_name)
            else:
                with run_hook.open_artifact_file(self.save_file, 'wb') as fd:
                    pickle.dump(self.fit_model, fd)

    def _predict_per_channel(self, data: np.ndarray, mask: Optional[np.ndarray],
                             model, new_ci: Dict[str, int]) -> np.ndarray:
        data = self._to_sample_feature_shape(data, self.image_features)
        if mask is not None:
            mask = mask.reshape(-1)
            data = data[mask]
        if self.predict_proba:
            if self.per_channel:
                predictions = [np.squeeze(actual_model.predict_proba(data[:, (new_ci[cn],)])) for cn, actual_model in
                               model.items()]
                return np.stack(predictions).transpose()
            else:
                return model.predict_proba(data).reshape((data.shape[0], -1)).transpose()
        else:
            if self.per_channel:
                predictions = [np.squeeze(actual_model.predict(data[:, (new_ci[cn],)])) for cn, actual_model in
                               model.items()]
                return np.stack(predictions).transpose()
            else:
                return model.predict(data).reshape((-1, 1)).transpose()

    def _predict_per_data_point(self, data: ArrayInMemoryDataset, mask: Optional[ArrayInMemoryDataset],
                                new_ci: Dict[str, int]) -> data.Dataset:
        if self.per_data_point:
            mask = [None] * len(data) if mask is None else mask
            combined = [self._predict_per_channel(self._rearrange_channel(data_entry, True),
                                                  self._rearrange_opt_channel(mask_entry, True),
                                                  data_model, new_ci).reshape((1,)+data_entry.shape[1:])
                        for data_entry, mask_entry, data_model in zip(data, mask, self.fit_model)]
            return ShapelessInMemoryDataset(combined)
        else:
            pred = self._predict_per_channel(self._rearrange_channel(data.data, False),
                                             self._rearrange_opt_channel(mask, False),
                                             self.fit_model,
                                             new_ci)
            if pred.shape[0] > 1:
                pred = np.concatenate(tuple(pred[i].reshape((data.data.shape[0], 1) + data.data.shape[2:])
                                            for i in range(pred.shape[0])), axis=1)
            else:
                pred = pred.reshape((data.data.shape[0], 1) + data.data.shape[2:])
            return ArrayInMemoryDataset(pred)

    def predict(self, indent: str, dataset: data.Dataset, meta: Meta,
                mask: Optional[Tuple[data.Dataset, Meta]]) -> data.Dataset:
        if not self.allow_no_fit and self.fit_model is None:
            raise RuntimeError('Cannot predict with a not yet fit model!')
        elif self.allow_no_fit and self.fit_model is None:
            self.fit_model = self._construct(dataset, meta.channel_names)
        new_ci = channel_index(meta)
        if self.predict_batch_size > 0:
            print(indent + f'Creating lazy prediction wrapper on dataset of size {len(dataset)} with batch-size = {self.predict_batch_size}.')
            assert mask is None, 'Masked predictions are not yet supported with lazy-batched-predictions...'
            return LazyBatchedPredDataset(dataset, self, self.predict_batch_size, new_ci)
        mask_dataset, mask_meta = (None, meta) if mask is None else (
            self._force_loaded_dataset(mask[0], indent), mask[1])
        dataset = self._force_loaded_dataset(dataset, indent)
        print(indent + f'Performing prediction on dataset of size {len(dataset)}.')
        t = time.time()
        res = self._predict_per_data_point(dataset, mask_dataset, new_ci)
        t = time.time() - t
        print(indent + f'Prediction took {t:.3f}s - average {(t / len(dataset)):.3f}s.')
        if self.clear_on_predict:
            del self.fit_model
            self.fit_model = None
        return res

    def __str__(self):
        return f'SupervisedSklearnAdaptor(class_name="{self.class_name}", params={str(self.params)}, ' \
               f'per_channel={self.per_channel}, per_data_point={self.per_data_point}, ' \
               f'clear_on_predict={self.clear_on_predict})'


class SupervisedSklearnAdaptorModule(MultiTransformerModule):
    def __init__(self, transformer: SupervisedSklearnAdaptor, feature_criterion: SelectionCriterion,
                 label_criterion: Optional[SelectionCriterion] = None,
                 mask_criterion: Optional[SelectionCriterion] = None, prediction_dataset_name: Optional[str] = None,
                 prediction_channel_name: Optional[str] = None, do_fit: bool = True, do_predict: bool = True,
                 delete_features: bool = False, delete_labels: bool = False, delete_mask: bool = False):
        super().__init__()
        self.transformer: SupervisedSklearnAdaptor = transformer
        self.prediction_dataset_name = prediction_dataset_name
        self.prediction_channel_name = prediction_channel_name
        self.feature_criterion = feature_criterion
        assert not (do_fit and (label_criterion is None or feature_criterion is None))
        assert not (do_predict and (feature_criterion is None or prediction_dataset_name is None or prediction_channel_name is None))
        self.label_criterion = label_criterion
        self.mask_criterion = mask_criterion
        self.do_fit = do_fit
        self.do_predict = do_predict
        self.delete_features: bool = delete_features
        self.delete_labels: bool = delete_labels
        self.delete_mask: bool = delete_mask

    def __call__(self, summary: Summary) -> Summary:
        feature_dataset, feature_meta = summary.by_criterion(self.feature_criterion, delete=self.delete_features)
        mask_info = None if self.mask_criterion is None else summary.by_criterion(self.mask_criterion,
                                                                                  delete=self.delete_mask)
        if self.do_fit:
            label_dataset, label_meta = (None, None) if self.label_criterion is None else summary.by_criterion(
                self.label_criterion, delete=self.delete_labels)
            self.print(summary, f'Performing fit on sklearn module: {str(self.transformer)}.')
            self.transformer.fit(summary.indent + '\t', feature_dataset, feature_meta, label_dataset, label_meta,
                                 mask_info)
            self.print(summary, f'Fit completed.')
        if self.do_predict:
            self.print(summary, f'Performing predict on sklearn module: {str(self.transformer)}.')
            dataset = self.transformer.predict(summary.indent + '\t', feature_dataset, feature_meta, mask_info)
            self.print(summary, f'Predict completed.')
            channel_names = [self.prediction_channel_name]
            meta = feature_meta._replace(channel_names=channel_names)
            summary.add(dataset, meta, self.prediction_dataset_name)
        return summary

class LazySupervisedSklearnAdaptorModule(MultiTransformerModule):
    def __init__(self, transformer: SupervisedSklearnAdaptor, feature_criterion: SelectionCriterion,
                 label_criterion: Optional[SelectionCriterion] = None,
                 mask_criterion: Optional[SelectionCriterion] = None, prediction_dataset_name: Optional[str] = None,
                 prediction_channel_name: Optional[str] = None,  predict_proba: bool = True,
                 delete_features: bool = False):
        super().__init__()
        self.transformer: SupervisedSklearnAdaptor = transformer
        self.prediction_dataset_name = prediction_dataset_name
        self.prediction_channel_name = prediction_channel_name
        self.feature_criterion = feature_criterion
        self.label_criterion = label_criterion
        self.mask_criterion = mask_criterion
        self.delete_features: bool = delete_features

    def __call__(self, summary: Summary) -> Summary:
        feature_dataset, feature_meta = summary.by_criterion(self.feature_criterion, delete=self.delete_features)
        mask_info = None if self.mask_criterion is None else summary.by_criterion(self.mask_criterion,
                                                                                  delete=self.delete_mask)
        if self.do_fit:
            label_dataset, label_meta = (None, None) if self.label_criterion is None else summary.by_criterion(
                self.label_criterion, delete=self.delete_labels)
            self.print(summary, f'Performing fit on sklearn module: {str(self.transformer)}.')
            self.transformer.fit(summary.indent + '\t', feature_dataset, feature_meta, label_dataset, label_meta,
                                 mask_info)
            self.print(summary, f'Fit completed.')
        if self.do_predict:
            self.print(summary, f'Performing predict on sklearn module: {str(self.transformer)}.')
            dataset = self.transformer.predict(summary.indent + '\t', feature_dataset, feature_meta, mask_info)
            self.print(summary, f'Predict completed.')
            channel_names = [self.prediction_channel_name]
            meta = feature_meta._replace(channel_names=channel_names)
            summary.add(dataset, meta, self.prediction_dataset_name)
        return summary

class SimpleCachedPipeline(MultiMetaModule):
    def __init__(self, debug_print_mismatch: bool = False):
        super().__init__()
        self.params = {}
        self.sub_module: Optional[MultiPipeline] = None
        self.cached_summary: Optional[Summary] = None
        self.debug_print_mismatch = debug_print_mismatch

    def recursive_list_mismatch_search(self, ref_list: List[Any], other_list: List[Any], key: Optional[str] = None):
        if len(ref_list) != len(other_list):
            print(f'Lists for {key} have differing length => {ref_list} != {other_list}')
        else:
            ref_set = set([(tuple(v.keys()) if isinstance(v, dict) else v) for v in ref_list])
            other_set = set([(tuple(v.keys()) if isinstance(v, dict) else v) for v in other_list])
            dif = (ref_set.union(other_set)).difference(ref_set.intersection(other_set))
            if dif:
                print(f'Lists for {key} have differing contents => {ref_list} != {other_list}')
            else:
                print(f'Lists for {key} seem to have a differing order => {ref_list} != {other_list}')
        for i, ref1, ref2 in zip(range(min(len(ref_list), len(other_list))), ref_list, other_list):
            if type(ref1) != type(ref2):
                print(f'Types ref={type(ref1)} and new={type(ref2)} are not equal for index {i}!')
            elif ref1 != ref2:
                if isinstance(ref1, dict):
                    print(f'Entering Dict (index {i}), as {ref1} != {ref2}')
                    self.recursive_mismatch_search(ref1, ref2)
                elif isinstance(ref1, list):
                    print(f'Entering List (index {i}), as {ref1} != {ref2}')
                    self.recursive_list_mismatch_search(ref1, ref2, key)
                else:
                    print(f'Found mismatch at index {i}! {ref1} != {ref2}')


    def recursive_mismatch_search(self, ref_params: Dict[str, Any], other_params: Dict[str, Any]):
        ref_keys: Set[str] = set(ref_params.keys())
        other_keys: Set[str] = set(other_params.keys())
        key_intersection = ref_keys.intersection(other_keys)
        key_union = ref_keys.union(other_keys)
        dif = key_union.difference(key_intersection)
        if dif:
            print(f'The following keys are not contained in both param dicts: {dif}')
        for key in key_intersection:
            if type(ref_params[key]) != type(other_params[key]):
                print(f'Types ref={type(ref_params[key])} and new={type(other_params[key])} are not equal for key {key}')
            elif ref_params[key] != other_params[key]:
                if isinstance(ref_params[key], dict):
                    print(f'Entering Dict (key {key}), as {ref_params[key]} != {other_params[key]}')
                    self.recursive_mismatch_search(ref_params[key], other_params[key])
                elif isinstance(ref_params[key], list):
                    print(f'Entering List (key {key}), as {ref_params[key]} != {other_params[key]}')
                    self.recursive_list_mismatch_search(ref_params[key], other_params[key], key)
                else:
                    print(f'Found mismatch at {key}! {ref_params[key]} != {other_params[key]}')

    def set_module(self, sub_module: MultiPipeline):
        sub_params = serialisation.serialize_dict(sub_module)
        if self.sub_module is None or self.cached_summary is None or sub_params != self.params:
            if self.sub_module is not None and self.debug_print_mismatch:
                self.recursive_mismatch_search(self.params, sub_params)
            self.sub_module = sub_module
            self.cached_summary = None
            self.params = sub_params

    def get_params_as_dict(self) -> Dict[str, Any]:
        return {'sub_module': self.sub_module} #, 'executes_cached': self.cached_summary is not None

    def __call__(self, summary: Summary) -> Summary:
        initial_indent = summary.indent
        if self.sub_module is None:
            raise RuntimeError('Cannot execute cached pipeline if the delegation module has not been set yet!')
        elif summary:
            raise RuntimeError('Cannot use simple cached pipeline with a summary that actually contains data!!!')
        if self.cached_summary is None:
            self.print(summary, f'Initialising Cache.')
            t = time.time()
            self.cached_summary = self.sub_module(summary.do_indent()).undo_indent()
            t = time.time() - t
            self.print(self.cached_summary, f'Initialisation completed in {t:.3f}s.')
        else:
            self.print(self.cached_summary, f'Utilising cached results.')
        self.cached_summary.exe_start = summary.exe_start
        self.cached_summary.set_hook(summary.run_hook)
        self.cached_summary['indent'] = initial_indent
        return self.cached_summary.copy()

class NoCachePipeline(SimpleCachedPipeline):
    def __call__(self, summary: Summary) -> Summary:
        initial_indent = summary.indent
        t = time.time()
        self.cached_summary = self.sub_module(summary.copy().do_indent()).undo_indent()
        t = time.time() - t
        self.cached_summary.exe_start = summary.exe_start
        self.cached_summary.set_hook(summary.run_hook)
        self.cached_summary['indent'] = initial_indent
        return self.cached_summary.copy()


class MaskDataset(TransformerDataset):
    def __init__(self, wrapped: data.Dataset, mask_label: int):
        super().__init__(wrapped)
        self.mask_label = mask_label

    def _transform(self, data):
        return data != self.mask_label


class MaskModule(MultiTransformerModule):
    def __init__(self, source_criterion: SelectionCriterion,
                 res_name: str,
                 mask_label: int,
                 res_mask_channel_name: str = 'mask'):
        super(MaskModule, self).__init__()
        self.source_criterion = source_criterion
        self.res_name = res_name
        self.mask_label = mask_label
        self.res_mask_channel_name = res_mask_channel_name

    def __call__(self, summary: Summary) -> Summary:
        source_data, source_meta = summary.by_criterion(self.source_criterion)
        if isinstance(source_data, ArrayInMemoryDataset):
            res_data = ArrayInMemoryDataset(source_data.data != self.mask_label)
        else:
            res_data = MaskDataset(source_data, self.mask_label)
        res_meta = source_meta._replace(channel_names=[self.res_mask_channel_name])
        return summary.add(res_data, res_meta, self.res_name)

class MultiTerminationModule(MultiTransformerModule):
    def __call__(self, summary: Summary) -> Summary:
        return None

class GeneralisedSklearnAdaptor(JSONSerializable):
    def __init__(self,
                 clazz: Union[str, Tuple[str, Callable]],
                 params: Optional[Dict[str, Any]] = None,
                 per_data_point: bool = False,
                 image_features: bool = False,
                 clear_on_predict: bool = False,
                 allow_no_fit: bool = False,
                 init_file: Optional[str] = None,
                 save_file: Optional[str] = None):
        class_name, cons_fun = (clazz, utils.get_class_constructor(clazz)) if isinstance(clazz, str) else clazz
        self.class_name = class_name
        self.cons_fun = cons_fun
        self.constructor = (lambda: cons_fun(**params)) if params is not None else cons_fun
        self.per_data_point = per_data_point
        self.image_features = image_features
        self.fit_model = None
        self.allow_no_fit = allow_no_fit
        self.clear_on_predict: bool = clear_on_predict
        self.params = params
        if init_file is not None and (not path.exists(init_file) or not path.isfile(init_file)):
            print(
                f'WARNING: Init-file "{init_file}" provided for adaptor of class {clazz} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            init_file = None
        self.init_file = init_file
        if self.init_file is not None:
            print(f'Valid init file found at {init_file}. Loading.')
            with open(init_file, 'rb') as fd:
                self.fit_model = pickle.load(fd)
        if init_file is not None and (not path.exists(save_file) or not path.isfile(save_file)):
            print(
                f'WARNING: Save-file "{save_file}" provided for adaptor of class {clazz} does not exist or is not a file! Ignoring!',
                file=sys.stderr)
            save_file = None
        self.save_file = save_file

    def get_params_as_dict(self) -> Dict[str, Any]:
        return {
            'class_name': self.class_name,
            'params': self.constructor().get_params(deep=True),
            'per_data_point': self.per_data_point,
            'image_features': self.image_features,
            'allow_no_fit': self.allow_no_fit,
            'clear_on_predict': self.clear_on_predict,
            'init_file': self.init_file,
            'save_file': self.save_file
        }

    @staticmethod
    def _rearrange_channel(data: np.ndarray, per_data_point: bool) -> np.ndarray:
        # swaps the first two dimensions, so that the channel is now the first one...
        if per_data_point:
            return data
        return np.transpose(data, (1, 0) + tuple(range(2, len(data.shape))))

    @staticmethod
    def _rearrange_opt_channel(data: Optional[Union[ArrayInMemoryDataset, np.ndarray]], per_data_point: bool) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, ArrayInMemoryDataset):
            data = data.data
        return SupervisedSklearnAdaptor._rearrange_channel(data, per_data_point)

    @staticmethod
    def _to_sample_feature_shape(data: np.ndarray, image_features: bool) -> np.ndarray:
        # remember, we assume that the last two dims are the image dimensions and that the first one is the channel dim
        if image_features and data.ndim >= 3:
            if data.ndim == 3:
                # if we only have 3 dimensions, then this is simple...
                return np.transpose(data, (1, 2, 0))
            else:
                # if we have to account for more dimensions - it get's a bit trickier
                # assumption: (C, something..., H, W)
                # first ensure that we have (reverse something..., W, H, C )
                data = np.transpose(data, tuple(range(data.ndim - 3, 0, -1)) + (0,))
                # then swap W and H
                data = np.swapaxes(data, axis1=-2, axis2=-3)
                # and now return with all other dimensions being flattened
                return data.reshape((-1,) + data.shape[-3:])
        # First reshape (C, ..., H, W) into (C, ... * H * W)
        # Then transpose into (... * H * W, C), aka the standard sklearn format
        return data.reshape((data.shape[0], -1)).transpose()

    def _do_train(self, data: Dict[str, Union[np.ndarray, torch.Tensor]], model: Any):
        model.fit(**data)

    def _train_per_data_point(self, loaded_data: Dict[str, Tuple[ShapelessInMemoryDataset, Meta]]):
        if self.per_data_point:
            per_data_entries = [{k: dataset[i] for k, dataset in loaded_data.items()}
                                for i in range(len(next(iter(loaded_data))[0]))]
            for data_dict, model in zip(per_data_entries, self.fit_model):
                self._do_train(data_dict, model)
        else:
            assert not any(map(lambda t: not isinstance(t, ArrayInMemoryDataset), loaded_data.values())), \
                'For non per-data training to work, all datasets must be loaded into memory!!!'
            self._do_train({k: dataset.data for k, (dataset, _) in loaded_data.items()}, self.fit_model)


    def _construct(self, size: int) -> Union[Any, Dict[str, Any]]:
        return [self.constructor() for _ in range(size)] \
            if self.per_data_point else self.constructor()

    def _force_loaded_dataset(self, dataset: data.Dataset, indent: str) -> ArrayInMemoryDataset:
        # test if both per_data_point and isinstance(dataset, ArrayInMemoryDataset) are false
        # hence: executing not per_data_point requires an ArrayInMemoryDataset
        # (which would not be given with the above condition)
        if not self.per_data_point and not isinstance(dataset, ArrayInMemoryDataset):
            print(indent + 'Found non-in-memory dataset but sklearn model requires in-memory-data. Loading!')
            t = time.time()
            dataset = ArrayInMemoryDataset(dataset)
            t = time.time() - t
            print(indent + f'Loading took {t:.3f}s!')
        return dataset

    def fit(self, indent: str, extracted: Dict[str, Tuple[data.Dataset, Meta]]):
        meta = next(iter(extracted.values()))[1]
        loaded_data: Dict[str, Tuple[ShapelessInMemoryDataset, Meta]] = \
            {key: (self._force_loaded_dataset(dataset, indent), meta)
             for key, (dataset, meta) in extracted.items()}
        print(indent + f'Constructing sklearn module(s) with '
                       f'per_data_point={self.per_data_point} for class {self.class_name}')
        if self.per_data_point:
            first_len = len(next(iter(loaded_data.values()))[0])
            assert not any(map(lambda t: len(t[0]) != first_len, loaded_data.values())), \
                'For per-data training to work, all datasets must have identical size!!!'
        else:
            first_len = -1
        t = time.time()
        self.fit_model = self._construct(first_len)
        t = time.time() - t
        print(indent + f'Construction completed successfully for class {self.class_name} and took {t:.3f}s. '
                       f'Training on dataset with {first_len} elements.')
        t = time.time()
        self._train_per_data_point(loaded_data)
        t = time.time() - t
        print(indent + f'Training completed successfully for class {self.class_name} and took {t:.3f}s.')
        if self.save_file is not None:
            print(indent + f'Saving trained model to "{self.save_file}".')
            with meta.run_hook.open_artifact_file(self.save_file, 'wb') as fd:
                pickle.dump(self.fit_model, fd)

    def _do_predict(self, loaded_data: Dict[str, Union[np.ndarray, torch.Tensor]], model) -> np.ndarray:
        loaded_data = {k: self._to_sample_feature_shape(data, self.image_features) for k, data in loaded_data.items()}
        return model.predict(**loaded_data).reshape((-1, 1)).transpose()

    def _predict_per_data_point(self, loaded_data: Dict[str, Tuple[ShapelessInMemoryDataset, Meta]]) -> data.Dataset:
        if self.per_data_point:
            first_len = len(next(iter(loaded_data.values()))[0])
            assert not any(map(lambda t: len(t[0]) != first_len, loaded_data.values())), \
                'For per-data training to work, all datasets must have identical size!!!'
            per_data_entries = [{k: dataset[i] for k, (dataset, meta) in loaded_data.items()}
                                for i in range(first_len)]
            combined = [self._do_predict(data_entry_map, model)
                        for data_entry_map, model in zip(per_data_entries, self.fit_model)]
            return ShapelessInMemoryDataset(combined)
        else:
            assert not any(map(lambda t: not isinstance(t, ArrayInMemoryDataset), loaded_data.values())), \
                'For non per-data training to work, all datasets must be loaded into memory!!!'
            pred = self._do_predict({k: dataset.data for k, (dataset, _) in loaded_data.items()}, self.fit_model)
            # it is unclear what the correct shape would be for this result
            # let's just keep it and hope it matches, copy of supervised adapter would do
            # pred.reshape((data.data.shape[0], 1, data.data.shape[2], data.data.shape[3]))
            return ArrayInMemoryDataset(pred)

    def predict(self, indent: str, extracted: Dict[str, Tuple[data.Dataset, Meta]]) -> data.Dataset:
        meta = next(iter(extracted.values()))[1]
        if not self.allow_no_fit and self.fit_model is None:
            raise RuntimeError('Cannot predict with a not yet fit model!')
        elif self.allow_no_fit and self.fit_model is None:
            self.fit_model = self._construct(len(meta.per_item_info))
        loaded_data: Dict[str, Tuple[ShapelessInMemoryDataset, Meta]] = \
            {key: (self._force_loaded_dataset(dataset, indent), meta)
             for key, (dataset, meta) in extracted.items()}
        print(indent + f'Performing prediction on dataset of size {len(meta.per_item_info)}.')
        t = time.time()
        res = self._predict_per_data_point(loaded_data)
        t = time.time() - t
        print(indent + f'Prediction took {t:.3f}s.')
        if self.clear_on_predict:
            del self.fit_model
            self.fit_model = None
        return res

    def __str__(self):
        return f'GeneralisedSklearnAdaptor(class_name="{self.class_name}", params={str(self.params)}, ' \
               f'per_data_point={self.per_data_point}, ' \
               f'clear_on_predict={self.clear_on_predict})'

@dataclass(frozen=True)
class DataInfo(JSONSerializable):
    criterion: SelectionCriterion
    param_name: str
    delete: bool = False

    def get_params_as_dict(self) -> Dict[str, Any]:
        return {'criterion': self.criterion, 'param_name': self.param_name, 'delete': self.delete}


# Generalises sklearn adaptor to accept more than one input (for example dbscan sample_weight)
# however dataset should already have the shape [N, C] and masking is not performed out of the box
# Furthermore channelwise predictions are not supported
# Also this should really be extended to support multiple outputs
class GeneralisedSklearnAdaptorModule(MultiTransformerModule):
    def __init__(self, transformer: GeneralisedSklearnAdaptor,
                 criteria_map: List[DataInfo],
                 prediction_dataset_name: Optional[str] = None,
                 prediction_channel_name: Optional[str] = None, do_fit: bool = True, do_predict: bool = True,
                 ):
        super().__init__()
        assert len(set(d_info.param_name for d_info in criteria_map)) == len(criteria_map), 'Param names must be unique, as they will be used as keys later!'
        self.transformer: GeneralisedSklearnAdaptor = transformer
        self.prediction_dataset_name = prediction_dataset_name
        self.prediction_channel_name = prediction_channel_name
        self.criteria_map = criteria_map.copy()
        self.do_fit = do_fit
        self.do_predict = do_predict

    def __call__(self, summary: Summary) -> Summary:
        extracted = {d_info.param_name: summary.by_criterion(d_info.criterion, delete=d_info.delete)
                     for d_info in self.criteria_map}
        if self.do_fit:
            self.print(summary, f'Performing fit on sklearn module: {str(self.transformer)} with params {extracted.keys()}.')
            self.transformer.fit(summary.indent + '\t', extracted)
            self.print(summary, f'Fit completed.')
        if self.do_predict:
            assert len(extracted) >= 1, 'Prediction with no-input is not supported, as a meta is required!!!'
            self.print(summary, f'Performing predict on sklearn module: {str(self.transformer)} with params {extracted.keys()}.')
            dataset = self.transformer.predict(summary.indent + '\t', extracted)
            self.print(summary, f'Predict completed.')
            channel_names = [self.prediction_channel_name]
            meta = next(iter(extracted.values()))[1]._replace(channel_names=channel_names)
            summary.add(dataset, meta, self.prediction_dataset_name)
        return summary

class RetainInSummaryModule(MultiPipeline):
    def __init__(self, to_retain: SelectionCriterion):
        super().__init__()
        self.to_retain = to_retain

    def __call__(self, summary: Summary) -> Summary:
        n, (d, m) = summary.by_criterion(self.to_retain, return_name=True)
        summary.data = {}
        summary.by_index = []
        summary.add(d, m, n)
        return summary

class RemoveFromSummaryModule(MultiPipeline):
    def __init__(self, to_remove: SelectionCriterion):
        super().__init__()
        self.to_retain = to_remove

    def __call__(self, summary: Summary) -> Summary:
        _ = summary.by_criterion(self.to_retain, delete=True)
        #summary.data = {}
        #summary.by_index = []
        #summary.add(d, m, n)
        return summary

class SaveToDatadings(MultiPipeline):

    def __init__(self, dataset_criterion: SelectionCriterion,
                 # WARNING: due to the run-hook being unable to detect the files written by datadings easily
                 # this will be the actual output path and not use the run-dir provided via runhook!!!
                 output_file: str,
                 delete: bool = False):
        super().__init__()
        self.dataset_criterion = dataset_criterion
        self.output_file_name = output_file
        self.delete = delete

    def __call__(self, summary: Summary) -> Summary:
        actual_file_name = path.abspath(self.output_file_name)
        dataset, meta = summary.by_criterion(self.dataset_criterion, self.delete)
        try:
            with datadings.writer.FileWriter(actual_file_name, total=len(dataset), overwrite=False) as data_writer:
                for data_point, per_item_meta in zip(dataset, meta.per_item_info):
                    data_writer.write({'image': data_point,
                                       'region': per_item_meta.region,
                                       'key': str(per_item_meta.id)})
        except:
            print(traceback.format_exc(), file=sys.stderr)
        return summary

class CombineDataset(TransformerDataset):

    def __init__(self, wrapped: data.Dataset, source: data.Dataset, mask_index: int = -1):
        super().__init__(wrapped)
        self.source = source
        self.mask_index = mask_index

    def __getitem__(self, item):
        target_data = self.wrapped[item].copy()
        source_data = self.source[item]
        mask = source_data == -1
        if source_data.shape[0] == 1 and target_data.shape[0] > 1:
            target_data = target_data
            mask = mask.reshape(source_data.shape[1:])
            for i in range(target_data.shape[0]):
                target_data[i, mask] = -1
        else:
            target_data[mask] = -1
        return target_data


class MaskCombine(MultiPipeline):
    def __init__(self, target_criterion: SelectionCriterion,
                 source_criterion: SelectionCriterion,
                 # WARNING: due to the run-hook being unable to detect the files written by datadings easily
                 # this will be the actual output path and not use the run-dir provided via runhook!!!
                 dataset_name: Optional[str] = None,
                 delete_target: bool = False,
                 delete_source: bool = True,
                 mask_index: int = -1):
        super().__init__()
        self.target_criterion = target_criterion
        self.source_criterion = source_criterion
        self.dataset_name = dataset_name
        self.delete_target = delete_target
        self.delete_source = delete_source
        self.mask_index = mask_index

    def __call__(self, summary: Summary) -> Summary:
        dataset, meta = summary.by_criterion(self.target_criterion, self.delete_target)
        source_dataset, source_meta = summary.by_criterion(self.source_criterion, self.delete_source)
        assert len(meta.per_item_info) == len(source_meta.per_item_info), \
            'Expected source and target dataset to be of equal size'
        summary.add(CombineDataset(dataset, source_dataset, self.mask_index), meta, name=self.dataset_name)
        return summary