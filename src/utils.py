import argparse
import importlib
import itertools as itert
import math
import os
import os.path as path
import shutil
import sys
import tarfile
import time
from typing import Union, Tuple, Optional, Callable, Iterable, Mapping, List, Iterator, Generic, TypeVar, Dict, Any

import numba
import numpy as np
import torch

def get_module_fun(module: str, fun: str) -> Callable:
    mod_name = module.rsplit('.', maxsplit=1)[1]
    ex = importlib.import_module(f'..{mod_name}', module)
    ex_fun = getattr(ex, fun)
    assert callable(ex_fun)
    return ex_fun

EPS = math.sqrt(np.finfo(float).eps)
ENABLE_PARALLEL = int(os.environ.get('ENABLE_NUMBA_PARALLEL', 1)) > 0
def njit(*args, **kwargs) -> Callable:
    if not ENABLE_PARALLEL and 'parallel' in kwargs:
        kwargs['parallel'] = False
    return numba.njit(*args, **kwargs)

def jit(*args, **kwargs) -> Callable:
    if not ENABLE_PARALLEL and 'parallel' in kwargs:
        kwargs['parallel'] = False
    return numba.jit(*args, **kwargs)


def recursive_simplify(val: Any):
    if isinstance(val, dict):
        return {key: recursive_simplify(v) for key, v in val.items()}
    elif isinstance(val, list) or isinstance(val, tuple):
        return [recursive_simplify(v) for v in val]
    elif hasattr(val, 'tolist'):
        return val.tolist()
    else:
        return val


def torch_as_np(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def np_as_torch(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if tensor is None:
        return None
    return tensor if isinstance(tensor, torch.Tensor) else torch.from_numpy(tensor)

def revert_dims(data: Union[torch.Tensor, np.ndarray], prefix: Optional[Tuple[int, ...]] = None,
                postfix: Optional[Tuple[int, ...]] = None) -> Union[torch.Tensor, np.ndarray]:
    perm_seq = tuple(range(len(data.shape) - 1, -1, -1))
    res = torch.permute(data, perm_seq) if isinstance(data, torch.Tensor) else np.transpose(data, perm_seq)
    shape = res.shape
    if prefix is not None:
        shape = prefix + shape
    if postfix is not None:
        shape = shape + postfix
    return res.reshape(shape)


def match_type(to_match: Union[torch.Tensor, np.ndarray], other: Union[torch.Tensor, np.ndarray]) \
        -> Tuple[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]], bool]:
    is_target_torch = isinstance(to_match, torch.Tensor)
    is_torch = isinstance(other, torch.Tensor)
    if is_target_torch and not is_torch:
        other = torch.from_numpy(other)
    elif not is_target_torch and is_torch:
        other = other.detach().cpu().numpy()
    return (to_match, other), is_torch


def move_channel_to_position(data: Union[torch.Tensor, np.ndarray], target_position: int = 0,
                             source_position: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
    """
    Moves a channel to index target_position from source_position and leaves the remainder untouched.
    :param data:
    :param target_position:
    :param source_position:
    :return:
    """
    if source_position == None:
        if target_position >= data.ndim - 1:
            return data
        source_position = data.ndim - 1
    perm_seq = tuple(i for i in range(target_position) if i != source_position) + (source_position,) + tuple(
        i for i in range(target_position, len(data.shape)) if i != source_position)
    return torch.permute(data, perm_seq) if isinstance(data, torch.Tensor) else np.transpose(data, perm_seq)


def property_function(stat: str, axis=1) -> Callable:
    if stat == 'majority':
        def majority(rm, intensity):
            if intensity.dtype == np.float64 or intensity.dtype == np.float32:
                return 0
            masked: np.ndarray = intensity[rm]
            if masked.size < 1:
                print('No elements in mask!!!', file=sys.stderr)
                return 0
            unique_values, unique_counts = np.unique(masked, return_counts=True)
            return unique_values[np.argmax(unique_counts)]

        return majority
    elif stat == 'var' or stat == 'variance' or stat == 'variance_intensity':
        def variance(rm, intensity):
            return np.nanvar(intensity[rm].T, axis=axis)

        return variance
    elif stat == 'std' or stat == 'standard_deviation' or stat == 'standard_deviation_intensity':
        def standard_deviation(rm, intensity):
            return np.nanstd(intensity[rm].T, axis=axis)

        return standard_deviation
    else:
        raise ValueError('Unknown Extraproperty ' + stat)


_EXTRA_STATS = {'majority', 'var', 'variance', 'variance_intensity', 'std', 'standard_deviation',
                'standard_deviation_intensity'}

ONE_PER_CHANNEL_STATISTICS = {'mean_intensity', 'var', 'variance', 'variance_intensity', 'std', 'standard_deviation',
                              'standard_deviation_intensity'}
ONE_CHANNEL_STATISTICS = {'label'}


def create_channel_calculator(stats_of_interest: Iterable[str]) -> Callable:
    stats_of_interest = set(stats_of_interest)
    per_channel_stats = sum(map(lambda p: 1, stats_of_interest.intersection(ONE_PER_CHANNEL_STATISTICS)))
    single_result_stats = sum(map(lambda p: 1, stats_of_interest.intersection(ONE_CHANNEL_STATISTICS)))
    return lambda c: c * per_channel_stats + single_result_stats


# A fast shift function: copied from https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
# shifts along the first axis
@njit()
def shift_1(xs: np.ndarray, shift_amount: int, fill):
    e = np.empty_like(xs)
    if shift_amount >= 0:
        e[:shift_amount] = fill
        e[shift_amount:] = xs[:-shift_amount]
    else:
        e[shift_amount:] = fill
        e[:shift_amount] = xs[-shift_amount:]
    return e


# adapted from shift_1 to shift along the second axis
@njit()
def shift_2(xs: np.ndarray, shift_amount: int, fill):
    e = np.empty_like(xs)
    if shift_amount >= 0:
        e[:, :shift_amount] = fill
        e[:, shift_amount:] = xs[:, :-shift_amount]
    else:
        e[:, shift_amount:] = fill
        e[:, :shift_amount] = xs[:, -shift_amount:]
    return e


def extract_bbs(props, sort: bool = False) -> np.ndarray:
    res = np.array([prop.bbox for prop in props]).reshape((-1, 2, 2))
    if sort:
        res = res[np.argsort(np.array([prop.label for prop in props]))]
    return res


@njit()
def index_by_bounding_box(ar: np.ndarray, bb: np.ndarray):
    return ar[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[1, 1]]


def index_by_bounding_box_nonumba(ar: np.ndarray, bb: np.ndarray) -> np.ndarray:
    return ar[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[1, 1]]


def align_bounding_box(to_align: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """
    Adjusts the boundaries of to_align to be relative to bb
    :param to_align: assumed to be contained within bb
    :param bb:
    :return:
    """
    to_align = to_align.copy()
    to_align[:, 0] -= bb[0, 0]
    to_align[:, 1] -= bb[0, 1]
    return to_align


@njit()
def assign_in_bounding_box(to_assign: np.ndarray, source: Union[np.ndarray, int, float], bb: np.ndarray):
    to_assign[bb[0, 0]:bb[1, 0], bb[0, 1]:bb[1, 1]] = source


# @numba.jit(nopython=True)
def bbs_union(bbs: Iterable[np.ndarray]) -> np.ndarray:
    """
    Union of boundingboxes
    :param bbs: Bbs should have the form [[min_x, min_y], [max_x, max_y]]. Though results will also be correct
    if x and y are swapped
    :return:
    """
    stacked = np.stack(tuple(bbs))
    return np.stack((np.min(stacked[:, 0], axis=0), np.max(stacked[:, 1], axis=0)))


def bbs_intersection(*bbs) -> np.ndarray:
    stacked = np.stack(bbs)
    return np.stack((np.max(stacked[:, 0], axis=0), np.min(stacked[:, 1], axis=0)))


@njit(numba.float32[:, ::1](numba.float32[:, :, :, ::1]))
def trace4d(ar: np.ndarray) -> np.ndarray:
    res = np.empty(ar.shape[2:], dtype=np.float32)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.trace(ar[:, :, i, j])
    return res


@njit(numba.types.UniTuple(numba.float32[::1], 3)(numba.float32[::1], numba.float32[::1], numba.float32[::1]),
            fastmath=False,
            parallel=True)  # )
def rgb_to_hsv(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hue = np.empty_like(red)
    saturation = np.empty_like(red)
    value = np.empty_like(red)
    for i in numba.prange(red.shape[0]):  # range(red.shape[0]):
        value[i] = max(red[i], green[i], blue[i])
        min_val = min(red[i], green[i], blue[i])
        if min_val == value[i]:
            saturation[i] = 0.0
            hue[i] = 0.0
            continue
        saturation[i] = value[i] - min_val
        if value[i] == red[i]:
            hue[i] = 60.0 * (green[i] - blue[i]) / saturation[i]
            # np.fmod(360.0 + 60.0 * (green[i] - blue[i]) / saturation[i], 360.0)
            if hue[i] >= 360.0:
                hue[i] -= 360.0
            elif hue[i] < 0:
                hue[i] += 360.0
        elif value[i] == green[i]:
            hue[i] = 120.0 + 60.0 * (blue[i] - red[i]) / saturation[i]
        elif value[i] == blue[i]:
            hue[i] = 240.0 + 60.0 * (red[i] - green[i]) / saturation[i]
        else:  # should never happen
            hue[i] = math.nan
        hue[i] = hue[i] / 360.0
        saturation[i] /= value[i]
    return hue, saturation, value


def rgb_to_hsv_np(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.stack((red, green, blue))
    max_indices = np.argmax(combined, axis=0)
    value = np.take_along_axis(combined, np.expand_dims(max_indices, axis=0), axis=0).flatten()
    min_indices = np.argmin(combined, axis=0)
    saturation = value - np.take_along_axis(combined, np.expand_dims(min_indices, axis=0), axis=0).flatten()
    hue = np.empty_like(red)
    eq_mask = max_indices == min_indices
    hue[eq_mask] = 0
    saturation[eq_mask] = 0
    eq_mask = np.logical_not(eq_mask)
    zero_mask, one_mask, two_mask = [np.logical_and(eq_mask, max_indices == i) for i in range(3)]
    hue[zero_mask] = np.fmod(360.0 + 60.0 * (green[zero_mask] - blue[zero_mask]) / saturation[zero_mask], 360.0)
    hue[one_mask] = 120.0 + 60.0 * (blue[one_mask] - red[one_mask]) / saturation[one_mask]
    hue[two_mask] = 240.0 + 60.0 * (red[two_mask] - green[two_mask]) / saturation[two_mask]
    saturation[eq_mask] /= value[eq_mask]
    hue /= 360.0
    return hue, saturation, value


def rgb_to_hsv_torch(red: torch.Tensor, green: torch.Tensor, blue: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    combined = torch.stack((red, green, blue))
    value, max_indices = torch.max(combined, dim=0)
    mins, min_indices = torch.min(combined, dim=0)
    saturation = value - mins
    hue = torch.empty_like(red)
    eq_mask = max_indices == min_indices
    hue[eq_mask] = 0
    saturation[eq_mask] = 0
    eq_mask = torch.logical_not(eq_mask)
    zero_mask, one_mask, two_mask = [torch.logical_and(eq_mask, max_indices == i) for i in range(3)]
    hue[zero_mask] = torch.fmod(360.0 + 60.0 * (green[zero_mask] - blue[zero_mask]) / saturation[zero_mask], 360.0)
    hue[one_mask] = 120.0 + 60.0 * (blue[one_mask] - red[one_mask]) / saturation[one_mask]
    hue[two_mask] = 240.0 + 60.0 * (red[two_mask] - green[two_mask]) / saturation[two_mask]
    hue /= 360.0
    saturation[eq_mask] /= value[eq_mask]
    return hue, saturation, value

_SPECIAl_CASES = {'torch.optim.adamw.AdamW': torch.optim.AdamW}
def get_class_constructor(class_name: str) -> Callable:
    if class_name in _SPECIAl_CASES:
        return _SPECIAl_CASES[class_name]
    split = class_name.split('.')
    # copied from https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module
    if len(split) > 1:
        module = __import__(split[0])
    else:
        print('WARNING: No module specification found!!! Interpreting this as being parts of the utils module, '
              'which is almost certainly not what you want!', file=sys.stderr)
        module = sys.modules[__name__]

    def recursive_constructor_search(cur_module, remaining_split, is_exc: bool = False) -> Callable:
        try:
            next_res = getattr(cur_module, remaining_split[0])
            if len(remaining_split) > 1:
                return recursive_constructor_search(next_res, remaining_split[1:])
            return next_res
        except Exception as e:
            if is_exc:
                raise e
            __import__('.'.join([mod for mod in split if mod not in remaining_split]) + '.' + remaining_split[0])
            return recursive_constructor_search(cur_module, remaining_split, True)

    return recursive_constructor_search(module, split[1:])


@njit()
def _find_first_index(item: int, ar: np.ndarray) -> int:
    ar = ar.flatten()
    for i, e in enumerate(ar):
        if e == item:
            return i
    return -1

def find_first_index(item: int, ar: Union[np.ndarray, torch.Tensor]) -> int:
    if isinstance(ar, torch.Tensor):
        return torch.argmax((ar == item).long()).item()
    else:
        return _find_first_index(item, ar)


_WRAPPED = TypeVar('_WRAPPED')


class IdentityWrapper(Generic[_WRAPPED]):
    def __init__(self, obj: _WRAPPED):
        self.obj: _WRAPPED = obj

    def __hash__(self):
        return hash(id(self.obj))

    def __eq__(self, other):
        return type(other) is IdentityWrapper and id(other.obj) == id(self.obj)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({str(self.obj)})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.obj)})'


class IdentityOrHashWrapper(IdentityWrapper[_WRAPPED]):
    def __init__(self, obj: _WRAPPED):
        super().__init__(obj)
        try:
            hash_code = hash(obj)
            equality = obj == obj
            self.eq_based = True
        except:
            self.eq_based = False

    def __hash__(self):
        if self.eq_based:
            return hash(self.obj)
        return super().__hash__()

    def __eq__(self, other):
        if self.eq_based and other.eq_based:
            return self.obj == other.obj
        return super().__eq__(other)


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class IdentityDict(Mapping[_KT, _VT]):
    @staticmethod
    def from_iterable(iterable: List[Tuple[_KT, _VT]], use_identity_only: bool = False) -> 'IdentityDict':
        new_obj = IdentityDict()
        cons = IdentityWrapper if use_identity_only else IdentityOrHashWrapper
        new_obj.delegate = {cons(key): value for key, value in iterable}
        return new_obj

    def __init__(self, use_identity_only: bool = False) -> None:
        super().__init__()
        self.delegate: Dict[IdentityWrapper[_KT], _VT] = {}
        self.wrapper_cons = IdentityWrapper if use_identity_only else IdentityOrHashWrapper

    def clear(self) -> None:
        self.delegate.clear()

    def copy(self) -> 'IdentityDict[_KT, _VT]':
        new = IdentityDict()
        new.delegate = self.delegate.copy()
        return new

    def keys(self) -> Iterator[_KT]:
        for key in self.delegate.keys():
            yield key.obj

    def values(self) -> Iterator[_KT]:
        for val in self.delegate.values():
            yield val

    def items(self) -> Iterator[Tuple[_KT, _VT]]:
        for key, val in self.delegate.items():
            return key.obj, val

    def __len__(self) -> int:
        return len(self.delegate)

    def __getitem__(self, k: _KT) -> _VT:
        return self.delegate[self.wrapper_cons(k)]

    def __setitem__(self, k: _KT, v: _VT) -> None:
        self.delegate[self.wrapper_cons(k)] = v

    def __delitem__(self, v: _KT) -> None:
        del self.delegate[self.wrapper_cons(v)]

    def __iter__(self) -> Iterator[_KT]:
        return self.keys()

    def __reversed__(self) -> Iterator[_KT]:
        for key in self.delegate.__reversed__():
            yield key.obj

    def __str__(self) -> str:
        return str({str(key.obj): val for key, val in self.delegate})


class DefaultedIdentityDict(IdentityDict[_KT, _VT]):
    def __init__(self, default_factory: Callable, set_default_on_query: bool = True, use_identity_only: bool = False) \
            -> None:
        super().__init__(use_identity_only)
        self.default_factory = default_factory
        self.set_default_on_query = set_default_on_query

    def clear(self) -> None:
        self.delegate.clear()

    def copy(self) -> 'IdentityDict[_KT, _VT]':
        new = DefaultedIdentityDict(self.default_factory(), self.set_default_on_query)
        new.delegate = self.delegate.copy()
        return new

    def __getitem__(self, k: _KT) -> _VT:
        wrapper = self.wrapper_cons(k)
        if wrapper not in self.delegate:
            def_value = self.default_factory()
            if self.set_default_on_query:
                self[k] = def_value
            return def_value
        return self.delegate[wrapper]


# adapted from https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        res_dict = {}
        for k, v in map(lambda kv: kv.split('=', maxsplit=1), values):
            key = (k if not ',' in k else tuple(k.split(',')))
            if isinstance(v, int) or v.isdigit():
                value = int(v)
            else:
                try:
                    value = float(v)
                except:
                    if v.casefold() == 'null'.casefold() or v.casefold() == 'none'.casefold():
                        value = None
                    elif v.casefold() == 'nan'.casefold():
                        value = math.nan
                    elif v.casefold() == 'true'.casefold():
                        value = True
                    elif v.casefold() == 'false'.casefold():
                        value = False
                    else:
                        value = v
            res_dict[key] = value
        setattr(namespace, self.dest, res_dict)

class StoreMultipleDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreMultipleDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        res_dict = [{k: (int(v) if v.isdigit() else v) for k, v in map(lambda kv: kv.split('=', maxsplit=1), v.split(';'))} for v in values]
        setattr(namespace, self.dest, res_dict)

class StoreMultiDict(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreMultiDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        res_dict ={k: v.split(';') for k, v in map(lambda kv: kv.split('='), values)}
        setattr(namespace, self.dest, res_dict)

# copied from https://stackoverflow.com/questions/7419665/python-move-and-overwrite-files-and-folders
def shutil_overwrite_dir(root_src_dir: str, root_dst_dir: str):
    if sys.version_info.major < 3 or sys.version_info.minor < 8:
        for src_dir, dirs, files in os.walk(root_src_dir):
            dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    # in case of the src and dst are the same file
                    if os.path.samefile(src_file, dst_file):
                        continue
                    os.remove(dst_file)
                shutil.move(src_file, dst_dir)
    else:
        shutil.copytree(root_src_dir, root_dst_dir, dirs_exist_ok=True)

def del_dir_content(dir: str):
    for f in os.listdir(dir):
        f = path.join(dir, f)
        if path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)

ITER_TYPE = TypeVar('ITER_TYPE')
def list_or_none(in_iterable: Optional[Iterable[ITER_TYPE]]) -> Optional[List[ITER_TYPE]]:
    if in_iterable is not None:
        return list(in_iterable)
    return None

def del_if_present(d: Dict[Any, Any], key: Any) -> Dict[Any, Any]:
    if key in d:
        del d[key]
    return d

TRIALS_INTERMEDIATES_DIR = 'trial_intermediates'

def parse_tar_generic(tar_file: str, process_experiment_head: Callable[[tarfile.TarFile, tarfile.TarInfo], None],
              process_entry: Callable[[tarfile.TarFile, tarfile.TarInfo, List[str], List[str]], tarfile.TarInfo]):
    print(f'Parsing tar at {tar_file}')
    t = time.time()
    i = 1
    with tarfile.open(tar_file, 'r:gz' if tar_file.endswith('gz') else 'r') as tar:
        next_entry = tar.next()
        print(f'({time.time() - t:.3f}) Processing entry {i}: {next_entry.name}')
        prefix = next_entry.name
        process_experiment_head(tar, next_entry)
        next_entry = tar.next()
        i += 1
        print(f'({time.time() - t:.3f}) Processing entry {i}: {next_entry.name}')
        skip_folders = [prefix + '/' + TRIALS_INTERMEDIATES_DIR, prefix + '/_sources']
        skip_files = [prefix + '/trial_copy.json', prefix + '/trial_copy.csv',
                      prefix + '/trial_distributions_copy.json', prefix+'/best_trials.json']
        while next_entry is not None:
            new_entry = process_entry(tar, next_entry, skip_folders, skip_files)
            next_entry = tar.next() if new_entry is None else new_entry
            i += 1
            print(
                f'({time.time() - t:.3f}) Processing entry {i}: {next_entry.name if next_entry is not None else "None"}')
    t = time.time() - t
    print(f'Finished parsing which took {t:.3f}s - {t / i:.3f}s on average.')


def right_slash_split(to_process: str) -> str:
    return to_process.rsplit('/', maxsplit=1)[-1]


def skip_with_prefix(tar: tarfile.TarFile, to_process: tarfile.TarInfo, prefixes: List[str]) -> tarfile.TarInfo:
    while to_process is not None and any(map(lambda p: to_process.name.startswith(p), prefixes)):
        to_process = tar.next()
    return to_process