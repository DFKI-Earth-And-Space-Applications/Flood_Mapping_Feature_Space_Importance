import argparse
import gc
import json
import math
import os.path as path
import time
from typing import Optional, Tuple, Dict, Any, List, Union, Iterable

import datadings.reader as ddreader
import datadings.torch
import numpy as np
from torch.utils import data as data

import pipeline as pl
import sen1floods11.dataset as sen_data


def _per_item_meta(f_name: str) -> pl.PerItemMeta:
    index = max(f_name.rfind('\\'), f_name.rfind('/'))
    f_name = f_name if index < 0 else f_name[index + 1:]
    split_str = path.basename(f_name).split('_')
    return pl.PerItemMeta(split_str[0], int(split_str[1]))


CHANNEL_NAMES_LABEL = ['label']
CHANNEL_NAMES_GBDT_LABEL = ['GBDT-Dry-Prob', 'GBDT-Water-Prob']
CHANNEL_NAMES_S1 = ['VV', 'VH']
CHANNEL_NAMES_S2 = ['Coastal', 'Blue', 'Green', 'Red', 'RedEdge-1', 'RedEdge-2', 'RedEdge-3', 'NIR', 'Narrow NIR',
                    'Water Vapor', 'Cirrus', 'SWIR-1', 'SWIR-2']
SPLIT_BOLIVIA = 'bolivia'
SPLIT_TEST = 'test'
SPLIT_TRAIN = 'train'
SPLIT_VALIDATION = 'valid'
SPLIT_WEAK = 'weak'
SPLIT_WEAK_NO_BOLIVIA = 'weak-bolivia'
VAILD_SPLITS = {SPLIT_BOLIVIA, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_WEAK, SPLIT_WEAK_NO_BOLIVIA}
TYPE_LABEL = 'label'
TYPE_S1 = 'S1'
TYPE_S2 = 'S2'
TYPE_S1_WEAK_LABEL = 'S1WeakLabel'
TYPE_S2_WEAK_LABEL = 'S2WeakLabel'
TYPE_GBDT_WEAK_LABEL = 'GBDTWeakLabel'
TYPE_GBDT_WEAK_LABEL_CALIBRATED = 'GBDTWeakLabelCalibrated'
TYPE_GBDT_WEAK_LABEL_CALIBRATED_TRINARY = 'GBDTWeakLabelCalibratedTrinary'
VALID_TYPES = {TYPE_LABEL, TYPE_S1, TYPE_S2, TYPE_S1_WEAK_LABEL, TYPE_S2_WEAK_LABEL, TYPE_GBDT_WEAK_LABEL,
               TYPE_GBDT_WEAK_LABEL_CALIBRATED, TYPE_GBDT_WEAK_LABEL_CALIBRATED_TRINARY}


def validate_type_and_split(type: str, split: str) -> Tuple[str, str]:
    split = split.lower().strip()
    type = type.strip()
    if split not in VAILD_SPLITS:
        raise ValueError('Unknown split type ' + split + '!' + 'Expected one of ' + str(VAILD_SPLITS))
    if type not in VALID_TYPES:
        raise ValueError('Unknown data type ' + type + '!' + 'Expected one of ' + str(VALID_TYPES))
    return type, split


class DataSource(pl.Pipeline):

    def get_params_as_dict(self) -> Dict[str, Any]:
        res = super().get_params_as_dict()
        if 'meta_type' in res and type(res['meta_type']) is int:
            res['meta_type'] = pl.types_as_name(pl.empty_meta()._replace(type=res['meta_type']))
        return res


class Sen1Floods11DataSource(DataSource):
    def __init__(self, base_folder: str, type: str, split: str, use_numpy=True, as_array: bool = False):
        type, split = validate_type_and_split(type, split)
        split_path_no_end = path.join(base_folder, 'split', 'flood_' + split + '_data')
        folder = path.join(base_folder, type)
        self.channel_names = [type]
        if 'label' == type:  # must be one of the hand labeled splits
            assert 'weak' not in split, 'Cannot create hand-labeled dataset for any weak split...'
            self.cons = lambda indent: sen_data.read_labels_only(folder,
                                                                 split_csv_file=split_path_no_end + '.csv',
                                                                 as_numpy=use_numpy,
                                                                 indent=indent)
            self.meta_type = pl.META_TYPE_LABEL
        elif 'Label' not in type:
            is_weak = 'weak' in split
            folder += 'Weak' if is_weak else 'Hand'
            filter_bolivia = split == SPLIT_WEAK_NO_BOLIVIA
            if type == TYPE_S1:
                self.cons = lambda indent: sen_data.read_s1_images(folder,
                                                                   split_csv_file=(
                                                                       None if is_weak else split_path_no_end + '.csv'),
                                                                   filter_bolivia=filter_bolivia,
                                                                   as_numpy=use_numpy,
                                                                   indent=indent)
                self.channel_names = CHANNEL_NAMES_S1
                self.meta_type = pl.META_TYPE_S1
            else:
                self.cons = lambda indent: sen_data.read_s2_images(folder,
                                                                   split_csv_file=(
                                                                       None if is_weak else split_path_no_end + '_s2.csv'),
                                                                   filter_bolivia=filter_bolivia,
                                                                   as_numpy=use_numpy,
                                                                   is_v10_data=is_weak,
                                                                   indent=indent)
                self.channel_names = CHANNEL_NAMES_S2
                self.meta_type = pl.META_TYPE_S2
        else:
            assert 'weak' in split, 'Cannot create weakly labeled dataset for a non-weak split.'
            filter_bolivia = split == SPLIT_WEAK_NO_BOLIVIA
            self.cons = lambda indent: sen_data.read_labels_only(folder,
                                                                 as_numpy=use_numpy,
                                                                 filter_bolivia=filter_bolivia,
                                                                 indent=indent)
            self.meta_type = pl.META_TYPE_LABEL
        self.base_folder = base_folder
        self.type = type
        self.split = split
        self.as_array = as_array

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        meta: pl.Meta = pl.empty_meta() if meta is None else meta
        if dataset is not None:
            self.print(meta, 'Overwriting already provided dataset!!!', True)
        self.print(meta, 'Constructing dataset')
        t = time.time()
        dataset = self.cons(meta.indent + '\t')
        assert isinstance(dataset, sen_data._FolderFileListDataset)
        meta = meta._replace(channel_names=self.channel_names,
                             per_item_info=list(map(_per_item_meta, dataset.files)),
                             type=self.meta_type,
                             split=self.split)
        if self.as_array:
            self.print(meta, f'Requested result to be returned as an ArrayInMemoryDataset. Concatenating.')
            t1 = time.time()
            dataset = pl.ArrayInMemoryDataset(dataset)
            t1 = time.time() - t1
            self.print(meta, f'Concat completed in {t1:.3f}s - {t1 / len(dataset):.3f}s on average.')
        t = time.time() - t
        self.print(meta, f'Construction completed in {t:.3f}s. Loaded {str(self.channel_names)} and an exemplary item '
                         f'info is {str(meta.per_item_info[0])}.')
        return dataset, meta

    def __repr__(self):
        return f'Sen1Floods11DataSource(({self.base_folder}, {self.type},{self.split}) => {self.channel_names})'

def evaluate_type_and_split(type: str, split: str) -> Tuple[str, List[str], pl.AMT_TYPE]:
    channel_names = CHANNEL_NAMES_LABEL
    if 'label' == type:  # must be one of the hand labeled splits
        assert 'weak' not in split, 'Cannot create hand-labeled dataset for any weak split...'
        meta_type = pl.META_TYPE_LABEL
    elif 'Label' not in type:
        if type == TYPE_S1:
            channel_names = CHANNEL_NAMES_S1
            meta_type = pl.META_TYPE_S1
        else:
            channel_names = CHANNEL_NAMES_S2
            meta_type = pl.META_TYPE_S2
    else:
        meta_type = pl.META_TYPE_LABEL
    if split == SPLIT_WEAK_NO_BOLIVIA:
        split = 'weak_data_no_bolivia'
    else:
        split += '_data'
    if type.lower() in {'label', 's1', 's2'}:
        type = type.lower()
    else:
        assert type == TYPE_S1_WEAK_LABEL or type == TYPE_S2_WEAK_LABEL or type == TYPE_GBDT_WEAK_LABEL or \
               type == TYPE_GBDT_WEAK_LABEL_CALIBRATED or type == TYPE_GBDT_WEAK_LABEL_CALIBRATED_TRINARY
        if type == TYPE_GBDT_WEAK_LABEL or type == TYPE_GBDT_WEAK_LABEL_CALIBRATED or TYPE_GBDT_WEAK_LABEL_CALIBRATED_TRINARY:
            channel_names = CHANNEL_NAMES_GBDT_LABEL
        if type == TYPE_GBDT_WEAK_LABEL_CALIBRATED:
            suffix = '_calibrated'
        elif type == TYPE_GBDT_WEAK_LABEL_CALIBRATED_TRINARY:
            suffix = '_calibrated_trinary'
        else:
            suffix = ''
        type = 'label_' + type[:2].lower() + suffix
    return split + '_' + type, channel_names, meta_type

class Sen1Floods11DataDingsDatasource(DataSource):
    def __init__(self, base_folders: Union[str, Iterable[str]], type: str, split: str, use_numpy=True, in_memory: bool = True,
                 buffering: int = 0, chunk_size: int = 256, as_array: bool = False, force_f_name: Optional[str] = None):
        type, split = validate_type_and_split(type, split)
        self.split = split
        self.f_name, self.channel_names, self.meta_type = evaluate_type_and_split(type, split)
        if force_f_name is not None:
            print(f'Forcing Datadings-Datasource file to {force_f_name}! Please note that this should behave like an '
                  f'replacement for type={type} and split={split} (=> channel_names={self.channel_names})!')
            self.f_name = force_f_name
        self.base_folders = [base_folders] if isinstance(base_folders, str) else list(base_folders)
        self.use_numpy = use_numpy
        self.in_memory = in_memory
        self.buffering = buffering
        self.chunk_size = chunk_size
        self.as_array = as_array

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        meta: pl.Meta = pl.empty_meta() if meta is None else meta
        if dataset is not None:
            self.print(meta, 'Overwriting already provided dataset!!!', True)
        f_name = next(f_name for f_name in map(lambda bf: path.join(bf, self.f_name, self.f_name + '.msgpack'),
                                               self.base_folders)
                      if path.exists(f_name) and path.isfile(f_name))
        self.print(meta, f'Constructing dataset at {f_name} {"in memory" if self.in_memory else "lazily"}.')
        t = time.time()
        reader = datadings.reader.MsgpackReader(f_name,
                                                buffering=self.buffering)
        if self.in_memory:
            dataset = datadings.torch.IterableDataset(reader, copy=True, chunk_size=self.chunk_size)
            msgpack_list = [data for data in dataset]
            images = [data['image'] for data in msgpack_list]
            per_item_info = [pl.PerItemMeta(data['region'], data['key']) for data in dataset]
            del msgpack_list
            del dataset
            dataset = pl.ShapelessInMemoryDataset(images)
            if self.as_array:
                self.print(meta, f'Requested result to be returned as an ArrayInMemoryDataset. Concatenating.')
                t1 = time.time()
                dataset = pl.ArrayInMemoryDataset(dataset)
                t1 = time.time() - t1
                self.print(meta, f'Concat completed in {t1:.3f}s - {t1 / len(images):.3f}s on average.')
        else:
            dataset = datadings.torch.IterableDataset(reader, copy=False, chunk_size=self.chunk_size)
            per_item_info = [pl.PerItemMeta(data['region'], data['key']) for data in dataset]
            del dataset
            dataset = datadings.torch.Dataset(reader, transforms=self._transform)
        res = dataset, meta._replace(per_item_info=per_item_info,
                                     channel_names=self.channel_names,
                                     type=self.meta_type,
                                     split=self.split)
        t = time.time() - t
        self.print(meta, f'Construction completed in {t:.3f}s - average is {t / len(dataset):.3f}s.')
        return res

    def _transform(self, d: Dict[str, Any]) -> np.ndarray:
        res: np.ndarray = d['image']
        if res.ndim == 4:# NN output
            res = res[0]
            res = np.concatenate((1-res, res), axis=0)
        return res


def print_info(folder: str, type: str, split: str, range_values: Dict[str, List[float]]):
    print('-----------------------------------------')
    print(f'Loading {type}-{split} from {folder}.')
    train_loader = Sen1Floods11DataDingsDatasource(folder, type, split)
    data, meta = train_loader(None, None)
    print('Converting to in-memory-array')
    data = pl.ArrayInMemoryDataset(data).data
    gc.collect()
    print(f'Evaluating channel-properties for index.')
    for i, cn in enumerate(meta.channel_names):
        sl_data = data[:, i]
        min_v = float(sl_data.min())
        max_v = float(sl_data.max())
        print('Channel', i, 'is', cn,
              f'with properties: [{min_v}, {max_v}] distributed according to N({sl_data.mean()}, {sl_data.std()})')
        cur_min, cur_max = range_values.get(cn, [math.inf, -math.inf])
        the_min = min(min_v, cur_min)
        range_values[cn] = [the_min, max(max_v - the_min, cur_max)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    args = parser.parse_args()
    range_values = {}
    for type in [TYPE_S1, TYPE_S2]:
        for split in [SPLIT_VALIDATION, SPLIT_TRAIN, SPLIT_WEAK]:
            print_info(args.data_folder, type, split, range_values)
    print('-----------------------------------------')
    range_file = path.abspath('range.json')
    print('Writing result to', range_file)
    with open(range_file, 'w') as fd:
        json.dump(range_values, fd, indent=4)
