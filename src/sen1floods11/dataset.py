import argparse
import csv
import itertools as itert
import os
import os.path as path
import sys
from typing import Callable, Optional, List, Iterable, Tuple, Union, Set

# import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.utils.data as data

NOFITY_CHECK_FILE_READ = False
try:
    import rasterio

    USE_TIFFFILE = False


    def _read_tif(fname):
        if NOFITY_CHECK_FILE_READ:
            print('Reading tif-file', fname)
            assert fname.endswith('.tif')
        return rasterio.open(fname).read()
except ModuleNotFoundError:
    # warning this has been changed from rasterio, as that didn't get any updates for recent python versions
    # However it seems to read it correctly
    print('WARNING: Failed to load rasterio. Trying to load data via tifffile however this is not the library used',
          'for loading the data in the original Sen1Floods11 paper. Use at own risk.', file=sys.stderr)
    import tifffile as tif

    USE_TIFFFILE = True


    def _read_tif(fname: str) -> np.ndarray:
        if NOFITY_CHECK_FILE_READ:
            print('Reading tif-file', fname)
            assert fname.endswith('.tif')
        return tif.imread(fname)


# checks whether the file exists and notifies if it doesn't
# returns True if the file should be skipped because it doesn't exist
def _check_notify_fnexists(f_name: str) -> bool:
    if not os.path.exists(f_name):
        print("Skipping", f_name, 'as it does not exist.', file=sys.stderr)
        return True
    return False


def _read_label_mask(as_numpy=False) -> Callable:
    def perform_read(f_name: str) -> Optional[Union[torch.Tensor, numpy.ndarray]]:
        if _check_notify_fnexists(f_name):
            return None
        mask_data = _read_tif(f_name)
        mask_data = np.int8(mask_data)
        if np.sum((mask_data != mask_data)) != 0:
            print('Found nan values in file', f_name, '- Skipping.', file=sys.stderr)
            return None
        if len(mask_data.shape) == 2:
            mask_data = np.expand_dims(mask_data, 0)
        return mask_data if as_numpy else torch.from_numpy(mask_data)

    return perform_read


def _read_dem_data(as_numpy=False) -> Callable:
    def perform_read(f_name: str) -> Optional[Union[torch.Tensor, numpy.ndarray]]:
        if _check_notify_fnexists(f_name):
            return None
        dem_data = _read_tif(f_name)
        # TODO verify this actually works!!!
        return dem_data if as_numpy else torch.from_numpy(dem_data.astype(np.float32))

    return perform_read


def _read_s1_data(as_numpy=False) -> Callable:
    def perform_read(f_name: str) -> Optional[Union[torch.Tensor, numpy.ndarray]]:
        if _check_notify_fnexists(f_name):
            return None
        s1_data = _read_tif(f_name)
        # The below transformations are done by the Sen1Floods11 on all S1 chips
        s1_data = np.nan_to_num(s1_data)
        s1_data = np.clip(s1_data, -50, 1)
        s1_data = (s1_data + 50) / 51
        return s1_data if as_numpy else torch.from_numpy(s1_data)

    return perform_read


def _read_s2_data(do_permute, as_numpy=False) -> Callable:
    def perform_read(f_name: str) -> Optional[Union[torch.Tensor, numpy.ndarray]]:
        if _check_notify_fnexists(f_name):
            return None
        # The example code provided by the Sen1Floods11 repo never uses Sentinel-2 chips
        # This is therefore reconstructed from their README notes and from
        # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric
        s2_data = _read_tif(f_name)
        # pytorch can't handle uint16 and I don't want to take the effort of testing out whether int16 works as well
        # (To my knowledge it should, but changing this might introduce very very subtle bugs)
        s2_data = np.transpose(s2_data, (2, 0, 1)) if USE_TIFFFILE and do_permute else s2_data
        return s2_data if as_numpy else torch.from_numpy(s2_data.astype(np.float32))

    return perform_read


# A simple map-style dataset representing all files (filtered by some function) in a folder.
# The filter function takes the file_name (without the base_folder) as input and outputs if this is a valid entry for
# the dataset. This can be used to for example filter for '.tif' files or to exclude bolivia data.
# In a similar way it is possible to use this function to filter for the predefined splits.
class _FolderFileListDataset(data.Dataset):
    def __init__(self, base_folder: str, load_function: Callable, filter_fun: Callable, indent=''):
        assert base_folder is not None and load_function is not None and filter_fun is not None, \
            'Expecting non-null arguments. Given base_folder="' + base_folder + '", load_function=' + str(load_function) \
            + ', filter_fun=' + str(filter_fun) + ' !'
        if not path.exists(base_folder):
            raise ValueError('Cannot create dataset for non existing folder ' + base_folder +
                             '. Expecting a valid directory!')
        if not path.isdir(base_folder):
            raise ValueError('Cannot create dataset for non-folder ' + base_folder +
                             '. Expecting a valid directory!')
        self.files = sorted([path.join(base_folder, file) for file in os.listdir(base_folder)
                             if filter_fun(file)])
        self._load_tif = load_function
        print(indent + 'Created FolderFileListDataset with', len(self.files), ' files.')

    def __getitem__(self, index) -> torch.Tensor:
        im_fname = self.files[index]
        read = self._load_tif(im_fname)
        if read is None:
            raise RuntimeError('Cannot read file ' + im_fname)
        return read

    def __len__(self) -> int:
        return len(self.files)


# Simple Wrapper for Datasets supporting the __len__ and __getitem__ operation that loads the data into a list to keep
# it in memory. Notice that no reference to the original dataset is kept and it is fully resolved at construction time!
class InMemoryDataset(data.Dataset):
    def __init__(self, wrapped: data.Dataset):
        self.data = [wrapped[i] for i in range(wrapped.__len__())]

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


# A simple Dataset that wraps multiple datasets into a tuple. For example data and labels...
class MergingDatasetWrapper(data.Dataset):
    def __init__(self, wrapped: Iterable[data.Dataset]):
        wrapped_datasets: List[data.Dataset] = list(itert.chain.from_iterable(
            [(d.wrapped if isinstance(d, MergingDatasetWrapper) else [d]) for d in wrapped]))
        first_obj = next(iter(wrapped_datasets))
        assert isinstance(first_obj, data.Dataset), f'{first_obj} should be a Dataset!!!'
        first_len: int = len(first_obj)
        assert not any(map(lambda f: True, filter(lambda d: len(d) != first_len, wrapped_datasets))), \
            'All wrapped datasets should have equal size'
        self.wrapped: List[data.Dataset] = wrapped_datasets
        self.size: int = first_len

    def __getitem__(self, index) -> Tuple:
        res = tuple(map(lambda d: d[index], self.wrapped))
        if any(filter(lambda r: r is None, res)):
            raise RuntimeError('Unable to query data (None-result) for index ' + index +
                               '. Retrieved joined result was:\n' + str(res))
        return res

    def __len__(self) -> int:
        return self.size


def _load_with_list_filter(base_folder: str, load_function: Callable, filter_list: List[str],
                           indent='') -> _FolderFileListDataset:
    filter_set = set(filter_list)
    return _FolderFileListDataset(base_folder, load_function, lambda f: f in filter_set, indent=indent)


def _load_with_tif_filter(base_folder: str, load_function: Callable, indent='') -> _FolderFileListDataset:
    return _FolderFileListDataset(base_folder, load_function, lambda f: f.endswith('.tif'), indent=indent)


def _csv_to_memory(split_csv_file: str, indent='') -> Tuple[List[str], List[str]]:
    assert split_csv_file is not None
    if not path.exists(split_csv_file):
        raise ValueError('Cannot create dataset for non existing csv index ' + split_csv_file +
                         '. Expecting a valid csv-file!')
    if not path.isfile(split_csv_file):
        raise ValueError('Cannot create dataset for non-folder ' + split_csv_file +
                         '. Expecting a valid csv-file!')
    print(indent + 'Reading csv split at', split_csv_file)
    with open(split_csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        as_list = list(map(lambda l: [s.strip() for s in l[0].split(',')], csv_reader))
    print(indent + 'Found', len(as_list), 'entries.')
    return [row[0] for row in as_list], [row[1] for row in as_list]


def _load_data_and_labels(data_supplier: Optional[Callable], label_supplier: Optional[Callable], indent='') \
        -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    datasets = []
    print(indent + 'Trying to create dataset(s).')
    if data_supplier is not None:
        print(indent + 'Creating data-dataset.')
        datasets.append(data_supplier())
    if label_supplier is not None:
        print(indent + 'Creating label-dataset.')
        datasets.append(label_supplier())
    if not datasets:
        raise ValueError('No data to be loaded was specified. '
                         'Please specify at least either a data folder or a label folder.')
    print(indent + 'Created', '1 dataset.' if len(datasets) == 1 else ' data and label datasets - merging.')
    return datasets[0] if len(datasets) == 1 else MergingDatasetWrapper(datasets)


def _load_from_split(load_function: Optional[Callable], label_load_function: Callable, split_csv_file: str,
                     data_folder: Optional[str] = None, label_folder: Optional[str] = None, indent='') \
        -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    data, labels = _csv_to_memory(split_csv_file, indent=indent)
    data_supplier = (lambda: _load_with_list_filter(data_folder, load_function, data, indent=indent)) \
        if data_folder is not None else None
    label_supplier = (lambda: _load_with_list_filter(label_folder, label_load_function, labels, indent=indent)) \
        if label_folder is not None else None
    return _load_data_and_labels(data_supplier, label_supplier, indent=indent)


def _load_tif_folder(load_function: Optional[Callable], label_load_function: Callable,
                     data_folder: Optional[str] = None, label_folder: Optional[str] = None, filter_bolivia=False,
                     indent='') \
        -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    filter = (lambda f: f.endswith('.tif') and 'Bolivia' not in f) if filter_bolivia else (lambda f: f.endswith('.tif'))
    data_supplier = (lambda: _FolderFileListDataset(data_folder, load_function, filter, indent=indent)) \
        if data_folder is not None else None
    label_supplier = (lambda: _FolderFileListDataset(label_folder, label_load_function, filter, indent=indent)) \
        if label_folder is not None else None
    return _load_data_and_labels(data_supplier, label_supplier, indent=indent)


def read_dem_only(dem_folder: str, split_csv_file: Optional[str] = None, filter_bolivia=False,
                  as_numpy=False, indent=''):
    if split_csv_file is None:
        return _load_tif_folder(_read_dem_data(as_numpy), _read_label_mask(as_numpy), dem_folder,
                                None, filter_bolivia, indent=indent)
    else:
        return _load_from_split(_read_dem_data(as_numpy), _read_label_mask(as_numpy), split_csv_file,
                                dem_folder, None, indent=indent)


def _append_dem(other_dataset: Union[MergingDatasetWrapper, _FolderFileListDataset], replace_ending: str,
                dem_folder: Optional[str] = None, split_csv_file: Optional[str] = None, filter_bolivia=False,
                as_numpy=False, indent='') -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    if dem_folder is not None:
        split_csv_file = '_dem.csv'.join(split_csv_file.rsplit(replace_ending, 1)) if split_csv_file else None
        return MergingDatasetWrapper((other_dataset,
                                      read_dem_only(dem_folder, split_csv_file, filter_bolivia, as_numpy,
                                                    indent=indent)))
    return other_dataset


def read_labels_only(label_folder: str, split_csv_file: Optional[str] = None, filter_bolivia=False,
                     as_numpy=False, indent='') -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    if split_csv_file is None:
        return _load_tif_folder(None, _read_label_mask(as_numpy), None, label_folder, filter_bolivia, indent=indent)
    else:
        return _load_from_split(None, _read_label_mask(as_numpy), split_csv_file, None, label_folder, indent=indent)


def read_s1_images(data_folder: str, label_folder: Optional[str] = None, dem_folder: Optional[str] = None,
                   split_csv_file: Optional[str] = None, filter_bolivia=False, as_numpy=False, indent='') \
        -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    if split_csv_file is None:
        res = _load_tif_folder(_read_s1_data(as_numpy), _read_label_mask(as_numpy), data_folder,
                               label_folder, filter_bolivia, indent=indent)
    else:
        res = _load_from_split(_read_s1_data(as_numpy), _read_label_mask(as_numpy), split_csv_file,
                               data_folder, label_folder, indent=indent)
    return _append_dem(res, '.csv', dem_folder, split_csv_file, filter_bolivia, as_numpy, indent=indent)


def read_s2_images(data_folder: str, label_folder: Optional[str] = None, dem_folder: Optional[str] = None,
                   split_csv_file: Optional[str] = None, filter_bolivia=False, is_v10_data: Optional[bool] = None,
                   as_numpy=False, indent='') -> Union[MergingDatasetWrapper, _FolderFileListDataset]:
    if is_v10_data is None and path.exists(data_folder) and path.isdir(data_folder):
        if any(map(lambda m: True, filter(lambda f: f.endswith('S2Hand.tif'), os.listdir(data_folder)))):
            print(indent + 'Did not specify whether images from the v1.1 or the v1.0 dataset are being read.',
                  'Notice that v1.0 stores the Tensors as [H,W,C] instead of [C,H,W] - autodetect recognized that',
                  'some files end with "S2Hand.tif" which is used by v1.1. Tensor dimensions are therefore not',
                  'permuted. If you notice that the loaded dimensions don\'t match [C,H,W] please specify',
                  '"is_v10_data=True".')
            is_v10_data = False
        elif any(map(lambda m: True, filter(lambda f: f.endswith('S2.tif'), os.listdir(data_folder)))):
            print(indent + 'Did not specify whether images from the v1.1 or the v1.0 dataset are being read.',
                  'Notice that v1.0 stores the Tensors as [H,W,C] instead of [C,H,W] - autodetect recognized that',
                  'some files end with "S2.tif" which is used by v1.0. Tensor dimensions are therefore',
                  'permuted. If you notice that the loaded dimensions don\'t match [C,H,W] please specify',
                  '"is_v10_data=False".')
            is_v10_data = True
        else:
            print(
                indent + '"is_v10_data" was not specified and also could not be inferred from the files in the folder.',
                'Defaulting to False. Notice that v1.0 stores the ensors as [H,W,C] instead of [C,H,W] and this may',
                'result in the loaded data having an unexpected dimensionality.')
            is_v10_data = False
    if split_csv_file is None:
        res = _load_tif_folder(_read_s2_data(is_v10_data, as_numpy), _read_label_mask(as_numpy),
                               data_folder, label_folder, filter_bolivia, indent=indent)
    else:
        res = _load_from_split(_read_s2_data(is_v10_data, as_numpy), _read_label_mask(as_numpy), split_csv_file,
                               data_folder, label_folder, indent=indent)
    return _append_dem(res, '_s2.csv', dem_folder, split_csv_file, filter_bolivia, as_numpy, indent=indent)


FILES_OF_INTEREST: Set[str] = {'flood_bolivia_data.csv', 'flood_test_data.csv', 'flood_train_data.csv',
                               'flood_valid_data.csv'}


def create_s2_split_files(split_csv_folder: str) -> None:
    def check_inputs(split_csv_folder: str):
        if not path.exists(split_csv_folder):
            raise ValueError(split_csv_folder + ' does not exist!')
        if not path.isdir(split_csv_folder):
            raise ValueError(split_csv_folder + ' should be a directory!')

    def check_file(file: str) -> Tuple[bool, str]:
        full_file = path.join(split_csv_folder, file)
        assert path.exists(full_file), f'os.listdir({split_csv_folder}) lists {full_file}, but this does not exist?!?'
        if not path.isfile(full_file):
            print('Skipping non-file', full_file)
            return True, full_file
        if not full_file.endswith('.csv'):
            print('Skipping non csv-file', full_file)
            return True, full_file
        if file not in FILES_OF_INTEREST:
            print(file, 'is not relevant to creating the splits. Skipping.')
            return True, full_file
        return False, full_file

    def process_file(full_file: str) -> Tuple[str, str]:
        def write_out(new_file, transformed_content: List[List[str]]):
            if path.exists(new_file):
                print('WARNING: overwriting existing split at', new_file, file=sys.stderr)
            else:
                print('Writing new split-file for the hand-labeled sentinel-2 data', new_file)
            with open(new_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in transformed_content:
                    csv_writer.writerow(row)

        with open(full_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            s2_transformed_file: List[List[str]] = list(
                map(lambda l: [element.replace('S1Hand', 'S2Hand') for element in l], csv_reader))
            dem_transformed_file: List[List[str]] = list(
                map(lambda l: [element.replace('S1Hand', 'dem') for element in l], csv_reader))
        # append an _s2 to the filename by replacing the last occurrence of ".csv" with "_s2.csv"
        # see also https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
        new_file1 = '_s2.csv'.join(full_file.rsplit('.csv', 1))
        new_file2 = '_dem.csv'.join(full_file.rsplit('.csv', 1))
        write_out(new_file1, s2_transformed_file)
        write_out(new_file2, dem_transformed_file)
        return new_file1, new_file2

    check_inputs(split_csv_folder)
    for file in os.listdir(split_csv_folder):
        skip, full_file = check_file(file)
        if skip:
            continue
        print('---------------------------------------------------------------------------')
        print('Processing', full_file)
        new_files = process_file(full_file)
        print('Finished writing splits', new_files, 'for source', full_file)


if __name__ == '__main__':
    pass # no-command-line interface, just utilities
