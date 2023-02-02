import sys

import numpy as np

from dataset import read_dem_only, read_labels_only, read_s1_images, read_s2_images, _FolderFileListDataset
import datadings.writer
import os.path as path
import torch.utils.data as data
import traceback
import argparse
from typing import Dict, Any, Union, Tuple

LABEL_HAND_FOLDER = 'label'
S1_HAND_FOLDER = 'S1Hand'
S2_HAND_FOLDER = 'S2Hand'
DEM_HAND_FOLDER = 'DEMHand'
LABEL_WEAK_S1_FOLDER = 'S1WeakLabel'
LABEL_WEAK_S2_FOLDER = 'S2WeakLabel'
S1_WEAK_FOLDER = 'S1Weak'
S2_WEAK_FOLDER = 'S2Weak'
DEM_WEAK_FOLDER = 'DEMWeak'

def key_and_region(f_name: str):
    index = max(f_name.rfind('\\'), f_name.rfind('/'))
    f_name = f_name if index < 0 else f_name[index + 1:]
    split_str = path.basename(f_name).split('_')
    return int(split_str[1]), split_str[0]

def Sen1Floods11Data(f_name: str, data, with_label: bool, with_dem: bool) -> Dict[str, Any]:
    key, region = key_and_region(f_name)
    if not with_label and not with_dem:
        assert isinstance(data, np.ndarray), 'If neither labels or dem data are included, expected a simple array image'
        image = data
        segmentation = None
        dem = None
    elif with_label and not with_dem:
        assert isinstance(data, tuple) and len(data) == 2 and \
               not any(map(lambda d: True, filter(lambda d: isinstance(d, np.ndarray), data))), \
            'If labels are present and dem data is not, expected a 2-element tuple of numpy arrays!'
        image = data[0]
        segmentation = data[1]
        dem = None
    elif not with_label and with_dem:
        assert isinstance(data, tuple) and len(data) == 2 and \
               not any(map(lambda d: True, filter(lambda d: isinstance(d, np.ndarray), data))), \
            'If labels are not present and dem data is, expected a 2-element tuple of numpy arrays!'
        image = data[0]
        segmentation = None
        dem = data[1]
    else:
        assert isinstance(data, tuple) and len(data) == 3 and \
               not any(map(lambda d: True, filter(lambda d: isinstance(d, np.ndarray), data))), \
            'If labels and dem data are present, expected a 3-element tuple of numpy arrays!'
        image = data[0]
        segmentation = data[1]
        dem = data[2]
    res = {
        'key': str(key), # TODO change this once datadings allows int keys
        'image': image,
        'region': region
    }
    if segmentation is not None:
        res['label_image'] = segmentation
    if dem is not None:
        res['dem_image'] = dem
    return res

def _yield_samples(dataset: _FolderFileListDataset, with_label: bool, with_dem: bool):
    for i in range(len(dataset)):
        yield Sen1Floods11Data(dataset.files[i], dataset[i], with_label, with_dem)

def _write_set(split: str, dataset: _FolderFileListDataset, output_folder: str, with_label: bool, with_dem: bool):
    output_file = path.join(output_folder, split+'.msgpack')
    try:
        with datadings.writer.FileWriter(output_file, total=len(dataset), overwrite=False) as data_writer:
            for sample in _yield_samples(dataset, with_label, with_dem):
                data_writer.write(sample)
    except Exception as e:
        print(e)

def _convert_split(base_folder: str, output_folder: str, with_separate_labels: bool = True,
                   with_separate_dem: bool = True):
    print('Converting hand-labeled Sen1Floods11 datasets to datadings-msgpack format.')
    print(f'Base-folder: {base_folder}')
    print(f'Output-folder: {output_folder}')
    print(f'Conversion is performed with separate labels: {str(with_separate_labels)}')
    print(f'Conversion is performed with separate dem: {str(with_separate_dem)}')
    dem_joined = path.join(base_folder, DEM_WEAK_FOLDER)
    label_joined = path.join(base_folder, LABEL_HAND_FOLDER)
    if not path.exists(dem_joined):
        print(f'DEM data-folder not found at {dem_joined}, will be omitted from dataset creation.')
        dem_joined = None
    if not path.exists(label_joined):
        print(f'Label-folder not found at {label_joined}, will be omitted from dataset creation.')
        label_joined = None
    dem_folder = None if with_separate_dem else dem_joined
    label_folder = None if with_separate_labels else label_joined
    descriptors = [(lambda sf: read_s1_images(path.join(base_folder, S1_HAND_FOLDER),
                                              label_folder=label_folder, dem_folder=dem_folder,
                                              split_csv_file=sf, as_numpy=True),
                    '_s1'),
                   (lambda sf: read_s2_images(path.join(base_folder, S2_HAND_FOLDER),
                                              label_folder=label_folder, dem_folder=dem_folder,
                                              split_csv_file=sf, as_numpy=True, is_v10_data=False),
                    '_s2')]
    if with_separate_dem and dem_joined is not None:
        descriptors.append((lambda sf: read_dem_only(dem_joined,
                                                      split_csv_file=sf,
                                                      as_numpy=True), '_dem'))
    if with_separate_labels and label_joined is not None:
        descriptors.append((lambda sf: read_labels_only(label_joined,
                                                      split_csv_file=sf,
                                                      as_numpy=True), '_label'))

    for split in ['train', 'valid', 'test', 'bolivia']:
        for constructor, suffix in descriptors:
            try:
                split_str = f'{split}_data{suffix}'
                sf_sstr = split_str if suffix != '_label' and suffix != '_s1' else f'{split}_data'
                split_file = path.join(base_folder, 'split', 'flood_'+sf_sstr+'.csv')
                dataset = constructor(split_file)
                set_folder = path.join(output_folder, split_str)
                _write_set(split_str, dataset, set_folder, not with_separate_labels, not with_separate_dem)
            except ValueError:
                print(f'unable to convert split {split} with suffix {suffix} as files do not exist!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            except FileExistsError:
                print(f'Skipping split {split} with suffix {suffix}.')


def _convert_weak(base_folder: str, output_folder: str, with_separate_labels: bool = True,
                  with_separate_dem: bool = True):
    print('Converting weak Sen1Floods11 datasets to datadings-msgpack format.')
    print(f'Base-folder: {base_folder}')
    print(f'Output-folder: {output_folder}')
    print(f'Conversion is performed with separate labels: {str(with_separate_labels)}')
    print(f'Conversion is performed with separate dem: {str(with_separate_dem)}')
    dem_joined = path.join(base_folder, DEM_WEAK_FOLDER)
    label_s1_joined = path.join(base_folder, LABEL_WEAK_S1_FOLDER)
    label_s2_joined = path.join(base_folder, LABEL_WEAK_S2_FOLDER)
    if not path.exists(dem_joined):
        print(f'DEM data-folder not found at {dem_joined}, will be omitted from dataset creation.')
        dem_joined = None
    if not path.exists(label_s1_joined):
        print(f'Weak S1 label-folder not found at {label_s1_joined}, will be omitted from dataset creation.')
        label_s1_joined = None
    if not path.exists(label_s2_joined):
        print(f'Weak S2 label-folder not found at {label_s2_joined}, will be omitted from dataset creation.')
        label_s2_joined = None
    dem_folder = None if with_separate_dem else dem_joined
    label_s1_folder = None if with_separate_labels else label_s1_joined
    label_s2_folder = None if with_separate_labels else label_s2_joined
    constructors = [(lambda fb: read_s1_images(path.join(base_folder, S1_WEAK_FOLDER), label_folder=label_s1_folder,
                                               dem_folder=dem_folder, filter_bolivia=fb, as_numpy=True), '_s1'),
                    (lambda fb: read_s2_images(path.join(base_folder, S2_WEAK_FOLDER), label_folder=label_s2_folder,
                                               dem_folder=dem_folder, filter_bolivia=fb, as_numpy=True,
                                               is_v10_data=True), '_s2')]
    if with_separate_labels:
        if label_s1_joined is not None:
            constructors.append((lambda fb: read_labels_only(label_s1_joined,
                                                    filter_bolivia=fb, as_numpy=True), '_label_s1'))
        if label_s2_joined is not None:
            constructors.append((lambda fb: read_labels_only(label_s2_joined,
                                                    filter_bolivia=fb, as_numpy=True), '_label_s2'))
    if with_separate_dem and dem_joined is not None:
        constructors.append((lambda fb: read_dem_only(dem_joined,
                                                     filter_bolivia=fb, as_numpy=True), '_dem'))
    for constructor, suffix in constructors:
        for filter_bolivia in [True, False]:
            try:
                dataset = constructor(filter_bolivia)
                split = 'weak_data' + ('_no_bolivia' if filter_bolivia else '') + suffix
                set_folder = path.join(output_folder, split)
                _write_set(split, dataset, set_folder, not with_separate_labels, not with_separate_dem)
            except ValueError:
                print(f'Unable to convert weak suffix {suffix} with filter_bolivia={filter_bolivia}', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            except FileExistsError:
                print(f'Skipping weak suffix {suffix} with filter_bolivia={filter_bolivia}.')


def convert_seperated_to_datadings(base_folder: str, output_folder: str, with_separate_labels: bool = True,
                  with_separate_dem: bool = True):
    _convert_split(base_folder, output_folder, with_separate_labels, with_separate_dem)
    _convert_weak(base_folder, output_folder, with_separate_labels, with_separate_dem)

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('indir')
    arg_parse.add_argument('outdir')
    arg_parse.add_argument('--separate_labels', default=True)
    arg_parse.add_argument('--separate_dem', default=True)
    args = arg_parse.parse_args()
    convert_seperated_to_datadings(args.indir, args.outdir, args.separate_labels, args.separate_dem)