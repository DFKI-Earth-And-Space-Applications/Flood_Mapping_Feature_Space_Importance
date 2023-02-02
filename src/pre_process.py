from typing import Iterable, Union, Callable

import numba
import numpy as np
import pipeline as pl
import scipy as sp
import scipy.integrate as integrate
import scipy.ndimage as spimg
import scipy.optimize as opt
import scipy.special as special
import skimage.measure as measure
import torch
import torch.nn.functional as tfunc
import functools as funct

import multipipeline as mpl
import serialisation
import utils
from data_source import *


class PreProcessingModule(pl.Pipeline):
    pass


class RangeCLippingNormalizationDataset(pl.TransformerDataset):
    def __init__(self, wrapped: data.Dataset, range_info: List[List[float]]):
        super().__init__(wrapped)
        self.range_info = range_info

    def _transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.type(torch.float32)
            for i, (offset, scale) in enumerate(self.range_info):
                data[i] = torch.clamp((data[i] - offset) / scale, min=0.0, max=1.0)
        else:
            data = data.astype(np.float32)
            for i, (offset, scale) in enumerate(self.range_info):
                data[i] = np.clip((data[i] - offset) / scale, a_min=0.0, a_max=1.0)
        return data


class RangeClippingNormalizationModule(PreProcessingModule):
    """
    Normalizes the input dataset to a 0 - 1 range as evaluated on the training set.
    Everything outside that range will be clipped.
    """

    def __init__(self, range_file: str = 'range.json'):
        self.range_file = range_file

    def _load_range_info(self, file: str) -> Dict[str, List[float]]:
        if path.exists(file) and path.isfile(file):
            with open(file, 'r') as fd:
                return json.load(fd)
        return {}

    def _update_range_info(self, dataset: data.Dataset, meta: pl.Meta, range_info: Dict[str, List[float]]) \
            -> Tuple[data.Dataset, Dict[str, List[float]]]:
        if not isinstance(dataset, pl.ArrayInMemoryDataset):
            self.print(meta, 'Range-info needs to be updated, but input is not an array-in-memory. Loading.')
            t = time.time()
            dataset = pl.ArrayInMemoryDataset(dataset)
            t = time.time() - t
            print(meta, f'Loading completed took {t:.3f}s - {t / len(dataset):.3f}s on average. Evaluating ranges.')
        else:
            self.print(meta, 'Computing missing range parameters.')
        t = time.time()
        data = dataset.data
        range_info = range_info.copy()
        i = 0
        for i, cn in filter(lambda t: t[1] not in range_info, enumerate(meta.channel_names)):
            min = data[:, i].min()
            max = data[:, i].max()
            scale = 1.0 if -utils.EPS < max - min < utils.EPS else max - min
            range_info[cn] = [float(min), float(scale)]
        t = time.time() - t
        self.print(meta, f'Computed {i} missing range parameters in {t:.3f}s - {t / i:.3f}s on average.')
        return dataset, range_info

    def _perform_range_update(self, dataset: pl.ArrayInMemoryDataset,
                              meta: pl.Meta,
                              range_info: Dict[str, List[float]]) \
            -> data.Dataset:
        self.print(meta, f'Scaling and clipping data to range [0.0, 1.0].')
        data = dataset.data.astype(np.float64)
        indented_meta = meta._replace(indent=meta.indent + '\t')
        t = time.time()
        for i, cn in enumerate(meta.channel_names):
            self.print(indented_meta, f'Updating channel {cn} with index {i}.')
            modified = (data[:, i] - range_info[cn][0]) / range_info[cn][1]
            data[:, i] = torch.clip(modified, min=0.0, max=1.0) if isinstance(data, torch.Tensor) \
                else np.clip(modified, a_min=0.0, a_max=1.0)
        t = time.time() - t
        self.print(meta,
                   f'Scaling and clipping completed. Took {t:.3f}s - {t / len(meta.channel_names):.3f}s on average.')
        return pl.ArrayInMemoryDataset(data)

    def _save_range_info(self, range_file: str, range_info: Dict[str, List[float]]):
        with open(range_file, 'w') as fd:
            json.dump(range_info, fd, indent=4)

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        dataset, meta = pl.check_meta_and_data(dataset, meta)
        range_file = path.abspath(self.range_file)
        self.print(meta, f'Performing range clipping with range file at {range_file}.')
        range_info = self._load_range_info(range_file)
        if any(map(lambda cn: cn not in range_info, meta.channel_names)):
            dataset, updated_range_info = self._update_range_info(dataset, meta, range_info)
        else:
            updated_range_info = range_info
        if isinstance(dataset, pl.ArrayInMemoryDataset):
            dataset = self._perform_range_update(dataset, meta, updated_range_info)
        else:
            self.print(meta, f'Found non-in-memory dataset. Range-clipping is performed lazily.')
            indexed_range_info = [updated_range_info[cn] for cn in meta.channel_names]
            dataset = RangeCLippingNormalizationDataset(dataset, indexed_range_info)
        if updated_range_info != range_info:
            self.print(meta, f'Saving to {range_file} as the range information changed.')
            self._save_range_info(range_file, updated_range_info)
        return dataset, meta


# Extracts some value from combined channels
# Asserts that the input is a torch tensor representing a EO image. Index for the extracted dims can be specified
# via the dim_list argument.
# All results are concatenated which allows Multiple Image extractors to be executed in sequence and the sequence order
# specifying the order of the resulting features.
class _EOImageExtractor(pl.TransformerDataset):
    def __init__(self, wrapped: data.Dataset, image_dims: Iterable[Union[int, Tuple[int, ...]]]):
        super().__init__(wrapped)
        self.relevant_dims: Iterable[Union[int, Tuple[int, ...]]] = image_dims

    def _transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return self._compute_feature(*[data[dim] for dim in self.relevant_dims]).reshape((-1,) + data.shape[1:])

    def _compute_feature(self, *args):
        raise NotImplementedError


class EOImageExtractorModule(PreProcessingModule):
    def __init__(self, name: str, image_dims: Tuple[str, ...]):
        super().__init__()
        self.name = name
        self.image_dims = image_dims
        self.multi_output = False

    def get_image_dims(self, meta: pl.Meta) -> Tuple[int, ...]:
        return tuple(meta.channel_names.index(image_dim) for image_dim in self.image_dims)

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        dataset, meta = pl.check_meta_and_data(dataset, meta)
        self.print(meta, f'Creating wrapper in order to extract {self.name}.')
        if self.name in meta.channel_names:
            print(f'Index {self.name} is already present as channel {meta.channel_names.index(self.name)}! Skipping.',
                  True)
            return dataset, meta
        image_dims = self.get_image_dims(meta)
        new_meta = meta._replace(channel_names=[self.name])
        if isinstance(dataset, pl.ArrayInMemoryDataset):
            dataset = self.extract_in_memory(dataset, image_dims)
            if not self.multi_output:
                data = dataset.data
                dataset.data = data.reshape((data.shape[0],) + (1,) + data.shape[1:])
            return dataset, new_meta
        return self._create_extractor_not_in_memory(meta, dataset, image_dims), new_meta

    def _create_extractor_not_in_memory(self, meta: pl.Meta, dataset: data.Dataset,
                                        image_dims: Tuple[int, ...]) -> data.Dataset:
        return self._create_extractor(dataset, image_dims)

    def _create_extractor(self, dataset: data.Dataset, image_dims: Tuple[int, ...]) -> data.Dataset:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement _create_extractor')

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, ...]) -> pl.ArrayInMemoryDataset:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement _create_in_memory')


# Extracts the VV / VH ratio
# Asserts that only two dimensions are given which are (in this order) VV and VH respectively
class _VV_VH_RatioExtractor(_EOImageExtractor):
    def __init__(self, wrapped: data.Dataset, image_dims: Iterable[Union[int, torch.Size, Tuple]] = (0, 1)):
        image_dims = list(image_dims)
        assert len(image_dims) == 2, 'Only VV and VH channels in this order are expected as relevant dims!!!'
        super().__init__(wrapped, image_dims)

    def _compute_feature(self, vv: Union[torch.Tensor, np.ndarray], vh: Union[torch.Tensor, np.ndarray]) \
            -> Union[torch.Tensor, np.ndarray]:
        return (vv + utils.EPS) / (vh + utils.EPS)


class VV_VH_RatioExtractorModule(EOImageExtractorModule):
    def _create_extractor(self, dataset: data.Dataset, image_dims: Tuple[int, int]) -> data.Dataset:
        return _VV_VH_RatioExtractor(dataset, image_dims)

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, int]) -> pl.ArrayInMemoryDataset:
        return pl.ArrayInMemoryDataset(dataset.data[:, image_dims[0]] / dataset.data[:, image_dims[1]])


# Extracts some X - Y ratio (optionally rescaled by a division of (X + Y) such that it forms an index)
# Asserts that only two dimensions are given which are (in this order) VV and VH respectively
class _IndexExtractor(_EOImageExtractor):
    def __init__(self, wrapped: data.Dataset,
                 image_dims: Iterable[Union[int, torch.Size, Tuple]],
                 rescale: bool = True):
        image_dims = list(image_dims)
        assert len(image_dims) == 2, 'Only two channels are expected as relevant dims to compute an index!!!'
        super().__init__(wrapped, image_dims)
        self.rescale: bool = rescale

    def _compute_feature(self, first: Union[torch.Tensor, np.ndarray], second: Union[torch.Tensor, np.ndarray]) \
            -> Union[torch.Tensor, np.ndarray]:
        sub = first - second
        if self.rescale:
            add = first + second
            mask = (add == 0.0)
            # on purpose test of equal to 0... as only those are relevant for producing zeros
            if isinstance(add, torch.Tensor):
                add = add.type('float32')
                sub = sub.type('float32')
            else:
                add = add.astype(np.float32)
                sub = sub.astype(np.float32)
            if (torch.any(mask) if isinstance(first, torch.Tensor) else np.any(mask)):
                add[mask] += utils.EPS
                sub[mask] += utils.EPS

            return sub / add
        return sub


class IndexExtractorModule(EOImageExtractorModule):
    def __init__(self, name: str, image_dims: Tuple[str, str], rescale: bool = True):
        super().__init__(name, image_dims)
        self.rescale = rescale

    def _create_extractor(self, dataset: data.Dataset, image_dims: Tuple[int, int]) -> data.Dataset:
        return _IndexExtractor(dataset, image_dims, self.rescale)

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, int]) -> pl.ArrayInMemoryDataset:
        first = dataset.data[:, image_dims[0]]
        second = dataset.data[:, image_dims[1]]
        sub = first - second
        if self.rescale:
            add = first + second
            add = add.type(torch.float32) if isinstance(add, torch.Tensor) else add.astype(np.float32)
            sub = sub.type(torch.float32) if isinstance(sub, torch.Tensor) else sub.astype(np.float32)
            mask = torch.isclose(add, torch.tensor([0.0])) if isinstance(add, torch.Tensor) else np.isclose(add, 0.0)
            # on purpose test of equal to 0... as only those are relevant for producing zeros
            if (torch.any(mask) if isinstance(first, torch.Tensor) else np.any(mask)):
                add[mask] = utils.EPS
                sub[mask] += utils.EPS
            return pl.ArrayInMemoryDataset(sub / add)
        return pl.ArrayInMemoryDataset(sub)


class _AWEIExtractor(_EOImageExtractor):
    def _compute_feature(self, green, swir1, nir, swir2):
        return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)


class AWEIExtractorModule(EOImageExtractorModule):
    def __init__(self, name: str = 'AWEI'):
        super().__init__(name, ('Green', 'NIR', 'SWIR-1', 'SWIR-2'))

    def _create_extractor(self, dataset: data.Dataset, image_dims: Tuple[int, int, int, int]) -> data.Dataset:
        return _AWEIExtractor(dataset, image_dims)

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, ...]) -> pl.ArrayInMemoryDataset:
        green, swir1, nir, swir2 = image_dims
        data = dataset.data
        return pl.ArrayInMemoryDataset(
            4 * (data[:, green] - data[:, swir1]) - (0.25 * data[:, nir] + 2.75 * data[:, swir2])
        )


class _AWEISHExtractor(_EOImageExtractor):
    def _compute_feature(self, blue, green, nir, swir1, swir2):
        return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2


class AWEISHExtractorModule(EOImageExtractorModule):
    def __init__(self, name: str = 'AWEISH'):
        super().__init__(name, ('Blue', 'Green', 'NIR', 'SWIR-1', 'SWIR-2'))

    def _create_extractor(self, dataset: data.Dataset, image_dims: Tuple[int, int, int, int, int]) -> data.Dataset:
        return _AWEISHExtractor(dataset, image_dims)

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, ...]) -> pl.ArrayInMemoryDataset:
        blue, green, nir, swir1, swir2 = image_dims
        data = dataset.data
        return pl.ArrayInMemoryDataset(
            data[:, blue] + 2.5 * data[:, green] - 1.5 * (data[:, nir] + data[:, swir1]) - 0.25 * data[:, swir2]
        )


class HSVDataset(_EOImageExtractor):
    def __init__(self, wrapped: data.Dataset,
                 image_dims: Tuple[int, int, int]):
        super().__init__(wrapped, image_dims)

    def _compute_feature(self, red, green, blue):
        shape = red.shape
        assert shape == green.shape == blue.shape
        red, green, blue = red.flatten(), blue.flatten(), green.flatten()
        is_torch = isinstance(red, torch.Tensor)
        hue, saturation, value = utils.rgb_to_hsv_torch(red, green, blue) if is_torch else \
            utils.rgb_to_hsv(red, green, blue)
        stack_tup = (hue.reshape(shape), saturation.reshape(shape), value.reshape(shape))
        data = torch.stack(stack_tup) if is_torch else np.stack(stack_tup)
        return data


uint12_max = 2.0 ** 12 - 1

class HSVExtractor(EOImageExtractorModule):
    def __init__(self, name: str, channels=('Red', 'Green', 'Blue'), ):
        super().__init__(name, channels)
        self.multi_output = True

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        res_data, res_meta = super().__call__(dataset, meta)
        channel_names_cat = ', '.join(self.image_dims)
        return res_data, res_meta._replace(channel_names=[f'hue({channel_names_cat})',
                                                          f'saturation({channel_names_cat})',
                                                          f'value({channel_names_cat})'])

    def _create_extractor_not_in_memory(self, meta: pl.Meta, dataset: data.Dataset,
                                        image_dims: Tuple[int, ...]) -> data.Dataset:
        # self.print(meta, 'Detected non-array-dataset for HSV. This is not supported as the value range needs to '
        #                  f'be normalized. Loading channels {str(self.image_dims)} [<=>{str(image_dims)}] into memory.')
        # t = time.time()
        # dataset = pl.ArrayInMemoryDataset(pl._ChannelWhitelistDataset(dataset, list(image_dims)))
        # t = time.time() - t
        # self.print(meta,
        #            f'Loading took {t:.3f}s - {t / len(dataset):.3f}s on average. Performing in-memory extraction.')
        # return self.extract_in_memory(dataset, tuple(map(lambda t: t[0], enumerate(image_dims))))
        return HSVDataset(dataset, image_dims)

    def extract_in_memory(self, dataset: pl.ArrayInMemoryDataset,
                          image_dims: Tuple[int, ...]) -> pl.ArrayInMemoryDataset:
        data = dataset.data
        red, green, blue = data[:, image_dims[0]], data[:, image_dims[1]], data[:, image_dims[2]]
        shape = red.shape
        assert shape == green.shape == blue.shape
        red, green, blue = red.flatten(), blue.flatten(), green.flatten()
        is_torch = isinstance(data, torch.Tensor)
        hue, saturation, value = utils.rgb_to_hsv_torch(red, green, blue) if is_torch \
            else utils.rgb_to_hsv(red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32))
        shape = (shape[0],) + (1,) + shape[1:]
        concat_tup = (hue.reshape(shape), saturation.reshape(shape), value.reshape(shape))
        data = torch.concat(concat_tup, dim=1) if is_torch else np.concatenate(concat_tup, axis=1)
        return pl.ArrayInMemoryDataset(data)


def VV_VH_LinearRatioExtractor(rescale: bool = True, name: Optional[str] = None) -> IndexExtractorModule:
    if name is None:
        name = 'VV-VH lin. Ratio'
    return IndexExtractorModule(name, ('VV', 'VH'), rescale)


# (Green - NIR)/(Green + NIR)
def NDWIExtractor(rescale: bool = True) -> IndexExtractorModule:
    # compare  Modification of normalised difference water index (ndwi) to enhance open water features in remotely
    # sensed imagery
    return IndexExtractorModule('NDWI', ('Green', 'NIR'), rescale)


# (Green - SWIR1)/(Green + SWIR1) also used by Konapala et al.
def MNDWI1Extractor(rescale: bool = True) -> IndexExtractorModule:
    # compare  Modification of normalised difference water index (ndwi) to enhance open water features in remotely
    # sensed imagery
    return IndexExtractorModule('MNDWI1', ('SWIR-1', 'Green'), rescale)


# (Green - SWIR2)/(Green + SWIR2)
def MNDWI2Extractor(rescale: bool = True) -> IndexExtractorModule:
    # compare  Modification of normalised difference water index (ndwi) to enhance open water features in remotely
    # sensed imagery
    return IndexExtractorModule('MNDWI2', ('SWIR-2', 'Green'), rescale)


# Normalized Difference Moisture Index (NIR - SWIR1)/(NIR - SWIR1)
def NDMI1Extractor(rescale: bool = True) -> IndexExtractorModule:
    # compare  Modification of normalised difference water index (ndwi) to enhance open water features in remotely
    # sensed imagery
    return IndexExtractorModule('NDMI1', ('NIR', 'SWIR-1'), rescale)


# Normalized Difference Moisture Index (NIR - SWIR2)/(NIR - SWIR2)
def NDMI2Extractor(rescale: bool = True) -> IndexExtractorModule:
    # compare  Modification of normalised difference water index (ndwi) to enhance open water features in remotely
    # sensed imagery
    return IndexExtractorModule('NDMI2', ('NIR', 'SWIR-2'), rescale)


# (NIR - Red)/(NIR + Red)
def NDVIExtractor(rescale: bool = True) -> IndexExtractorModule:
    # compare "A commentary review on the use of normalized difference vegetation index (NDVI) in the era of popular
    # remote sensing"
    return IndexExtractorModule('NDVI', ('NIR', 'Red'), rescale)


# Implementation courtesy of PyRAT: https://github.com/birgander2/PyRAT/blob/master/pyrat/filter/Filter.py
def median_filter(array: np.ndarray,
                  win: int = 7,
                  type='amplitude') -> np.ndarray:
    """
        Median (speckle) filter.
        :author: Andreas Reigber
    """
    phase = type.lower().strip() != 'amplitude'
    # if not np.iscomplexobj(array) and not phase:
    #     if win == 3:
    #         return utils.median3_filter(array)
    #     elif win == 5:
    #         return utils.median5_filter(array)
    #     elif win == 7:
    #         return utils.median7_filter(array)
    #     elif win == 9:
    #         return utils.median9_filter(array)
    win = [win, win]
    if array.ndim == 3:
        win = [1] + win
    if array.ndim == 4:
        win = [1, 1] + win

    if np.iscomplexobj(array):
        return sp.ndimage.filters.median_filter(array.real, size=win) + 1j * sp.ndimage.filters.median_filter(
            array.imag, size=win)
    elif phase:
        tmp = np.exp(1j * array)
        tmp = sp.ndimage.filters.uniform_filter(tmp.real, win) + 1j * sp.ndimage.filters.uniform_filter(tmp.imag, win)
        return np.angle(tmp)
    else:
        return sp.ndimage.filters.median_filter(array, size=win, mode='mirror')


def boxcar_filter(array: np.ndarray,
                  win: int = 7,
                  type='amplitude') -> np.ndarray:
    """
        Boxcar / Moving average (speckle) filter.
        :author: Andreas Reigber
    """
    phase = type.lower().strip() != 'amplitude'
    win = [win, win]
    array = array.copy()
    if array.ndim == 3:
        win = [1] + win
    if array.ndim == 4:
        win = [1, 1] + win
    array[np.isnan(array)] = 0.0
    if np.iscomplexobj(array):
        return sp.ndimage.filters.uniform_filter(array.real, win) + 1j * sp.ndimage.filters.uniform_filter(
            array.imag, win)
    elif phase:
        tmp = np.exp(1j * array)
        tmp = sp.ndimage.filters.uniform_filter(tmp.real, win) + 1j * sp.ndimage.filters.uniform_filter(tmp.imag,
                                                                                                        win)
        return np.angle(tmp)
    else:
        return sp.ndimage.filters.uniform_filter(array, win, mode='mirror')


# Implementation courtesy of PyRAT: https://github.com/birgander2/PyRAT/blob/master/pyrat/filter/Despeckle.py
def lee_filter(array: np.ndarray,
               win: int = 7,
               looks: float = 4.4) -> np.ndarray:
    """
        Lee's classical speckle filter from 1981. Not the best one...
        :author: Andreas Reigber
        :param array: The image to filter (2D np.ndarray)
        :type array: float
        :param win: The filter window size (default: [7,7])
        :type win: integer
        :param looks=1.0: The effective number of looks of the input image.
        :type looks: float
        :returns: filtered image
        """
    shape = array.shape
    array[np.isnan(array)] = 0.0
    if array.ndim == 3:
        array = np.abs(array)
        span = np.sum(array ** 2, axis=0)
        array = array.reshape((1,) + shape)
    elif array.ndim == 4:
        span = np.abs(utils.trace4d(array))
    else:
        array = np.abs(array)
        span = array ** 2
        array = array.reshape((1, 1,) + shape)

    sig2 = 1.0 / looks
    sfak = 1.0 + sig2
    m2arr = sp.ndimage.filters.uniform_filter(span ** 2, size=win)
    marr = sp.ndimage.filters.uniform_filter(span, size=win)
    vary = (m2arr - marr ** 2).clip(1e-10)
    varx = ((vary - marr ** 2 * sig2) / sfak).clip(0)
    kfac = varx / vary

    out = np.empty_like(array)
    for k in range(array.shape[0]):
        for l in range(array.shape[1]):
            if np.iscomplexobj(array):
                out[k, l, ...] = sp.ndimage.filters.uniform_filter(array[k, l, ...].real, size=win) + \
                                 1j * sp.ndimage.filters.uniform_filter(array[k, l, ...].imag, size=win)
            else:
                out[k, l, ...] = sp.ndimage.filters.uniform_filter(array[k, l, ...], size=win)
            out[k, l, ...] += (array[k, l, ...] - out[k, l, ...]) * kfac
    return np.squeeze(out)


def refined_lee_filter(array: np.ndarray,
                       win: int = 7,
                       looks: float = 4.4,
                       threshold: float = 0.5,
                       original_edge_detector: bool = True):
    """
    Refined Lee speckle filter
    further information:
    J.S.Lee et al.: "Speckle Reduction in Multipolarization Multifrequency SAR Imagery",
    Trans. on Geoscience and Remote Sensing, Vol. 29, No. 4, pp. 535-544, 1991
    :author: Andreas Reigber
    :param array: The image to filter (2D np.ndarray)
    :type array: float
    :param win: The filter window size (default: [7,7]) - range [3, 999]
    :type win: integer
    :param looks=1.0: The effective number of looks of the input image. - range [1.0, 99.0]
    :type looks: float
    :param threshold=0.5: Threshold on which switch to normal Lee filtering - range [0.0, 9.0]
    :type threshold: float
    :param original_edge_detector: The edge detection method used - if true use the original version
    :type method: bool
    :returns: filtered image
    """
    # array = array.copy()
    print('Refined Lee')
    shape = array.shape
    array[np.isnan(array)] = 0.0
    if array.ndim == 3:
        array = np.abs(array)
        span = np.array([np.sum(array[i] ** 2) for i in range(shape[0])])
        array = array.reshape((1,) + shape)
    elif array.ndim == 4:
        span = np.abs(utils.trace4d(array))
    else:
        array = np.abs(array)
        span = array ** 2
        array = array.reshape((1, 1,) + shape)
    # lshape = array.shape[0:2]

    # ---------------------------------------------
    # INIT & SPAN
    # ---------------------------------------------

    sig2 = 1.0 / looks
    sfak = 1.0 + sig2

    # nrx = array.shape[-1]
    #
    # lshape = array.shape[0:-2]
    # if len(lshape) == 2:
    # # span = np.abs(np.trace(array,axis1=0,axis2=1))
    #     span = np.abs(array[0, 0, ...] + array[1, 1, ...] + array[2, 2, ...])
    # else:
    #     logging.error("Data not in matrix form")

    # ---------------------------------------------
    # TURNING BOX
    # ---------------------------------------------

    cbox = np.zeros((9, win, win), dtype=np.float32)
    chbox = np.zeros((win, win), dtype=np.float32)
    chbox[0:win // 2 + 1, :] = 1
    cvbox = np.zeros((win, win), dtype=np.float32)
    for k in range(win):
        cvbox[k, 0:k + 1] = 1

    cbox[0, ...] = np.rot90(chbox, 3)
    cbox[1, ...] = np.rot90(cvbox, 1)
    cbox[2, ...] = np.rot90(chbox, 2)
    cbox[3, ...] = np.rot90(cvbox, 0)
    cbox[4, ...] = np.rot90(chbox, 1)
    cbox[5, ...] = np.rot90(cvbox, 3)
    cbox[6, ...] = np.rot90(chbox, 0)
    cbox[7, ...] = np.rot90(cvbox, 2)
    for k in range(win // 2 + 1):
        for l in range(win // 2 - k, win // 2 + k + 1):
            cbox[8, k:win - k, l] = 1

    for k in range(9):
        cbox[k, ...] /= np.sum(cbox[k, ...])

    ampf1 = np.empty((9,) + span.shape)
    ampf2 = np.empty((9,) + span.shape)
    for k in range(9):
        ampf1[k, ...] = spimg.filters.correlate(span ** 2, cbox[k, ...])
        ampf2[k, ...] = spimg.filters.correlate(span, cbox[k, ...]) ** 2

    # ---------------------------------------------
    # GRADIENT ESTIMATION
    # ---------------------------------------------
    np.seterr(divide='ignore', invalid='ignore')

    if original_edge_detector:
        xs = [+2, +2, 0, -2, -2, -2, 0, +2]
        ys = [0, +2, +2, +2, 0, -2, -2, -2]
        samp = sp.ndimage.filters.uniform_filter(span, win // 2)
        grad = np.empty((8,) + span.shape)
        for k in range(8):
            grad[k, ...] = np.abs(np.roll(np.roll(samp, ys[k], axis=0), xs[k], axis=1) / samp - 1.0)
        magni = np.amax(grad, axis=0)
        direc = np.argmax(grad, axis=0)
        direc[magni < threshold] = 8
    else:  # 'cov' method
        grad = np.empty((8,) + span.shape)
        for k in range(8):
            grad[k, ...] = np.abs((ampf1[k, ...] - ampf2[k, ...]) / ampf2[k, ...])
        direc = np.argmin(grad, axis=0)

    np.seterr(divide='warn', invalid='warn')
    # ---------------------------------------------
    # FILTERING
    # ---------------------------------------------
    out = np.empty_like(array)
    dbox = np.zeros((1, 1) + (win, win))
    for l in range(9):
        grad = ampf1[l, ...]
        mamp = ampf2[l, ...]
        dbox[0, 0, ...] = cbox[l, ...]

        vary = (grad - mamp).clip(1e-10)
        varx = ((vary - mamp * sig2) / sfak).clip(0)
        kfac = varx / vary
        if np.iscomplexobj(array):
            mamp = spimg.filters.correlate(array.real, dbox) + 1j * sp.ndimage.filters.convolve(array.imag,
                                                                                                dbox)
        else:
            mamp = sp.ndimage.filters.correlate(array, dbox)
        idx = np.argwhere(direc == l)
        out[:, :, idx[:, 0], idx[:, 1]] = (mamp + (array - mamp) * kfac)[:, :, idx[:, 0], idx[:, 1]]

    return out


@utils.njit()
def _execute_improved_lee_sigma(array: np.ndarray, span: np.ndarray,
                                bounds: Tuple[int, int] = (1, 2), thres: float = 10.0,
                                looks: float = 4.4,
                                win: Tuple[int, int] = (7, 7),
                                newsig: float = 0.9) -> np.ndarray:
    """
            Lee's improved sigma speckle filter. Fast implementation in Cython.
            J.S. Lee et al.: Improved Sigma Filter for Speckle Filtering of SAR Imagery.
            IEEE Transactions on Geoscience and Remote Sensing, Vol. 47, No.1, pp. 202-213, 2009']
            :author: Andreas Reigber
    """
    out: np.ndarray = np.zeros_like(array,
                                    dtype=np.float32)  # originally this should be the dtype, however I don't think we need that
    sig2: float = 1.0 / looks
    sfak: float = 1.0 + sig2
    nsig2: float = newsig
    nsfak: float = 1.0 + nsig2
    # xtilde: float = 0.0
    nv: int = array.shape[0]
    nz: int = array.shape[1]
    ny: int = array.shape[2]
    nx: int = array.shape[3]
    ym: int = win[0] // 2
    xm: int = win[1] // 2
    # norm: int = win[0] * win[1]
    # k, l, x, y, v, z = 0, 0, 0, 0, 0, 0
    # m2arr, marr, vary, varx, kfac, i1, i2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    res = np.zeros((nv, nz),
                   dtype=np.float32)  # originally this should be the dtype, however I don't think we need that

    n = 0
    for k in range(ym, ny - ym):
        for l in range(xm, nx - xm):
            m2arr = 0.0
            marr = 0.0
            n = 0
            for y in range(-1, 2):  # check 3x3 neighbourhood
                for x in range(-1, 2):
                    m2arr += span[k + y, l + x] ** 2
                    marr += span[k + y, l + x]
                    if span[k + y, l + x] > thres:
                        n += 1
            if n >= 6:  # keep all point targets
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        for v in range(nv):
                            for z in range(nz):
                                if span[k + y, l + x] > thres:
                                    out[v, z, k + y, l + x] = array[v, z, k + y, l + x]

            if out[0, 0, k, l] == 0.0:  # no point target, also not prior
                m2arr /= 9.0
                marr /= 9.0
                vary = (m2arr - marr ** 2)
                if vary < 1e-10: vary = 1e-10
                varx = ((vary - marr ** 2 * sig2) / sfak)
                if varx < 0: varx = 0
                kfac = varx / vary

                xtilde = (span[k, l] - marr) * kfac + marr

                i1 = xtilde * bounds[0]
                i2 = xtilde * bounds[1]
                m2arr = 0.0
                marr = 0.0
                n = 0
                for v in range(nv):
                    for z in range(nz):
                        res[v, z] = 0.0

                for y in range(-ym, ym + 1):
                    for x in range(-xm, xm + 1):
                        if span[k + y, l + x] > i1 and span[k + y, l + x] < i2:
                            m2arr += span[k + y, l + x] ** 2
                            marr += span[k + y, l + x]
                            n += 1
                            for v in range(nv):
                                for z in range(nz):
                                    res[v, z] = res[v, z] + array[v, z, k + y, l + x]
                if n == 0:
                    for v in range(nv):
                        for z in range(nz):
                            out[v, z, k, l] = 0.0
                else:
                    m2arr /= n
                    marr /= n
                    vary = (m2arr - marr ** 2)
                    if vary < 1e-10: vary = 1e-10
                    varx = ((vary - marr ** 2 * nsig2) / nsfak)
                    if varx < 0.0: varx = 0.0
                    kfac = varx / vary
                    for v in range(nv):
                        for z in range(nz):
                            out[v, z, k, l] = (array[v, z, k, l] - res[v, z] / n) * kfac + res[v, z] / n
    array = np.asarray(out)
    return array


def improved_lee_sigma(array, looks=4.4, win=(7, 7), sigma=0.9, perc: float = 98.0,
                       type='amplitude'):  # , dtype=np.float32):
    def estimate_percentile(perc=98.0):
        if array.ndim == 3:  # polarimetric vector
            if np.iscomplexobj(array) or type == "amplitude":
                span = np.sum(np.abs(array) ** 2, axis=0)
            else:
                span = np.sum(np.abs(array), axis=0)
        elif array.ndim == 4:  # covariance data
            span = np.abs(np.trace(array, axis1=0, axis2=1))
        else:  # single channel data
            if np.iscomplexobj(array) or type == "amplitude":
                span = np.abs(array) ** 2
            else:
                span = np.abs(array)
        return np.percentile(span, perc)

    # @numba.jit(nopython=True)
    def meanpdf(i, looks=1.0) -> float:
        if i < 0.0:
            return 0.0
        else:
            return ((looks ** looks) * (i ** (looks - 1.0))) / special.gamma(looks) * np.exp(-looks * i) * i

    # @numba.jit(nopython=True)
    def specklepdf(i, looks=1.0) -> float:
        if i < 0.0:
            return 0.0
        else:
            return ((looks ** looks) * (i ** (looks - 1.0))) / special.gamma(looks) * np.exp(-looks * i)

    # @numba.jit(nopython=True)
    def sigpdf(i, looks=1.0) -> float:
        return (i - 1.0) ** 2 * specklepdf(i, looks)

    # @numba.jit(nopython=True)
    def newsig(i1, i2, sigrng=0.9, looks=1.0) -> float:
        return 1 / sigrng * integrate.quad(sigpdf, i1, i2, args=(looks,))[0]

    # @numba.jit(nopython=True)
    def sigmarange(i1, i2, looks=1.0) -> np.ndarray:
        return np.clip(integrate.quad(specklepdf, i1, i2, args=(looks,))[0], 1e-10, 1.0)

    # @numba.jit(nopython=True)
    def intmean(i1, i2, looks=1.0) -> float:
        return 1.0 / sigmarange(i1, i2, looks) * integrate.quad(meanpdf, i1, i2, args=(looks,))[0]

    # @numba.jit(nopython=True)
    def optf(i, looks, sigr) -> float:
        return (sigmarange(i[0], i[1], looks) - sigr) ** 2 + (intmean(i[0], i[1], looks) - 1.0) ** 2

    if array.ndim == 3:  # polarimetric vector
        if np.iscomplexobj(array) or type == "amplitude":
            array = np.abs(array) ** 2
            # type = "amplitude"
        else:
            array = np.abs(array)
        span = np.sum(array, axis=0)
        array = np.expand_dims(array, axis=0)  # array[np.newaxis, ...]
        type = "amplitude"
    elif array.ndim == 4:  # covariance data
        span = np.abs(np.trace(array))
        type = "intensity"
    else:  # single channel data
        if np.iscomplexobj(array) or type == "amplitude":
            array = np.abs(array) ** 2
            type = "amplitude"
        else:
            array = np.abs(array)
        span = array.copy()
        array = array[np.newaxis, np.newaxis, ...]

    # this originates from the pre-processing code in the run method
    # https://github.com/birgander2/PyRAT/blob/b4236269c329d6941d01d1cd9a8689776650ff4e/pyrat/filter/Despeckle.py#L657
    bounds = opt.fmin(optf, [0.5, 2.0], args=(looks, sigma), disp=False)  # calc sigma bounds
    newsig = newsig(bounds[0], bounds[1], sigrng=sigma, looks=looks)  # calc new stddev
    thres = np.mean(estimate_percentile(perc))

    # and now that the paramaters are there, we're back in leeimproved
    array = _execute_improved_lee_sigma(array, span, bounds, thres, looks, win, newsig)
    array[~np.isfinite(array)] = 0.0
    if type == "amplitude":
        array[array < 0] = 0.0
        array = np.sqrt(array)
    return np.squeeze(array)


FILTER_ALIASES = {
    'median': ['median'],
    'boxcar': ['boxcar', 'mean'],
    'lee': ['lee', 'lee_sigma'],
    'lee_improved': ['lee_improved', 'improved_sigma_lee', 'improved_sigma']
}
FILTER_METHODS = list(FILTER_ALIASES.keys())


def get_filter_fun(method: str):
    method = method.lower().strip()
    if method == 'lee_enhanced' or method == 'refined_lee':
        raise NotImplementedError('Error somewhere in the reference implementation of lee-enhanced.')
        # return refined_lee_filter
    elif method in FILTER_ALIASES['lee_improved']:
        return improved_lee_sigma
    elif method in FILTER_ALIASES['lee']:
        return lee_filter
    elif method in FILTER_ALIASES['boxcar']:
        return boxcar_filter
    elif method in FILTER_ALIASES['median']:
        return median_filter
    else:
        raise ValueError('Unknown method: ' + method)


class SARFilterDataset(pl.TransformerDataset):

    def __init__(self, wrapped: data.Dataset, method: str, params: Dict[str, Any]):
        super().__init__(wrapped)
        self.method = method
        self.filter_fun = get_filter_fun(method)
        self.params = params

    def _transform(self, data):
        data = self.filter_fun(data, **self.params)
        return data


class SARFilterModule(PreProcessingModule):
    def __init__(self, method: str = 'refined_lee', params: Optional[Dict[str, Any]] = None):
        self.method = method
        self.params = {} if params is None else params
        self.filter_fun = get_filter_fun(method)

    def get_params_as_dict(self) -> Dict[str, Any]:
        def_params = serialisation.get_function_arguments(self.filter_fun, {'array'})
        def_params.update(self.params)
        return {
            'method': self.method,
            'params': def_params
        }

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        self.print(meta,
                   f'Creating SARFilterDataset wrapper for filter method {self.method} with params {str(self.params)}.')
        if len(meta.channel_names) > 2 or (meta.channel_names[0] not in ['VV', 'VH']) or \
                (meta.channel_names[1] not in ['VV', 'VH']):
            self.print(meta,
                       f'Potentially incorrect usage of SAR-Filter detected. Provided channel names are {meta.channel_names}.'
                       f'However only ["VV", "VH"] was expected. Proceeding anyway.')
        dataset = SARFilterDataset(dataset, self.method, self.params)
        self.print(meta, f'Successfully created SARFilterDataset wrapper for filter method {self.method}.')
        return dataset, meta


StandardizationSave = Dict[str, Dict[Optional[str], List[float]]]


@utils.njit()
def jit_mean_std(data: np.ndarray, means_and_scales: np.ndarray) -> np.ndarray:
    res = np.empty_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        res[i] = (data[i] - means_and_scales[i, 0]) / means_and_scales[i, 1]
    return res


class StandardizationDataset(pl.TransformerDataset):

    def __init__(self, wrapped: data.Dataset, means_and_scales: List[List[float]]):
        super().__init__(wrapped)
        self.means_and_scales = np.array(means_and_scales)
        assert self.means_and_scales.ndim == 2 and self.means_and_scales.shape[1] == 2

    def _transform(self, data):
        assert data.shape[0] == self.means_and_scales.shape[0], \
            f'Expected {data.shape[0]} == {self.means_and_scales.shape[0]} but got {data.shape} and {self.means_and_scales.shape}'
        return jit_mean_std(data, self.means_and_scales)


class StandardizationModule(PreProcessingModule):
    def __init__(self, standard_file: str = 'standard.json', filter_method: Optional[str] = None):
        self.standard_file = standard_file
        self.filter_method = filter_method

    def _load_standard_values(self, meta: pl.Meta) -> StandardizationSave:
        standard_file = path.abspath(self.standard_file)
        if path.exists(standard_file):
            self.print(meta, f'Loading standard file at {standard_file}.')
            with open(standard_file, 'r') as fd:
                loaded = json.load(fd)
                return {cn: {(None if k == 'null' else k.replace('_null', 'null')): val for k, val in filters.items()}
                        for cn, filters in loaded.items()}
        else:
            self.print(meta, f'No standard file found. Returning empty dictionary.')
        return {}

    def _save_standard_values(self, meta: pl.Meta, to_save: StandardizationSave):
        standard_file = path.abspath(self.standard_file)
        self.print(meta, f'Saving standard file at {standard_file}.')
        with open(standard_file, 'w') as fd:
            json.dump(to_save, fd, indent=4)

    def _check_parameters_known(self, meta: pl.Meta, known_standard: StandardizationSave) -> List[str]:
        is_filtered = self.filter_method is not None
        is_train = meta.split == SPLIT_TRAIN
        present_channels = []
        for cn in meta.channel_names:
            if is_train:
                if cn in known_standard and not isinstance(known_standard[cn], dict):
                    raise ValueError(f'Loaded illegal types for {cn}'
                                     f'{f"and filter method {self.filter_method}" if is_filtered else ""}!')
                elif cn in known_standard and isinstance(known_standard[cn], dict) and self.filter_method in \
                        known_standard[cn]:
                    present_channels.append(cn)
            else:
                if cn not in known_standard or self.filter_method not in known_standard[cn]:
                    raise ValueError(f'Standardization parameters for {cn} '
                                     f'{f"and filter method {self.filter_method}" if is_filtered else ""}'
                                     f'are unknown and cannot be calculated on split {meta.split}.')
                elif not isinstance(known_standard[cn], dict):
                    raise ValueError(f'Loaded illegal types for {cn}'
                                     f'{f"and filter method {self.filter_method}" if is_filtered else ""}!')
                else:
                    present_channels.append(cn)
        return present_channels

    @staticmethod
    def _calc_mean_scale(data: np.ndarray) -> Tuple[float, float]:
        # calculate using high-precision arithmetic
        mean = np.nanmean(data.flatten(), dtype=np.float64)
        std = np.nanstd(data.flatten(), dtype=np.float64)
        return float(mean), (1.0 if -utils.EPS < std < utils.EPS else std)

    def _perform_standardization(self, dataset: data.Dataset, meta: pl.Meta,
                                 known_standard: StandardizationSave,
                                 present_channels: List[str]) \
            -> Tuple[StandardizationSave, data.Dataset]:
        if not any(map(lambda cn: cn not in present_channels, meta.channel_names)):
            self.print(meta, 'All required standardization parameters are already calculated. Returning wrapper')
            return known_standard, StandardizationDataset(dataset, [known_standard[cn][self.filter_method]
                                                                    for cn in meta.channel_names])
        self.print(meta, f'Calculating missing standardization parameters for channels '
                         f'{str([cn for cn in meta.channel_names if cn not in present_channels])}')
        if not isinstance(dataset, pl.ArrayInMemoryDataset):
            self.print(meta, f'However this requires the data to be loaded into memory as an array.')
            t = time.time()
            dataset = pl.ArrayInMemoryDataset(dataset)
            self.print(meta, f'Loading into an in-memory array took {time.time() - t:.3f}s.')
        if isinstance(dataset.data, torch.Tensor):
            dataset.data = utils.torch_as_np(dataset.data)
        # copy to allow for a change check
        known_standard = {cn: {fm: ls.copy() for fm, ls in fm_map.items()} for cn, fm_map in known_standard.items()}
        data = dataset.data
        t_complete = time.time()
        for i, cn in enumerate(meta.channel_names):
            if cn not in present_channels:
                self.print(meta, f'Calculating mean and scale for channel {cn} and filter method {self.filter_method}.')
                t = time.time()
                if cn not in known_standard:
                    known_standard[cn] = {}
                known_standard[cn][self.filter_method] = list(self._calc_mean_scale(data[:, i]))
                self.print(meta, f'Calculation took {time.time() - t:.3f}s.')
            self.print(meta, f'Applying mean and scale for channel {cn} and filter method {self.filter_method}.')
            mean, scale = known_standard[cn][self.filter_method]
            data[:, i] = (data[:, i] - mean) / scale
        self.print(meta, f'Calculation took {time.time() - t_complete:.3f}s in total.')
        return known_standard, dataset

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        dataset, meta = pl.check_meta_and_data(dataset, meta)
        self.print(meta, f'Performing standardization for channels {str(meta.channel_names)}.')
        t = time.time()
        known_standard = self._load_standard_values(meta)
        self.print(meta, f'Loading standardization took {time.time() - t:.3f}s.')
        present_channels = self._check_parameters_known(meta, known_standard)
        new_standard, dataset = self._perform_standardization(dataset, meta, known_standard, present_channels)
        if new_standard != known_standard:
            self._save_standard_values(meta, new_standard)
            self.print(meta, f'Save completed successfully. Returning.')
        else:
            self.print(meta, f'As the standardization info did not change, no save was deemed necessary. Returning.')
        return dataset, meta


# Constructs the tested feature spaces for the hand labeled dataset
# split should be one of {'bolivia', 'test', 'train', 'valid'}
def construct_feature_spaces(base_folder: str, split: str, add_optical=False, use_numpy=True, in_memory=True,
                             filter_method: Optional[Union[str, Tuple[str], Tuple[str, Dict[str, Any]]]] = None,
                             use_datadings: bool = True,
                             standard_file: Optional[str] = 'standard.json',
                             range_file: str = 'range.json', as_array: bool = False,
                             allow_lin_ratio: bool = False, allow_sar_hsv: bool = False) \
        -> List[Tuple[str, pl.Pipeline]]:
    SAR_VARIANTS = [
        ('SAR', True, [], []),  # 'VV-VH lin. Ratio'
        # , #
        #
    ]
    if allow_lin_ratio:
        SAR_VARIANTS.append(('SAR+R', True, [VV_VH_LinearRatioExtractor()], []))
    if allow_sar_hsv:
        SAR_VARIANTS.append(('HSV(SAR+R)', True, [VV_VH_LinearRatioExtractor()],
                             [pl.WhitelistModule(['VV-VH lin. Ratio', 'VV', 'VH']),
                              RangeClippingNormalizationModule(range_file),
                              HSVExtractor('hsv', channels=('VV', 'VV-VH lin. Ratio', 'VH'))]))
    OPTICAL_VARIANTS = [
        # optical feature space as used by L. Landuyt - this excludes all 60m res band
        ('OPT', False, [], [pl.BlacklistModule(['Coastal', 'Water Vapor', 'Cirrus'])]),
        # o3 feature space as used by L. Landuyt
        ('O3', False, [], [pl.WhitelistModule(['Red', 'NIR', 'SWIR-2'])]),
        # all
        ('S2', False, [], []),
        # RGB
        ('RGB', False, [], [pl.WhitelistModule(['Red', 'Green', 'Blue'])]),
        # RGB
        ('RGBN', False, [], [pl.WhitelistModule(['Red', 'Green', 'Blue', 'NIR'])]),
        # HSV from RGB
        ('HSV(RGB)', False, [], [pl.WhitelistModule(['Red', 'Green', 'Blue']),
                                 RangeClippingNormalizationModule(range_file),
                                 HSVExtractor('hsv')]),
        # HSV but this time from Red NIR and SWIR-2 bands - this is what Konpala et al. call HSV
        ('HSV(O3)', False, [], [pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                RangeClippingNormalizationModule(range_file),
                                HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))]),
        # cNDWI Feature space as by Konpala et al.
        ('cNDWI', False, [NDWIExtractor(), MNDWI1Extractor()], [pl.WhitelistModule(('NDWI', 'MNDWI1'))]),
        # cAWEISH Feature space as by Konpala et al.
        ('cAWEI', False, [AWEIExtractorModule(), AWEISHExtractorModule()], [pl.WhitelistModule(('AWEI', 'AWEISH'))]),
        # cAWEISH Feature space as by Konpala et al.
        ('cAWEI+cNDWI', False, [AWEIExtractorModule(), AWEISHExtractorModule(), NDWIExtractor(), MNDWI1Extractor()],
         [pl.WhitelistModule(('AWEI', 'AWEISH', 'NDWI', 'MNDWI1'))]),
        # HSVFeature space as by Konpala et al. + the Normalized Difference Vegetation Index
        #('cNDWI+NDVI', False, [NDWIExtractor(), MNDWI1Extractor(), NDVIExtractor()],
        # [pl.WhitelistModule(('NDWI', 'MNDWI1', 'NDVI'))]),
        ('HSV(O3)+cAWEI+cNDWI', False,
         [pl.SequenceModule([pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                RangeClippingNormalizationModule(range_file),
                                HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))]),
          AWEIExtractorModule(), AWEISHExtractorModule(), NDWIExtractor(), MNDWI1Extractor()],
         [pl.WhitelistModule(('AWEI', 'AWEISH', 'NDWI', 'MNDWI1',  'hue(SWIR-2, NIR, Red)',
                              'saturation(SWIR-2, NIR, Red)', 'value(SWIR-2, NIR, Red)'))]),

        # # now some of my own Ideas for feature-spaces
        # # could potentially allow the algorithm to recognize clouds
        # 'opt-Cir': MergingDatasetWrapper((ChannelBlacklist(
        #     sen_data.read_s2_images(s2_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy),(0, 9)),
        #     sen_data.read_labels_only(label_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy))),
        # # and for completeness also all the broader IR channels
        # 'RGB+IR': lambda: MergingDatasetWrapper((ChannelWhitelist(
        #     sen_data.read_s2_images(s2_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy),
        #     (1, 2, 3, 7, 11, 12)),
        #     sen_data.read_labels_only(label_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy))),
        # # of course we have to check full spectrum as well
        # 'full-opt': lambda: sen_data.read_s2_images(s2_path, label_path, split_csv_file=split_path_no_end + '_s2.csv',
        #                                             as_numpy=use_numpy),
    ]
    DATASET_VARIANTS = finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                   filter_method, standard_file, use_numpy, [SAR_VARIANTS]))
    if add_optical:
        DATASET_VARIANTS.extend(finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                            filter_method, standard_file, use_numpy,
                                                            [OPTICAL_VARIANTS])))
    DATASET_VARIANTS.extend(finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                        filter_method, standard_file, use_numpy,
                                                        [SAR_VARIANTS, OPTICAL_VARIANTS])))
    return DATASET_VARIANTS


def finish_build(incomplete: List[Tuple[str, pl.List[pl.Pipeline]]]) -> List[Tuple[str, pl.Pipeline]]:
    return [(name, (pl.DistributorModule(ls) if len(ls) > 1 else ls[0])) for name, ls in incomplete]


def build_combined(use_datadings, base_folder, split, in_memory, as_array, filter_method, standard_file, use_numpy,
                   source_tuples: List[List[Tuple[str, bool, List[pl.Pipeline], List[pl.Pipeline]]]],
                   verbose: bool =True) \
        -> List[Tuple[str, List[pl.Pipeline]]]:
    if not source_tuples:
        raise ValueError('No input data given!')
    cur: List[Tuple[str, bool, List[pl.Pipeline], List[pl.Pipeline]]] = source_tuples[0]
    if len(source_tuples) == 1:
        if verbose:
            print(f'Only one set of constructors given. Returning feature-spaces {", ".join([n[0] for n in cur])}')
        return [(t[0], [create_module(use_datadings, base_folder, split, in_memory, as_array, filter_method,
                                      standard_file, use_numpy, t[1:])]) for t in cur]

    res: List[Tuple[str, List[pl.Pipeline]]] = []
    sub: List[Tuple[str, List[pl.Pipeline]]] = build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                              filter_method, standard_file, use_numpy,
                                                              source_tuples[1:])
    for t_cur in cur:
        for sub_name, ls in sub:
            if verbose:
                print(f'Combining {t_cur[0]} and {sub_name} to the feature-space {t_cur[0]}_{sub_name}')
            ls: List[pl.Pipeline] = ls.copy()
            ls.append(create_module(use_datadings, base_folder, split, in_memory, as_array, filter_method,
                                    standard_file, use_numpy, t_cur[1:]))
            res.append((t_cur[0] + '_' + sub_name, ls))
    return res


def create_module(use_datadings, base_folder, split, in_memory, as_array, filter_method, standard_file, use_numpy,
                  parallel_features: Tuple[bool, List[pl.Pipeline], List[pl.Pipeline]]) -> pl.Pipeline:
    is_s1, extractors, finisher = parallel_features
    type = ('S1' if is_s1 else 'S2')
    if use_datadings:
        seq: List[pl.Pipeline] = [Sen1Floods11DataDingsDatasource(base_folder, type, split, True,
                                                                  in_memory=in_memory, as_array=as_array)]
    else:
        seq: List[pl.Pipeline] = [
            Sen1Floods11DataSource(base_folder, type, split, use_numpy=True, as_array=as_array)]
    if is_s1 and filter_method is not None:
        fm = (filter_method,) if isinstance(filter_method, str) else tuple(filter_method)
        seq.append(SARFilterModule(*fm))
    else:
        fm = None
    if extractors:
        seq.append(pl.DistributorModule(extractors, keep_source=True))
    if finisher:
        seq.extend(finisher)
    if standard_file is not None:
        seq.append(StandardizationModule(standard_file, None if fm is None else fm[0]))
    if not use_numpy:
        seq.append(pl.TorchConversionModule())
    if in_memory:
        seq.append(pl.InMemoryModule() if as_array else pl.ShapelessInMemoryModule())
    if len(seq) > 1:
        return pl.SequenceModule(seq)
    return seq[0]


def default_feature_space_construction(data_folder: str, filters: Optional[List[str]], use_datadings: bool,
                                       eval_split: str = SPLIT_VALIDATION, in_memory: bool = True,
                                       train_split: str = SPLIT_TRAIN) \
        -> Tuple[Dict[str, Dict[str, Tuple[pl.Pipeline, pl.Pipeline]]], Dict[str, List[str]], List[str], List[str]]:
    print('Constructing feature spaces.')
    cons_by_filter: Dict[str, Dict[str, Tuple[Any, Any]]] = {}
    all_filters = FILTER_METHODS.copy()
    all_filters.append(None)
    if filters is not None:
        flattened_aliases = set(funct.reduce(lambda l1, l2: l1 + l2, FILTER_ALIASES.values()))
        for filter_method in filters:
            if filter_method not in flattened_aliases:
                raise ValueError(f'Invalid filter {filter_method}. Expected one of {str(all_filters)}.')
        selected_filters = filters
    else:
        selected_filters = [None]
    f_names = {}

    for filter_method in all_filters:
        train_cons = construct_feature_spaces(data_folder, split=train_split, in_memory=in_memory,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=in_memory)
        valid_cons = construct_feature_spaces(data_folder, split=eval_split, in_memory=in_memory,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=in_memory)
        cur_f_names = [t[0] for t in
                       train_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: (v1, v2) for (k1, v1), (k2, v2) in zip(train_cons, valid_cons)}
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')
    return cons_by_filter, f_names, all_filters, selected_filters

class StatsExtractorModule(mpl.MultiTransformerModule):
    def __init__(self, label_criterion: mpl.SelectionCriterion,
                 intensity_criterion: Optional[mpl.SelectionCriterion],
                 stats_of_interest: List[str],
                 separator: str = '-',
                 dataset_result_name: Optional[str] = 'zonal_stats',
                 label_dataset_result_name: Optional[str] = 'label',
                 label_channel_result_name: Optional[str] = 'label',
                 keep_label: bool = False,
                 keep_intensity: bool = False,
                 no_stat_suffix: bool = False,
                 cache: bool = True):
        super().__init__()
        self.label_criterion = label_criterion
        self.intensity_criterion = intensity_criterion
        include_label = 'label' in stats_of_interest
        if include_label:
            stats_of_interest.remove('label')
        if len(stats_of_interest) > 1 and no_stat_suffix:
            raise ValueError('Statistic suffixes are required to correctly identify the individual statistics due to '
                             'more than one statistic being requested!')
        self.stats_of_interest = stats_of_interest
        self.separator: str = separator
        self.include_label: bool = include_label
        self.dataset_result_name: str = dataset_result_name
        self.label_dataset_result_name: str = label_dataset_result_name
        self.label_channel_result_name = label_channel_result_name
        self.keep_label = keep_label
        self.keep_intensity = keep_intensity
        self.no_stat_suffix = no_stat_suffix
        self.cache = cache

    def _create_stats_extract(self, label_dataset, intensity_dataset, stats_of_interest: List[str]) -> data.Dataset:
        raise NotImplementedError

    def __call__(self, summary: mpl.Summary) -> mpl.Summary:
        self.print(summary, f'Generating wrapper to extract the following zonal statistics {self.stats_of_interest}.')
        label_dataset, label_meta = summary.by_criterion(self.label_criterion, delete=not self.keep_label)
        intensity_dataset, intensity_meta = summary.by_criterion(self.intensity_criterion,
                                                                 delete=not self.keep_intensity) \
            if self.intensity_criterion is not None else (None, None)
        if self.stats_of_interest:
            intensity_meta = intensity_meta if intensity_meta is not None else label_meta
            assert intensity_meta is not None, 'Either a label data set or an intensity dataset is required for stats extraction!'
            intensity_dataset = self._create_stats_extract(label_dataset, intensity_dataset, self.stats_of_interest)
            channel_names = [cn + ('' if self.no_stat_suffix else self.separator + stat)
                             for stat in self.stats_of_interest
                             for cn in intensity_meta.channel_names]
            intensity_meta = intensity_meta._replace(channel_names=channel_names)
            summary = summary.add(intensity_dataset, intensity_meta, name=self.dataset_result_name)
        if self.include_label:
            label_meta = label_meta._replace(channel_names=[self.label_channel_result_name])
            summary = summary.add(self._create_stats_extract(label_dataset, None, ['label']), label_meta,
                                  name=self.label_dataset_result_name)
        self.print(summary, 'Successfully created zonal-stats wrapper featuring output channels ' +
                   f'{str(intensity_meta.channel_names) if intensity_meta is not None else "None"} and output'
                   f' dataset name {self.dataset_result_name}'
                   + ((f'and label channels {str(label_meta.channel_names)} with output dataset name '
                       f'{self.label_dataset_result_name}.') if self.include_label else '.'))
        return summary


# Extracts zonal statistics (some value for each label in the individual images)
# accepts images of the form (C, H, W)
# returns results of the form (num_samples, num_features)
class _ZonalStatsExtractor(mpl.MultiTransformerDataset):
    def __init__(self, label_dataset: data.Dataset,
                 intensity_dataset: Optional[data.Dataset],
                 stats_of_interest: List[str],
                 cache: bool = True):
        super().__init__(label_dataset)
        self.intensity_dataset = intensity_dataset
        self.stats_of_interest = list(stats_of_interest)
        self.extra_properties = [utils.property_function(stat) for stat in stats_of_interest if
                                 stat in utils._EXTRA_STATS]
        self.cache = cache

    def _transform(self, data, index):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        label_image = np.squeeze(data)
        intensity_image = None if self.intensity_dataset is None else self.intensity_dataset[index]
        intensity_image = utils.move_channel_to_position(intensity_image, len(intensity_image.shape), 0) \
            if intensity_image is not None else None
        # avoid zero background value
        offset = -np.min(label_image) + 1
        label_image = label_image.copy() + offset
        # note that this will keep the order of stats_of_interest as dicts are nowadays ordered in python
        if intensity_image.ndim > label_image.ndim and intensity_image.shape[-1] == 1:
            intensity_image = np.squeeze(intensity_image)
        tab = measure.regionprops_table(label_image.astype(np.int), intensity_image, properties=self.stats_of_interest,
                                        separator='-', cache=self.cache, extra_properties=self.extra_properties)
        # TODO fix this for 2D properties... These will mess up now!
        res = np.array([(val - offset if name == 'label' else val) for name, val in tab.items()])
        return res
        # return np.array([[getattr(p, stat) for p in props] for stat in self.stats_of_interest])


# WARNING: this works as of now only for 1D properties
class PerImageZonalStatsExtractorModule(StatsExtractorModule):

    def _create_stats_extract(self, label_dataset, intensity_dataset, stats_of_interest) -> data.Dataset:
        return _ZonalStatsExtractor(label_dataset, intensity_dataset, stats_of_interest, self.cache)


class _StatsExtractor(mpl.MultiTransformerDataset):
    def __init__(self, label_dataset: data.Dataset,
                 intensity_dataset: Optional[data.Dataset],
                 stats_of_interest: List[str],
                 cache: bool = True):
        super().__init__(label_dataset)
        self.intensity_dataset = intensity_dataset
        self.stats_of_interest = stats_of_interest.copy()
        self.extra_properties = [utils.property_function(stat) for stat in stats_of_interest if
                                 stat in utils._EXTRA_STATS]
        self.cache = cache

    def get_statistic(self, statistic: str) -> Callable:
        if statistic == 'mean_intensity':
            def mean_intensity(rm, intensity, label):
                return np.nanmean(intensity[rm].T, axis=1)

            return mean_intensity
        elif statistic == 'label':
            def label(rm, intensity, label):
                return np.array([label])

            return label
        else:
            fun = utils.property_function(statistic)
            return lambda rm, intensity, label: fun(rm, intensity)

    def _transform(self, data, index):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        label_image = np.squeeze(data)
        intensity_image = None if self.intensity_dataset is None else self.intensity_dataset[index]
        intensity_image = utils.move_channel_to_position(intensity_image, len(intensity_image.shape), 0) \
            if intensity_image is not None else None
        masks = [(label_image == val, val) for val in np.unique(label_image)]
        stats = [self.get_statistic(statistic) for statistic in self.stats_of_interest]
        res = np.array([stat(mask, intensity_image, label) for mask, label in masks for stat in stats])
        res = res.reshape((len(masks), -1))
        res = utils.move_channel_to_position(res, target_position=0)
        return res  # np.transpose(res)
        # return np.array([[getattr(p, stat) for p in props] for stat in self.stats_of_interest])


class PerDataPointStatsExtractorModule(StatsExtractorModule):
    def _create_stats_extract(self, label_dataset, intensity_dataset, stats_of_interest) -> data.Dataset:
        return _StatsExtractor(label_dataset, intensity_dataset, stats_of_interest, self.cache)


WindowType = Union[int, Tuple[int, ...]]


def to_tuple_window(a: WindowType, shape: Union[Tuple, torch.Size]) -> Tuple[int, ...]:
    return a if isinstance(a, tuple) else tuple(a for _ in range(len(shape) - 2))


def conc_iter_gen(iter_in: Iterable[Iterable[Any]]) -> Iterable:
    iterators = [iter(i) for i in iter_in]
    while True:
        try:
            res = tuple(next(i) for i in iterators)
        except RuntimeError:
            break
        else:
            yield res


def extract_windows(data: Union[torch.Tensor, np.ndarray], kernel_size: WindowType, dilation: WindowType,
                    padding: WindowType, stride: WindowType, aggregation: Optional[str] = None) \
        -> Union[torch.Tensor, np.ndarray]:
    if aggregation == 'center':
        return data
    was_torch = isinstance(data, torch.Tensor)
    if not was_torch:
        data = torch.from_numpy(data)

    start_shape = data.shape
    prev_type = data.dtype
    data = data.type(torch.float)
    tup = to_tuple_window(kernel_size, start_shape), \
          to_tuple_window(dilation, start_shape), \
          to_tuple_window(padding, start_shape), \
          to_tuple_window(stride, start_shape)
    # see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Unfold
    mapped_sizes = tuple(1 + ((sh + 2 * p - d * (k - 1) - 1) // st)
                         for k, d, p, st, sh in conc_iter_gen(tup + (start_shape[2:],)))
    data = tfunc.unfold(data, *tup)
    data = data.reshape(data.shape[:2] + mapped_sizes)
    if aggregation == 'majority':
        data, _ = torch.mode(data, 1)
        data = data.reshape((data.shape[0],) + (1,) + data.shape[1:])

    data = data.type(prev_type)
    # data = data.reshape((n, c) + self.kernel_size + (-1,))
    # data = data.permute((0, len(data.shape)-1) + tuple(range(1, len(data.shape)-1))).reshape((n, c, h, w)+self.kernel_size)
    if not was_torch:
        data = data.detach().cpu().numpy()

    return data


class SlidingWindowExtractorDataset(pl.TransformerDataset):

    def __init__(self, wrapped: data.Dataset, kernel_size: WindowType, dilation: WindowType, padding: WindowType,
                 stride: WindowType, aggregation: Optional[str] = None):
        super().__init__(wrapped)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.aggregation = aggregation

    def _transform(self, data: Union[torch.Tensor, np.ndarray]):
        data = data.reshape((1,) + data.shape)

        data = extract_windows(data, self.kernel_size, self.dilation, self.padding, self.stride, self.aggregation)

        return data.reshape(data.shape[1:])


class SlidingWindowExtractorModule(PreProcessingModule):
    def __init__(self, kernel_size: WindowType, dilation: WindowType = 1, padding: WindowType = 0,
                 stride: WindowType = 1, aggregation: Optional[str] = None):

        assert kernel_size >= (tuple(1 for _ in range(len(kernel_size))) if isinstance(kernel_size, tuple) else 1) \
               and dilation >= 1 and stride >= 1, \
            f'Kernel size must be >= 1, given {kernel_size}. ' \
            f'Dilation must be >= 1, given {dilation}. Stride must be >= 1, given {stride}.'
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.aggregation = aggregation

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        if self.kernel_size == 1:
            return dataset, meta
        if isinstance(dataset, pl.ArrayInMemoryDataset):
            data = extract_windows(dataset.data, self.kernel_size, self.dilation, self.padding,
                                   self.stride, self.aggregation)
            return pl.ArrayInMemoryDataset(data), meta
        return SlidingWindowExtractorDataset(dataset, self.kernel_size, self.dilation, self.padding,
                                             self.stride, self.aggregation), meta


def construct_clustering_feature_spaces(base_folder: str, split: str, use_numpy=True, in_memory=True,
                                        filter_method: Optional[Union[str, Tuple[str], Tuple[str, Dict[str, Any]]]] = None,
                                        use_datadings: bool = True, combine_all: bool = True,
                                        standard_file: Optional[str] = 'standard.json',
                                        range_file: str = 'range.json', as_array: bool = False,
                                        verbose: bool = True):
    SAR_VARIANTS = [
        ('SAR', True, [], [])
    ]
    OPTICAL_VARIANTS = [
        ('RGB', False, [], [pl.WhitelistModule(['Red', 'Green', 'Blue'])]),
        # o3 feature space as used by L. Landuyt
        ('O3', False, [], [pl.WhitelistModule(['Red', 'NIR', 'SWIR-2'])]),
        # HSV but this time from Red NIR and SWIR-2 bands - this is what Konpala et al. call HSV
        ('HSV(O3)', False, [], [pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                RangeClippingNormalizationModule(range_file),
                                HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))]),
        # cAWEISH Feature space as by Konpala et al.
        ('cAWEI', False, [AWEIExtractorModule(), AWEISHExtractorModule()], [pl.WhitelistModule(('AWEI', 'AWEISH'))]),
        ('cAWEI+NDVI', False, [AWEIExtractorModule(), AWEISHExtractorModule(), NDVIExtractor()], [pl.WhitelistModule(('AWEI', 'AWEISH', 'NDVI'))]),
        # cAWEISH Feature space as by Konpala et al. extended by Hue and Value
        ('cAWEI+HV(O3)', False, [AWEIExtractorModule(), AWEISHExtractorModule(),
                          pl.SequenceModule([pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                             RangeClippingNormalizationModule(range_file),
                                             HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))])],
          [pl.WhitelistModule(('AWEI', 'AWEISH', 'hue(SWIR-2, NIR, Red)', 'value(SWIR-2, NIR, Red)'))]),
        ('HSV(O3)+NDVI', False, [pl.SequenceModule([pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                                    RangeClippingNormalizationModule(range_file),
                                                    HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))]),
                                 NDVIExtractor()], []),
        # cAWEISH Feature space as by Konpala et al. combined with NDWI
        ('cAWEI+cNDWI', False, [AWEIExtractorModule(), AWEISHExtractorModule(), NDWIExtractor(), MNDWI1Extractor()],
         [pl.WhitelistModule(('AWEI', 'AWEISH', 'NDWI', 'MNDWI1'))]),
        # cNDWI Feature space as by Konpala et al.
        ('cNDWI', False, [NDWIExtractor(), MNDWI1Extractor()], [pl.WhitelistModule(('NDWI', 'MNDWI1'))]),
        ('cNDWI+NDVI', False, [NDWIExtractor(), MNDWI1Extractor(), NDVIExtractor()], [pl.WhitelistModule(('NDWI', 'MNDWI1', 'NDVI'))]),

        # cAWEISH Feature space as by Konpala et al. extended by Hue and Saturation
        ('cNDWI+HS(O3)', False, [NDWIExtractor(), MNDWI1Extractor(),
                          pl.SequenceModule([pl.WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                                             RangeClippingNormalizationModule(range_file),
                                             HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))])],
          [pl.WhitelistModule(('NDWI', 'MNDWI1', 'hue(SWIR-2, NIR, Red)', 'saturation(SWIR-2, NIR, Red)'))]),

        # # now some of my own Ideas for feature-spaces
        # # could potentially allow the algorithm to recognize clouds
        # 'opt-Cir': MergingDatasetWrapper((ChannelBlacklist(
        #     sen_data.read_s2_images(s2_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy),(0, 9)),
        #     sen_data.read_labels_only(label_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy))),
        # # and for completeness also all the broader IR channels
        # 'RGB+IR': lambda: MergingDatasetWrapper((ChannelWhitelist(
        #     sen_data.read_s2_images(s2_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy),
        #     (1, 2, 3, 7, 11, 12)),
        #     sen_data.read_labels_only(label_path, split_csv_file=split_path_no_end + '.csv', as_numpy=use_numpy))),
        # # of course we have to check full spectrum as well
        # 'full-opt': lambda: sen_data.read_s2_images(s2_path, label_path, split_csv_file=split_path_no_end + '_s2.csv',
        #                                             as_numpy=use_numpy),
    ]
    DATASET_VARIANTS = finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                   filter_method, standard_file, use_numpy, [SAR_VARIANTS]))
    DATASET_VARIANTS.extend(finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                            filter_method, standard_file, use_numpy,
                                                            [OPTICAL_VARIANTS], verbose=verbose)))
    if combine_all:
        DATASET_VARIANTS.extend(finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                            filter_method, standard_file, use_numpy,
                                                            [SAR_VARIANTS, OPTICAL_VARIANTS], verbose=verbose)))
    else:
        DATASET_VARIANTS.extend(finish_build(build_combined(use_datadings, base_folder, split, in_memory, as_array,
                                                            filter_method, standard_file, use_numpy,
                                                            [SAR_VARIANTS,
                                                             [t for t in OPTICAL_VARIANTS if t[0] in
                                                              ['cNDWI', 'HSV(O3)', 'cAWEI']]], verbose=verbose)))
    return DATASET_VARIANTS

def default_clustering_feature_space_construction(data_folder: str, filters: Optional[List[str]], use_datadings: bool,
                                       eval_split: str = SPLIT_VALIDATION) \
        -> Tuple[Dict[str, Dict[str, Tuple[pl.Pipeline, pl.Pipeline]]], Dict[str, List[str]], List[str], List[str]]:
    cons_by_filter: Dict[str, Dict[str, Tuple[Any, Any]]] = {}
    all_filters = FILTER_METHODS.copy()
    all_filters.append(None)
    if filters is not None:
        flattened_aliases = set(funct.reduce(lambda l1, l2: l1 + l2, FILTER_ALIASES.values()))
        for filter_method in filters:
            if filter_method not in flattened_aliases:
                raise ValueError(f'Invalid filter {filter_method}. Expected one of {str(all_filters)}.')
        selected_filters = filters
    else:
        selected_filters = all_filters
    f_names = {}

    for filter_method in all_filters:
        train_cons = construct_clustering_feature_spaces(data_folder, split='train', in_memory=True,
                                                         filter_method=filter_method, use_datadings=use_datadings,
                                                         as_array=True)
        valid_cons = construct_clustering_feature_spaces(data_folder, split=eval_split, in_memory=True,
                                                         filter_method=filter_method, use_datadings=use_datadings,
                                                         as_array=True)
        cur_f_names = [t[0] for t in
                       train_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: (v1, v2) for (k1, v1), (k2, v2) in zip(train_cons, valid_cons)}
        f_names[filter_method] = cur_f_names
    return cons_by_filter, f_names, all_filters, selected_filters
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    args = parser.parse_args()
    for name, pipe in construct_feature_spaces(args.data_folder,
                                               'train',
                                               add_optical=True,
                                               filter_method=None,
                                               use_datadings=True):
        print('Name:', name, 'filter:', None)
        pipe(None, None)
    for filter_method in FILTER_METHODS:
        for name, pipe in construct_feature_spaces(args.data_folder,
                                                   'train',
                                                   add_optical=True,
                                                   filter_method=filter_method,
                                                   use_datadings=True):
            print('Name:', name, 'filter:', filter_method)
            pipe(None, None)
