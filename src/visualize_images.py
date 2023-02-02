import argparse
import shutil
import sys
import time

import datadings.torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import skimage.color as color
import skimage.exposure as expose
import skimage.segmentation as seg
import PIL.Image as Image

from optuna_sacred_adaptor import DirectFileStorage, accuracy_from_contingency, iou_from_contingency
from plot_utils import *
from pre_process import *
from metrics import *
from serialisation import serialize


def image_remap(image: np.ndarray, vmin: Union[int, float], vmax: Union[int, float]) -> np.ndarray:
    return np.clip(image - vmin, 0.0, vmax - vmin) / (vmax - vmin)


class PlotDatasetChannels(Pipeline):
    def __init__(self, fig_title: str, channels: Optional[Union[str, List[str]]] = None, res_file_name_template: Optional[str] = None,
                 res_image_title_template:Optional[str] = None, num_consecutive_images: int = 5,
                 show: bool = True, save: bool = False, vmin: Optional[Union[int, float]] = 0.0,
                 vmax: Optional[Union[int, float]] = 1.0,
                 equalize: bool = True,
                 save_jpg: bool = True):
        assert not (res_file_name_template is None and save)
        self.fig_title = fig_title
        if channels is not None and isinstance(channels, str):
            channels = [channels]
        self.channels = channels
        self.res_image_title_template = res_image_title_template
        self.res_file_name_template = res_file_name_template
        self.num_consecutive_images = num_consecutive_images
        self.show = show
        self.save = save
        self.vmin = vmin
        self.vmax = vmax
        self.equalize = equalize
        self.save_jpg = save_jpg

    def save_f_name(self, hook: Optional[RunHook], n_saves: int, meta: Meta, i: int) -> Optional[str]:
        if hook is None:
            return None
        info: PerItemMeta = meta.per_item_info[i]
        formatted = self.res_file_name_template.format(n_saves=n_saves, channels='_'.join(self.channels),
                                                       region=info.region, id=info.id)
        return hook.get_artifact_file_name(formatted)

    def do_save(self, hook: Optional[pl.RunHook], meta: Meta, n_saves: int, fig: plt.Figure, ld: int, n_rows: int, n_cols: int, create_new: bool = False) \
            -> Tuple[Optional[plt.Figure], Optional[np.ndarray], Optional[int]]:
        if self.save and self.res_file_name_template is not None:
            file = hook.get_artifact_file_name(self.save_f_name(hook, n_saves, meta, n_saves*n_rows))
        else:
            file = None
        save_and_show(fig, file, self.show, ld, self.save)
        if create_new:
            assert n_rows >= 1 and n_cols >= 1
            return create_figure_with_subplots(self.fig_title, n_rows, n_cols)
        else:
            return None, None, None

    def get_title_str(self, meta: Meta, i: int, j: int, channel: str, template: Optional[str]) -> Optional[str]:
        if template is not None:
            info: PerItemMeta = meta.per_item_info[i]
            return template.format(region=info.region, id=info.id, n_image=i, n_channel=j, channel=channel)
        return None

    def set_title(self, ax: plt.Axes, meta: Meta, i: int, j: int, channel: str):
        tstr = self.get_title_str(meta, i, j, channel, self.res_image_title_template)
        ax.set_title(tstr)

    def write_jpg(self, data: np.ndarray, suffix: str, hook: Optional[RunHook], n_saves: int,
                  meta: Meta, i: int, j: int, channel: str, template: Optional[str]):
        f_name = self.save_f_name(hook, n_saves, meta, i)
        t_str = self.get_title_str(meta, i, j, channel, template)
        if f_name is not None and t_str is not None:
            f_name = f_name + '_' + "".join((x if x.isalnum() else '_') for x in t_str) + suffix + '.jpg'
            data = (data * 255).astype(np.uint8)
            if data.ndim == 3 and data.shape[0] == 3:
                data = data.transpose((1, 2, 0))
            pil_image = Image.fromarray(data)
            pil_image.save(f_name)
            hook.add_artifact(f_name)

    def do_plot(self, hook: Optional[RunHook], channels: List[str], i, axs: np.ndarray, image: np.ndarray, ci: Dict[str, int], meta: Meta, n_saves: int):
        for j, channel in enumerate(channels):
            ax: plt.Axes = axs[i % self.num_consecutive_images, j]
            img = image[ci[channel]]
            vmin = img.min() if self.vmin is None else self.vmin
            vmax = img.max() if self.vmax is None else self.vmax
            img = image_remap(img, vmin, vmax)
            if self.equalize:
                img = expose.equalize_adapthist(img)
            ax.set_axis_off()
            ax.imshow(img, vmin=0.0, vmax=1.0, cmap='gray')
            self.set_title(ax, meta, i, j, channel)
            if self.save_jpg:
                self.write_jpg(img, '_'+channel, hook, n_saves, meta, i, j, channel, self.res_image_title_template)


    def n_plots(self, channels: List[str]):
        return len(channels)

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        if dataset is None or meta is None:
            raise ValueError
        n_saves = 0
        ci = pl.channel_index(meta)
        channels = self.channels if self.channels is not None else meta.channel_names
        assert not any(map(lambda c: c not in meta.channel_names, channels)), \
            f'Found channel that is not present in channel index! Expected {channels} to be present but {meta.channel_names} were found!'
        self.print(meta, f'Visualizing the channels {channels} for {len(dataset)} images. save={self.save} and '
                         f'show={self.show} with file_name_template={self.res_file_name_template}...')
        t = time.time()
        fig, axs, ld = create_figure_with_subplots(self.fig_title, min(self.num_consecutive_images, len(dataset)),
                                                   self.n_plots(channels))
        for i, image in enumerate(dataset):
            self.do_plot(hook, channels, i, axs, image, ci, meta, n_saves)
            if i % self.num_consecutive_images == self.num_consecutive_images - 1:
                fig, axs, ld = self.do_save(meta.run_hook, meta, n_saves, fig, ld,
                                            min(self.num_consecutive_images, len(dataset)-(i+1)),
                                            self.n_plots(channels),
                                            i < len(dataset) - 1)
                n_saves += 1
        if fig is not None: # remember to check for completion!
            self.do_save(meta.run_hook, meta, n_saves, fig, ld, -1, -1)
        t = time.time() - t
        self.print(meta, f'Finished visualizing the channels {channels} for {len(dataset)} images. '
                         f'Took {t:.3f}s - {t/len(dataset) if len(dataset) > 0 else -1.0:.3f}s on average.')
        return dataset, meta

class PlotDatasetRGBComposition(PlotDatasetChannels):

    def __init__(self, fig_title: str, channels: Iterable[str],
                 res_file_name_template: Optional[str] = None, res_image_title_template: Optional[str] = None,
                 num_consecutive_images: int = 5, show: bool = True, save: bool = False,
                 vmin: Union[Union[int, float], Tuple[Union[int, float],...]] = 0.0,
                 vmax: Union[Union[int, float], Tuple[Union[int, float],...]] = 1.0):
        channels = list(channels)
        assert len(channels) % 3 == 0, '3 channels are required per composite, therefore the given channels should be divisible by 3'
        super().__init__(fig_title, channels, res_file_name_template, res_image_title_template, num_consecutive_images,
                         show, save, vmin, vmax)
        if not isinstance(vmin, tuple):
            self.vmin = tuple(vmin for _ in channels)
        else:
            assert len(vmin) == len(channels) // 3
            self.vmin = vmin
        if not isinstance(vmax, tuple):
            self.vmax = tuple(vmax for _ in channels)
        else:
            assert len(vmax) == len(channels) // 3
            self.vmax = vmax

    def n_plots(self, channels: List[str]):
        return len(channels) // 3

    def do_plot(self, hook: Optional[RunHook], channels: List[str], i, axs: np.ndarray, image: np.ndarray, ci: Dict[str, int], meta: Meta, n_saves: int):
        for j in range(0, len(channels), 3):
            ax: plt.Axes = axs[i % self.num_consecutive_images, j//3]
            img = image[tuple(ci[cn] for cn in channels[j:j+3]),...]
            img = utils.move_channel_to_position(img, img.ndim, 0)
            img = image_remap(img, self.vmin[j//3], self.vmax[j//3])
            ax.set_axis_off()
            ax.imshow(img, vmin=0.0, vmax=1.0)
            self.set_title(ax, meta, i, j, '+'.join(channels[j:j+3]))
            if self.save_jpg:
                cname = '_'+'_'.join(channels[j:j+3])
                self.write_jpg(img, cname, hook, n_saves, meta, i, j, cname, self.res_image_title_template)


class GreyChannelDataset(TransformerDataset):

    def __init__(self, wrapped: data.Dataset, channels: List[int], vmin: float = .0, vmax: float = 1.,
                 equalize_contrast: bool = False):
        super().__init__(wrapped)
        assert len(channels) == 1 or len(channels) == 3
        self.channels = tuple(channels)
        self.vmin = vmin
        self.vmax = vmax
        self.equalize_contrast = equalize_contrast

    def _transform(self, data: Union[torch.Tensor, np.ndarray]):
        data: np.ndarray = utils.torch_as_np(data)
        data = data[self.channels,...]
        img = image_remap(data, self.vmin, self.vmax)
        if img.shape[0] != 1:
            grey = utils.move_channel_to_position(img, data.ndim, 0)
            if self.equalize_contrast:
                grey = expose.equalize_adapthist(grey)
            grey = color.rgb2gray(grey)
            grey = grey.reshape((1,)+grey.shape)
        elif self.equalize_contrast:
            grey = expose.equalize_adapthist(img)
        else:
            grey = img
        return img, grey

def gt_bw_map(gt_image: np.ndarray, is_prob: bool = False) -> np.ndarray:
    res = np.zeros((3,) + gt_image.shape[1:],dtype=np.float32)
    gt_image = gt_image.squeeze()
    if is_prob:
        res[0] = gt_image
        res[1] = gt_image
        res[2] = gt_image
    else:
        mask = gt_image == 1
        res[0,mask] = 1.0
        res[1,mask] = 1.0
        res[2,mask] = 1.0
    return res
# The following colores are curtesy of
# https://volkmarmeyd.de/infos/edv/farbtabelle.html#pastell

# TODO use torch-cuda implementation if available
@utils.njit(parallel=True)
def gt_color_map(gt_image: np.ndarray):
    gt_image = gt_image.ravel()
    color_image_r = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_g = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_b = np.empty(gt_image.shape[0], dtype=np.float32)
    for i in numba.prange(gt_image.shape[0]):
        if gt_image[i] < 0: # Invalid => black color
            color_image_r[i] = 0.0
            color_image_g[i] = 0.0
            color_image_b[i] = 0.0
        elif gt_image[i] == 0: # dry => green - darkolivegreen  pastell - darkkhaki
            color_image_r[i] = 0.33333333#0.74117647
            color_image_g[i] = 0.41960784#0.71764705
            color_image_b[i] = 0.18431372#0.41960784
        else: # should be water => blues - royalblue lightcyan
            color_image_r[i] = 0.25490196#0.87843137
            color_image_g[i] = 0.4117670#1.0
            color_image_b[i] = 0.88235294# 1.0
    return np.stack((color_image_r, color_image_g, color_image_b))

def assign_gt_color_map(gt_image: np.ndarray, actual_colors: np.ndarray, grey_image: Optional[np.ndarray]):
    copy = np.copy(actual_colors)
    gt_image = gt_image.squeeze()
    mask = gt_image == 1
    copy[0,mask] = 0.25490196
    copy[1,mask] = 0.41960784
    copy[2,mask] = 0.88235294
    if grey_image is not None:
        copy[:,mask] *= grey_image.squeeze()[mask]
    return copy

# TODO use torch-cuda implementation if available
@utils.njit(parallel=True)
def pred_color_map(gt_image: np.ndarray, pred_image: np.ndarray, is_prob: bool = False) -> np.ndarray:
    gt_image = gt_image.ravel()
    pred_image = pred_image.ravel()
    color_image_r = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_g = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_b = np.empty(gt_image.shape[0], dtype=np.float32)
    for i in numba.prange(gt_image.shape[0]):
        pred_flooded = (pred_image[i] > 0.5) if is_prob else (pred_image[i] == 1)
        gt_flooded = gt_image[i] == 1
        is_invalid = gt_image[i] < 0
        if is_invalid: # Invalid => black color
            color_image_r[i] = 0.0
            color_image_g[i] = 0.0
            color_image_b[i] = 0.0
        elif not gt_flooded and not pred_flooded:# dry, TN => green - darkolivegreen  pastell - darkkhaki
            color_image_r[i] = 0.33333333#0.74117647
            color_image_g[i] = 0.41960784#0.71764705
            color_image_b[i] = 0.18431372#0.41960784
        elif gt_flooded and pred_flooded: # water, TP => blues - royalblue lightcyan
            color_image_r[i] = 0.25490196#0.87843137
            color_image_g[i] = 0.4117670#1.0
            color_image_b[i] = 0.88235294# 1.0
        elif not gt_flooded and pred_flooded: # FP - chartreuse
            color_image_r[i] = 0.5
            color_image_g[i] = 1.0
            color_image_b[i] = 0.0
        elif gt_flooded and not pred_flooded: # FN- mediumvioletred
            color_image_r[i] = 0.78039215#1.0
            color_image_g[i] = 0.0823529#0.0
            color_image_b[i] = 0.52156862#0.0
        else:
            color_image_r[i] = 1.0
            color_image_g[i] = 1.0
            color_image_b[i] = 1.0
        if is_prob:
            if is_invalid:
                invalid_recognition = 1 - np.abs(pred_image[i] - 0.5) * 2
                color_image_r[i] = invalid_recognition
                color_image_g[i] = invalid_recognition
                color_image_b[i] = invalid_recognition
            else:
                prop_modifier = ((pred_image[i] - 0.5) * 2) if pred_flooded else (1 - ( pred_image[i] * 2))
                color_image_r[i] *= prop_modifier
                color_image_g[i] *= prop_modifier
                color_image_b[i] *= prop_modifier
    return np.stack((color_image_r, color_image_g, color_image_b))

@utils.njit(parallel=True)
def assign_pred_color_map(gt_image: np.ndarray, pred_image: np.ndarray, source_image: np.ndarray,
                          is_prob: bool = False) -> np.ndarray:
    gt_image = gt_image.ravel()
    pred_image = pred_image.ravel()
    source_image = source_image.reshape((3, -1))
    color_image_r = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_g = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_b = np.empty(gt_image.shape[0], dtype=np.float32)
    for i in numba.prange(gt_image.shape[0]):
        pred_flooded = (pred_image[i] > 0.5) if is_prob else (pred_image[i] == 1)
        gt_flooded = gt_image[i] == 1
        if gt_image[i] < 0 or (not gt_flooded and not pred_flooded):
            color_image_r[i] = source_image[0, i]
            color_image_g[i] = source_image[1, i]
            color_image_b[i] = source_image[2, i]
        elif gt_flooded and pred_flooded: # water, TP => blues - royalblue lightcyan
            color_image_r[i] = 0.25490196#0.87843137
            color_image_g[i] = 0.4117670#1.0
            color_image_b[i] = 0.88235294# 1.0
        elif not gt_flooded and pred_flooded: # FP - chartreuse
            color_image_r[i] = 0.5
            color_image_g[i] = 1.0
            color_image_b[i] = 0.0
        elif gt_flooded and not pred_flooded: # FN- mediumvioletred
            color_image_r[i] = 0.78039215#1.0
            color_image_g[i] = 0.0823529#0.0
            color_image_b[i] = 0.52156862#0.0
        else:
            color_image_r[i] = source_image[0, i]
            color_image_g[i] = source_image[1, i]
            color_image_b[i] = source_image[2, i]
        if is_prob:
            prop_modifier = ((pred_image[i] - 0.5) * 2) if pred_flooded else (1 - ( pred_image[i] * 2))
            color_image_r[i] *= prop_modifier
            color_image_g[i] *= prop_modifier
            color_image_b[i] *= prop_modifier

    return np.stack((color_image_r, color_image_g, color_image_b))


@utils.njit(parallel=True)
def assign_pred_color_map_structured(gt_image: np.ndarray, pred_image: np.ndarray, source_image: np.ndarray,
                                     grey_image: np.ndarray, is_prob: bool = False) -> np.ndarray:
    gt_image = gt_image.ravel()
    pred_image = pred_image.ravel()
    grey_image = grey_image.ravel()
    source_image = source_image.reshape((3, -1))
    color_image_r = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_g = np.empty(gt_image.shape[0], dtype=np.float32)
    color_image_b = np.empty(gt_image.shape[0], dtype=np.float32)
    for i in numba.prange(gt_image.shape[0]):
        pred_flooded = (pred_image[i] > 0.5) if is_prob else (pred_image[i] == 1)
        gt_flooded = gt_image[i] == 1
        if gt_image[i] < 0 or (not gt_flooded and not pred_flooded):
            color_image_r[i] = source_image[0, i]
            color_image_g[i] = source_image[1, i]
            color_image_b[i] = source_image[2, i]
        elif gt_flooded and pred_flooded: # water, TP => blues - royalblue lightcyan
            color_image_r[i] = 0.25490196 * grey_image[i]#0.87843137
            color_image_g[i] = 0.4117670 * grey_image[i]#1.0
            color_image_b[i] = 0.88235294 * grey_image[i]# 1.0
        elif not gt_flooded and pred_flooded: # FP - chartreuse
            color_image_r[i] = 0.5 * grey_image[i]
            color_image_g[i] = 1.0 * grey_image[i]
            color_image_b[i] = 0.0
        elif gt_flooded and not pred_flooded: # FN- mediumvioletred
            color_image_r[i] = 0.78039215 * grey_image[i]#1.0
            color_image_g[i] = 0.0823529 * grey_image[i]#0.0
            color_image_b[i] = 0.52156862 * grey_image[i]#0.0
        else:
            color_image_r[i] = source_image[0, i]
            color_image_g[i] = source_image[1, i]
            color_image_b[i] = source_image[2, i]
        if is_prob:
            prop_modifier = ((pred_image[i] - 0.5) * 2) if pred_flooded else (1 - ( pred_image[i] * 2))
            color_image_r[i] *= prop_modifier
            color_image_g[i] *= prop_modifier
            color_image_b[i] *= prop_modifier

    return np.stack((color_image_r, color_image_g, color_image_b))

class PlotChannelOverlay(MultiPipeline):
    # TODO support for colormaps that match something else than Sen1Floods11
    # (This only supports hardcoded colors for the individual classes)
    def __init__(self, fig_title: str,
                 reference_criterion: NameSelectionCriterion,
                 ground_truth_criterion: NameSelectionCriterion,
                 prediction_module: Optional[MultiPipeline] = None,
                 prediction_selection_criterion: Optional[SelectionCriterion] = None,
                 luminance_channels: Optional[Union[str, List[str]]] = None,
                 res_file_name_template: Optional[str] = None,
                 ref_image_title_template: Optional[str] = None,
                 label_image_title_template: Optional[str] = None,
                 pred_image_title_template: Optional[str] = None,
                 num_consecutive_images: int = 5,
                 show: bool = True, save: bool = False,
                 vmin: Union[int, float] = 0.0, vmax: Union[int, float] = 1.0,
                 equalize_contrast: bool = False,
                 is_prob: bool = False,
                 save_jpg: bool = False):
        super().__init__()
        self.reference_criterion = reference_criterion
        self.ground_truth_criterion = ground_truth_criterion
        assert not (res_file_name_template is None and save)
        self.fig_title = fig_title
        if luminance_channels is not None and isinstance(luminance_channels, str):
            luminance_channels = [luminance_channels]
        assert len(luminance_channels) == 1 or len(luminance_channels) == 3, 'Expected either single (grey-scale) channel or three ' \
                                                         'channels that will be interpreted as RGB'
        self.prediction_module = prediction_module
        self.prediction_selection_criterion = prediction_selection_criterion
        self.luminance_channels = luminance_channels
        self.ref_image_title_template = ref_image_title_template
        self.label_image_title_template = label_image_title_template
        self.res_file_name_template = res_file_name_template
        self.pred_image_title_template = pred_image_title_template
        self.num_consecutive_images = num_consecutive_images
        self.show = show
        self.save = save
        self.vmin = vmin
        self.vmax = vmax
        self.equalize_contrast = equalize_contrast
        self.is_prob = is_prob
        self.save_jpg = save_jpg

    def save_f_name(self, hook: Optional[RunHook], n_saves: int, meta: Meta, i: int) -> Optional[str]:
        if hook is None:
            return None
        info: PerItemMeta = meta.per_item_info[i]
        formatted = self.res_file_name_template.format(n_saves=n_saves, channels='_'.join(self.luminance_channels),
                                                       is_prob=self.is_prob, region=info.region, id=info.id)
        return hook.get_artifact_file_name(formatted)

    def do_save(self, hook: Optional[pl.RunHook], n_saves: int, fig: plt.Figure, ld: int, n_rows: int, n_cols: int, meta: Meta,
                create_new: bool = False) \
            -> Tuple[Optional[plt.Figure], Optional[np.ndarray], Optional[int]]:
        if self.save and self.res_file_name_template is not None:
            file = self.save_f_name(hook, n_saves, meta, n_saves*n_rows)
        else:
            file = None
        save_and_show(fig, file, self.show, ld, self.save)
        if create_new:
            assert n_rows >= 1 and n_cols >= 1
            return create_figure_with_subplots(self.fig_title, n_rows, n_cols)
        else:
            return None, None, None

    def get_title_str(self, meta: Meta, i: int, j: int, channel: str, template: Optional[str]) -> Optional[str]:
        if template is not None:
            info: PerItemMeta = meta.per_item_info[i]
            return template.format(region=info.region, id=info.id, n_image=i, n_channel=j, channel=channel)
        return None

    def set_title(self, ax: plt.Axes, meta: Meta, i: int, j: int, channel: str, template: Optional[str]):
        tstr = self.get_title_str(meta, i, j, channel, template)
        ax.set_title(tstr)

    def write_jpg(self, data: np.ndarray, suffix: str, hook: Optional[RunHook], n_saves: int,
                  meta: Meta, i: int, j: int, channel: str, template: Optional[str]):
        f_name = self.save_f_name(hook, n_saves, meta, i)
        t_str = self.get_title_str(meta, i, j, channel, template)
        if f_name is not None and t_str is not None:
            f_name = f_name + '_' + "".join((x if x.isalnum() else '_') for x in t_str) + suffix + '.jpg'
            pil_image = Image.fromarray((data * 255).astype(np.uint8).transpose((1, 2, 0)))
            pil_image.save(f_name)
            hook.add_artifact(f_name)

    def plot_reference(self, axs: np.ndarray, i: int, ref_image: np.ndarray, ref_meta: Meta, n_saves: int):
        ax: plt.Axes = axs[i % self.num_consecutive_images, 0]
        ax.set_axis_off()
        ax.imshow(utils.move_channel_to_position(ref_image, ref_image.ndim, 0), vmin=.0, vmax=1.)
        self.set_title(ax, ref_meta, i, 0, '+'.join(self.luminance_channels), self.ref_image_title_template)
        if self.save_jpg:
            self.write_jpg(ref_image, '_ref_image', hook, n_saves, ref_meta, i, 3,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)

    def plot_labels(self, axs: np.ndarray, i: int, gt_image: np.ndarray, ref_image: np.ndarray,
                    ref_grey: np.ndarray, ref_meta: Meta, hook: Optional[RunHook], n_saves: int) -> np.ndarray:
        gt_colors = gt_color_map(gt_image).reshape((3,) + gt_image.shape[1:])
        assign_colors = assign_gt_color_map(gt_image, ref_image, None).reshape((3,) + gt_image.shape[1:])
        assign_colors_structured = assign_gt_color_map(gt_image, gt_colors, ref_grey).reshape((3,) + gt_image.shape[1:])
        bw_map = gt_bw_map(gt_image)

        if self.save_jpg:
            self.write_jpg(gt_colors, '_gt_colors', hook, n_saves, ref_meta, i, 0,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(gt_colors, '_gt_colors', hook, n_saves, ref_meta, i, 0,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(assign_colors, '_gt_assign_colors', hook, n_saves, ref_meta, i, 1,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(assign_colors_structured, '_gt_assign_colors_structured', hook, n_saves, ref_meta, i, 2,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(bw_map, '_gt_bw_map', hook, n_saves, ref_meta, i, 3,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
        for j, rgb in enumerate([assign_colors_structured, assign_colors, gt_colors, bw_map]):
            rgb = utils.move_channel_to_position(rgb, rgb.ndim, 0)
            ax: plt.Axes = axs[i % self.num_consecutive_images, 1+j]
            ax.set_axis_off()
            ax.imshow(rgb, vmin=.0, vmax=1.)
            self.set_title(ax, ref_meta, i, j, '+'.join(self.luminance_channels), self.label_image_title_template)
        return gt_colors

    def plot_results(self, axs: np.ndarray, i: int, gt_colors: np.ndarray, gt_image: np.ndarray,ref_image: np.ndarray,
                     ref_grey: np.ndarray, pred_image: np.ndarray, pred_meta: Meta, hook: Optional[RunHook], n_saves: int):
        if self.is_prob:
            assert np.all(0.0 <= pred_image) and np.all(pred_image <= 1.0)
        else:
            unique = np.unique(pred_image)
            assert 0 <= np.min(unique) <= np.max(unique) <= 1
        pred_colors = pred_color_map(gt_image, pred_image, self.is_prob).reshape(gt_colors.shape)
        assign_colors = assign_pred_color_map(gt_image, pred_image, ref_image, self.is_prob).reshape(gt_colors.shape)
        assign_colors_structured = assign_pred_color_map_structured(gt_image, pred_image, ref_image, ref_grey, self.is_prob).reshape(gt_colors.shape)
        bw_map = gt_bw_map(pred_image, is_prob=self.is_prob)
        if self.save_jpg:
            self.write_jpg(pred_colors, '_pred_colors', hook, n_saves, pred_meta, i, 0,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(assign_colors, '_assign_colors', hook, n_saves, pred_meta, i, 1,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(assign_colors_structured, '_assign_colors_structured', hook, n_saves, pred_meta, i, 2,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
            self.write_jpg(bw_map, '_bw_map', hook, n_saves, pred_meta, i, 3,
                           '+'.join(self.luminance_channels), self.pred_image_title_template)
        # clip instead of interpolate in order to brighten the colors of incorrect predictions
        for j, rgb in enumerate([assign_colors_structured, assign_colors, pred_colors, bw_map]):
            rgb = utils.move_channel_to_position(rgb, rgb.ndim, 0)
            ax: plt.Axes = axs[i % self.num_consecutive_images, 5+j]
            ax.set_axis_off()
            ax.imshow(rgb, vmin=.0, vmax=1.)
            self.set_title(ax, pred_meta, i, j, '+'.join(self.luminance_channels), self.pred_image_title_template)

    def get_pred_data(self, summary: Summary):
        if self.prediction_module is not None:
            s = self.prediction_selection_criterion.name if self.prediction_selection_criterion is not None else ""
            self.print(summary, f'Calculating predictions {s}')
            pred_summary = Summary([], lambda: '')
            pred_summary.indent = summary.indent
            pred_summary = self.prediction_module(pred_summary.do_indent())
            if len(pred_summary) == 1:
                pred_dataset, pred_meta = pred_summary.by_index[0]
            else:
                assert self.prediction_selection_criterion is not None, \
                    'If the prediction module outputs a summary with more than one element, ' \
                    'then a selection criterion is required to identify the corresponding output'
                pred_dataset, pred_meta = pred_summary.by_criterion(self.prediction_selection_criterion)
            return pred_dataset, pred_meta
        return None, None

    def __call__(self, summary: Summary) -> Summary:
        self.print(summary, f'Visualizing reference {self.reference_criterion.name} and ground truth '
                            f'{self.ground_truth_criterion}.')
        ref_dataset, ref_meta = summary.by_criterion(self.reference_criterion)
        gt_dataset, gt_meta = summary.by_criterion(self.ground_truth_criterion)
        ref_ci = channel_index(ref_meta)
        ref_dataset = GreyChannelDataset(ref_dataset, [ref_ci[cn] for cn in self.luminance_channels],
                                         vmin=self.vmin, vmax=self.vmax, equalize_contrast=self.equalize_contrast)
        assert len(gt_dataset) == len(ref_dataset), \
            'Both the ground truth and the reference dataset should have the same length!'
        pred_dataset, pred_meta = self.get_pred_data(summary)

        n_cols = 5+(0 if pred_dataset is None else 4)
        fig, axs, ld = create_figure_with_subplots(self.fig_title, min(self.num_consecutive_images, len(gt_dataset)),
                                                   n_cols)
        n_saves = 1
        self.print(summary, f'Plotting images.')
        t = time.time()
        for i, gt_image, (ref_image, ref_grey) in zip(range(len(gt_dataset)),gt_dataset, ref_dataset):
            assert gt_meta.per_item_info[i] == ref_meta.per_item_info[i], \
                'Both the ground truth and the reference image should originate from the same location!'
            assert gt_image.shape[1:] == ref_image.shape[1:], \
                'Both the ground truth and the reference image should have the same size!'
            assert gt_image.shape[0] == 1

            self.plot_reference(axs, i, ref_image, ref_meta, n_saves)
            gt_colors = self.plot_labels(axs, i, gt_image, ref_image, ref_grey, ref_meta, summary.run_hook, n_saves)
            if pred_dataset is not None:
                assert pred_meta is not None and len(pred_meta.per_item_info) > i \
                       and pred_meta.per_item_info[i] == gt_meta.per_item_info[i]
                self.plot_results(axs, i, gt_colors, gt_image, ref_image, ref_grey, pred_dataset[i], pred_meta,
                                  summary.run_hook, n_saves)

            if i % self.num_consecutive_images == self.num_consecutive_images - 1:
                fig, axs, ld = self.do_save(summary.run_hook, n_saves, fig, ld,
                                            min(self.num_consecutive_images, len(gt_dataset)-(i+1)),
                                            n_cols,
                                            create_new=i < len(gt_dataset) - 1,
                                            meta=ref_meta)
                n_saves += 1
        if fig is not None: # remember to check for completion!
            self.do_save(summary.run_hook, n_saves, fig, ld, -1, -1, meta=ref_meta)
        t = time.time() - t
        self.print(summary, f'Plotting images took {t:.3f}s - '
                            f'{t / len(gt_dataset) if len(gt_dataset) > 0 else -1.0:.3f}s on average.')
        return summary

class PlotSegmentationOverlay(PlotChannelOverlay):
    def plot_results(self, axs: np.ndarray, i: int, gt_colors: np.ndarray, gt_image: np.ndarray,ref_image: np.ndarray,
                     ref_grey: np.ndarray, pred_image: np.ndarray, pred_meta: Meta, hook: Optional[RunHook], n_saves: int):
        pred_image: np.ndarray = np.squeeze(pred_image)
        assert pred_image.ndim == 2
        assert pred_image.shape == ref_image.shape[1:]
        ref_image = utils.move_channel_to_position(ref_image, ref_image.ndim, 0)
        gt_colors = utils.move_channel_to_position(gt_colors, gt_colors.ndim, 0)
        pred_image = pred_image - np.min(pred_image) + 1 # ensure no background...
        marked_image = seg.mark_boundaries(ref_image, pred_image)#, outline_color=(0, 0, 0))
        marked_colors = seg.mark_boundaries(gt_colors, pred_image)#, outline_color=(0, 0, 0))
        # clip instead of interpolate in order to brighten the colors of incorrect predictions
        for j, rgb in enumerate([marked_image, marked_colors]):
            ax: plt.Axes = axs[i % self.num_consecutive_images, 5+j]
            ax.set_axis_off()
            ax.imshow(rgb, vmin=.0, vmax=1.)
            self.set_title(ax, pred_meta, i, j, '+'.join(self.luminance_channels), self.pred_image_title_template)


def sentinel_1_vis_pipe(res_file_name_template: Optional[str] = None,  show: bool = True) -> SequenceModule:
    return SequenceModule([RangeClippingNormalizationModule('range.json'),
                           PlotDatasetChannels('Sentinel-1 Image Visualisations',
                                                channels=['VV', 'VH'],
                                                res_file_name_template=res_file_name_template,
                                                res_image_title_template=
                                                '"{channel}" of image {id} in "{region}"',
                                                show=show, save=res_file_name_template is not None)
    ])

def sentinel_2_vis_pipe(res_file_name_template: Optional[str] = None, show: bool = True,
                        res_channel_file_name_template: Optional[str] = None) -> SequenceModule:
    res_channel_file_name_template = 'raw_'+res_file_name_template if res_channel_file_name_template is None else res_channel_file_name_template
    return SequenceModule([RangeClippingNormalizationModule('range.json'),
                           PlotDatasetRGBComposition('Sentinel-2 Image Visualisations',
                                                     ('Red', 'Green', 'Blue', 'Red', 'NIR', 'SWIR-2'),
                                                     res_file_name_template=res_file_name_template,
                                                     res_image_title_template=
                                                     'RGB-Composite of "{channel}" ({id} in "{region}")',
                                                     show=show, save=res_file_name_template is not None,
                                                     # clouds result in a severe offset... Therefore: rescale needed
                                                     vmax=(0.2, 0.2)),
                           PlotDatasetChannels('Sentinel-2 Channel Visualisations',
                                               channels=CHANNEL_NAMES_S2,
                                               res_file_name_template=res_channel_file_name_template,
                                               res_image_title_template=
                                               'Channel "{channel}" ({id} in "{region}")',
                                               show=show, save=res_file_name_template is not None,
                                               # clouds result in a severe offset... Therefore: rescale needed,
                                               vmin=None,
                                               vmax=None)
    ])

def visualize_dataset_s2(data_folder: str, hook: RunHook, res_file_name_template: Optional[str] = None, split = SPLIT_TEST,
                         show: bool = False, filter_method: str = 'lee_improved'):
    constructors = dict(
        construct_feature_spaces(data_folder, split=split, use_datadings=True, add_optical=True,
                                 filter_method=filter_method, in_memory=False, standard_file=None,
                                 range_file='range.json'))
    pipe = SequenceModule([
        constructors['S2'],
        sentinel_2_vis_pipe(res_file_name_template=res_file_name_template, show=show)
    ])
    meta = empty_meta()
    meta = meta._replace(run_hook=hook)
    d, m = pipe(None, meta)
    finish_visualization()

def visualize_dataset_s1(data_folder: str, hook: RunHook, res_file_name_template: Optional[str] = None, split = SPLIT_TEST,
                         show: bool = False, filter_method: str = 'lee_improved'):
    constructors = dict(
        construct_feature_spaces(data_folder, split=split, use_datadings=True, add_optical=True,
                                 filter_method=filter_method, in_memory=False, standard_file=None,
                                 range_file='range.json'))
    pipe = SequenceModule([
        constructors['SAR'],
        sentinel_1_vis_pipe(res_file_name_template=res_file_name_template, show=show)
    ])
    meta = empty_meta()
    meta = meta._replace(run_hook=hook)
    d, m = pipe(None, meta)
    finish_visualization()

def visualize_dataset_feature_spaces(data_folder: str, hook: RunHook, res_file_name_template: Optional[str] = None,
                                     split = SPLIT_TEST, show: bool = True):
    constructors = dict(
        construct_feature_spaces(data_folder, split=split, use_datadings=True, add_optical=True,
                                 filter_method='lee_improved', in_memory=False, standard_file=None,
                                 range_file='range.json'))
    for key in ['O3']:#constructors.keys():
        try:
            template = res_file_name_template + '_' + key if res_file_name_template is not None else None
            meta = empty_meta()
            meta = meta._replace(run_hook=hook)
            d, m = constructors[key](None, meta)
            d, m = PlotDatasetChannels('Channel Visualisations',
                                       channels=m.channel_names,
                                       res_file_name_template=template,
                                        res_image_title_template=
                                       'Channel "{channel}" ({id} in "{region}")',
                                       show=show, save=res_file_name_template is not None,
                                       # clouds result in a severe offset... Therefore: rescale needed,
                                       vmin=None,
                                       vmax=None)(d, m)
        except:
            print(traceback.format_exc(), file=sys.stderr)
    finish_visualization()

def visualize_classification(data_folder: str, hook: RunHook, label_pipe: MultiPipeline, show: bool = True,
                             res_file_name_template: Optional[str] = None, split = SPLIT_TEST,
                             channels: Iterable[str] = ('Red', 'Green', 'Blue'), is_probability: bool = False,
                             label_type = TYPE_LABEL, calculate_metrics: bool = False) -> Optional[Tuple[float, float]]:
    print('f_name template', res_file_name_template)
    channels = list(channels)
    pipe_seq = [
        MultiDistributorModule([
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=SequenceModule([DistributorModule([
                                          Sen1Floods11DataDingsDatasource(data_folder, TYPE_S2, split, in_memory=False),
                                          Sen1Floods11DataDingsDatasource(data_folder, TYPE_S1, split, in_memory=False)
                                      ]),
                                      RangeClippingNormalizationModule('range.json')]),
                                  dataset_name='features'),
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, label_type,
                                                                              split, in_memory=False),
                                  dataset_name='labels')
        ]),
        PlotChannelOverlay('Image Wise Label Distribution',
                           reference_criterion=NameSelectionCriterion('features'),
                           ground_truth_criterion=NameSelectionCriterion('labels'),
                           luminance_channels=channels,
                           label_image_title_template='Labels for image {id} in "{region}" ({channel})',
                           ref_image_title_template='{channel} of ({id} in "{region}")',
                           pred_image_title_template='Predictions for image {id} in "{region}" ({channel})',
                           res_file_name_template=res_file_name_template,
                           vmax=0.1 if len(channels) > 1 else 1.0,
                           prediction_module=label_pipe,
                           prediction_selection_criterion=NameSelectionCriterion('valid_prediction'),
                           save=res_file_name_template is not None,
                           show=show,
                           is_prob=is_probability,
                           save_jpg=res_file_name_template is not None
                           )
    ]
    if calculate_metrics and not is_probability:
        pipe_seq.extend([
            label_pipe,
            MetricsModule(prediction_criterion=NameSelectionCriterion('predictions'),
                          label_criterion=NameSelectionCriterion('labels'),
                          per_data_computation=ContingencyMatrixComputation('contingency'))
        ])
        hook.add_result_metric_prefix('contingency')
    elif calculate_metrics and is_probability:
        print('Cannot calculate metrics for probability targets. Look at the output for non-probability visualizations.')
    pipe = MultiSequenceModule(pipe_seq)
    summary = Summary([], lambda: '')
    summary.set_hook(hook)
    pipe(summary)
    finish_visualization()
    if calculate_metrics and not is_probability:
        final_contingency_matrices = [d[0] for k, d in hook.recorded_metrics.items() if k.startswith('contingency')]
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        iou_values = [iou_from_contingency(d) for d in final_contingency_matrices]
        acc = float(sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values)))
        iou = float(sum(map(lambda t: t[0], iou_values)) / sum(map(lambda t: t[1], iou_values)))
        return acc, iou

def visualize_clusters(data_folder: str, hook: RunHook, label_pipe: MultiPipeline, show: bool = True,
                     res_file_name_template: Optional[str] = None, split = SPLIT_TEST,
                     channels: Iterable[str] = ('Red', 'Green', 'Blue')):
    # This really should not be a copy, but for now it is one in order to get this to work faster...

    print('f_name template', res_file_name_template)
    channels = list(channels)
    pipe = MultiSequenceModule([
        MultiDistributorModule([
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=SequenceModule([DistributorModule([
                                          Sen1Floods11DataDingsDatasource(data_folder, TYPE_S2, split, in_memory=False),
                                          Sen1Floods11DataDingsDatasource(data_folder, TYPE_S1, split, in_memory=False)
                                      ]),
                                      RangeClippingNormalizationModule('range.json')]),
                                  dataset_name='features'),
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, TYPE_LABEL,
                                                                              split, in_memory=False),
                                  dataset_name='labels')
        ]),
        PlotSegmentationOverlay('Image Wise Label Distribution',
                               reference_criterion=NameSelectionCriterion('features'),
                               ground_truth_criterion=NameSelectionCriterion('labels'),
                               luminance_channels=channels,
                               label_image_title_template='Labels for image {id} in "{region}" ({channel})',
                               ref_image_title_template='{channel} of ({id} in "{region}")',
                               pred_image_title_template='Predictions for image {id} in "{region}" ({channel})',
                               res_file_name_template=res_file_name_template,
                               vmax=0.1 if len(channels) > 1 else 1.0,
                               prediction_module=label_pipe,
                               prediction_selection_criterion=NameSelectionCriterion('valid_prediction'),
                               save=res_file_name_template is not None,
                               show=show
                           )
    ])
    with hook.open_artifact_file('pipeline.json', 'w') as fd:
        serialize(fd, pipe)
    sum = Summary([], lambda: '')
    sum.set_hook(hook)
    pipe(sum)
    finish_visualization()

def finish_visualization(successful: bool = True):
    if successful:
        utils.shutil_overwrite_dir(hook.temp_dir_name, obs.dir)
    utils.del_dir_content(hook.temp_dir_name)

def plot_pipeline(args: argparse.Namespace):
    from experiments.simple_methods import raw_pipeline
    feature_spaces = {#'HSV(O3)+cAWEI+cNDWI':
    #                       {'filter': 'lee_improved',
    #                        'method': 'GB-RF',
    #                        'boosting_type': 'gbdt',
    #                        'num_leaves':64,
    #                        'max_depth': -1,
    #                        'learning_rate': 0.1,
    #                        'n_estimators': 200,
    #                        'subsample_for_bin': 4 * 512 * 512,
    #                        'class_weight': None,
    #                        'min_split_gain': 0.0,
    #                        'min_child_weight': 50.0,
    #                        'min_child_samples': 100,
    #                        'reg_alpha': 0.0,
    #                        'reg_lambda': 1.0,
    #                        'z_feature_space': 'HSV(O3)+cAWEI+cNDWI'},
                      'SAR_HSV(O3)+cAWEI+cNDWI':
                          {'filter': 'lee_improved',
                           'method': 'GB-RF',
                           'boosting_type': 'gbdt',
                           'num_leaves': 128,
                           'max_depth': -1,
                           'learning_rate': 0.1,
                           'n_estimators': 200,
                           'subsample_for_bin': 512 * 512,
                           'class_weight': None,
                           'min_split_gain': 0.0,
                           'min_child_weight': 50.0,
                           'min_child_samples': 100,
                           'reg_alpha': 0.0,
                           'reg_lambda': 1.0,
                           'z_feature_space': 'HSV(O3)+cAWEI+cNDWI',
                           'seed': 797387},
                      # 'SAR':
                      #     {'filter': 'lee_improved',
                      #      'method': 'GB-RF',
                      #      'boosting_type': 'gbdt',
                      #      'num_leaves': 2,
                      #      'max_depth': -1,
                      #      'learning_rate': 0.1,
                      #      'n_estimators': 50,
                      #      'subsample_for_bin': 512 * 512,
                      #      'class_weight': None,
                      #      'min_split_gain': 0.0,
                      #      'min_child_weight': 50.0,
                      #      'min_child_samples': 100,
                      #      'reg_alpha': 0.0,
                      #      'reg_lambda': 1.0,
                      #      'z_feature_space': 'SAR'}
                      }
    cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(args.data_folder,
                                                                                                ['lee_improved'],
                                                                                                True,
                                                                                                eval_split=args.split)
    label_pipe_cache = SimpleCachedPipeline()
    feature_pipe_cache = SimpleCachedPipeline()
    total_cache = SimpleCachedPipeline()
    for feature_space, config in feature_spaces.items():
        if 'SAR' in feature_space:
            vis_channels = [['VH']]
            if '_' in feature_space:
                vis_channels.append(['Red', 'Green', 'Blue'])
        else:
            vis_channels = [['Red', 'Green', 'Blue']]
        for set_of_vis_channels in vis_channels:
            try:
                obs.start({})
                pipe = raw_pipeline(args.data_folder, args.split, label_pipe_cache, feature_pipe_cache, config,
                                    args.seed, cons_by_filter, False)
                total_cache.set_module(pipe)
                visualize_classification(args.data_folder, hook, total_cache, args.show,
                                         args.dataset_label_template, args.split, set_of_vis_channels)
            except Exception as e:
                obs.fail()
                print('Execution Failed!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                finish_visualization(False)
            else:
                obs.complete(None)

def plot_exported_gbdt(df, args: argparse.Namespace):
    for plot_prob in [True, False]:
        print('Plotting probability:', plot_prob)
        try:
            obs.start({'plotting_probability': plot_prob, 'split': args.split, 'labels': args.label_type,
                       'force_f_name': args.force_f_name, 'data_folders': df})
            loading = Sen1Floods11DataDingsDatasource(df, TYPE_GBDT_WEAK_LABEL, args.split, in_memory=False,
                                                      force_f_name=args.force_f_name)
            if plot_prob:
                loading = SequenceModule([
                    loading,
                    WhitelistModule(['GBDT-Water-Prob'])
                ])
            else:
                loading = SequenceModule([
                    loading,
                    ProbabilityToValue()
                ])
            pipe = PipelineAdaptorModule(selection_criterion=None,
                                         pipe_module=loading,
                                         dataset_name='predictions')
            res = visualize_classification(df, hook, pipe, args.show,
                                           args.dataset_label_template, args.split, is_probability=plot_prob,
                                           label_type=args.label_type, calculate_metrics=args.calc_metrics)
        except Exception as e:
            obs.fail()
            print('Execution Failed!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            finish_visualization(False)
        else:
            obs.complete(res)
            if res is not None:
                print('Mean Accuracy:', res[0])
                print('Mean IoU:', res[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('--split', dest='split', default=SPLIT_TEST, type=str)
    parser.add_argument('--seed', dest='seed', default=42, type=int)
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--calc_metrics', dest='calc_metrics', action='store_true')
    parser.add_argument('--no_metrics', dest='calc_metrics', action='store_false')
    parser.add_argument('--label_type', dest='label_type', default=TYPE_LABEL, type=str)
    parser.add_argument('--force_f_name', dest='force_f_name', default=None, type=str)
    parser.add_argument('--dataset_template', dest='dataset_template', default=None, type=str)
    parser.add_argument('--dataset_fs_vis', dest='dataset_fs_vis', default=None, type=str)
    parser.add_argument('--dataset_label_template', dest='dataset_label_template', default=None, type=str)
    parser.add_argument('--filter_method', dest='filter_method', default=None, type=str)
    parser.add_argument('--feature_spaces', dest='feature_spaces', nargs='+',
                        default=['SAR', 'HSV(O3)+cAWEI+cNDWI', 'SAR_HSV(O3)+cAWEI+cNDWI'])
    parser.add_argument('--feature_space_name', dest='feature_space_name', default='z_feature_space')

    parser.add_argument('--additional_data_folders', nargs='+', dest='additional_data_folders', default=None,
                        help='Varags param for additional data-folders that should be considered for finding '
                             'datadings-files')
    parser.set_defaults(show=True, save=True, calc_metrics=True)
    args = parser.parse_args()

    obs = DirectFileStorage(args.save_folder, redirect_sysout=False, do_host=False)
    print('Starting hook')
    # --dataset_label_template "gbdt_vis={is_prob}_{n_saves}"
    # --dataset_template "{region}_{id}_{channel}"

    df = args.data_folder
    if args.additional_data_folders is not None:
        df = [df] + args.additional_data_folders
    with RunHook(obs, None) as hook:
        if args.dataset_template is not None:
            try:
                obs.start({'filter_method': args.filter_method,
                           'vis_type': 'Sentinel-1',
                           'split': args.split})
                visualize_dataset_s1(df, hook, args.dataset_template, args.split, show=args.show, filter_method=args.filter_method)
            except Exception as e:
                obs.fail()
                print('Execution Failed!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                finish_visualization(False)
            else:
                obs.complete(None)
            try:
                obs.start({'vis_type': 'Sentinel-2',
                           'split': args.split})
                visualize_dataset_s2(df, hook, args.dataset_template, args.split, show=args.show, filter_method=args.filter_method)
            except Exception as e:
                obs.fail()
                print('Execution Failed!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                finish_visualization(False)
            else:
                obs.complete(None)
        if args.dataset_fs_vis is not None:
            try:
                obs.start({})
                visualize_dataset_feature_spaces(df, hook, args.dataset_fs_vis, args.split,
                                                 show=args.show)
            except Exception as e:
                obs.fail()
                print('Execution Failed!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                finish_visualization(False)
            else:
                obs.complete(None)
        if args.dataset_label_template is not None:
            plot_exported_gbdt(df, args)
