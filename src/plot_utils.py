import os
import traceback
from numbers import Number

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os.path as path
import numpy as np
from typing import Tuple, List, Iterable, Optional, Union, Callable, Dict, Any

from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d, Axes3D
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import utils
from queries import perform_averaging
from experiment_processing import COLUMN_METRIC_CLASS, COLUMN_METRIC_DATA_ID, COLUMN_METRIC_VALUE, COLUMN_EXPERIMENT_SEED

import pandas as pd


def create_figure_with_subplots(title: str, n_rows: int, num_cols: int, subplot_kw: Optional[Dict[str, Any]] = None,
                                fig_width = 6.0, fig_height = 6.0, with_title: bool = True) \
        -> Tuple[plt.Figure, np.ndarray, int]:
    #print((fig_width, fig_height))
    if subplot_kw is None:
        subplot_kw = {}
    fig: plt.Figure = plt.figure(figsize=[fig_width * num_cols, fig_height * n_rows])
    if with_title:
        fig.suptitle(title)
    axs = fig.subplots(n_rows, num_cols, subplot_kw=subplot_kw)
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        axs = axs.reshape((n_rows, num_cols))
    return fig, axs, n_rows

def create_figure_with_mosaic(title: str, mosaic_spec: List[List[str]], subplot_kw: Optional[Dict[str, Any]] = None) \
        -> Tuple[plt.Figure, Dict[str, plt.Axes], int]:
    if subplot_kw is None:
        subplot_kw = {}
    num_cols = max([len(ls) for ls in mosaic_spec])
    n_rows = len(mosaic_spec)
    #, constrained_layout=True
    fig: plt.Figure = plt.figure(figsize=[6.0 * num_cols, 6.0 * n_rows])
    if 'title.visibility' not in plt.rcParams or plt.rcParams['title.visibility']:
        fig.suptitle(title)
    axs = fig.subplot_mosaic(mosaic_spec, subplot_kw=subplot_kw)
    return fig, axs, n_rows

def save_and_show(fig: plt.Figure, f_name: str, show: bool, layout_dividend: int, save: bool, eps_dpi: Optional[int] = None,
                  with_title: bool = True):
    assert not (f_name is None and save)
    if with_title:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1 - (0.05 / layout_dividend)))
    else:
        fig.tight_layout()
    if f_name is not None and path.exists(f_name+'.pdf'):
        print(f'WARNING: {f_name}.pdf already exists. Overwriting.')
    if save and f_name is not None:
        try:

            fig.savefig(f_name+'.pdf')
            if eps_dpi is not None:
                fig.savefig(f_name+'.eps', dpi=eps_dpi)
            if path.exists(f_name+'.png'):
                print(f'WARNING: {f_name}.png already exists. Overwriting.')
            fig.savefig(f_name+'.png')
        except Exception:
            print('Caught exception whilst saving!')
            print(traceback.format_exc())
    if show:
        fig.show()
    plt.close(fig)


def ensure_is_dir(dir: str):
    if path.exists(dir) and not path.isdir(dir):
        raise ValueError(f'{dir} exists already but is not a directory!!!')
    elif not path.exists(dir):
        os.makedirs(dir)

FLOOD_LABELS = {
    -1 : 'Unclassified Data',
    0: 'Dry Land',
    1: 'Surface Water' # 'Flooded'
}

FLOOD_CLASSES = np.array([-1, 0, 1])

cmap = cm.get_cmap('tab20c')(np.array([4, 8, 0, 16]))
FLOOD_COLORS = {(i-1): cmap[i] for i in range(3)}

METRIC_INFO_MAP = {
    'Precision': ('per_data.precision.no_clouds', (0.0, 1.0)),
    'Recall': ('per_data.recall.no_clouds', (0.0, 1.0)),
    'IoU(0) Score': ('per_data.iou0.no_clouds', (0.0, 1.0)),
    'IoU Score': ('per_data.iou1.no_clouds', (0.0, 1.0)),
    'F1 Score': ('per_data.f1_score.no_clouds', (0.0, 1.0)),
    'Completeness': ('per_data.completeness.no_clouds', (0.0, 1.0)),
    'Homogeneity': ('per_data.homogenity.no_clouds', (0.0, 1.0)),
    'V1-Measure': ('per_data.v1_measure.no_clouds', (0.0, 1.0)),
    'Accuracy': ('per_data.accuracy.no_clouds', (0.0, 1.0)),
    'Rand Score': ('per_data.rand_score.pair_confusion.no_clouds', (0.0, 1.0)),
    'NC-Rand Score': ('per_data.rand_score.normalized_class_size_pair_confusion.no_clouds', (0.0, 1.0)),
    'Adjusted Rand Score': ('per_data.adjusted_rand_score.pair_confusion.no_clouds', (0.0, 1.0)),
    'Mutual Information': ('per_data.mutual_information_score.no_clouds', (0.0, 1.0)),
    'Adjusted Mutual Information': ('per_data.adjusted_mutual_information_score.no_clouds', (0.0, 1.0)),
    # 'Batch (2) IOU ': ('batch.iou_2', (0.0, 1.0)),
    # 'Batch (4) IOU ': ('batch.iou_4', (0.0, 1.0)),
    # 'Batch (8) IOU ': ('batch.iou_8', (0.0, 1.0)),
    # 'Batch (16) IOU ': ('batch.iou_16', (0.0, 1.0)),
    'Total Precision': ('total.precision.no_clouds', (0.0, 1.0)),
    'Total Recall': ('total.recall.no_clouds', (0.0, 1.0)),
    'Total F1 Score': ('total.f1_score.no_clouds', (0.0, 1.0)),
    'Total Cluster-F1 Score': ('total.cluster_f1_score.no_clouds', (0.0, 1.0)),
    'Total IoU Score': ('total.iou0.no_clouds', (0.0, 1.0)),
    'Total Mutual Information': ('total.mutual_information_score.no_clouds', (0.0, 1.0)),
    #'Total Normalized Mutual Information': ('total.normalized_mutual_information_score.no_clouds', (0.0, 1.0)),
    'Total Adjusted Mutual Information': ('total.adjusted_mutual_information_score.no_clouds', (0.0, 1.0)),
    'Total Completeness': ('total.completeness.no_clouds', (0.0, 1.0)),
    'Total Homogeneity': ('total.homogenity.no_clouds', (0.0, 1.0)),
    'Total V1-Measure': ('total.v1_measure.no_clouds', (0.0, 1.0)),
    'Total Accuracy': ('total.accuracy.no_clouds', (0.0, 1.0)),
    'Total Rand Score': ('total.rand_score.pair_confusion.no_clouds', (0.0, 1.0)),
    'Total NC-Rand Score': ('total.rand_score.normalized_class_size_pair_confusion.no_clouds', (0.0, 1.0)),
    'Total Adjusted Rand Score': ('total.adjusted_rand_score.pair_confusion.no_clouds', (0.0, 1.0)),
    'Max Agree Precision': ('per_data.precision.max_agree.no_clouds', (0.0, 1.0)),
    'Max Agree Recall': ('per_data.recall.max_agree.no_clouds', (0.0, 1.0)),
    'Max Agree IoU(0) Score': ('per_data.iou0.max_agree.no_clouds', (0.0, 1.0)),
    'Max Agree IoU(1) Score': ('per_data.iou1.max_agree.no_clouds', (0.0, 1.0)),
    'Max Agree F1 Score': ('per_data.f1_score.max_agree.no_clouds', (0.0, 1.0)),
    'Max Agree Accuracy': ('per_data.accuracy.max_agree.no_clouds', (0.0, 1.0)),
    'Cluster-F1 Score': ('per_data.cluster_f1_score.no_clouds', (0.0, 1.0)),
}
METRIC_NAME_MAP: Dict[str, str] = {name: key for name, (key, _) in METRIC_INFO_MAP.items()}
METRIC_RANGE_MAP = {name: range for name, (_, range) in METRIC_INFO_MAP.items()}
METRIC_NAME_REVERSE_MAP: Dict[str, str] = {key:name for name, key in METRIC_NAME_MAP.items()}
COMBINATIONS = [('Recall', 'Precision'), ('Total Recall', 'Total Precision'), ('Completeness', 'Homogeneity'),
                ('Total Completeness', 'Total Homogeneity')]
STANDARD_RETAIN_COLUMNS = (COLUMN_METRIC_VALUE, COLUMN_METRIC_VALUE + '_std', COLUMN_METRIC_VALUE + '_mstd',
                           COLUMN_METRIC_DATA_ID)
STANDARD_SEED_RETAIN_COLUMNS = (COLUMN_METRIC_VALUE, COLUMN_METRIC_VALUE + '_std', COLUMN_METRIC_VALUE + '_mstd',
                                COLUMN_METRIC_DATA_ID, '_seed', 'random_state', 'random_seed', COLUMN_EXPERIMENT_SEED)
def param_boxplot(modules: pd.DataFrame, title: Callable, save_file: Callable,
                  params_of_interest: List[Union[List[str], str]],
                  metric_name: str, name_map: Dict[Optional[Union[str, int, float]], Union[Dict[str, str], str]],
                  show: bool, save: bool, add_title: bool = True, add_mean_line: bool = True,
                  metric_y_axis: bool = False, show_x_axis_label: bool = True, max_unique_classes: int = -1,
                  with_title: bool = True, fig_width = 6.0, fig_height = 6.0):

    has_class_column = COLUMN_METRIC_CLASS in modules.columns
    has_data_column = COLUMN_METRIC_DATA_ID in modules.columns
    config_metric_modules = perform_averaging(modules, [c for c in modules.columns if c != COLUMN_METRIC_VALUE],
                                              drop_class=False, reset_index=False, allow_seed_averaging=False)
    unique_classes = config_metric_modules.index.get_level_values(
        COLUMN_METRIC_CLASS).unique() if has_class_column else [None]
    if max_unique_classes > 0 and max_unique_classes < len(unique_classes):
        unique_classes = unique_classes[-max_unique_classes:]
    for param in params_of_interest:
        print(f'Inspecting parameter {param}')
        fig, axes, ld = create_figure_with_subplots(title(param), (2 if add_mean_line else 1), len(unique_classes),
                                                    fig_width=fig_width, fig_height=fig_height, with_title=with_title)
        if param not in ['seed', '_seed', 'random_state', 'random_seed']:
            modules_to_use = perform_averaging(config_metric_modules,
                                               [c for c in modules.columns if c != COLUMN_METRIC_VALUE],
                                               drop_class=False, reset_index=False, allow_seed_averaging=True)
        else:
            modules_to_use = config_metric_modules
        modules_to_use = modules_to_use[COLUMN_METRIC_VALUE+'_mean']
        for j, clazz in enumerate(unique_classes):
            if clazz is not None:
                relevant_values = modules_to_use.xs(clazz, level=COLUMN_METRIC_CLASS)
            else:
                relevant_values = modules_to_use
            sub_dict: List[Tuple[Tuple[str, pd.Series], Tuple[float, float]]] = \
                [(t, (np.nanmax(t[1].dropna().to_numpy().astype(np.float)),
                             np.nanmean(t[1].dropna().to_numpy().astype(np.float))
                             if np.count_nonzero(t[1].notna()) > 0 else (0.0, 0.0)))
                        for t in relevant_values.groupby(level=param, dropna=False)]
            # sort by maximum
            sub_dict: List[Tuple[Tuple[str, pd.Series], Tuple[float, float]]] = [t for t in sorted(sub_dict, key=lambda t: t[1], reverse=True)]
            means = [mean for _, (_, mean) in sub_dict]
            sub_dict: Dict[str, pd.Series] = {k: v.dropna() for (k, v), _ in sub_dict}

            for i, plot_mean_line in enumerate(([True, False] if add_mean_line else [False])):
                print(f'Plotting clazz={clazz} and plot_mean_line={plot_mean_line}')
                ax: plt.Axes = axes[i, j]
                ax.grid(visible=True)
                ax.boxplot(sub_dict.values(), whis=(0., 100.))
                metric_str = f'{"Mean " if has_data_column else ""}{metric_name}'
                if add_title:
                    ax.set_title(metric_str)
                if show_x_axis_label:
                    ax.set_xlabel(f'Tested values of {param}')
                # must be a list to keep the ordering...
                keys = [(None if isinstance(k, Number) and np.isnan(k) else k) for k in sub_dict.keys()]
                ax.set_xticklabels([(k if k not in name_map else
                                     ((name_map[k])[param] if isinstance(name_map[k], dict) else name_map[k])
                                     ) for k in keys],
                                   rotation=45, ha='right')
                if metric_y_axis:
                    ax.set_ylabel(metric_str)
                else:
                    ax.set_ylabel(f'{"" if clazz is None else f"{FLOOD_LABELS[clazz]}"} Score')
                ax.set_ylim(*METRIC_RANGE_MAP[metric_name])
                #ax.tick_params(axis='x', which='major', labelsize=8)
                if plot_mean_line:
                    ax.plot(np.arange(len(means)) + 1, means, label='Mean')
                    ax.legend()

        save_and_show(fig, save_file(param), show, ld, save, with_title=with_title)

def plot_scatter_regression_line(ax: plt.Axes, mean_xs: pd.Series, mean_ys: pd.Series, x_range: Tuple[float, float],
                                 y_range: Tuple[float, float], regression_steps: int = 200, test_y_log: bool = True,
                                 implicit_x_log: bool = False, scale_x: bool = True,
                                 scale_y: bool = True, min_c_exp: int = -5, max_c_exp: int = 3, steps: int = 40):
    def log_add(ser: pd.Series) -> float:
        min_val = ser.min()
        return min_val + utils.EPS if min_val <= 0 else 0.0
    x_scaler, y_scaler, y_log_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    x_add = log_add(mean_xs)
    y_add = log_add(mean_ys)
    mean_reg_xs = mean_xs.to_numpy().astype(np.float32).reshape((-1, 1))
    if implicit_x_log:
        mean_reg_xs = np.log(mean_reg_xs + x_add)
    if scale_x:
        mean_reg_xs = x_scaler.fit_transform(mean_reg_xs)
    mean_reg_ys = mean_ys.to_numpy().astype(np.float32).reshape((-1, 1))
    mean_log_ys = np.log(mean_reg_ys + y_add)
    if scale_y:
        mean_reg_ys = y_scaler.fit_transform(mean_reg_ys)
        mean_log_ys = y_log_scaler.fit_transform(mean_log_ys)
    regressor = RidgeCV(alphas=[10.0 ** x for x in np.linspace(-min_c_exp, max_c_exp, steps)])
    regressor.fit(mean_reg_xs, mean_reg_ys)
    log_regressor = None
    if test_y_log:
        log_regressor = RidgeCV(alphas=[10.0 ** x for x in np.linspace(-min_c_exp, max_c_exp, steps)])
        log_regressor.fit(mean_reg_xs, mean_log_ys)
        use_log = log_regressor.best_score_ > regressor.best_score_
        log_regressor = None if not use_log else log_regressor
    to_draw: np.ndarray = np.linspace(x_range[0], x_range[1], regression_steps)
    if implicit_x_log:
        to_draw = np.log(to_draw + x_add)
    mask = np.isfinite(to_draw)
    if scale_x:
        to_draw[mask] = x_scaler.transform(to_draw[mask].reshape((-1, 1))).flatten()
    predicted = np.empty_like(to_draw)
    predicted[~mask] = np.nan
    if log_regressor is not None:
        predicted[mask] = log_regressor.predict(to_draw[mask].reshape((-1, 1))).flatten()
    else:
        predicted[mask] = regressor.predict(to_draw[mask].reshape((-1, 1))).flatten()
    if scale_y:
        predicted[mask] = y_scaler.inverse_transform(predicted[mask].reshape((-1, 1))).flatten()
    if log_regressor is not None:
        predicted[mask] = np.exp(predicted[mask] - y_add)
    with np.errstate(invalid='ignore'):
        final_mask = np.logical_and(np.isfinite(predicted),
                              np.logical_and(predicted >= y_range[0],
                                             predicted <= y_range[1]))
    if implicit_x_log: # the logarithm is taken implicitly...
        to_draw: np.ndarray = np.linspace(x_range[0], x_range[1], regression_steps).reshape((-1, 1))
        #if scale_x:
        #    to_draw[mask] = x_scaler.inverse_transform(to_draw[mask].reshape((-1, 1)))
    ax.plot(to_draw[final_mask], predicted[final_mask])

def plot_scatter_regression_line2(ax: plt.Axes, mean_xs: pd.Series, mean_ys: pd.Series,
                                  x_range: Tuple[float, float],
                                 y_range: Tuple[float, float], regression_steps: int = 200):
    poly_fun = np.poly1d(np.polyfit(np.log(mean_xs), mean_ys, 1))
    to_draw: np.ndarray = np.linspace(x_range[0], x_range[1], regression_steps)
    predicted = poly_fun(np.log(to_draw))
    final_mask = np.logical_and(predicted >= y_range[0],
                                predicted <= y_range[1])
    ax.plot(to_draw[final_mask], predicted[final_mask], '--')

def join_config_values(config_values: Dict[str, str]) -> str:
    return "_".join(map(lambda t: "-".join(t), config_values.items()))

def join_multiple_config_values(config_values: List[Optional[Dict[str, str]]]) -> str:
    return "_".join(map(lambda t: ("None" if t is None else "-".join(map(lambda v: f'{v[0]}={v[1]}', t.items()))),
                        config_values))


def add_point(ax: Axes3D, x, y, z, fc=None, ec=None, radius=0.005):
    # copied from https://stackoverflow.com/questions/51241367/matplotlib-surface-plot-hides-scatter-points-which-should-be-in-front
    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation = {'z': ((x, y, z), axis_length[1] / axis_length[0]),
                     'y': ((x, z, y), axis_length[2] / axis_length[0] * xy_len / z_len),
                     'x': ((y, z, x), axis_length[2] / axis_length[1] * xy_len / z_len)}
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width=radius, height=radius * ratio, fc=fc, ec=ec)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)