import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas.core.groupby as groupby
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from queries import *
from plot_utils import *
from utils import StoreDictKeyPair, StoreMultiDict

def criterion_file_add(args: argparse.Namespace) -> str:
    return f'{"" if args.label_criterion is None else f"_{args.label_criterion}"}' \
           f'{"" if args.prediction_criterion is None else f"_{args.prediction_criterion}"}'

def plot_metric_histogram(ax:plt.Axes, values: pd.DataFrame, metric_name: str, clazz: Optional[int],
                          remaining_columns: Union[str, Iterable[str]] = STANDARD_RETAIN_COLUMNS,
                          add_title: bool = False):
    means = perform_averaging(values, [c for c in values.columns if c != COLUMN_METRIC_VALUE], drop_class=False).rename(columns={COLUMN_METRIC_VALUE+'_mean':COLUMN_METRIC_VALUE})
    print(f'Describing {metric_name} for class {clazz}:')
    print(means.describe())
    best_idx = np.argmax(means[COLUMN_METRIC_VALUE])
    print(f'According to {metric_name} and class {clazz}, the best configuration is:')
    print(means.reset_index().iloc[best_idx])
    hist, edges = np.histogram(means[COLUMN_METRIC_VALUE].dropna(), bins='auto', range=METRIC_RANGE_MAP[metric_name])
    center_values = np.convolve(edges, np.array([0.5, 0.5]), mode='valid')
    #plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    x_values = np.arange(hist.shape[0])
    ax.grid(visible=True)
    ax.bar(x_values, height=hist, width=0.8)#,
           #tick_label=[f'{x:.3f}' for x in center_values])
    #ax.set_xlim(xmin=np.min(center_values), xmax=np.max(center_values))
    if hist.shape[0] > 10:
        positions = np.array([i * hist.shape[0] // 10 for i in range(10)] + [hist.shape[0]-1])
    else:
        positions = x_values
    ax.set_title(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}'
                  f'{metric_name} Histogram{"" if clazz is None else f" for class {FLOOD_LABELS[clazz]}"}')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{center_values[p]:.3f}' for p in positions], rotation=45, ha='right')
    ax.set_xlabel(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}'
                  f'{metric_name}{"" if clazz is None else f" for class {FLOOD_LABELS[clazz]}"}')
    ax.set_ylabel('Occurrences')

def plot_class_scatter_plot(ax:plt.Axes, values: pd.DataFrame, metric_name: str, classes: Union[np.array, pd.Series],
                            remaining_columns: Union[str, Iterable[str]] = STANDARD_RETAIN_COLUMNS,
                            add_stds: bool = False, add_title: bool = True):
    assert len(classes) == 2
    sorted_classes = np.sort(classes)
    means = perform_averaging(values, [c for c in values.columns if c != COLUMN_METRIC_VALUE], drop_class=False,
                              reset_index=False)
    means = means.rename(columns={COLUMN_METRIC_VALUE+'_mean':COLUMN_METRIC_VALUE})
    first_class, second_class = means.xs(sorted_classes[0], level=COLUMN_METRIC_CLASS), \
                                means.xs(sorted_classes[1], level=COLUMN_METRIC_CLASS)
    assert first_class.shape == second_class.shape
    merged = pd.merge(left=first_class, right=second_class, left_index=True, right_index=True, suffixes=('_x', '_y'))
    mean_xs, mean_ys = merged[COLUMN_METRIC_VALUE+'_x'], merged[COLUMN_METRIC_VALUE+'_y']
    ax.grid(visible=True)
    ax.scatter(mean_xs, mean_ys, s=plt.rcParams['lines.markersize']/2)
    ax.set_xlim(*METRIC_RANGE_MAP[metric_name])
    if add_title:
        ax.set_title(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}{metric_name}')
    ax.set_xlabel(f'Class {FLOOD_LABELS[sorted_classes[0]]}')
    ax.set_ylim(*METRIC_RANGE_MAP[metric_name])
    ax.set_ylabel(f'Class {FLOOD_LABELS[sorted_classes[1]]}')
    print(means.describe())
    if add_stds and COLUMN_METRIC_DATA_ID in values.columns:
        stds: pd.DataFrame = means[COLUMN_METRIC_VALUE+'_mstd']
        first_std, second_std = stds.xs(sorted_classes[0], level=COLUMN_METRIC_CLASS), \
                                stds.xs(sorted_classes[1], level=COLUMN_METRIC_CLASS)
        stds = pd.merge(left=first_std, right=second_std, left_index=True, right_index=True, suffixes=('_x', '_y'))
        ax.errorbar(mean_xs, mean_ys,
                    xerr=stds[COLUMN_METRIC_VALUE + '_x'], yerr=stds[COLUMN_METRIC_VALUE + '_y'])


def plot_metric_scatter_plot(ax:plt.Axes, values: pd.DataFrame, metric_names: List[str], clazz: Optional[int],
                             remaining_columns: Union[str, Iterable[str]] = STANDARD_RETAIN_COLUMNS,
                             add_stds: bool = False, plot_regression_line: bool = True, add_title: bool = True):
    means = perform_averaging(values, [c for c in values.columns if c != COLUMN_METRIC_VALUE], drop_class=False, reset_index=False)
    means = means.rename(columns={COLUMN_METRIC_VALUE+'_mean':COLUMN_METRIC_VALUE})
    first_metric, second_metric = means.xs(metric_names[0], level=COLUMN_MCB_METRIC_NAME), \
                                  means.xs(metric_names[1], level=COLUMN_MCB_METRIC_NAME)
    merged = pd.merge(left=first_metric, right=second_metric, left_index=True, right_index=True, suffixes=('_x', '_y'))
    mean_xs, mean_ys = merged[COLUMN_METRIC_VALUE+'_x'], merged[COLUMN_METRIC_VALUE+'_y']
    ax.scatter(mean_xs, mean_ys, s=plt.rcParams['lines.markersize'] / 2)
    ax.grid(visible=True)
    if add_title:
        ax.set_title(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}{metric_names[0]} vs. '
                     f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}{metric_names[1]}'
                     f'{"" if clazz is None else f" for class {FLOOD_LABELS[clazz]}"}')
    ax.set_xlim(*METRIC_RANGE_MAP[metric_names[0]])
    ax.set_xlabel(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}'
                  f'{metric_names[0]}')
    ax.set_ylim(*METRIC_RANGE_MAP[metric_names[1]])
    ax.set_ylabel(f'{"Mean " if COLUMN_METRIC_DATA_ID in values.columns else ""}'
                  f'{metric_names[1]}')
    print(means.describe())
    if add_stds and COLUMN_METRIC_DATA_ID in values.columns:
        stds: pd.DataFrame = means[COLUMN_METRIC_VALUE+'_mstd']
        first_std, second_std = stds.xs(metric_names[0], level=COLUMN_MCB_METRIC_NAME), \
                                stds.xs(metric_names[1], level=COLUMN_MCB_METRIC_NAME)
        stds = pd.merge(left=first_std, right=second_std, left_index=True, right_index=True, suffixes=('_x', '_y'))
        ax.errorbar(mean_xs, mean_ys,
                    xerr=stds[COLUMN_METRIC_VALUE + '_x'], yerr=stds[COLUMN_METRIC_VALUE + '_y'])
    if plot_regression_line:
        plot_scatter_regression_line(ax, mean_xs, mean_ys,
                                     METRIC_RANGE_MAP[metric_names[0]], METRIC_RANGE_MAP[metric_names[1]])

def plot_general_statistics(db: PandasDatabase, folder: str, args: argparse.Namespace, orig_metric_modules: pd.DataFrame, add_title: bool):
    folder = path.join(folder, 'general_statistics')
    ensure_is_dir(folder)
    metrics = args.metrics if args.metrics is not None else METRIC_NAME_MAP.keys()
    for metric in metrics:
        if METRIC_NAME_MAP[metric] not in db:
            print(f'Metric {metric} not found. Skipping!', file=sys.stderr)
            continue
        config_metric_modules = merge_with_metrics_and_clear_common_columns(db, orig_metric_modules,
                                                                            COLUMN_PIPELINE_PREDECESSOR,
                                                                            METRIC_NAME_MAP[metric],
                                                                            retain_columns=(COLUMN_METRIC_VALUE,
                                                                                            COLUMN_METRIC_CLASS,
                                                                                            COLUMN_METRIC_DATA_ID,
                                                                                            COLUMN_METRIC_STEP_ID,
                                                                                            COLUMN_PIPELINE_T_NUM))
        if len(config_metric_modules) == 0:
            print(f'Found no evaluations for metric {metric} ({METRIC_NAME_MAP[metric]})! Skipping!', file=sys.stderr)
            continue

        unique_classes = config_metric_modules[
            COLUMN_METRIC_CLASS].unique() if COLUMN_METRIC_CLASS in config_metric_modules.columns else [None]
        n_classes = len(unique_classes)
        scatter_plot = n_classes == 2
        fig, axes, ld = create_figure_with_subplots(f'{args.experiment_name}'
                                                    f'{f"({args.config_values})" if args.config_values is not None else ""}:'
                                                    f'\n{metric} ', 1, n_classes + 1 if scatter_plot else n_classes,
                              fig_width=args.fig_width,
                              fig_height=args.fig_height)
        config_metric_modules = config_metric_modules.drop(columns=[COLUMN_PIPELINE_T_NUM])
        for i, clazz in enumerate(unique_classes):
            modules_to_use = config_metric_modules if clazz is None else config_metric_modules.loc[
                config_metric_modules[COLUMN_METRIC_CLASS] == clazz]
            plot_metric_histogram(axes[0, i], modules_to_use, metric, clazz, add_title=add_title)
        if scatter_plot:
            plot_class_scatter_plot(axes[0, -1], config_metric_modules, metric, unique_classes,
                                    add_stds=args.std_errors, add_title=args.add_title)
        save_and_show(fig, path.join(folder, f'{metric}'
                                             f'{"_" + join_config_values(args.config_values) if has_config_values else ""}'
                                             +criterion_file_add(args)+
                                             f'_final_distribution'), args.show, ld, args.save)

def plot_combinations(db: PandasDatabase, folder: str, orig_metric_modules: pd.DataFrame, args: argparse.Namespace):
    folder = path.join(folder, 'metric_combinations')
    ensure_is_dir(folder)
    has_config_values = args.config_values is not None and len(args.config_values) > 0
    combinations = [s.split(',') for s in args.combinations] if args.combinations is not None else COMBINATIONS
    for m1, m2 in combinations:
        if METRIC_NAME_MAP[m1] not in db:
            print(f'metric {m1} not found. Skipping!', file=sys.stderr)
            continue
        if METRIC_NAME_MAP[m2] not in db:
            print(f'metric {m2} not found. Skipping!', file=sys.stderr)
            continue
        fig, axes, ld = create_figure_with_subplots(f'{args.experiment_name}'
                                                    f'{f"({args.config_values})" if args.config_values is not None else ""}:'
                                                    f'\n{m1} vs {m2} ', 2, 2,
                              fig_width=args.fig_width,
                              fig_height=args.fig_height)
        for j, plot_regression_line in enumerate([True, False]):
            config_metric_modules1 = merge_with_metrics_and_clear_common_columns(db, orig_metric_modules,
                                                                                 COLUMN_PIPELINE_PREDECESSOR,
                                                                                 METRIC_NAME_MAP[m1])

            if len(config_metric_modules1) == 0:
                print(f'Found no evaluations for metric {m1} ({METRIC_NAME_MAP[m1]})! Skipping!',
                      file=sys.stderr)
                continue
            config_metric_modules1[COLUMN_MCB_METRIC_NAME] = m1
            unique_classes1 = config_metric_modules1[
                COLUMN_METRIC_CLASS].unique() if COLUMN_METRIC_CLASS in config_metric_modules1.columns else [None]

            config_metric_modules2 = merge_with_metrics_and_clear_common_columns(db, orig_metric_modules,
                                                                                 COLUMN_PIPELINE_PREDECESSOR,
                                                                                 METRIC_NAME_MAP[m2])
            if len(config_metric_modules2) == 0:
                print(f'Found no evaluations for metric {m2} ({METRIC_NAME_MAP[m2]})! Skipping!',
                      file=sys.stderr)
                continue

            config_metric_modules2[COLUMN_MCB_METRIC_NAME] = m2
            unique_classes2 = config_metric_modules2[
                COLUMN_METRIC_CLASS].unique() if COLUMN_METRIC_CLASS in config_metric_modules2.columns else [None]

            config_metric_modules = pd.concat((config_metric_modules1, config_metric_modules2))
            assert len(unique_classes1) == len(unique_classes2) == 2
            for i, clazz in enumerate(unique_classes1):
                modules_to_use = config_metric_modules if clazz is None else config_metric_modules.loc[
                    config_metric_modules[COLUMN_METRIC_CLASS] == clazz]
                plot_metric_scatter_plot(axes[j, i], modules_to_use, [m1, m2], clazz, add_stds=args.std_errors,
                                         plot_regression_line=plot_regression_line, add_title=args.add_title)
        save_and_show(fig, path.join(folder, f'{m1}_vs_{m2}'
                                             f'{"_" + join_config_values(args.config_values) if has_config_values else ""}'
                                             +criterion_file_add(args)+
                                             f'_final_distribution'), args.show, ld, args.save)

def plot_param_analysis(db: PandasDatabase, folder: str, orig_metric_modules: pd.DataFrame, args: argparse.Namespace):
    folder = path.join(folder, 'param_analysis')
    ensure_is_dir(folder)
    params_of_interest: Optional[List[str]] = args.params_of_interest
    if params_of_interest is None or not params_of_interest:
        return
    metrics = args.metrics if args.metrics is not None else METRIC_NAME_MAP.keys()
    has_config_values = args.config_values is not None
    no_seed_params = [param for param in params_of_interest if param not in ['seed', '_seed']]
    seed_params = [param for param in params_of_interest if param in ['seed', '_seed']]
    for metric in metrics:
        if METRIC_NAME_MAP[metric] not in db:
            print(f'metric {metric} not found. Skipping!', file=sys.stderr)
            continue
        print(f'Performing Param analysis for metric {metric} and params_of_interest={params_of_interest}')
        if no_seed_params:
            config_metric_modules = merge_with_metrics_and_clear_common_columns(db, orig_metric_modules,
                                                                                COLUMN_PIPELINE_PREDECESSOR,
                                                                                METRIC_NAME_MAP[metric])
            if len(config_metric_modules) > 0:
                param_boxplot(config_metric_modules, lambda param: f'{args.experiment_name}'
                                                                   f'{f"({args.config_values})" if has_config_values else ""}:'
                                                                   f'\nTested values of {param}',
                              lambda param:path.join(folder, f'{param}_{metric}'
                                                             f'{"_" + join_config_values(args.config_values) if has_config_values else ""}'
                                                             +criterion_file_add(args)+
                                                             f'_boxplots'),
                              no_seed_params,
                              metric, args.value_map,
                              args.show, args.save,
                              add_title=args.add_title,
                              add_mean_line=args.mean_line,
                              metric_y_axis=args.metric_y_axis,
                              fig_width=args.fig_width,
                              fig_height=args.fig_height,
                              show_x_axis_label=args.x_axis_label,
                              max_unique_classes=args.max_n_classes,
                              with_title=args.with_title)
            else:
                print(f'Found no evaluations for metric {metric} ({METRIC_NAME_MAP[metric]})! Skipping!',
                      file=sys.stderr)
        if seed_params:
            config_metric_modules = merge_with_metrics_and_clear_common_columns(db, orig_metric_modules,
                                                                                COLUMN_PIPELINE_PREDECESSOR,
                                                                                METRIC_NAME_MAP[metric])
            if len(config_metric_modules) == 0:
                print(f'Found no evaluations for metric {metric} ({METRIC_NAME_MAP[metric]})! Skipping!',
                      file=sys.stderr)
                continue
            param_boxplot(config_metric_modules, lambda param: f'{args.experiment_name}'
                                                               f'{f"({args.config_values})" if has_config_values else ""}:'
                                                               f'\nTested values of {param}',
                          lambda param: path.join(folder, f'{param}_{metric}'
                                                          f'{"_" + join_config_values(args.config_values) if has_config_values else ""}'
                                                          +criterion_file_add(args)+
                                                          f'_boxplots'),
                          seed_params,
                          metric, args.value_map,
                          args.show, args.save,
                          add_title=args.add_title,
                          add_mean_line=args.mean_line,
                          metric_y_axis=args.metric_y_axis,
                          fig_width=args.fig_width,
                          fig_height=args.fig_height,
                          show_x_axis_label=args.x_axis_label,
                          max_unique_classes=args.max_n_classes,
                          with_title=args.with_title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('plot_folder')
    parser.add_argument('experiment_name')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--metrics', nargs='+', dest='metrics', default=None)
    parser.add_argument('--combinations', nargs='+', dest='combinations', default=None)
    parser.add_argument('--label_criterion', dest='label_criterion', default=None)
    parser.add_argument('--prediction_criterion', dest='prediction_criterion', default=None)
    parser.add_argument("--config_values", dest="config_values", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")
    parser.add_argument('--std_errors', dest='std_errors', action='store_true')
    parser.add_argument('--no-std_errors', dest='std_errors', action='store_false')
    parser.add_argument('--params_of_interest', nargs='+', dest='params_of_interest', default=None)
    parser.add_argument('--plot_combinations', dest='plot_combinations', action='store_true')
    parser.add_argument('--no-plot_combinations', dest='plot_combinations', action='store_false')
    parser.add_argument('--plot_general', dest='plot_general', action='store_true')
    parser.add_argument('--no-plot_general', dest='plot_general', action='store_false')
    parser.add_argument('--plot_params', dest='plot_params', action='store_true')
    parser.add_argument('--no-plot_params', dest='plot_params', action='store_false')
    parser.add_argument("--combine_columns", dest="combine_columns", action=StoreMultiDict, nargs="+", default=None)
    # Additional arguments for publishing in Remote Sensing of Environment.
    # Note that the defaults have been set upon creation of these arguments, so that they match the requirements of
    # Remote Sensing of Environment
    parser.add_argument('--rc_params', dest='rc_params', action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL",
                        default={'font.family': 'arial', 'font.size': 11})
    parser.add_argument('--add_title', dest='add_title', action='store_true')
    parser.add_argument('--no-title', dest='add_title', action='store_false')
    # Use the Metric name on the y-Axis for param-analysis plots [doesn't make sense to have them there if the title also contains the metric...]
    parser.add_argument('--metric_y_axis', dest='metric_y_axis', action='store_true')
    parser.add_argument('--no-metric_y_axis', dest='metric_y_axis', action='store_false')
    parser.add_argument('--x_axis_label', dest='x_axis_label', action='store_true')
    parser.add_argument('--no-x_axis_label', dest='x_axis_label', action='store_false')
    parser.add_argument('--max_n_classes', dest='max_n_classes', default=1, type=int)
    parser.add_argument('--add_mean_line', dest='mean_line', action='store_true')
    parser.add_argument('--no-mean_line', dest='mean_line', action='store_false')
    parser.add_argument('--value_map', dest='value_map', action=StoreDictKeyPair, default={None: 'no re-balancing'}) # 'nan': 'no re-balancing',
    parser.add_argument('--fig_width', dest='fig_width', default=6.0, type=float)
    parser.add_argument('--fig_height', dest='fig_height', default=6.0, type=float)
    parser.add_argument('--with_figure_title', dest='with_title', action='store_true')
    parser.add_argument('--no-figure_title', dest='with_title', action='store_false')
    parser.set_defaults(show=True, save=True, std_errors=False, plot_general=True, plot_combinations=True,
                        plot_params=True, add_title=False, metric_y_axis=True, mean_line=False, x_axis_label=False,
                        with_title=False)
    parser.set_defaults()
    args = parser.parse_args()
    folder = path.join(args.plot_folder, args.experiment_name)
    if args.config_values is not None:
        folder = path.join(folder, join_config_values(args.config_values))
    ensure_is_dir(folder)
    db = ReadOnlyPandasDatabase(args.data_folder)
    orig_metric_modules: pd.DataFrame = last_experiment_evaluation_metric_modules(db, args.experiment_name,
                                         metric_label_criterion=args.label_criterion,
                                         metric_prediction_criterion=args.prediction_criterion,
                                         merge_columns=args.combine_columns, config_filter=args.config_values)
    orig_metric_modules = orig_metric_modules.dropna(axis='columns', how='all')
    has_config_values = args.config_values is not None and len(args.config_values) > 0
    if has_config_values:
        orig_metric_modules = orig_metric_modules.loc[np.all(orig_metric_modules[
                                                                 list(args.config_values.keys())
                                                             ] == list(args.config_values.values()),
                                                             axis=1)]
    plt.rcParams.update(args.rc_params)
    print('Font Sizes:', plt.rcParams['font.size'], plt.rcParams['legend.fontsize'], plt.rcParams['legend.title_fontsize'])
    print('Font Properties:', plt.rcParams['font.family'])
    print('Font Weight:', plt.rcParams['font.weight'])
    if args.plot_general:
        plot_general_statistics(db, folder, args, orig_metric_modules, add_title=args.add_title)
    if args.plot_combinations:
        try:
            plot_combinations(db, folder, orig_metric_modules, args)
        except Exception:
            print('Caught exception...')
            print(traceback.format_exc())
    if args.plot_params:
        plot_param_analysis(db, folder, orig_metric_modules, args)
