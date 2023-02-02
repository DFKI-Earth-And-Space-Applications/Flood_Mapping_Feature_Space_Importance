import time

import pandas as pd

from plot_utils import *
from utils import *
from queries import *
from experiment_processing import *

def plot_comparison(folder: str, df: pd.DataFrame, metric_name: str, modules: List[str], name_map: Dict[str, str],
                    show: bool, save: bool, has_filters: bool,
                    module_param_filters: Optional[List[Dict[str, Any]]]):
    params_of_interest = [[COLUMN_PIPELINE_TYPE, COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME]]
    print(f'Comparing params_of_interest={params_of_interest} for metric {metric_name} with module filters '
          f'{module_param_filters}.')
    param_boxplot(df, lambda param: f'Comparison{f" ({module_param_filters})" if has_filters else ""}',
                  lambda param: path.join(folder, f'{"_".join(modules)}_{param}_{metric_name}'
                                                  f'{"_" + join_multiple_config_values(module_param_filters) if has_filters else ""}'
                                                  f'_boxplots'),
                  params_of_interest,
                  metric_name, name_map,
                  show, save)

def broadcast_params(ref_ls: List[str], param: Optional[List[Any]], param_name: str, ref_name: str) -> List[Any]:
    if param is None:
        param = [None]
    assert len(param) == 1 or len(ref_ls) == len(param_name), \
            f'{param_name} should either be a single value or be one value per value of {ref_name}!'
    # enforce identical length
    if len(param) == 1:
        param = param * len(ref_ls)
    return param

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('plot_folder')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--modules', nargs='+', dest='modules', default=None)
    parser.add_argument('--module_param_filters', action=StoreMultipleDictKeyPair, nargs='+', dest='module_filters', default=None)
    parser.add_argument('--metrics', nargs='+', dest='metrics', default=None)
    parser.add_argument('--experiments', nargs='+', dest='experiments', default=None)
    parser.add_argument('--label_criteria', nargs='+', dest='label_criteria', default=None)
    parser.add_argument('--name_map', action=StoreDictKeyPair, nargs='+', dest='name_map', default=None)
    parser.add_argument('--expand_adaptors', dest='expand_adaptors', action='store_true')
    parser.add_argument('--no-expand_adaptors', dest='expand_adaptors', action='store_false')
    parser.add_argument('--adaptor_module', dest='adaptor_module', type=str)
    parser.set_defaults(show=True, save=True, expand_adaptors=False)
    args = parser.parse_args()
    assert args.modules is not None and len(args.modules) >= 2, 'Cannot compare less than 2 modules!!!'
    folder = args.plot_folder
    ensure_is_dir(folder)
    specified_modules = args.modules
    metrics: Iterable[str] = args.metrics if args.metrics is not None else METRIC_NAME_MAP.keys()
    experiments: List[str] = broadcast_params(specified_modules, args.experiments, 'experiments', 'modules')
    label_criteria: List[str] = broadcast_params(specified_modules, args.label_criteria, 'label_criteria', 'modules')
    has_filters: bool = args.module_filters is not None
    module_filters: List[Optional[Dict[str, Any]]] = broadcast_params(specified_modules, args.module_filters, 'module_param_filters', 'modules')
    db = ReadOnlyPandasDatabase(args.data_folder)
    modules = [find_closest_metric_for_module(db, module, experiment_name=ex, module_param_filter=param_filter,
                                              label_criterion=label_criterion, keep_experiment_table=True)
               for module, ex, param_filter, label_criterion in zip(specified_modules, experiments,
                                                                    module_filters, label_criteria)]
    # convert the list of predecessor-successor dataframes into a single dataframe...
    # each predecessor corresponds to one module of interest, each successor to the corresponding metric
    module_df: pd.DataFrame = pd.concat(modules)
    # now we often have a sklearn adaptor, so fill the transformer-name column with a value
    # notice that currently the code is only correct if the parameters provided by the requested modules are filled in
    if args.expand_adaptors:
        module_df = merge_with_adaptors(db, module_df, COLUMN_PIPELINE_PREDECESSOR, adaptor_module=args.adaptor_module)
    else:
        module_df[COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME] = np.nan
    module_df = pd.merge(left=module_df, right=db.create_pipeline_table_if_absent(), left_on=COLUMN_PIPELINE_PREDECESSOR, right_index=True)
    module_df = module_df.drop(columns=[COLUMN_PIPELINE_ROOT, COLUMN_PIPELINE_FINAL])
    db.save() # Free memory
    for metric in metrics:
        if METRIC_NAME_MAP[metric] in db:
            print('Merging with metric', metric)
        else:
            print('Metric', metric, 'is not present. Skipping.')
            continue
        t = time.time()
        metric_df = merge_with_metrics_and_clear_common_columns(db, module_df, COLUMN_PIPELINE_SUCCESSOR,
                                                                METRIC_NAME_MAP[metric],
                                                                retain_columns=(COLUMN_PIPELINE_TYPE,COLUMN_PIPELINE_T_NUM))
        print(f'Merge took {time.time() - t:.3f}s.')
        plot_comparison(folder, metric_df, metric, specified_modules, args.name_map,
                        args.show, args.save, has_filters, module_filters)