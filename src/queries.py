import argparse
import sys
import time

from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

from utils import StoreDictKeyPair, StoreMultiDict
from experiment_processing import *
from sci_analysis import analyze

def recursive_search(db: PandasDatabase,
                     start_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                     target_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                     search_stop_module_ids: Optional[Union[np.ndarray, pd.Series, pd.Index]],
                     terminate_at_match: bool, print_iter: bool,
                     source_name: str, target_name: str) -> pd.DataFrame:

    def rec_append(to_append_to: List[pd.DataFrame], next_level: pd.DataFrame):
        mask = next_level[target_name].isin(target_module_ids)
        next_found = next_level.loc[mask]
        if len(next_found) > 0:
            to_append_to.append(next_found)
        if terminate_at_match:
            next_level = next_level.loc[~mask]
        return next_level.loc[~next_level[target_name].isin(search_stop_module_ids)]

    relation = db.create_pipeline_successor_table_if_absent()
    predecessor_index = relation.loc[:, source_name]
    found = []
    search_index = pd.DataFrame({source_name: start_module_ids,
                                 target_name: start_module_ids})
    print(f'Searching for Pipeline successors of {len(start_module_ids)} modules.')
    i = 0
    t = time.time()
    search_index = rec_append(found, search_index) # start recursion with depth 0 modules
    while len(search_index) > 0:
        i += 1
        if print_iter:
            print(f'Performing Recursive-Search Iteration {i}')
        next_level = relation.loc[
            predecessor_index.isin(search_index[target_name]), [source_name, target_name]]
        next_level = pd.merge(search_index, next_level, how='inner', left_on=target_name,
                              right_on=source_name, suffixes=('_x', '_y'))\
            .drop(columns=[target_name+'_x', source_name+'_y'])\
            .rename(columns={source_name+'_x': source_name,
                            target_name+'_y': target_name})
        search_index = rec_append(found, next_level)
    print(f'Processing took {time.time() - t:.3f}s.')
    found = pd.concat(found, ignore_index=True)
    print(f'Found successors for {len(found)} modules. Returning source module id\'s as key '
          f'{source_name} and target module id\'s as key {target_name}')
    return found

def recursive_successor_search(db: PandasDatabase,
                               start_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                               target_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                               final_module_ids: Optional[Union[np.ndarray, pd.Series, pd.Index]] = None,
                               terminate_at_match: bool = True, print_iter: bool = True) -> pd.DataFrame:

    if final_module_ids is None:
        pipe_table = db.create_pipeline_table_if_absent()
        final_module_ids = pipe_table.loc[pipe_table[COLUMN_PIPELINE_FINAL] == True].index

    return recursive_search(db, start_module_ids, target_module_ids, final_module_ids, terminate_at_match, print_iter,
                            COLUMN_PIPELINE_PREDECESSOR, COLUMN_PIPELINE_SUCCESSOR)

def recursive_predecessor_search(db: PandasDatabase,
                                 start_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                                 target_module_ids: Union[np.ndarray, pd.Series, pd.Index],
                                 root_module_ids: Optional[Union[np.ndarray, pd.Series, pd.Index]] = None,
                                 terminate_at_match: bool = True, print_iter: bool = True) -> pd.DataFrame:
    if root_module_ids is None:
        pipe_table = db.create_pipeline_table_if_absent()
        root_module_ids = pipe_table.loc[pipe_table[COLUMN_PIPELINE_ROOT] == True].index

    return recursive_search(db, start_module_ids, target_module_ids, root_module_ids, terminate_at_match, print_iter,
                            COLUMN_PIPELINE_SUCCESSOR, COLUMN_PIPELINE_PREDECESSOR)

def get_metrics_index_with_criterion(db: PandasDatabase, pipe_modules: Optional[pd.DataFrame] = None,
                                     label_criterion: Optional[str] = None, prediction_criterion: Optional[str] = None) -> pd.DataFrame:
    if pipe_modules is None:
        pipe_modules = db.create_pipeline_table_if_absent()
    metric_modules: pd.DataFrame = pipe_modules.loc[pipe_modules[COLUMN_PIPELINE_TYPE] == 'metrics.MetricsModule']
    if label_criterion is not None:
        metric_modules = pd.merge(metric_modules, db['metrics.MetricsModule'], left_index=True, right_index=True)
        metric_modules = metric_modules.loc[metric_modules['label_criterion'] == label_criterion]
    if prediction_criterion is not None:
        metric_modules = pd.merge(metric_modules, db['metrics.MetricsModule'], left_index=True, right_index=True, suffixes=('','_rem'))
        # if label_criterion is not none, there will be suffixes
        drop_columns = [c for c in metric_modules.columns if c.endswith('_rem')]
        if drop_columns:
            metric_modules = metric_modules.drop(columns=drop_columns)
        metric_modules = metric_modules.loc[metric_modules['prediction_criterion'] == prediction_criterion]
    return metric_modules

def find_closest_metric_for_module(db: PandasDatabase, module_name: str,
                                   experiment_name: Optional[str] = None,
                                   module_param_filter: Optional[Dict[str, Any]] = None,
                                   merge_module_params: bool = False,
                                   label_criterion: Optional[str] = None,
                                   keep_experiment_table: bool = False,
                                   prediction_criterion: Optional[str] = None):
    pipeline_table = db.create_pipeline_table_if_absent()
    modules_of_interest = pipeline_table.loc[pipeline_table[COLUMN_PIPELINE_TYPE] == module_name]
    # first reduce the search space by ignoring all that are from potentially irrelevant experiments
    if experiment_name is not None:
        experiment_table = db.create_experiment_table_if_absent()
        experiment_table = experiment_table.loc[experiment_table[COLUMN_EXPERIMENT_NAME] == experiment_name]
        modules_of_interest = pd.merge(modules_of_interest, experiment_table, how='inner',
                                       left_on=COLUMN_PIPELINE_EX_ID, right_index=True)
        if not keep_experiment_table:
            del db[TABLE_EXPERIMENT] # free memory
    if module_param_filter is not None or merge_module_params:
        corresponding_module = db[module_name]
        corresponding_module = corresponding_module.loc[np.all(corresponding_module[list(module_param_filter.keys())] ==
                                                        list(module_param_filter.values()), axis=1)]
        if module_param_filter is not None:
            # don't merge yet in order to avoid having too much data in the recursive merge
            modules_of_interest = modules_of_interest.loc[modules_of_interest.index.isin(corresponding_module.index)]
    else:
        corresponding_module = None
    #pd.merge(modules_of_interest, corresponding_module, how='inner',
                          #         left_index=True, right_index=True, sort=False)
    metric_module_index = get_metrics_index_with_criterion(db, label_criterion=label_criterion, prediction_criterion=prediction_criterion).index

    match_df = recursive_successor_search(db, modules_of_interest.index, metric_module_index)
    if corresponding_module is not None and merge_module_params:
        match_df = pd.merge(left=match_df, right=corresponding_module, left_on=COLUMN_PIPELINE_PREDECESSOR, right_index=True)
    return match_df

def merge_module_with_metrics(db: PandasDatabase, to_merge: pd.DataFrame, merge_columns: Union[str, List[str]],
                              metric_name: str, clear_metric: bool = True,
                              metric_columns: Union[str, List[str]] = COLUMN_METRIC_MODULE_ID) -> pd.DataFrame:
    assert type(metric_columns) == type(merge_columns) and (not isinstance(metric_columns, list) or len(metric_columns) == len(merge_columns))
    metric_table = db[metric_name]
    if isinstance(metric_columns, list):
        metric_table = metric_table.dropna(axis=1, how='all')
        combined = [(metric_c, merge_c) for metric_c, merge_c in zip(metric_columns, merge_columns)
                    if (merge_c in to_merge.columns and metric_c in metric_table.columns)]
        merge_columns = [merge_c for _, merge_c in combined]
        metric_columns = [metric_c for metric_c, _ in combined]
        del combined
    try:
        res = pd.merge(to_merge, metric_table, left_on=merge_columns, right_on=metric_columns)
    except:
        print(f'Something went wrong and pandas has problems with nan values again, even though it should not! '
              f'Dropping nan values!', file=sys.stderr)
        # Also dropna has a problem with non sequence subset arguments even though the docs say that this is allowed
        if not isinstance(metric_columns, list):
            metric_columns = [metric_columns]
        if not isinstance(merge_columns, list):
            merge_columns = [merge_columns]
        metric_table = metric_table.dropna(subset=metric_columns)
        to_merge = to_merge.dropna(subset=merge_columns)
        res = pd.merge(to_merge, metric_table, left_on=merge_columns, right_on=metric_columns)
    if clear_metric:
        del db[metric_name]
    return res

def restrict_to_columns(to_clean: pd.DataFrame, to_restrict_to: Set[str]) -> pd.DataFrame:
    return to_clean.drop(columns=[c for c in to_clean.columns.values if c not in to_restrict_to])

def value_class_seed_only(to_clean: pd.DataFrame) -> pd.DataFrame:
    no_drop = {COLUMN_METRIC_VALUE, COLUMN_METRIC_CLASS, COLUMN_EXPERIMENT_SEED}
    return restrict_to_columns(to_clean, no_drop)

def merge_adaptors_with_transformers(db: PandasDatabase, to_merge: pd.DataFrame) -> pd.DataFrame:
    grouped = restrict_to_columns(to_merge, {COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME,
                                             COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID})\
        .groupby(COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME)[COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID].apply(list).to_dict()
    merge_frames = [pd.concat((db[t_name].loc[id_s].reset_index(name=COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID),
                               pd.DataFrame({COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME: t_name})), axis=1)
                    for t_name, id_s in grouped.items()]
    merge_frames = pd.concat(merge_frames, ignore_index=True)
    return pd.merge(left=to_merge, right=merge_frames,
                    on=[COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME, COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID])

def last_experiment_evaluation_metric_modules(db: PandasDatabase, experiment_name: str,
                                              metric_label_criterion: Optional[str] = None,
                                              include_config: bool = True,
                                              merge_columns: Optional[Dict[str, List[str]]] = None,
                                              config_filter: Optional[Dict[str, Any]] = None,
                                              metric_prediction_criterion: Optional[str] = None) \
        -> Union[pd.Series, pd.DataFrame]:
    experiment_table = db.create_experiment_table_if_absent()
    experiment_table = experiment_table.loc[experiment_table[COLUMN_EXPERIMENT_NAME] == experiment_name]
    modules = db.create_pipeline_table_if_absent()
    modules = pd.merge(modules, experiment_table, left_on=COLUMN_PIPELINE_EX_ID, right_index=True)
    metric_module_index = get_metrics_index_with_criterion(db, modules, metric_label_criterion, metric_prediction_criterion).index
    root_module_ids = modules.loc[modules[COLUMN_PIPELINE_ROOT] == True].index
    final_module_ids = modules.loc[modules[COLUMN_PIPELINE_FINAL] == True].index
    predecessors = recursive_predecessor_search(db, final_module_ids, metric_module_index, root_module_ids)[COLUMN_PIPELINE_PREDECESSOR]
    res = pd.merge(left=predecessors, right=modules, left_on=COLUMN_PIPELINE_PREDECESSOR, right_index=True)
    if include_config:
        res = pd.merge(left=res, right=db[experiment_name],
                               left_on=[COLUMN_PIPELINE_T_NUM, COLUMN_PIPELINE_EX_ID],
                               right_on=[COLUMN_CONFIG_TRIAL_NUM, COLUMN_CONFIG_EX_ID])
        if merge_columns is not None:
            for target_column, source_columns in merge_columns.items():
                source_columns = [sc for sc in source_columns if sc in res.columns]
                if not source_columns:
                    print(f'Got empty list of source columns that should have been assigned to {target_column}!')
                    continue
                new_column = res[source_columns[0]].copy()
                for source_column in source_columns[1:]:
                    mask = new_column.isna()
                    try:
                        new_column.loc[mask] = res.loc[mask,source_column]
                    except: # Fix pandas sometimes being a little bit too specific with it's types...
                        print('Caught exception', file=sys.stderr)
                        new_column = new_column.astype(np.float64)
                        new_column.loc[mask] = res.loc[mask,source_column]
                res.drop(columns=source_columns, inplace=True)
                res.loc[:,target_column] = new_column
        if config_filter is not None and len(config_filter) > 0:
            res = res.loc[np.all(res[list(config_filter.keys())] == list(config_filter.values()), axis=1)]
    return res

COMMON_CLEAR_COLUMNS = [COLUMN_PIPELINE_PREDECESSOR, COLUMN_PIPELINE_SUCCESSOR, COLUMN_METRIC_MODULE_ID, COLUMN_PIPELINE_ROOT,
                      COLUMN_PIPELINE_TYPE, COLUMN_EXPERIMENT_NAME, COLUMN_PIPELINE_EX_ID, COLUMN_PIPELINE_T_NUM,
                      COLUMN_PIPELINE_EX_ID+'_x', COLUMN_PIPELINE_EX_ID+'_y', COLUMN_PIPELINE_FINAL]

def merge_with_metrics_and_clear_common_columns(db: PandasDatabase, to_merge: pd.DataFrame,
                                                merge_column: Union[str, List[str]], metric_name: str,
                                                clear_metric: bool = True, retain_columns: Iterable[str] =
                                                (COLUMN_METRIC_VALUE, COLUMN_METRIC_CLASS, COLUMN_METRIC_DATA_ID,
                                                 COLUMN_METRIC_STEP_ID),
                                                metric_columns: Union[str, List[str]] = COLUMN_METRIC_MODULE_ID) -> pd.DataFrame:
    res = merge_module_with_metrics(db, to_merge, merge_column, metric_name, clear_metric, metric_columns)
    res = res.drop(columns=[c for c in COMMON_CLEAR_COLUMNS if c in res.columns and c not in retain_columns])
    return res

def merge_with_adaptors(db: PandasDatabase, to_merge: pd.DataFrame, merge_column: str,
                        include_unsupervised: bool = True, include_supervised:bool = True,
                        retain_name_only: bool = True, adaptor_module: str = 'pipeline.UnsupervisedSklearnAdaptorModule') -> pd.DataFrame:
    def merge_adaptor_columns(to_merge: pd.DataFrame, column: str, suffixes: Tuple[str, str]) -> pd.DataFrame:
        # We can safely assume that the unsupervised and the supervised adaptor never refer to the same module...
        to_merge.loc[:, column] = to_merge[column + suffixes[0]]
        # careful: Not all modules are one of the adaptors, therefore isna on the already set column must be used for
        # correct masking
        is_na_mask = to_merge[column].isna()
        to_merge.loc[is_na_mask,column] = to_merge.loc[is_na_mask, column + suffixes[1]]
        return to_merge

    if include_supervised:
        to_merge = pd.merge(left=to_merge, right=db['multipipeline.SupervisedSklearnAdaptorModule'], how='outer',
                             left_on=merge_column, right_index=True)
    if include_unsupervised:
        suffixes = ('_x', '_y')
        to_merge = pd.merge(left=to_merge, right=db[adaptor_module], how='outer',
                             left_on=merge_column, right_index=True, suffixes=suffixes)

        if include_supervised: # both are present => suffixes were added unintentionally
            to_merge = merge_adaptor_columns(to_merge, COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME, suffixes)
            to_merge = merge_adaptor_columns(to_merge, COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID, suffixes)
            to_merge = merge_adaptor_columns(to_merge, COLUMN_SKLEARN_ADAPTOR_DO_FIT, suffixes)
            to_merge = merge_adaptor_columns(to_merge, COLUMN_SKLEARN_ADAPTOR_DO_PREDICT, suffixes)
            to_merge = to_merge.drop(columns=[c + s
                                      for c in [COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME,
                                                COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID,
                                                COLUMN_SKLEARN_ADAPTOR_DO_FIT,
                                                COLUMN_SKLEARN_ADAPTOR_DO_PREDICT]
                                      for s in suffixes])

    if retain_name_only:
        to_merge = to_merge.drop(columns=[COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID,
                                          COLUMN_SKLEARN_ADAPTOR_DO_FIT,
                                          COLUMN_SKLEARN_ADAPTOR_DO_PREDICT])
    return to_merge

def _merge_metrics(db: PandasDatabase, metrics: Iterable[str], df: pd.DataFrame, class_of_interest: Optional[int]):
    metrics = list(metrics)
    orig_columns = list(df.columns)
    retain_columns: List[str] = orig_columns + [COLUMN_METRIC_DATA_ID, COLUMN_METRIC_CLASS, COLUMN_METRIC_STEP_ID]
    retain_columns: Set[str] = set(retain_columns)
    for metric in metrics:
        print(f'Merging {df.shape[0]} elements with metric {metric}')
        merge_columns = [COLUMN_PIPELINE_PREDECESSOR,
                         COLUMN_METRIC_CLASS,
                         COLUMN_METRIC_DATA_ID,
                         COLUMN_METRIC_STEP_ID,
                         COLUMN_METRIC_BATCH_SIZE]
        df = merge_with_metrics_and_clear_common_columns(db, df, merge_columns, metric,
                                                         retain_columns=(COLUMN_METRIC_VALUE,
                                                                         COLUMN_METRIC_CLASS,
                                                                         COLUMN_METRIC_DATA_ID,
                                                                         COLUMN_METRIC_STEP_ID,
                                                                         COLUMN_METRIC_BATCH_SIZE,
                                                                         COLUMN_PIPELINE_PREDECESSOR,
                                                                         'Mean of Batches-Iteration Mean',
                                                                         COLUMN_EXPERIMENT_SEED),
                                                         metric_columns=[COLUMN_METRIC_MODULE_ID,
                                                                         COLUMN_METRIC_CLASS,
                                                                         COLUMN_METRIC_DATA_ID,
                                                                         COLUMN_METRIC_STEP_ID,
                                                                         COLUMN_METRIC_BATCH_SIZE])
        #df = perform_averaging(df, orig_columns, drop_class=False, allow_seed_averaging=False)

        if COLUMN_METRIC_BATCH_SIZE in df.columns:
            unique_batch_sizes = df[COLUMN_METRIC_BATCH_SIZE].unique()
            new_c = {'Mean of Batches-Iteration Mean', 'Mean of Batches-Iteration Std.',
                     'Std. of Batches-Iteration Mean'}
            df = df.drop(columns=[c for c in df.columns if c not in retain_columns and c not in new_c and c != COLUMN_METRIC_BATCH_SIZE])
            rename = {'Mean of Batches-Iteration Mean': metric,
                      'Mean of Batches-Iteration Std.': metric + ' (SEM)',
                      'Std. of Batches-Iteration Mean': metric + ' (MSTD)'}
            non_batch_size_columns = [c for c in df.columns if c != COLUMN_METRIC_BATCH_SIZE]
            new_df = [df.loc[df[COLUMN_METRIC_BATCH_SIZE] == bs, non_batch_size_columns].rename(columns={k:v+f' ({bs})'
                                                                                                 for k,v in
                                                                                                 rename.items()})
                      for bs in unique_batch_sizes]
            non_batch_size_columns = [c for c in non_batch_size_columns if c not in new_c]
            while len(new_df) > 1:
                new_df = [pd.merge(new_df[i], new_df[i+1], on=non_batch_size_columns) for i in range(0, len(new_df)-1, 2)]
            df = new_df[0]
            retain_columns.update([v+f' ({bs})' for v in rename.values() for bs in unique_batch_sizes])
            del new_df
        else:
            rename = {COLUMN_METRIC_VALUE: metric} if COLUMN_METRIC_VALUE in df.columns else \
                {COLUMN_METRIC_VALUE+'.0': metric+ '.0',
                 COLUMN_METRIC_VALUE+'.1': metric + '.1'}
            df = df.rename(columns=rename)
            retain_columns.update(rename.values())
            df = df.drop(columns=[c for c in df.columns if c not in retain_columns])
    orig_columns.append(COLUMN_METRIC_CLASS)
    df = perform_averaging(df, orig_columns, drop_class=False)
    if COLUMN_METRIC_CLASS in df.columns and class_of_interest is not None:
        df = df.loc[np.logical_or(df[COLUMN_METRIC_CLASS] == class_of_interest, df[COLUMN_METRIC_CLASS].isna())]
    elif COLUMN_METRIC_CLASS in df.columns:
        columns_of_interest = [c for c in df.columns if c != COLUMN_METRIC_CLASS]
        df = pd.merge(left=df.loc[df[COLUMN_METRIC_CLASS] == 0, columns_of_interest],
                      right=df.loc[df[COLUMN_METRIC_CLASS] == 1, columns_of_interest],
                      left_on=[c for c in orig_columns if c != COLUMN_METRIC_CLASS and c in df.columns],
                      right_on=[c for c in orig_columns if c != COLUMN_METRIC_CLASS and c in df.columns],
                      suffixes=('.0', '.1'))
    orig_columns.remove(COLUMN_METRIC_CLASS)
    return df, orig_columns

def perform_averaging(df: pd.DataFrame, orig_columns: List[str], reset_index: bool = True, drop_class: bool = True,
                      allow_seed_averaging: bool = True, retain_columns: Iterable[str] = tuple()) -> pd.DataFrame:
    print(f'Aggregating over the data points.')
    orig_columns = set(orig_columns + [COLUMN_METRIC_STEP_ID])
    # first: average over data-points per-seed
    if COLUMN_METRIC_DATA_ID in df.columns:
        columns = [COLUMN_METRIC_DATA_ID]
        if drop_class and COLUMN_METRIC_CLASS in df.columns:
            columns.append(COLUMN_METRIC_CLASS)
        df = df.drop(columns=columns)
        grouped = df.groupby(by=[c for c in df.columns if c in orig_columns], dropna=False)
        df = grouped.agg([np.nanmean]).droplevel(axis=1, level=1)
        stds: pd.DataFrame = grouped.agg([np.nanstd]).droplevel(axis=1, level=1)
        stds = pd.DataFrame({c: ser.replace(0.0, np.nan) for c, ser in stds.items()}, index=stds.index)
        stds = stds.dropna(axis=1, how='all')
        df = pd.merge(left=df, right=stds, left_index=True, right_index=True, suffixes=('_mean', '_mstd'))
        del grouped
        del stds
    else:
        df = df.rename(columns={COLUMN_METRIC_VALUE:COLUMN_METRIC_VALUE+'_mean'})
    # second average over seeds
    if allow_seed_averaging:
        reset = df.reset_index()
        columns = [c for c in COMMON_CLEAR_COLUMNS if c not in retain_columns]
        columns.extend([COLUMN_EXPERIMENT_SEED, '_seed', 'random_state', 'random_seed'])
        to_remove = [c for c in columns if c in reset.columns]
        if to_remove:
            df = reset.drop(columns=to_remove)
            grouped = df.groupby(by=[c for c in df.columns if c in orig_columns], dropna=False)
            df = grouped.agg([np.nanmean]).droplevel(axis=1, level=1)
            stds: pd.DataFrame = grouped.agg([np.nanstd]).droplevel(axis=1, level=1)
            stds = pd.DataFrame({c: ser.replace(0.0, np.nan) for c, ser in stds.items()}, index=stds.index)
            stds = stds.dropna(axis=1, how='all')
            if len(stds.columns) > 0:  # no seed to calculate standard on
                df = pd.merge(left=df, right=stds, left_index=True, right_index=True, suffixes=('', '_std'))
            del grouped
            del stds
    if reset_index:
        df = df.reset_index()
    elif not isinstance(df.index, pd.MultiIndex):
        df = df.set_index([c for c in orig_columns if c in df.columns])
    return df.round(5)

def best_according_to_metric(db: PandasDatabase, df: pd.DataFrame, metrics: Iterable[str],
                             class_of_interest: Optional[int] = 1, feature_space_column_name = 'feature_space',
                             max_display: int = 5, save_folder: Optional[str] = None):

    df, orig_columns = _merge_metrics(db, metrics, df, class_of_interest)
    #pre_average_columns = list(df.columns)
    #df = perform_averaging(df, orig_columns)

    # now calculate whether some feature space is mixed and/or contains SAR data
    df['SAR_included'] = df[feature_space_column_name].str.contains(r'SAR[^"]*(_[^"]*)*', regex=True)
    df['mixed'] = df[feature_space_column_name].str.contains(r'[^"]*_[^"]*', regex=True)
    print('Calculating results.')
    metrics = [((m+'_mean.1' if m+'_mean.1' in df.columns else (m+'_mean' if m+'_mean' in df.columns else m+'.1')) if m not in df.columns else m)
               for m in metrics]
    for sar_included, mixed in [(True, True), (True, False), (False, False)]:
        print(f'Depicting results on feature spaces with sar_included={sar_included} and mixed={mixed}.')
        filter_columns = ['SAR_included', 'mixed']
        df_to_use: pd.DataFrame = df.loc[np.all(df[filter_columns] == [sar_included, mixed], axis=1)]
        df_to_use = df_to_use.drop(columns=filter_columns)
        print(f'The tested feature spaces are {np.unique(df_to_use[feature_space_column_name]).tolist()} and there are'
              f'{df_to_use.shape[0]} entries.')
        if df_to_use.shape[0] == 0:
            print(f'No values match the given predicate. Skipping.')
            continue
        df_to_use = df_to_use.sort_values(by=metrics, ascending=False, kind='stable', ignore_index=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_to_use.head(max_display))
        if save_folder is not None:
            print(f'Saving to {save_folder}')
            if not path.exists(save_folder):
                os.makedirs(save_folder)
            if sar_included and mixed:
                f_name = 'sar_opt.csv'
            elif sar_included and not mixed:
                f_name = 'sar.csv'
            elif not sar_included and not mixed:
                f_name = 'opt.csv'
            else:
                raise RuntimeError(f'Illegal combination sar_included={sar_included} and mixed={mixed}')
            df_to_use.to_csv(path.join(save_folder, f_name), index=True)
        else:
            print('No save file provided. Outputs will only be dumped to the console.')

def plot_region_comparison(db: PandasDatabase, df: pandas.DataFrame, folder: str,
                           region_metrics: Optional[List[str]], class_of_interest,
                           single_plot: bool = True, with_title: bool = False,
                           add_title: bool = False, x_axis_label: bool = False, metric_y_axis: bool = True):
    from plot_utils import METRIC_NAME_MAP, METRIC_NAME_REVERSE_MAP, METRIC_RANGE_MAP, save_and_show, create_figure_with_subplots, FLOOD_LABELS, plot_scatter_regression_line2
    metrics = region_metrics if region_metrics else METRIC_NAME_MAP.keys()
    metrics = [m.replace('per_data', 'per_region', 1) for m in metrics if 'per_data' in m]
    df = pd.merge(left=df, right=db['per_region.label_distribution.no_clouds'],
                  left_on=COLUMN_PIPELINE_PREDECESSOR, right_on=COLUMN_METRIC_MODULE_ID)
    df = df.rename(columns={COLUMN_METRIC_VALUE: 'Region-Distribution', COLUMN_METRIC_CLASS: 'Distribution Class'})
    #df = df.set_index(['Distribution Class'], append=True)
    #to_add = df.xs(0, level='Distribution Class')['Region-Distribution'].to_numpy() - df.xs(1, level='Distribution Class')['Region-Distribution'].to_numpy()
    df = df.drop(columns=COLUMN_METRIC_MODULE_ID)
    if single_plot:
        fig, axes, ld = create_figure_with_subplots('Regionwise Metrics', len(metrics), 3, with_title=with_title)
    else:
        fig, axes, ld = [] , [], []
        for _ in range(len(metrics)):
            fls, axls, ldls = [] , [], []
            for _ in range(3):
                f, a, l = create_figure_with_subplots('Regionwise Metrics', 1, 1, with_title=with_title)
                fls.append(f)
                axls.append(a[0,0])
                ldls.append(l)
            fig.append(fls)
            axes.append(axls)
            ld.append(ldls)

    feature_spaces = df['feature_space'].unique()
    data_ids = df[COLUMN_METRIC_DATA_ID].unique()
    to_sci_analyze = []
    for metric, axs in zip(metrics, axes):
        merged = merge_with_metrics_and_clear_common_columns(db, df, [COLUMN_PIPELINE_PREDECESSOR,
                                                                      COLUMN_METRIC_DATA_ID,
                                                                      COLUMN_METRIC_STEP_ID], metric,
                                                             retain_columns=(COLUMN_METRIC_VALUE,
                                                                             COLUMN_METRIC_CLASS,
                                                                             COLUMN_METRIC_DATA_ID,
                                                                             COLUMN_METRIC_STEP_ID,
                                                                             COLUMN_METRIC_BATCH_SIZE,
                                                                             COLUMN_PIPELINE_PREDECESSOR),
                                                             metric_columns=[COLUMN_METRIC_MODULE_ID,
                                                                             COLUMN_METRIC_DATA_ID,
                                                                             COLUMN_METRIC_STEP_ID])
        to_index = ['feature_space', COLUMN_METRIC_DATA_ID, 'Distribution Class']
        query_levels = ['feature_space', COLUMN_METRIC_DATA_ID, 'Distribution Class']
        if COLUMN_METRIC_CLASS in merged.columns:
            to_index.append(COLUMN_METRIC_CLASS)
            query_levels.append(COLUMN_METRIC_CLASS)
        merged = merged.set_index(to_index, append=True)
        print(merged)
        ax: plt.Axes = axs[0]
        width = 0.3
        x_pos = np.arange(len(data_ids))
        values = {d_id: {fs:merged.xs(((fs, d_id, 1) if len(query_levels) == 3 else (fs, d_id, 1, class_of_interest)),
                                 level=query_levels)[COLUMN_METRIC_VALUE].mean()  for fs in feature_spaces}
                  for d_id in data_ids}
        sorted_values = list(sorted(values.items(), key=lambda t: np.max(list(t[1].values()))))
        data_counts = {d_id: {clazz: merged.xs((d_id, clazz), level=[COLUMN_METRIC_DATA_ID, 'Distribution Class'])['Region-Distribution'].mean() for clazz in [0, 1]}
                       for d_id in data_ids}
        m_name = METRIC_NAME_REVERSE_MAP[metric.replace('per_region', 'per_data', 1)]
        for i, fs in enumerate(feature_spaces):
            heights = [sub_dict[fs] for _, sub_dict in sorted_values]
            positions = x_pos+(i - len(feature_spaces) //2)*width
            ax.bar(positions, height=heights, width=width, label=fs)
            for i, ax_to_use in enumerate(axs[1:]):
                x_positions = pd.Series([sub_dict[i] for _, sub_dict in data_counts.items()], name=FLOOD_LABELS[i])
                y_positions = pd.Series([values[d_id][fs] for d_id, _ in data_counts.items()], name=m_name)
                ax_to_use.scatter(x_positions, y_positions, label=fs)
                plot_scatter_regression_line2(ax_to_use, x_positions, y_positions,
                                              (x_positions.min(), x_positions.max()), METRIC_RANGE_MAP[m_name])
                to_sci_analyze.append([m_name, FLOOD_LABELS[i], f'{fs} Analysis of {m_name}', x_positions, y_positions])
                #
                #plot_scatter_regression_line(ax_to_use, x_positions, y_positions, (x_positions.min(), x_positions.max()),
                #                             METRIC_RANGE_MAP[m_name], test_y_log=False, implicit_x_log=True, scale_x=False, scale_y=False)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([d_id for d_id, _ in sorted_values], rotation=45, ha='right')
        for i, ax_to_use in enumerate(axs):
            if add_title:
                ax_to_use.set_title(m_name)
            ax_to_use.legend()
            ax_to_use.set_ylim(METRIC_RANGE_MAP[m_name])
            if i > 0:
                ax_to_use.set_xscale('log')
                if x_axis_label:
                    ax_to_use.set_xlabel(FLOOD_LABELS[i-1])
            if metric_y_axis:
                ax_to_use.set_ylabel(m_name)
    if isinstance(fig, list) and isinstance(ld, list):
        for i, (fls, ldls) in enumerate(zip(fig, ld)):
            for j, (f, l) in enumerate(zip(fls, ldls)):
                save_and_show(f, path.join(folder, f'plot_{FLOOD_LABELS[class_of_interest]}_{i}_{j}'), True, l, True)
    else:
        save_and_show(fig, path.join(folder, f'plot_{FLOOD_LABELS[class_of_interest]}'), True, ld, True)
    for m_name, flood_label, title, x_positions, y_positions in to_sci_analyze:
        print('-----------------------------------------------------------------------')
        print(f'Analysis of "{m_name}" for flood pixels of type "{flood_label}" and with image title "{title}"')
        print('Metric statistics:')
        print(y_positions.describe())
        analyze(np.log(x_positions), y_positions, xname=flood_label, yname=m_name, title=title)


DEFAULT_METRICS = [
    'per_data.iou1.no_clouds',
    'total.iou0.no_clouds',
    'per_data.accuracy.no_clouds',
    'total.accuracy.no_clouds',
    'per_data.recall.no_clouds',
    'total.recall.no_clouds',
    'per_data.f1_score.no_clouds',
    'total.f1_score.no_clouds',
    'per_data.precision.no_clouds',
    'total.precision.no_clouds',
    #'batch.iou_8',
    #'batch.accuracy_8',
    #'batch.f1_score_8',
    #'batch.recall_8',
    #'batch.precision_8',
    #'batch.iou_2',
    #'batch.accuracy_2',
    #'batch.f1_score_2',
    #'batch.recall_2',
    #'batch.precision_2',
    #'batch.iou_4',
    #'batch.accuracy_4',
    #'batch.f1_score_4',
    #'batch.recall_4',
    #'batch.precision_4',
    #'batch.iou_16',
    #'batch.accuracy_16',
    #'batch.f1_score_16',
    #'batch.recall_16',
    #'batch.precision_16'
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database_folder', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--metrics', nargs='+', type=str, default=DEFAULT_METRICS)
    parser.add_argument('--label_criterion', dest='label_criterion', default=None)
    parser.add_argument('--prediction_criterion', dest='prediction_criterion', default=None)
    parser.add_argument("--config_values", dest="config_values", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", default=None)
    parser.add_argument("--combine_columns", dest="combine_columns", action=StoreMultiDict, nargs="+", default=None)
    parser.add_argument('-coi','--class_of_interest', dest='class_of_interest', type=int, default=1)
    parser.add_argument('--feature_space_name', dest='fs_name', default='feature_space', type=str)
    parser.add_argument('--max_display', dest='max_display', default=5, type=int)
    parser.add_argument('--save_folder', dest='save_folder', default=None, type=str)
    parser.add_argument('--save_query', dest='save_query', action='store_true')
    parser.add_argument('--no-save_query', dest='save_query', action='store_false')
    parser.add_argument('--plot_regions', dest='plot_regions', action='store_true')
    parser.add_argument('--no-plot_regions', dest='plot_regions', action='store_false')
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
    parser.add_argument('--with_figure_title', dest='with_title', action='store_true')
    parser.add_argument('--no-figure_title', dest='with_title', action='store_false')
    parser.add_argument('--single_plot', dest='single_plot', action='store_true')
    parser.add_argument('--multi_plot', dest='single_plot', action='store_false')
    parser.add_argument('--fig_width', dest='fig_width', default=6.0, type=float)
    parser.add_argument('--fig_height', dest='fig_height', default=6.0, type=float)
    parser.set_defaults(save_query=True, plot_regions=False, add_title=False, metric_y_axis=True, x_axis_label=True,
                        with_title=False, single_plot = False)
    args = parser.parse_args()
    db = ReadOnlyPandasDatabase(args.database_folder)
    print('Loading metric modules')
    df = last_experiment_evaluation_metric_modules(db, args.experiment_name,
                                                   metric_label_criterion=args.label_criterion,
                                                   metric_prediction_criterion=args.prediction_criterion,
                                                   merge_columns=args.combine_columns,
                                                   config_filter=args.config_values)
    df = df.dropna(axis=1, how='all')
    plt.rcParams.update(args.rc_params)
    print('Font Sizes:', plt.rcParams['font.size'], plt.rcParams['legend.fontsize'], plt.rcParams['legend.title_fontsize'])
    print('Font Properties:', plt.rcParams['font.family'])
    print('Font Weight:', plt.rcParams['font.weight'])
    # Here a dirty fix for some seed column being merged too much, before I have to hand in... Not sure why this happens
    if 'seed_x' in df.columns and 'seed_y' in df.columns:
        print('Hackfixing seed merge...')
        df = df.drop(columns=['seed_y'])
        df = df.rename(columns={'seed_x':COLUMN_EXPERIMENT_SEED})
    if args.save_query:
        best_according_to_metric(db, df, args.metrics, args.class_of_interest, args.fs_name, args.max_display,
                                 save_folder=args.save_folder)
    if args.plot_regions:
        plot_region_comparison(db, df, args.save_folder, args.metrics, args.class_of_interest,
                               single_plot=args.single_plot, with_title=args.with_title, add_title=args.add_title,
                               x_axis_label=args.x_axis_label, metric_y_axis=args.metric_y_axis)

