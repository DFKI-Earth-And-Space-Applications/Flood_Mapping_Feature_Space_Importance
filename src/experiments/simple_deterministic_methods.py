import functools as funct

import optuna
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'simple_deterministic_classifiers'

def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                 feature_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True, predict_factory: bool = False) -> Union[MultiPipeline, Tuple[MultiPipeline, Callable]]:
    label_pipe = MultiSequenceModule([
        MultiDistributorModule([
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=SPLIT_TRAIN, as_array=True),
                                  dataset_name='train_labels'
                                  ),
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=eval_split,
                                                                              as_array=True),
                                  dataset_name='valid_labels'
                                  )
        ]),
        MaskModule(source_criterion=NameSelectionCriterion('train_labels'),
                   res_name='train_mask',
                   mask_label=-1)
    ])
    label_pipe_cache.set_module(label_pipe)
    feature_space = _config['z_feature_space']
    train_cons, valid_cons = cons_by_filter[_config['filter']][feature_space]
    feature_pipe = MultiDistributorModule([
        PipelineAdaptorModule(selection_criterion=None,
                              pipe_module=train_cons,
                              dataset_name='train_features'
                              ),
        PipelineAdaptorModule(selection_criterion=None,
                              pipe_module=valid_cons,
                              dataset_name='valid_features'
                              )
    ])
    feature_pipe_cache.set_module(feature_pipe)
    data_construction = MultiDistributorModule([
        feature_pipe_cache,
        label_pipe_cache
    ])
    method = _config[feature_space + '_method']
    eval_module: Callable = lambda feature_dataset_name, prediction_dataset_name: None
    train_module: Callable = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: None
    if method == 'threshold':
        assert feature_space == 'SAR', 'Cannot threshold other feature spaces than SAR!'
        threshold_args = {
            'print_warning': False,
            'bin_count': _config['bin_count'],
            'use_tiled': _config['threshold_use_tiled']
        }
        if _config['threshold_use_tiled']:
            threshold_args['tile_dim'] = _config['threshold_tile_dim']
            threshold_args['force_merge_on_failure'] = _config['threshold_force_merge_on_failure']
            threshold_args['percentile'] = _config['threshold_percentile']
        threshold_alg = 'classification.KIPerImageThresholdingClassifier'
        if _config['threshold_alg'] == 'otsu':
            threshold_alg = 'classification.OtsuPerImageThresholdingClassifier'
        thresholding = UnsupervisedSklearnAdaptor(threshold_alg,
                                                  params=threshold_args,
                                                  per_channel=True,
                                                  per_data_point=True,
                                                  image_features=True,
                                                  clear_on_predict=False,
                                                  save_file='thresholds.pkl')

        target_eval = _config['target_eval']
        if target_eval == 'VV':
            eval_module = lambda feature_dataset_name, prediction_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([WhitelistModule('VV'),
                                            UnsupervisedSklearnAdaptorModule(thresholding, do_fit=False)]),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=prediction_dataset_name,
                keep_source=True)
            train_module = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([WhitelistModule('VV'),
                                            UnsupervisedSklearnAdaptorModule(thresholding, do_predict=False),
                                            TerminationModule()],
                                           ignore_none=True),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=feature_dataset_name,
                keep_source=True)
        elif target_eval == 'VH':
            eval_module = lambda feature_dataset_name, prediction_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([WhitelistModule('VH'),
                                            UnsupervisedSklearnAdaptorModule(thresholding,
                                                                             do_fit=False)]),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=prediction_dataset_name,
                keep_source=True)
            train_module = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([WhitelistModule('VH'),
                                            UnsupervisedSklearnAdaptorModule(thresholding, do_predict=False),
                                            TerminationModule()], ignore_none=True),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=None,
                keep_source=True)
        elif target_eval == 'VV & VH':
            bitwise_combine = UnsupervisedSklearnAdaptor('classification.BitwiseAndChannelCombiner',
                                                         params={'reduction_axis': 0},
                                                         per_data_point=True,
                                                         image_features=True,
                                                         allow_no_fit=True)
            eval_module = lambda feature_dataset_name, prediction_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([UnsupervisedSklearnAdaptorModule(thresholding, do_fit=False),
                                            UnsupervisedSklearnAdaptorModule(bitwise_combine, do_fit=False)],
                                           ignore_none=True),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=prediction_dataset_name,
                keep_source=True)
            train_module = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([UnsupervisedSklearnAdaptorModule(thresholding, do_predict=False),
                                            TerminationModule()], ignore_none=True),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=None,
                keep_source=True)
        elif target_eval == 'VV | VH':
            bitwise_combine = UnsupervisedSklearnAdaptor('classification.BitwiseOrChannelCombiner',
                                                         params={'reduction_axis': 0},
                                                         per_data_point=True,
                                                         image_features=True,
                                                         allow_no_fit=True)
            eval_module = lambda feature_dataset_name, prediction_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([UnsupervisedSklearnAdaptorModule(thresholding, do_fit=False),
                                            UnsupervisedSklearnAdaptorModule(bitwise_combine, do_fit=False)]),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=prediction_dataset_name,
                keep_source=True)
            train_module = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: PipelineAdaptorModule(
                pipe_module=SequenceModule([UnsupervisedSklearnAdaptorModule(thresholding, do_predict=False),
                                            TerminationModule()], ignore_none=True),
                selection_criterion=NameSelectionCriterion(feature_dataset_name),
                dataset_name=None,
                keep_source=True)
        else:
            raise RuntimeError(f'Unknown thresholding eval method {target_eval}!')
    elif method == 'k-NN':
        adaptor = SupervisedSklearnAdaptor('sklearn.neighbors.KNeighborsClassifier',
                                           params={'n_neighbors': _config[_config['weights'] + '_k'],
                                                   'weights': _config['weights'],
                                                   'leaf_size': 30,
                                                   'metric': _config['metric'],
                                                   'n_jobs': 2},
                                           save_file=None)
        eval_module = lambda feature_dataset_name, prediction_dataset_name: \
            SupervisedSklearnAdaptorModule(adaptor, feature_criterion=NameSelectionCriterion(feature_dataset_name),
                                           prediction_dataset_name=prediction_dataset_name,
                                           prediction_channel_name=prediction_dataset_name,
                                           do_fit=False)
        train_module = lambda feature_dataset_name, label_dataset_name, mask_dataset_name: \
            SupervisedSklearnAdaptorModule(adaptor, feature_criterion=NameSelectionCriterion(feature_dataset_name),
                                           label_criterion=NameSelectionCriterion(label_dataset_name),
                                           mask_criterion=NameSelectionCriterion(mask_dataset_name),
                                           do_predict=False)
        thresholding = None
    else:
        raise RuntimeError(f'Unknown method {method}!')
    seq = [
        data_construction,
        train_module('train_features', 'train_labels', 'train_mask'),
        eval_module('valid_features', 'valid_prediction')
    ]
    if include_save_operations:
        if eval_split == SPLIT_TEST:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     per_region_computation=ContingencyMatrixComputation('valid.contingency'),
                                     delete_prediction=True))
        else:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     delete_prediction=True))
    complete_pipe = MultiSequenceModule(seq)

    if predict_factory:
        return complete_pipe, (lambda criterion: eval_module(criterion.name, 'valid_prediction'))
    return complete_pipe

def pipeline(data_folder: str, cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
             eval_split: str = SPLIT_VALIDATION) -> Callable:
    feature_pipe_cache = SimpleCachedPipeline()
    label_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        complete_pipe = raw_pipeline(data_folder, eval_split, label_pipe_cache, feature_pipe_cache, _config, _seed, cons_by_filter)
        print(f'Serializing pipeline for config {_config}')
        with _hook.open_artifact_file('pipeline.json', 'w') as fd:
            serialize(fd, complete_pipe)
        print('Serialisation completed to file pipeline.json')
        summary = Summary([], lambda a: '')
        summary.set_hook(_hook.add_result_metric_prefix('valid.contingency'))
        print('Starting Pipeline')
        t = time.time()
        res_summary = complete_pipe(summary)
        t = time.time() - t
        print(f'Pipeline completed in {t:.3f}s')

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if k.startswith('valid')]
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values))
        print(f'Mean Overall Accuracy is {res}.')
        return res

    return main

def execute_grid_search_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_grid_search'
    cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters, use_datadings)

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        # prefix with z so that it will be iterated last and we can thus maximise cache hits
        feature_space = trial.suggest_categorical('z_feature_space', f_names[filter], condition=True)
        method_choices = ['k-NN']
        if feature_space == 'SAR':
           method_choices = ['threshold'] + method_choices
        method = trial.suggest_categorical(feature_space+'_method', method_choices, condition=True)
        if method == 'threshold':
            trial.suggest_categorical('threshold_alg', ['otsu', 'ki'])
            trial.suggest_categorical('bin_count', ['tile_dim', 'auto'])
            use_tiled = trial.suggest_categorical('threshold_use_tiled', [True, False], condition=True)
            if use_tiled:
                trial.suggest_categorical('threshold_tile_dim', [32, 64, 128, 256])
                trial.suggest_categorical('threshold_force_merge_on_failure', [True, False])
                trial.suggest_float('threshold_percentile', 0.8, 0.99, selected_choices=[0.8, 0.85, 0.9, 0.95, 0.99])
            trial.leave_condition('threshold_use_tiled')
            target_eval = trial.suggest_categorical('target_eval', ['VV', 'VH', 'VV & VH', 'VV | VH'])
        elif method == 'k-NN':
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'], selected_choices=['uniform'], condition=True)
            if weights == 'uniform':
                k = trial.suggest_int('uniform_k', 1, 1000, selected_choices=[1, 2, 3, 4, 5, 10, 20])
            else:
                k = trial.suggest_int('distance_k', 1, 1000, selected_choices=[2, 3, 4, 5, 10, 20])
            trial.leave_condition('weights')
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'],
                                               selected_choices=['euclidean'])
        else:
            raise NotImplementedError(f'Method {method} is not supported yet')
        trial.leave_condition(['z_feature_space', feature_space+'_method'])

    run_sacred_grid_search(experiment_base_folder, ex_name, pipeline(data_folder, cons_by_filter), config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)


def execute_final_experiments(data_folder: str, experiment_base_folder: str,
                              use_datadings: bool = True, seed: Optional[int] = None,
                              timeout: Optional[int] = None,
                              filters: Optional[List[str]] = None,
                              redirect_output: bool = True):
    ex_name = NAME + '_final'
    for split in [SPLIT_TEST, SPLIT_BOLIVIA]:
        cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters, use_datadings, eval_split=split)

        def config(trial: ConditionedTrial):
            filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
            # prefix with z so that it will be iterated last and we can thus maximise cache hits
            feature_space = trial.suggest_categorical('z_feature_space', f_names[filter],
                                                      selected_choices=['SAR'],
                                                      condition=True)
            method_choices = ['threshold']
            method = trial.suggest_categorical(feature_space + '_method', method_choices, condition=True)
            if method == 'threshold':
                trial.suggest_categorical('threshold_alg', ['otsu'])
                trial.suggest_categorical('bin_count', ['tile_dim'], selected_choices=['tile_dim'])
                use_tiled = trial.suggest_categorical('threshold_use_tiled', [True, False], selected_choices=[False])
                target_eval = trial.suggest_categorical('target_eval', ['VH'])
            else:
                raise NotImplementedError(f'Method {method} is not supported yet')
            trial.leave_condition(['z_feature_space', feature_space + '_method'])
        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)