import functools as funct

import optuna
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'simple_discriminant_analysis'


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
    method = _config['method']
    if method == 'naive':
        adaptor = SupervisedSklearnAdaptor('sklearn.naive_bayes.GaussianNB',
                                           params={},
                                           save_file='naive_bayes.pkl',
                                           clear_on_predict=False)
    elif method == 'linear':
        params = {'solver': _config['solver']}
        if 'shrinkage' in _config:
            params['shrinkage'] = _config['shrinkage']
        adaptor = SupervisedSklearnAdaptor('sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
                                           params=params,
                                           save_file='linear_discriminant_analysis.pkl',
                                           clear_on_predict=False)
    elif method == 'quadratic':
        adaptor = SupervisedSklearnAdaptor('sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
                                           params={'reg_param': _config['reg_param']},
                                           save_file='quadratic_discriminant_analysis.pkl',
                                           clear_on_predict=False)
    else:
        raise RuntimeError(f'Unknown method {method}!')
    seq = [
        data_construction,
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('train_features'),
                                       label_criterion=NameSelectionCriterion('train_labels'),
                                       mask_criterion=NameSelectionCriterion('train_mask'),
                                       do_predict=False)
    ]
    if eval_split == SPLIT_VALIDATION:
        seq.extend([
            SupervisedSklearnAdaptorModule(adaptor,
                                           feature_criterion=NameSelectionCriterion('train_features'),
                                           do_fit=False,
                                           prediction_channel_name='train_prediction',
                                           prediction_dataset_name='train_prediction'),
            MetricsModule(prediction_criterion=NameSelectionCriterion('train_prediction'),
                          label_criterion=NameSelectionCriterion('train_labels'),
                          source_criterion=NameSelectionCriterion('train_features'),
                          per_data_computation=ContingencyMatrixComputation('train.contingency'),
                          delete_prediction=True)
        ])
    seq.extend([
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('valid_features'),
                                       do_fit=False,
                                       prediction_channel_name='valid_prediction',
                                       prediction_dataset_name='valid_prediction')
    ])
    if include_save_operations:
        if eval_split == SPLIT_TEST:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     per_region_computation=ContingencyMatrixComputation('valid.region.contingency'),
                                     delete_prediction=True))
        else:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     delete_prediction=True))
    complete_pipe = MultiSequenceModule(seq)
    if predict_factory:
        return complete_pipe, (lambda criterion: SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=criterion,
                                       do_fit=False,
                                       prediction_channel_name='valid_prediction',
                                       prediction_dataset_name='valid_prediction'))
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
        method = trial.suggest_categorical('method', ['naive', 'linear', 'quadratic'], condition=True)
        if method == 'linear':
            solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'], condition=True)
            if solver in ['lsqr', 'eigen']:
                trial.suggest_uniform('shrinkage', 0.0, 1.0, selected_choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                                               0.7, 0.8, 0.9, 1.0])
            trial.leave_condition('solver')
        elif method == 'quadratic':
            trial.suggest_uniform('reg_param', 0.0, 1e3, selected_choices=[0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5,
                                                                           1, 2, 4, 8, 10])
        elif method == 'naive':
            pass
        else:
            raise NotImplementedError(f'Method {method} is not supported yet')
        trial.leave_condition(['method', 'z_feature_space'])



    run_sacred_grid_search(experiment_base_folder, ex_name, pipeline(data_folder, cons_by_filter), config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)

# TODO setup correct parameter configurations
def execute_final_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_final'

    for split in [SPLIT_TEST, SPLIT_BOLIVIA]:
        cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters,
                                                                                                    use_datadings, eval_split=split)

        def config(trial: ConditionedTrial):
            filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
            # prefix with z so that it will be iterated last and we can thus maximise cache hits
            method = trial.suggest_categorical('method', ['naive', 'linear', 'quadratic'], condition=True)
            res = {}
            if method == 'linear':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_cAWEI+cNDWI', 'cAWEI', 'SAR'],
                                                          condition=True)
                res['z_feature_space'] = feature_space
                solver = trial.suggest_categorical(feature_space+'solver',
                                                   ['eigen'],
                                                   condition=True)
                res['solver'] = solver
                if solver in ['lsqr', 'eigen']:
                    shrinkage = trial.suggest_uniform(feature_space+'shrinkage', 0.0, 1.0,
                                          selected_choices=[(0.1 if feature_space == 'SAR_cAWEI+cNDWI' else 1.0)])
                    res['shrinkage'] = shrinkage
                trial.leave_condition(feature_space+'solver')
            elif method == 'quadratic':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_HSV(O3)+cAWEI+cNDWI', 'cAWEI', 'SAR'],
                                                          condition=True)
                res['z_feature_space'] = feature_space
                reg = 0.0
                if feature_space == 'cAWEI':
                    reg = 0.001
                elif feature_space == 'SAR':
                    reg = 1.0
                trial.suggest_uniform(feature_space+'reg_param', 0.0, 1e3, selected_choices=[reg])
                res['reg_param'] = reg
            elif method == 'naive':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_HSV(O3)+cAWEI+cNDWI', 'cAWEI+cNDWI', 'SAR'],
                                                          condition=False)
                res['z_feature_space'] = feature_space
                pass
            else:
                raise NotImplementedError(f'Method {method} is not supported yet')
            trial.leave_condition([method+'_z_feature_space', 'method'])
            return res
        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split), config, seed,
                                   timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
