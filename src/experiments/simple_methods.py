import functools as funct
import sys

import optuna
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'simple_classifiers'

# used for visualisation - see visualize_images
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
    if method == 'SGDClassifier':
        params = {'random_state': _seed,
                  'loss': _config['loss'],
                  'alpha': _config['alpha'],
                  'penalty': _config['penalty'],
                  'learning_rate': _config['sgd_learning_rate'],
                  'class_weight': _config['class_weight'],
                  'max_iter': _config['max_iter'],
                  'n_iter_no_change': 4,
                  'n_jobs': 4}
        if 'epsilon' in _config:
            params['epsilon'] = _config['epsilon']
        if 'eta0' in _config:
            params['eta0'] = _config['eta0']
        if 'l1_ratio' in _config:
            params['l1_ratio'] = _config['l1_ratio']
        adaptor = SupervisedSklearnAdaptor('sklearn.linear_model.SGDClassifier',
                                           params=params,
                                           save_file='sgd_classifier.pkl',
                                           clear_on_predict=False)
    elif method == 'GB-RF':
        adaptor = SupervisedSklearnAdaptor('lightgbm.LGBMClassifier',
                                           params={'random_state': _seed,
                                                   'boosting_type': _config['boosting_type'],
                                                   'num_leaves': _config['num_leaves'],
                                                   'max_depth': _config['max_depth'],
                                                   'learning_rate': _config['learning_rate'],
                                                   'n_estimators': _config['n_estimators'],
                                                   'subsample_for_bin': _config['subsample_for_bin'],
                                                   'class_weight': _config['class_weight'],
                                                   'min_split_gain': _config['min_split_gain'],
                                                   'reg_alpha': _config['reg_alpha'],
                                                   'reg_lambda': _config['reg_lambda'],
                                                   'n_jobs': -1},
                                           save_file=('gbm_forest.pkl' if include_save_operations else None),
                                           clear_on_predict=False)
    else:
        raise RuntimeError(f'Unknown method {method}!')
    seq = [
        data_construction,
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('train_features'),
                                       label_criterion=NameSelectionCriterion('train_labels'),
                                       mask_criterion=NameSelectionCriterion('train_mask'),
                                       do_predict=False),
        # SupervisedSklearnAdaptorModule(adaptor,
        #                                feature_criterion=NameSelectionCriterion('train_features'),
        #                                do_fit=False,
        #                                prediction_channel_name='train_prediction',
        #                                prediction_dataset_name='train_prediction'),
        # MetricsModule(prediction_criterion=NameSelectionCriterion('train_prediction'),
        #               label_criterion=NameSelectionCriterion('train_labels'),
        #               source_criterion=NameSelectionCriterion('train_features'),
        #               per_data_computation=ContingencyMatrixComputation('train.contingency'),
        #               delete_prediction=True),
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('valid_features'),
                                       do_fit=False,
                                       prediction_channel_name='valid_prediction',
                                       prediction_dataset_name='valid_prediction'),

    ]
    if include_save_operations:
        if eval_split == SPLIT_TEST:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     # WARNING: This accidentally combines both computations!!!
                                     per_region_computation=ContingencyMatrixComputation('valid.contingency'),
                                     delete_prediction=False))
        else:
            seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                     label_criterion=NameSelectionCriterion('valid_labels'),
                                     source_criterion=NameSelectionCriterion('valid_features'),
                                     per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                     delete_prediction=False))
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
        complete_pipe = raw_pipeline(data_folder, eval_split, label_pipe_cache, feature_pipe_cache, _config,
                                     _seed, cons_by_filter)
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
        # [SAR, OPT, O3, S2, RGB, RGBN, HSV(RGB), HSV(O3), cNDWI, cAWEI, cAWEI+cNDWI, HSV(O3)+cAWEI+cNDWI, SAR_OPT, SAR_O3, SAR_S2, SAR_RGB, SAR_RGBN, SAR_HSV(RGB), SAR_HSV(O3), SAR_cNDWI, SAR_cAWEI, SAR_cAWEI+cNDWI, SAR_HSV(O3)+cAWEI+cNDWI]

        feature_space = trial.suggest_categorical('feature_space', f_names[filter], condition=True)  # )
        num_leaves = None
        if method == 'GB-RF':
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss', 'rf'],
                                                      selected_choices=['gbdt'])
            # feature spaces with more than 9 features
            if feature_space in ['OPT', 'S2', 'SAR_OPT', 'SAR_S2', 'SAR_HSV(O3)+cAWEI+cNDWI']:
                leaf_choices = [32, 64, 128]
            elif feature_space in ['SAR', 'cNDWI', 'cAWEI']:
                leaf_choices = [2, 4]
            # feature spaces with at most 3 features
            elif feature_space in ['SAR', 'O3', 'RGB', 'HSV(RGB)', 'HSV(O3)', 'cNDWI', 'cAWEI']:#, 'cNDWI+NDVI']:
                leaf_choices = [4, 8]
            # 4 features
            elif feature_space in ['SAR_cNDWI', 'SAR_cAWEI', 'RGBN', 'cAWEI+cNDWI']:
                leaf_choices = [4, 8, 16]
            # 5 features
            elif feature_space in [ 'SAR_O3', 'SAR_RGB', 'SAR_HSV(RGB)', 'SAR_HSV(O3)']:#, 'SAR_cNDWI+NDVI']:
                leaf_choices = [8, 16, 32]
            # 6 or 7 features
            elif feature_space in ['SAR_RGBN', 'SAR_cAWEI+cNDWI', 'HSV(O3)+cAWEI+cNDWI']:
                leaf_choices = [16, 32, 64]
            else:
                raise ValueError(f'Unknown search space {feature_space}')
            num_leaves = trial.suggest_int(feature_space+'_num_leaves', 1, 1000, selected_choices=leaf_choices)
            max_depth = trial.suggest_int('max_depth', -1, 1000, selected_choices=[-1])
            learning_rate = trial.suggest_uniform('learning_rate', 1e-7, 1e3, selected_choices=[0.1, 0.01])
            n_estimators = trial.suggest_int('n_estimators', 1, 1000, selected_choices=[50, 100, 200])
            subsample_for_bin = trial.suggest_int('subsample_for_bin', 1, 1_000_000_000,
                                                  selected_choices=[512 * 512, 4 * 512 * 512])
            class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
            min_split_gain = trial.suggest_uniform('min_split_gain', 0.0, 1000., selected_choices=[0.0])
            min_child_weight = trial.suggest_uniform('min_child_weight', 0.0, 1e6, selected_choices=[50.0])
            min_child_samples = trial.suggest_int('min_child_samples', 1, 1_000_000, selected_choices=[100])
            reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 10.0, selected_choices=[0.0])
            reg_lambda = trial.suggest_uniform('reg_lambda', 0.0, 10.0, selected_choices=[0.0, 0.01, 1])
        elif method == 'SGDClassifier':
            loss = trial.suggest_categorical('loss', ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                                             selected_choices=['hinge', 'log', 'modified_huber'],
                                             condition=True)
            if loss in ['hinge', 'modified_huber', 'squared_hinge']:
                epsilon = trial.suggest_uniform('epsilon', 1e-7, 1e3, selected_choices=[0.1])
            trial.leave_condition('loss')
            alpha = trial.suggest_uniform('alpha', 1e-7, 1e3, selected_choices=[0.0001, 0.001, 0.01, 0.1, 1])
            penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'],
                                                selected_choices=['l2'],
                                                condition=True)
            if penalty == 'elasticnet':
                l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1, selected_choices=[0.15, 0.5])
            trial.leave_condition('penalty')

            learning_rate = trial.suggest_categorical('sgd_learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']
                                                      , selected_choices=['adaptive'], condition=True)
            if learning_rate in ['constant', 'invscaling', 'adaptive']:
                eta0 = trial.suggest_uniform('eta0', 1e-8, 1e3, selected_choices=[1e-4])
            trial.leave_condition('sgd_learning_rate')

            class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
            max_iter = trial.suggest_int('max_iter', 1, 1000, selected_choices=[20])
        else:
            raise NotImplementedError(f'Method {method} is not supported yet')
        trial.leave_condition(['z_feature_space', 'method'])
        if num_leaves is not None:
            return {'num_leaves': num_leaves}


    run_sacred_grid_search(experiment_base_folder, ex_name, pipeline(data_folder, cons_by_filter), config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)

# TODO check best parameters
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
            # [SAR, OPT, O3, S2, RGB, RGBN, HSV(RGB), HSV(O3), cNDWI, cAWEI, cAWEI+cNDWI, HSV(O3)+cAWEI+cNDWI, SAR_OPT, SAR_O3, SAR_S2, SAR_RGB, SAR_RGBN, SAR_HSV(RGB), SAR_HSV(O3), SAR_cNDWI, SAR_cAWEI, SAR_cAWEI+cNDWI, SAR_HSV(O3)+cAWEI+cNDWI]
            method = trial.suggest_categorical('method', ['GB-RF', 'SGDClassifier'], condition=True)
            res = {}
            if method == 'GB-RF':
                feature_space = trial.suggest_categorical('feature_space_gb', f_names[filter],
                                                          selected_choices=['SAR', 'SAR_HSV(O3)+cAWEI+cNDWI',
                                                                            'HSV(O3)+cAWEI+cNDWI'],
                                                          condition=True)  # )
                res['z_feature_space'] = feature_space
                boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss', 'rf'],
                                                          selected_choices=['gbdt'])
                # feature spaces with more than 9 features
                if feature_space in ['OPT', 'S2', 'SAR_OPT', 'SAR_S2', 'SAR_HSV(O3)+cAWEI+cNDWI']:
                    leaf_choices = [128]#[32, 64, 128]
                elif feature_space in ['SAR', 'cNDWI', 'cAWEI']:
                    leaf_choices = [2]#[2, 4]
                # feature spaces with at most 3 features
                elif feature_space in ['SAR', 'O3', 'RGB', 'HSV(RGB)', 'HSV(O3)', 'cNDWI', 'cAWEI']:  # , 'cNDWI+NDVI']:
                    leaf_choices = [4, 8]
                # 4 features
                elif feature_space in ['SAR_cNDWI', 'SAR_cAWEI', 'RGBN', 'cAWEI+cNDWI']:
                    leaf_choices = [4, 8, 16]
                # 5 features
                elif feature_space in ['SAR_O3', 'SAR_RGB', 'SAR_HSV(RGB)', 'SAR_HSV(O3)']:  # , 'SAR_cNDWI+NDVI']:
                    leaf_choices = [8, 16, 32]
                # 6 or 7 features
                elif feature_space in ['SAR_RGBN', 'SAR_cAWEI+cNDWI', 'HSV(O3)+cAWEI+cNDWI']:
                    leaf_choices = [64]#[16, 32, 64]
                else:
                    raise ValueError(f'Unknown search space {feature_space}')
                num_leaves = trial.suggest_int(feature_space + '_num_leaves', 1, 1000, selected_choices=leaf_choices)
                res['num_leaves'] = num_leaves
                max_depth = trial.suggest_int('max_depth', -1, 1000, selected_choices=[-1])
                learning_rate = trial.suggest_uniform('learning_rate', 1e-7, 1e3, selected_choices=[0.1])
                n_estimators = trial.suggest_int(feature_space+'_n_estimators', 1, 1000, selected_choices=[(50 if feature_space == 'SAR' else 200)])
                res['n_estimators'] = n_estimators
                subsample_for_bin = trial.suggest_int(feature_space+'_subsample_for_bin', 1, 1_000_000_000,
                                                      selected_choices=([4 * 512 * 512]
                                                                        if feature_space == 'HSV(O3)+cAWEI+cNDWI'
                                                                        else [512 * 512]))
                res['subsample_for_bin'] = subsample_for_bin
                class_weight = trial.suggest_categorical(feature_space+'_class_weight_gb', [None])
                res['class_weight'] = class_weight
                min_split_gain = trial.suggest_uniform('min_split_gain', 0.0, 1000., selected_choices=[0.0])
                min_child_weight = trial.suggest_uniform('min_child_weight', 0.0, 1e6, selected_choices=[50.0])
                min_child_samples = trial.suggest_int('min_child_samples', 1, 1_000_000, selected_choices=[100])
                reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 10.0, selected_choices=[0.0])
                reg_lambda = trial.suggest_uniform(feature_space+'reg_lambda', 0.0, 10.0,
                                                   selected_choices=([1.0]))
                res['reg_lambda'] = reg_lambda
                trial.leave_condition('feature_space_gb')
            elif method == 'SGDClassifier':
                feature_space = trial.suggest_categorical('feature_space_sgd', f_names[filter],
                                                          selected_choices=['SAR', 'SAR_HSV(O3)',
                                                                            'HSV(O3)'],
                                                          condition=True)  # )
                res['z_feature_space'] = feature_space
                loss = trial.suggest_categorical(feature_space+'_loss', ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                                                 selected_choices=(['hinge'] if feature_space == 'SAR' else ['log']),
                                                 condition=True)
                res['loss'] = loss
                if loss in ['hinge', 'modified_huber', 'squared_hinge']:
                    epsilon = trial.suggest_uniform('epsilon', 1e-7, 1e3, selected_choices=[0.1])
                trial.leave_condition(feature_space+'_loss')
                alpha = trial.suggest_uniform(feature_space+'_alpha', 1e-7, 1e3,
                                              selected_choices=([0.1] if feature_space == 'SAR' else [0.0001]))
                res['alpha'] = alpha
                penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'],
                                                    selected_choices=['l2'],
                                                    condition=True)
                if penalty == 'elasticnet':
                    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1, selected_choices=[0.15, 0.5])
                trial.leave_condition('penalty')

                learning_rate = trial.suggest_categorical('sgd_learning_rate',
                                                          ['constant', 'optimal', 'invscaling', 'adaptive']
                                                          , selected_choices=['adaptive'], condition=True)
                if learning_rate in ['constant', 'invscaling', 'adaptive']:
                    eta0 = trial.suggest_uniform('eta0', 1e-8, 1e3, selected_choices=[1e-4])
                trial.leave_condition('sgd_learning_rate')

                class_weight = trial.suggest_categorical(feature_space+'_class_weight',
                                                         (['balanced'] if feature_space == 'SAR' else [None]))
                res['class_weight'] = class_weight
                max_iter = trial.suggest_int('max_iter', 1, 1000, selected_choices=[20])
                trial.leave_condition('feature_space_sgd')
            else:
                raise NotImplementedError(f'Method {method} is not supported yet')
            trial.leave_condition('method')
            return res
        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)