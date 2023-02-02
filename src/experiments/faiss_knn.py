import functools as funct

import optuna
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper
from pre_process import *
from serialisation import *

NAME = 'faiss_knn'


def execute_grid_search_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_grid_search'
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
        selected_filters = all_filters
    f_names = {}

    for filter_method in all_filters:
        train_cons = construct_feature_spaces(data_folder, split='train', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=True)
        valid_cons = construct_feature_spaces(data_folder, split='valid', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=True)
        cur_f_names = [t[0] for t in
                       train_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: (v1, v2) for (k1, v1), (k2, v2) in zip(train_cons, valid_cons)}
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        feature_space = trial.suggest_categorical('feature_space', f_names[filter], condition=True)
        weights = trial.suggest_categorical('weights', ['uniform', 'sklearn', 'direct'], selected_choices=['uniform'])
        trial.suggest_int('k', 1, 10_000, selected_choices=[1,1024])
        metric = trial.suggest_categorical('metric', ['euclidean', 'cosine'], selected_choices=['euclidean'])
        trial.leave_condition(['feature_space, filter'])

    feature_pipe_cache = SimpleCachedPipeline()
    label_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        label_pipe = MultiSequenceModule([
            MultiDistributorModule([
                PipelineAdaptorModule(selection_criterion=None,
                                      pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                                  split=SPLIT_TRAIN, as_array=True),
                                      dataset_name='train_labels'
                                      ),
                PipelineAdaptorModule(selection_criterion=None,
                                      pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                                  split=SPLIT_VALIDATION,
                                                                                  as_array=True),
                                      dataset_name='valid_labels'
                                      )
            ]),
            MaskModule(source_criterion=NameSelectionCriterion('train_labels'),
                       res_name='train_mask',
                       mask_label=-1)
        ])
        label_pipe_cache.set_module(label_pipe)
        feature_space = _config['feature_space']
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
        adaptor = SupervisedSklearnAdaptor('classification.FaissKNNClassifier',
                                           params={'k': _config['k'],
                                                   'weights': _config['weights'],
                                                   'metric': _config['metric'],
                                                   'use_gpu': True},
                                           save_file=None)

        complete_pipe = MultiSequenceModule([
            data_construction,
            SupervisedSklearnAdaptorModule(adaptor, feature_criterion=NameSelectionCriterion('train_features'),
                                           label_criterion=NameSelectionCriterion('train_labels'),
                                           mask_criterion=NameSelectionCriterion('train_mask'),
                                           do_predict=False),
            # SupervisedSklearnAdaptorModule(adaptor, feature_criterion=NameSelectionCriterion('train_features'),
            #                                prediction_dataset_name='train_prediction',
            #                                prediction_channel_name='train_prediction',
            #                                do_fit=False),
            # MetricsModule(prediction_criterion=NameSelectionCriterion('train_prediction'),
            #               label_criterion=NameSelectionCriterion('train_labels'),
            #               source_criterion=NameSelectionCriterion('train_features'),
            #               per_data_computation=ContingencyMatrixComputation('train.contingency'),
            #               delete_prediction=True),
            SupervisedSklearnAdaptorModule(adaptor, feature_criterion=NameSelectionCriterion('valid_features'),
                                           prediction_dataset_name='valid_prediction',
                                           prediction_channel_name='valid_prediction',
                                           do_fit=False),
            MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                          label_criterion=NameSelectionCriterion('valid_labels'),
                          source_criterion=NameSelectionCriterion('valid_features'),
                          per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                          delete_prediction=False)
        ])
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

        def accuracy_from_contingency(d: dict) -> Tuple[float, int]:
            matrix = d[KEY_CONTINGENCY_MATRIX]
            ul = d[KEY_CONTINGENCY_UNIQUE_LABELS]
            up = d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
            divisor = matrix[ul!=-1].sum()
            dividend = (matrix[ul == 0, up == 0].sum()
                        + matrix[ul == 1, up == 1].sum())
            return (0.0, 0) if divisor == 0 else (float(dividend / divisor), 1)

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if k.startswith('valid')]
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values))
        print(f'Mean Overall Accuracy is {res}.')
        return res

    run_sacred_grid_search(experiment_base_folder, ex_name, main, config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
