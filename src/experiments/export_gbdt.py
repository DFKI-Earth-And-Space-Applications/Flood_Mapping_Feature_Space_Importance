import functools as funct
import random
import sys
import os.path as path
import traceback
from typing import Tuple, List

import lightgbm
import numpy as np
import optuna
import torch
from classification import SavableLGBMClassifier
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, \
    iou_from_contingency, TrialAbort, DirectFileStorage
from pre_process import *
from serialisation import *

NAME = 'export_gbdt'

# used for visualisation - see visualize_images
def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                 feature_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True, predict_factory: bool = False) -> Union[MultiPipeline, Tuple[MultiPipeline, Callable]]:

    if predict_factory:
        return complete_pipe, (lambda criterion: SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=criterion,
                                       do_fit=False,
                                       prediction_channel_name='valid_prediction',
                                       prediction_dataset_name='valid_prediction'))
    return complete_pipe


def export(data_folder: str, experiment_base_folder: str,
           use_datadings: bool = True, seed: Optional[int] = None,
           filters: Optional[List[str]] = None,
           redirect_output: bool = True, num_workers: int = 4,
           **kwargs):
    # TODO allow-non-datadings (aka tiff-file-collection) outputs

    obs = DirectFileStorage(path.join(experiment_base_folder, NAME), redirect_sysout=redirect_output, do_host=False)
    print(f'Starting export to folder {experiment_base_folder}')
    with RunHook(obs, None, seed=seed) as hook:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        for filter in filters: # should always be 'lee_improved'
            if filter != 'lee_improved':
                print(f'WARNING: The model has been trained on SAR-Data with the improved lee-sigma filter '
                      f'but you are requesting the filter {filter}! Export will commence, but this may have unwanted side-'
                      f'effects.', file=sys.stderr)
            p = {'random_state': seed, 'boosting_type': 'gbdt',
                                                       'num_leaves': 128,
                                                       'max_depth': -1,
                                                       'learning_rate': 0.1,
                                                       'n_estimators': 200,
                                                       'subsample_for_bin': 262144, # = 512 * 512
                                                       'class_weight': None,
                                                       'min_split_gain': 0.0,
                                                       'reg_alpha': 0.0,
                                                       'reg_lambda': 1.0,
                                                       'n_jobs': num_workers}
            lgbmclassifier = lightgbm.LGBMClassifier(**p)
            adaptor = SupervisedSklearnAdaptor(('lightgbm.LGBMClassifier', SavableLGBMClassifier('gbm_forest')),
                                               params={'n_jobs': num_workers},
                                               clear_on_predict=False,
                                               predict_proba=True,
                                               predict_batch_size=128,
                                               allow_no_fit=True)

            cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder,
                                                                                                        filters,
                                                                                                        use_datadings,
                                                                                                        eval_split=SPLIT_VALIDATION,
                                                                                                        in_memory=False)
            train_cons, _ = cons_by_filter[filter]['SAR_HSV(O3)+cAWEI+cNDWI']
            seq: List[MultiPipeline] = []
            try:
                obs.start({'seed': seed,
                           'params': p})
                for split in VAILD_SPLITS:
                    is_final_split = split in [SPLIT_TEST, SPLIT_BOLIVIA]
                    f_name, cn, _ = evaluate_type_and_split(TYPE_GBDT_WEAK_LABEL, split)
                    cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters,
                                                                                                            use_datadings, eval_split=split,
                                                                                                            in_memory=False)
                    _, eval_cons = cons_by_filter[filter]['SAR_HSV(O3)+cAWEI+cNDWI']
                    sub_seq = [
                        PipelineAdaptorModule(selection_criterion=None,
                                              pipe_module=eval_cons,
                                              dataset_name='features'
                                              ),
                        SupervisedSklearnAdaptorModule(adaptor,
                                                       feature_criterion=NameSelectionCriterion('features'),
                                                       do_fit=False,
                                                       prediction_channel_name=cn[0],
                                                       prediction_dataset_name='prediction',
                                                       delete_features=True),
                        SaveToDatadings(NameSelectionCriterion('prediction'),
                                        output_file=path.join(obs.dir, f_name, f_name + '.msgpack'),
                                        delete=not is_final_split)

                    ]
                    if is_final_split:
                        sub_seq.extend([
                            PipelineAdaptorModule(selection_criterion=None,
                                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder,
                                                                                              type=TYPE_LABEL,
                                                                                              split=split,
                                                                                              as_array=False,
                                                                                              in_memory=False),
                                                  dataset_name='labels'
                                              ),
                            PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('prediction'),
                                                  pipe_module=ProbabilityToValue(),
                                                  dataset_name='prediction'
                                              ),
                            MetricsModule(prediction_criterion=NameSelectionCriterion('prediction'),
                                          label_criterion=NameSelectionCriterion('labels'),
                                          per_data_computation=ContingencyMatrixComputation(split+'.contingency'),
                                          delete_label=True,
                                          delete_prediction=True)])
                    seq.append(MultiSequenceModule(sub_seq))
                complete_pipe = MultiDistributorModule(seq)
                seq: List[MultiPipeline] = [
                    # PipelineAdaptorModule(selection_criterion=None,
                    #                       pipe_module=train_cons,
                    #                       dataset_name='features'
                    #                       ),
                    # PipelineAdaptorModule(selection_criterion=None,
                    #                       pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                    #                                                                   split=SPLIT_TRAIN, as_array=True),
                    #                       dataset_name='train_labels'
                    #                       ),
                    # MaskModule(source_criterion=NameSelectionCriterion('train_labels'),
                    #            res_name='train_mask',
                    #            mask_label=-1),
                    # SupervisedSklearnAdaptorModule(adaptor,
                    #                                feature_criterion=NameSelectionCriterion('features'),
                    #                                label_criterion=NameSelectionCriterion('train_labels'),
                    #                                mask_criterion=NameSelectionCriterion('train_mask'),
                    #                                do_predict=False,
                    #                                delete_features=True,
                    #                                delete_labels=True,
                    #                                delete_mask=True),
                    complete_pipe
                ]
                complete_pipe = MultiSequenceModule(seq)
                summary = Summary([], lambda a: '')
                summary.set_hook(hook.add_result_metric_prefix('test.contingency')
                                 .add_result_metric_prefix('bolivia.contingency'))
                complete_pipe(summary)
                def calc_res(key: str) -> Tuple[float, float]:
                    def masked_list_average(ls: List[Tuple[float, int]]) -> float:
                        return float(sum(map(lambda t: t[0], ls)) / sum(map(lambda t: t[1], ls)))

                    final_contingency_matrices = [d[0] for k, d in hook.recorded_metrics.items()
                                                  if k.startswith(key)]
                    if not final_contingency_matrices:
                        return None, None
                    acc_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
                    iou_values = [iou_from_contingency(d) for d in final_contingency_matrices]
                    return masked_list_average(acc_values), masked_list_average(iou_values)

                test_acc, test_iou = calc_res('test.contingency')
                bolivia_acc, bolivia_iou = calc_res('bolivia.contingency')

                print(f'Mean Overall IOU is {test_iou} and {bolivia_iou} on the test and '
                      f'bolivia test sets respectively.')
                print(f'Mean Overall Accuracy is {test_acc} and {bolivia_acc} on the test and '
                      f'bolivia test sets respectively.')
            except Exception:
                obs.fail()
                print('Execution Failed!', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            else:
                obs.complete({'test_acc': test_acc, 'test_iou': test_iou,
                              'bolivia_acc': bolivia_acc, 'bolivia_iou': bolivia_iou})
            # Average is 0.467s per image on a i7-8750H @ 2.2Ghz (utilizing all 12 cores)