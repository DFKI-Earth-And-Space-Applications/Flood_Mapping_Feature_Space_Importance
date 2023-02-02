import argparse
import os
import sys
import traceback

import torch
import utils

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # make it work deterministically on cuda 10.2+
DEF_TIMEOUT = None
if __name__ == '__main__':
    try:
        # avoid linux problems... - see https://pytorch.org/docs/stable/notes/multiprocessing.html
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print(traceback.format_exc(), file=sys.stderr)
        print('Failed to set mp-start-method... Hopefully this works nonetheless!')
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('experiment_folder')
    parser.add_argument('module',  help='Specify the python module in which the code to execute resides.')
    parser.add_argument('--fun', dest='fun', default='execute_grid_search_experiments',
                        help='Specify the function to execute')
    parser.add_argument('--datadings', dest='use_datadings', action='store_true')
    parser.add_argument('--no-datadings', dest='use_datadings', action='store_false')
    parser.add_argument('--redirect-console', dest='redirect_output', action='store_true')
    parser.add_argument('--no-redirect-console', dest='redirect_output', action='store_false')
    parser.add_argument('--set_torch_deterministic', dest='torch_deterministic', action='store_true')
    parser.add_argument('--torch_allow_nondeterministic', dest='torch_deterministic', action='store_false')
    parser.add_argument('--num_workers', dest='num_workers', default=6, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--n_trials', dest='n_trials', type=int, default=None)
    parser.add_argument('--n_init', dest='n_init', type=int, default=None)
    parser.add_argument('-s', '--seed', dest='seed', default=None, type=int, help='Seed to use for reproducability.')
    parser.add_argument('-ss', '--seeds', nargs='+', dest='seeds', default=None, help='Set of seeds to use for reproducability.')
    parser.add_argument('-t', '--timeout', dest='timeout', default=DEF_TIMEOUT, type=int, help='Optuna timeout in s')

    parser.add_argument('--additional_data_folders', nargs='+', dest='additional_data_folders', default=None,
                        help='Varags param for additional data-folders that should be considered for finding '
                             'datadings-files')
    #parser.add_argument('-j', '--n_jobs', dest='n_jobs', default=-1, type=int, help='Number of concurrent jobs to run, if supported by the pipeline')
    parser.add_argument('--filters', nargs='+', dest='filters', default=None, help='Filters to use for the experiments')
    parser.add_argument('--additional_list', nargs='+', dest='additional_list', default=None,
                        help='For the combined experiment this represents the tested combined algorithms. Set to None to test all')
    # only for calibration:
    parser.add_argument('--ensemble', dest='ensemble', action='store_true')
    parser.add_argument('--single_predictor', dest='ensemble', action='store_false')

    parser.set_defaults(use_datadings=True, redirect_output=True, torch_deterministic=True, ensemble=False)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(args.torch_deterministic)
    timeout = args.timeout
    if timeout is not None and timeout < 0:
        raise ValueError('Negative Timeout given!')
    if args.seed is not None and args.seeds is not None:
        raise RuntimeError('Cannot specify both a set of seeds to test and a single seed to test!!!')
    elif args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        print('No seed specified. Running without seed.')
        seeds = [None]
    ex_fun = utils.get_module_fun(args.module, args.fun)
    df = args.data_folder
    if args.additional_data_folders is not None:
        df = [df] + args.additional_data_folders
    for seed in seeds:
        experiment_instances = ex_fun(data_folder=df, experiment_base_folder=args.experiment_folder,
                                      use_datadings=args.use_datadings, seed=seed, timeout=timeout, filters=args.filters,
                                      redirect_output=args.redirect_output, num_workers=args.num_workers,
                                      additional_list=args.additional_list, n_trials=args.n_trials,
                                      batch_size=args.batch_size, ensemble=args.ensemble)
