# General Notes

The code has been deliberately designed for re-usability. The class Pipeline (see pipeline.py) represents a re-usable 
module (somewhat similiar to sklearns pipeline), which accepts a pytorch dataset (with numpy elements) as inputs and 
outputs a new (transformed, predicted, etc.) dataset. Similarily, MultiPipelines are pipelines operating on multiple 
datasets at once.

To re-use part of the code, simply copy the relevent module over and pass it a pytorch dataset as input.

# Running the code

## Dependencies
To create an enviornment with the required dependencies, the following sequence of commands should help get you started. If you find a missing dependency, please let me know in 
an issue.
```
conda create -n env_name python=3.9 numpy scipy pandas numba matplotlib
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install datadings==3.4.2 scikit-image==0.18.3 optuna==2.10.0 sacred==0.8.2 lightgbm==3.3.2 frozendict==2.3.1 sci-analysis
pip install -U scikit-learn==1.0.2 
conda install -y -c conda-forge faiss-gpu
```

## Preparing Data

Before running any experiments please download the Sen1Floods11 dataset. For the experiments in this repository
only the hand-labeled data is required. For this follow the [README](sen1floods11/README.md) of the 
dataloading code and also remember to split the data into datadings files as described there!!!
Note that this splitting process requires fewer dependencies and you will later no longer need tifffile or rasterio
so you may choose to do this in a seperate enviornment.

## Running Experiments

Experiments are started via `main.py` and **not via the individual experiment scripts**. 
Instead specify the folder containing the previously prepared data (`data_folder`), where to store experiment outputs
(`experiment_folder`) and the module (aka experiment) which you would like to run. If you want to only run inference on the final
test-splits, you have to specify additionally `--fun execute_final_experiments`. Also remember to pass `--filters lee_improved` 
when using SAR data.

You may also want to adjust `--seed` and `--n_jobs` to specify the seed to use and/or the number of parallel processes 
that a parallelizable training algorithm may employ. Note that there is some additionall paralellism due to the use of numba which you can only disable by setting the
`ENABLE_NUMBA_PARALLEL=0` enviornment variable and furthermore it is recommended to also set the `PYTHON_HASH_SEED` to the same value. Seeds used: 

42 1234 4998 125143 252134 296459 338544 365132 444068 637291 691561 702734 797387 908314 947120 984340

Due to differing paralell executions, results might still not be 100% identical but should be very close, when averaged.

Further note, that some experiment-modules are designed to run for more than 1 week on at least 12 cores, due to the large number
of different hyperparameter configurations being tested. In order to re-create validation set results for only the best-performing
configurations (and hence run the training similiar to how it would be done with `--fun execute_final_experiments`) copy the 
config function in the `execute_final_experiments` function of the respective module and use it to override the config function 
embedded into `execute_grid_search_experiments`.

## Processing Results

After you have run your experiments, you will have a lot of sacred-style folders representing individual runs. Since these 
may have a significant size, these should be compressed on the remote machine into a `.tar.gz` file (simply invoke tar on 
the folder previously specified as `experiment_folder`) before downloading. However, these do
not contain metric evaluations yet, but rather only confusion/contingency matrices. To obtain metrics in a csv based data base, 
run `experiment_processing.py` with a path to where you want to store your csv files (`database_folder`) and an additional 
`--result_folder` argument specifying a folder containing multiple of these tar archives or a `--result_files` argument
specifying the tar archives individually (multiple files such as multiple seeds should be combined in one database in order 
to permit comparison, however you can always add more results to database by invoking experiment_processing with the same
database folder again).

Note that TAR-ARCHIVES CREATED USING WINDOWS WILL MOST LIKELY NOT WORK since they often do not retain the property that all 
files in a folder are completely compressed into the archive before another folder is started. Instead use a linux istrubtion
to compress the experiment folders (wsl under windows 11 might also work, but hasn't been tested).

## Inspecting/Analyzing Results

Since reading large csv-files is unwieldly, `queries.py` permits printing result summaries given a `database_folder` and 
`experiment_name` (corresponds to the initial "name-id" of the experiments that were run, for instance `simple_classifiers_final` for
SGD and GBDT results on the test-set). Via `--config_values` a selection to specific sub-results may be done and combine_columns
for combining columns with identical meaning. For example, the following command will compute results on the test-set for the 
gradient boosting classifier (for acquiring bolivia results this would refer to `simple_classifiers_final_bolivia`, or for validation results
it would be `simple_classifiers_grid_search` (with additional config_values filters for e.g. class_weight, num_leaves, etc.), ...):

```
[<data_base_path>]
simple_classifiers_final_test
--label_criterion
valid_labels
--config_values
method=GB-RF
--combine_columns
"num_leaves=HSV(O3)+cAWEI+cNDWI_num_leaves;SAR_HSV(O3)+cAWEI+cNDWI_num_leaves;SAR_num_leaves"
"subsample_for_bin=SAR_HSV(O3)+cAWEI+cNDWI_subsample_for_bin;HSV(O3)+cAWEI+cNDWI_subsample_for_bin;SAR_subsample_for_bin;subsample_for_bin"
"alpha=SAR_HSV(O3)_alpha;SAR_alpha;HSV(O3)_alpha;alpha"
"class_weight=SAR_class_weight;class_weight"
"reg_lambda=SAR_HSV(O3)+cAWEI+cNDWIreg_lambda;HSV(O3)+cAWEI+cNDWIreg_lambda;SARreg_lambda;reg_lambda"
"n_estimators=HSV(O3)+cAWEI+cNDWI_n_estimators;SAR_HSV(O3)+cAWEI+cNDWI_n_estimators;SAR_n_estimators;n_estimators"
"feature_space=feature_space_gb;feature_space"
--save_folder
[<Output path for later latex-conversion>]
````

The optional save_folder argument can be used to produce output files that can be passed to `query_to_latex.py` for creating rows
in a latex table from these results given the specified subset of metrics...

To create plots `experiment_plots.py` may be used, again taking a database, output (plot) folder ,experiment_name and optional combine_columns and
config_values arguments.
