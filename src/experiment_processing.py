import dataclasses
import itertools
import os
import pickle as pkl
import tarfile
from os import path
from typing import Generator

import joblib as job

from metrics import *
from pre_process import *
from dnn import *
from utils import get_class_constructor, right_slash_split, skip_with_prefix

TRIALS_DF_FILE = 'trials.csv'
TRIALS_DISTRIBUTIONS = 'trial_distributions.json'
TRIALS_INTERMEDIATES_DIR = 'trial_intermediates'

FILE_TRIALS = 'trials.csv'
FILE_TRIAL_DISTRIBUTIONS = 'trial_distributions.json'
FILE_CONFIG = 'config.json'
FILE_PIPELINE = 'pipeline.json'
FILE_RUN = 'run.json'
FILE_METRICS = 'metrics.json'
FILE_COUT = 'cout.txt'
SACRED_FILES = frozenset([FILE_CONFIG, FILE_RUN, FILE_METRICS, FILE_PIPELINE])

TABLE_EXPERIMENT = 'Experiment'
TABLE_TRIAL = 'Trial'
TABLE_PIPELINE = 'Pipeline'
TABLE_PIPELINE_SUCCESSOR = 'is_successor'
TABLE_ARTIFACT = 'Artifact'
TABLE_EX_ARTIFACT = 'Experiment_Artifacts'
TABLE_METRICS_COMPUTED_BY = 'computed_by'

COLUMN_EXPERIMENT_NAME = 'name'
COLUMN_EXPERIMENT_SEED = 'seed'
COLUMN_TRIAL_EX_ID = 'ex_id'
COLUMN_TRIAL_TRIAL_NUM = 'trial_num'
COLUMN_TRIAL_START_TIME = 'start_time'
COLUMN_TRIAL_END_TIME = 'end_time'
COLUMN_PIPELINE_T_NUM = 'trial_num'
COLUMN_PIPELINE_EX_ID = 'ex_id'
COLUMN_PIPELINE_ROOT = 'root'
COLUMN_PIPELINE_FINAL = 'final'
COLUMN_PIPELINE_TYPE = 'type'
COLUMN_EX_ARTIFACT_EX_ID = 'ex_id'
COLUMN_EX_ARTIFACT_FNAME = 'f_name'
COLUMN_ARTIFACT_T_NUM = 'trial_num'
COLUMN_ARTIFACT_EX_ID = 'ex_id'
COLUMN_ARTIFACT_FNAME = 'f_name'
COLUMN_PIPELINE_SUCCESSOR = 'successor'
COLUMN_PIPELINE_PREDECESSOR = 'predecessor'
COLUMN_MCB_MODULE_ID = 'module_id'
COLUMN_MCB_METRIC_NAME = 'metric'
COLUMN_MCB_METRIC_ID = 'metric_id'

COLUMN_CONFIG_EX_ID = 'ex_id'
COLUMN_CONFIG_TRIAL_NUM = 'trial_num'

COLUMN_METRIC_MODULE_ID = 'module'
COLUMN_METRIC_STEP_ID = 'step_id'
COLUMN_METRIC_BATCH_SIZE = 'batch_size'
COLUMN_METRIC_DATA_ID = 'data_id'
COLUMN_METRIC_CLASS = 'class_name'
COLUMN_METRIC_VALUE = 'value'

COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME = 'transformer'
COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID = 'transformer_id'
COLUMN_SKLEARN_ADAPTOR_DO_FIT = 'do_fit'
COLUMN_SKLEARN_ADAPTOR_DO_PREDICT = 'do_predict'

# This is needed so that the pandas database can recognize the special handling of indices for these tables
_SPECIAL_TABLES = {TABLE_EXPERIMENT: lambda d: d.create_experiment_table_if_absent(True),
                   TABLE_TRIAL: lambda d: d.create_trial_table_if_absent(True),
                   TABLE_ARTIFACT: lambda d: d.create_artifact_table_if_absent(True),
                   TABLE_PIPELINE: lambda d: d.create_pipeline_table_if_absent(True),
                   TABLE_PIPELINE_SUCCESSOR: lambda d: d.create_pipeline_successor_table_if_absent(True),
                   TABLE_EX_ARTIFACT: lambda d: d.create_exartifact_table_if_absent(True)}


class PandasDatabase:
    def __init__(self, database_folder: Optional[str]):
        self.database_folder = database_folder
        self._scan_db_folder()
        self.tables: Dict[str, pd.DataFrame] = {}

    def _scan_db_folder(self) -> 'PandasDatabase':
        if self.database_folder is None:
            self.stored_tables = {}
        else:
            if not path.exists(self.database_folder):
                raise ValueError(f'{self.database_folder} does not exist')
            elif not path.isdir(self.database_folder):
                raise ValueError(f'Cannot have database in a non-dir: {self.database_folder}')
            self.stored_tables: Dict[str, str] = {
                f_name.rsplit('.csv', 1)[0]: path.join(self.database_folder, f_name)
                for f_name in os.listdir(self.database_folder) if f_name.endswith('.csv')
            }
        return self

    def get(self, item, default) -> pd.DataFrame:
        if item not in self:
            return default
        return self[item]

    def compute_if_absent(self, key, default_fun: Callable, prototype: bool = False) -> pd.DataFrame:
        use_def = key not in self.tables if prototype else key not in self
        if use_def:
            def_val = default_fun()
            self.tables[key] = def_val
            return def_val
        return self[key]

    def create_table_if_absent(self, key, column_spec: Iterable[Tuple[str, str]],
                               index_spec: Optional[Iterable[Tuple[str, str]]] = None,
                               prototype: bool = False) -> pd.DataFrame:
        index: Optional[Union[pd.Index, pd.MultiIndex]] = None
        if index_spec is not None:
            index_spec = list(index_spec)
            if len(index_spec) > 1:
                empty_data = [[]] * len(index_spec)
                index = pd.MultiIndex(levels=empty_data, codes=empty_data,
                                      names=[n for n, _ in index_spec])
                index = index.set_levels([pd.Series(dtype=t) for _, t in index_spec])
            elif len(index_spec) == 1:
                index: pd.Index = pd.Index(data=[], name=index_spec[0][0], dtype=index_spec[0][1])
            else:
                print(
                    'Got empty index spec! This is almost certainly not intentional... Continuing with default index.',
                    file=sys.stderr)
        res = self.compute_if_absent(key,
                                     lambda: pd.DataFrame({c: pd.Series(dtype=t) for c, t in column_spec}, index=index),
                                     prototype)
        #res = res.convert_dtypes()
        return res

    def create_experiment_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_EXPERIMENT,
                                           [(COLUMN_EXPERIMENT_NAME, 'str'), (COLUMN_EXPERIMENT_SEED, 'Int64')],
                                           prototype=prototype)

    def create_trial_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_TRIAL,
                                           [(COLUMN_TRIAL_START_TIME, 'datetime64[ns]'),
                                            (COLUMN_TRIAL_END_TIME, 'datetime64[ns]')],
                                           [(COLUMN_TRIAL_EX_ID, 'Int64'),
                                            (COLUMN_TRIAL_TRIAL_NUM, 'Int64')],
                                           prototype=prototype)

    def create_artifact_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_ARTIFACT,
                                           [(COLUMN_ARTIFACT_T_NUM, 'Int64'), (COLUMN_ARTIFACT_EX_ID, 'Int64'),
                                            (COLUMN_ARTIFACT_FNAME, 'str')],
                                           prototype=prototype)

    def create_pipeline_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_PIPELINE,
                                           [(COLUMN_PIPELINE_T_NUM, 'Int64'), (COLUMN_PIPELINE_EX_ID, 'Int64'),
                                            (COLUMN_PIPELINE_TYPE, 'str'),
                                            (COLUMN_PIPELINE_ROOT, 'bool'),
                                            (COLUMN_PIPELINE_FINAL, 'bool')],
                                           prototype=prototype)

    def create_exartifact_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_ARTIFACT,
                                           [(COLUMN_EX_ARTIFACT_EX_ID, 'Int64'), (COLUMN_ARTIFACT_FNAME, 'str')],
                                           prototype=prototype)

    def create_pipeline_successor_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_PIPELINE_SUCCESSOR,
                                           [(COLUMN_PIPELINE_PREDECESSOR, 'Int64'),
                                            (COLUMN_PIPELINE_SUCCESSOR, 'Int64')],
                                           prototype=prototype)

    def create_metric_computed_by_table_if_absent(self, prototype: bool = False) -> pd.DataFrame:
        return self.create_table_if_absent(TABLE_METRICS_COMPUTED_BY,
                                           [(COLUMN_MCB_MODULE_ID, 'Int64'),
                                            (COLUMN_MCB_METRIC_ID, 'Int64')],#,
                                            #(COLUMN_MCB_METRIC_NAME, 'str')],
                                           prototype=prototype)

    def __getitem__(self, item: str) -> pd.DataFrame:
        if item not in self.tables:
            index_col = 0
            if item in _SPECIAL_TABLES:
                tab: pd.DataFrame = _SPECIAL_TABLES[item](self)
                if tab.index.nlevels > 1:
                    index_col = list(range(tab.index.nlevels))
            if item not in self.stored_tables:
                self._scan_db_folder()
            if item not in self.stored_tables:
                raise KeyError(f'Cannot find table {item} in {self.database_folder} and it has not been set yet!!!')
            res = pd.read_csv(self.stored_tables[item], parse_dates=True, index_col=index_col).convert_dtypes()
            self.tables[item] = res
            return res
        return self.tables[item]

    def __contains__(self, item: str):
        return item in self.stored_tables or item in self.tables

    def __setitem__(self, key: str, value: pd.DataFrame):
        if key in self.tables:
            cur_value = self.tables[key]
            for col_name in cur_value.columns.values:
                if col_name not in value.columns.values:
                    print(f'WARNING: Found deleted column {col_name} in database for key {key}. If this is intentional,'
                          f'this warning can be safely ignored, otherwise you might be overriding an incompatible table.')
                elif cur_value[col_name].dtype != value[col_name].dtype:
                    print(
                        f'WARNING: Found incompatible (dat) column {col_name} in database for key {key}. If this is intentional,'
                        f'this warning can be safely ignored, otherwise you might be overriding an incompatible table.')
        self.tables[key] = value

    def _save_item(self, key: str, table: pd.DataFrame):
        f_name = path.join(self.database_folder, key + '.csv')
        self.stored_tables[key] = f_name
        table.to_csv(f_name, index=True)

    def __delitem__(self, key: str):
        if key in self.tables:
            self._save_item(key, self.tables[key])
            del self.tables[key]

    def save(self, clear: bool = True):
        assert self.database_folder is not None, 'Cannot save virtual database!'
        for key, table in self.tables.items():
            self._save_item(key, table)
        if clear:
            self.tables.clear()

class ReadOnlyPandasDatabase(PandasDatabase):
    def _save_item(self, key: str, table: pd.DataFrame):
        print(f'Key {key} should have been saved, but as this is a read only database, this operation could not be performed.')
        f_name = path.join(self.database_folder, key + '.csv')
        self.stored_tables[key] = f_name
        #table.to_csv(f_name, index=True)

@dataclass
class MetricOptions:
    exclude_clouds: bool = True
    include_lsas: bool = False
    enforce_flood_no_flood_present: bool = True
    is_non_total_calculation: bool = True
    force_max_agree: bool = False
    is_parallel_data_computation: bool = True
    allow_cluster_metrics: bool = True

@dataclass
class MetricConfiguration:
    parallel_instance: job.Parallel
    options: MetricOptions
    random_state: int = 42
    batch_iter: int = 1000
    batch_sizes: Optional[List[int]] = dataclasses.field(default_factory=lambda:[2, 4, 8, 16])
    allow_total_from_per_data: bool = True
    max_num_unique_predictions: Optional[int] = None

def process_experiment_head(tar: tarfile.TarFile, ex_info: tarfile.TarInfo, database: PandasDatabase) -> Tuple[
    int, str, Optional[int]]:
    """
    Processes the top-level-entry (folder) in a tar-gzipped experiment file.
    :param tar: The tar in which it is contained
    :param ex_info: the info to be examined
    :param database: the database in which the current experiments are stored and to which it is to be added
    :return: The index of the resulting experiment
    """
    assert ex_info.isdir(), 'Expected first tar entry to be the top-level directory (depicting the experiment)!'
    experiment_table = database.create_experiment_table_if_absent()
    split = ex_info.name.rsplit('_', maxsplit=1)
    if len(split) > 1 and split[-1].isdigit():
        name, seed = split[0], int(split[-1])
    else:
        name, seed = ex_info.name, None
    if seed is None:
        seed_mask = experiment_table[COLUMN_EXPERIMENT_SEED].isna()
    else:
        seed_mask = experiment_table[COLUMN_EXPERIMENT_SEED].eq(seed, fill_value=seed+1)
    matching_indices = experiment_table.index[(experiment_table[COLUMN_EXPERIMENT_NAME] == name) & (seed_mask)].tolist()
    if not matching_indices:
        id = len(experiment_table)
        name_type, seed_type = experiment_table[COLUMN_EXPERIMENT_NAME].dtype, experiment_table[
            COLUMN_EXPERIMENT_SEED].dtype
        experiment_table.loc[id] = (name, seed)
        experiment_table[COLUMN_EXPERIMENT_NAME] = experiment_table[COLUMN_EXPERIMENT_NAME].astype(name_type)
        experiment_table[COLUMN_EXPERIMENT_SEED] = experiment_table[COLUMN_EXPERIMENT_SEED].astype(seed_type)
        return id, name, seed
    elif len(matching_indices) == 1:
        return matching_indices[0], ex_info.name, seed
    else:
        raise ValueError(f'Found inconsistent database - expected no duplicates to exist for {name} with seed {seed}. '
                         f'However found duplicates in rows {str(matching_indices)}.')


class SerializedObject:
    def __init__(self, id: int, type_name: str, actual_type: Any, serialized_content: Dict[str, Any]):
        self.id = id
        self.type_name = type_name
        self.actual_type = actual_type
        self.serialized_content: Dict[str, Any] = serialized_content

    def __getitem__(self, item: str) -> Any:
        return self.serialized_content[item]

    def __contains__(self, item: str) -> bool:
        return item in self.serialized_content

    def __str__(self) -> str:
        return f'SerializedObject(id={self.id}, type_name={self.type_name}, serialized_content={self.serialized_content})'

    def __repr__(self) -> str:
        return f'SerializedObject(id={self.id}, type_name={self.type_name}, serialized_content={self.serialized_content})'


class UnresolvedVariable:
    def __init__(self, id: int, type_name: str, value: Dict[str, Any]):
        self.id = id
        self.type_name = type_name
        self.value = value

    def resolve(self, var_store: 'VariableStore') -> SerializedObject:
        def read_values(cur_val: Any) -> Any:
            if isinstance(cur_val, dict):
                type_attribute = cur_val['type']
                if type_attribute == 'REFERENCE_VARIABLE':
                    return var_store[int(cur_val['value']['id'])]
                elif type_attribute == 'dict':
                    assert isinstance(cur_val['value'], dict), \
                        f'Type was declared as dictionary, however non dictonary given! ' \
                        f'Failed to read values from {cur_val}'
                    return {k: read_values(v) for k, v in cur_val['value'].items()}
                else:
                    return read_values(cur_val['value'])
            elif isinstance(cur_val, list):
                return tuple(read_values(v) for v in cur_val)
            else:
                return cur_val

        return SerializedObject(self.id, self.type_name, get_class_constructor(self.type_name), read_values(self.value))


class VariableStore:
    def __init__(self, pipeline: Dict[str, Any]):
        if 'root' not in pipeline:
            raise RuntimeError(f'Expected pipeline save to always know about the root module! Given: {pipeline}')
        if 'variables' not in pipeline:
            raise RuntimeError(f'Expected pipeline save to always contain variables! Given: {pipeline}')
        if any(map(lambda d: 'type' not in d or d['type'] != 'VALUE_VARIABLE' or 'value' not in d,
                   pipeline['variables'])):
            raise RuntimeError(
                f'Cannot handle non-value-variable entries in the variable definition! Given: {pipeline}')
        pipeline = {'root': pipeline['root'],
                    'variables': [d['value'] for d in pipeline['variables']]}
        if any(map(lambda d: 'value' not in d or 'type' not in d or 'id' not in d, pipeline['variables'])):
            raise RuntimeError(f'Cannot handle value-variables lacking "id", "type" or "value" keys! Given: {pipeline}')
        self.root: int = int(pipeline['root'])
        self.vars: Dict[int, UnresolvedVariable] = {
            int(d['id']): UnresolvedVariable(int(d['id']), d['type'], d['value'])
            for d in pipeline['variables']}
        self.resolved_vars: Dict[int, SerializedObject] = {}

    def __getitem__(self, item: int) -> SerializedObject:
        if item in self.resolved_vars:
            return self.resolved_vars[item]
        elif item in self.vars:
            resolved = self.vars[item].resolve(self)
            self.resolved_vars[item] = resolved
            return resolved
        else:
            raise RuntimeError(
                f'Queried id {item}, but it can neither be found in the unresolved nor in the resolved store!')

    def resolve(self) -> SerializedObject:
        if self.root not in self.vars:
            raise RuntimeError(f'Root {self.root} not found!')
        return self[self.root]

    def max_id(self) -> int:
        return max(self.vars.keys())


# The idea of a reference is the following:
# As id's and similiar stuff are only valid locally whilst iterating the tar, keep them in a reference
# Once iteration is complete a second pass can transform them into valid (database wide unique) id's
@dataclass(eq=True, frozen=False, unsafe_hash=True)
class Reference():
    referenced: int

@dataclass
class PipeInfo():
    pipe_references: List[Reference]
    pipe_successors: List[Tuple[Reference, Reference]]
    _type_instances: List[Tuple[str, List[Tuple[Reference, Dict[str, Any]]]]]
    artifact_ref_map: Dict[str, Reference]
    module_metrics: List[Tuple[Reference, str, Dict[str, Any]]]

    def metric_instances(self) -> Generator[Tuple[str, List[Tuple[Reference, Reference, Dict[str, Any]]]],None,None]:
        for name in {name for _, name, _ in self.module_metrics}:
            yield (name, [(mod, value) for mod, name1, value in self.module_metrics if name1 == name])

    # def metric_references(self) -> Generator[Reference,None,None]:
    #     for _, _, mid, _ in self.module_metrics:
    #         yield mid
    #
    # def metrics_relation(self) -> Generator[Tuple[Reference, Reference],None,None]:
    #     for mod, _, mid, _ in self.module_metrics:
    #         yield (mod, mid)

    def type_instances(self) -> Generator[Tuple[str, List[Tuple[Reference, Dict[str, Any], bool, bool]]],None,None]:
        unfolded_predecessors = {ref for ref, _ in self.pipe_successors}
        unfolded_successors = {ref for _, ref in self.pipe_successors}
        in_relation = unfolded_predecessors.union(unfolded_successors)

        def is_root(ref: Reference) -> bool:
            return (ref in in_relation) and (ref not in unfolded_successors)

        def is_final(ref: Reference) -> bool:
            return (ref in in_relation) and (ref not in unfolded_predecessors)

        for name, ls in self._type_instances:
            yield (name, [(ref, value, is_root(ref), is_final(ref)) for ref, value in ls])

@dataclass
class BuildingPipeInfo():
    next_free_id: int # next module id
    pipe_successors: List[Tuple[Reference, Reference]]
    type_instances: Dict[str, List[Tuple[Reference, Dict[str, Any]]]]
    pipe_ref_map: Dict[int, Reference]
    artifact_ref_map: Dict[str, Reference]
    # Tuples of Form (module_ref, metric_name, metric_id)
    module_metrics: List[Tuple[Reference, str, Dict[str, Any]]]
    #metric_ref_map: List[Reference]
    next_free_aid: int = 0 # next artifact id
    #next_free_mid: int = 0 # next metric id


    def __init__(self, next_free_id: int) -> None:
        super().__init__()
        self.next_free_id = next_free_id
        self.pipe_successors = []
        self.type_instances = {}
        self.pipe_ref_map = {}
        self.artifact_ref_map = {}
        #self.metric_ref_map = []
        self.module_metrics = []

        self.next_free_aid = 0
        self.next_free_mid = 0
        self.next_free_config_id = 0

    def add_relations(self, new_relations: Iterable[Tuple[int, int]]):
        self.pipe_successors.extend(map(lambda t: (self.pipe_ref(t[0]), self.pipe_ref(t[1])), new_relations))

    def pipe_ref(self, id: int) -> Reference:
        ref = self.pipe_ref_map.get(id, Reference(id))
        self.pipe_ref_map[id] = ref
        return ref

    def artifact_ref(self, name: str, create: bool = False) -> Reference:
        if name in self.artifact_ref_map:
            return self.artifact_ref_map[name]
        elif not create:
            raise ValueError(f'{name} is not a valid artifact!')
        ref = Reference(self.next_free_aid)
        self.next_free_aid += 1
        self.artifact_ref_map[name] = ref
        return ref

    def add_artifacts(self, names: Iterable[str]):
        self.artifact_ref_map.update({name: self.artifact_ref(name, True) for name in names})

    def add_artifact_prefix(self, prefix: str):
        self.artifact_ref_map = {prefix + name: ref for name, ref in self.artifact_ref_map.items()}

    def add_module(self, id: Optional[int], name: str, data: Dict[str, Any]) -> Reference:
        if id is None:
            id = self.next_free_id
            self.next_free_id += 1
        is_new = id not in self.pipe_ref_map
        ref = self.pipe_ref(id)
        if is_new:
            ls = self.type_instances.get(name, [])
            ls.append((ref, data))
            self.type_instances[name] = ls
        return ref

    def add_metric(self, module_id: Union[int, Reference], name: str, data: Dict[str, Any]):
        if isinstance(module_id, Reference):
            module_id = module_id.referenced
        if module_id not in self.pipe_ref_map:
            raise ValueError('Cannot handle non-registered module!!!')
        module_ref = self.pipe_ref(module_id)
        #metric_ref = Reference(self.next_free_mid)
        #self.next_free_mid += 1
        #self.metric_ref_map.append(metric_ref)
        self.module_metrics.append((module_ref, name, data)) #metric_ref, data))

    def add_metrics_bulk(self, module_id: Union[int, Reference], name: str, data_list: List[Dict[str, Any]]):
        if isinstance(module_id, Reference):
            module_id = module_id.referenced
        if module_id not in self.pipe_ref_map:
            raise ValueError('Cannot handle non-registered module!!!')
        module_ref = self.pipe_ref(module_id)
        #metric_refs = [Reference(id) for id in range(self.next_free_mid, self.next_free_mid+len(data_list))]
        #self.next_free_mid += len(data_list)
        #self.metric_ref_map.extend(metric_refs)
        self.module_metrics.extend([(module_ref, name, data) #metric_ref, data)
                                    for data in data_list])
                                    #for metric_ref, data in zip(metric_refs, data_list)])

    def build(self) -> PipeInfo:
        return PipeInfo(list(self.pipe_ref_map.values()),
                        self.pipe_successors,
                        # ignore pycharms type inspection here... This is correct
                        list(self.type_instances.items()),
                        self.artifact_ref_map,
                        self.module_metrics)


def recursive_metric_merge(res: Dict[str, Any], loaded: Dict[str, Any]):
    for k, v in loaded.items():
        if k in res:
            cur_res = res[k]
            if isinstance(cur_res, dict) and isinstance(v, dict):
                res[k] = recursive_metric_merge(cur_res, v)
            elif isinstance(cur_res, tuple):
                res[k] = cur_res + (v,)
            else:
                res[k] = (cur_res, v)
        else:
            res[k] = v
    return res

KEYS_TO_FIX = {'contingency_matrix', 'contingency_unique_labels', 'contingency_unique_predictions',
               'center', 'num_samples', 'intra_dists', 'centroids'}
def load_metrics(seq_computation: SerializedObject, loaded_files: Dict[str, Any]) -> Dict[str, Any]:
    def fix_keys(key: Any) -> Any:
        if isinstance(key, str) and key in KEYS_TO_FIX:
            return 'key:'+key
        elif isinstance(key, str) and key.isdigit():
            return int(key)
        else:
            return key

    def to_numpy(value: Any) -> Any:
        if isinstance(value, dict):
            return {fix_keys(k): to_numpy(v) for k, v in value.items()}
        elif isinstance(value, list) or isinstance(value, tuple):
            return np.array(value)
        else:
            return value

    def fix_contingencies(value: Any) -> Any:
        if isinstance(value, dict):
            if KEY_CONTINGENCY_MATRIX in value and KEY_CONTINGENCY_UNIQUE_LABELS in value and \
               KEY_CONTINGENCY_UNIQUE_PREDICTIONS in value:
                pred_ar = value[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
                if pred_ar[pred_ar >= 0].shape[0] < 2:
                    cm = value[KEY_CONTINGENCY_MATRIX]
                    if not np.any(pred_ar == 0):
                        pred_ar = np.concatenate((pred_ar, [0]))
                        cm = np.concatenate((cm, np.zeros((cm.shape[0], 1), dtype=np.int32)), axis=1)
                    if not np.any(pred_ar == 1):
                        pred_ar = np.concatenate((pred_ar, [1]))
                        cm = np.concatenate((cm, np.zeros((cm.shape[0], 1), dtype=np.int32)), axis=1)
                    value[KEY_CONTINGENCY_MATRIX] = cm
                    value[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = pred_ar
                return value
            return {k: fix_contingencies(v) for k, v in value.items()}
        else:
            return value

    if issubclass(seq_computation.actual_type, SequenceMetricComputation) or isinstance(seq_computation.actual_type,
                                                                                        DistributedMetricComputation):
        res = {}
        for sub_module in seq_computation['sub_modules']:
            loaded = load_metrics(sub_module, loaded_files)
            recursive_metric_merge(res, loaded)
        return res
    elif issubclass(seq_computation.actual_type, ContingencyMatrixComputation) or issubclass(
            seq_computation.actual_type, ClusterSpatialDistributionComputation):
        f_name = seq_computation['name'] + '.pkl'
        as_numpy = to_numpy(loaded_files[f_name])
        if issubclass(seq_computation.actual_type, ContingencyMatrixComputation):
            as_numpy = fix_contingencies(as_numpy)
        return as_numpy
    else:
        raise RuntimeError(f'Not manageable computation type {seq_computation.type_name}!')


def remove_non_scalar(metrics: Any) -> Optional[Any]:
    if isinstance(metrics, dict):
        res = {k: remove_non_scalar(v) for k, v in metrics.items() if not isinstance(k, str) or not k.startswith('key:')}
        res = {k: v for k, v in res.items() if v is not None}
        return res if res else None
    elif isinstance(metrics, list) or isinstance(metrics, np.ndarray):
        return None
    elif isinstance(metrics, tuple):
        res = list(filter(lambda v: v is not None, (remove_non_scalar(v) for v in metrics)))
        return tuple(res) if res else None
    else:
        return metrics

def do_metric_computation(step: int, kwargs: Dict[str, Any], metric_options: MetricOptions) -> Dict[str, Any]:
    def standard_contingency_computations() -> List[MetricComputation]:
        res = [
            PrecisionComputation('precision'),
            RecallComputation('recall'),
            FScoreComputation('f1_score', 1.0),
            AccuracyComputation('accuracy')
        ]
        if metric_options.is_non_total_calculation:
            res.extend([
                IOUComputation('iou1', 1),
                IOUComputation('iou0', 0),
            ])
        else:
            res.append(IOUComputation('iou0', 0))
        return res

    def pair_confusion_calculations() -> List[MetricComputation]:
        return [RandScore('rand_score'),
                AdjustedRandScore('adjusted_rand_score'),
                #FowlkesMallowsScore('fowlkes_mallows_score'),
                # PairPrecisionComputation('pair_precision'),
                # PairRecallComputation('pair_recall'),
                # PairFScore('pair_f1_score', 1.0),
                #PairIOUScore('iou_score')
        ]

    computations: List[MetricComputation] = []
    if KEY_CONTINGENCY_MATRIX in kwargs and KEY_CONTINGENCY_UNIQUE_LABELS in kwargs and KEY_CONTINGENCY_UNIQUE_PREDICTIONS in kwargs:
        cm, ul, up = get_contingency_predictions(kwargs)
        contingency_computations: List[MetricComputation] = [LabelDistribution('label_distribution')]
        if metric_options.is_non_total_calculation and metric_options.exclude_clouds:
            mask = ul != -1
            perform = np.any(mask)
            cm = cm[mask]
            ul = ul[mask]
            kwargs[KEY_CONTINGENCY_MATRIX] = cm
            kwargs[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
        else:
            perform = True

        if perform:
            if metric_options.allow_cluster_metrics:
                contingency_computations.extend([
                    MutualInfoScore('mutual_information_score'),
                    #AdjustedMutualInfoScore('adjusted_mutual_information_score'),
                    #NormalizedMutualInfoScore('normalized_mutual_information_score'),
                    #PredictionEntropy('pred_entropy'),
                    #LabelEntropy('label_entropy'),
                    Homogenity('homogenity'),
                    Completeness('completeness'),
                    VMeasure('v1_measure', 1.0) # <=> NMI with average normalisation
                ])
                # if metric_options.is_non_total_calculation and not metric_options.force_max_agree:
                #     contingency_computations.extend([
                #         ClusterFMeasure('cluster_f1_score0', 1.0, 0),
                #         ClusterFMeasure('cluster_f1_score1', 1.0, 1)
                #     ])
                # else:
                pc_comp = PairConfusionMatrix(pair_confusion_calculations()).append_to_name('pair_confusion')
                ncpc_comp = NormalizedClassSizePairConfusionMatrix([RandScore('rand_score')],
                                                                   # parallel only in total mode or when it is not parallel...
                                                                   (not metric_options.is_parallel_data_computation or
                                                                    not metric_options.is_non_total_calculation)) \
                    .append_to_name('normalized_class_size_pair_confusion')
                contingency_computations.extend([
                    # BCubedRecall('b3_recall'),
                    # BCubedPrecision('b3_precision'),
                    # BCubedF('b3f1_score', 1.0),
                    ClusterFMeasure('cluster_f1_score', 1.0, 1),
                    pc_comp,
                    ncpc_comp
                ])
                if ul.shape[0] >= up.shape[0] and not metric_options.force_max_agree:
                    if metric_options.include_lsas:
                        lsas = LinearSumAssignment(standard_contingency_computations()).append_to_name('linear_sum_assignment')
                        contingency_computations.append(lsas)
                    else:
                        contingency_computations.extend(standard_contingency_computations())
                else:
                    comp = MaximumAgreementComputation(standard_contingency_computations()).append_to_name('max_agree')
                    contingency_computations.append(comp)
            else:
                contingency_computations.extend(standard_contingency_computations())
            #else:
            #    print('Found Cloud only image, therefore no computation could be performed!')
            cc = SequenceMetricComputation(contingency_computations)
            if metric_options.exclude_clouds:
                cc = cc.append_to_name('no_clouds')
            if metric_options.is_non_total_calculation and metric_options.enforce_flood_no_flood_present:
                computations.append(ToSen1Floods11NormalizedContingency(cc))
            else:
                computations.append(cc)

    res = SequenceMetricComputation(computations).forward(step, True, **kwargs)
    return remove_non_scalar(res)

def execute_metric_computation_per_data(seq_computation: SerializedObject, loaded_files: Dict[str, Any], metric_config: MetricConfiguration) \
        -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    def compute_steps(id: int, step_dict: Dict[int, Dict[str, Any]], metric_options: MetricOptions) -> Tuple[int, Dict[int, Dict[str, Any]]]:
        return id, {step: do_metric_computation(step, kwargs, metric_options) for step, kwargs in step_dict.items()}
    loaded_metrics = load_metrics(seq_computation, loaded_files)
    if metric_config.max_num_unique_predictions is not None:
        lengths = [len(kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]) for _, step_dict in loaded_metrics.items() for step, kwargs in step_dict.items()]
        lengths = pd.Series(lengths)
        print('Found metrics with characteristics:')
        print(lengths.describe())
        if lengths.mean() >= metric_config.max_num_unique_predictions:
            print(f'Mean {lengths.mean()} is larger than or equal to the configured maximum for per-data computations'
                  f' of  {metric_config.max_num_unique_predictions}. Skipping computations!')
            return None
        else:
            print(f'Mean {lengths.mean()} is smaller than the configured maximum for per-data computations of '
                  f'{metric_config.max_num_unique_predictions}. Continuing with computation.')
    metrics_dict = metric_config.parallel_instance(job.delayed(compute_steps)(id, step_dict, metric_config.options)
                                                   for id, step_dict in loaded_metrics.items())
    metrics_dict = dict(metrics_dict)
    return metrics_dict, loaded_metrics

def compute_metrics_per_data(seq_computation: SerializedObject, pipe_info: BuildingPipeInfo,
                             loaded_files: Dict[str, Any], metric_config: MetricConfiguration,
                             module_ref: Reference, prefix='per_data.', int_data_ids: bool = True) -> Optional[Dict[str, Any]]:
    # the int_data_ids param is there only to circumvent a pipeline specification error... I accidentally set the
    # region and the per_data computations to output to the same contingency file :face_palm: (see simple_methods.py)
    t = execute_metric_computation_per_data(seq_computation, loaded_files, metric_config)
    if t is None:
        return None

    per_data_metrics, loaded_metrics = t
    # TODO generalize for stepwise metrics
    per_data_metrics = [(metric_type,
                         data_id,
                         (int(step) if len(step_dict) > 1 else None), clazz,
                            (v.item() if type(v).__module__ == np.__name__ else v)) # unbox numpy types...
                        for data_id, step_dict in per_data_metrics.items()
                        if int_data_ids ^ (type(data_id) is not int)
                        for step, value_dict in step_dict.items()
                        if value_dict is not None
                        for metric_type, value_item in value_dict.items()
                        for clazz, v in (value_item.items() if isinstance(value_item, dict) else [(None,value_item)])]
    column_names =['metric_type', COLUMN_METRIC_DATA_ID, COLUMN_METRIC_STEP_ID, COLUMN_METRIC_CLASS, COLUMN_METRIC_VALUE]
    apply_metrics_from_data(pipe_info, per_data_metrics, column_names, module_ref, prefix)
    return loaded_metrics


def extract_metrics_total_from_data(loaded_metrics: Dict[str, Any],
                                    metric_config: MetricConfiguration) -> \
        Optional[Tuple[List[Tuple[Any,...]], List[str]]]:
    contingencies = [{step: (kwargs[KEY_CONTINGENCY_MATRIX],
                                  kwargs[KEY_CONTINGENCY_UNIQUE_LABELS],
                                  kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS])
                     for step, kwargs in step_dict.items()}
                     for _, step_dict in loaded_metrics.items()]
    try:
        contingencies = {step: [d[step] for d in contingencies]
                         for step in set(itert.chain(*[d.keys() for d in contingencies]))}
    except KeyError as e:
        raise RuntimeError('Expected all steps to be present for all data entries. '
                           'However this does not seem to be the case!') from e
    step_wise_consistency_check = lambda cls, ref: any(map(lambda t: t[2].shape[0] != ref.shape[0] or
                                                                np.any(np.sort(t[2], kind='stable') != ref) or
                                                                np.any((t[1] > 1) | (t[1] < -1)),
                                                      cls))
    if any(map(lambda cls: step_wise_consistency_check(cls, np.sort(cls[0][2], kind='stable')), contingencies.values())):
        # There exists some contingency matrix with incompatible predictions or incompatible labels...
        print('Skipping total computation due to incompatibility of Predictions!')
        return None

    options = dataclasses.replace(metric_config.options)
    per_step_metrics = [(metric_type,
                         (int(step) if len(contingencies) > 1 else None),
                         clazz,
                         (v.item() if type(v).__module__ == np.__name__ else v)) # unbox numpy types...
                        for step, cls in contingencies.items()
                        for metric_type, value_item in
                        do_metric_computation(step, accumulate_contingency(cls, options.exclude_clouds), options).items()
                        for clazz, v in (value_item.items() if isinstance(value_item, dict) else [(None, value_item)])]
    column_names = ['metric_type', COLUMN_METRIC_STEP_ID, COLUMN_METRIC_CLASS, COLUMN_METRIC_VALUE]
    return per_step_metrics, column_names

def apply_metrics_from_data(pipe_info: BuildingPipeInfo,
                            per_step_metrics: List[Tuple[Any,...]],
                            column_names: List[str],
                            module_ref: Reference,
                            metric_prefix: str):
    rem_cn = column_names[1:]
    for metric_type in {t[0] for t in per_step_metrics}:
        relevant_data = [{n:v for n,v in zip(rem_cn, t[1:]) if v is not None} for t in per_step_metrics if t[0] == metric_type]
        m_name = metric_prefix + metric_type
        pipe_info.add_metrics_bulk(module_ref, m_name, relevant_data)

def compute_batchwise_metrics_from_data(loaded_metrics: Dict[str, Any], metric_config: MetricConfiguration) \
        -> Optional[Tuple[List[Tuple[Any,...]], List[str]]]:
    if metric_config.batch_sizes is None:
        return None
    contingencies = [{step: (kwargs[KEY_CONTINGENCY_MATRIX],
                             kwargs[KEY_CONTINGENCY_UNIQUE_LABELS],
                             kwargs[KEY_CONTINGENCY_UNIQUE_PREDICTIONS])
                      for step, kwargs in step_dict.items()}
                     for _, step_dict in loaded_metrics.items()]
    try:
        contingencies = {step: [d[step] for d in contingencies]
                         for step in set(itert.chain(*[d.keys() for d in contingencies]))}
    except KeyError as e:
        raise RuntimeError('Expected all steps to be present for all data entries. '
                           'However this does not seem to be the case!') from e
    step_wise_consistency_check = lambda cls, ref: any(map(lambda t: t[2].shape[0] != ref.shape[0] or
                                                                     np.any(np.sort(t[2], kind='stable') != ref) or
                                                                     np.any((t[1] > 1) | (t[1] < -1)),
                                                           cls))
    if any(map(lambda cls: step_wise_consistency_check(cls, np.sort(cls[0][2], kind='stable')),
               contingencies.values())):
        # There exists some contingency matrix with incompatible predictions or incompatible labels...
        print('Skipping total computation due to incompatibility of Predictions!')
        return None

    results = [(batch_size,
                   [(step, batchwise_metrics_df(cms, metric_config.batch_iter, batch_size, metric_config.random_state))
                    for step, cms in contingencies.items()])
               for batch_size in metric_config.batch_sizes]
    per_step_metrics = [((f'{metric}_{batch_size}', step)+tuple(m_df.xs(key=metric, axis=1, level=BM_CLEVEL_METRIC).squeeze(axis=0)))
                        for batch_size, step_list in results
                        for step, m_df in step_list
                        for metric in BM_METRIC_LEVEL]
    column_names = ['metric_type', COLUMN_METRIC_STEP_ID]
    column_names.extend([bl+'-'+il for bl, il in itert.product(BM_BATCH_LEVEL, BM_ITERATION_LEVEL)])
    return per_step_metrics, column_names


# TODO depict algorithm in Thesis
def process_pipeline(pipe_obj: SerializedObject, pipe_info: BuildingPipeInfo, loaded_files: Dict[str, Any],
                     metric_config: MetricConfiguration) \
        -> Tuple[List[int], List[int]]:
    if issubclass(pipe_obj.actual_type, MultiMetaModule) or issubclass(pipe_obj.actual_type, MetaModule):
        # print('MultiMetaModule!')
        if issubclass(pipe_obj.actual_type, MultiSequenceModule) or issubclass(pipe_obj.actual_type, SequenceModule):
            sub_modules = pipe_obj['sub_modules']
            if sub_modules:
                start, seq_end = process_pipeline(sub_modules[0], pipe_info, loaded_files, metric_config)
                for module in sub_modules[1:]:
                    assert isinstance(module, SerializedObject)
                    seq_ids, next_end = process_pipeline(module, pipe_info, loaded_files, metric_config)
                    pipe_info.add_relations(itertools.product(seq_end, seq_ids))
                    seq_end = next_end
                return start, seq_end
            else:
                print('Found empty (Multi) SequenceModule?!?')
                return [], []
        elif issubclass(pipe_obj.actual_type, MultiDistributorModule) or issubclass(pipe_obj.actual_type,
                                                                                    DistributorModule):
            start, seq_end = [], []
            distributed_modules = pipe_obj['distributed_modules']
            for module in distributed_modules:
                assert isinstance(module, SerializedObject)
                new_start, new_seq_end = process_pipeline(module, pipe_info, loaded_files, metric_config)
                start.extend(new_start)
                seq_end.extend(new_seq_end)
            return start, seq_end
        elif issubclass(pipe_obj.actual_type, PipelineAdaptorModule):
            pipe_module = pipe_obj['pipe_module']
            return process_pipeline(pipe_module, pipe_info, loaded_files, metric_config)
        elif issubclass(pipe_obj.actual_type, SimpleCachedPipeline):
            sub_module = pipe_obj['sub_module']
            return process_pipeline(sub_module, pipe_info, loaded_files, metric_config)
        else:
            raise RuntimeError
    else:
        value = {}

        def non_null_artifact_add(obj: SerializedObject, key: str, append_to: Dict[str, Any]):
            file = obj[key]
            if file is not None:
                append_to[key] = pipe_info.artifact_ref(file)

        if issubclass(pipe_obj.actual_type, Sen1Floods11DataDingsDatasource):
            value = {'in_memory': pipe_obj['in_memory'],
                     'meta_type': pipe_obj['meta_type'],
                     'split': pipe_obj['split']}
        elif issubclass(pipe_obj.actual_type, WhitelistModule):
            value = {'whitelisted': ';'.join(pipe_obj['whitelisted'])}
        elif issubclass(pipe_obj.actual_type, BlacklistModule):
            value = {'blacklisted': ';'.join(pipe_obj['blacklisted'])}
        elif issubclass(pipe_obj.actual_type, SARFilterModule):
            method = pipe_obj['method']
            params = pipe_obj['params']
            if 'win' in params:
                params['win_x'] = params['win'][0]
                params['win_y'] = params['win'][1]
                del params['win']
            method_id = pipe_info.add_module(None, method, params)
            value = {'method': method,
                     'method_id': method_id}
        elif issubclass(pipe_obj.actual_type, UnsupervisedSklearnAdaptorModule) \
                or issubclass(pipe_obj.actual_type, SupervisedSklearnAdaptorModule) \
                or issubclass(pipe_obj.actual_type, GeneralisedSklearnAdaptorModule):
            transformer: SerializedObject = pipe_obj['transformer']
            transformer_ref = pipe_info.add_module(transformer.id, transformer['class_name'], transformer['params'])
            value = {COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_NAME: transformer['class_name'],
                     COLUMN_SKLEARN_ADAPTOR_TRANSFORMER_ID: transformer_ref,
                     COLUMN_SKLEARN_ADAPTOR_DO_FIT: pipe_obj['do_fit'],
                     COLUMN_SKLEARN_ADAPTOR_DO_PREDICT: pipe_obj['do_predict']}
            non_null_artifact_add(transformer, 'save_file', transformer['params'])
            non_null_artifact_add(transformer, 'init_file', transformer['params'])
        elif issubclass(pipe_obj.actual_type, SkorchAdaptorModule):
            source_ref = pipe_info.add_module(pipe_obj.id, pipe_obj.type_name, value)
            transformer: SerializedObject = pipe_obj['transformer']
            callbacks: Dict[str, SerializedObject] = transformer['callbacks']
            for key, callback in callbacks.items():
                if issubclass(callback.actual_type, AccumulatingMetricScoring):
                    # TODO: save to extra file - these metrics otherwise take up way too much memory
                    pass
                    # is_train = callback['on_train']
                    # repeat_factor = callback['repeat_factor']
                    # ref = pipe_info.add_module(pipe_obj.id, pipe_obj.type_name,
                    #                            {'source': source_ref, 'is_train': is_train, 'repeat_factor': repeat_factor})
                    # comp: SerializedObject = callback['contingency_comp']
                    # loaded_metrics = compute_metrics_per_data(comp, pipe_info, loaded_files,
                    #                                           metric_config, ref)
                    # if metric_config.allow_total_from_per_data and loaded_metrics is not None:
                    #     t = extract_metrics_total_from_data(loaded_metrics, metric_config)
                    #     if t is not None:
                    #         apply_metrics_from_data(pipe_info, t[0], t[1], ref, 'total.')

        elif issubclass(pipe_obj.actual_type, MetricsModule):
            ref = pipe_info.add_module(pipe_obj.id, pipe_obj.type_name, value)
            if pipe_obj['per_data_computation'] is not None:
                loaded_metrics = compute_metrics_per_data(pipe_obj['per_data_computation'], pipe_info, loaded_files, metric_config, ref)
                value['per_data'] = True
            else:
                loaded_metrics = None
                value['per_data'] = False
            if loaded_metrics is not None:
                t = compute_batchwise_metrics_from_data(loaded_metrics, metric_config)
                if t is not None:
                    apply_metrics_from_data(pipe_info, t[0], t[1], ref, 'batch.')
                    value['batchwise'] = True
                else:
                    value['batchwise'] = False
            if pipe_obj['per_region_computation'] is not None:
                #print('WARNING: per region computation is not processed yet!!!')
                _ = compute_metrics_per_data(pipe_obj['per_region_computation'],
                                                            pipe_info, loaded_files, metric_config, ref,
                                                            prefix='per_region.', int_data_ids=False)
                del _
                value['per_region'] = True
            else:
                value['per_region'] = False
            if pipe_obj['total_computation'] is not None:
                print('WARNING: total computation is not processed yet!!!')
                value['total'] = True
            else:
                if metric_config.allow_total_from_per_data and loaded_metrics is not None:
                    t = extract_metrics_total_from_data(loaded_metrics, metric_config)
                    if t is not None:
                        apply_metrics_from_data(pipe_info, t[0], t[1], ref, 'total.')
                        total_calculated = True
                    else:
                        total_calculated = False
                else:
                    total_calculated = False
                value['total'] = total_calculated
            del loaded_metrics
            value['prediction_criterion'] = pipe_obj['prediction_criterion']['name']
            value['label_criterion'] = pipe_obj['label_criterion']['name']
        # elif issubclass(pipe_obj.actual_type, PerImageClusteringModule):
        #     method = pipe_obj['method']
        #     method_id = pipe_info.add_module(None, method, pipe_obj['params'])
        #     value = {'method': method,
        #              'method_id': method_id}
        # elif issubclass(pipe_obj.actual_type, PerImageClusteringRefinementModule):
        #     value = pipe_obj.serialized_content.copy()
        #     del value['intensity_criterion']
        #     del value['label_criterion']
        #     del value['cache']
        #     del value['result_name']
        elif issubclass(pipe_obj.actual_type, PerImageZonalStatsExtractorModule) \
                or issubclass(pipe_obj.actual_type, PerDataPointStatsExtractorModule):
            value = {'stats_of_interest': ';'.join(pipe_obj['stats_of_interest'])}
        elif issubclass(pipe_obj.actual_type, EOImageExtractorModule):
            value = pipe_obj.serialized_content
        elif issubclass(pipe_obj.actual_type, (StandardizationModule, RangeClippingNormalizationModule,
                                               ShapelessInMemoryModule, InMemoryModule, TerminationModule,
                                               AssemblerModule, MaskModule, NoCachePipeline, RemoveFromSummaryModule,
                                               RetainInSummaryModule, ProbabilityToValue, MaskCombine)):
            pass
        else:
            raise RuntimeError(f'Not manageable type {pipe_obj.type_name}!')
        pipe_info.add_module(pipe_obj.id, pipe_obj.type_name, value)
        return [pipe_obj.id], [pipe_obj.id]


def pipeline_to_tables(pipeline: Dict, folder_name: str, loaded_files: Dict[str, Any],
                       artifact_files: List[str], metric_config: MetricConfiguration) -> PipeInfo:
    var_store = VariableStore(pipeline)
    root = var_store.resolve()
    # print(root.id, root.type_name, root.serialized_content)
    pipe_info = BuildingPipeInfo(var_store.max_id() + 1)
    pipe_info.add_artifacts(loaded_files.keys())
    process_pipeline(root, pipe_info, loaded_files, metric_config)
    pipe_info.add_artifact_prefix(folder_name + '/')
    pipe_info.add_artifacts(artifact_files)
    return pipe_info.build()


@dataclass
class RunInfo:
    folder_name: str
    config: Dict[str, Any]
    pipeline: PipeInfo
    artifact_files: List[str]


@dataclass
class ExperimentInfo:
    trials: pd.DataFrame
    trial_distributions: Dict[int, Dict[str, Dict[str, Any]]]
    runs: List[RunInfo]
    experiment_artifacts: List[str]
    name: str
    id: int
    seed: Optional[int] = None


def process_output_folder(tar: tarfile.TarFile, to_process: tarfile.TarInfo,
                          ex_info: ExperimentInfo, metric_config: MetricConfiguration,
                          folder_name: Optional[str] = None) -> tarfile.TarInfo:
    folder_name = to_process.name if folder_name is None else folder_name
    artifact_files = []
    loaded_files: Dict[str, Any] = {}
    while to_process is not None and to_process.name.startswith(folder_name):
        if to_process.isdir():
            # Because subfolders should be skipped it is necessary to perform the tar.next() as the last thing in the loop
            # This results in this special case for directories where the folder name needs to be skipped...
            if to_process.name != folder_name:
                artifact_files.append(to_process.name)
                to_process = skip_with_prefix(tar, to_process, [to_process.name])
            else:
                to_process = tar.next()
        elif to_process.isfile():
            artifact_files.append(to_process.name)
            actual_name = right_slash_split(to_process.name)
            _, ending = actual_name.rsplit('.', maxsplit=1)
            try:
                if ending == 'json':
                    loaded_files[actual_name] = json.load(tar.extractfile(to_process))
                elif ending == 'pkl':
                    loaded_files[actual_name] = pkl.load(tar.extractfile(to_process))
                elif ending == 'txt':
                    pass
                elif ending == 'pt':
                    pass # skorch saves don't need to be loaded...
                else:
                    print(f'WARNING: Cannot load file "{actual_name}" with unknown ending "{ending}".')
            except:
                print(f'WARNING: Cannot load file "{actual_name}" due to an error.')
                print(traceback.format_exc())
            to_process = tar.next()
        else:
            raise RuntimeError(f'Object {to_process.name} in output folder {folder_name} is neither a file nor a '
                               f'directory. This should not be the case!')
    if 'pipeline.json' not in loaded_files:
        print(f'No pipeline provided for output folder {folder_name}! Skipping!')
        return to_process
    elif 'config.json' not in loaded_files:
        print(f'No config provided for output folder {folder_name}! Skipping!')
        return to_process
    elif 'run.json' not in loaded_files:
        print(f'No run info provided for output folder {folder_name}! Skipping!')
        return to_process
    pipeline = loaded_files['pipeline.json']
    config = loaded_files['config.json']
    run_info = loaded_files['run.json']
    if run_info['status'] != 'COMPLETED':
        print(f'Skipping output folder {folder_name} with run status "{run_info["status"]}" which is definetly not '
              f'"COMPLETED".')
    else:
        pipe_info = pipeline_to_tables(pipeline, folder_name, loaded_files, artifact_files, metric_config)
        del loaded_files
        ex_info.runs.append(RunInfo(folder_name, config, pipe_info, artifact_files))
    return to_process


def process_trials(tar: tarfile.TarFile, to_process: tarfile.TarInfo, ex_info: ExperimentInfo) -> None:
    if len(ex_info.trials) >= 1:
        raise RuntimeError('Found duplicate trials file - reading trials would overwrite already read trials!')
    trials = pd.read_csv(tar.extractfile(to_process), index_col=0)
    trials['datetime_start'] = pd.to_datetime(trials['datetime_start'].str.replace(' ', 'T', regex=False))
    trials['datetime_complete'] = pd.to_datetime(trials['datetime_complete'].str.replace(' ', 'T', regex=False))

    ex_info.trials = trials
    return None


def process_trial_distributions(tar: tarfile.TarFile, to_process: tarfile.TarInfo, ex_info: ExperimentInfo) -> None:
    if len(ex_info.trial_distributions) > 0:
        raise RuntimeError(
            'Found duplicate trial-distributions file - reading distributions would overwrite already read distributions!')
    ex_info.trial_distributions = json.load(tar.extractfile(to_process))
    return None


def process_entry(tar: tarfile.TarFile, to_process: tarfile.TarInfo, ex_info: ExperimentInfo, skip_folders: List[str],
                  skip_files: List[str], metric_info: MetricConfiguration) -> Optional[tarfile.TarInfo]:
    ex_info.experiment_artifacts.append(to_process.name)
    if to_process.isdir():
        if to_process.name in skip_folders:
            return skip_with_prefix(tar, to_process, skip_folders)
        # Experiment folders only consist of numbers
        elif right_slash_split(to_process.name).isdigit():
            return process_output_folder(tar, to_process, ex_info, metric_info)
        else:
            raise RuntimeError(f'Cannot handle directory {to_process.name}!')
    elif to_process.isfile():
        if to_process.name in skip_files:
            return skip_with_prefix(tar, to_process, skip_files)
        # The trials file is always named "trials.csv"
        elif right_slash_split(to_process.name) == TRIALS_DF_FILE:
            return process_trials(tar, to_process, ex_info)
        elif right_slash_split(to_process.name) == TRIALS_DISTRIBUTIONS:
            return process_trial_distributions(tar, to_process, ex_info)
        elif len(to_process.name.rsplit('/')) >= 2 and to_process.name.rsplit('/')[-2].isdigit():
            print('WARNING: DETECTED FOLDER NOT BEING OPENED PREVIOUSLY. THIS IS INTERPRETED AS AN IMPLICIT OPEN!')
            process_output_folder(tar, to_process, ex_info, metric_info, folder_name='/'.join(to_process.name.rsplit('/')[:-1]))
        else:
            print(f'Cannot handle file {to_process.name}!')
    else:
        raise ValueError('Found non-file and non-directory entry in experiment tar. This should not be the case!')
    return None


def insert_trails_into_database(ex_info: ExperimentInfo, database: PandasDatabase):
    print(f'Adding trials to database.')
    t = time.time()
    cur_trials = database.create_trial_table_if_absent()
    ex_trials: pd.DataFrame = ex_info.trials
    ex_trials = ex_trials.loc[ex_trials['state'] == 'COMPLETE']
    ex_trials = ex_trials.drop(
        columns=[key for key in ex_trials.columns.values if key.startswith('values_')] + ['state'] +
                (['value'] if 'value' in ex_trials.columns else []),
        inplace=False)
    ex_trials.loc[:, COLUMN_TRIAL_EX_ID] = ex_info.id
    ex_trials = ex_trials.rename(columns={'number': COLUMN_TRIAL_TRIAL_NUM,
                                          'datetime_start': COLUMN_TRIAL_START_TIME,
                                          'datetime_complete': COLUMN_TRIAL_END_TIME})
    ex_trials.set_index(keys=[COLUMN_TRIAL_EX_ID, COLUMN_TRIAL_TRIAL_NUM], drop=True, inplace=True)
    cur_trials = cur_trials.append(ex_trials)
    cur_trials = cur_trials.loc[
        cur_trials[COLUMN_TRIAL_START_TIME].notnull() & cur_trials[COLUMN_TRIAL_END_TIME].notnull()]
    database[TABLE_TRIAL] = cur_trials
    t = time.time() - t
    print(
        f'Adding trials took {t:.3f}s and added {len(ex_trials)} rows - {(t / len(ex_trials) if len(ex_trials) > 0 else -0.0):.3f}s on average.')


def insert_ex_artifacts(ex_info: ExperimentInfo, database: PandasDatabase):
    print(f'Adding experiment artifacts to database.')
    t = time.time()
    cur_artifacts = database.create_exartifact_table_if_absent()
    ex_artifacts = pd.DataFrame({COLUMN_EX_ARTIFACT_FNAME: ex_info.experiment_artifacts})
    ex_artifacts.loc[:, COLUMN_EX_ARTIFACT_EX_ID] = ex_info.id
    database[TABLE_EX_ARTIFACT] = cur_artifacts.append(ex_artifacts)
    t = time.time() - t
    print(f'Adding experiment artifacts took {t:.3f}s and added {len(ex_artifacts)} rows - '
          f'{(t / len(ex_artifacts) if len(ex_artifacts) > 0 else -0.0):.3f}s on average.')


def is_super_dict(d1: dict, d2: dict) -> bool:
    return not any(map(lambda t: t[0] not in d1 or d1[t[0]] != t[1], d2.items()))


def do_matching(ex_info: ExperimentInfo, database: PandasDatabase, raise_duplicates: bool = False):
    def single_matching_run(runs: List[RunInfo], dist: Dict[str, Any], num: int, duplicates_returned: Set[int]) -> \
    Optional[RunInfo]:
        """
        Iterates the runs completely to verify that only exactly one run matches the distribution values. A descriptive
        exception is thrown otherwise.
        """
        found: Optional[Tuple[RunInfo, int]] = None
        duplicate: bool = False
        for i, run in enumerate(runs):
            if is_super_dict(run.config, dist):
                if found is not None:
                    r, j = found
                    desc_str = f'Found more than one matching run for trial num {num}! Candidates are {j} and ' \
                               f'{i} with distribution values {dist}.'
                    found_id = found[0].folder_name.rsplit('/', maxsplit=1)[-1]
                    new_id = run.folder_name.rsplit('/', maxsplit=1)[-1]
                    prefer_old = True
                    if not found_id.isdigit():
                        print('ERROR: Cannot determine which duplicate entry is the newer one as the old entry '
                              'does not have a numeric folder name!', file=sys.stderr)
                        prefer_old = False
                    if not new_id.isdigit():
                        print('ERROR: Cannot determine which duplicate entry is the newer one as the new entry '
                              'does not have a numeric folder name!', file=sys.stderr)
                    if raise_duplicates:
                        raise RuntimeError(desc_str)
                    elif found_id.isdigit() and new_id.isdigit() and int(found_id) > int(new_id):
                        print('WARNING:', desc_str, 'Replacing previous match as per raise_duplicates = False!',
                              file=sys.stderr)
                        found = run, i
                    elif prefer_old:
                        print('WARNING:', desc_str, 'Skipping newest occurrence as per raise_duplicates = False!',
                              file=sys.stderr)
                    else:
                        print('WARNING:', desc_str, 'Replacing previous match as it\'s folder name could not be parse '
                                                    'as per raise_duplicates = False!',
                              file=sys.stderr)
                        found = run, i
                    duplicate = True
                else:
                    found = run, i
        if found is None:
            print(f'WARNING: No matching run for trial num {num} found! Distribution values are {dist}.',
                  file=sys.stderr)
            return None
        if duplicate:
            if found[1] in duplicates_returned:
                print('Skipping later occurrence of duplicate.')
                return None
            else:
                duplicates_returned.add(found[1])
        return found[0]

    print(f'Matching trials and runs.')
    t = time.time()
    trials = database.create_trial_table_if_absent()
    trials_by_num = trials.xs(ex_info.id, level=0)
    cleaned_dists: Dict[int, Dict[str, Any]] = {int(num): {key: desc['value'] for key, desc in dist_config.items()}
                                                for num, dist_config in ex_info.trial_distributions.items()}
    duplicates_returned = set()
    run_by_trial_id = [(num, single_matching_run(ex_info.runs, dist, num, duplicates_returned))
                       for num, dist in sorted(cleaned_dists.items(), key=lambda t: t[0])
                       if num in trials_by_num.index]
    run_by_trial_id = {num: run for num, run in run_by_trial_id if run is not None}
    print(f'Matching completed in {time.time() - t:.3f}s.')
    return run_by_trial_id


def match_runs_and_trials(ex_info: ExperimentInfo, database: PandasDatabase, raise_duplicates: bool = False):
    run_by_trial_id = do_matching(ex_info, database, raise_duplicates)
    ex_id, ex_name = ex_info.id, ex_info.name
    del ex_info
    print(f'Adding trial artifacts to database.')
    t = time.time()
    cur_artifacts = database.create_artifact_table_if_absent()
    artifacts = pd.DataFrame(data=[(ex_id, int(id), artifact)
                                   for id, run in run_by_trial_id.items()
                                   for artifact in run.artifact_files],
                             columns=[COLUMN_ARTIFACT_EX_ID, COLUMN_TRIAL_TRIAL_NUM, COLUMN_ARTIFACT_FNAME])
    cur_artifacts = cur_artifacts.append(artifacts, ignore_index=True)
    artifact_id_values = cur_artifacts.index.to_numpy()[len(cur_artifacts) - len(artifacts):]
    artifact_names = pd.Index((cur_artifacts.loc[artifact_id_values])[COLUMN_ARTIFACT_FNAME])

    # to get the correct id's
    artifact_ids = {name: artifact_id_values[i] for i, name in enumerate(artifact_names.values)}
    database[TABLE_ARTIFACT] = cur_artifacts
    del database[TABLE_ARTIFACT]
    t = time.time() - t
    print(f'Adding trial artifacts took {t:.3f}s and added {len(artifacts)} rows - '
          f'{(t / len(artifacts) if len(artifacts) > 0 else -0.0):.3f}s on average.')
    print(f'Saving changes so far, in order to free memory.')
    database.save()
    print(f'Save completed.')
    print(f'Aligning Pipeline-Modules.')
    t = time.time()
    modules = database.create_pipeline_table_if_absent()
    next_pipe_id = modules.index.max()
    del database[TABLE_PIPELINE]
    #module_metric_rel = database.create_metric_computed_by_table_if_absent()
    #next_metric_id = module_metric_rel[COLUMN_MCB_METRIC_ID].max()
    #del database[TABLE_METRICS_COMPUTED_BY]
    next_config_id = 0 if ex_name is not database else database[ex_name].max()
    def fix_na(val: int):
        if pd.isna(val):
            return 0
        else:
            return val + 1
    next_pipe_id = fix_na(next_pipe_id)
    #next_metric_id = fix_na(next_metric_id)
    next_config_id = fix_na(next_config_id)

    pipe_tables: Dict[str, Tuple[List[pd.DataFrame], bool]] = {}

    def add_to_tables(t_name: str, df: pd.DataFrame, do_ignore: Optional[bool] = None):
        ls, ignore_index = pipe_tables.get(t_name, ([], False))
        ls.append(df)
        if do_ignore is not None:
            ignore_index = do_ignore
        pipe_tables[t_name] = ls, ignore_index

    def expand_dicts(d: Any) -> Any:
        if isinstance(d, dict):
            expanded = [([(k + '_' + k1, v) for k1, v in expand_dicts(v).items()] if isinstance(v, dict) else [(k, expand_dicts(v))])
                        for k, v in d.items()]
            return {k: v for ls in expanded for k, v in ls}
        elif isinstance(d, Reference):
            return d.referenced
        else:
            return d

    for num, run in run_by_trial_id.items():
        print(f'Aligning trial {num} with matched folder name {run.folder_name}')
        run: RunInfo = run  # help type inspection...
        pipe_info: PipeInfo = run.pipeline
        for artifact, artifact_ref in pipe_info.artifact_ref_map.items():
            artifact_ref.referenced = artifact_ids[artifact]
        for pipe_reference in pipe_info.pipe_references:
            pipe_reference.referenced = next_pipe_id
            next_pipe_id += 1
        #for metric_reference in pipe_info.metric_references():
        #    metric_reference.referenced = next_metric_id
        #    next_metric_id += 1
        print(f'Aligning trial {num} completed. Adding modules.')
        collected_modules: List[Tuple[Reference, str, bool, bool]] = []
        for t_name, entry in pipe_info.type_instances():
            add_to_tables(t_name,
                          pd.DataFrame.from_records(data=[expand_dicts(d) for _, d, _, _ in entry],
                                                    index=[r.referenced for r, _, _, _ in entry]))
            collected_modules.extend([(ref, t_name, r, f) for ref, _, r, f in entry])
        for t_name, entry in pipe_info.metric_instances():
            add_to_tables(t_name,
                          pd.DataFrame.from_records(data=[expand_dicts(dict(d, module=mod)) for mod, d in entry]),
                          do_ignore=True)
        module_index = pd.DataFrame.from_records(data=[{COLUMN_PIPELINE_EX_ID: ex_id, COLUMN_PIPELINE_T_NUM: num,
                                                        COLUMN_PIPELINE_TYPE: name, COLUMN_PIPELINE_ROOT: r,
                                                        COLUMN_PIPELINE_FINAL: f}
                                                       for _, name, r, f in collected_modules],
                                                 index=[r.referenced for r, _, _, _ in collected_modules])
        module_relations = pd.DataFrame.from_records(data=[{COLUMN_PIPELINE_PREDECESSOR: p.referenced,
                                                            COLUMN_PIPELINE_SUCCESSOR: s.referenced}
                                                           for p, s in pipe_info.pipe_successors])
        # metric_computed_by = pd.DataFrame.from_records(data=[{COLUMN_MCB_MODULE_ID: mod_id.referenced,
        #                                                       COLUMN_MCB_METRIC_ID: met_id.referenced}
        #                                                      for mod_id, met_id in pipe_info.metrics_relation()])
        if 'z_feature_space' in run.config and 'feature_space' not in run.config:
            run.config['feature_space'] = run.config['z_feature_space']
            del run.config['z_feature_space']
        if COLUMN_CONFIG_TRIAL_NUM in run.config:
            print(f'{COLUMN_CONFIG_TRIAL_NUM} is in config {run.config}! Renaming.')
            run.config[COLUMN_CONFIG_TRIAL_NUM+'_1'] = run.config[COLUMN_CONFIG_TRIAL_NUM]
            del run.config[COLUMN_CONFIG_TRIAL_NUM]
        if COLUMN_CONFIG_EX_ID in run.config:
            print(f'{COLUMN_CONFIG_EX_ID} is in config {run.config}! Renaming.')
            run.config[COLUMN_CONFIG_EX_ID+'_1'] = run.config[COLUMN_CONFIG_EX_ID]
            del run.config[COLUMN_CONFIG_EX_ID]
        run.config.update({COLUMN_CONFIG_EX_ID:ex_id, COLUMN_CONFIG_TRIAL_NUM:num})
        config = pd.DataFrame.from_records(data=[expand_dicts(run.config)],
                                           index=[next_config_id])
        next_config_id+=1
        add_to_tables(TABLE_PIPELINE, module_index)
        add_to_tables(TABLE_PIPELINE_SUCCESSOR, module_relations, True)

        #add_to_tables(TABLE_METRICS_COMPUTED_BY, metric_computed_by)
        add_to_tables(ex_name, config)
        print(f'Trial {num} added to the following modules: {set(name for _,name, _, _ in collected_modules)}')
    t = time.time() - t
    print(f'Aligning Pipeline modules took {t:.3f}s - {t / len(run_by_trial_id):.3f}s on average.')
    del run_by_trial_id
    print(f'Merging and adding pipeline elements to database.')
    t = time.time()
    for t_name, (frame_list, ignore_index) in pipe_tables.items():
        print(f'Merging frames with name {t_name}.')
        t1 = time.time()
        cur_frame: Optional[pd.DataFrame] = database.get(t_name, None)
        if cur_frame is not None:
            frame_list = [cur_frame] + frame_list
        cat = pd.concat(frame_list, ignore_index=ignore_index)
        database[t_name] = cat
        t1 = time.time() - t1
        print(f'Merge completed in {t1:.3f}s - average {t1 / len(frame_list)}s. '
              f'Saving database with frame of name {t_name}.')
        t1 = time.time()
        # always save, to avoid caching too much data in memory
        database.save()
        t1 = time.time() - t1
        print(f'Save completed in {t1:.3f}s.')
    t = time.time() - t
    print(f'Adding pipeline elements took {t:.3f}s.')


def parse_tar(tar_file: str, database: PandasDatabase, metric_config: MetricConfiguration) -> ExperimentInfo:
    ex_info: Optional[ExperimentInfo] = None
    def do_process_experiment_head(tar, next_entry):
        nonlocal ex_info
        ex_id, name, seed = process_experiment_head(tar, next_entry, database)
        ex_info = ExperimentInfo(pd.DataFrame(), {}, [], [], name, ex_id, seed)

    def do_process_entry(tar, next_entry, skip_folders, skip_files):
        nonlocal ex_info
        assert ex_info is not None
        return process_entry(tar, next_entry, ex_info, skip_folders, skip_files, metric_config)

    utils.parse_tar_generic(tar_file, do_process_experiment_head, do_process_entry)
    assert ex_info is not None
    return ex_info


def process_tar(tar_file: str, database: PandasDatabase, metric_config: MetricConfiguration,
                raise_duplicates: bool = False) -> PandasDatabase:
    database.save()
    print(f'Processing tar {tar_file}.')
    t = time.time()
    # Do this whole thing as a transaction. if it fails, nothing is going to be changed! (unless shutil fails... which it hopefully doesn't)
    with tmp.TemporaryDirectory(suffix='_tmp_database') as dir_name:
        shutil.copytree(database.database_folder, dir_name, dirs_exist_ok=True)
        tmp_database = PandasDatabase(dir_name)
        ex_info = parse_tar(tar_file, tmp_database, metric_config)
        insert_trails_into_database(ex_info, tmp_database)
        insert_ex_artifacts(ex_info, tmp_database)
        match_runs_and_trials(ex_info, tmp_database, raise_duplicates=raise_duplicates)
        utils.shutil_overwrite_dir(tmp_database.database_folder, database.database_folder)
    print(f'Finished processing tar {tar_file} after {time.time() - t:.3f}s.')
    return database


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database_folder', type=str)
    parser.add_argument('--result_files', type=str, nargs='+', default=None, dest='result_files')
    parser.add_argument('--result_folder', type=str, default=None, dest='result_folder')
    parser.add_argument('-ecc', '--exclude_clouds', dest='exclude_clouds', action='store_true')
    parser.add_argument('-icc', '--include_clouds', dest='exclude_clouds', action='store_false')
    parser.add_argument('-eclsas', '--exclude_lsas', dest='include_lsas', action='store_false')
    parser.add_argument('-iclsas', '--include_lsas', dest='include_lsas', action='store_true')
    parser.add_argument('--allow_cluster_metrics', dest='allow_cluster_metrics', action='store_true')
    parser.add_argument('--no_cluster_metrics', dest='allow_cluster_metrics', action='store_false')
    parser.add_argument('--auto_max_agree', dest='force_max_agree', action='store_false')
    parser.add_argument('--force_max_agree', dest='force_max_agree', action='store_true')
    parser.add_argument('--only_total_from_data', dest='allow_total_from_per_data', action='store_false')
    parser.add_argument('--allow_total_assembly', dest='allow_total_from_per_data', action='store_true')
    parser.add_argument('--enforce_flood_no_flood_present', dest='enforce_flood_no_flood_present', action='store_true')
    parser.add_argument('--allow_dry_or_flood_only', dest='enforce_flood_no_flood_present', action='store_false')
    parser.add_argument('--n_jobs', dest='n_jobs', default=-1)
    parser.add_argument('-s','--seed', dest='seed', default=42)
    parser.add_argument('--n_iter', dest='n_iter', default=1000)
    parser.add_argument('--batch_sizes', nargs='+', dest='batch_sizes', default=[2, 4, 8, 16], type=int)
    parser.add_argument('--batch_random_state', dest='batch_random_state', default=42, type=int)
    parser.add_argument('--batch_iter', dest='batch_iter', default=1000, type=int)
    parser.add_argument('--batch_calc', dest='batch_calc', action='store_true')
    parser.add_argument('--no-batch_calc', dest='batch_calc', action='store_false')
    parser.add_argument('--raise_duplicates', dest='raise_duplicates', action='store_true')
    parser.add_argument('--ignore_duplicates', dest='raise_duplicates', action='store_false')
    parser.add_argument('--memmap_border', dest='memmap_border', type=str, default='2M')
    parser.add_argument('--max_num_unique_predictions', dest='max_num_unique_predictions', type=int, default=None)
    parser.set_defaults(exclude_clouds=True, include_lsas=False, enforce_flood_no_flood_present=True,
                        raise_duplicates=False, batch_calc=False, force_max_agree=False, allow_total_from_per_data=True,
                        allow_cluster_metrics=False)
    args = parser.parse_args()
    if not path.exists(args.database_folder):
        os.makedirs(args.database_folder)
        print(f'Created directory {args.database_folder} as it did not exist.')
    elif not path.isdir(args.database_folder):
        raise ValueError(f'{args.database_folder} exists but is not a directory! Expected the directory for the database!')
    n_jobs = int(args.n_jobs)
    # Override metrics type checking to allow memmap arrays
    DO_TYPE_CHECK = False
    with job.Parallel(n_jobs=n_jobs, max_nbytes=args.memmap_border) as parallel:
        parallel_mode = n_jobs < 0 or n_jobs > 1
        if parallel_mode:
            print('Operating in data parallel mode')
        else:
            print('Operating in sequential data mode')
        mc = MetricConfiguration(parallel,
                                 MetricOptions(exclude_clouds=args.exclude_clouds,
                                               include_lsas=args.include_lsas,
                                               enforce_flood_no_flood_present=args.enforce_flood_no_flood_present,
                                               force_max_agree=args.force_max_agree,
                                               is_parallel_data_computation=parallel_mode,
                                               allow_cluster_metrics=args.allow_cluster_metrics),
                                 random_state=args.batch_random_state,
                                 batch_iter=args.batch_iter,
                                 batch_sizes=(args.batch_sizes if args.batch_calc else None),
                                 allow_total_from_per_data=args.allow_total_from_per_data,
                                 max_num_unique_predictions=args.max_num_unique_predictions)
        db = PandasDatabase(args.database_folder)
        result_files = args.result_files
        if result_files is None:
            result_files = [path.join(args.result_folder, file)
                            for file in os.listdir(args.result_folder)
                            if path.isfile(file) and file.endswith('.tar') or file.endswith('.tar.gz') ]
        for result_file in result_files:
            if not path.exists(result_file):
                raise ValueError(f'{result_file} does not exist, however queried for it to be extracted into the database!')
            elif not path.isfile(result_file):
                raise ValueError(f'{result_file} is not a file, however queried for it to be extracted into the database!')
            elif not (result_file.endswith('.tar') or result_file.endswith('.tar.gz')):
                raise ValueError(f'{result_file} does not seem to be a tar archive, however queried for it to be extracted into the database!'
                                 f'If it actually is a tar archive, please supply it with the file ending ".tar" or ".tar.gz"!')
        for result_file in result_files:
            process_tar(result_file, db, mc)
