import datetime
import inspect
import itertools as itert
import os
import os.path as path
import random
import shutil
import sys
import time
import traceback
import traceback as tb
from collections import defaultdict
from typing import Callable, Optional, Set, Iterable, Tuple, Generator, List, DefaultDict
from metrics import KEY_CONTINGENCY_MATRIX, KEY_CONTINGENCY_UNIQUE_LABELS, KEY_CONTINGENCY_UNIQUE_PREDICTIONS, \
    AccuracyComputation, MaximumAgreementComputation, ToSen1Floods11NormalizedContingency, IOUComputation
import utils
from utils import torch_as_np

import numpy as np
import numpy.random
import optuna
import pandas as pd
import sacred.host_info as host_info
import sacred.metrics_logger
import sortedcontainers
import sortedcontainers as sc
import torch
from optuna import Study
from optuna.distributions import *
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import BaseTrial, FrozenTrial, TrialState
from sacred.observers import FileStorageObserver
from sacred.observers.file_storage import DEFAULT_FILE_STORAGE_PRIORITY
from sacred.utils import PathType

from pipeline import RunHook, RunFileContext


def _inverse_value_check(super_val: Any, sub_val: Any) -> bool:
    if type(super_val) != type(sub_val):
        return True
    elif isinstance(super_val, dict):
        return not check_dict_contains(super_val, sub_val)
    else:
        return super_val != sub_val


def check_dict_contains(super_set: Dict[Any, Any], to_check: Dict[Any, Any]) -> bool:
    if not set(super_set.keys()).issuperset(set(to_check.keys())):
        return False
    res = not any(map(lambda t: _inverse_value_check(*t), map(lambda t: (super_set[t[0]], t[1]), to_check.items())))
    return res


CategoricalChoiceType = Union[CategoricalChoiceType, List]


class ConditionedTrial(BaseTrial):
    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float] = None,
                      log: bool = False, condition: bool = False, enter: bool = True, random: bool = False,
                      selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> float:
        # copied from Trial.py
        if step is not None:
            if log:
                raise ValueError("The parameter `step` is not supported when `log` is True.")
            else:
                return self.suggest_discrete_uniform(name, low, high, step, condition, enter, random, selected_choices)
        else:
            if condition and selected_choices is None:
                raise ValueError('Cannot handle non-discrete conditioning! '
                                 'Please discretize this variable with a suitable step argument!')
            if log:
                return self.suggest_loguniform(name, low, high, random, condition, enter, selected_choices)
            else:
                return self.suggest_uniform(name, low, high, random, condition, enter, selected_choices)

    def suggest_uniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                        selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> float:
        raise NotImplementedError

    def suggest_loguniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                           selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> float:
        raise NotImplementedError

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float, condition: bool = False,
                                 enter: bool = True, random: bool = False,
                                 selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> float:
        raise NotImplementedError

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False, condition: bool = False,
                    enter: bool = True, random: bool = False,
                    selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> int:
        raise NotImplementedError

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType], condition: bool = False,
                            enter: bool = True, random: bool = False,
                            selected_choices: Optional[
                                Sequence[CategoricalChoiceType]] = None) -> CategoricalChoiceType:
        raise NotImplementedError

    def suggest_singleton(self, name: str, value: CategoricalChoiceType, condition: bool = False, enter: bool = True) \
            -> CategoricalChoiceType:
        return self.suggest_categorical(name, (value,), condition, enter)

    def enter_condition(self, name: Union[str, Iterable[str]]):
        raise NotImplementedError

    def leave_condition(self, name: Union[str, Iterable[str]]):
        raise NotImplementedError

    def report(self, value: float, step: int) -> None:
        raise NotImplementedError

    def should_prune(self) -> bool:
        raise NotImplementedError

    def set_user_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def set_system_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        raise NotImplementedError

    @property
    def user_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def system_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        raise NotImplementedError

    @property
    def number(self) -> int:
        raise NotImplementedError


class ConditionNotExploredException(Exception):
    def __init__(self, condition_name: str):
        super(ConditionNotExploredException, self).__init__(f'No value given for condition {condition_name}')
        self.condition_name = condition_name


class TrialRecorder(ConditionedTrial):
    def __init__(self, condition_params_to_use: Dict[str, Any],
                 dist_by_name: Dict[str, BaseDistribution],
                 dist_graph: DefaultDict[str, DefaultDict[Any, Set[str]]]):
        self.condition_params_used = condition_params_to_use
        self.dist_by_name = dist_by_name
        self.dist_graph = dist_graph
        self.active_conditions: sc.SortedList[str] = sc.SortedList()
        self.name_transform: Dict[str, str] = {}
        self.random_vars: Set[str] = set()

    def _record_dist(self, name: str, dist: BaseDistribution, random: bool) -> str:
        new_name = '_'.join([f'[{active_name}={self.condition_params_used[active_name]}]'
                             for active_name in self.active_conditions]) + (
                       '_' if self.active_conditions else '') + name
        if name in self.name_transform:
            raise ValueError(f'Found non-unique name "{name}" - complete name would have been "{new_name}" but it would'
                             ' not have been possible to guarantee uniqueness!')
        self.name_transform[name] = new_name
        for condition in self.active_conditions:
            self.dist_graph[condition][self.condition_params_used[condition]].add(new_name)
        if new_name not in self.dist_by_name:
            self.dist_by_name[new_name] = dist
        if random:
            self.random_vars.add(new_name)
        return new_name

    def _check_condition(self, name: str, default: Any, condition: bool, enter: bool) -> Any:
        if condition:
            if name not in self.condition_params_used:
                raise ConditionNotExploredException(name)
            if enter:
                self.enter_condition(name)
            return self.condition_params_used[name]
        return default

    def suggest_uniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                        selected_choices: Optional[Sequence[float]] = None) -> float:
        replace_dist = CategoricalDistribution(selected_choices) if selected_choices is not None else None
        assert selected_choices is None or not any(map(lambda f: low > f or high < f, selected_choices)), \
            f'Expected a valid range in [{low}, {high}] but got {selected_choices}!'
        name = self._record_dist(name, replace_dist or UniformDistribution(low, high), random)
        return self._check_condition(name, low, condition, enter)

    def suggest_loguniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                           selected_choices: Optional[Sequence[float]] = None) -> float:
        selected_choices = None if selected_choices is None else [np.log(f) for f in selected_choices]
        replace_dist = None if selected_choices is None else CategoricalDistribution(selected_choices)
        assert selected_choices is None or not any(map(lambda f: np.log(low) > f or np.log(high) < f, selected_choices))
        name = self._record_dist(name, replace_dist or LogUniformDistribution(low, high), random)
        return self._check_condition(name, np.log(low), condition, enter)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float, condition: bool = False,
                                 enter: bool = True, random: bool = False,
                                 selected_choices: Optional[Sequence[float]] = None) -> float:
        replace_dist = CategoricalDistribution(selected_choices) if selected_choices is not None else None
        assert selected_choices is None or not any(map(lambda f: low > f or high < f, selected_choices))
        name = self._record_dist(name, replace_dist or DiscreteUniformDistribution(low, high, q), random)
        return self._check_condition(name, low, condition, enter)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False, condition: bool = False,
                    enter: bool = True, random: bool = False,
                    selected_choices: Optional[Sequence[int]] = None) -> int:
        replace_dist = CategoricalDistribution(selected_choices) if selected_choices is not None else None
        if log:
            if step != 1:
                raise ValueError('Step != 1 is not supported for log distribtutions')
            assert selected_choices is None or not any(
                map(lambda i: np.log(low) > i or np.log(high) < i, selected_choices))
            name = self._record_dist(name, replace_dist or IntLogUniformDistribution(low, high, log), random)
        else:
            assert selected_choices is None or not any(map(lambda i: low > i or high < i, selected_choices))
            name = self._record_dist(name, replace_dist or IntUniformDistribution(low, high, step=step), random)
        return self._check_condition(name, np.log(low) if log else low, condition, enter)

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType], condition: bool = False,
                            enter: bool = True, random: bool = False,
                            selected_choices: Optional[
                                Sequence[CategoricalChoiceType]] = None) -> CategoricalChoiceType:
        assert selected_choices is None or not any(map(lambda c: c not in choices, selected_choices)), \
            f'Expected all selected choices ({str(selected_choices)}) to be present in {str(choices)}'

        selected_choices = selected_choices or choices
        name = self._record_dist(name, CategoricalDistribution(selected_choices), random)
        return self._check_condition(name, selected_choices[0], condition, enter)

    def enter_condition(self, name: Union[str, Iterable[str]]):
        if isinstance(name, str):
            if name in self.active_conditions:
                print(f'WARNING: Entering condition {name} more than once. '
                      f'Ignoring, but this may indicate a programming error!')
                return
            entered = self.name_transform[name] if name in self.name_transform else name
            self.active_conditions.add(entered)
        else:
            for sub_name in name:
                self.enter_condition(sub_name)

    def leave_condition(self, name: Union[str, Iterable[str]]):
        if isinstance(name, str):
            entered = self.name_transform[name] if name in self.name_transform else name
            if entered not in self.active_conditions:
                print(f'WARNING: Leaving condition {name} more than once. '
                      f'Ignoring, but this may indicate a programming error!')
                return
            self.active_conditions.remove(entered)
        else:
            for sub_name in name:
                self.leave_condition(sub_name)


def construct_search_space(objective: Callable) -> Generator[Dict[str, Any], None, None]:
    def search_space_iter(distributions: Dict[str, BaseDistribution], ignored_vars: Set[str],
                          allow_continuous: bool = False) -> Generator[Dict[str, Any], None, None]:

        def dist_to_discrete_space(distribution: BaseDistribution) -> Iterable[Any]:
            if isinstance(distribution, CategoricalDistribution):
                return distribution.choices
            elif isinstance(distribution, DiscreteUniformDistribution):
                low, high = decimal.Decimal.from_float(distribution.low), decimal.Decimal.from_float(distribution.high)
                q = decimal.Decimal.from_float(distribution.q)
                n_steps = int(((high - low) // q).to_integral_value(decimal.ROUND_HALF_UP)) + 1
                return [float(v) for v in np.linspace(distribution.low, distribution.high, n_steps)]
            elif isinstance(distribution, IntUniformDistribution):
                return range(distribution.low,
                             distribution.high + distribution.step,
                             distribution.step)
            elif isinstance(distribution, IntLogUniformDistribution):
                return [float(v) for v in np.log(np.arange(distribution.low,
                                                           distribution.high + distribution.step,
                                                           distribution.step))]
            else:
                raise TypeError(f'Cannot discretize distribution of type {type(distribution)}')

        item_list = [item for item in distributions.items()
                     if not isinstance(item[1], (UniformDistribution, LogUniformDistribution))
                     and item[0] not in ignored_vars] \
            if allow_continuous else list(distributions.items())
        for value_tup in itert.product(*map(dist_to_discrete_space, map(lambda t: t[1], item_list))):
            configuration_instance = {key: value
                                      for (key, _), value in zip(item_list, value_tup)}
            yield configuration_instance

    def search_space_generator(variables_in_layer: Set[str], graph: Dict[str, Dict[Any, Set[str]]],
                               dist_by_name: Dict[str, BaseDistribution], ignored_vars: Set[str], all_parents=None,
                               cur_assignment: Optional[Dict[str, Any]] = None) \
            -> Generator[Dict[str, Any], None, None]:
        if not variables_in_layer:
            return
        if all_parents is None:
            # this has a runtime in O(n^2), however due to the efficiency of python comprehensions this is much
            # faster for than a loop over the nodes would be...
            all_parents = {name: {parent: {k for k, s in edge_map.items() if name in s}
                                  for parent, edge_map in graph.items()}
                           for name in dist_by_name.keys()}
            # remove nodes that aren't actually parents...
            all_parents = {name: {parent: values for parent, values in parent_map.items() if values}
                           for name, parent_map in all_parents.items()}
        if cur_assignment is None:
            cur_assignment = {}
        for generated_assignment in search_space_iter(
                sortedcontainers.SortedDict({name: dist_by_name[name] for name in variables_in_layer}),
                ignored_vars,
                True):
            found_any = False
            generated_assignment.update(cur_assignment)
            next_variables = {name for layer_name in variables_in_layer
                              if layer_name not in ignored_vars and layer_name in graph and generated_assignment[
                                  layer_name] in graph[layer_name]
                              for name in graph[layer_name][generated_assignment[layer_name]]
                              if name not in ignored_vars}
            key_set = set(generated_assignment.keys())
            next_variables = {name for name in next_variables
                              if key_set.issuperset(all_parents[name].keys()) and not
                              any(map(lambda p_v: generated_assignment[p_v[0]] not in p_v[1],
                                      all_parents[name].items()))
                              }
            for sub_assignment in search_space_generator(next_variables,
                                                         graph,
                                                         dist_by_name,
                                                         ignored_vars,
                                                         all_parents,
                                                         generated_assignment.copy()):
                # Check if there exists some element in the sub_assignment that was to be generated in the immediately
                # following level for which there exists a parent that is not in the child_assignment
                if any(map(lambda child: child in next_variables and any(
                        map(lambda p: p not in generated_assignment, all_parents[child])),
                           sub_assignment.keys())):
                    print('ERROR: SKIPPING ASSIGNMENT! CHECK search_space_generator!', file=sys.stderr)
                    continue
                found_any = True
                yield sortedcontainers.SortedDict(sub_assignment)
            if not found_any:
                yield generated_assignment

    def calc_free_variables(dist_by_name: Dict[str, BaseDistribution],
                            dist_graph: DefaultDict[str, DefaultDict[Any, Set[str]]]) \
            -> Set[str]:
        free_variables: Set[str] = dist_by_name.keys() - {name
                                                          for sub_dict in dist_graph.values()
                                                          for sub_set in sub_dict.values()
                                                          for name in sub_set}
        return free_variables

    def inspect_objective() -> Tuple[Dict[str, BaseDistribution], DefaultDict[str, DefaultDict[Any, Set[str]]],
                                     Dict[str, str], Set[str]]:
        dist_by_name: Dict[str, BaseDistribution] = {}
        # directed graph with distribution name as node id and edges which are uniquely identified by the target node
        # and the value that leads to this node
        # notice that transitive dependencies will be reflected in this graph
        dist_graph: DefaultDict[str, DefaultDict[Any, Set[str]]] = defaultdict(lambda: defaultdict(set))
        # the reverse map of what the trial_recorders need and create - map a unique name to the real name
        name_map_reverse: Dict[str, str] = {}
        ignored_vars = set()
        recorder = TrialRecorder({}, dist_by_name, dist_graph)
        try:
            objective(recorder)
        except ConditionNotExploredException as e:
            # print('Condition discovered:', e.condition_name)
            condition_distributions: Set[str, BaseDistribution] = {e.condition_name}
            free_variables = calc_free_variables(dist_by_name, dist_graph)
            finished = False
            while not finished:
                # print('Initiating search.')
                i = 0
                t = time.time()
                try:
                    to_ignore = {key for key in dist_by_name.keys() if key not in condition_distributions}
                    for search_space_instance in search_space_generator(free_variables,
                                                                        {key: {k: v.copy() for k, v in edge_map.items()}
                                                                         for key, edge_map in dist_graph.items()},
                                                                        dist_by_name.copy(),
                                                                        to_ignore):
                        recorder = TrialRecorder(search_space_instance, dist_by_name, dist_graph)
                        objective(recorder)
                        name_map_reverse.update(
                            {unique_name: name for name, unique_name in recorder.name_transform.items()})
                        ignored_vars.update(recorder.random_vars)
                        i += 1
                    t = time.time() - t
                    # print(f'Searched {i} combinations without discovering another condition! Took {t:.3f}s.')
                    # unique_set = {tuple([key_val_pair for key_val_pair in search_space_instance.items()])
                    #               for search_space_instance in search_space_generator(free_variables,
                    #                                                                   dist_graph,
                    #                                                                   dist_by_name, to_ignore)
                    #               }
                    # print(f'Found {len(unique_set)} unique combinations')
                except ConditionNotExploredException as e_inner:
                    t = time.time() - t
                    # print(f'Condition "{e_inner.condition_name}" was discovered after {i} iterations which took {t:.3f}s')
                    condition_distributions.add(e_inner.condition_name)
                    free_variables = calc_free_variables(dist_by_name, dist_graph)
                else:
                    finished = True
        else:
            name_map_reverse.update({unique_name: recorder.name_transform[unique_name]
                                     for unique_name in itert.chain.from_iterable(recorder.name_transform.values())})
            ignored_vars.update(recorder.random_vars)
        return dist_by_name, dist_graph, name_map_reverse, ignored_vars

    print('Inspecting objective function.')
    t = time.time()
    dist_by_name, dist_graph, name_map, ignored_vars = inspect_objective()
    t = time.time() - t
    print(f'Inspection completed. Took {t:.3f}s.')
    # print('Distribution by name:')
    # print(json.dumps({name: repr(dist) for name, dist in dist_by_name.items()}, indent=4))
    # print('Distribution Graph:')
    # print(json.dumps({key: {k:list(s) for k, s in edge_map.items()}for key, edge_map in dist_graph.items()}, indent=4))
    # print('Name Map:')
    # print(json.dumps(name_map, indent=4))
    # print('Ignored (Random) Variables:', ignored_vars)
    free_variables = calc_free_variables(dist_by_name, dist_graph)
    # print('Free Variables:', free_variables)
    for assignment in search_space_generator(free_variables, dist_graph, dist_by_name, ignored_vars):
        yield {name_map[name]: val for name, val in assignment.items()}


class TrialWrapper(ConditionedTrial):
    def __init__(self, delegate: BaseTrial) -> None:
        super().__init__()
        self.delegate = delegate
        self.config = {}

    def replacement_dist(self, name: str, random: bool,
                         selected_choices: Optional[Sequence[CategoricalChoiceType]] = None) -> Optional:
        if selected_choices is not None:
            return self.delegate.suggest_categorical(name, selected_choices)
        return None

    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float] = None,
                      log: bool = False, condition: bool = False, enter: bool = True, random: bool = False,
                      selected_choices: Optional[Sequence[float]] = None) -> float:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_float(name, low, high, step=step, log=log)
        self.config[name] = res
        return res

    def suggest_uniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                        selected_choices: Optional[Sequence[float]] = None) -> float:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_uniform(name, low, high)
        self.config[name] = res
        return res

    def suggest_loguniform(self, name: str, low: float, high: float, random: bool = False, condition: bool = False, enter: bool = True,
                           selected_choices: Optional[Sequence[float]] = None) -> float:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_loguniform(name, low, high)
        self.config[name] = res
        return res

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float, condition: bool = False,
                                 enter: bool = True, random: bool = False,
                                 selected_choices: Optional[Sequence[float]] = None) -> float:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_discrete_uniform(name, low, high, q)
        self.config[name] = res
        return res

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False, condition: bool = False,
                    enter: bool = True, random: bool = False,
                    selected_choices: Optional[Sequence[int]] = None) -> int:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_int(name, low, high, step, log)
        self.config[name] = res
        return res

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType], condition: bool = False,
                            enter: bool = True, random: bool = False,
                            selected_choices: Optional[
                                Sequence[CategoricalChoiceType]] = None) -> CategoricalChoiceType:
        res = self.replacement_dist(name, random, selected_choices)
        if res is None:
            res = self.delegate.suggest_categorical(name, choices if selected_choices is None else selected_choices) # choices)
        self.config[name] = res
        return res

    def enter_condition(self, name: Union[str, Iterable[str]]):
        pass

    def leave_condition(self, name: Union[str, Iterable[str]]):
        pass

    def report(self, value: float, step: int) -> None:
        return self.delegate.report(value, step)

    def should_prune(self) -> bool:
        return self.delegate.should_prune()

    def set_user_attr(self, key: str, value: Any) -> None:
        return self.delegate.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        return self.set_user_attr(key, value)

    @property
    def params(self) -> Dict[str, Any]:
        return self.delegate.params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        return self.delegate.distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self.delegate.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self.delegate.system_attrs

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        return self.delegate.datetime_start

    @property
    def number(self) -> int:
        return self.delegate.number


class TrialAbort(Exception):
    def __init__(self, other: Exception, text: Optional[str] = None) -> None:
        super().__init__(f'{text or ""} Trial failed due to {type(other)}! Please inspect the logs')


class AutoGridSampler(BaseSampler):
    def __init__(self, search_space_iterable: Iterable[Dict[str, Any]], seed: Optional[int] = None) -> None:
        iterator = iter(search_space_iterable)
        try:
            self.next_value = next(iterator)
        except StopIteration:
            raise ValueError('Empty serach space provided!')
        else:
            self.iterator = iterator
        self.random_sampler = RandomSampler(seed)

    def reseed_rng(self) -> None:
        self.random_sampler.reseed_rng()

    def _validate_no_repeated_trial(self, study: Study):
        while any(map(lambda t: t.state == TrialState.COMPLETE and t.params == self.next_value,
                      study.trials)):
            self.next_value = next(self.iterator)

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> Dict[str, BaseDistribution]:
        try:
            self._validate_no_repeated_trial(study)
        except StopIteration as e:
            raise TrialAbort(e, 'Search space is completely covered by the given study!') from e
        return {}

    def sample_relative(self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]) -> Dict[
        str, Any]:
        return {}

    def sample_independent(self, study: Study, trial: FrozenTrial, param_name: str,
                           param_distribution: BaseDistribution) -> Any:
        if param_name in self.next_value:
            return self.next_value[param_name]
        return self.random_sampler.sample_independent(study, trial, param_name, param_distribution)

    def after_trial(self, study: Study, trial: FrozenTrial, state: TrialState,
                    values: Optional[Sequence[float]]) -> None:
        try:
            self.next_value = next(self.iterator)
            while any(map(lambda t: t.state == TrialState.COMPLETE and check_dict_contains(t.params, self.next_value),
                          study.trials)):
                self.next_value = next(self.iterator)
        except StopIteration:
            study.stop()


def get_objective_value_from_result(result: Any, key_objective_value: str = 'optimized_param') -> float:
    if isinstance(result, (float, int, tuple)):
        return result
    elif isinstance(result, dict):
        res = result[key_objective_value]
        return float(res)
    else:
        raise ValueError('Unsupported return type ' + str(type(result)))


def json_dict_to_dist(json_dict: Dict[str, Any]) -> BaseDistribution:
    if json_dict["name"] == CategoricalDistribution.__name__:
        json_dict["attributes"]["choices"] = tuple(json_dict["attributes"]["choices"])

    for cls in optuna.distributions.DISTRIBUTION_CLASSES:
        if json_dict["name"] == cls.__name__:
            return cls(**json_dict["attributes"])

    raise ValueError("Unknown distribution class: {}".format(json_dict["name"]))


DistributionMap = Dict[int, Dict[str, Dict[str, Any]]]
TRIALS_DF_FILE = 'trials.csv'
TRIALS_INTERMEDIATES_DIR = 'trial_intermediates'
TRIALS_DISTRIBUTIONS = 'trial_distributions.json'


def load_previous_run(study: optuna.study, experiment_folder: str) -> optuna.study:
    def add_trial_to_study(study: optuna.study, row: pd.Series, trial_intermediates: Dict[int, Dict[int, float]],
                           dist_map: DistributionMap):
        number = int(row['number'])
        value = row['value'] if 'value' in row.keys() else None
        values = [val for key, val in sorted(row.items(), key=lambda i: i[0]) if key.startswith('values_')]
        values = values if values else None
        intermediate = trial_intermediates[number] if number in trial_intermediates else None
        trial = optuna.create_trial(state=TrialState[row['state']],
                                    params=({param_name: json_dict['value']
                                            for param_name, json_dict in dist_map[number].items()} if number in dist_map else {}),
                                    distributions=({param_name: json_dict_to_dist(json_dict)
                                                   for param_name, json_dict in dist_map[number].items()} if number in dist_map else {}),
                                    intermediate_values=intermediate,
                                    value=value, values=values)
        if number in dist_map:
            trial.datetime_start = row['datetime_start']
            trial.datetime_complete = row['datetime_complete']
        study.add_trial(trial)

    trials_csv = path.join(experiment_folder, TRIALS_DF_FILE)
    if path.exists(trials_csv):
        trial_table = pd.read_csv(trials_csv)
        trial_intermediate_folder = path.join(experiment_folder, TRIALS_INTERMEDIATES_DIR)
        trial_intermediates = {int(file.rsplit('.csv', 1)[-1]): pd.r(file)
                               for file in os.listdir(trial_intermediate_folder)} \
            if path.exists(trial_intermediate_folder) and path.isdir(trial_intermediate_folder) else {}
        with open(path.join(experiment_folder, TRIALS_DISTRIBUTIONS)) as fd:
            dist_map = {int(k): v for k, v in json.load(fd).items()}
        trial_table.apply(lambda r: add_trial_to_study(study, r, trial_intermediates, dist_map), axis=1)
    return study


def dist_json_dict(dist: BaseDistribution, value: Any) -> Dict[str, Any]:
    return {'name': dist.__class__.__name__, 'attributes': dist._asdict(), 'value': value}

def save_current_run(study: optuna.study, experiment_folder: str, is_multicriteria: bool):
    print(f'Saving run in folder {experiment_folder} with trials in {TRIALS_DF_FILE}')
    trials_df: pd.DataFrame = study.trials_dataframe(
        attrs=('number', 'value', 'state', 'datetime_start', 'datetime_complete'))
    trials_df_file = path.join(experiment_folder, TRIALS_DF_FILE)
    if path.exists(trials_df_file):
        shutil.copyfile(trials_df_file, trials_df_file[:-5] + '_copy.csv')
    trials_df.to_csv(trials_df_file)
    trial_intermediate_folder = path.join(experiment_folder, TRIALS_INTERMEDIATES_DIR)
    trial_dist_file = path.join(experiment_folder, TRIALS_DISTRIBUTIONS)
    if not path.exists(trial_intermediate_folder):
        os.mkdir(trial_intermediate_folder)
    print(f'Found {len(study.trials)} trials. ')
    print(f'Potential intermediate values, which are to be saved in {trial_intermediate_folder}.')
    print(f'Potential intermediate values, which are to be saved in {trial_intermediate_folder}.')
    dist_map: DistributionMap = {}
    if path.exists(trial_dist_file):
        try:
            with open(trial_dist_file, 'r') as fd:
                dist_map = json.load(fd)
            shutil.copyfile(trial_dist_file, trial_dist_file[:-5] + '_copy.json')
        except Exception:
            print('Caught exception reading trial distribution file!!! Leaving copy untouched', file=sys.stderr)
            print(traceback.format_exc())
    for trial in study.trials:
        trial: optuna.trial.FrozenTrial = trial
        if trial.intermediate_values:
            trial_file = path.join(trial_intermediate_folder, f'{trial.number}.json')
            try:
                with open(trial_file, 'w') as fd:
                    json.dump(trial.intermediate_values, fd, indent=4)
            except Exception:
                print('Caught exception writing trial intermediate file!!! Skipping!', file=sys.stderr)
                print(traceback.format_exc())
        for param_name, dist in trial.distributions.items():
            if trial.number not in dist_map:
                dist_map[trial.number] = {param_name: dist_json_dict(dist, trial.params[param_name])}
            else:
                dist_map[trial.number][param_name] = dist_json_dict(dist, trial.params[param_name])
    with open(trial_dist_file, 'w') as fd:
        json.dump(dist_map, fd, indent=4)
    if is_multicriteria:
        best = [{'id': trial._trial_id, 'number': trial.number, 'state': str(trial.state), 'values': trial.values,
                 'datetime_start': str(trial.datetime_start), 'datetime_complete': str(trial.datetime_complete)}
                for trial in study.best_trials]
        best_file = path.join(experiment_folder, 'best_trials.json')
        if path.exists(best_file) and path.isfile(best_file):
            shutil.copyfile(best_file, path.join(experiment_folder, 'best_trials_copy.json'))
        with open(best_file, 'w') as fd:
            json.dump(best, fd, indent=4)


class DirectFileStorage(FileStorageObserver, RunFileContext):
    def __init__(self, basedir: PathType, resource_dir: Optional[PathType] = None,
                 source_dir: Optional[PathType] = None, template: Optional[PathType] = None,
                 priority: int = DEFAULT_FILE_STORAGE_PRIORITY, redirect_sysout: bool = True,
                 do_host: bool = True):
        super().__init__(basedir, resource_dir, source_dir, template, priority)
        self.logger = sacred.metrics_logger.MetricsLogger()
        self.redirect_sysout = redirect_sysout
        self.do_host = do_host

    def start(self, config: Dict[str, Any], _id: Optional[int] = None):
        os.makedirs(self.basedir, exist_ok=True)
        self._make_run_dir(_id)
        cout_file, errout_file = self.open_out()
        self.run_entry = {
            "experiment": {
                'base_dir': self.basedir,
                'cout': cout_file,
                'errout': errout_file
            },
            "command": None,
            "host": dict(sacred.host_info.get_host_info()) if self.do_host else {},
            "start_time": datetime.datetime.today().isoformat(),
            "meta": {},
            "status": "RUNNING",
            "resources": [],
            'sources': [],
            "artifacts": [],
            "heartbeat": None,
        }
        self.config = config
        self.info = {}
        self.cout = ""
        self.cout_write_cursor = 0

        self.save_json(self.run_entry, "run.json")
        self.save_json(self.config, "config.json")

    def _get_metrics_dict(self) -> Dict[str, Dict[str, Any]]:
        metrics = self.logger.get_last_metrics()
        grouped = {s.name: [s1 for s1 in metrics if s1.name == s.name] for s in metrics}
        return {name: {'values': [s.value for s in ls],
                       'steps': [s.step for s in ls],
                       'timestamps': [s.timestamp for s in ls]}
                for name, ls in grouped}

    def open_out(self) -> Tuple[Optional[str], Optional[str]]:
        if self.redirect_sysout:
            cout_file = path.join(self.dir, 'cout.txt')
            errout_file = path.join(self.dir, 'errout.txt')
            self.orig_stdout = sys.stdout
            self.orig_stdout.flush()
            self.orig_stderr = sys.stderr
            self.orig_stderr.flush()
            sys.stdout = open(cout_file, 'w')
            sys.stderr = open(errout_file, 'w')
            return cout_file, errout_file
        else:
            return None, None

    def close_out(self):
        if self.redirect_sysout:
            sys.stdout.flush()
            sys.stdout.close()
            sys.stderr.flush()
            sys.stderr.close()
            sys.stdout = self.orig_stdout
            sys.stderr = self.orig_stderr

    def fail(self):
        self.log_metrics(self._get_metrics_dict(), None)
        self.run_entry["stop_time"] = datetime.datetime.today().isoformat()
        self.run_entry["status"] = "FAILED"
        self.run_entry["fail_trace"] = tb.format_exc()
        self.save_json(self.run_entry, "run.json")
        self.close_out()

    def complete(self, result):
        self.log_metrics(self._get_metrics_dict(), None)
        self.run_entry["stop_time"] = datetime.datetime.today().isoformat()
        self.run_entry["result"] = result
        self.run_entry["status"] = "COMPLETED"
        self.save_json(self.run_entry, "run.json")
        self.close_out()

    def add_artifact(self, filename, name=None, metadata=None, content_type=None):
        if name is None:
            name = path.basename(filename)
        self.artifact_event(name, filename, metadata, content_type)

    # override to avoid having a lot of duplicate artifacts in the run_entry...
    def artifact_event(self, name, filename, metadata=None, content_type=None):
        self.save_file(filename, name)
        if name not in self.run_entry["artifacts"]:
            self.run_entry["artifacts"].append(name)
        self.save_json(self.run_entry, "run.json")

    def log_scalar(self, metric_name, value, step=None):
        self.logger.log_scalar_metric(metric_name, value, step=step)

    def get_execution_dir(self) -> str:
        return self.basedir

    def get_active_dir(self) -> Optional[str]:
        return self.dir


class SubFileStorage(RunFileContext):
    def __init__(self, delegate: RunFileContext):
        super().__init__()
        self.delegate = delegate
        self.dir = None
        self._id = None
        self.run_entry = None

    def add_artifact(self, artifact_file, name: Optional[str] = None):
        if name is None:
            name = path.basename(artifact_file)
        self.run_entry['artifacts'].append(name)
        self.delegate.add_artifact(artifact_file, path.join(str(self._id), name))

    def get_execution_dir(self) -> str:
        return self.delegate.get_active_dir()

    def get_active_dir(self) -> Optional[str]:
        assert self.dir is not None
        return self.dir

    def start(self, config: Dict[str, Any], _id: Optional[int] = None):
        os.makedirs(self.get_execution_dir(), exist_ok=True)
        self.dir = None
        if _id is None:
            dir_nrs = [
                int(d)
                for d in os.listdir(self.get_execution_dir())
                if path.isdir(path.join(self.get_execution_dir(), d)) and d.isdigit()
            ]
            self._id = max(dir_nrs, default=0) +1
        else:
            self._id = _id
        self.dir = path.join(self.get_execution_dir(), str(self._id))
        os.mkdir(self.dir)

        self.run_entry = {
            "experiment": {
                'base_dir': self.get_execution_dir()
            },
            "start_time": datetime.datetime.today().isoformat(),
            "status": "RUNNING",
            "resources": [],
            'sources': [],
            "artifacts": [],
            'config': config
        }
        with open(path.join(self.get_active_dir(), 'config.json'), "w") as f:
            json.dump(self.run_entry, f, sort_keys=True, indent=2)

    def fail(self):
        self.run_entry["stop_time"] = datetime.datetime.today().isoformat()
        self.run_entry["status"] = "FAILED"
        self.run_entry["fail_trace"] = tb.format_exc()

        with open(path.join(self.get_active_dir(), 'run.json'), "w") as f:
            json.dump(self.run_entry, f, sort_keys=True, indent=2)

        self.dir = None
        self._id = None

    def complete(self, result):
        self.run_entry["stop_time"] = datetime.datetime.today().isoformat()
        self.run_entry["result"] = result
        self.run_entry["status"] = "COMPLETED"

        with open(path.join(self.get_active_dir(), 'run.json'), "w") as f:
            json.dump(self.run_entry, f, sort_keys=True, indent=2)

        self.dir = None
        self._id = None
        self._run_entry = None

def run_sacred_grid_search(experiment_base_folder: str, ex_name: str, main_fun: Callable,
                           trial_config_fun: Callable, seed: Optional[int] = None,
                           direction: Optional[Union[optuna.study.StudyDirection,
                                                     Sequence[optuna.study.StudyDirection]]] = None,
                           pruner: Optional[optuna.pruners.BasePruner] = None, timeout=None,
                           key_objective_value: str = 'optimized_param',
                           redirect_output: bool = True):
    print('WARNING: Hacking away the sacred cpu info gatherer as it takes ages on my machine...', file=sys.stderr)
    host_info._host_info_gatherers_list = list(
        filter(lambda f: f is not host_info._cpu, host_info._host_info_gatherers_list))
    if seed is not None:
        ex_name += '_' + str(seed)
    experiment_folder = path.join(experiment_base_folder, ex_name)
    obs = DirectFileStorage(experiment_folder, redirect_sysout=redirect_output)

    def objective(trial: BaseTrial) -> float:
        wrapped = TrialWrapper(trial)
        additional = trial_config_fun(wrapped)
        config = wrapped.config
        if additional is not None:
            if not isinstance(additional, dict):
                print(f'WARNING: got non dict additional values {additional}')
            else:
                config.update(additional)
        if seed is not None:
            config['_seed'] = seed
        try:
            obs.start(config)
            result = main_fun(trial)(config, obs, random.randint(0, 2_000_000))
        except Exception as e:
            obs.fail()
            raise TrialAbort(e) from e
        else:
            obs.complete(result)
            return get_objective_value_from_result(result, key_objective_value)

    is_single_objective = direction is None or isinstance(direction, optuna.study.StudyDirection)
    def callback(study: optuna.study, frozen_trial: optuna.trial.FrozenTrial):
        # obs.clear_saved_metrics()
        save_current_run(study, experiment_folder, not is_single_objective)

    ls = list(construct_search_space(trial_config_fun))
    print('Search-space has size', len(ls))
    # print(json.dumps(ls))
    sampler = AutoGridSampler(construct_search_space(trial_config_fun), seed)
    if is_single_objective:
        study = optuna.create_study(sampler=sampler, study_name=ex_name, pruner=pruner, direction=direction)
    else:
        study = optuna.create_study(sampler=sampler, study_name=ex_name, pruner=pruner, directions=direction)
    study = load_previous_run(study, experiment_folder)
    study.optimize(objective, gc_after_trial=True, catch=(TrialAbort,), timeout=timeout,
                   callbacks=[callback])

def run_tpe_optimize(experiment_base_folder: str, ex_name: str, main_fun: Callable,
                           trial_config_fun: Callable, seed: Optional[int] = None,
                           direction: Optional[Union[optuna.study.StudyDirection,
                                                     Sequence[optuna.study.StudyDirection]]] = None,
                           pruner: Optional[optuna.pruners.BasePruner] = None, n_trials=None,
                           key_objective_value: str = 'optimized_param',
                           redirect_output: bool = True, configs_to_test: Iterable[Dict[str, Any]] = tuple(), n_startup: int = 10, multivariate: bool = True):
    print('WARNING: Hacking away the sacred cpu info gatherer as it takes ages on my machine...', file=sys.stderr)
    host_info._host_info_gatherers_list = list(
        filter(lambda f: f is not host_info._cpu, host_info._host_info_gatherers_list))
    if seed is not None:
        ex_name += '_' + str(seed)
    experiment_folder = path.join(experiment_base_folder, ex_name)
    obs = DirectFileStorage(experiment_folder, redirect_sysout=redirect_output)

    def objective(trial: BaseTrial) -> float:
        wrapped = TrialWrapper(trial)
        additional = trial_config_fun(wrapped)
        config = wrapped.config
        if additional is not None:
            if not isinstance(additional, dict):
                print(f'WARNING: got non dict additional values {additional}')
            else:
                config.update(additional)
        if seed is not None:
            config['_seed'] = seed
        try:
            obs.start(config)
            result = main_fun(trial)(config, obs, random.randint(0, 2_000_000))
        except Exception as e:
            obs.fail()
            raise TrialAbort(e) from e
        else:
            obs.complete(result)
            return get_objective_value_from_result(result, key_objective_value)

    is_single_objective = direction is None or isinstance(direction, optuna.study.StudyDirection)
    def callback(study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial):
        # obs.clear_saved_metrics()
        save_current_run(study, experiment_folder, not is_single_objective)

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, multivariate=multivariate, seed=seed)
    if is_single_objective:
        study = optuna.create_study(sampler=sampler, study_name=ex_name, pruner=pruner, direction=direction)
    else:
        study = optuna.create_study(sampler=sampler, study_name=ex_name, pruner=pruner, directions=direction)
    study = load_previous_run(study, experiment_folder)
    for config in configs_to_test:
        study.enqueue_trial(config)
    study.optimize(objective, gc_after_trial=True, catch=(TrialAbort,), n_trials=n_trials,
                   callbacks=[callback])


def trial_wrapper(fun: Callable) -> Callable:
    """
    Decorator that allows a sacred main function to be used nicely with optuna - in particular the function can
    have all standard sacred arguments as well as admitting an additional "_trial" and "_hook" parameter.
    :param fun:
    :return:
    """
    args = inspect.signature(fun).parameters
    valid_keyword = {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    kwargs_param = [k for k, p in args.items() if p.kind == inspect.Parameter.VAR_KEYWORD]
    kwargs_param = kwargs_param[0] if kwargs_param else None

    def wrapped_fun(trial):
        def res_fun(_config, _run, _seed):
            kwargs = {key: value for key, value in _config.items() if key in args and args[key].kind in valid_keyword}
            if kwargs_param is not None:
                kwargs[kwargs_param] = _config
            if '_config' in args:
                kwargs['_config'] = _config
            if '_run' in args:
                kwargs['_run'] = _run
            if '_seed' in args:
                seed = _config['_seed'] if '_seed' in _config else _seed
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)
                kwargs['_seed'] = seed
            else:
                seed = None
            if '_trial' in args:
                kwargs['_trial'] = trial
            if '_hook' in args:
                with RunHook(_run, trial, seed) as hook:
                    kwargs['_hook'] = hook
                    return fun(**kwargs)
            else:
                return fun(**kwargs)

        return res_fun

    return wrapped_fun

def accuracy_from_contingency(d: dict, max_agree: bool = False) -> Tuple[float, int]:
    cm = torch_as_np(d[KEY_CONTINGENCY_MATRIX])
    ul = torch_as_np(d[KEY_CONTINGENCY_UNIQUE_LABELS])
    up = torch_as_np(d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS])
    d[KEY_CONTINGENCY_MATRIX] = cm
    d[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
    d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = up
    comp = AccuracyComputation('accuracy')
    if max_agree:
        comp = MaximumAgreementComputation([comp])
    res = ToSen1Floods11NormalizedContingency(comp).forward(step=None, final=True, **d)
    return (res['accuracy'], 1) if 'accuracy' in res else (0.0, 0)

def iou_from_contingency(d: dict, max_agree: bool = False) -> Tuple[float, int]:
    cm = torch_as_np(d[KEY_CONTINGENCY_MATRIX])
    ul = torch_as_np(d[KEY_CONTINGENCY_UNIQUE_LABELS])
    up = torch_as_np(d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS])
    d[KEY_CONTINGENCY_MATRIX] = cm
    d[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
    d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS] = up
    comp = IOUComputation('iou')
    if max_agree:
        comp = MaximumAgreementComputation([comp])
    res = ToSen1Floods11NormalizedContingency(comp).forward(step=None, final=True, **d)
    return (res['iou'][1], 1) if 'iou' in res and 1 in res['iou'] else (0.0, 0)

def find_folder_matching_config(_hook: RunHook, _config: Dict[str, Any],
                                config_skip_predicate: Callable[[Dict[str, Any]], bool],
                                source_deletables: Iterable[str] = tuple(),
                                target_deletables: Iterable[str] = tuple()) -> str:
    dir = _hook.run_obj.get_execution_dir()
    print(f'Inspecting {dir} for a model to finetune (matching config {_config})')
    config_copy = _config.copy()
    for to_del in source_deletables:
        #config_copy = utils.del_if_present(utils.del_if_present(_config.copy(), 'train_type'), 'finetune_lr')
        config_copy = utils.del_if_present(config_copy, to_del)
    config_copy = tuple(sorted(config_copy.items()))
    existing_runs = [int(f) for f in os.listdir(dir) if
                     path.exists(path.join(dir, f)) and path.isdir(path.join(dir, f)) and f.isnumeric()]
    existing_runs = list(sorted(existing_runs, reverse=True))
    # print(f'Inspecting runs: {existing_runs}')
    found_folder = None
    for run in existing_runs:
        d = path.join(dir, str(run))
        config_file = path.join(d, 'config.json')
        run_file = path.join(d, 'run.json')
        if not path.exists(config_file) or not path.exists(run_file):
            continue
        try:
            with open(config_file, 'r') as fd:
                loaded_config = json.load(fd)
            with open(run_file, 'r') as fd:
                loaded_run = json.load(fd)
        except:
            continue
        if config_skip_predicate(loaded_config):#loaded_config['train_type'] != 'weak':
            continue
        if loaded_run['status'] != 'COMPLETED':
            continue
        for to_del in target_deletables:
            loaded_config = utils.del_if_present(loaded_config, to_del)
        #del loaded_config['train_type']
        #utils.del_if_present(loaded_config, 'weak_lr')
        loaded_config = tuple(sorted(loaded_config.items()))
        # print(f'Comparing config: {loaded_config} against {config_copy}')
        if loaded_config == config_copy:
            found_folder = d
    if found_folder is None:
        raise RuntimeError(f'Cannot find any models to finetune in experiment folder {dir}!')
    return found_folder


if __name__ == '__main__':
    # should yield exactly 4 * 25 = 100 trials (aka finish with trial 99)
    def test_trail_config_fun(trial: ConditionedTrial):
        # in total 3 + 1 possibilities
        cluster_alg = trial.suggest_categorical('cluster_alg', ('K-Means', 'MeanShift'), condition=True)
        if cluster_alg == 'K-Means':
            # 3 possibilities
            trial.suggest_int('num_clusters', 2, 100, selected_choices=[2, 3, 4])
        else:
            # 1 possibility as it is set to be random
            # value is expected to be interpreted as 10^{value}
            trial.suggest_uniform('bandwidth', -2.0, 2.0, selected_choices=[-2.0, 2.0])  # random=True)
        trial.leave_condition('cluster_alg')
        # in total 9 + 16 = 25 possibilities
        class_alg = trial.suggest_categorical('class_alg', ('SVM', 'NN'), condition=True)
        if class_alg == 'SVM':
            # yields 9 possibilities
            trial.suggest_uniform('C', -2, 2, selected_choices=[-2.0 + i * 0.5 for i in range(9)])
        else:
            # 4 * 4 possibilities = 16
            n_layers = trial.suggest_int('n_layers', 2, 5, condition=True, enter=False)
            init_width = trial.suggest_int('width', 4, 16, condition=True, enter=False, selected_choices=[4, 8, 12, 16])
            width = init_width
            trial.enter_condition(('width', 'n_layers'))
            for i in range(n_layers):
                if i <= n_layers // 2:
                    trial.suggest_singleton(f'layer {i} of {n_layers - 1} started with {init_width}', width * i)
                else:
                    trial.suggest_singleton(f'layer {i} of {n_layers - 1} started with {init_width}',
                                            width * (n_layers - i))


    @trial_wrapper
    def main_fun(_config, _hook: RunHook) -> Tuple[float, float]:
        print('Generated: ', str(_config))
        print('Trial is', _hook.trial_obj)
        return random.random(), random.random()


    ex_dir = path.abspath('experiments/experiments')
    # if path.exists(ex_dir):
    #    shutil.rmtree(ex_dir)
    # os.mkdir(ex_dir)
    print('Starting search')
    run_sacred_grid_search(ex_dir, 'test', main_fun, trial_config_fun=test_trail_config_fun,
                           direction=[optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MINIMIZE])

def gaussian_sampling(dists: Dict[str, Union[Tuple[float, ...], List[Tuple[float, ...]]]], gen: np.random.Generator) -> Dict[str, float]:
    return {k: (float(gen.normal(*v[gen.integers(0, len(v))])) if isinstance(v, list) else float(gen.normal(*v)))
            for k, v in dists.items()}