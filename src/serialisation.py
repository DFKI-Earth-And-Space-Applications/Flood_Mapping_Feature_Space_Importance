import inspect
import json
import sys
from collections import deque
from types import FunctionType
from typing import Any, Dict, List, Union, Callable, Optional, Set, TextIO

import numpy as np
import torch

import utils

TYPE_KEY = 'type'
VALUE_KEY = 'value'

def get_type_str(type_val: Any, check_type: bool = True) -> str:
    mod = inspect.getmodule(type_val)
    clazz = type_val if check_type and isinstance(type_val, type) else type_val.__class__
    if mod is None and clazz is None:
        raise ValueError('No name available for type ' + type(type_val))
    type_val = mod.__name__ if mod is not None else ''
    if mod is not None and clazz is not None:
        type_val += '.'
    type_val += clazz.__qualname__ if clazz is not None else ''
    return type_val

def type_value_embed(type_val: Any, value: Any) -> Dict[str, Any]:
    if type(type_val) is not str:
        if isinstance(type_val, JSONSerializable):
            type_val = type_val.type_str()
        else:
            type_val = get_type_str(type_val)

    return {
        TYPE_KEY: type_val,
        VALUE_KEY: value
    }


def get_function_arguments(fun: Callable, ignored_params: Optional[Set[str]] = None, raise_missing: bool =True) -> Dict[str, Any]:
    ignored_params = ignored_params or set()
    sig = inspect.signature(fun)
    param_dict: Dict[str, Any] = {
        key: param.default for key, param in sig.parameters.items() if param.default is not inspect.Parameter.empty
    }
    if raise_missing:
        _, _, varkw, _, _, _, _ = inspect.getfullargspec(fun)
        if varkw is not None:
            ignored_params.add(varkw)
        # store in list for debugging purposes...
        ls = list(
            filter(lambda b: b, map(lambda p: p not in ignored_params and p not in param_dict, sig.parameters.keys())))
        if any(ls):
            raise ValueError(f'Some arguments do not provide default values!!! Inspecting {repr(fun)} which gave '
                             f'signature {str(sig)} (with keys: {str(sig.parameters.keys())}) and param_dict '
                             f'{str(param_dict)}. However only {str(ignored_params)} is/are ignored')
    return param_dict


class JSONSerializable:
    def get_params_as_dict(self) -> Dict[str, Any]:
        res = {}
        for key, value in self.__dict__.items():
            if isinstance(value, FunctionType):
                print(f'Found function type object attribute "{key}" in class {self.__class__.__name__}.'
                      f'This should be avoided by overriding the get_params_as_dict function!!!', file=sys.stderr)
            else:
                res[key] = value
        return res

    def type_str(self):
        return get_type_str(self)

class TreeAnalyzer:
    def __init__(self):
        self.reference_set: utils.DefaultedIdentityDict[JSONSerializable, int] = utils.DefaultedIdentityDict(lambda: 0)
        self.parent_children: utils.DefaultedIdentityDict[JSONSerializable, List[JSONSerializable]] = \
            utils.DefaultedIdentityDict(list)
        self.child_parent: utils.IdentityDict[JSONSerializable, JSONSerializable] = utils.IdentityDict()

    def register(self, parent: JSONSerializable, value: Any):
        if isinstance(value, list) or isinstance(value, tuple) \
                or isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if isinstance(value, torch.Tensor) and len(value.shape) == 0:
                self.register(parent, value.item())
            else:
                for val in value:
                    self.register(parent, val)
        elif isinstance(value, dict):
            for val in value.values():
                self.register(parent, val)
        elif isinstance(value, JSONSerializable):
            self.reference_set[value] = self.reference_set[value] + 1
            self.child_parent[value] = parent
            self.parent_children[parent].append(value)
            self.count_complex_objects(value) # TODO add print state in case none of those matches

    def count_complex_objects(self, obj: JSONSerializable):
        params = obj.get_params_as_dict()
        for value in params.values():
            self.register(obj, value)


VARIABLE_MARKER_VALUE = 'VALUE_VARIABLE'
VARIABLE_MARKER_REFERENCE = 'REFERENCE_VARIABLE'


class Variable(JSONSerializable):
    def __init__(self, id: int, delegate: JSONSerializable):
        self.id = id
        self.delegate = delegate

    def get_params_as_dict(self) -> Dict[str, Any]:
        embedding = type_value_embed(self.delegate, self.delegate.get_params_as_dict())
        embedding['id'] = self.id
        return type_value_embed(VARIABLE_MARKER_VALUE, embedding)

    def get_as_reference(self) -> Dict[str, Union[int, str]]:
        return type_value_embed(VARIABLE_MARKER_REFERENCE, {'id': self.id})


# can be applied exactly like this also to a general graph, however as I don't need
# reverse_graph: utils.IdentityDict[JSONSerializable, List[JSONSerializable]], I've left it at the more simple type
# that enforces at most one parent per-node
def topological_sort_forest(forest: utils.IdentityDict[JSONSerializable, List[JSONSerializable]],
                            reverse_forest: utils.IdentityDict[JSONSerializable, JSONSerializable]) \
        -> List[JSONSerializable]:
    """
    Calculates an inverse topological sort for the given forest - children will be first...
    :param forest: The forest from
    :param reverse_forest:
    :return:
    """
    to_inspect = deque([parent for parent in forest.keys() if parent not in reverse_forest])
    visited: utils.DefaultedIdentityDict[JSONSerializable, bool] = utils.DefaultedIdentityDict(lambda: False)
    res = []
    while to_inspect:
        next = to_inspect.popleft()
        if visited[next]:
            continue
        for new_node in forest[next]:
            if not visited[new_node]:
                to_inspect.append(new_node)
        res.append(next)
        visited[next] = True
    return res[::-1]

def recursive_var_replace(val: Any, obj_variable_map: utils.IdentityDict[JSONSerializable, Variable]) -> Any:
    if isinstance(val, dict):
        if TYPE_KEY in val and VALUE_KEY in val:
            return {key: recursive_var_replace(v, obj_variable_map) for key, v in val.items()}
        return type_value_embed(val, {key: recursive_var_replace(v, obj_variable_map) for key, v in val.items()})
        # {'value':{key: recursive_var_replace(v) for key, v in val.items()}, 'type': 'dict'}
    elif isinstance(val, str):
        return val
    elif isinstance(val, JSONSerializable):
        return obj_variable_map[val].get_as_reference()
    elif hasattr(val, 'tolist'):
        # notice that tolist sometimes returns a float for one element tensors/arrays (not sure whether thats numpy or
        # torch), therefore just pass it down and let the iterable code handle that
        return recursive_var_replace(val.tolist(), obj_variable_map)
    else:
        try:
            iterable = iter(val)
        except TypeError:
            return val
        else:
            return type_value_embed(val, [recursive_var_replace(v, obj_variable_map) for v in iterable])

def to_dict_list(variables: List[Variable], obj_variable_map: utils.IdentityDict[JSONSerializable, Variable]):
    res: List[Dict] = []
    for var in variables:
        new_dict = recursive_var_replace(var.get_params_as_dict(), obj_variable_map)
        res.append(new_dict)
    return res

def recursive_serializable_extract(serializable: JSONSerializable):
    d = serializable.get_params_as_dict()
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = {k1: (recursive_serializable_extract(v1) if isinstance(v1, JSONSerializable) else v1)
                      for k1, v1 in v.items()}
        else:
            try:
                res[k] = [(recursive_serializable_extract(v1) if isinstance(v1, JSONSerializable) else v1)
                          for v1 in v]
            except TypeError:
                res[k] = v
    return res


def serialize_dict(serializable: JSONSerializable) -> Dict[str, Any]:
    tree = TreeAnalyzer()
    tree.count_complex_objects(serializable)
    sorted = topological_sort_forest(tree.parent_children, tree.child_parent)
    variables: List[Variable] = [Variable(i, obj) for i, obj in enumerate(sorted)]
    obj_variable_map = utils.IdentityDict.from_iterable([(var.delegate, var) for var in variables])
    root: int = obj_variable_map[sorted[-1]].id
    return {
        'variables': to_dict_list(variables, obj_variable_map),
        'root': root
    }


def serialize_str(serializable: JSONSerializable) -> str:
    res = serialize_dict(serializable)
    return json.dumps(res, indent=4, sort_keys=True)


def serialize(out_path: Union[str, TextIO], serializable: JSONSerializable):
    res = serialize_dict(serializable)
    if isinstance(out_path, str):
        with open(out_path, 'w') as fd:
            json.dump(res, fd, indent=4, sort_keys=True)
    else:
        json.dump(res, out_path, indent=4, sort_keys=True)

from frozendict import frozendict
class FixedJsonSerializable(JSONSerializable):
    def __init__(self, type_string: str, params: Dict[str, Any]):
        self.type_string = type_string
        self.params = frozendict(params)

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.params

    def type_str(self):
        return self.type_string

    def __str__(self) -> str:
        return f'{self.type_string}{self.params}'

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FixedJsonSerializable) and o.type_string == self.type_string and o.params == self.params

    def __hash__(self) -> int:
        return hash(self.type_string) ^ hash(self.params)

# class SklearnSerializableAdaptor(JSONSerializable):
#     def __init__(self, wrapped, update_params: Dict[str, Any]):
#         self.wrapped = wrapped
#         self.update_params = update_params
#
#     def get_params_as_dict(self) -> Dict[str, Any]:
#         res = self.wrapped.get_params(deep=True)
#         res = {k: self.update_params.get(k, v) for k, v in res.items()}
#         while any(map(lambda k: k.count('__') >=2, res.keys())):
#             splits = [(k.rsplit('__', 2), v) for k, v in res.items()]
#             to_modify = None # TODO
#
# def sklearn_to_serializable(obj, update_params: Optional[Dict[str, Any]] = None) -> JSONSerializable:
#     if isinstance(obj, JSONSerializable):
#         return obj
#     else:
#         if update_params is None:
#             update_params = {}
#         return SklearnSerializableAdaptor(obj, update_params)
