import argparse
import math

import pandas as pd
import decimal
from typing import List, Dict, Union, Optional, Tuple, Mapping

from utils import StoreMultiDict, StoreDictKeyPair


def multirowcell(num_rows: int, text: str, escape_underscore: bool = True) -> str:
    if escape_underscore:
        text = text.replace('_', '$\\_$')
    return f'\\multirowcell{{{num_rows}}}{{{text}}}'


def next_param(i: int, row: Mapping[str, Union[int, str, float, bool]], param_keys: List[str],
               param_dict: Dict[str, str]) -> Tuple[int, Optional[str]]:
    next_param_key = None
    while i < len(param_keys) and next_param_key is None:
        p = param_keys[i]
        if p in row:
            next_param_key = p
        i += 1
    if next_param_key is not None:
        val = row[next_param_key]
        if isinstance(val, float) and math.isfinite(val) and int(val) == val:
            val = int(val)
        return i, param_dict[next_param_key].format(val)
    return i, None


EMPTY_CELL = ' &'
QUANTIZATION_OBJECT = decimal.Decimal('.0001')

def write_metric(metric: str, row_dict: Mapping[str, Union[int, float, str, bool]], num_rows: int,
                 include_batch_stdm: bool, batch_sizes: List[int]) -> str:
    try:
        if metric.startswith('per_data'):
            m_metric = (metric[:-2] + '_mean' + metric[-2:]) if metric.endswith('.0') or metric.endswith('.1') else (metric + '_mean')
            s_metric = (metric[:-2] + '_mstd' + metric[-2:]) if metric.endswith('.0') or metric.endswith('.1') else (metric + '_mstd')
            if m_metric not in row_dict:
                raise ValueError(f'Requested metric {metric} but the mean ({m_metric}) is not in the given row: {row_dict}!')
            if s_metric not in row_dict:
                raise ValueError(f'Requested metric {metric} but the mean standard deviation ({s_metric}) is not in '
                                 f'the given row: {row_dict}!')
            value = decimal.Decimal(row_dict[m_metric]).quantize(QUANTIZATION_OBJECT)
            std = decimal.Decimal(row_dict[s_metric]).quantize(QUANTIZATION_OBJECT)
            return multirowcell(num_rows, f'${str(value)} (\\pm {str(std)})$')
        elif metric.startswith('batch'):
            text = ''
            for i, bs in enumerate(batch_sizes):
                if i>= 1:
                    text+='\\\\'
                suffixed_metric = metric + f'_{bs}'
                if suffixed_metric not in row_dict:
                    raise ValueError(
                        f'Requested metric {suffixed_metric} but the mean ({suffixed_metric}) is not in the given row: {row_dict}!')
                if suffixed_metric + f' (MSTD)' not in row_dict:
                    raise ValueError(f'Requested metric {suffixed_metric} but the mean standard deviation ({metric}_{bs} (MSTD)) is '
                                     f'not in the given row: {row_dict}!')
                value = decimal.Decimal(row_dict[suffixed_metric]).quantize(QUANTIZATION_OBJECT)
                std = decimal.Decimal(row_dict[suffixed_metric + f' (MSTD)']).quantize(QUANTIZATION_OBJECT)
                text+=f'{bs}: ${str(value)} (\\pm {str(std)}'
                if include_batch_stdm:
                    if metric + f'_{bs} (STDM)' not in row_dict:
                        raise ValueError(
                            f'Requested metric {suffixed_metric} but the mean iteration standard deviation ({suffixed_metric} (STDM)) is '
                            f'not in the given row: {row_dict}!')
                    stdm = decimal.Decimal(row_dict[suffixed_metric+" (STDM)"]).quantize(QUANTIZATION_OBJECT)
                    text+=f', {str(stdm)})$'
                else:
                    text+= ')$'
            return multirowcell(num_rows, text)
        elif metric.startswith('total'):
            if metric not in row_dict:
                raise ValueError(f'Requested metric {metric} but it is not in the given row: {row_dict}!')
            value = decimal.Decimal(row_dict[metric]).quantize(QUANTIZATION_OBJECT)
            return multirowcell(num_rows, f'${str(value)}$')
        else:
            RuntimeError(f'Cannot handle metric {metric} as it is of an unknown type')
    except decimal.InvalidOperation:
        return multirowcell(num_rows, f'NaN')

def print_as_latex(query_file: str, method_name: Optional[str], relevant_params: Dict[str, str],
                   relevant_metrics: List[str], include_batch_stdm: bool, batch_sizes: List[int],
                   feature_space_column: str = 'feature_space', end_hline: bool = False, limit: int = 5):
    print(f'Processing {query_file} with relevant params {relevant_params} and metrics {relevant_metrics}')
    df: pd.DataFrame = pd.read_csv(query_file)
    num_out_rows = max(len(relevant_params), 1)
    if any(map(lambda m: m.startswith('batch'), relevant_metrics)):
        num_out_rows = max(num_out_rows, len(batch_sizes))
    if method_name is not None:
        num_out_rows = max(num_out_rows, method_name.count('\\\\')+1)
    param_keys = list(relevant_params.keys())
    decimal.getcontext().prec = 4
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    for r_idx, row in df.iterrows():
        if int(r_idx) >= limit:
            break
        print(f'---------------------------------------------------')
        as_dict: Mapping[str, Union[str, int, float, bool]] = row.to_dict()
        print(f'Processing row {r_idx} with index {as_dict["Unnamed: 0"]}.')
        out_string = ''
        if method_name is not None:
            out_string += multirowcell(num_out_rows, method_name, False) + EMPTY_CELL
        else:
            out_string += ' ' + EMPTY_CELL
        out_string += ' ' + multirowcell(num_out_rows, as_dict[feature_space_column]) + EMPTY_CELL
        i = 0
        i, p_str = next_param(i, as_dict, param_keys, relevant_params)
        if p_str is None:
            out_string += multirowcell(num_out_rows, '-') + EMPTY_CELL
        else:
            out_string += ' ' + p_str + EMPTY_CELL
        for j, metric in enumerate(relevant_metrics):
            out_string += ' ' + write_metric(metric, as_dict, num_out_rows, include_batch_stdm, batch_sizes)
            if j < len(relevant_metrics) - 1:
                out_string+=EMPTY_CELL
        for j in range(1, num_out_rows):
            if p_str is None:
                out_string += '\\\\ \n'
            else:
                out_string += '\\\\ \\cline{3-3}\n'
            out_string += EMPTY_CELL * 2
            if p_str is not None:
                i, p_str = next_param(i, as_dict, param_keys, relevant_params)
                if p_str is None:
                    out_string += multirowcell(num_out_rows-j, '-') + EMPTY_CELL
                else:
                    out_string += ' ' + p_str + EMPTY_CELL
            else:
                out_string += EMPTY_CELL
            out_string += EMPTY_CELL * (len(relevant_metrics) - 1)
        if end_hline:
            out_string += f'\\\\ \\hline\n'
        else:
            out_string += f'\\\\ \\cline{{2-{len(relevant_metrics)+3}}}\n'
        print(out_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query_file', type=str)  # should be the output of queries.py
    parser.add_argument('--method_name', default=None, type=str)
    parser.add_argument('--relevant_params', action=StoreDictKeyPair, nargs="+", default={}, type=str)
    parser.add_argument('--relevant_metrics', nargs="+", default=[], type=str)
    parser.add_argument('--include_batch_stdm', dest='include_batch_stdm', action='store_true')
    parser.add_argument('--exclude_batch_stdm', dest='include_batch_stdm', action='store_false')
    parser.add_argument('--end_hline', dest='end_hline', action='store_true')
    parser.add_argument('--end_cline', dest='end_hline', action='store_false')
    parser.add_argument('--batch_sizes', nargs="+", default=[2,4,8,16], type=int)
    parser.add_argument('--feature_space_column', default='feature_space', type=str)
    parser.add_argument('--limit', dest='limit',  default=5, type=int)
    parser.set_defaults(include_batch_stdm=True, end_hline=False)
    args = parser.parse_args()
    print_as_latex(args.query_file, args.method_name,args.relevant_params, args.relevant_metrics,
                   args.include_batch_stdm, args.batch_sizes, args.feature_space_column, args.end_hline, args.limit)

