from typing import Union, Dict, Set, Tuple

import cirq
import sympy
import numpy as np


def resolve_all(circuit: cirq.Circuit,
                all_symbols: Tuple[sympy.Symbol],
                symbols_dict: Dict[str, Set[sympy.Symbol]],
                to_resolve: Dict[str, Union[np.array, float]]) -> (cirq.Circuit, Tuple[sympy.Symbol]):
    result_circuit = circuit
    not_resolved_symbols = set(all_symbols)
    for name, value in to_resolve.items():
        result_circuit = resolve(result_circuit, symbols_dict[name], value)
        not_resolved_symbols = not_resolved_symbols.difference(symbols_dict[name])
    return result_circuit, tuple(not_resolved_symbols)


def resolve(circuit: cirq.Circuit, symbols: Set[sympy.Symbol], params: Union[np.array, float]) -> cirq.Circuit:
    return cirq.resolve_parameters(
        circuit,
        _get_resolver(symbols, params)
    )


def _get_resolver(symbols: Set[sympy.Symbol], params: Union[np.array, float]) -> cirq.ParamResolver:
    if isinstance(params, float):
        params = [params for _ in range(len(symbols))]
    return cirq.ParamResolver({el[0]: el[1] for el in zip(symbols, params)})
