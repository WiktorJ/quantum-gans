from typing import Dict, Union, List, Callable, Tuple

import cirq
import copy
import numpy as np
import sympy


class CircuitEvaluator:

    def __init__(self, circuit: cirq.Circuit,
                 label_symbols: Tuple[sympy.Symbol],
                 label_value_provider: Callable = None,
                 symbol_value_pairs: Union[Dict[sympy.Symbol, float], Dict[str, float]] = None,
                 symbols: Tuple[sympy.Symbol] = None) -> None:
        self.label_value_provider = label_value_provider
        if symbols:
            symbol_value_pairs = {symbol: symbol_value_pairs[symbol.name] for symbol in symbols}
        self.symbol_value_pairs = symbol_value_pairs if symbol_value_pairs else {}
        self.label_symbols = label_symbols
        self.circuit = circuit

    def get_state(self, resolver: cirq.ParamResolver, trace_dims: list = None, max_traced: bool = True):
        state_vector = cirq.final_state_vector(cirq.resolve_parameters(self.circuit, resolver))
        if trace_dims:
            state_vector = cirq.partial_trace_of_state_vector_as_mixture(state_vector, trace_dims)
            if max_traced:
                prob, state_vector = max(state_vector, key=lambda el: el[0])
                print(f"Max probability state after tracing has probability: {prob}")
        return state_vector

    def get_state_from_params(self, labels=None, trace_dims: list = None, max_traced: bool = True):
        return self.get_state(self.get_resolver(labels), trace_dims, max_traced)

    def get_resolved_circuit(self, labels=None):
        return cirq.resolve_parameters(self.circuit, self.get_resolver(labels))

    def get_resolver(self, labels=None):
        if labels is not None and self.label_value_provider:
            params = copy.deepcopy(self.symbol_value_pairs)
            for symbol, value in zip(self.label_symbols, self.label_value_provider(labels)):
                params[symbol] = value
        else:
            params = self.symbol_value_pairs
        return cirq.ParamResolver(params)
