from typing import Tuple

import cirq
import sympy
import numpy as np


class CircuitEvaluator:

    def __init__(self, circuit, symbols) -> None:
        self.symbols = symbols
        self.circuit = circuit

    def get_state(self, resolver: cirq.ParamResolver, trace_dims: list = None, max_traced: bool = True):
        state_vector = cirq.final_state_vector(cirq.resolve_parameters(self.circuit, resolver))
        if trace_dims:
            state_vector = cirq.partial_trace_of_state_vector_as_mixture(state_vector, trace_dims)
            if max_traced:
                prob, state_vector = max(state_vector)
                print(f"Max probability state after tracing has probability: {prob}")
        return state_vector

    def get_state_from_params(self, params: np.array, trace_dims: list = None, max_traced: bool = True):
        return self.get_state(self.get_resolver(params), trace_dims, max_traced)

    def get_resolver(self, params: np.array):
        return cirq.ParamResolver({el[0]: el[1] for el in zip(self.symbols, params)})


