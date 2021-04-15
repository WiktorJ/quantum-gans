from typing import Dict, Union, List, Callable, Tuple

import cirq
import sympy
import tensorflow as tf


class CircuitEvaluator:

    def __init__(self, circuit: cirq.Circuit,
                 symbols: Tuple[sympy.Symbol] = None,
                 value_provider: Callable = None,
                 symbol_value_pairs: Union[
                     List[Tuple[float, Dict[sympy.Symbol, float]]], Dict[sympy.Symbol, float]] = None,
                 g_values: List[float] = None) -> None:
        self.value_provider = value_provider
        self.symbols = symbols

        self.circuit = circuit
        if symbols is not None and value_provider is not None and g_values is not None:
            self.symbol_value_pairs = [(1 / (len(g_values)), g, {s: w for s, w in zip(symbols, value_provider(g))}) for
                                       g in
                                       g_values]
        elif isinstance(symbol_value_pairs, dict):
            self.symbol_value_pairs = [(1, 1, symbol_value_pairs)]
        else:
            self.symbol_value_pairs = symbol_value_pairs

    def get_state(self, resolver: cirq.ParamResolver, trace_dims: list = None, max_traced: bool = True):
        state_vector = cirq.final_state_vector(cirq.resolve_parameters(self.circuit, resolver))
        if trace_dims and len(trace_dims) < self.get_circuit_size():
            state_vector = cirq.partial_trace_of_state_vector_as_mixture(state_vector, trace_dims)
            if max_traced:
                prob, state_vector = max(state_vector, key=lambda el: el[0])
                print(f"Max probability state after tracing has probability: {prob}")
                return state_vector, abs(state_vector)
        return state_vector, abs(state_vector)

    def get_state_from_params(self, labels=None, pair=None, trace_dims: list = None, max_traced: bool = True):
        if labels is None and pair is None:
            pair = self.symbol_value_pairs[0][2]
        return self.get_state(self.get_resolver(labels, pair), trace_dims, max_traced)

    def get_all_states_from_params(self, trace_dims: list = None, max_traced: bool = True):
        return [(p, l, *self.get_state_from_params(pair=pair, trace_dims=trace_dims, max_traced=max_traced))
                for p, l, pair in self.symbol_value_pairs]

    def get_resolved_circuit(self, labels=None, pair=None):
        if labels is None and pair is None:
            pair = self.symbol_value_pairs[0][2]
        return cirq.resolve_parameters(self.circuit, self.get_resolver(labels, pair))

    def get_all_resolved_circuits(self):
        return [(p, l, self.get_resolved_circuit(pair=pair)) for p, l, pair in self.symbol_value_pairs]

    def get_resolver(self, labels=None, pair=None):
        if labels is not None and self.value_provider:
            params = {el[0]: el[1] for el in zip(self.symbols, self.value_provider(labels))}
        else:
            params = pair
        return cirq.ParamResolver(params)

    def get_circuit_size(self):
        return len(self.circuit.qid_shape())

    def get_density_matrix(self, modulo: bool = True):
        """
        Returns the density matrix corresponding to the given pure ensemble.

        This function first treats every state as a column vector, then takes the
        outer product of each state with its adjoint.  Each outer product is weighted
        according to its probability in the ensemble.  Then these weighted outer
        products are summed to obtain the corresponding density matrix.

        """
        probs, labels, pure_states, abs_states = zip(*self.get_all_states_from_params())
        if modulo:
            pure_states = abs_states
        expanded_s = tf.expand_dims(pure_states, 1)
        col_s = tf.transpose(expanded_s, [0, 2, 1])
        adj_s = tf.linalg.adjoint(col_s)
        prod = tf.linalg.matmul(col_s, adj_s)
        density_matrix = tf.reduce_sum(
            tf.cast(prod, tf.complex64) * tf.cast(tf.expand_dims(tf.expand_dims(probs, 1), 2), tf.complex64),
            0)
        return density_matrix

    def set_symbol_value_pairs(self,
                               pairs: Union[
                                   List[Tuple[float, any, Dict[sympy.Symbol, float]]], Dict[sympy.Symbol, float]]):
        if isinstance(pairs, dict):
            self.symbol_value_pairs = [(1, 1, pairs)]
        else:
            self.symbol_value_pairs = pairs

cirq.Circuit().all_operations()