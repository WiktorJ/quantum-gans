from typing import Iterable, Callable, Tuple, List

import cirq
import neptune
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from scipy.optimize import linprog
from qsgenerator.evaluators.circuit_evaluator import CircuitEvaluator
from qsgenerator.plotting.Plotter import Plotter
from qsgenerator.quwgans.circuits import get_discriminator
from qsgenerator.utils import get_zero_ones_array


class Trainer:

    def __init__(self,
                 real: cirq.Circuit,
                 real_symbols: Tuple[sympy.Symbol],
                 gen: cirq.Circuit,
                 gs: Tuple[sympy.Symbol],
                 g_values: Iterable,
                 real_values_provider: Callable,
                 use_analytical_expectation: bool = True,
                 gradient_method_provider: Callable = None,
                 sampling_repetitions: int = 500,
                 use_neptune=False,
                 compare_on_fidelity=True):
        gradient_method_provider = gradient_method_provider if gradient_method_provider else lambda: tfq.differentiators.ForwardDifference()
        self.size = len(real.all_qubits())
        self.real = real
        self.real_symbols = real_symbols
        self.gs = gs
        self.gen = gen
        self.disc_hamiltonians, self.qubit_to_string_index = get_discriminator(self.real)
        self.A = np.array([get_zero_ones_array(len(self.disc_hamiltonians), indices) for indices in
                           self.qubit_to_string_index.values()])
        self.b = np.ones(len(self.qubit_to_string_index))
        self.gen_weights = tf.Variable(np.random.normal(scale=1e-2, size=(len(self.gs),)), dtype=tf.float32)
        self.g_values = g_values
        self.sampling_repetitions = sampling_repetitions
        self.g_provider = lambda: np.random.choice(self.g_values)
        self.real_values_provider = real_values_provider
        self.gen_evaluator = CircuitEvaluator(self.gen)
        self.real_evaluator = CircuitEvaluator(self.real, real_symbols, real_values_provider)
        self.compare_on_fidelity = compare_on_fidelity
        self.use_neptune = use_neptune
        if use_analytical_expectation:
            self.real_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
            self.gen_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
        else:
            self.real_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))
            self.gen_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))

    def train(self,
              opt,
              epochs=100,
              gen_iteration=2,
              snapshot_interval_epochs=20,
              plot=True):

        plotter = Plotter()

        for epoch in range(epochs):
            w, h, c = self.find_max_w_h_pairs()
            # TODO: Normalize w per qubit
            em_distance = sum(x * y for x, y in zip(w, c))
            states_and_fidelity = [(el, self.get_states_and_fidelty_for_real(el)) for el in self.g_values]
            fidelities = {g: s[2] for g, s in states_and_fidelity}
            abs_fidelities = {g: s[3] for g, s in states_and_fidelity}

            if plot:
                plotter.plot_quwgans(em_distance, fidelities, abs_fidelities, epoch % snapshot_interval_epochs == 0)

            if epoch % snapshot_interval_epochs == 0:
                print("----------------------------------------------------")
                print(f"Epoch {epoch}:  distance = {em_distance}")
                for i, e in enumerate(c):
                    print(f"h={h[i]}, w={w[i]}, e={e}")

                for k, v in fidelities.items():
                    print(f"Fidelity for g={k} : {v[0]}")
                for k, v in abs_fidelities.items():
                    print(f"Fidelity modulo for g={k} : {v[1]}")

            for step in range(gen_iteration):
                opt.minimize(lambda: self.gen_cost(w, h), self.gen_weights)

    def get_states_and_fidelty_for_real(self, g):
        gen_pairs = {el[0]: el[1] for el in zip(self.gs, self.gen_weights[:].numpy())}
        self.gen_evaluator.symbol_value_pairs = gen_pairs
        generated = self.gen_evaluator.get_state_from_params(trace_dims=list(range(self.size)))
        real = self.real_evaluator.get_state_from_params(g)
        return generated, real, cirq.fidelity(generated, real), cirq.fidelity(abs(generated), abs(real))

    def find_max_w_h_pairs(self):
        c = (self.get_generator_expectation(self.disc_hamiltonians).numpy() - self.get_real_expectation(
            self.disc_hamiltonians).numpy()).flatten()
        res = linprog(-c, A_ub=self.A, b_ub=self.b, bounds=(0, 1))

        return [res.x[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5], \
               [self.disc_hamiltonians[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5], \
               [c[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5]

    def gen_cost(self, max_w, max_h):
        exps = self.get_generator_expectation(max_h)
        return tf.reduce_sum([w * exp for w, exp in zip(max_w, exps)])

    def real_distance(self, max_w, max_h):
        exps = self.get_real_expectation(max_h)
        return tf.reduce_sum([w * exp for w, exp in zip(max_w, exps)])

    def get_generator_expectation(self, operators: List[cirq.PauliString]):
        full_weights = tf.keras.layers.Layer()(self.gen_weights)
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
        return self.gen_expectation([self.gen], symbol_names=self.gs, symbol_values=full_weights, operators=operators)

    def get_real_expectation(self, operators: List[cirq.PauliString], g: float = None):
        if not g:
            g = self.g_provider()
        full_weights = tf.keras.layers.Layer()(tf.Variable(np.array(self.real_values_provider(g), dtype=np.float32)))
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
        return self.real_expectation([self.real], symbol_names=self.real_symbols, symbol_values=full_weights,
                                     operators=operators)

    def _get_sampled_expectation(self, expectation):
        return lambda circuit, symbol_names, symbol_values, operators: \
            expectation(circuit,
                        symbol_names=symbol_names,
                        symbol_values=symbol_values,
                        operators=operators,
                        repetitions=self.sampling_repetitions)
