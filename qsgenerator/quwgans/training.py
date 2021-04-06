import io
import json
from typing import Callable, Tuple, List, Dict

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
from qsgenerator.utils import get_zero_ones_array, \
    get_fidelity_grid, \
    get_generator_fidelity_grid, \
    FidelityGrid, \
    GeneratorsFidelityGrid


class Trainer:

    def __init__(self,
                 real: cirq.Circuit,
                 real_symbols: Tuple[sympy.Symbol],
                 gen: cirq.Circuit,
                 gs: Tuple[sympy.Symbol],
                 g_values: List[float],
                 real_values_provider: Callable,
                 rank: int = 1,
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
        self.rank = rank
        initial_prob = 1 / rank
        self.gen_weights = [
            (tf.Variable(initial_prob, name=f"p{i}"), i, tf.Variable(
                np.array([0] * len(gs)) + np.random.normal(scale=1e-2, size=(len(self.gs),)),
                dtype=tf.float32, name=f"w{i}")) for i in range(rank)]
        self.var_list = [el for t in self.gen_weights for el in t if isinstance(el, tf.Variable)]
        self.g_values = g_values
        self.sampling_repetitions = sampling_repetitions
        self.g_provider = lambda: np.random.choice(self.g_values)
        self.real_values_provider = real_values_provider
        self.gen_evaluator = CircuitEvaluator(self.gen)
        self.real_evaluator = CircuitEvaluator(self.real, real_symbols, real_values_provider, g_values=g_values)
        self.compare_on_fidelity = compare_on_fidelity
        self.use_neptune = use_neptune
        self.last_run_generator_weights = []
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
        final_figures = {}
        for epoch in range(epochs):
            w, h, c = self.find_max_w_h_pairs()
            # TODO: Normalize w per qubit
            em_distance = sum(x * y for x, y in zip(w, c))
            fidelities, gen_fidelities, generated, real = self.get_fidelty_for_real()
            trace_distance = self._get_trace_distance(False)
            abs_trace_distance = self._get_trace_distance(True)
            if plot:
                final_figures = plotter.plot_quwgans(em_distance, trace_distance, abs_trace_distance, fidelities,
                                                     gen_fidelities,
                                                     epoch % snapshot_interval_epochs == 0)

            if epoch % snapshot_interval_epochs == 0:
                print("----------------------------------------------------")
                print(
                    f"Epoch {epoch}:  EM distance = {em_distance}, trace distance = {trace_distance}, abs trace distance = {abs_trace_distance}")
                for i, e in enumerate(c):
                    print(f"h={h[i]}, w={w[i]}, e={e}")
            for step in range(gen_iteration):
                opt.minimize(lambda: self.gen_cost(w, h), self.var_list)
                # probabilities normalization
                s = sum(el[0] for el in self.gen_weights)
                for p, l, _ in self.gen_weights:
                    p.assign(p / s)

            self._update_snapshot(em_distance, fidelities, gen_fidelities, trace_distance, abs_trace_distance,
                                  {k: v for k, v in zip(w, h)}, generated, real, epoch, snapshot_interval_epochs)

        def __json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tf.Variable):
                return obj.numpy()
            if isinstance(obj, WeightSnapshot):
                return obj.__dict__
            if isinstance(obj, FidelityGrid):
                return obj.__dict__
            if isinstance(obj, GeneratorsFidelityGrid):
                return obj.__dict__
            if isinstance(obj, cirq.PauliString):
                return obj.__str__()
            if type(obj) in (str, int, bool, float):
                return obj
            return obj.__repr__()

        print("-------------------------------------")
        print("----------- TRAINING DONE -----------")
        json_result = json.dumps(self.last_run_generator_weights, default=__json_default, indent=2)
        if self.use_neptune:
            self._upload_images_to_neptune(final_figures)
            neptune.log_artifact(io.StringIO(self.gen_evaluator.get_resolved_circuit().to_qasm()), 'gen_qasm.txt')
            neptune.log_artifact(io.StringIO(json_result), 'full_snapshot.json')
        return json_result

    def get_fidelty_for_real(self) -> Tuple[
        List[FidelityGrid],
        List[GeneratorsFidelityGrid],
        List[Tuple[float, any, np.array, np.array]],
        List[Tuple[float, any, np.array, np.array]]]:
        gen_pairs = [
            (weights[0].numpy(), weights[1], {el[0]: float(el[1]) for el in zip(self.gs, weights[2][:].numpy())})
            for weights in self.gen_weights]
        self.gen_evaluator.set_symbol_value_pairs(gen_pairs)
        generated = self.gen_evaluator.get_all_states_from_params()
        real = self.real_evaluator.get_all_states_from_params()
        return get_fidelity_grid(generated, real), get_generator_fidelity_grid(generated), generated, real

    def find_max_w_h_pairs(self):
        c = (self.get_all_generator_expectations(self.disc_hamiltonians).numpy() - self.get_real_expectation(
            self.disc_hamiltonians).numpy()).flatten()
        res = linprog(-c, A_ub=self.A, b_ub=self.b, bounds=(0, 1))

        return [res.x[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5], \
               [self.disc_hamiltonians[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5], \
               [c[i] for i in range(len(self.disc_hamiltonians)) if res.x[i] > 1.e-5]

    def gen_cost(self, max_w, max_h):
        exps = self.get_all_generator_expectations(max_h)
        return tf.reduce_sum([w * exp for w, exp in zip(max_w, exps)])

    def real_distance(self, max_w, max_h):
        exps = self.get_real_expectation(max_h)
        return tf.reduce_sum([w * exp for w, exp in zip(max_w, exps)])

    def get_all_generator_expectations(self, operators: List[cirq.PauliString]):
        expectation = None
        for p, l, theta in self.gen_weights:
            full_weights = tf.keras.layers.Layer()(theta)
            full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
            partial_expectation = p * self.gen_expectation([self.gen], symbol_names=self.gs, symbol_values=full_weights,
                                                           operators=operators)
            if expectation is None:
                expectation = partial_expectation
            else:
                expectation += partial_expectation

        return expectation

    def get_real_expectation(self, operators: List[cirq.PauliString]):
        expectation = None
        for g in self.g_values:
            full_weights = tf.keras.layers.Layer()(
                tf.Variable(np.array(self.real_values_provider(g), dtype=np.float32)))
            full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
            partial_expectation = (1 / (len(self.g_values))) * self.real_expectation([self.real],
                                                                                     symbol_names=self.real_symbols,
                                                                                     symbol_values=full_weights,
                                                                                     operators=operators)
            if expectation is None:
                expectation = partial_expectation
            else:
                expectation += partial_expectation
        return expectation

    def _get_trace_distance(self, modulo: bool = True):
        eigen_values, _ = np.linalg.eig(
            (self.real_evaluator.get_density_matrix(modulo) - self.gen_evaluator.get_density_matrix(modulo)).numpy())
        return sum(abs(np.real(eigen_values))) / 2

    def _get_sampled_expectation(self, expectation):
        return lambda circuit, symbol_names, symbol_values, operators: \
            expectation(circuit,
                        symbol_names=symbol_names,
                        symbol_values=symbol_values,
                        operators=operators,
                        repetitions=self.sampling_repetitions)

    def _update_snapshot(self, em_distance: float, fidelities: List[FidelityGrid],
                         gen_fidelities: List[GeneratorsFidelityGrid], trace_dist: float, abs_trace_dist: float,
                         disc_h_w: Dict[float, cirq.PauliString],
                         generated: List[Tuple[float, any, np.array, np.array]],
                         real: List[Tuple[float, any, np.array, np.array]], epoch: int, snapshot_interval_epochs: int):
        gen_pairs = [
            (weights[0].numpy(), weights[1], {el[0].name: float(el[1]) for el in zip(self.gs, weights[2][:].numpy())})
            for weights in self.gen_weights]

        snap = WeightSnapshot(gen_pairs, em_distance, trace_dist, abs_trace_dist, epoch, fidelities, gen_fidelities,
                              disc_h_w, generated, real, self.compare_on_fidelity)
        self.__update_best_generator_weights(
            snap,
            epoch % snapshot_interval_epochs != 0)
        if self.use_neptune:
            self._upload_metrics_to_neptune(em_distance, fidelities, gen_fidelities, trace_dist, abs_trace_dist)

    def __update_best_generator_weights(self, weight_snapshot: 'WeightSnapshot', replace: bool = True):
        if not replace \
                or not self.last_run_generator_weights \
                or weight_snapshot.is_better_than(self.last_run_generator_weights[-1]):
            if replace and self.last_run_generator_weights:
                self.last_run_generator_weights.pop()
            self.last_run_generator_weights.append(weight_snapshot)

    def _upload_metrics_to_neptune(self, em_distance: float, fidelities: List[FidelityGrid],
                                   gen_fidelities: List[GeneratorsFidelityGrid], trace_dist: float,
                                   abs_trace_dist: float):
        neptune.log_metric("em_distance", em_distance)
        neptune.log_metric("trace_distance", trace_dist)
        neptune.log_metric("abs_trace_distance", abs_trace_dist)
        for f in fidelities:
            neptune.log_metric(f"fidelity (real:gen) ({f.label_real}:{f.label_gen})", f.fidelity)
            neptune.log_metric(f"fidelity modulo (real:gen) ({f.label_real}:{f.label_gen})", f.abs_fidelity)
            neptune.log_metric(f"prob (real - gen) ({f.label_real} - {f.label_gen})", f.prob_real - f.prob_gen)
        for f in gen_fidelities:
            neptune.log_metric(f"fidelity ({f.label_gen1}:{f.label_gen2})", f.fidelity)
            neptune.log_metric(f"fidelity modulo ({f.label_gen1}:{f.label_gen2})", f.abs_fidelity)

    def _upload_images_to_neptune(self, images_dict: Dict):
        def fig2img(fig):
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            return buf

        for k, v in images_dict.items():
            neptune.log_artifact(fig2img(v), f"{k}.png")


class WeightSnapshot(object):

    def __init__(self,
                 gen_pairs: List[Tuple[float, any, Dict[str, float]]],
                 em_distance: float,
                 trace_distance: float,
                 abs_trace_distance: float,
                 epoch: int,
                 fidelities: List[FidelityGrid],
                 abs_fidelities: List[GeneratorsFidelityGrid],
                 disc_w_h: Dict[float, cirq.PauliString],
                 generated: List[Tuple[float, any, np.array, np.array]],
                 real: List[Tuple[float, any, np.array, np.array]],
                 compare_on_trace: bool = True) -> None:
        self.abs_trace_distance = abs_trace_distance
        self.trace_distance = trace_distance
        self.em_distance = em_distance
        self.fidelities = fidelities
        self.abs_fidelities = abs_fidelities
        self.gen_pairs = gen_pairs
        self.disc_w_h = disc_w_h
        self.real = real
        self.generated = generated
        self.epoch = epoch
        self.compare_on_fidelity = compare_on_trace

    def is_better_than(self, other: 'WeightSnapshot') -> bool:
        return self.abs_trace_distance >= other.abs_trace_distance if self.compare_on_fidelity \
            else self.em_distance <= other.em_distance

    def get_serializable_dict(self):
        return self.__dict__
