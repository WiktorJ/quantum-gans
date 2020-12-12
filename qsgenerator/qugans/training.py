from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple, Iterable

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from qsgenerator.plotting.Plotter import Plotter
from qsgenerator.utils import map_to_radians


class Trainer:

    def __init__(self,
                 g_values: Iterable,
                 size: int,
                 disc: cirq.Circuit,
                 gen: cirq.Circuit,
                 real: cirq.Circuit,
                 out_qubit: cirq.GridQubit,
                 ds: Tuple[sympy.Symbol],
                 gs: Tuple[sympy.Symbol],
                 real_symbols: Tuple[sympy.Symbol],
                 ls: sympy.Symbol,
                 real_values_provider: Callable,
                 label_value_provider: Callable = None,
                 use_analytical_expectation=False,
                 sampling_repetitions=500,
                 gradient_method_provider=None):
        gradient_method_provider = gradient_method_provider if gradient_method_provider is not None else lambda: tfq.differentiators.ForwardDifference()
        self.label_value_provider = label_value_provider if label_value_provider is not None else map_to_radians
        self.real_values_provider = real_values_provider
        self.sampling_repetitions = sampling_repetitions
        self.use_analytical_expectation = use_analytical_expectation
        self.real = real
        self.size = size
        self.real_symbols = real_symbols
        self.ls = ls
        self.ds = ds
        self.out_qubit = out_qubit
        self.gen = gen
        self.disc = disc
        self.gs = gs
        self.g_values = list(g_values)
        self.g_provider = lambda: np.random.choice(self.g_values)
        self.last_run_generator_weights = []
        if use_analytical_expectation:
            self.disc_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
            self.gen_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
        else:
            self.disc_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))
            self.gen_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))

    def real_disc_circuit_eval(self, disc_weights, g=None):
        # cirq.Simulator().simulate(real)
        if not g:
            g = self.g_provider()
        full_weights = tf.keras.layers.Concatenate(axis=0)([
            np.array(self.real_values_provider(g),
                     dtype=np.float32),
            disc_weights,
            np.array([self.label_value_provider(g)], dtype=np.float32)
        ])
        return self.disc_expectation([self.real],
                                     symbol_names=self.real_symbols + self.ds + (self.ls,),
                                     symbol_values=tf.reshape(full_weights, (
                                         1, full_weights.shape[0])),
                                     operators=[cirq.Z(self.out_qubit)])

    def gen_disc_circuit_eval(self, gen_weights, disc_weights, g=None):
        if not g:
            g = self.g_provider()
        full_weights = tf.keras.layers.Concatenate(axis=0)([
            disc_weights,
            gen_weights,
            np.array([self.label_value_provider(g)], dtype=np.float32)
        ])
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))

        return self.gen_expectation([self.gen],
                                    symbol_names=self.ds + self.gs + (self.ls,),
                                    symbol_values=full_weights,
                                    operators=[cirq.Z(self.out_qubit)])

    def _get_sampled_expectation(self, expectation):
        return lambda circuit, symbol_names, symbol_values, operators: \
            expectation(circuit,
                        symbol_names=symbol_names,
                        symbol_values=symbol_values,
                        operators=operators,
                        repetitions=self.sampling_repetitions)

    def prob_real_true(self, disc_weights, g=None):
        true_disc_output = self.real_disc_circuit_eval(disc_weights, g)
        # convert to probability
        prob_real_true = (true_disc_output + 1) / 2
        return prob_real_true

    def prob_fake_true(self, gen_weights, disc_weights, g=None):
        fake_disc_output = self.gen_disc_circuit_eval(gen_weights, disc_weights, g)
        # convert to probability
        prob_fake_true = (fake_disc_output + 1) / 2
        return prob_fake_true

    def default_disc_cost(self, disc_weights, gen_weights, g=None):
        cost = self.prob_fake_true(gen_weights, disc_weights) - self.prob_real_true(disc_weights, g)
        return cost

    def default_gen_cost(self, disc_weights, gen_weights, g=None):
        return -self.prob_fake_true(gen_weights, disc_weights, g)

    def train(self,
              disc_weights,
              gen_weights,
              opt,
              disc_cost=None,
              gen_cost=None,
              epochs=100,
              disc_iteration=100,
              gen_iteration=2,
              snapshot_interval_epochs=20,
              print_weights=False):
        if disc_cost is None:
            disc_cost = lambda: self.default_disc_cost(disc_weights, gen_weights)
        if gen_cost is None:
            gen_cost = lambda: self.default_gen_cost(disc_weights, gen_weights)
        self.last_run_generator_weights = []
        plotter = Plotter()
        gen_cost_val = -0.5

        for epoch in range(epochs):
            for step in range(disc_iteration):
                opt.minimize(disc_cost, disc_weights)
            disc_cost_val = disc_cost().numpy()[0][0]

            prob_fake_real, prob_real_real = self.__get_probs_and_update_snapshots(
                gen_weights,
                disc_weights,
                epoch,
                snapshot_interval_epochs,
                TrainingPhaseLabel.DISCRIMINATOR
            )
            plotter.on_epoch_end(disc_cost_val, gen_cost_val, prob_fake_real, prob_real_real)
            if epoch % snapshot_interval_epochs == 0:
                print("----------------------------------------------------")
                print("----------- AFTER DISCRIMINATOR TRAINING -----------")
                print("Epoch {}: generator cost = {}".format(epoch, gen_cost().numpy()))
                print("Epoch {}: discriminator cost = {}".format(epoch, disc_cost_val))

                ##############################################################################
                # For comparison, we check how the discriminator classifies the
                # generator’s (still unoptimized) fake data:

                print("Prob(fake classified as real): ", prob_fake_real)

                ##############################################################################
                # At the discriminator’s optimum, the probability for the discriminator to
                # correctly classify the real data should be close to one.

                print("Prob(real classified as real): ", prob_real_real)

            ##############################################################################
            # In the adversarial game we now have to train the generator to better
            # fool the discriminator. For this demo, we only perform one stage of the
            # game. For more complex models, we would continue training the models in an
            # alternating fashion until we reach the optimum point of the two-player
            # adversarial game.

            for step in range(gen_iteration):
                opt.minimize(gen_cost, gen_weights)
            gen_cost_val = gen_cost().numpy()[0][0]

            prob_fake_real, prob_real_real = self.__get_probs_and_update_snapshots(
                gen_weights,
                disc_weights,
                epoch,
                snapshot_interval_epochs,
                TrainingPhaseLabel.GENERATOR
            )

            plotter.on_epoch_end(disc_cost_val, gen_cost_val, prob_fake_real, prob_real_real)
            if epoch % snapshot_interval_epochs == 0:
                print("----------- AFTER GENERATOR TRAINING -----------")
                print("Epoch {}: generator cost = {}".format(epoch, gen_cost_val))
                ##############################################################################
                # At the joint optimum the discriminator cost will be close to zero,
                # indicating that the discriminator assigns equal probability to both real and
                # generated data.
                print("Epoch {}: discriminator cost = {}".format(epoch, disc_cost().numpy()))

                ##############################################################################
                # At the optimum of the generator, the probability for the discriminator
                # to be fooled should be close to 1.
                print("Prob(fake classified as real): ", prob_fake_real)
                print("Prob(real classified as real): ", prob_real_real)

                if print_weights:
                    print("Generator weights:", gen_weights)
                    print("Discriminator weights", disc_weights)

        def __json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tf.Variable):
                return obj.numpy()
            if isinstance(obj, WeightSnapshot):
                return obj.__dict__
            if isinstance(obj, TrainingPhaseLabel):
                return obj.name
            elif type(obj) in (str, int, bool, float):
                return obj
            else:
                return obj.__repr__()

        print("-------------------------------------")
        print("----------- TRAINING DONE -----------")
        print("Weights:", json.dumps(self.last_run_generator_weights, default=__json_default, indent=2))
        return self.last_run_generator_weights

    def __update_best_generator_weights(self, weight_snapshot: WeightSnapshot, replace: bool = True):
        if not replace \
                or not self.last_run_generator_weights \
                or weight_snapshot.is_better_than(self.last_run_generator_weights[-1]):
            if replace and self.last_run_generator_weights:
                self.last_run_generator_weights.pop()
            self.last_run_generator_weights.append(weight_snapshot)

    def __get_probs_and_update_snapshots(self, gen_weights: np.array, disc_weights: np.array, epoch: int,
                                         snapshot_interval_epochs: int, label: TrainingPhaseLabel):
        prob_fake_real = statistics.mean(
            [self.prob_fake_true(gen_weights, disc_weights, g).numpy()[0][0] for g in self.g_values])
        prob_real_real = statistics.mean([self.prob_real_true(disc_weights, g).numpy()[0][0] for g in self.g_values])

        self.__update_best_generator_weights(
            WeightSnapshot(gen_weights, disc_weights, prob_fake_real, prob_real_real, epoch, label),
            epoch % snapshot_interval_epochs != 0)

        return prob_fake_real, prob_real_real


class TrainingPhaseLabel(Enum):
    GENERATOR = 1
    DISCRIMINATOR = 2


@dataclass
class WeightSnapshot(object):
    gen_weights: np.array
    disc_weights: np.array
    prob_fake_real: float
    prob_real_real: float
    epoch: int
    label: TrainingPhaseLabel

    def is_better_than(self, other: WeightSnapshot) -> bool:
        return self.prob_fake_real >= other.prob_fake_real
