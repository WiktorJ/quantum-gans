import math
from typing import Callable, Tuple

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from qsgenerator.phase.analitical import get_theta_v, get_theta_w, get_theta_r
from qsgenerator.phase.simulators import PhaseTransitionFinalStateSimulator
from qsgenerator.utils import map_to_radians


class Trainer:

    def __init__(self,
                 g_provider: Callable,
                 size: int,
                 disc: cirq.Circuit,
                 gen: cirq.Circuit,
                 real: cirq.Circuit,
                 out_qubit: cirq.GridQubit,
                 ds: Tuple[sympy.Symbol],
                 gs: Tuple[sympy.Symbol],
                 real_symbols: Tuple[sympy.Symbol],
                 ls: sympy.Symbol):
        self.real = real
        self.size = size
        self.real_symbols = real_symbols
        self.ls = ls
        self.ds = ds
        self.out_qubit = out_qubit
        self.gen = gen
        self.disc = disc
        self.gs = gs
        self.g_provider = g_provider

    def real_disc_circuit_eval_mocked(self, disc_weights):
        full_weights = tf.keras.layers.Concatenate(axis=0)([
            disc_weights,
            np.array([map_to_radians(self.g_provider())], dtype=np.float32)
        ])

        return tfq.layers.Expectation(backend=PhaseTransitionFinalStateSimulator(self.g_provider, self.size)) \
            ([self.disc],
             symbol_names=self.ds + (self.ls,),
             symbol_values=tf.reshape(
                 full_weights, (
                     1, full_weights.shape[0])),
             operators=[cirq.Z(self.out_qubit)])

    def real_disc_circuit_eval(self, disc_weights):
        # cirq.Simulator().simulate(real)
        full_weights = tf.keras.layers.Concatenate(axis=0)([
            np.array([get_theta_v(self.g_provider()), get_theta_w(self.g_provider()), get_theta_r(self.g_provider())],
                     dtype=np.float32),
            disc_weights,
            np.array([map_to_radians(self.g_provider())], dtype=np.float32)
        ])

        return tfq.layers.Expectation()([self.real],
                                        symbol_names=self.real_symbols + self.ds + (self.ls,),
                                        symbol_values=tf.reshape(full_weights, (
                                            1, full_weights.shape[0])),
                                        operators=[cirq.Z(self.out_qubit)])

    def gen_disc_circuit_eval(self, gen_weights, disc_weights):
        full_weights = tf.keras.layers.Concatenate(axis=0)([
            disc_weights,
            gen_weights,
            np.array([map_to_radians(self.g_provider())], dtype=np.float32)
        ])
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
        return tfq.layers.Expectation()([self.gen],
                                        symbol_names=self.ds + self.gs + (self.ls,),
                                        symbol_values=full_weights,
                                        operators=[cirq.Z(self.out_qubit)])

    def prob_real_true(self, disc_weights):
        true_disc_output = self.real_disc_circuit_eval(disc_weights)
        # convert to probability
        prob_real_true = (true_disc_output + 1) / 2
        return prob_real_true

    def prob_fake_true(self, gen_weights, disc_weights):
        fake_disc_output = self.gen_disc_circuit_eval(gen_weights, disc_weights)
        # convert to probability
        prob_fake_true = (fake_disc_output + 1) / 2
        return prob_fake_true

    def default_disc_cost(self, disc_weights, gen_weights):
        cost = self.prob_fake_true(gen_weights, disc_weights) - self.prob_real_true(disc_weights)
        return cost

    def default_gen_cost(self, disc_weights, gen_weights):
        return -self.prob_fake_true(gen_weights, disc_weights)

    def train(self,
              disc_weights,
              gen_weights,
              opt,
              disc_cost=None,
              gen_cost=None,
              epochs=50,
              disc_iteration=100,
              gen_iteration=2):
        if disc_cost is None:
            disc_cost = self.default_disc_cost
        if gen_cost is None:
            gen_cost = self.default_gen_cost

        for epoch in range(epochs):
            for step in range(disc_iteration):
                opt.minimize(disc_cost(disc_weights, gen_weights), disc_weights)
                # if step % 5 == 0:
            cost_val = disc_cost(disc_weights, gen_weights).numpy()
            print("Epoch {}: cost = {}".format(epoch, cost_val))

            ##############################################################################
            # At the discriminator’s optimum, the probability for the discriminator to
            # correctly classify the real data should be close to one.

            print("Prob(real classified as real): ", self.prob_real_true(disc_weights).numpy())

            ##############################################################################
            # For comparison, we check how the discriminator classifies the
            # generator’s (still unoptimized) fake data:

            print("Prob(fake classified as real): ", self.prob_fake_true(gen_weights, disc_weights).numpy())

            ##############################################################################
            # In the adversarial game we now have to train the generator to better
            # fool the discriminator. For this demo, we only perform one stage of the
            # game. For more complex models, we would continue training the models in an
            # alternating fashion until we reach the optimum point of the two-player
            # adversarial game.

            for step in range(gen_iteration):
                opt.minimize(gen_cost(disc_weights, gen_weights), gen_weights)
                # if step % 5 == 0:
            cost_val = gen_cost(disc_weights, gen_weights).numpy()
            print("Epoch {}: cost = {}".format(epoch, cost_val))

            ##############################################################################
            # At the optimum of the generator, the probability for the discriminator
            # to be fooled should be close to 1.

            print("Prob(fake classified as real): ", self.prob_fake_true(gen_weights, disc_weights).numpy())

            ##############################################################################
            # At the joint optimum the discriminator cost will be close to zero,
            # indicating that the discriminator assigns equal probability to both real and
            # generated data.

            print("Discriminator cost: ", disc_cost(disc_weights, gen_weights).numpy())
            print("Generator weights:", gen_weights)
            print("Discriminator weights", disc_weights)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return max(10 * math.e ** - (step / (self.warmup_steps / math.log(100))), 0.1)


# epochs = 50
# disc_iteration = 100
# gen_iteration = 2
# Trainer(0,0,0,0,0,0,0,0,0).train([],
#       [],
#       0,
#       epochs=epochs,
#       disc_iteration=disc_iteration,
#       gen_iteration=gen_iteration)