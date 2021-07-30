import random
import time
from collections import defaultdict
from enum import Enum
from typing import List, Any, Callable, Tuple, Dict, Set

import cirq
import neptune
import sympy
import statistics as st
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from abc import ABC, abstractmethod
from scipy import interpolate
from tensorflow.keras import layers

from qsgenerator.quwgans.circuits import get_discriminator


class RealExpectationsProvider(ABC):

    @abstractmethod
    def get_expectations_for_parameters(self, pauli_strings: Set[cirq.PauliString], parameters: List[Any],
                                        filter_small_expectations: bool = True) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        pass

    @abstractmethod
    def initialize(self, pauli_strings: Set[cirq.PauliString] = None):
        pass


class PrecomputedExpectationsProvider(RealExpectationsProvider):

    def __init__(self,
                 real: cirq.Circuit,
                 real_symbols: Tuple[sympy.Symbol],
                 real_state_parameters: List[Any],
                 real_values_provider: Callable,
                 eps: float = 0.05,
                 sampling_repetitions: int = 1000,
                 use_analytical_expectation: bool = True,
                 gradient_method_provider: Callable = None,
                 ):
        self.real = real
        self.real_symbols = real_symbols
        gradient_method_provider = gradient_method_provider if gradient_method_provider else lambda: tfq.differentiators.ForwardDifference()
        self.eps = eps
        self.sampling_repetitions = sampling_repetitions
        self.real_values_provider = real_values_provider
        # self.pauli_strings_and_indexes = pauli_strings_and_indexes
        # if pauli_strings_and_indexes is not None:
        #     self.pauli_strings = pauli_strings_and_indexes[0]
        #     self.qubit_to_string_index = pauli_strings_and_indexes[1]
        # else:
        #     self.pauli_strings, self.qubit_to_string_index = get_discriminator(real)
        self.real_state_parameters = real_state_parameters
        if use_analytical_expectation:
            self.real_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
            self.gen_expectation = tfq.layers.Expectation(differentiator=gradient_method_provider())
        else:
            self.real_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))
            self.gen_expectation = self._get_sampled_expectation(
                tfq.layers.SampledExpectation(differentiator=gradient_method_provider()))
        self.expectations = {}

    def get_expectations_for_parameters(self,
                                        pauli_strings: Set[cirq.PauliString],
                                        parameters: List[Any] = None,
                                        filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        if parameters is None:
            parameters = self.real_state_parameters

        expectations_dict = defaultdict(dict)
        for param in parameters:
            if param in self.real_state_parameters:
                for pauli_string in pauli_strings:
                    if (param, pauli_string) not in self.expectations:
                        self.expectations[(param, pauli_string)] = \
                            self._get_real_expectation(param, pauli_string)[0][0].numpy()
                    expectations_dict[param][pauli_string] = self.expectations[(param, pauli_string)]

        # expectations_dict = \
        #     {param: {pauli_string: expectation for pauli_string, expectation
        #              in self.expectations.setdefault(param, {ps: e for ps, e in zip(self.pauli_strings,
        #                                                                             [el.numpy() for el in
        #                                                                              self._get_real_expectation(
        #                                                                                  param)[0]])}).items()}
        #      for param in parameters if param in self.real_state_parameters}
        if filter_small_expectations:
            string_with_non_negligible_expectations = set()
            for _, strings_to_exp in expectations_dict.items():
                for pauli_string, exp in strings_to_exp.items():
                    if abs(exp) > self.eps:
                        string_with_non_negligible_expectations.add(pauli_string)
            strings_with_negligible_expectations = set(pauli_strings) - string_with_non_negligible_expectations
            for strings_to_exp in expectations_dict.values():
                for pauli_string in strings_with_negligible_expectations:
                    strings_to_exp.pop(pauli_string, None)
        return expectations_dict

    def get_expectations_for_random_batch(self, pauli_strings: Set[cirq.PauliString], size: int = None,
                                          filter_small_expectations: bool = False) \
            -> Tuple[List[List[float]], List[cirq.PauliString]]:
        if size is None or size >= len(self.real_state_parameters):
            expectations_dict = self.get_expectations_for_parameters(pauli_strings,
                                                                     filter_small_expectations=filter_small_expectations)
        else:
            expectations_dict = self.get_expectations_for_parameters(pauli_strings,
                                                                     random.sample(list(self.real_state_parameters),
                                                                                   min(size, len(
                                                                                       self.real_state_parameters))),
                                                                     filter_small_expectations=filter_small_expectations)
        result = []
        string_used = list(list(expectations_dict.values())[0].keys())
        for _, string_exp in expectations_dict.items():
            exps = []
            for string in string_used:
                exps.append(string_exp[string])
            result.append(exps)
        return result, string_used

    # def get_pauli_strings_and_indexes(self) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
    #     return self.pauli_strings_and_indexes

    def initialize(self, pauli_strings: Set[cirq.PauliString] = None):
        pass

    def _get_real_expectation(self, parameter, pauli_strings):
        full_weights = tf.keras.layers.Layer()(
            tf.Variable(np.array(self.real_values_provider(parameter), dtype=np.float32)))
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
        return self.real_expectation([self.real],
                                     symbol_names=self.real_symbols,
                                     symbol_values=full_weights,
                                     operators=pauli_strings)

    def _get_sampled_expectation(self, expectation):
        return lambda circuit, symbol_names, symbol_values, operators: \
            expectation(circuit,
                        symbol_names=symbol_names,
                        symbol_values=symbol_values,
                        operators=operators,
                        repetitions=self.sampling_repetitions)


class Interpolation1DExpectationsProvider(RealExpectationsProvider):

    def __init__(self, precomputed_expectations_provider: PrecomputedExpectationsProvider):
        self.precomputed_expectations_provider = precomputed_expectations_provider
        self.x = self.precomputed_expectations_provider.real_state_parameters
        self.interpolation_tuples: Dict[cirq.PauliString, Tuple] = {}
        self.expectations = {}

    def get_expectations_for_parameters(self, pauli_strings: Set[cirq.PauliString], parameters: List[Any],
                                        filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        self._update_interpolation_tuples(pauli_strings)
        expectations = defaultdict(dict)
        for param in parameters:
            for pauli_string in pauli_strings:
                if (param, pauli_string) not in self.expectations:
                    self.expectations[(param, pauli_string)] = \
                        interpolate.splev(param, self.interpolation_tuples[ pauli_string], der=0)
                expectations[param][pauli_string] = self.expectations[(param, pauli_string)].tolist()
        return expectations

        # return \
        #     {
        #         param:
        #             {
        #                 pauli_string: expectation.tolist() for pauli_string, expectation in
        #                 self.expectations.setdefault(
        #                     param,
        #                     {
        #                         ps: interpolate.splev(param, self.interpolation_tuples[ps], der=0)
        #                         for ps in pauli_strings
        #                     }).items()
        #             }
        #         for param in parameters
        #     }

    def initialize(self, pauli_strings: Set[cirq.PauliString] = None):
        pass

    def _get_expectations_by_pauli_string_sorted(self, pauli_strings: Set[cirq.PauliString]):
        precomputed_expectations = self.precomputed_expectations_provider.get_expectations_for_parameters(pauli_strings)
        expectations = defaultdict(list)
        for param in sorted(precomputed_expectations.keys()):
            expectation_for_param = precomputed_expectations[param]
            for pauli_string, exp in expectation_for_param.items():
                expectations[pauli_string].append(exp)
        return expectations

    def _update_interpolation_tuples(self, pauli_strings: Set[cirq.PauliString]):
        new_pauli_strings = pauli_strings.difference(self.interpolation_tuples.keys())
        for pauli_string, expectations in self._get_expectations_by_pauli_string_sorted(new_pauli_strings).items():
            self.interpolation_tuples[pauli_string] = interpolate.splrep(self.x, expectations, s=0)

        # return {pauli_string: interpolate.splrep(self.x, expectations, s=0) for pauli_string, expectations in
        #         self._get_expectations_by_pauli_string_sorted(pauli_strings).items()}


class WassersteinGanExpectationProvider(RealExpectationsProvider):

    def __init__(self,
                 precomputed_expectations_provider: PrecomputedExpectationsProvider,
                 gen_input_dim: int,
                 hidden_dim: List[int],
                 penalty_factor: int = 10,
                 epochs: int = 50,
                 batch_size: int = 4,
                 n_crit: int = 5,
                 report_interval_epochs: int = 200,
                 use_neptune: bool = False,
                 filter_small_expectations: bool = True,
                 use_convolutions: bool = False,
                 alpha: float = 0.0001,
                 beta_1: float = 0,
                 beta_2: float = 0.9,
                 seed: int = None):
        self.use_convolutions = use_convolutions
        self.filter_small_expectations = filter_small_expectations
        self.use_neptune = use_neptune
        self.n_crit = n_crit
        self.batch_size = batch_size
        self.epochs = epochs
        self.gen_input_dim = gen_input_dim
        self.hidden_dim = hidden_dim
        self.penalty_factor = penalty_factor
        self.generator_optimizer = tf.keras.optimizers.Adam(alpha, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(alpha, beta_1=beta_1, beta_2=beta_2)
        self.precomputed_expectations_provider = precomputed_expectations_provider
        # self.used_pauli_strings = precomputed_expectations_provider.pauli_strings
        self.used_pauli_strings = None
        self.input_dim = None
        self.discriminator = None
        self.generator = None
        self.expectations = {}
        self.initialized = False
        self.report_interval_epochs = report_interval_epochs
        self.seed = np.random.randint(0, 2 ** 31 - 1) if seed is None else seed

    def get_expectations_for_parameters(self, pauli_strings: Set[cirq.PauliString], parameters: List[Any],
                                        filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        if not self.initialized:
            self.initialize(pauli_strings)
        generated_expectations = {param: self.expectations.setdefault(param,
                                                                      {pauli_string: exp for pauli_string, exp in
                                                                       zip(self.used_pauli_strings,
                                                                           self._get_scaled_generated_vector())})
                                  for param in parameters}
        for param, string_to_expectation in generated_expectations.items():
            for pauli_string in pauli_strings:
                if pauli_string not in string_to_expectation:
                    string_to_expectation[pauli_string] = 0

        return generated_expectations

    def initialize(self, pauli_strings: Set[cirq.PauliString] = None):
        self.initialized = True
        expectations, pauli_strings = self.precomputed_expectations_provider.get_expectations_for_random_batch(
            pauli_strings,
            filter_small_expectations=self.filter_small_expectations)
        self.used_pauli_strings = pauli_strings
        self.input_dim = len(self.used_pauli_strings)
        if self.use_convolutions:
            conv_size = 108
            real_size = len(pauli_strings)
            extra_dims = conv_size - real_size
            for exps in expectations:
                for _ in range(extra_dims):
                    exps.append(0)
            self.input_dim = conv_size

        self.discriminator = self._discriminator()
        self.generator = self._generator()
        dataset = tf.data.Dataset.from_tensor_slices((np.array(expectations, dtype='float32') + 1) / 2)
        # dataset = tf.data.Dataset.from_tensor_slices(np.array(expectations, dtype='float32'))
        dataset = dataset.shuffle(buffer_size=len(self.precomputed_expectations_provider.real_state_parameters)) \
            .batch(self.batch_size)
        start = time.time()
        for epoch in range(self.epochs):
            epoch_gen_losses = []
            epoch_disc_losses = []
            for batch in dataset:
                gen_loss, disc_loss, pen_loss = self.train_step(batch)

                epoch_gen_losses.append(gen_loss)
                epoch_disc_losses.append(disc_loss)
            average_epoch_gen_loss = st.mean(epoch_gen_losses)
            average_epoch_disc_loss = st.mean(epoch_disc_losses)
            if self.use_neptune:
                neptune.log_metric("wgan_gen_loss", average_epoch_gen_loss)
                neptune.log_metric("wgan_disc_loss", average_epoch_disc_loss)
                neptune.log_metric("wgan_pen_loss", average_epoch_disc_loss)
                neptune.log_text("wgan_seed", str(self.seed))
            if epoch % self.report_interval_epochs == 0:
                print(f"Epoch: {epoch}, time for last {self.report_interval_epochs} epochs {time.time() - start}")
                print(f"Last epoch gen loss: {average_epoch_gen_loss}, disc loss: {average_epoch_disc_loss}")
                start = time.time()

    def train_step(self, batch):
        avg_disc_loss = 0
        avg_pen_loss = 0
        for _ in range(self.n_crit):
            noise = tf.random.normal([batch.shape[0], self.gen_input_dim])

            with tf.GradientTape() as disc_tape:
                generated_expectations = self.generator(noise, training=True)
                real_output = self.discriminator(batch, training=True)
                fake_output = self.discriminator(generated_expectations, training=True)

                disc_loss, pen_loss = self._discriminator_loss(real_output, fake_output, batch, generated_expectations)
            avg_disc_loss += (1 / self.n_crit) * disc_loss.numpy()
            avg_pen_loss += (1 / self.n_crit) * pen_loss.numpy()
            gradients_of_discriminator = disc_tape.gradient(pen_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch.shape[0], self.gen_input_dim])
        # noise = tf.random.uniform([batch.shape[0], self.gen_input_dim], minval=-1, maxval=1)
        with tf.GradientTape() as gen_tape:
            generated_expectations = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_expectations, training=True)
            gen_loss = self._generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss.numpy(), avg_disc_loss, avg_pen_loss

    def _generator(self):
        model = tf.keras.Sequential()
        dims = [self.gen_input_dim] + self.hidden_dim
        if self.use_convolutions:
            model.add(layers.Dense(24 * 256,
                                   use_bias=True,
                                   input_shape=(self.gen_input_dim,),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

            model.add(layers.Reshape((24, 256)))

            model.add(layers.Conv1DTranspose(128,
                                             (13,),
                                             (2,),
                                             padding='same',
                                             use_bias=False,
                                             kernel_initializer=tf.keras.initializers.GlorotNormal(
                                                 seed=self.seed)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

            model.add(layers.Conv1DTranspose(64,
                                             (13,),
                                             (2,),
                                             padding='same',
                                             use_bias=False,
                                             kernel_initializer=tf.keras.initializers.GlorotNormal(
                                                 seed=self.seed)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

            # model.add(layers.Conv1DTranspose(64,
            #                                  (13,),
            #                                  (2,),
            #                                  padding='same',
            #                                  use_bias=False,
            #                                  kernel_initializer=tf.keras.initializers.GlorotNormal(
            #                                      seed=self.seed)))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())

            model.add(layers.Conv1DTranspose(1, (13,), (1,), use_bias=False, activation='sigmoid'))
            assert model.output_shape == (None, self.input_dim, 1), print(model.output_shape)
        else:
            for i in range(len(dims) - 1):
                model.add(layers.Dense(dims[i + 1],
                                       use_bias=True,
                                       input_shape=(dims[i],),
                                       kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))
                model.add(layers.BatchNormalization())
                model.add(layers.LeakyReLU())
            model.add(layers.Dense(self.input_dim,
                                   use_bias=True,
                                   input_shape=(dims[-1],),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))
            model.add(layers.Activation(tf.nn.sigmoid))
            # model.add(layers.Activation(tf.nn.tanh))
        return model

    @staticmethod
    def _generator_loss(fake_vector):
        return -tf.reduce_mean(fake_vector)

    def _discriminator(self):
        model = tf.keras.Sequential()
        dims = [self.input_dim] + self.hidden_dim
        if self.use_convolutions:
            model.add(layers.Conv1D(64, (13,), strides=(2,), padding='same', input_shape=(self.input_dim, 1)))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

            model.add(layers.Conv1D(128, (13,), strides=(2,), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

            model.add(layers.Conv1D(256, (13,), strides=(2,), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

            # model.add(layers.Conv1D(512, (13,), strides=(2,), padding='same'))
            # model.add(layers.LeakyReLU())
            # model.add(layers.Dropout(0.3))

            model.add(layers.Flatten())
            model.add(layers.Dense(1))
        else:
            for i in range(len(dims) - 1):
                model.add(layers.Dense(dims[i + 1],
                                       use_bias=True,
                                       input_shape=(dims[i],),
                                       kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))
                model.add(layers.LeakyReLU())
            model.add(layers.Dense(1,
                                   use_bias=True,
                                   input_shape=(dims[-1],),
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(seed=self.seed)))
        return model

    def _discriminator_loss(self, real_vector, fake_vector, real_sample, gen_sample):
        eps = tf.Variable(tf.random.uniform([self.input_dim], minval=0., maxval=1.))
        gen_sample = tf.reshape(gen_sample, [gen_sample.shape[0], gen_sample.shape[1]])
        x_hat = eps * real_sample + (1 - eps) * gen_sample
        with tf.GradientTape() as pen_tape:
            pen_tape.watch(x_hat)
            out = self.discriminator(x_hat, training=True)
        grad = pen_tape.gradient(out, x_hat)
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1))
        grad_penalty = tf.maximum(0, self.penalty_factor * tf.reduce_mean((grad_norm - 1) ** 2))
        base_loss = tf.reduce_mean(fake_vector) - tf.reduce_mean(real_vector)
        return base_loss, base_loss + grad_penalty

    def _get_scaled_generated_vector(self):
        if self.use_convolutions:
            return [el[0] for el in ((self.generator(
                tf.random.uniform([1, self.gen_input_dim], minval=-1, maxval=1)).numpy()[0] * 2) - 1)[
                                    :len(self.used_pauli_strings)]]
        # return self.generator(tf.random.uniform([1, self.gen_input_dim], minval=-1, maxval=1)).numpy()[0]
        return (self.generator(tf.random.uniform([1, self.gen_input_dim], minval=-1, maxval=1)).numpy()[0] * 2) - 1

    # def _get_scaled_generated_vector(self):
    #     if self.use_convolutions:
    #         return [el[0] for el in ((self.generator(tf.random.normal([1, self.gen_input_dim])).numpy()[0] * 2) - 1)[
    #                :len(self.used_pauli_strings)]]
    #     return (self.generator(tf.random.normal([1, self.gen_input_dim])).numpy()[0] * 2) - 1

    # def _get_scaled_generated_vector(self):
    #     if self.use_convolutions:
    #         return [el[0] for el in
    #                 self.generator(tf.random.normal([1, self.gen_input_dim])).numpy()[0][:len(self.used_pauli_strings)]]
    #     return self.generator(tf.random.normal([1, self.gen_input_dim])).numpy()[0]


class ExpectationProviderType(Enum):
    ONLY_KNOWN = 1,
    INTERPOLATION1D = 2,
    WGAN = 3
