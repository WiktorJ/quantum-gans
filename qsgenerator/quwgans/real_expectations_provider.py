import random
import time
from collections import defaultdict
from enum import Enum
from typing import List, Any, Callable, Tuple, Dict

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from abc import ABC, abstractmethod
from scipy import interpolate
from tensorflow.keras import layers

from qsgenerator.quwgans.circuits import get_discriminator


class RealExpectationsProvider(ABC):

    @abstractmethod
    def get_expectations_for_parameters(self, parameters: List[Any], filter_small_expectations: bool = True) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_pauli_strings_and_indexes(self) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
        pass


class PrecomputedExpectationsProvider(RealExpectationsProvider):

    def __init__(self,
                 real: cirq.Circuit,
                 real_symbols: Tuple[sympy.Symbol],
                 real_state_parameters: List[Any],
                 real_values_provider: Callable,
                 pauli_strings_and_indexes: Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]] = None,
                 eps: float = 1.e-5,
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
        self.pauli_strings_and_indexes = pauli_strings_and_indexes
        if pauli_strings_and_indexes is not None:
            self.pauli_strings = pauli_strings_and_indexes[0]
            self.qubit_to_string_index = pauli_strings_and_indexes[1]
        else:
            self.pauli_strings, self.qubit_to_string_index = get_discriminator(real)
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

    def get_expectations_for_parameters(self, parameters: List[Any] = None, filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        if parameters is None:
            parameters = self.real_state_parameters
        expectations_dict = \
            {param: {pauli_string: expectation for pauli_string, expectation
                     in self.expectations.setdefault(param, {ps: e for ps, e in zip(self.pauli_strings,
                                                                                    [el.numpy() for el in
                                                                                     self._get_real_expectation(
                                                                                         param)[0]])}).items()}
             for param in parameters if param in self.real_state_parameters}
        if filter_small_expectations:
            string_with_non_negligible_expectations = set()
            for _, strings_to_exp in expectations_dict.items():
                for pauli_string, exp in strings_to_exp.items():
                    if abs(exp) > self.eps:
                        string_with_non_negligible_expectations.add(pauli_string)
            strings_with_negligible_expectations = set(self.pauli_strings) - string_with_non_negligible_expectations
            for strings_to_exp in expectations_dict.values():
                for pauli_string in strings_with_negligible_expectations:
                    strings_to_exp.pop(pauli_string, None)
        return expectations_dict

    def get_expectations_for_random_batch(self, size: int = None, filter_small_expectations: bool = False) \
            -> Tuple[List[List[float]], List[cirq.PauliString]]:
        if size is None or size >= len(self.real_state_parameters):
            expectations_dict = self.get_expectations_for_parameters(
                filter_small_expectations=filter_small_expectations)
        else:
            expectations_dict = self.get_expectations_for_parameters(
                random.sample(list(self.real_state_parameters), min(size, len(self.real_state_parameters))),
                filter_small_expectations=filter_small_expectations)
        result = []
        string_used = list(list(expectations_dict.values())[0].keys())
        for _, string_exp in expectations_dict.items():
            exps = []
            for string in string_used:
                exps.append(string_exp[string])
            result.append(exps)
        return result, string_used

    def get_pauli_strings_and_indexes(self) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
        return self.pauli_strings_and_indexes

    def initialize(self):
        pass

    def _get_real_expectation(self, parameter):
        full_weights = tf.keras.layers.Layer()(
            tf.Variable(np.array(self.real_values_provider(parameter), dtype=np.float32)))
        full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
        return self.real_expectation([self.real],
                                     symbol_names=self.real_symbols,
                                     symbol_values=full_weights,
                                     operators=self.pauli_strings)

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
        self.interpolation_tuples = None
        self.expectations = {}

    def get_expectations_for_parameters(self, parameters: List[Any], filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        if self.interpolation_tuples is None:
            self.initialize()
        return \
            {
                param:
                    {
                        pauli_string: expectation.tolist() for pauli_string, expectation in
                        self.expectations.setdefault(
                            param,
                            {
                                ps: interpolate.splev(param, self.interpolation_tuples[ps], der=0)
                                for ps in self.precomputed_expectations_provider.pauli_strings
                            }).items()
                    }
                for param in parameters
            }

    def initialize(self):
        self.interpolation_tuples = self._create_interpolation_tuples()

    def get_pauli_strings_and_indexes(self) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
        return self.precomputed_expectations_provider.pauli_strings_and_indexes

    def _get_expectations_by_pauli_string_sorted(self):
        precomputed_expectations = self.precomputed_expectations_provider.get_expectations_for_parameters()
        expectations = defaultdict(list)
        for param in sorted(precomputed_expectations.keys()):
            expectation_for_param = precomputed_expectations[param]
            for pauli_string, exp in expectation_for_param.items():
                expectations[pauli_string].append(exp)
        return expectations

    def _create_interpolation_tuples(self) -> Dict[cirq.PauliString, Tuple]:

        return {pauli_string: interpolate.splrep(self.x, expectations, s=0) for pauli_string, expectations in
                self._get_expectations_by_pauli_string_sorted().items()}


class WassersteinGanExpectationProvider(RealExpectationsProvider):

    def __init__(self,
                 precomputed_expectations_provider: PrecomputedExpectationsProvider,
                 gen_input_dim: int,
                 hidden_dim: List[int],
                 penalty_factor: int = 10,
                 epochs: int = 50,
                 batch_size: int = 4):
        self.batch_size = batch_size
        self.epochs = epochs
        self.gen_input_dim = gen_input_dim
        self.hidden_dim = hidden_dim
        self.penalty_factor = penalty_factor
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.precomputed_expectations_provider = precomputed_expectations_provider
        self.used_pauli_strings = precomputed_expectations_provider.pauli_strings
        self.input_dim = len(self.used_pauli_strings)
        self.discriminator = None
        self.generator = None
        self.expectations = {}
        self.initialized = False

    def get_expectations_for_parameters(self, parameters: List[Any], filter_small_expectations: bool = False) \
            -> Dict[Any, Dict[cirq.PauliString, float]]:
        if not self.initialized:
            self.initialize()
        generated_expectations = {param: self.expectations.setdefault(param,
                                                                      {pauli_string: exp for pauli_string, exp in
                                                                       zip(self.used_pauli_strings,
                                                                           self._get_scaled_generated_vector())})
                                  for param in parameters}
        for param, string_to_expectation in generated_expectations.items():
            for pauli_string in self.precomputed_expectations_provider.pauli_strings:
                if pauli_string not in string_to_expectation:
                    string_to_expectation[pauli_string] = 0

        return generated_expectations

    def initialize(self):
        expectations, pauli_strings = self.precomputed_expectations_provider.get_expectations_for_random_batch(
            filter_small_expectations=True)
        self.used_pauli_strings = pauli_strings
        self.input_dim = len(self.used_pauli_strings)
        self.discriminator = self._discriminator()
        self.generator = self._generator()
        dataset = tf.data.Dataset.from_tensor_slices((np.array(expectations) + 1) / 2)
        dataset = dataset.shuffle(buffer_size=len(self.precomputed_expectations_provider.real_state_parameters)) \
            .batch(self.batch_size)
        for epoch in range(self.epochs):
            start = time.time()

            for batch in dataset:
                self.train_step(batch)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        self.initialized = True

    def get_pauli_strings_and_indexes(self) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
        return self.precomputed_expectations_provider.pauli_strings_and_indexes

    def train_step(self, batch):
        noise = tf.random.normal([batch.shape[0], self.gen_input_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_expectations = self.generator(noise, training=True)
            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_expectations, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output, generated_expectations, batch)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def _generator(self):
        model = tf.keras.Sequential()
        dims = [self.gen_input_dim] + self.hidden_dim
        for i in range(len(dims) - 1):
            model.add(layers.Dense(dims[i + 1], use_bias=True, input_shape=(dims[i],)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        model.add(layers.Dense(self.input_dim, use_bias=True, input_shape=(dims[-1],)))
        model.add(layers.Activation(tf.nn.sigmoid))
        return model

    @staticmethod
    def _generator_loss(fake_vector):
        return -tf.reduce_mean(fake_vector)

    def _discriminator(self):
        model = tf.keras.Sequential()
        dims = [self.input_dim] + self.hidden_dim
        for i in range(len(dims) - 1):
            model.add(layers.Dense(dims[i + 1], use_bias=True, input_shape=(dims[i],)))
            model.add(layers.LeakyReLU())
        model.add(layers.Dense(1, use_bias=True, input_shape=(dims[-1],)))
        return model

    def _discriminator_loss(self, real_vector, fake_vector, real_sample, gen_sample):
        eps = tf.Variable(tf.random.uniform([self.input_dim], minval=0., maxval=1.))
        x_hat = tf.Variable(eps * real_sample + (1 - eps) * gen_sample)
        with tf.GradientTape() as pen_tape:
            out = self.discriminator(x_hat, training=True)
        grad = pen_tape.gradient(out, x_hat)
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1))
        grad_penalty = self.penalty_factor * tf.reduce_mean((grad_norm - 1) ** 2)
        return tf.reduce_mean(fake_vector) - tf.reduce_mean(real_vector) + grad_penalty

    def _get_scaled_generated_vector(self):
        return (self.generator(tf.random.normal([1, self.gen_input_dim])).numpy()[0] * 2) - 1


class ExpectationProviderType(Enum):
    ONLY_KNOWN = 1,
    INTERPOLATION1D = 2,
    WGAN = 3
