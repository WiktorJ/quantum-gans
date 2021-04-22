import neptune
import tensorflow as tf

import io
import cirq
import numpy as np
from qsgenerator import circuits
from qsgenerator.quwgans import circuits as quwgans_circuits
from qsgenerator.quwgans.training import Trainer
from qsgenerator.phase.circuits import PhaseCircuitBuilder
from qsgenerator.evaluators.circuit_evaluator import CircuitEvaluator
from qsgenerator.phase.analitical import get_theta_v, get_theta_w, get_theta_r, get_g_parameters_provider

use_neptune = False
generator_layers = 3
data_bus_size = 5
real_phase = True
generic_generator = True
zxz = False
all_gates_parametrized = False
data_qubits = qubits = cirq.GridQubit.rect(1, data_bus_size)
builder = PhaseCircuitBuilder(all_gates_parametrized=False)
real, real_symbols, symbols_dict_real = builder.build_ground_state_circuit(qubits=data_qubits)
pauli_strings, qubit_to_string_index = quwgans_circuits.get_discriminator(real)
if generic_generator:
    gen, gs = circuits.build_circuit(generator_layers, data_qubits, "g")
    symbols_dict_gen = {}
else:
    builder = PhaseCircuitBuilder(all_gates_parametrized=all_gates_parametrized)
    gen, gs, symbols_dict_gen = builder.build_ground_state_circuit(qubits=data_qubits, full_parametrization=True, zxz=zxz)
g_values = [-0.5]
real_values_provider = get_g_parameters_provider()
opt = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
trainer = Trainer(real,
                  real_symbols,
                  gen,
                  gs,
                  g_values,
                  real_values_provider,
                  use_neptune=use_neptune)
epochs = 2
gen_iteration = 1
snapshot_interval_epochs = 100
json_result = trainer.train(opt, epochs, gen_iteration, snapshot_interval_epochs, plot=True)
gen_evaluator = trainer.gen_evaluator
real_evaluator = trainer.real_evaluator
c = gen_evaluator.get_resolved_circuit()
c.to_qasm()
