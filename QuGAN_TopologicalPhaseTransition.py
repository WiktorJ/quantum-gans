from typing import Any

import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import numpy as np
from cirq import GridQubit, ops

cirq.SingleQubitGate

def construct_hamiltonian(l, gzz, gx, gzxz, pbc=True):
    """
    Construct a quantum Hamiltonian as full matrix.

    Reference:
      Adam Smith, Bernhard Jobst, Andrew G. Green, Frank Pollmann
      Crossing a topological phase transition with a quantum computer
      arXiv: 1910.05351
    """
    sigma = [np.array([[1, 0], [0, 1]]),
             np.array([[0, 1], [1, 0]]),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]])]
    # Hamiltonian is actually real-valued
    H = np.zeros((2 ** l, 2 ** l))
    for i in range(1, l):
        H -= gzz * np.kron(np.identity(2 ** (i - 1)),
                           np.kron(sigma[3], np.kron(sigma[3], np.identity(2 ** (l - i - 1)))))
    # external field
    for i in range(1, l + 1):
        H -= gx * np.kron(np.identity(2 ** (i - 1)), np.kron(sigma[1], np.identity(2 ** (l - i))))
    for i in range(1, l - 1):
        H += gzxz * np.kron(np.identity(2 ** (i - 1)),
                            np.kron(sigma[3], np.kron(sigma[1], np.kron(sigma[3], np.identity(2 ** (l - i - 2))))))
    if pbc:
        # periodic boundary conditions
        H -= gzz * np.kron(sigma[3], np.kron(np.identity(2 ** (l - 2)), sigma[3]))
        H += gzxz * np.kron(sigma[3], np.kron(np.identity(2 ** (l - 3)), np.kron(sigma[3], sigma[1])))
        H += gzxz * np.kron(sigma[1], np.kron(sigma[3], np.kron(np.identity(2 ** (l - 3)), sigma[3])))
    return H


l = 3

out_qubit, label_disc, data1, data2, data3, label_gen = cirq.GridQubit.rect(1, 6)

gs = sympy.symbols("g:22")
gen = cirq.Circuit(
    cirq.rx(gs[0]).on(data1),
    cirq.rx(gs[1]).on(data2),
    cirq.rx(gs[2]).on(data3),
    cirq.rx(gs[3]).on(label_gen),
    cirq.rz(gs[4]).on(data1),
    cirq.rz(gs[5]).on(data2),
    cirq.rz(gs[6]).on(data3),
    cirq.rz(gs[7]).on(label_gen),
    cirq.ZZ(data1, data2) ** gs[8],
    cirq.ZZ(data3, label_gen) ** gs[9],
    cirq.ZZ(data2, data3) ** gs[10],
    cirq.rx(gs[11]).on(data1),
    cirq.rx(gs[12]).on(data2),
    cirq.rx(gs[13]).on(data3),
    cirq.rx(gs[14]).on(label_gen),
    cirq.rz(gs[15]).on(data1),
    cirq.rz(gs[16]).on(data2),
    cirq.rz(gs[17]).on(data3),
    cirq.rz(gs[18]).on(label_gen),
    cirq.ZZ(data1, data2) ** gs[19],
    cirq.ZZ(data3, label_gen) ** gs[20],
    cirq.ZZ(data2, data3) ** gs[21],
)

pure_gen = gen.copy()

ds = sympy.symbols("d:28")
disc = cirq.Circuit(
    cirq.rx(ds[0]).on(out_qubit),
    cirq.rx(ds[1]).on(label_disc),
    cirq.rx(ds[2]).on(data1),
    cirq.rx(ds[3]).on(data2),
    cirq.rx(ds[4]).on(data3),
    cirq.rz(ds[5]).on(out_qubit),
    cirq.rz(ds[6]).on(label_disc),
    cirq.rz(ds[7]).on(data1),
    cirq.rz(ds[8]).on(data2),
    cirq.rz(ds[9]).on(data3),
    cirq.ZZ(out_qubit, label_disc) ** ds[10],
    cirq.ZZ(data1, data2) ** ds[11],
    cirq.ZZ(label_disc, data1) ** ds[12],
    cirq.ZZ(data2, data3) ** ds[13],
    cirq.rx(ds[14]).on(out_qubit),
    cirq.rx(ds[15]).on(label_disc),
    cirq.rx(ds[16]).on(data1),
    cirq.rx(ds[17]).on(data2),
    cirq.rx(ds[18]).on(data3),
    cirq.rz(ds[19]).on(out_qubit),
    cirq.rz(ds[20]).on(label_disc),
    cirq.rz(ds[21]).on(data1),
    cirq.rz(ds[22]).on(data2),
    cirq.rz(ds[23]).on(data3),
    cirq.ZZ(out_qubit, label_disc) ** ds[24],
    cirq.ZZ(data1, data2) ** ds[25],
    cirq.ZZ(label_disc, data1) ** ds[26],
    cirq.ZZ(data2, data3) ** ds[27],
)

gen.append([disc])

np.random.seed(0)
eps = 1e-2
init_gen_weights = np.array([np.pi] + [0] * 21) + \
                   np.random.normal(scale=eps, size=(22,))
init_disc_weights = np.random.normal(size=(28,))

gen_weights = tf.Variable(init_gen_weights, dtype=tf.float32)
disc_weights = tf.Variable(init_disc_weights, dtype=tf.float32)

opt = tf.keras.optimizers.Adam(0.001)


class PhaseTransitionFinalStateSimulator(cirq.Simulator):
    def simulate(self, program: 'cirq.Circuit', param_resolver: 'study.ParamResolverOrSimilarType' = None,
                 qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                 initial_state: Any = None) -> 'SimulationTrialResult':
        g = np.random.choice([-1, 1])
        label = [1, 0] if g > 0 else [0, 1]
        gzz = 2 * (1 - g ** 2)
        gx = (1 + g) ** 2
        gzxz = (g - 1) ** 2

        H = construct_hamiltonian(l, gzz, gx, gzxz)
        print('symerr (should be zero):', np.linalg.norm(H - H.conj().T))

        lam, V = np.linalg.eigh(H)

        # ground state wavefunction
        psi = V[:, 0] / np.linalg.norm(V[:, 0])
        psi = np.kron([1, 0], np.kron(label, psi))
        return super().simulate(program, param_resolver, qubit_order, psi)


class PhaseTransitionFinalStateSimulatorGen(cirq.Simulator):
    def simulate(self, program: 'cirq.Circuit', param_resolver: 'study.ParamResolverOrSimilarType' = None,
                 qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                 initial_state: Any = None) -> 'SimulationTrialResult':
        g = np.random.choice([-1, 1])
        label = [1, 0] if g > 0 else [0, 1]
        initial_state = [1, 0]
        for _ in range(l - 1):
            initial_state = np.kron([1, 0], initial_state)
        initial_state = np.kron([1, 0], np.kron(label, np.kron(initial_state, label)))
        return super().simulate(program, param_resolver, qubit_order, initial_state)


def real_disc_circuit_eval(disc_weights):
    # cirq.Simulator().simulate(real)
    return tfq.layers.Expectation(backend=PhaseTransitionFinalStateSimulator())([disc],
                                                                                symbol_names=ds,
                                                                                symbol_values=tf.reshape(disc_weights, (
                                                                                    1, disc_weights.shape[0])),
                                                                                operators=[cirq.Z(out_qubit)])


def gen_disc_circuit_eval(gen_weights, disc_weights):
    full_weights = tf.keras.layers.Concatenate(axis=0)([disc_weights, gen_weights])
    full_weights = tf.reshape(full_weights, (1, full_weights.shape[0]))
    return tfq.layers.Expectation(backend=PhaseTransitionFinalStateSimulatorGen())([gen],
                                                                                   symbol_names=ds + gs,
                                                                                   symbol_values=full_weights,
                                                                                   operators=[cirq.Z(out_qubit)])


def prob_real_true(disc_weights):
    true_disc_output = real_disc_circuit_eval(disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    fake_disc_output = gen_disc_circuit_eval(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true


def disc_cost(disc_weights):
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost


def gen_cost(gen_weights):
    return -prob_fake_true(gen_weights, disc_weights)


def train():
    cost = lambda: disc_cost(disc_weights)
    cost_gen = lambda: gen_cost(gen_weights)
    for epoch in range(20):
        for step in range(20):
            opt.minimize(cost, disc_weights)
            # if step % 5 == 0:
        cost_val = cost().numpy()
        print("Epoch {}: cost = {}".format(epoch, cost_val))

        ##############################################################################
        # At the discriminator’s optimum, the probability for the discriminator to
        # correctly classify the real data should be close to one.

        print("Prob(real classified as real): ", prob_real_true(disc_weights).numpy())

        ##############################################################################
        # For comparison, we check how the discriminator classifies the
        # generator’s (still unoptimized) fake data:

        print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())

        ##############################################################################
        # In the adversarial game we now have to train the generator to better
        # fool the discriminator. For this demo, we only perform one stage of the
        # game. For more complex models, we would continue training the models in an
        # alternating fashion until we reach the optimum point of the two-player
        # adversarial game.

        for step in range(20):
            opt.minimize(cost_gen, gen_weights)
            # if step % 5 == 0:
        cost_val = cost_gen().numpy()
        print("Epoch {}: cost = {}".format(epoch, cost_val))

        ##############################################################################
        # At the optimum of the generator, the probability for the discriminator
        # to be fooled should be close to 1.

        print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())

        ##############################################################################
        # At the joint optimum the discriminator cost will be close to zero,
        # indicating that the discriminator assigns equal probability to both real and
        # generated data.

        print("Discriminator cost: ", disc_cost(disc_weights).numpy())


g = 1
label = [1, 0]
gzz = 2 * (1 - g ** 2)
gx = (1 + g) ** 2
gzxz = (g - 1) ** 2

H = construct_hamiltonian(l, gzz, gx, gzxz)
print('symerr (should be zero):', np.linalg.norm(H - H.conj().T))

lam, V = np.linalg.eigh(H)
# ground state wavefunction
psi = V[:, 0] / np.linalg.norm(V[:, 0])

trained_disc_weights = tf.Variable(np.array([-5.902424, 5.235119, 2.9735384, -4.027759,
                                             -0.45231304, -10.262014, 2.189722, 6.306804,
                                             1.9912083, -13.428224, -9.827148, 0.3823985,
                                             -3.0864358, -9.370758, 8.842436, -8.806886,
                                             7.2321877, 7.3172007, 6.5709624, -15.352012,
                                             -2.5790832, 3.435183, 7.1098614, 7.181435,
                                             -8.872321, -4.213799, -5.463598, -7.8322635]), dtype=tf.float32)

trained_gen_weights = tf.Variable(np.array([4.68485, -5.360671, -36.346577, -5.1716895, -10.068207,
                                            7.2207055, -2.4580982, -36.35788, -1.0866196, 3.1072195,
                                            -36.354927, -36.34182, 2.7561631, -36.35514, -36.35192,
                                            -36.353027, -36.341427, -1.7640233, 4.3496346, -36.364895,
                                            -36.381893, 7.0489244]), dtype=tf.float32)

state_vector = tfq.layers.State()(pure_gen, symbol_names=gs, symbol_values=tf.reshape(trained_gen_weights, (
    1, trained_gen_weights.shape[0])))
print(state_vector)
print(cirq.wavefunction_partial_trace_as_mixture(state_vector, [1, 2, 3]))
