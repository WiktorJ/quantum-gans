import numpy as np
import cirq
import sympy


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


def build_ground_state_circuit(g, size):
    """
    Builds the circuit necessary to generate ground state
    :param g: parametrization for the transition curve
    :param size: circuit size excluding boundary qubits
    :return: cirq Circuit
    """

    circuti_size = size + 2
    qubits = cirq.GridQubit.rect(1, circuti_size)
    circuit = cirq.Circuit()

    circuit.append([build_u1_gate(qubits[0], qubits[1], g)])

    for i in range(1, circuti_size - 1):
        circuit.append(build_u_gate(qubits[i], qubits[i + 1], g))

    return circuit


def build_u1_gate(q1, q2, g):
    u1 = cirq.Circuit(
        cirq.H(q1),
        cirq.CNOT(q1, q2),
        cirq.Z(q2),  # R gate from the paper == Z and RY gate
        cirq.ry(_get_theta_r(g)).on(q2)
    )

    if g > 0:
        u1 = u1.append([cirq.H(q2), cirq.CNOT(q1, q2), cirq.H(q2)])
    return u1


def build_u_gate(q1, q2, g):
    return cirq.Circuit(
        cirq.X(q1),
        cirq.ry(_get_theta_w(g)).on(q2),  # W tilda gate from the paper == ry
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        cirq.ry(_get_theta_w(g)).on(q2),
        cirq.ry(_get_theta_v(g)).on(q2),  # V tilda gate from the paper == ry
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        cirq.ry(_get_theta_v(g)).on(q2),
    )


# Methods to construct angles for U1 and U gates from the paper
def _get_theta_v(g):
    return np.arcsin(np.sqrt(np.abs(g)) / np.sqrt(1 + np.abs(g)))


def _get_theta_w(g):
    return np.arccos((np.sign(g) * np.sqrt(np.abs(g))) / np.sqrt(1 + np.abs(g)))


def _get_theta_r(g):
    return 2 * np.arcsin(1 / np.sqrt(1 + np.abs(g)))
