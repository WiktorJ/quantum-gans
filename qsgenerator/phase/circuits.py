import cirq
import sympy
import numpy as np
from cirq.type_workarounds import NotImplementedType
from typing import Union, Iterable


def build_ground_state_circuit(size=None, qubits=None):
    """
    Builds the circuit necessary to generate ground state
    :param size: circuit size excluding boundary qubits
    :param size: qubits circuit of the circuit excluding boundary qubits
    :return: cirq Circuit
    """

    if not size and not qubits:
        raise ValueError("One of the (size, qubits) must be specified")

    if not qubits:
        circuit_size = size + 2
        qubits = cirq.GridQubit.rect(1, circuit_size)
    else:
        q1, qn = cirq.GridQubit.rect(1, 2)
        qubits = [q1] + qubits + [qn]
        circuit_size = len(qubits)

    # theta_v: symbol for V tilda gate parametrization
    # theta_w: symbol for W tilda gate parametrization
    # theta_r: symbol for R tilda gate parametrization
    theta_v, theta_w, theta_r = sympy.symbols("theta_v, theta_w, theta_r")
    circuit = cirq.Circuit()

    circuit.append([build_u1_gate(qubits[0], qubits[1], theta_r)])

    for i in range(1, circuit_size - 1):
        circuit.append(build_u_gate(qubits[i], qubits[i + 1], theta_v, theta_w))

    return circuit, (theta_v, theta_w, theta_r)


def build_u1_gate(q1, q2, theta_r):
    u1 = cirq.Circuit(
        cirq.H(q1),
        cirq.CNOT(q1, q2),
        *_get_r_gate(q2, theta_r)
    )

    # For g > 0
    #     u1 = u1.append([cirq.H(q2), cirq.CNOT(q1, q2), cirq.H(q2)])
    return u1


def build_u_gate(q1, q2, theta_v, theta_w):
    return cirq.Circuit(
        cirq.X(q1),
        *_get_wv_tilda_gate(q2, theta_w),
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        *_get_wv_tilda_transpose_gate(q2, theta_w),
        *_get_wv_tilda_gate(q2, theta_v),
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        *_get_wv_tilda_transpose_gate(q2, theta_v),
    )


def _get_wv_tilda_gate(q, theta):
    return [cirq.ry(theta).on(q)]


def _get_wv_tilda_transpose_gate(q, theta):
    return [cirq.X(q), cirq.ry(theta).on(q), cirq.X(q)]


def _get_r_gate(q, theta):
    return [cirq.Z(q), cirq.ry(theta).on(q)]


class U3(cirq.Gate):
    def __init__(self, theta, phi, lambd):
        self.theta = theta
        self.lambd = lambd
        self.phi = phi
        self.u3 = np.array([
            [sympy.cos(self.theta / 2), -np.e ** (1j * self.lambd) * sympy.sin(self.theta / 2)],
            [np.e ** (1j * self.phi) * sympy.sin(self.theta / 2),
             np.e ** (1j * (self.lambd + self.phi)) * sympy.cos(self.theta / 2)]
        ])

    def _num_qubits_(self) -> int:
        return 1

    def _circuit_diagram_info_(self, _) -> Union[str, Iterable[str],
                                                 cirq.CircuitDiagramInfo]:
        return [f"U3({self.theta, self.phi, self.lambd})"]

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        return self.u3

    def _resolve_parameters_(self, param_resolver) -> 'U3':
        return type(self)(param_resolver.value_of(self.theta),
                          param_resolver.value_of(self.phi),
                          param_resolver.value_of(self.lambd)
                          )

    def transpose(self) -> 'U3':
        self.u3 = self.u3.conj().T
        return self
