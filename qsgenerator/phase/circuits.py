import cirq
import sympy
import numpy as np
from cirq.type_workarounds import NotImplementedType
from typing import Union, Iterable


def build_ground_state_circuit(size=None, qubits=None, full_circuit=True, full_parametrization=False, zxz=False):
    """
    Builds the circuit necessary to generate ground state
    :param size: circuit size
    :param qubits: qubits to create the SPT circuit
    :param full_circuit: if False boundary qubits will be added, if True the provided are used
    :param full_parametrization: if True all parametrized gates get separate symbol
    :return: cirq Circuit
    """

    if not size and not qubits:
        raise ValueError("One of the (size, qubits) must be specified")

    circuit_size = len(qubits) if qubits else size
    if not full_circuit:
        circuit_size += 2

    if (size and size < circuit_size) or (qubits and len(qubits) < circuit_size):
        raise ValueError(f"At least 5 qubits are required, but the size is only {circuit_size}.")

    if not qubits:
        qubits = cirq.GridQubit.rect(1, circuit_size)
    elif not full_circuit:
        qubits = cirq.GridQubit.rect(1, 1, 1, qubits[0].col) + qubits + cirq.GridQubit.rect(1, 1, 1, qubits[-1].col)

    # theta_v: symbol for V tilda gate parametrization
    # theta_w: symbol for W tilda gate parametrization
    # theta_r: symbol for R tilda gate parametrization

    if not full_parametrization:
        thetas = sympy.symbols("theta_r, theta_v, theta_w")
    else:
        size = ((circuit_size - 2) * 4) + 1
        if zxz:
            size *= 3
        thetas = sympy.symbols(f"theta:{size}")

    circuit = cirq.Circuit()

    multiplier = 3 if zxz else 1
    y_supplier = _zxz_ry if zxz else _default_ry

    circuit.append([build_u1_gate(qubits[0], qubits[1], thetas[0:multiplier], y_supplier)])
    base = 0
    for i in range(0, circuit_size - 2):
        if full_parametrization:
            circuit.append(
                build_u_gate(qubits[i + 1],
                             qubits[i + 2],
                             thetas[multiplier + base: (2 * multiplier) + base],
                             thetas[(2 * multiplier) + base: (3 * multiplier) + base],
                             thetas[(3 * multiplier) + base: (4 * multiplier) + base],
                             thetas[(4 * multiplier) + base: (5 * multiplier) + base], y_supplier))
            base += (4 * multiplier)
        else:
            circuit.append(
                build_u_gate(qubits[i + 1], qubits[i + 2], thetas[1:2], thetas[1:2], thetas[2:3], thetas[2:3],
                             _default_ry))

    return circuit, thetas


def build_u1_gate(q1, q2, theta_r, y_supplier):
    u1 = cirq.Circuit(
        cirq.H(q1),
        cirq.CNOT(q1, q2),
        *_get_r_gate(q2, theta_r, y_supplier)
    )

    # For g > 0
    #     u1 = u1.append([cirq.H(q2), cirq.CNOT(q1, q2), cirq.H(q2)])
    return u1


def build_u_gate(q1, q2, theta_v, theta_vt, theta_w, theta_wt, ry_supplier):
    return cirq.Circuit(
        cirq.X(q1),
        *_get_wv_tilda_gate(q2, theta_w, ry_supplier),
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        *_get_wv_tilda_transpose_gate(q2, theta_wt, ry_supplier),
        *_get_wv_tilda_gate(q2, theta_v, ry_supplier),
        cirq.CNOT(q1, q2),
        cirq.X(q1),
        *_get_wv_tilda_transpose_gate(q2, theta_vt, ry_supplier),
    )


def _get_wv_tilda_gate(q, theta, ry_supplier):
    return ry_supplier(q, theta)


def _get_wv_tilda_transpose_gate(q, theta, ry_supplier):
    return [cirq.X(q)] + ry_supplier(q, theta) + [cirq.X(q)]


def _get_r_gate(q, theta, ry_supplier):
    return [cirq.Z(q)] + ry_supplier(q, theta)


def _default_ry(q, theta):
    return [cirq.ry(theta[0]).on(q)]


def _zxz_ry(q, thetas):
    return [cirq.rz(thetas[0]).on(q), cirq.rx(thetas[1]).on(q), cirq.rz(thetas[2]).on(q)]


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
