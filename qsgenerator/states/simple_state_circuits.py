import cirq
import sympy


def build_x_rotation_state(size=None, qubits=None):
    if not size and not qubits:
        raise ValueError("One of the (size, qubits) must be specified")

    if not qubits:
        qubits = cirq.GridQubit.rect(1, size)
    else:
        size = len(qubits)
    symbols = sympy.symbols(f"r:{size}")
    circuit = cirq.Circuit()

    for i in range(size):
        circuit.append([cirq.rx(symbols[i]).on(qubits[i])])

    return circuit, symbols
