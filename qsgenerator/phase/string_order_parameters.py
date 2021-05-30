import cirq
import tensorflow_quantum as tfq


def measure_s1_pauli_string(circuit):
    s1_string = get_s1_pauli_string(circuit)
    print(s1_string)
    return tfq.layers.Expectation()([circuit], operators=s1_string)


def measure_szy_pauli_string(circuit):
    szy_string = get_szy_pauli_string(circuit)
    print(szy_string)
    return tfq.layers.Expectation()([circuit], operators=szy_string)


def get_s1_pauli_string(circuit):
    qubits = sorted(circuit.all_qubits())
    return _get_x_pauli_string(qubits[1:-1])


def get_szy_pauli_string(circuit):
    qubits = sorted(circuit.all_qubits())
    x_string = _get_x_pauli_string(qubits[3:-3])
    return cirq.PauliString(cirq.PauliString([cirq.Z(qubits[1]), cirq.Y(qubits[2])])
                            + x_string
                            + cirq.PauliString([cirq.Y(qubits[-3]), cirq.Z(qubits[-2])]))



def _get_x_pauli_string(qubits):
    return cirq.PauliString([cirq.X(q) for q in qubits])
