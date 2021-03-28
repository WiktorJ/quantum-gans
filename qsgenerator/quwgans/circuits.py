from collections import defaultdict
from typing import List, Dict, Tuple

import cirq


def get_discriminator(circuit: cirq.Circuit):
    return __get_pauli_strings(list(circuit.all_qubits()))


def __get_pauli_strings(qubits: List[cirq.Qid]) -> \
        Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[cirq.PauliString]]]:
    pauli_gates = {cirq.I, cirq.X, cirq.Y, cirq.Z}
    pauli_strings = []
    qubit_to_string_index = defaultdict(list)
    n = len(qubits)
    for i, j in ((x, y) for x in range(n - 1) for y in range(x + 1, n)):
        for p, q in ((p, q) for p in pauli_gates for q in pauli_gates):
            pauli_string = cirq.PauliString([p(qubits[i]), q(qubits[j])])
            pauli_strings.append(pauli_string)
            qubit_to_string_index[qubits[i]].append(len(pauli_strings) - 1)
            qubit_to_string_index[qubits[j]].append(len(pauli_strings) - 1)
    return pauli_strings, qubit_to_string_index
