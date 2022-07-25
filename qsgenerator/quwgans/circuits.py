from collections import defaultdict
from typing import List, Dict, Tuple, Set

import cirq

from qsgenerator.phase.string_order_parameters import get_s1_pauli_string, \
    get_szy_pauli_string


def __get_strings_for_index(index: List[int], qubits: List[cirq.Qid]):
    k = len(index)
    non_trivial_pauli_gates = {cirq.X, cirq.Y, cirq.Z}
    pauli_gates = [[el] for el in non_trivial_pauli_gates]
    for _ in range(1, k):
        new_pauli_gates = []
        for gates in pauli_gates:
            for gate in non_trivial_pauli_gates:
                new_pauli_gates.append(gates + [gate])
        pauli_gates = new_pauli_gates
    pauli_strings = set()
    for gates in pauli_gates:
        pauli_string_list = []
        for i, gate in zip(index, gates):
            pauli_string_list.append(gate(qubits[i]))
        pauli_strings.add(cirq.PauliString(pauli_string_list))
    return pauli_strings


def __get_qubit_indexes(n: int, k: int) -> List[List[int]]:
    m = n - k + 1
    indexes = [[el] for el in list(range(m))]
    while m < n:
        m += 1
        new_indexes = []
        for ind in indexes:
            for i in range(ind[-1] + 1, m):
                new_indexes.append(ind + [i])
        indexes = new_indexes
    return indexes


def __get_pauli_strings_for_k(qubits: List[cirq.Qid], k) -> Set[
    cirq.PauliString]:
    pauli_strings = set()
    for index in __get_qubit_indexes(len(qubits), k):
        pauli_strings = pauli_strings.union(
            __get_strings_for_index(index, qubits))
    return pauli_strings


def __get_paul_strings_up_to_k(qubits: List[cirq.Qid], k) -> Set[
    cirq.PauliString]:
    pauli_strings = set()
    for i in range(k):
        pauli_strings = pauli_strings.union(
            __get_pauli_strings_for_k(qubits, i + 1))
    return pauli_strings


def get_discriminator(
        circuit: cirq.Circuit,
        k: int = 2,
        add_phase_strings: bool = False
) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
    return __get_pauli_strings(circuit, k, add_phase_strings)


def __get_pauli_strings(
        circuit: cirq.Circuit,
        k: int = 2,
        add_phase_strings: bool = False
) -> Tuple[List[cirq.PauliString], Dict[cirq.Qid, List[int]]]:
    qubits = list(circuit.all_qubits())
    pauli_strings: Set[cirq.PauliString] = __get_paul_strings_up_to_k(qubits, k)
    if add_phase_strings:
        pauli_strings.add(get_s1_pauli_string(circuit))
        pauli_strings.add(get_szy_pauli_string(circuit))
    pauli_strings_list = []
    qubit_to_string_index = defaultdict(list)
    for pauli_string in pauli_strings:
        pauli_strings_list.append(pauli_string)
        for qubit in pauli_string.keys():
            qubit_to_string_index[qubit].append(len(pauli_strings_list) - 1)

    return pauli_strings_list, qubit_to_string_index

    # non_trivial_pauli_gates = {cirq.X, cirq.Y, cirq.Z}
    # pauli_strings = []
    #
    # n = len(qubits)
    # for i, j, k in ((x, y, z) for x in range(n - 2) for y in range(x + 1, n - 1) for z in range(y + 1, n)):
    #     for p, q, s in ((p, q, s) for p in non_trivial_pauli_gates for q in non_trivial_pauli_gates for s in
    #                     non_trivial_pauli_gates):
    #         pauli_string = cirq.PauliString([p(qubits[i]), q(qubits[j]), s(qubits[k])])
    #         pauli_strings.append(pauli_string)
    #         qubit_to_string_index[qubits[i]].append(len(pauli_strings) - 1)
    #         qubit_to_string_index[qubits[j]].append(len(pauli_strings) - 1)
    #         qubit_to_string_index[qubits[k]].append(len(pauli_strings) - 1)
    #
    # for i, j in ((x, y) for x in range(n - 1) for y in range(x + 1, n)):
    #     for p, q in ((p, q) for p in non_trivial_pauli_gates for q in non_trivial_pauli_gates):
    #         pauli_string = cirq.PauliString([p(qubits[i]), q(qubits[j])])
    #         pauli_strings.append(pauli_string)
    #         qubit_to_string_index[qubits[i]].append(len(pauli_strings) - 1)
    #         qubit_to_string_index[qubits[j]].append(len(pauli_strings) - 1)
    # for q in qubits:
    #     for g in non_trivial_pauli_gates:
    #         pauli_string = cirq.PauliString([g(q)])
    #         pauli_strings.append(pauli_string)
    #         qubit_to_string_index[q].append(len(pauli_strings) - 1)
    # return pauli_strings, qubit_to_string_index
