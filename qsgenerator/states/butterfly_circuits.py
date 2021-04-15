import cirq
import sympy


class ButterflyCircuitBuilder:

    def __init__(self):
        self.thetas = ()

    def build(self, size=None, qubits=None):
        """

        :param size: circuit size
        :param qubits: qubits to create the SPT circuit
        :return:
        """
        if not size and not qubits:
            raise ValueError("One of the (size, qubits) must be specified")

        if not qubits:
            qubits = cirq.GridQubit.rect(1, size)

        circuit = cirq.Circuit()

        skip_factor = 1

        while (skip_factor - 1) ** 2 < len(qubits):
            circuit.append(self.__build_simple_layer(qubits))
            circuit.append(self.__build_mixing_layer(qubits, skip_factor))
            skip_factor += 1

        return circuit, self.thetas

    def __build_simple_layer(self, qubits):
        new_thetas = sympy.symbols(f"theta{len(self.thetas)}:{len(self.thetas) + len(qubits)}")
        self.thetas += new_thetas
        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            circuit.append([cirq.rx(new_thetas[i]).on(q)])
        return circuit

    def __build_mixing_layer(self, qubits, skip_factor):
        curent_skip = skip_factor
        current_qubit_index = 0
        circuit = cirq.Circuit()
        while current_qubit_index + skip_factor < len(qubits):
            while current_qubit_index + skip_factor < len(qubits) and curent_skip > 0:
                self.thetas += (sympy.symbols(f"theta{len(self.thetas)}"),)
                circuit.append([cirq.CXPowGate(exponent=self.thetas[-1]).on(qubits[current_qubit_index],
                                                                            qubits[current_qubit_index + skip_factor])])
                curent_skip -= 1
                current_qubit_index += 1
            current_qubit_index += skip_factor
            curent_skip = skip_factor
        return circuit


