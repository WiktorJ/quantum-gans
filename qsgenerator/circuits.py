import cirq
import sympy


def build_circuit(layers: int, data_qubits, data_qubits_prefix, label_qubit=None, label_symbol: sympy.Symbol = None,
                  full_layer_labeling: bool = False, all_layers_labeling: bool = False):
    circuit = cirq.Circuit()
    layer_symbols_size = len(data_qubits) * 3 - 1
    symbols = sympy.symbols(f"{data_qubits_prefix}:{layers * layer_symbols_size}")
    for i in range(0, layers):
        if i > 0 and not all_layers_labeling:
            circuit.append([_build_layer(None, data_qubits, None,
                                         symbols[i * layer_symbols_size: (i + 1) * layer_symbols_size],
                                         False)])
        else:
            circuit.append([_build_layer(label_qubit, data_qubits, label_symbol,
                                         symbols[i * layer_symbols_size: (i + 1) * layer_symbols_size],
                                         full_layer_labeling)])

    return circuit, symbols


def _build_layer(label_qubit, data_qubits, label_symbol, data_symbols, full_layer_labeling=True):
    assert len(data_symbols) == len(data_qubits) * 3 - 1

    # Add the label rotation
    layer = cirq.Circuit()
    if full_layer_labeling:
        for data_qubit in data_qubits:
            layer.append(cirq.rx(label_symbol).on(data_qubit))
    if label_qubit is not None and label_symbol is not None:
        layer.append(
            cirq.rx(label_symbol).on(label_qubit))

    # Define symbol iterator
    i = 0

    # Add rx rotation for all qubits
    for data_qubit in data_qubits:
        layer.append([cirq.rx(data_symbols[i]).on(data_qubit)])
        i += 1

    # Add rz rotation to all qubits
    for data_qubit in data_qubits:
        layer.append([cirq.rz(data_symbols[i]).on(data_qubit)])
        i += 1

    # Add first moment of ZZ two qubit gates starting from 0th qubit
    j = 0
    while j < len(data_qubits) - 1:
        layer.append(
            [cirq.ZZPowGate(exponent=data_symbols[i], global_shift=-0.5).on(data_qubits[j], data_qubits[j + 1])])
        j += 2
        i += 1
    # Add second moment of ZZ two qubit gates starting from 1st qubit
    j = 1
    while j < len(data_qubits) - 1:
        layer.append(
            [cirq.ZZPowGate(exponent=data_symbols[i], global_shift=-0.5).on(data_qubits[j], data_qubits[j + 1])])
        j += 2
        i += 1
    return layer
