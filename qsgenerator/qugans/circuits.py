import cirq
import sympy


def build_gan_circuits(generator_layers: int, discriminator_layers: int, data_bus_size: int):
    qubits = cirq.GridQubit.rect(1, 3 + data_bus_size)
    out_qubit, label_disc_qubit = qubits[0], qubits[1]
    data_qubits = qubits[2:-1]
    label_gen_qubit = qubits[-1]
    ls = sympy.symbols("l")
    gen, gen_symbols = build_circuit(generator_layers, label_gen_qubit, data_qubits + [label_gen_qubit], ls, "g")
    disc, disc_symbols = build_circuit(discriminator_layers, label_disc_qubit,
                                       [out_qubit, label_disc_qubit] + data_qubits, ls, "d")
    return gen, gen_symbols, disc, disc_symbols, ls, data_qubits, out_qubit


def build_circuit(layers: int, label_qubit, data_qubits, label_symbol, data_qubits_prefix):
    circuit = cirq.Circuit()
    layer_symbols_size = len(data_qubits) * 3 - 1
    symbols = sympy.symbols(f"{data_qubits_prefix}:{layers * layer_symbols_size}")
    circuit.append([_build_layer(label_qubit, data_qubits, label_symbol,
                                 symbols[0:  layer_symbols_size])])
    for i in range(1, layers):
        circuit.append([_build_layer(None, data_qubits, None,
                                     symbols[i * layer_symbols_size: (i + 1) * layer_symbols_size])])

    return circuit, symbols


def _build_layer(label_qubit, data_qubits, label_symbol, data_symbols):
    assert len(data_symbols) == len(data_qubits) * 3 - 1

    # Add the label rotation
    if label_qubit is not None and label_symbol is not None:
        layer = cirq.Circuit(
            cirq.rz(label_symbol).on(label_qubit))
    else:
        layer = cirq.Circuit()

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
        layer.append([cirq.ZZ(data_qubits[j], data_qubits[j + 1]) ** data_symbols[i]])
        j += 2
        i += 1

    # Add second moment of ZZ two qubit gates starting from 1st qubit
    j = 1
    while j < len(data_qubits) - 1:
        layer.append([cirq.ZZ(data_qubits[j], data_qubits[j + 1]) ** data_symbols[i]])
        j += 2
        i += 1

    return layer
