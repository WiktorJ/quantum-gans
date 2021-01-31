import cirq
import sympy


def build_gan_circuits(generator_layers: int, discriminator_layers: int, data_bus_size: int,
                       full_layer_labeling: bool = True, all_layers_labeling: bool = False,
                       use_gen_label_qubit: bool = False, use_disc_label_qubit: bool = False):
    total_size = data_bus_size + 1
    disc_exclusive_qubits = 1
    if use_gen_label_qubit:
        total_size += 1
    if use_disc_label_qubit:
        total_size += 1
        disc_exclusive_qubits += 1


    qubits = cirq.GridQubit.rect(1, total_size)
    out_qubit = qubits[0]
    if use_gen_label_qubit:
        data_qubits = qubits[disc_exclusive_qubits:-1]
        label_gen_qubit = qubits[-1]
    else:
        data_qubits = qubits[disc_exclusive_qubits:]
        label_gen_qubit = None

    if use_disc_label_qubit:
        label_disc_qubit = qubits[1]
    else:
        label_disc_qubit = None

    ls = sympy.symbols("l")
    gen_qubits = data_qubits + ([label_gen_qubit] if label_gen_qubit else [])
    disc_qubits = [out_qubit] + ([label_disc_qubit] if label_disc_qubit else []) + data_qubits
    gen, gen_symbols = build_circuit(generator_layers, label_gen_qubit, gen_qubits, ls, "g",
                                     full_layer_labeling, all_layers_labeling)
    disc, disc_symbols = build_circuit(discriminator_layers, label_disc_qubit, disc_qubits, ls, "d",
                                       full_layer_labeling, all_layers_labeling)
    return gen, gen_symbols, disc, disc_symbols, ls, data_qubits, out_qubit


def build_circuit(layers: int, label_qubit, data_qubits, label_symbol, data_qubits_prefix,
                  full_layer_labeling: bool = True, all_layers_labeling: bool = False):
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
