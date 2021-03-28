import cirq
import sympy

from qsgenerator.circuits import build_circuit


def build_gan_circuits(generator_layers: int, discriminator_layers: int, data_bus_size: int,
                       generator_bath_size: int = 0, discriminator_bath_size: int = 0,
                       full_layer_labeling: bool = True, all_layers_labeling: bool = False,
                       use_gen_label_qubit: bool = False, use_disc_label_qubit: bool = False):
    total_size = data_bus_size + generator_bath_size + discriminator_bath_size + 1
    disc_exclusive_qubits = discriminator_bath_size + 1
    if use_gen_label_qubit:
        total_size += 1
    if use_disc_label_qubit:
        total_size += 1
        disc_exclusive_qubits += 1

    qubits = cirq.GridQubit.rect(1, total_size)
    out_qubit = qubits[0]
    if use_gen_label_qubit:
        data_qubits = qubits[disc_exclusive_qubits:-(generator_bath_size + 1)]
        label_gen_qubit = qubits[-1]
    else:
        data_qubits = qubits[disc_exclusive_qubits:-generator_bath_size] if generator_bath_size > 0 \
            else qubits[disc_exclusive_qubits:]
        label_gen_qubit = None
    if use_disc_label_qubit:
        label_disc_qubit = qubits[1]
    else:
        label_disc_qubit = None

    ls = sympy.symbols("l")
    gen_qubits = data_qubits + ([label_gen_qubit] if label_gen_qubit else []) + qubits[
                                                                                total_size - generator_bath_size:]
    disc_qubits = [out_qubit] + qubits[1:discriminator_bath_size + 1] + (
        [label_disc_qubit] if label_disc_qubit else []) + data_qubits
    gen, gen_symbols = build_circuit(generator_layers, gen_qubits, "g", label_gen_qubit, ls,
                                     full_layer_labeling, all_layers_labeling)
    disc, disc_symbols = build_circuit(discriminator_layers, disc_qubits, "d", label_disc_qubit, ls,
                                       full_layer_labeling, all_layers_labeling)
    return gen, gen_symbols, disc, disc_symbols, ls, data_qubits, out_qubit
