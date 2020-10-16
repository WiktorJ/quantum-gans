import cirq
import sympy


def build_gan_circuits(generator_layers: int, discriminator_layers: int, data_bus_size: int):
    pass


def build_generator(generator_layers: int, label_qubit, data_qubits, label_symbol):
    generator = cirq.Circuit()
    layer_symbols_number = (len(data_qubits) + 1) * 2 + (len(data_qubits))
    symbols = sympy.symbols(f"g:{generator_layers * layer_symbols_number}")

    for i in range(generator_layers):
        generator.append([_build_generator_layer(label_qubit, data_qubits, label_symbol,
                                                 symbols[i * layer_symbols_number: (i + 1) * layer_symbols_number])])

    return generator, symbols


def _build_generator_layer(label_qubit, data_qubits, label_symbol, data_symbols):
    assert len(data_symbols) == (len(data_qubits) + 1) * 2 + (len(data_qubits))

    layer = cirq.Circuit(
        cirq.rz(label_symbol).on(label_qubit))
    i = 0
    for data_qubit in data_qubits:
        layer.append([cirq.rx(data_symbols[i]).on(data_qubit)])
        i += 1
    layer.append([cirq.rx(data_symbols[i]).on(label_qubit)])
    i += 1

    for data_qubit in data_qubits:
        layer.append([cirq.rz(data_symbols[i]).on(data_qubit)])
        i += 1
    layer.append([cirq.rz(data_symbols[i]).on(label_qubit)])
    i += 1

    j = 0
    while j < len(data_qubits) // 2:
        layer.append([cirq.ZZ(data_qubits[j], data_qubits[j + 1]) ** data_symbols[i]])
        j += 2
        i += 1
    if len(data_qubits) % 2 == 1:
        layer.append([cirq.ZZ(data_qubits[-1], label_qubit) ** data_symbols[i]])
        i += 1

    j = 1
    while j < len(data_qubits) // 2:
        layer.append([cirq.ZZ(data_qubits[j], data_qubits[j + 1]) ** data_symbols[i]])
        j += 2
        i += 1

    if len(data_qubits) % 2 == 0:
        layer.append([cirq.ZZ(data_qubits[-1], label_qubit) ** data_symbols[i]])

    return layer

    out_qubit, label_disc, data1, data2, data3, label_gen = cirq.GridQubit.rect(1, 6)

    ls = sympy.symbols("l")

    gs = sympy.symbols("g:22")
    gen = cirq.Circuit(
        cirq.rz(ls).on(label_gen),
        cirq.rx(gs[0]).on(data1),
        cirq.rx(gs[1]).on(data2),
        cirq.rx(gs[2]).on(data3),
        cirq.rx(gs[3]).on(label_gen),
        cirq.rz(gs[4]).on(data1),
        cirq.rz(gs[5]).on(data2),
        cirq.rz(gs[6]).on(data3),
        cirq.rz(gs[7]).on(label_gen),
        cirq.ZZ(data1, data2) ** gs[8],
        cirq.ZZ(data3, label_gen) ** gs[9],
        cirq.ZZ(data2, data3) ** gs[10],
        cirq.rx(gs[11]).on(data1),
        cirq.rx(gs[12]).on(data2),
        cirq.rx(gs[13]).on(data3),
        cirq.rx(gs[14]).on(label_gen),
        cirq.rz(gs[15]).on(data1),
        cirq.rz(gs[16]).on(data2),
        cirq.rz(gs[17]).on(data3),
        cirq.rz(gs[18]).on(label_gen),
        cirq.ZZ(data1, data2) ** gs[19],
        cirq.ZZ(data3, label_gen) ** gs[20],
        cirq.ZZ(data2, data3) ** gs[21],
    )

    pure_gen = gen.copy()

    ds = sympy.symbols("d:28")
    disc = cirq.Circuit(
        cirq.rz(ls).on(label_disc),
        cirq.rx(ds[0]).on(out_qubit),
        cirq.rx(ds[1]).on(label_disc),
        cirq.rx(ds[2]).on(data1),
        cirq.rx(ds[3]).on(data2),
        cirq.rx(ds[4]).on(data3),
        cirq.rz(ds[5]).on(out_qubit),
        cirq.rz(ds[6]).on(label_disc),
        cirq.rz(ds[7]).on(data1),
        cirq.rz(ds[8]).on(data2),
        cirq.rz(ds[9]).on(data3),
        cirq.ZZ(out_qubit, label_disc) ** ds[10],
        cirq.ZZ(data1, data2) ** ds[11],
        cirq.ZZ(label_disc, data1) ** ds[12],
        cirq.ZZ(data2, data3) ** ds[13],
        cirq.rx(ds[14]).on(out_qubit),
        cirq.rx(ds[15]).on(label_disc),
        cirq.rx(ds[16]).on(data1),
        cirq.rx(ds[17]).on(data2),
        cirq.rx(ds[18]).on(data3),
        cirq.rz(ds[19]).on(out_qubit),
        cirq.rz(ds[20]).on(label_disc),
        cirq.rz(ds[21]).on(data1),
        cirq.rz(ds[22]).on(data2),
        cirq.rz(ds[23]).on(data3),
        cirq.ZZ(out_qubit, label_disc) ** ds[24],
        cirq.ZZ(data1, data2) ** ds[25],
        cirq.ZZ(label_disc, data1) ** ds[26],
        cirq.ZZ(data2, data3) ** ds[27],
    )

    gen.append([disc])
