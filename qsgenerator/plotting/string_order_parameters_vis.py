from collections import defaultdict
from pathlib import Path

import json
import cirq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qsgenerator import circuits
from qsgenerator.evaluators.circuit_evaluator import CircuitEvaluator
from qsgenerator.phase.analitical import get_g_parameters_provider
from qsgenerator.phase.circuits import PhaseCircuitBuilder
from qsgenerator.phase.string_order_parameters import measure_s1_pauli_string, measure_szy_pauli_string
from qsgenerator.plotting.utils import get_ids, id_prefix_wqgans, get_data_for_id

colors = sns.color_palette("husl", 5)
FIG_PATH_PREFIX = "/Users/wiktorjurasz/Studies/thesis/paper/figures"

sns.set()


def get_qasm(ids, project='thesis-em2'):
    data = defaultdict(list)
    for rid in ids:
        run = get_data_for_id(rid, project)
        fidelity_name = [k for k in run.get_structure()['logs'].keys() if k.startswith("fidelity (real:gen)")][0]
        g = fidelity_name.split(" ")[2].split(":")[0][1:]
        size = int(run[f"parameters/size"].fetch())
        Path(f".neptune/snapshots/{size}").mkdir(exist_ok=True, parents=True)
        run['artifacts/full_snapshot.json'].download(f".neptune/snapshots/{size}/snapshot_{float(g)}.txt")
    return data


def analytical_transition(g):
    return 4 * np.abs(g) / (1 + np.abs(g)) ** 2


def analytical_transition_1(g):
    return analytical_transition(g) if g > 0 else 0


def analytical_transition_zy(g):
    return analytical_transition(g) if g < 0 else 0


size_5_ids = get_ids([485, 487, 488, 490, 492, 452, 454, 456, 458, 459, 499, 500, 501, 502, 503, 529, 530, 531, 532, 533], id_prefix_wqgans)
size_6_ids = get_ids([434, 420, 419, 418, 417, 461, 463, 465, 467, 469, 505, 506, 507, 508, 509, 535, 536, 537, 538, 539], id_prefix_wqgans)
size_7_ids = get_ids([435, 425, 424, 423, 422, 471, 474, 475, 476, 479, 511, 512, 513, 514, 515, 541, 542, 543, 544, 545], id_prefix_wqgans)
size_8_ids = get_ids([428, 429, 430, 427, 431, 481, 484, 486, 489, 491, 517, 518, 519, 520, 521, 547, 548, 549, 550, 551], id_prefix_wqgans)
size_9_ids = get_ids([442, 443, 445, 446, 440], id_prefix_wqgans)
metadata = [
    {'ids': size_5_ids, 'size': 5},
    {'ids': size_6_ids, 'size': 6},
    {'ids': size_7_ids, 'size': 7},
    {'ids': size_8_ids, 'size': 8},
    # {'ids': size_9_ids, 'size': 9},
]

for meta in metadata:
    get_qasm(meta['ids'])

s1s = {}
szys = {}

all_g = [np.round(el, 3) for el in np.linspace(-1, 1, 21)]
for p in Path(".neptune/snapshots").glob("*"):
    size = int(p.name.split("/")[-1])
    g_to_s1 = {}
    g_to_szy = {}

    for qp in Path(p).glob("*"):
        with open(str(qp)) as snapshot_file:
            snapshot = json.load(snapshot_file)
            g = float(".".join(snapshot_file.name.split("/")[-1].split("_")[-1].split(".")[:2]))
            data_qubits = cirq.GridQubit.rect(1, size)
            name_weight_dict = snapshot[-1]['gen_pairs'][0][2]
            gen, gs = circuits.build_circuit(int(len(name_weight_dict) / (size * 3 - 1)), data_qubits, "g")
            symbol_weight_dict = {s: name_weight_dict[s.name] for s in gs}
            evaluator = CircuitEvaluator(gen, symbol_value_pairs=symbol_weight_dict)
            g_to_s1[g] = measure_s1_pauli_string(evaluator.get_resolved_circuit())
            g_to_szy[g] = measure_szy_pauli_string(evaluator.get_resolved_circuit())

    real_values_provider = get_g_parameters_provider()
    real_pos, real_symbols_pos, symbols_dict_real_pos = PhaseCircuitBuilder(
        all_gates_parametrized=False, g_positive=True).build_ground_state_circuit(size=size)
    real_neg, real_symbols_neg, symbols_dict_real_neg = PhaseCircuitBuilder(
        all_gates_parametrized=False, g_positive=True).build_ground_state_circuit(size=size)
    for g in all_g:
        if g not in g_to_s1:
            if g <= 0:
                evaluator = CircuitEvaluator(real_pos, real_symbols_pos, real_values_provider, g_values=[g])
            else:
                evaluator = CircuitEvaluator(real_neg, real_symbols_neg, real_values_provider, g_values=[g])
            g_to_s1[g] = measure_s1_pauli_string(evaluator.get_resolved_circuit())
            g_to_szy[g] = measure_szy_pauli_string(evaluator.get_resolved_circuit())
    if size < 8:
        s1s[size] = {k: v.numpy()[0][0] for k, v in sorted(g_to_s1.items())}
    if size > 6:
        szys[size] = {k: v.numpy()[0][0] for k, v in sorted(g_to_szy.items())}

marker_types = ['o', '^', 's', '*', '+']
all_g = [np.round(el, 3) for el in np.linspace(-1, 1, 100)]
fig, ax = plt.subplots()
for i, (size, s) in enumerate(s1s.items()):
    ax.plot(list(s.keys()), list(s.values()), marker_types[i], c=colors[i], label=f"N = {size}")
ax.plot(all_g, [analytical_transition_1(g) for g in all_g], color='black', label='exact')
ax.legend()
ax.set_xlabel("g", fontsize=12)
ax.set_ylabel("String order parameters $S^\mathbb{1}$", fontsize=12)
Path(f"{FIG_PATH_PREFIX}/string_order_s1").mkdir(exist_ok=True)
fig.savefig(f"{FIG_PATH_PREFIX}/string_order_s1/plot.png")
plt.show()

fig, ax = plt.subplots()
for i, (size, s) in enumerate(szys.items()):
    ax.plot(list(s.keys()), list(s.values()), marker_types[i], c=colors[i], label=f"N = {size}")

ax.set_xlabel("g", fontsize=12)
ax.set_ylabel("String order parameters $S^{ZY}$", fontsize=12)
ax.plot(all_g, [analytical_transition_zy(g) for g in all_g], color='black', label='exact')
ax.legend()
Path(f"{FIG_PATH_PREFIX}/string_order_szy").mkdir(exist_ok=True)
fig.savefig(f"{FIG_PATH_PREFIX}/string_order_szy/plot.png")
plt.show()
