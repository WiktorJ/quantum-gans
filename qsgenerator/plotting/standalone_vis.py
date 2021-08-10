from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qsgenerator.plotting.utils import get_ids, \
    id_prefix_sqgans, \
    id_prefix_wqgans, \
    get_data_for_id, \
    id_prefix_wqgans_gans

sns.set()

FIG_PATH_PREFIX = "/Users/wiktorjurasz/Studies/thesis/paper/figures"

colors = sns.color_palette("husl", 5)

size_3_ids = get_ids([64, 72, 73, 74, 75], id_prefix_sqgans)
size_4_ids = get_ids([66, 67, 69, 78, 79], id_prefix_sqgans)
size_5_ids = get_ids([42, 44, 46, 84, 85], id_prefix_sqgans)

size_4_ids_wqgans_phase_k_3_gen_4 = get_ids([39, 45, 51, 56, 60], id_prefix_wqgans)
size_5_ids_wqgans_phase_k_3_gen_4 = get_ids([40, 46, 52, 57, 61], id_prefix_wqgans)
size_6_ids_wqgans_phase_k_3_gen_4 = get_ids([42, 48, 53, 58, 294], id_prefix_wqgans)
size_7_ids_wqgans_phase_k_3_gen_4 = get_ids([43, 50, 55, 59, 63], id_prefix_wqgans)
size_8_ids_wqgans_phase_k_3_gen_4 = get_ids([43, 50, 55, 59, 63], id_prefix_wqgans)

size_6_ids_wqgans_phase_k_3_gen_5 = get_ids([64, 69, 74], id_prefix_wqgans)
size_7_ids_wqgans_phase_k_3_gen_5 = get_ids([65, 71, 75], id_prefix_wqgans)
size_8_ids_wqgans_phase_k_3_gen_5 = get_ids([67, 72, 76], id_prefix_wqgans)

size_6_ids_wqgans_phase_k_4_gen_5 = {
    'ids': get_ids([184, 188, 192, 196, 200], id_prefix_wqgans),
    'size': 6,
    'k': 4,
    'gen': 5,
    'type': 'phase',
    'color': colors[2]
}

size_7_ids_wqgans_phase_k_4_gen_5 = {
    'ids': get_ids([185, 189, 193, 197, 201], id_prefix_wqgans),
    'size': 7,
    'k': 4,
    'gen': 5,
    'type': 'phase',
    'color': colors[2]
}

size_8_ids_wqgans_phase_k_4_gen_5 = {
    'ids': get_ids([329, 336, 338, 341, 347], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 5,
    'type': 'phase',
    'color': colors[2]
}

size_9_ids_wqgans_phase_k_4_gen_5 = {
    'ids': get_ids([187, 191, 199, 203], id_prefix_wqgans),
    'size': 9,
    'k': 4,
    'gen': 5,
    'type': 'phase',
    'color': colors[2]
}

size_6_ids_wqgans_phase_k_4_gen_4 = {
    'ids': get_ids([78, 83, 86, 89, 92], id_prefix_wqgans),
    'size': 6,
    'k': 4,
    'gen': 4,
    'type': 'phase',
    'color': colors[2]
}
size_7_ids_wqgans_phase_k_4_gen_4 = {
    'ids': get_ids([80, 84, 87, 90, 93], id_prefix_wqgans),
    'size': 7,
    'k': 4,
    'gen': 4,
    'type': 'phase',
    'color': colors[2]
}
size_8_ids_wqgans_phase_k_4_gen_4 = {
    'ids': get_ids([82, 85, 88, 91, 93], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 4,
    'type': 'phase',
    'color': colors[2]
}

size_4_ids_wqgans_butterfly_k_3_gen_4 = {
    'ids': get_ids([124, 128, 132, 136, 140], id_prefix_wqgans),
    'size': 4,
    'k': 3,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}
size_5_ids_wqgans_butterfly_k_3_gen_4 = {
    'ids': get_ids([125, 129, 137, 141, 144], id_prefix_wqgans),
    'size': 5,
    'k': 3,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}
size_6_ids_wqgans_butterfly_k_3_gen_4 = {
    'ids': get_ids([126, 130, 134, 138, 142], id_prefix_wqgans),
    'size': 6,
    'k': 3,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}
size_7_ids_wqgans_butterfly_k_3_gen_4 = {
    'ids': get_ids([127, 131, 135, 139, 143], id_prefix_wqgans),
    'size': 7,
    'k': 3,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}

size_5_ids_wqgans_butterfly_k_4_gen_4 = {
    'ids': get_ids([164, 168, 172, 176, 180], id_prefix_wqgans),
    'size': 5,
    'k': 4,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}

size_6_ids_wqgans_butterfly_k_4_gen_4 = {
    'ids': get_ids([165, 213, 173, 177, 181], id_prefix_wqgans),
    'size': 6,
    'k': 4,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}

size_7_ids_wqgans_butterfly_k_4_gen_4 = {
    'ids': get_ids([170, 174, 182, 205, 214], id_prefix_wqgans),
    'size': 7,
    'k': 4,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}

size_8_ids_wqgans_butterfly_k_4_gen_4 = {
    'ids': get_ids([167, 183, 209, 212, 218], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 4,
    'type': 'butterfly',
    'color': colors[3]
}

size_8_ids_wqgans_butterfly_k_4_gen_5 = {
    'ids': get_ids([304, 306, 307, 328, 329], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 5,
    'type': 'butterfly',
    'color': colors[3]
}

size_4_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([246, 264, 270], id_prefix_wqgans),
    'size': 4,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_5_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([247, 253, 259, 265, 272], id_prefix_wqgans),
    'size': 5,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_6_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([248, 254, 260, 266, 273, 279], id_prefix_wqgans),
    'size': 6,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_7_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([249, 255, 267, 274, 280], id_prefix_wqgans),
    'size': 7,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_8_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([256, 262, 268, 281], id_prefix_wqgans),
    'size': 8,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_4_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([330, 339, 349, 363, 370], id_prefix_wqgans),
    'size': 4,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_5_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([340, 350, 357, 371, 377], id_prefix_wqgans),
    'size': 5,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_6_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([342, 351, 358, 378], id_prefix_wqgans),
    'size': 6,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_7_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([343, 352, 359, 373, 379], id_prefix_wqgans),
    'size': 7,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_8_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([344, 353, 367], id_prefix_wqgans),
    'size': 8,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}
size_9_ids_wqgans_butterfly_k_2_gen_4_same = {
    'ids': get_ids([335, 345, 361, 368, 375], id_prefix_wqgans),
    'size': 9,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}
size_10_ids_wqgans_butterfly_k_3_gen_4_same = {
    'ids': get_ids([384, 436, 439, 441], id_prefix_wqgans),
    'size': 10,
    'k': 3,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3]
}

size_4_ids_wqgans_phase_k_3_gen_4_interpolation = {
    'ids': get_ids([433, 415, 414, 413, 412], id_prefix_wqgans),
    'size': 4,
    'k': 3,
    'gen': "4",
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}

size_5_ids_wqgans_phase_k_3_gen_4_interpolation = {
    'ids': get_ids([433, 415, 414, 413, 412], id_prefix_wqgans),
    'size': 5,
    'k': 3,
    'gen': "4",
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}

size_6_ids_wqgans_phase_k_3_gen_4_interpolation = {
    'ids': get_ids([434, 420, 419, 418, 417], id_prefix_wqgans),
    'size': 6,
    'k': 3,
    'gen': "4",
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}

size_7_ids_wqgans_phase_k_3_gen_4_interpolation = {
    'ids': get_ids([434, 420, 419, 418, 417], id_prefix_wqgans),
    'size': 7,
    'k': 3,
    'gen': "4",
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}

size_8_ids_wqgans_phase_k_4_gen_4_interpolation = {
    'ids': get_ids([402, 403, 404, 405, 406], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 4,
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}
size_8_ids_wqgans_phase_k_4_gen_5_interpolation = {
    'ids': get_ids([408, 409, 410, 411, 407], id_prefix_wqgans),
    'size': 8,
    'k': 4,
    'gen': 5,
    'type': 'phase',
    'color': colors[3],
    'suffix': 'interpolation'
}

size_4_ids_wqgans_butterfly_k_2_gen_same_gans = {
    'ids': get_ids([266, 264, 263, 262, 261], id_prefix_wqgans_gans),
    'size': 4,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3],
    'suffix': 'gan'
}

size_6_ids_wqgans_butterfly_k_2_gen_same_gans = {
    'ids': get_ids([272, 275, 197, 280, 278], id_prefix_wqgans_gans),
    'size': 6,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3],
    'suffix': 'gan'
}

size_8_ids_wqgans_butterfly_k_2_gen_same_gans = {
    'ids': get_ids([301, 297, 298, 299, 296], id_prefix_wqgans_gans),
    'size': 8,
    'k': 2,
    'gen': "same as real",
    'type': 'butterfly',
    'color': colors[3],
    'suffix': 'gan'
}

metadata = [
    # {'ids': size_3_ids, 'dir': 'sqgans_size=3', 'sub_title': f"real input qubits = {3}"},
    # {'ids': size_4_ids, 'dir': 'sqgans_size=4', 'sub_title': f"real input qubits = {4}"},
    # {'ids': size_5_ids, 'dir': 'sqgans_size=5', 'sub_title': f"real input qubits = {5}"},
    #     {
    #         'ids': size_4_ids_wqgans_phase_k_3_gen_4,
    #         'dir': 'wqgans_phase_size=4_k=3_gen=4',
    #         'sub_title': f"real input qubits = {4}, gen layers = {4}, k = 3",
    #         'color': colors[0]
    #     },
    #     {
    #         'ids': size_5_ids_wqgans_phase_k_3_gen_4,
    #         'dir': 'wqgans_phase_size=5_k=3_gen=4',
    #         'sub_title': f"real input qubits = {5}, gen layers = {4}, k = 3",
    #         'color': colors[0]
    #     },
    #     {
    #         'ids': size_6_ids_wqgans_phase_k_3_gen_4,
    #         'dir': 'wqgans_phase_size=6_k=3_gen=4',
    #         'sub_title': f"real input qubits = {6}, gen layers = {4}, k = 3",
    #         'color': colors[0]
    #     },
    #     {
    #         'ids': size_7_ids_wqgans_phase_k_3_gen_4,
    #         'dir': 'wqgans_phase_size=7_k=3_gen=4',
    #         'sub_title': f"real input qubits = {7}, gen layers = {4}, k = 3",
    #         'color': colors[0]
    #     },
    #     {
    #         'ids': size_8_ids_wqgans_phase_k_3_gen_4,
    #         'dir': 'wqgans_phase_size=8_k=3_gen=4',
    #         'sub_title': f"real input qubits = {8}, gen layers = {4}, k = 3",
    #         'color': colors[0]
    #     },
    #     {
    #         'ids': size_6_ids_wqgans_phase_k_3_gen_5,
    #         'dir': 'wqgans_phase_size=6_k=3_gen=5',
    #         'sub_title': f"real input qubits = {6}, gen layers = {5}, k = 3",
    #         'color': colors[1]
    #     },
    #     {
    #         'ids': size_7_ids_wqgans_phase_k_3_gen_5,
    #         'dir': 'wqgans_phase_size=7_k=3_gen=5',
    #         'sub_title': f"real input qubits = {7}, gen layers = {5}, k = 3",
    #         'color': colors[1]
    #     },
    #     {
    #         'ids': size_8_ids_wqgans_phase_k_3_gen_5,
    #         'dir': 'wqgans_phase_size=8_k=3_gen=5',
    #         'sub_title': f"real input qubits = {8}, gen layers = {5}, k = 3",
    #         'color': colors[1]
    #     },
]

for meta_dict in [
    # size_6_ids_wqgans_phase_k_4_gen_4,
    # size_7_ids_wqgans_phase_k_4_gen_4,
    # size_8_ids_wqgans_phase_k_4_gen_4,
    # size_4_ids_wqgans_butterfly_k_3_gen_4,
    # size_5_ids_wqgans_butterfly_k_3_gen_4,
    # size_6_ids_wqgans_butterfly_k_3_gen_4,
    # size_7_ids_wqgans_butterfly_k_3_gen_4,
    # size_6_ids_wqgans_phase_k_4_gen_5,
    # size_7_ids_wqgans_phase_k_4_gen_5,
    # size_8_ids_wqgans_phase_k_4_gen_5,
    # size_9_ids_wqgans_phase_k_4_gen_5,
    # size_5_ids_wqgans_butterfly_k_4_gen_4,
    # size_6_ids_wqgans_butterfly_k_4_gen_4,
    # size_7_ids_wqgans_butterfly_k_4_gen_4,
    # size_8_ids_wqgans_butterfly_k_4_gen_4,
    # size_4_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_5_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_6_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_7_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_8_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_8_ids_wqgans_butterfly_k_4_gen_5,
    # size_4_ids_wqgans_phase_k_3_gen_4_interpolation,
    # size_5_ids_wqgans_phase_k_3_gen_4_interpolation,
    # size_6_ids_wqgans_phase_k_3_gen_4_interpolation,
    # size_7_ids_wqgans_phase_k_3_gen_4_interpolation,
    # size_4_ids_wqgans_butterfly_k_2_gen_4_same,
    # size_5_ids_wqgans_butterfly_k_2_gen_4_same,
    # size_6_ids_wqgans_butterfly_k_2_gen_4_same,
    # size_7_ids_wqgans_butterfly_k_2_gen_4_same,
    # size_8_ids_wqgans_butterfly_k_2_gen_4_same,
    # size_9_ids_wqgans_butterfly_k_2_gen_4_same,
    size_10_ids_wqgans_butterfly_k_3_gen_4_same,
    # size_8_ids_wqgans_phase_k_4_gen_4_interpolation,
    # size_8_ids_wqgans_phase_k_4_gen_5_interpolation,
    # size_4_ids_wqgans_butterfly_k_2_gen_same_gans,
    # size_6_ids_wqgans_butterfly_k_2_gen_same_gans,
    # size_8_ids_wqgans_butterfly_k_2_gen_same_gans,
]:
    if meta_dict.get('suffix'):
        directory = f"wqgans_{meta_dict['type']}_size={meta_dict['size']}_k={meta_dict['k']}_gen={str(meta_dict['gen']).replace(' ', '_')}_{meta_dict['suffix']}"
    else:
        directory = f"wqgans_{meta_dict['type']}_size={meta_dict['size']}_k={meta_dict['k']}_gen={str(meta_dict['gen']).replace(' ', '_')}"
    metadata.append(
        {
            'ids': meta_dict['ids'],
            'dir': directory,
            'sub_title': f"real input qubits = {meta_dict['size']}, gen layers = {meta_dict['gen']}, k = {meta_dict['k']}",
            'color': meta_dict['color']
        }
    )

metadata_per_project = {'thesis': {
    'fidelity_g': {'name': 'fidelity', 'color': colors[0]},
    'prob_real_real': {'name': 'probability real as real', 'color': colors[1]},
    'prob_fake_real': {'name': 'probability generated as real', 'color': colors[2]}
},
    'thesis-em2': {
        'fidelity (': {'name': 'fidelity', 'color': colors[0]},
        'em_distance': {'name': 'Wasserstein Distance', 'color': colors[1], 'ylim': (0, 5)}
    },
    'thesis-em-exps': {
        'fidelity (': {'name': 'fidelity', 'color': colors[0]},
        'em_distance': {'name': 'Wasserstein Distance', 'color': colors[1], 'ylim': (0, 5)}
    }
}


def get_data(ids, param_prefixes, project='thesis', filter_epoch=True):
    data = defaultdict(list)
    for rid in ids:
        run = get_data_for_id(rid, project)
        for param_prefix in param_prefixes:
            field_name = [k for k in run.get_structure()['logs'].keys() if k.startswith(param_prefix)][0]
            res = run[f"logs/{field_name}"].fetch_values()
            res = res[['value']].to_numpy().flatten()
            res = res[:401]
            if filter_epoch:
                res = [el for i, el in enumerate(res) if i % 2 == 0]
            data[param_prefix].append(res)
    return data


def plot_metric(metric, data_name, color, sub_title, ylim):
    if ylim is None:
        ylim = (0, 1)
    fig, ax = plt.subplots()
    mean = np.mean(np.stack(metric), axis=0)
    mi = np.min(np.stack(metric), axis=0)
    ma = np.max(np.stack(metric), axis=0)
    std = np.std(np.stack(metric), axis=0)
    ax.plot(list(range(len(metric[0]))), mean, c=color)
    ax.fill_between(list(range(len(metric[0]))), mi, ma, alpha=0.3, facecolor=color)
    # ax.fill_between(list(range(len(metric[0]))), mean - std, mean + std, alpha=0.3, facecolor=color)
    ax.set_xlabel("epoch", fontsize=20)
    ax.set_ylabel(data_name, fontsize=20)
    ax.set_ylim(ylim[0], ylim[1])
    fig.suptitle(sub_title, fontsize=18)
    plt.show()
    return fig


# project = 'thesis-em-exps'
project = 'thesis-em2'
# prefix_to_metrics_metadata = metadata_per_project['thesis']
prefix_to_metrics_metadata = metadata_per_project[project]
for meta in metadata:
    data = get_data(meta['ids'], prefix_to_metrics_metadata.keys(), project, filter_epoch=False)
    for prefix, metric in data.items():
        name = prefix_to_metrics_metadata[prefix]['name']
        fig = plot_metric(metric,
                          name,
                          prefix_to_metrics_metadata[prefix]['color'],
                          meta['sub_title'],
                          prefix_to_metrics_metadata[prefix].get('ylim')
                          )
        Path(f"{FIG_PATH_PREFIX}/{meta['dir']}").mkdir(exist_ok=True)
        fig.savefig(f"{FIG_PATH_PREFIX}/{meta['dir']}/{name.replace(' ', '_')}.png")
