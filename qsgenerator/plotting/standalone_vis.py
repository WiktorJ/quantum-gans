from collections import defaultdict
from pathlib import Path

import neptune.new as neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

FIG_PATH_PREFIX = "/Users/wiktorjurasz/Studies/thesis/paper/figures"

id_prefix_sqgans = "THES-"


def get_ids(pure_ids, prefix):
    return [f"{prefix}{el}" for el in pure_ids]


size_3_ids = get_ids([64, 72, 73, 74, 75], id_prefix_sqgans)
size_4_ids = get_ids([66, 67, 69, 78, 79], id_prefix_sqgans)
size_5_ids = get_ids([42, 44, 46, 84, 85], id_prefix_sqgans)

metadata = [
    {'ids': size_3_ids, 'dir': 'sqgans_size=3', 'width': 3},
    {'ids': size_4_ids, 'dir': 'sqgans_size=4', 'width': 4},
    {'ids': size_5_ids, 'dir': 'sqgans_size=5', 'width': 5}
]

colors = sns.color_palette("husl", 3)
prefix_to_metrics_metadata = {
    'fidelity_g': {'name': 'fidelity', 'color': colors[0]},
    'prob_real_real': {'name': 'probability real as real', 'color': colors[1]},
    'prob_fake_real': {'name': 'probability generated as real', 'color': colors[2]}
}


def get_data_for_id(rid):
    return neptune.init(
        project='wiktor.jurasz/thesis',
        api_token=None,  # put the token in NEPTUNE_API_TOKEN env variable
        run=rid)


def get_data(ids, param_prefixes):
    data = defaultdict(list)
    for rid in ids:
        run = get_data_for_id(rid)
        for param_prefix in param_prefixes:
            field_name = [k for k in run.get_structure()['logs'].keys() if k.startswith(param_prefix)][0]
            res = run[f"logs/{field_name}"].fetch_values()
            res = res[['value']].to_numpy().flatten()
            res = [el for i, el in enumerate(res) if i % 2 == 0]
            data[param_prefix].append(res)
    return data


def plot_metric(metric, data_name, color, width):
    fig, ax = plt.subplots()
    mean = np.mean(np.stack(metric), axis=0)
    mi = np.min(np.stack(metric), axis=0)
    ma = np.max(np.stack(metric), axis=0)
    std = np.std(np.stack(metric), axis=0)
    ax.plot(list(range(len(metric[0]))), mean, c=color)
    ax.fill_between(list(range(len(metric[0]))), mi, ma, alpha=0.3, facecolor=color)
    # ax.fill_between(list(range(len(metric[0]))), mean - std, mean + std, alpha=0.3, facecolor=color)
    ax.set_xlabel("epoch", fontsize=18)
    ax.set_ylabel(data_name, fontsize=18)
    ax.set_ylim(0, 1)
    fig.suptitle(f"real input qubits = {width}")
    plt.show()
    return fig


for meta in metadata:
    data = get_data(meta['ids'], prefix_to_metrics_metadata.keys())
    for prefix, metric in data.items():
        name = prefix_to_metrics_metadata[prefix]['name']
        fig = plot_metric(metric,
                          name,
                          prefix_to_metrics_metadata[prefix]['color'],
                          meta['width'])
        Path(f"{FIG_PATH_PREFIX}/{meta['dir']}").mkdir(exist_ok=True)
        fig.savefig(f"{FIG_PATH_PREFIX}/{meta['dir']}/{name.replace(' ', '_')}.png")
