import numpy as np
from collections import defaultdict
from typing import List

from matplotlib import pyplot as plt
from IPython.display import clear_output

from qsgenerator.utils import FidelityGrid, GeneratorsFidelityGrid


class Plotter:
    def __init__(self):
        self.disc_costs = []
        self.gen_costs = []
        self.prob_fake_real_history = []
        self.prob_real_real_history = []
        self.em_distance_history = []
        self.trace_distance_history = []
        self.abs_trace_distance_history = []
        self.fidelities_history = defaultdict(lambda: defaultdict(list))
        self.abs_fidelities_history = defaultdict(lambda: defaultdict(list))
        self.gen_fidelities_history = defaultdict(lambda: defaultdict(list))
        self.gen_abs_fidelities_history = defaultdict(lambda: defaultdict(list))
        self.fig = plt.figure()
        self.val_losses = []
        self.x = []
        self.i = 0

    def on_epoch_end(self, disc_cost, gen_cost, prob_fake_real, prob_real_real, fidelities, abs_fidelities, refresh):
        self.x.append(self.i)
        self.disc_costs.append(disc_cost)
        self.gen_costs.append(gen_cost)
        self.prob_fake_real_history.append(prob_fake_real)
        self.prob_real_real_history.append(prob_real_real)
        for item in fidelities.items():
            self.fidelities_history[item[0]].append(item[1])
        for item in abs_fidelities.items():
            self.abs_fidelities_history[item[0]].append(item[1])
        self.i += 1
        if refresh:
            clear_output(wait=True)
            try:
                mng = plt.get_current_fig_manager()
                mng.frame.Maximize(True)
                fig, axes = plt.subplots(nrows=1, ncols=4)
            except:
                # doesn't work on mac
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
            axes[0].plot(self.x, self.disc_costs, label="discriminator cost")
            axes[0].plot(self.x, self.gen_costs, label="generator cost")
            axes[0].legend()
            axes[1].plot(self.x, self.prob_real_real_history, label="prob real real")
            axes[1].plot(self.x, self.prob_fake_real_history, label="prob fake real")
            axes[1].legend()
            for item in self.fidelities_history.items():
                axes[2].plot(self.x, item[1], label=f"fidelity g={item[0]}")
            axes[2].legend()
            for item in self.abs_fidelities_history.items():
                axes[3].plot(self.x, item[1], label=f"abs_fidelity g={item[0]}")
            axes[3].legend()
            fig.tight_layout()
            plt.show()

    def plot_quwgans(self, em_distance: float, trace_distace: float, abs_trace_distance: float,
                     fidelities: List[FidelityGrid], gen_fidelities: List[GeneratorsFidelityGrid], refresh: bool):
        self.x.append(self.i)
        for fg in fidelities:
            self.fidelities_history[fg.label_real][fg.label_gen].append(fg)
        for fg in gen_fidelities:
            self.gen_fidelities_history[fg.label_gen1][fg.label_gen2].append(fg)
        self.em_distance_history.append(em_distance)
        self.trace_distance_history.append(trace_distace)
        self.abs_trace_distance_history.append(abs_trace_distance)
        self.i += 1
        ret_figures = {}
        if refresh:
            distinct_real_labels = len(self.fidelities_history)
            distinct_gen_labels = len(self.gen_fidelities_history)
            clear_output(wait=True)
            try:
                mng = plt.get_current_fig_manager()
                mng.frame.Maximize(True)
            except:
                pass
            fig_m, axes_m = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
            fig_p, axes_p = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
            fig_f, axes_f = plt.subplots(nrows=distinct_real_labels, ncols=3,
                                         figsize=(18, 5 * distinct_real_labels))
            if distinct_gen_labels > 0:
                fig_g, axes_g = plt.subplots(nrows=distinct_gen_labels, ncols=2, figsize=(18, 5 * distinct_gen_labels))
                fig_g.suptitle("Generator-Generator Fidelities")

                if distinct_gen_labels == 1:
                    axes_g = [axes_g]

                for al in axes_g:
                    for a in al:
                        a.set_ylim([0, 1])

                self._plot_fidelity(self.gen_fidelities_history, axes_g)
                ret_figures['gen_fidelities'] = fig_g
            if distinct_real_labels == 1:
                axes_f = [axes_f]

            for al in axes_f:
                for a in al:
                    a.set_ylim([0, 1])

            fig_f.suptitle("Generator-Real Fidelities")
            axes_m.plot(self.x, self.em_distance_history, label="earth mover distance")
            axes_m.plot(self.x, self.trace_distance_history, label="trace distance")
            axes_m.plot(self.x, self.abs_trace_distance_history, label="trace distance modulo")
            axes_m.legend()
            real_labels = sorted(list(self.fidelities_history.keys()))
            gen_labels = list(list(self.fidelities_history.values())[0].keys())
            fidelities = []
            abs_fidelities = []
            probs = []
            for rl in real_labels:
                gen_labels_dict = self.fidelities_history[rl]
                fidelities.append([])
                abs_fidelities.append([])
                probs.append([])
                for gl in gen_labels:
                    fg = gen_labels_dict[gl]
                    fidelities[-1].append(self._get_average_from_last_n(fg, lambda el: el.fidelity))
                    abs_fidelities[-1].append(self._get_average_from_last_n(fg, lambda el: el.abs_fidelity))
                    probs[-1].append(self._get_average_from_last_n(fg, lambda el: el.prob_real - el.prob_gen))

            self._plot_matrix(axes_p[0], fidelities, real_labels, gen_labels, "Fidelity")
            self._plot_matrix(axes_p[1], abs_fidelities, real_labels, gen_labels, "Fidelity modulo")
            self._plot_matrix(axes_p[2], probs, real_labels, gen_labels, "Probabilities")
            self._plot_fidelity_and_probs(self.fidelities_history, axes_f)
            fig_m.tight_layout()
            fig_p.tight_layout()
            fig_f.tight_layout()
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.show()
            ret_figures['distances'] = fig_m
            ret_figures['grids'] = fig_p
            ret_figures['fidelities'] = fig_f
        return ret_figures

    def _get_average_from_last_n(self, array, field_supplier, n=10):
        n = min(n, len(array))
        return np.mean([field_supplier(el) for el in array[-n:]])

    def _plot_fidelity(self, fids, axes):
        for i, (l1, gen_dict) in enumerate(fids.items()):
            for l2, fg_list in gen_dict.items():
                axes[i][0].plot(self.x, [fb.fidelity for fb in fg_list],
                                label=f"fidelity ({l1}:{l2})")
                axes[i][1].plot(self.x, [fb.abs_fidelity for fb in fg_list],
                                label=f"fidelity modulo ({l1}:{l2})")
            axes[i][0].legend()
            axes[i][1].legend()

    def _plot_fidelity_and_probs(self, fids, axes_f):
        for i, (real_label, gen_label_dict) in enumerate(fids.items()):
            for gen_label, fb_list in gen_label_dict.items():
                axes_f[i][0].plot(self.x, [fb.fidelity for fb in fb_list],
                                  label=f"fidelity (real:gen), ({real_label}:{gen_label})")
                axes_f[i][1].plot(self.x, [fb.abs_fidelity for fb in fb_list],
                                  label=f"fidelity modulo (real:gen), ({real_label}:{gen_label})")
                axes_f[i][2].plot(self.x, [fb.prob_real - fb.prob_gen for fb in fb_list],
                                  label=f"prob (real - gen) ({real_label}:{gen_label})")
            axes_f[i][0].legend()
            axes_f[i][1].legend()
            axes_f[i][2].legend()
            axes_f[i][2].set_ylim([-0.5, 0.5])

    def _plot_matrix(self, ax, matrix, rows, columns, title):
        ax.imshow(matrix, vmin=0, vmax=1)

        # We want to show all ticks...
        ax.set_yticks(np.arange(len(rows)))
        ax.set_xticks(np.arange(len(columns)))
        # ... and label them with the respective list entries
        ax.set_yticklabels(rows)
        ax.set_xticklabels(columns)
        for j in range(len(columns)):
            for i in range(len(rows)):
                ax.text(j, i, round(matrix[i][j], 2),
                        ha="center", va="center", color="w")
        ax.set_title(title)
