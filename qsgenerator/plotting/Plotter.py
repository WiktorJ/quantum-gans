from collections import defaultdict

from matplotlib import pyplot as plt
from IPython.display import clear_output


class Plotter:
    def __init__(self):
        self.disc_costs = []
        self.gen_costs = []
        self.prob_fake_real_history = []
        self.prob_real_real_history = []
        self.em_distance_history = []
        self.fidelities_history = defaultdict(list)
        self.abs_fidelities_history = defaultdict(list)
        self.fig = plt.figure()
        self.val_losses = []
        self.x = []
        self.i = 0

    def on_epoch_end(self, disc_cost, gen_cost, prob_fake_real, prob_real_real, fidelities, refresh):
        self.x.append(self.i)
        self.disc_costs.append(disc_cost)
        self.gen_costs.append(gen_cost)
        self.prob_fake_real_history.append(prob_fake_real)
        self.prob_real_real_history.append(prob_real_real)
        for item in fidelities.items():
            self.fidelities_history[item[0]].append(item[1])
        self.i += 1
        if refresh:
            clear_output(wait=True)
            try:
                mng = plt.get_current_fig_manager()
                mng.frame.Maximize(True)
                fig, axes = plt.subplots(nrows=1, ncols=3)
            except:
                # doesn't work on mac
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
            axes[0].plot(self.x, self.disc_costs, label="discriminator cost")
            axes[0].plot(self.x, self.gen_costs, label="generator cost")
            axes[0].legend()
            axes[1].plot(self.x, self.prob_real_real_history, label="prob real real")
            axes[1].plot(self.x, self.prob_fake_real_history, label="prob fake real")
            axes[1].legend()
            for item in self.fidelities_history.items():
                axes[2].plot(self.x, item[1], label=f"fidelity g={item[0]}")
            axes[2].legend()
            fig.tight_layout()
            plt.show()

    def plot_quwgans(self, em_distance, fidelities, abs_fidelities, refresh):
        self.x.append(self.i)
        for item in fidelities.items():
            self.fidelities_history[item[0]].append(item[1])
        for item in abs_fidelities.items():
            self.abs_fidelities_history[item[0]].append(item[1])
        self.em_distance_history.append(em_distance)
        self.i += 1
        if refresh:
            clear_output(wait=True)
            try:
                mng = plt.get_current_fig_manager()
                mng.frame.Maximize(True)
                fig, axes = plt.subplots(nrows=1, ncols=3)
            except:
                # doesn't work on mac
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
            axes[0].plot(self.x, self.em_distance_history, label="earth mover distance")
            axes[0].legend()
            for item in self.fidelities_history.items():
                axes[1].plot(self.x, item[1], label=f"fidelity g={item[0]}")
            axes[1].legend()
            for item in self.abs_fidelities_history.items():
                axes[2].plot(self.x, item[1], label=f"fidelity modulo g={item[0]}")
            axes[2].legend()
            fig.tight_layout()
            plt.show()
