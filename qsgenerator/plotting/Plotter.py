from matplotlib import pyplot as plt
from IPython.display import clear_output


class Plotter:
    def __init__(self):
        self.disc_costs = []
        self.gen_costs = []
        self.prob_fake_real_history = []
        self.prob_real_real_history = []
        self.fig = plt.figure()
        self.val_losses = []
        self.x = []
        self.i = 0

    def on_epoch_end(self, disc_cost, gen_cost, prob_fake_real, prob_real_real):
        self.x.append(self.i)
        self.disc_costs.append(disc_cost)
        self.gen_costs.append(gen_cost)
        self.prob_fake_real_history.append(prob_fake_real)
        self.prob_real_real_history.append(prob_real_real)
        self.i += 1

        clear_output(wait=True)
        try:
            mng = plt.get_current_fig_manager()
            mng.frame.Maximize(True)
            fig, axes = plt.subplots(nrows=1, ncols=2)
        except:
            # doesn't work on mac
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        axes[0].plot(self.x, self.disc_costs, label="discriminator cost")
        axes[0].plot(self.x, self.gen_costs, label="generator cost")
        axes[0].legend()
        axes[1].plot(self.x, self.prob_real_real_history, label="prob real real")
        axes[1].plot(self.x, self.prob_fake_real_history, label="prob fake real")
        axes[1].legend()
        fig.tight_layout()
        plt.show()
