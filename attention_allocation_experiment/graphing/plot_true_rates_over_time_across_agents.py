import numpy as np
from matplotlib import pyplot as plt


def plot_true_rates_over_time_across_agents(tot_eval_data):
    """
    Plots average episodic true rates over time.
    aggregated_tot_ep_true_rates: the true rates per timestep per site per agent, should be {agent_name: tot_ep_true_rates} where
                            tot_ep_true_rates shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_true_rates = tot_eval_data[name]['tot_true_rates']  # (num_episodes, num_timesteps, num_locations)
        timesteps = list(range(len(tot_ep_true_rates[0])))
        num_eps = len(tot_ep_true_rates)

        ax.set_title(f'True Rates Over Time For {name} Agent')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('True Rates')

        means = np.mean(tot_ep_true_rates, axis=0)

        for i in range(means.shape[-1]):
            ax.plot(timesteps, means[:, i], label=f'Site {i}')

        ax.legend()

    plt.show()
    plt.close()