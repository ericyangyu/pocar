import numpy as np
from matplotlib import pyplot as plt


def plot_att_all_over_time_across_agents(tot_eval_data):
    """
    Plots average episodic attention allocated over time.
    aggregated_tot_ep_att_all: the attention allocated per timestep per agent, should be {agent_name: tot_ep_att_all} where
                            tot_ep_att_all shape is (num_episodes, num_timesteps, num_locations)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_att_all = tot_eval_data[name]['tot_att_all']  # (num_episodes, num_timesteps, num_locations)
        timesteps = list(range(len(tot_ep_att_all[0])))
        num_eps = len(tot_ep_att_all)

        ax.set_title(f'Average Attention Allocated Over Time For {name} Agent')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Attention Allocated')

        means = np.mean(tot_ep_att_all, axis=0)

        for i in range(means.shape[-1]):
            ax.plot(timesteps, means[:, i], label=f'Site {i}')

        ax.legend()

    plt.show()
