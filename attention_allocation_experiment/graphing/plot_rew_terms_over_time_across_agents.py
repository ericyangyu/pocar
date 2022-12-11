import copy
import random

import numpy as np
from matplotlib import pyplot as plt


def plot_rew_terms_over_time_across_agents(tot_eval_data):
    """
    Plots average episodic reward terms over time, with maxs and mins.
    aggregated_tot_infos: the info for reward terms per timestep over time per agent, should be {agent_name: tot_ep_infos}, where
                            tot_ep_infos has shape (num_eps, num_timesteps)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_ep_infos = tot_eval_data[name]['tot_rew_infos']  # (num_episodes, num_timesteps, {})
        timesteps = list(range(len(tot_ep_infos[0])))
        num_eps = len(tot_ep_infos)

        term_names = list(tot_ep_infos[0][0].keys())

        # Extract out term values, might be slightly inefficient... oh wells ;P
        term_vals = {}
        for term_name in term_names:
            mean_term_vals = []
            for i in range(num_eps):
                ep_term_vals = []
                for info in tot_ep_infos[i]:
                    ep_term_vals.append(info[term_name])
                mean_term_vals.append(copy.deepcopy(ep_term_vals))
            term_vals[term_name] = np.mean(mean_term_vals, axis=0)

        ax.set_title(
            f'Average Reward Term Values Over Time For {name} Agent')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward Magnitude')

        for term_name, term_means in list(term_vals.items()):
            ax.plot(timesteps, term_vals[term_name], label=term_name)

        ax.legend()

    plt.show()