import numpy as np
from matplotlib import pyplot as plt


def plot_rew_over_time_across_agents(tot_eval_data):
    """
    Plots average episodic rewards over time, with maxs and mins.
    aggregated_tot_ep_rews: the reward per timestep over time per agent, should be {agent_name: tot_ep_rews} where
                            tot_ep_rews shape is (num_episodes, num_timesteps)
    """
    # aggregated_tot_ep_rews = tot_eval_data[]

    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_rews = [tot_eval_data[name]['tot_rews'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_rews[0][0])))

    plt.title(f'Reward Over Time, Base Environment', fontsize=20)
    plt.xlabel('Timestep', fontsize=18)
    plt.ylabel('Reward', fontsize=18)

    means = np.mean(aggregated_tot_ep_rews, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()