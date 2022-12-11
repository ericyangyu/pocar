import numpy as np
from matplotlib import pyplot as plt


def plot_percent_healthy_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_rews = [tot_eval_data[name]['tot_percent_healthy_over_time'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_rews[0][0])))
    num_eps = len(aggregated_tot_ep_rews[0])

    plt.title(f'Average Percent Healthy Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Percent Healthy')

    means = np.mean(aggregated_tot_ep_rews, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()
