import numpy as np
from matplotlib import pyplot as plt


def plot_rews_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_rews = [tot_eval_data[name]['tot_rews_over_time'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_rews[0][0])))
    num_eps = len(aggregated_tot_ep_rews[0])

    plt.title(f'Average Reward Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    means = np.mean(aggregated_tot_ep_rews, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(agent_names))]
    for i in range(len(agent_names)):
        # plt.plot(timesteps, means[i], color=colors[i], alpha=0.8)
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()
