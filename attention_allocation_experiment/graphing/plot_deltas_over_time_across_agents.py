import numpy as np
from matplotlib import pyplot as plt


def plot_deltas_over_time_across_agents(tot_eval_data):
    """
    Plots average deltas over time
    aggregated_tot_ep_deltas: the delta per timestep per agent, should be {agent_name: tot_ep_deltas} where
                            tot_ep_rews shape is (num_episodes, num_timesteps)
    """
    agent_names = list(tot_eval_data.keys())

    aggregated_tot_ep_deltas = [tot_eval_data[name]['tot_deltas'] for name in agent_names]  # (num_agents, num_eps, num timesteps)

    timesteps = list(range(len(aggregated_tot_ep_deltas[0][0])))
    num_eps = len(aggregated_tot_ep_deltas[0])

    plt.title(f'Average Delta Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Delta')

    means = np.mean(aggregated_tot_ep_deltas, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    for i in range(len(agent_names)):
        # plt.plot(timesteps, means[i], color=colors[i], alpha=0.8)
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()

    # Next, show the average delta for each agent
    plt.title('Average Delta Mean')
    plt.ylabel('Delta')
    plt.bar(agent_names, np.mean(means, axis=1))
    plt.show()
    plt.close()