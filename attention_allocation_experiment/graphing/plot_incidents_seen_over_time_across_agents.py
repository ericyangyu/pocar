import numpy as np
from matplotlib import pyplot as plt


def plot_incidents_seen_over_time_across_agents(tot_eval_data):
    """
    Plots average deltas over time
    aggregated_tot_ep_incidents_seen: the delta per timestep per agent, should be {agent_name: tot_ep_deltas} where
                            tot_ep_rews shape is (num_episodes, num_timesteps)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_incidents_seen = [tot_eval_data[name]['tot_incidents_seen'] for name in agent_names]  # (num_agents, num_eps, num timesteps, num locations)
    aggregated_tot_ep_incidents_seen = np.sum(aggregated_tot_ep_incidents_seen, axis=3)

    timesteps = list(range(len(aggregated_tot_ep_incidents_seen[0][0])))
    num_eps = len(aggregated_tot_ep_incidents_seen[0])

    plt.title(f'Average Incidents Seen Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Incidents Seen')

    means = np.mean(aggregated_tot_ep_incidents_seen, axis=1) # Shape: (num human_designed_policies, num timesteps)
    maxs = np.max(aggregated_tot_ep_incidents_seen, axis=1) # Shape: (num human_designed_policies, num timesteps)
    mins = np.min(aggregated_tot_ep_incidents_seen, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(agent_names))]
    for i in range(len(agent_names)):
        # plt.plot(timesteps, means[i], color=colors[i], alpha=0.8)
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()