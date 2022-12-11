import numpy as np
from matplotlib import pyplot as plt


def plot_incidents_missed_over_time_across_agents(tot_eval_data):
    """
    Plots average incidents missed over time
    aggregated_tot_ep_incidents_missed: the incidents missed per timestep per agent, should be {agent_name: tot_ep_incidents_missed} where
                            tot_ep_incidents_missed shape is (num_episodes, num_timesteps)
    Returns:
    """
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_incidents_missed = np.sum([tot_eval_data[name]['tot_incidents_missed'] for name in agent_names], axis=3)  # (num_agents, num_eps, num timesteps)

    timesteps = list(range(len(aggregated_tot_ep_incidents_missed[0][0])))
    num_eps = len(aggregated_tot_ep_incidents_missed[0])

    plt.title(f'Average Incidents Missed Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Incidents Missed')

    means = np.mean(aggregated_tot_ep_incidents_missed, axis=1) # Shape: (num human_designed_policies, num timesteps)
    maxs = np.max(aggregated_tot_ep_incidents_missed, axis=1) # Shape: (num human_designed_policies, num timesteps)
    mins = np.min(aggregated_tot_ep_incidents_missed, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(agent_names))]
    for i in range(len(agent_names)):
        # plt.plot(timesteps, means[i], color=colors[i], alpha=0.8)
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()