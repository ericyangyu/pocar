import numpy as np
from matplotlib import pyplot as plt


def plot_tpr_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())

    for an in agent_names:
        tpr_over_time = np.mean(tot_eval_data[an]['tot_tpr_over_time'], axis=0)
        timesteps = np.arange(tpr_over_time.shape[0])
        for group in range(tpr_over_time.shape[1]):
            plt.plot(timesteps, tpr_over_time[:, group], label=f'{an}: Group {group + 1}')

    plt.title('Average TPR Over Time Across Agents')
    plt.xlabel('Timestep')
    plt.ylabel('Average TPR')
    plt.legend()
    plt.show()