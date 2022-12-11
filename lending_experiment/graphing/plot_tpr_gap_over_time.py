import numpy as np
from matplotlib import pyplot as plt


def plot_tpr_gap_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())

    for an in agent_names:
        tpr_over_time = np.mean(tot_eval_data[an]['tot_tpr_over_time'], axis=0)
        tpr_gap_over_time = np.abs(np.subtract(tpr_over_time[:, 0], tpr_over_time[:, 1]))
        timesteps = np.arange(tpr_gap_over_time.shape[0])
        plt.plot(timesteps, tpr_gap_over_time, label=f'{an}')

    plt.title('Average Delta Over Time Across Agents')
    plt.xlabel('Timestep')
    plt.ylabel('Average Delta')
    plt.legend()
    plt.show()
