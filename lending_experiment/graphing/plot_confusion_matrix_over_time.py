import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())

    fig, axs = plt.subplots(len(agent_names))
    fig.tight_layout()
    if len(agent_names) == 1:
        axs = [axs]

    for name, ax in zip(agent_names, axs):
        tot_tp_over_time = tot_eval_data[name]['tot_tp_over_time']  # (num_eps, num_timesteps, num_groups)
        tot_fp_over_time = tot_eval_data[name]['tot_fp_over_time']
        tot_tn_over_time = tot_eval_data[name]['tot_tn_over_time']
        tot_fn_over_time = tot_eval_data[name]['tot_fn_over_time']

        timesteps = list(range(len(tot_tp_over_time[0])))
        num_eps = len(tot_tp_over_time)

        ax.set_title(f'Average Confusion Terms Over Time For {name} Agent')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Occurrences')

        tot_tp_means = np.mean(tot_tp_over_time, axis=0)
        tot_fp_means = np.mean(tot_fp_over_time, axis=0)
        tot_tn_means = np.mean(tot_tn_over_time, axis=0)
        tot_fn_means = np.mean(tot_fn_over_time, axis=0)

        # for means in range(means.shape[-1]):
        for metric, means in zip(['TP', 'FP', 'TN', 'FN'], [tot_tp_means, tot_fp_means, tot_tn_means, tot_fn_means]):
            for group_id in range(tot_tp_means.shape[1]):
                ax.plot(timesteps, means[:, group_id], label=f'{metric}: Group {group_id + 1}')

        ax.legend()

    plt.show()
