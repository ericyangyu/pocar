import numpy as np
from matplotlib import pyplot as plt


def plot_bank_cash_over_time(tot_eval_data):
    agent_names = list(tot_eval_data.keys())
    aggregated_tot_ep_bank_cash = [tot_eval_data[name]['tot_bank_cash_over_time'] for name in agent_names]

    timesteps = list(range(len(aggregated_tot_ep_bank_cash[0][0])))
    num_eps = len(aggregated_tot_ep_bank_cash[0])

    plt.title(f'Average Bank Cash Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Bank Cash')

    means = np.mean(aggregated_tot_ep_bank_cash, axis=1) # Shape: (num human_designed_policies, num timesteps)

    # Plot means and add a legend
    # colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(agent_names))]
    for i in range(len(agent_names)):
        plt.plot(timesteps, means[i], alpha=0.8)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()
