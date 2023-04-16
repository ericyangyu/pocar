import os

import pandas
from matplotlib import pyplot as plt

def plot_rets(exp_path, save_png=False):
    if not os.path.isdir(exp_path):
        exit(f"{exp_path} not found!!!")

    df = pandas.read_csv(f'./{exp_path}/progress.csv')
    xs = df['time/total_timesteps']
    ys = df['rollout/ep_rew_mean']

    plt.plot(xs, ys)
    plt.title('PPO Training: Average Episodic Reward Over Time')
    plt.xlabel('Total Timesteps Trained So Far')
    plt.ylabel('Average Episodic Reward')
    if save_png:
        plt.savefig(exp_path + 'train_rew_over_time')
    else:
        plt.show()

    plt.close()