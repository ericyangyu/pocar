from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import functools
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from absl import flags
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from yaml import full_load

import sys; sys.path.append('..')

from lending_experiment.agents.human_designed_policies import oracle_lending_agent
from lending_experiment.agents.human_designed_policies.classifier_agents import ScoringAgentParams
from lending_experiment.agents.human_designed_policies.threshold_policies import ThresholdPolicy
from lending_experiment.config import CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT, BURNIN, MAXIMIZE_REWARD, EQUALIZE_OPPORTUNITY, EP_TIMESTEPS, NUM_GROUPS, EVAL_ZETA_0, \
    EVAL_ZETA_1, TRAIN_TIMESTEPS, SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, EVAL_MODEL_PATHS, \
    CPO_EVAL_MODEL_PATHS
from lending_experiment.agents.cpo.cpo import CPO
from lending_experiment.agents.cpo.cpo_wrapper_env import CPOEnvWrapper
from lending_experiment.agents.cpo.models import build_diag_gauss_policy, build_bernouilli_policy, build_mlp
from lending_experiment.agents.cpo.simulators import SinglePathSimulator
from lending_experiment.agents.cpo.torch_utils.torch_utils import get_device
from lending_experiment.environments import params, rewards
from lending_experiment.environments.lending import DelayedImpactEnv
from lending_experiment.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending_experiment.environments.rewards import LendingReward
from lending_experiment.graphing.plot_bank_cash_over_time import plot_bank_cash_over_time
from lending_experiment.graphing.plot_confusion_matrix_over_time import plot_confusion_matrix_over_time
from lending_experiment.graphing.plot_fn_over_time import plot_fn_over_time
from lending_experiment.graphing.plot_fp_over_time import plot_fp_over_time
from lending_experiment.graphing.plot_loans_over_time import plot_loans_over_time
from lending_experiment.graphing.plot_rets import plot_rets
from lending_experiment.graphing.plot_rews_over_time import plot_rews_over_time
from lending_experiment.graphing.plot_tn_over_time import plot_tn_over_time
from lending_experiment.graphing.plot_tp_over_time import plot_tp_over_time
from lending_experiment.graphing.plot_tpr_gap_over_time import plot_tpr_gap_over_time
from lending_experiment.graphing.plot_tpr_over_time import plot_tpr_over_time
from lending_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from lending_experiment.agents.ppo.sb3.ppo import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()


def load_cpo_policy(model_path):
    policy_dims = [64, 64]
    # (7) OHE of credit score + (2) group +  (2) TPRs of each group
    state_dim = 11
    action_dim = 1

    # policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
    policy = build_bernouilli_policy(state_dim, policy_dims, action_dim)

    policy.to('cpu')

    ckpt = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state_dict'])

    return policy
def train_cpo(env_list):
    import sys
    sys.path.insert(0,'cpo/')

    config = full_load(open('cpo_config.yaml', 'r'))['lending']

    n_episodes = config['n_episodes']
    env_name = config['env_name']
    n_episodes = config['n_episodes']
    n_trajectories = config['n_trajectories']
    trajectory_len = config['max_timesteps']
    policy_dims = config['policy_hidden_dims']
    vf_dims = config['vf_hidden_dims']
    cf_dims = config['cf_hidden_dims']
    max_constraint_val = config['max_constraint_val']
    bias_red_cost = config['bias_red_cost']
    device = get_device()

    print(env_list[0].observation_space)
    print(env_list[0].action_space)
    print(env_list[0].reset())
    state_dim = env_list[0].observation_space.shape[0]
    # action_dim = env_list[0].action_space.n
    action_dim = 1

    # policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
    policy = build_bernouilli_policy(state_dim, policy_dims, action_dim)
    value_fun = build_mlp(state_dim + 1, vf_dims, 1)
    cost_fun = build_mlp(state_dim + 1, cf_dims, 1)

    policy.to(device)
    value_fun.to(device)
    cost_fun.to(device)

    simulator = SinglePathSimulator(env_list, policy, n_trajectories, trajectory_len)

    cpo = CPO(policy, value_fun, cost_fun, simulator, model_path='agents/cpo/save-dir/lending.pt',
              bias_red_cost=bias_red_cost, max_constraint_val=max_constraint_val)

    print(f'Training policy {env_name} environment...\n')

    cpo.train(n_episodes)

def train(train_timesteps, env):

    exp_exists = False
    if os.path.isdir(SAVE_DIR):
        exp_exists = True
        if input(f'{SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper(env=env, reward_fn=LendingReward)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = None
    should_load = False
    if exp_exists:
        resp = input(f'\nWould you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
        if resp != 'y' and resp != 'n':
            exit('Invalid response for resp: ' + resp)
        should_load = resp == 'y'

    if should_load:
        model_name = input(f'Specify the model you would like to load in. Do not include the .zip: ')
        model = PPO.load(EXP_DIR + "models/" + model_name, verbose=1, device=device)
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=1,
                    learning_rate=LEARNING_RATE,
                    device=device)

        shutil.rmtree(EXP_DIR, ignore_errors=True)
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=EXP_DIR))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(SAVE_DIR + '/final_model')

    # Once we finish learning, plot the returns over time and save into the experiments directory
    plot_rets(EXP_DIR)

def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, algorithm=None):
    print()
    print(f"Evaluating {name}")
    Path(f'{eval_path}/{name}/').mkdir(parents=True, exist_ok=True)
    eval_data = {
        'tot_loans': np.zeros((num_eps, NUM_GROUPS)),  # The number of loans per group per episode
        'tot_tp': np.zeros((num_eps, NUM_GROUPS)),  # The number of true positives, or no default given loan accepted, per group per episode
        'tot_fp': np.zeros((num_eps, NUM_GROUPS)),  # The number of false positives, or default given loan accepted, per group per episode
        'tot_tn': np.zeros((num_eps, NUM_GROUPS)),  # The number of true negatives, or default given loan rejected, per group per episode
        'tot_fn': np.zeros((num_eps, NUM_GROUPS)),  # The number of false negatives, or no default given loan rejected, per group per episode
        'tot_tpr': np.zeros((num_eps, NUM_GROUPS)),  # The TPR per group per episode
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_loans_over_time': np.zeros((num_eps, num_timesteps,  NUM_GROUPS)),  # The number of loans per group per timestep per episode
        'tot_bank_cash_over_time': np.zeros((num_eps, num_timesteps)),  # The amount of bank cash per timestep per episode
        'tot_tp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TP per group per timestep per episode
        'tot_fp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FP per group per timestep per episode
        'tot_tn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TN per group per timestep per episode
        'tot_fn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FN per group per timestep per episode
        'tot_tpr_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TPR per group per timestep per episode
    }

    reward_fn = LendingReward()

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False
        print(f'Episode {ep}:')
        for t in tqdm.trange(num_timesteps):
            will_default = env.state.will_default


            action = None
            if algorithm == 'cpo':
                action = int(agent(torch.FloatTensor(obs).squeeze()).sample().item())
            else:
                if isinstance(agent, PPO):
                    action = agent.predict(obs)[0]
                else:
                    action = agent.act(obs, done)

            # Logging
            group_id = np.argmax(env.state.group)
            # Add to loans if the agent wants to loan
            if action == 1:
                eval_data['tot_loans'][ep][group_id] += 1
                # Check if agent would default
                if env.state.will_default:
                    eval_data['tot_fp'][ep][group_id] += 1
                else:
                    eval_data['tot_tp'][ep][group_id] += 1
            elif action == 0:
                if env.state.will_default:
                    eval_data['tot_tn'][ep][group_id] += 1
                else:
                    eval_data['tot_fn'][ep][group_id] += 1

            # Update TPR for both groups per timestep.
            # In the edge case where denom is 0, set TPR to 0
            eval_data['tot_tpr'][ep] = np.divide(
                eval_data['tot_tp'][ep],
                eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep],
                out=np.zeros_like(eval_data['tot_tp'][ep]),
                where=(eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep])!=0)
            # Update total loans and TPR observed so far
            eval_data['tot_loans_over_time'][ep][t] = eval_data['tot_loans'][ep]
            eval_data['tot_tpr_over_time'][ep][t] = eval_data['tot_tpr'][ep]

            eval_data['tot_tp_over_time'][ep][t] = eval_data['tot_tp'][ep]
            eval_data['tot_fp_over_time'][ep][t] = eval_data['tot_fp'][ep]
            eval_data['tot_tn_over_time'][ep][t] = eval_data['tot_tn'][ep]
            eval_data['tot_fn_over_time'][ep][t] = eval_data['tot_fn'][ep]

            old_bank_cash = env.state.bank_cash

            obs, _, done, _ = env.step(action)

            bank_cash = env.state.bank_cash

            r = reward_fn(old_bank_cash=old_bank_cash,
                          bank_cash=bank_cash,
                          tpr=eval_data['tot_tpr'][ep],
                          zeta0=EVAL_ZETA_0,
                          zeta1=EVAL_ZETA_1)

            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_bank_cash_over_time'][ep][t] = bank_cash

            if done:
                break

    Path(f'{eval_path}/{name}/').mkdir(parents=True, exist_ok=True)
    for key in eval_data.keys():
        np.save(f'{eval_path}/{name}/{key}.npy', eval_data[key])

    return eval_data

def display_eval_results(eval_dir):
    tot_eval_data = {}
    agent_names = copy.deepcopy(next(os.walk(eval_dir))[1])
    for agent_name in agent_names:
        tot_eval_data[agent_name] = {}
        for key in next(os.walk(f'{eval_dir}/{agent_name}'))[2]:
            key = key.split('.npy')[0]
            tot_eval_data[agent_name][key] = np.load(f'{eval_dir}/{agent_name}/{key}.npy', allow_pickle=True)


    # Plot all agent evaluations
    plot_rews_over_time(tot_eval_data)
    plot_loans_over_time(tot_eval_data)
    plot_bank_cash_over_time(tot_eval_data)
    plot_tpr_over_time(tot_eval_data)
    plot_tpr_gap_over_time(tot_eval_data)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'cpo'])
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=None)
    parser.add_argument('--display_eval_path', dest='display_eval_path', type=str, default=None)
    parser.add_argument('--show_train_progress', action='store_true')
    args = parser.parse_args()

    env_params = DelayedImpactParams(
        applicant_distribution=two_group_credit_clusters(
            cluster_probabilities=CLUSTER_PROBABILITIES,
            group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
        bank_starting_cash=BANK_STARTING_CASH,
        interest_rate=INTEREST_RATE,
        cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
    )
    env = DelayedImpactEnv(env_params)

    if args.train:
        if args.algorithm == 'cpo':
            n_trajectories = full_load(open('cpo_config.yaml', 'r'))['lending']['n_trajectories']
            env_list = [CPOEnvWrapper(env=env, reward_fn=LendingReward, ep_timesteps=EP_TIMESTEPS) for _ in range(n_trajectories)]
            train_cpo(env_list)
        else:
            train(train_timesteps=TRAIN_TIMESTEPS, env=env)
        plot_rets(exp_path=EXP_DIR, save_png=True)

    if args.show_train_progress:
        plot_rets(exp_path=EXP_DIR, save_png=False)

    if args.display_eval_path is not None:
        display_eval_results(eval_dir=args.display_eval_path)

    if args.eval_path is not None:

        assert(args.eval_path is not None)
        p = Path(args.eval_path)
        if p.exists():
            resp = input(f'{args.eval_path} already exists; do you want to override it? (y/n): ')
            if resp != 'y':
                exit('Exiting.')

        # Initialize eval directory to store eval information
        shutil.rmtree(args.eval_path, ignore_errors=True)
        Path(args.eval_path).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 10
        eval_timesteps = 10000
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(args.eval_path + '/seeds.txt', 'w') as f:
            f.write(str(seeds))

        # First, evaluate PPO human_designed_policies
        for name, model_path in EVAL_MODEL_PATHS.items():
            env = DelayedImpactEnv(env_params)
            agent = PPO.load(model_path, verbose=1)
            evaluate(env=PPOEnvWrapper(env=env, reward_fn=LendingReward, ep_timesteps=eval_timesteps),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     seeds=seeds,
                     eval_path=args.eval_path)

        # Evaluate CPO agent
        for name, model_path in CPO_EVAL_MODEL_PATHS.items():
            agent = load_cpo_policy(model_path)
            evaluate(env=CPOEnvWrapper(env=env, reward_fn=LendingReward, ep_timesteps=eval_timesteps),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     seeds=seeds,
                     eval_path=args.eval_path,
                     algorithm='cpo')

        # Evaluate threshold policies
        for name, threshold_policy in zip(['EO', 'Greedy'], [EQUALIZE_OPPORTUNITY, MAXIMIZE_REWARD]):
            env = DelayedImpactEnv(env_params)
            agent_params = ScoringAgentParams(
                feature_keys=['applicant_features'],
                group_key='group',
                default_action_fn=(lambda: 1),
                burnin=BURNIN,
                convert_one_hot_to_integer=True,
                threshold_policy=threshold_policy,
                skip_retraining_fn=lambda action, observation: action == 0,
                cost_matrix=params.CostMatrix(
                    fn=0, fp=-1, tp=env_params.interest_rate, tn=0))

            agent = oracle_lending_agent.OracleThresholdAgent(
                action_space=env.action_space,
                reward_fn=rewards.BinarizedScalarDeltaReward(
                    'bank_cash', baseline=env.initial_params.bank_starting_cash),
                observation_space=env.observation_space,
                params=agent_params,
                env=env)

            evaluate(env=env,
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=threshold_policy.name,
                     seeds=seeds,
                     eval_path=args.eval_path)


if __name__ == '__main__':
    main()