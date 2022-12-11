import argparse
import copy
import random
import time
import os
import shutil
from pathlib import Path

import numpy as np
from sympy import E
import torch
import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from attention_allocation_experiment.agents.human_designed_policies.allocation_agents import MLEGreedyAgent, MLEGreedyAgentParams
from config import N_LOCATIONS, INCIDENT_RATES, N_ATTENTION_UNITS, DYNAMIC_RATE, EVAL_ZETA_0, EVAL_ZETA_1, EVAL_ZETA_2, \
    SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, TRAIN_TIMESTEPS, EVAL_MODEL_PATHS, CPO_EVAL_MODEL_PATHS
from config import OBS_HIST_LEN
from environments.attention_allocation import LocationAllocationEnv, Params
from environments.rewards import AttentionAllocationReward
from graphing.plot_att_all_over_time_across_agents import plot_att_all_over_time_across_agents
from graphing.plot_deltas_over_time_across_agents import plot_deltas_over_time_across_agents
from graphing.plot_incidents_missed_over_time_across_agents import plot_incidents_missed_over_time_across_agents
from graphing.plot_incidents_seen_over_time_across_agents import plot_incidents_seen_over_time_across_agents
from graphing.plot_rews import plot_rets
from graphing.plot_rew_over_time_across_agents import plot_rew_over_time_across_agents
from graphing.plot_rew_terms_over_time_across_agents import plot_rew_terms_over_time_across_agents
from graphing.plot_true_rates_over_time_across_agents import plot_true_rates_over_time_across_agents
from attention_allocation_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from attention_allocation_experiment.agents.ppo.sb3.ppo import PPO

# For CPO
from yaml import full_load
from attention_allocation_experiment.agents.cpo.models import build_diag_gauss_policy, build_mlp
from attention_allocation_experiment.agents.cpo.torch_utils.torch_utils import get_device
from attention_allocation_experiment.agents.cpo.cpo_wrapper_env import CPOEnvWrapper
from attention_allocation_experiment.agents.cpo.simulators import SinglePathSimulator
from attention_allocation_experiment.agents.cpo.cpo import CPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


def load_cpo_policy(model_path):
    policy_dims = [64, 64]
    state_dim = N_LOCATIONS * 4 * OBS_HIST_LEN
    action_dim = N_LOCATIONS

    policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)

    policy.to('cpu')

    ckpt = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state_dict'])

    return policy

def train_cpo(env_list):
    import sys
    sys.path.insert(0,'cpo/')

    config = full_load(open('cpo_config.yaml', 'r'))['attention_allocation']

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

    state_dim = N_LOCATIONS * 4 * OBS_HIST_LEN
    action_dim = N_LOCATIONS

    policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
    value_fun = build_mlp(state_dim + 1, vf_dims, 1)
    cost_fun = build_mlp(state_dim + 1, cf_dims, 1)

    policy.to(device)
    value_fun.to(device)
    cost_fun.to(device)

    simulator = SinglePathSimulator(env_list, policy, n_trajectories, trajectory_len, params=env_params)

    cpo = CPO(policy, value_fun, cost_fun, simulator, model_path='agents/cpo/save-dir/cpo_agent.pt',
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

    env = PPOEnvWrapper(env=env,
                        reward_fn=AttentionAllocationReward)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = None
    should_load = False
    if exp_exists:
        resp = input(f'Would you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
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
    plot_rews(EXP_DIR)

def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, algorithm=None):
    print(f'\nEvaluating {name}!\n')

    Path(f'{eval_path}/{name}/').mkdir(parents=True, exist_ok=True)

    reward_fn = AttentionAllocationReward()

    eval_data = {
        'tot_rews': np.zeros((num_eps, num_timesteps)),  # The rewards per timestep per episode
        'tot_att_all': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The attention allocated per site per timestep per episode
        'tot_true_rates': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The true rates per site per timestep per episode
        'tot_deltas': np.zeros((num_eps, num_timesteps)),  # The deltas per timestep per episode
        'tot_incidents_seen': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents seen per site per timestep per episode
        'tot_incidents_occurred': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents occurred per site per timestep per episode
        'tot_incidents_missed': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents missed per site per timestep per episode
        'tot_rew_infos': []  # The values of each term in the reward per timestep per episode, shape is (num_eps, num_timesteps, dict)
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        ep_data = {
            'rews': np.zeros(num_timesteps),  # The reward per timestep of this episode
            'att_all': np.zeros((num_timesteps, env.state.params.n_locations)),  # The attention allocated per site per timestep of this episode
            'true_rates': np.zeros((num_timesteps, env.state.params.n_locations)),  # The true rates per site per timestep of this episode
            'deltas': np.zeros(num_timesteps),  # The deltas per timestep of this episode
            'ep_incidents_seen': np.zeros((num_timesteps, env.state.params.n_locations)),  # The incidents seen per site per timestep of this episode
            'ep_incidents_occurred': np.zeros((num_timesteps, env.state.params.n_locations)),  # The incidents occurred per site per timestep of this episode
            'rew_infos': [],  # The values of each term in the reward per timestep of this episode
        }

        obs = env.reset()
        done = False

        print(f'{name} EPISODE {ep}:')
        for t in tqdm.trange(num_timesteps):

            action = None
            if algorithm == 'cpo':
                act = agent(torch.FloatTensor(obs).squeeze())
                action = act.mean.detach().cpu().numpy()
            else:
                if isinstance(agent, MLEGreedyAgent):
                    action = agent.act(obs, done)
                elif isinstance(agent, PPO):
                    action = agent.predict(obs)[0]

            obs, _, done, _ = env.step(action)

            # Update total incidents variables
            ep_data['ep_incidents_seen'][t] = env.state.incidents_seen
            ep_data['ep_incidents_occurred'][t] = env.state.incidents_occurred

            r = reward_fn(incidents_seen=env.state.incidents_seen,
                          incidents_occurred=env.state.incidents_occurred,
                          ep_incidents_seen=ep_data['ep_incidents_seen'],
                          ep_incidents_occurred=ep_data['ep_incidents_occurred'],
                          zeta0=EVAL_ZETA_0,
                          zeta1=EVAL_ZETA_1,
                          zeta2=EVAL_ZETA_2,
                          )

            ep_data['rew_infos'].append(reward_fn.rew_info)
            ep_data['rews'][t] = r
            ep_data['att_all'][t] = env.process_action(action) if isinstance(env, PPOEnvWrapper) or algorithm=='cpo' else action
            ep_data['true_rates'][t] = env.state.params.incident_rates
            ep_data['deltas'][t] = reward_fn.calc_delta(ep_incidents_seen=ep_data['ep_incidents_seen'],
                                                        ep_incidents_occurred=ep_data['ep_incidents_occurred'])

            if done:
                break

        # Store the episodic data in eval data
        eval_data['tot_rews'][ep] = ep_data['rews']
        eval_data['tot_att_all'][ep] = ep_data['att_all']
        eval_data['tot_true_rates'][ep] = ep_data['true_rates']
        eval_data['tot_deltas'][ep] = ep_data['deltas']
        eval_data['tot_incidents_seen'][ep] = ep_data['ep_incidents_seen']
        eval_data['tot_incidents_occurred'][ep] = ep_data['ep_incidents_occurred']
        eval_data['tot_incidents_missed'][ep] = ep_data['ep_incidents_occurred'] - ep_data['ep_incidents_seen']
        eval_data['tot_rew_infos'].append(copy.deepcopy(ep_data['rew_infos']))

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

    # Plot all human_designed_policies evaluations
    plot_rew_over_time_across_agents(tot_eval_data)
    plot_incidents_seen_over_time_across_agents(tot_eval_data)
    plot_incidents_missed_over_time_across_agents(tot_eval_data)
    plot_att_all_over_time_across_agents(tot_eval_data)
    plot_true_rates_over_time_across_agents(tot_eval_data)
    plot_deltas_over_time_across_agents(tot_eval_data)
    plot_rew_terms_over_time_across_agents(tot_eval_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'cpo'])
    parser.add_argument('--show_train_progress', action='store_true')
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=None)
    parser.add_argument('--display_eval_path', dest='display_eval_path', type=str, default=None)
    args = parser.parse_args()

    env_params = Params(
        n_locations=N_LOCATIONS,
        prior_incident_counts=tuple(500 for _ in range(N_LOCATIONS)),
        incident_rates=INCIDENT_RATES,
        n_attention_units=N_ATTENTION_UNITS,
        miss_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
        extra_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
        dynamic_rate=DYNAMIC_RATE)

    # Initialize the environment
    env = LocationAllocationEnv(params=env_params)

    # Train the PPO model
    if args.train:
        if args.algorithm == 'cpo':
            n_trajectories = full_load(open('cpo_config.yaml', 'r'))['attention_allocation']['n_trajectories']
            env_list = [CPOEnvWrapper(LocationAllocationEnv(params=env_params), reward_fn=AttentionAllocationReward) for _ in range(n_trajectories)]
            train_cpo(env_list)
        else:
            train(train_timesteps=TRAIN_TIMESTEPS, env=env)
        plot_rets(exp_path=EXP_DIR, save_png=True)

    if args.show_train_progress:
        plot_rews(exp_path=EXP_DIR, save_png=False)

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
        eval_timesteps = 1000
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(args.eval_path + '/seeds.txt', 'w') as f:
            f.write(str(seeds))

        # First, evaluate PPO human_designed_policies
        for name, model_path in EVAL_MODEL_PATHS.items():
            agent = PPO.load(model_path, verbose=1)
            evaluate(env=PPOEnvWrapper(env=env, reward_fn=AttentionAllocationReward),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     seeds=seeds,
                     eval_path=args.eval_path)

        # Evaluate CPO agent
        for name, model_path in CPO_EVAL_MODEL_PATHS.items():
            agent = load_cpo_policy(model_path)
            evaluate(env=CPOEnvWrapper(env=env, reward_fn=AttentionAllocationReward),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     seeds=seeds,
                     eval_path=args.eval_path,
                     algorithm='cpo')

        # Next, Evaluate the greedy agent
        model_class = MLEGreedyAgent
        agent_params = MLEGreedyAgentParams(
            burn_steps=25, window=100)
        agent = model_class(
            action_space=env.action_space,
            reward_fn=env.reward_fn,
            observation_space=env.observation_space,
            params=agent_params)
        evaluate(env=env,
                 agent=agent,
                 num_eps=eval_eps,
                 num_timesteps=eval_timesteps,
                 name='Purely Greedy',
                 seeds=seeds,
                 eval_path=args.eval_path)

print('Finished!')

