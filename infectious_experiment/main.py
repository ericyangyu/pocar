import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import tqdm
from absl import flags
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from networkx.algorithms import community

import sys; sys.path.append('..')

from infectious_experiment.agents.human_designed_policies import infectious_disease_agents
from infectious_experiment.agents.human_designed_policies.infectious_disease_agents import CentralityAgent, RandomAgent, MaxNeighborsAgent
from infectious_experiment.config import INFECTION_PROBABILITY, INFECTED_EXIT_PROBABILITY, NUM_TREATMENTS, BURNIN, \
    SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, TRAIN_TIMESTEPS, GRAPH_NAME, EVAL_MODEL_PATHS, \
    EVAL_ZETA_1, EVAL_ZETA_0, CPO_EVAL_MODEL_PATHS
from infectious_experiment.environments import infectious_disease, rewards
from infectious_experiment.environments.rewards import InfectiousReward, calc_percent_healthy
from infectious_experiment.graphing.plot_delta_over_time import plot_delta_over_time
from infectious_experiment.graphing.plot_percent_healthy_over_time import plot_percent_healthy_over_time
from infectious_experiment.graphing.plot_percent_sick_over_time import plot_percent_sick_over_time
from infectious_experiment.graphing.plot_rets import plot_rets
from infectious_experiment.graphing.plot_rews_over_time import plot_rews_over_time
from infectious_experiment.agents.ppo.ppo_wrapper_env import PPOEnvWrapper
from infectious_experiment.agents.ppo.sb3.ppo import PPO


# For CPO
from yaml import full_load
from infectious_experiment.agents.cpo.models import build_diag_gauss_policy, build_mlp, build_categorical_policy
from infectious_experiment.agents.cpo.torch_utils.torch_utils import get_device
from infectious_experiment.agents.cpo.cpo_wrapper_env import CPOEnvWrapper
from infectious_experiment.agents.cpo.simulators import SinglePathSimulator
from infectious_experiment.agents.cpo.cpo import CPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


GRAPHS = {'karate': nx.karate_club_graph()}


def render(env,
           color_map,
           mode='human',
           savedir=None,
           **kwargs):
    if mode != 'human':
        raise ValueError('Unsupported mode \'%s\'.' % mode)
    colors = [
        color_map[health_state] for health_state in env.state.health_states]
    nx.draw_circular(env.state.population_graph, node_color=colors, with_labels=True)
    if savedir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(savedir)
        plt.close()


def train_cpo(env_list):
    import sys
    sys.path.insert(0,'cpo/')

    config = full_load(open('cpo_config.yaml', 'r'))['infectious']

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

    print(env_list[0].observation_space.shape)
    print(env_list[0].action_space.n)
    print(env_list[0].reset())
    state_dim = env_list[0].observation_space.shape[0]
    action_dim = env_list[0].action_space.n

    policy = build_categorical_policy(state_dim, policy_dims, action_dim)
    value_fun = build_mlp(state_dim + 1, vf_dims, 1)
    cost_fun = build_mlp(state_dim + 1, cf_dims, 1)

    policy.to(device)
    value_fun.to(device)
    cost_fun.to(device)

    simulator = SinglePathSimulator(env_list, policy, n_trajectories, trajectory_len)

    cpo = CPO(policy, value_fun, cost_fun, simulator, model_path='agents/cpo/save-dir/infectious.pt',
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

    env = PPOEnvWrapper(env=env, reward_fn=InfectiousReward)
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

def evaluate(env, agent, num_eps, num_timesteps, name, should_render, seeds, eval_path, algorithm=None):
    print()
    print(f"Evaluating {name}")

    Path(f'{eval_path}/{name}/').mkdir(parents=True, exist_ok=True)

    reward_fn = InfectiousReward()
    eval_data = {
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_percent_sick_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_percent_healthy_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_deltas_over_time': np.zeros((num_eps, num_timesteps)),  # The delta per timestep per episode
    }

    communities_generator = community.girvan_newman(env.state.population_graph)
    communities = tuple(sorted(c) for c in next(communities_generator))
    num_communities = len(communities)
    # Map individuals in the graph to a community
    communities_map = {
        individual: comm_i for comm_i, comm in enumerate(communities) for individual in comm
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        # Keep track of how many vaccines go to each community
        num_vaccines_per_community = np.zeros(num_communities)
        # Newly infected in each community
        num_newly_infected_per_community = np.zeros(num_communities)

        obs = env.reset()
        done = False

        print(f'Episode {ep}:')
        if should_render:
            Path(f"./evaluation_plots/{name}/ep_{ep}").mkdir(parents=True, exist_ok=True)

        for t in tqdm.trange(num_timesteps):
            color_map = {
                0: 'gray',
                1: 'red',
                2: 'green'
            }
            if should_render:
                render(env,
                       color_map,
                       mode='human',
                       savedir=f'./evaluation_plots/{name}/ep_{ep}/timestep_{t}')

            prev_health_states = copy.deepcopy(env.state.health_states)

            action = None
            a = None  # action placeholder for PPO after processing it
            if algorithm == 'cpo':
                action = agent(torch.FloatTensor(obs).squeeze())
                action = torch.argmax(action.probs)
                action = action.detach().cpu().numpy()
                a = env.process_action(action)
            else:
                if isinstance(agent, PPO):
                    action = agent.predict(obs)[0]
                    a = env.process_action(action)
                else:
                    action = agent.act(obs, done)
                    a = action


            obs, _, done, _ = env.step(action)

            # Update the number of vaccines in each community
            if a is not None:
                comm_i = communities_map[np.array([a]).flatten()[0]]
                num_vaccines_per_community[comm_i] += 1
            # Compute newly infected
            for i, (health_state, prev_health_state) in enumerate(
                    zip(env.state.health_states, prev_health_states)):
                # 1 is the index in self.env.state.params.state_names for infected
                if health_state == 1 and health_state != prev_health_state:
                    comm_i = communities_map[i]
                    num_newly_infected_per_community[comm_i] += 1

            r = reward_fn(health_states=env.state.health_states,
                          num_vaccines_per_community=num_vaccines_per_community,
                          num_newly_infected_per_community=num_newly_infected_per_community,
                          eta0=EVAL_ZETA_0,
                          eta1=EVAL_ZETA_1)


            percent_healthy = calc_percent_healthy(env.state.health_states)
            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_percent_sick_over_time'][ep][t] = 1 - percent_healthy
            eval_data['tot_percent_healthy_over_time'][ep][t] = percent_healthy
            eval_data['tot_deltas_over_time'][ep][t] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community,
                                                                            num_newly_infected_per_community=num_newly_infected_per_community)

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


    # Plot all human_designed_policies evaluations
    plot_rews_over_time(tot_eval_data)
    plot_percent_sick_over_time(tot_eval_data)
    # plot_percent_healthy_over_time(tot_eval_data)
    plot_delta_over_time(tot_eval_data)

def vis_karate_graph():
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    G = nx.karate_club_graph()

    communities_generator = community.girvan_newman(G)
    communities = tuple(sorted(c) for c in next(communities_generator))
    print(communities)

    ##### Plot edge betweenness #####
    colors = list(nx.betweenness_centrality(G).values())
    # min max normalize colors
    colors = np.array(colors)
    colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

    cmap = plt.cm.Greens

    nx.draw_circular(G, node_color=cmap(colors), with_labels=True, ax=ax1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  # only needed for matplotlib < 3.1

    divider = make_axes_locatable(ax1)
    colorbar_axes = divider.append_axes("left",
                                        size="5%",
                                        pad=0.1)
    # Using new axes for colorbar
    f.colorbar(sm, cax=colorbar_axes)

    ax1.set_title('Betweenness Centrality', fontsize=30)

    ##### Plot communities #####

    color_map = {
        0: 'magenta',
        1: 'turquoise',
    }
    colors = []
    for i, comm in enumerate(communities):
        color = color_map[i]
        colors += [color for _ in range(len(comm))]

    patch1 = mpatches.Patch(color='magenta', label='Community 1')
    patch2 = mpatches.Patch(color='turquoise', label='Community 2')

    nx.draw_circular(G, node_color=colors, with_labels=True, ax=ax2)
    ax2.set_title('Communities', fontsize=30)
    ax2.legend(handles=[patch1, patch2], loc='lower right', fontsize='x-large')


    f.set_size_inches(h=10, w=25)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.savefig("infectious_betweenness_and_communities_plot.png", bbox_inches='tight')

def load_cpo_policy(model_path, state_dim=None, action_dim=None):
    policy_dims = [64, 64]

    # policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
    policy = build_categorical_policy(state_dim, policy_dims, action_dim)
    
    policy.to('cpu')

    ckpt = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(ckpt['policy_state_dict'])

    return policy
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'cpo'])
    parser.add_argument('--show_train_progress', action='store_true')
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--vis_karate_graph', action='store_true')
    parser.add_argument('--display_eval_path', dest='display_eval_path', type=str, default=None)
    args = parser.parse_args()

    graph = GRAPHS[GRAPH_NAME]
    # Randomly initialize a node to infected
    initial_health_state = [0 for _ in range(graph.number_of_nodes())]
    initial_health_state[0] = 1
    env = infectious_disease.build_sir_model(
        population_graph=graph,
        infection_probability=INFECTION_PROBABILITY,
        infected_exit_probability=INFECTED_EXIT_PROBABILITY,
        num_treatments=NUM_TREATMENTS,
        max_treatments=1,
        burn_in=BURNIN,
        # Treatments turn susceptible people into recovered without having them
        # get sick.
        treatment_transition_matrix=np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [0, 0, 1]]),
        initial_health_state = copy.deepcopy(initial_health_state)
    )
    
    if args.train:
        if args.algorithm == 'cpo':
            n_trajectories = full_load(open('cpo_config.yaml', 'r'))['infectious']['n_trajectories']
            env_list = [CPOEnvWrapper(infectious_disease.build_sir_model(
                                        population_graph=graph,
                                        infection_probability=INFECTION_PROBABILITY,
                                        infected_exit_probability=INFECTED_EXIT_PROBABILITY,
                                        num_treatments=NUM_TREATMENTS,
                                        max_treatments=1,
                                        burn_in=BURNIN,
                                        # Treatments turn susceptible people into recovered without having them
                                        # get sick.
                                        treatment_transition_matrix=np.array([[0, 0, 1],
                                                                            [0, 1, 0],
                                                                            [0, 0, 1]]),
                                        initial_health_state = copy.deepcopy(initial_health_state)
                                    ), reward_fn=InfectiousReward) for _ in range(n_trajectories)]
            train_cpo(env_list)
        else:
            train(train_timesteps=TRAIN_TIMESTEPS, env=env)
        plot_rets(exp_path=EXP_DIR, save_png=True)

    if args.show_train_progress:
        plot_rets(exp_path=EXP_DIR, save_png=False)

    if args.display_eval_path is not None:
        display_eval_results(eval_dir=args.display_eval_path)

    if args.vis_karate_graph:
        vis_karate_graph()

    if args.eval_path is not None:

        assert(args.eval_path is not None)
        p = Path(args.eval_path)
        if p.exists():
            resp = input(f'{args.eval_path} already exists; do you want to override it? (y/n): ')
            if resp != 'y':
                exit('Exiting.')

        if args.render:
            # Set up render directory
            shutil.rmtree('./evaluation_plots/', ignore_errors=True)
            Path('./evaluation_plots/').mkdir(parents=True, exist_ok=True)

        # Initialize eval directory to store eval information
        shutil.rmtree(args.eval_path, ignore_errors=True)
        Path(args.eval_path).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 200
        eval_timesteps = 20
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(args.eval_path + '/seeds.txt', 'w') as f:
            f.write(str(seeds))

        # First, evaluate PPO human_designed_policies
        for name, model_path in EVAL_MODEL_PATHS.items():
            # Set up agent render directory
            if args.render:
                Path(f'./evaluation_plots/{name}').mkdir(parents=True, exist_ok=True)
            agent = PPO.load(model_path, verbose=1)
            evaluate(env=PPOEnvWrapper(env=env, reward_fn=InfectiousReward, ep_timesteps=eval_timesteps),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     should_render=args.render,
                     seeds=seeds,
                     eval_path=args.eval_path)

        # Evaluate CPO agent
        for name, model_path in CPO_EVAL_MODEL_PATHS.items():
            agent = load_cpo_policy(model_path, state_dim=graph.number_of_nodes() * 3, action_dim=graph.number_of_nodes() + 1)
            evaluate(env=CPOEnvWrapper(env=env, reward_fn=InfectiousReward),
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     should_render=args.render,
                     seeds=seeds,
                     eval_path=args.eval_path,
                     algorithm="cpo")

        for agent_class, name in zip([RandomAgent, MaxNeighborsAgent],
                                     ['Random', 'Max Neighbors']):
            agent = agent_class(
                env.action_space,
                rewards.NullReward(),
                env.observation_space,
                params=infectious_disease_agents.env_to_agent_params(
                    env.initial_params))
            evaluate(env=env,
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     should_render=args.render,
                     seeds=seeds,
                     eval_path=args.eval_path)




if __name__ == '__main__':
    main()