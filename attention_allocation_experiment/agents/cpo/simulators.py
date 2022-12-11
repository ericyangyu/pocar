from collections import defaultdict, namedtuple
import numpy as np
import torch

from agents.cpo.autoassign import autoassign
from agents.cpo.memory import Memory, Trajectory
from agents.cpo.torch_utils.torch_utils import get_device



class Simulator:
    @autoassign(exclude=('env_list', 'env_args'))
    def __init__(self, env_list, policy, n_trajectories, trajectory_len, obs_filter=None, **env_args):
        # NOTE: DEPRECATED param: n_trajectories. Fixed to 1
        # assert n_trajectories == 1, "No longer supports n_trajectories != 1"
        self.env = np.asarray(env_list)
        self.n_trajectories = len(env_list)

        for env in self.env:
            env._max_episode_steps = trajectory_len

        self.device = get_device()


class SinglePathSimulator:
    def __init__(self, env_list, policy, n_trajectories, trajectory_len, state_filter=None,
                 **env_args):
        Simulator.__init__(self, env_list, policy, n_trajectories, trajectory_len, state_filter,
                           **env_args)

    def run_sim(self):
        self.policy.eval()

        with torch.no_grad():
            trajectories = np.asarray([Trajectory() for i in range(self.n_trajectories)])
            continue_mask = np.ones(self.n_trajectories)

            for env, trajectory in zip(self.env, trajectories):
                obs = torch.tensor(env.reset()).float()

                # Maybe batch this operation later
                if self.obs_filter:
                    obs = self.obs_filter(obs)

                trajectory.observations.append(obs)

            while np.any(continue_mask):
                continue_indices = np.where(continue_mask)
                trajs_to_update = trajectories[continue_indices]
                continuing_envs = self.env[continue_indices]

                policy_input = torch.stack([torch.tensor(trajectory.observations[-1]).to(self.device)
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(policy_input)
                actions = action_dists.sample()
                actions = actions.cpu()

                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    obs, reward, trajectory.done, info = env.step(action.numpy())

                    obs = torch.tensor(obs).float()
                    reward = torch.tensor(reward, dtype=torch.float)
                    cost = torch.tensor(info['constraint_cost'], dtype=torch.float)

                    if self.obs_filter:
                        obs = self.obs_filter(obs)

                    trajectory.actions.append(action)
                    trajectory.rewards.append(reward)
                    trajectory.costs.append(cost)

                    if not trajectory.done:
                        trajectory.observations.append(obs)

                continue_mask = np.asarray([1 - trajectory.done for trajectory in trajectories])

        memory = Memory(trajectories)

        return memory
