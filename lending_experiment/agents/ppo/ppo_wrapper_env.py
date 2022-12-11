import gym
import numpy as np
import torch
from gym import spaces

from lending_experiment.config import EP_TIMESTEPS, ZETA_0, ZETA_1


class PPOEnvWrapper(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               ep_timesteps=EP_TIMESTEPS):
    super(PPOEnvWrapper, self).__init__(env)

    self.observation_space = spaces.Box(
      low=np.inf,
      high=np.inf,
      # (7) OHE of credit score + (2) group +  (2) TPRs of each group
      shape=(env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
    )

    self.action_space = spaces.Discrete(n=2)

    self.env = env
    self.reward_fn = reward_fn()

    self.timestep = 0
    self.ep_timesteps = ep_timesteps

    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

  def process_observation(self, obs):
    credit_score = obs['applicant_features']
    group = obs['group']

    return np.concatenate(
      (credit_score,
       group,
       self.tpr,
       ),
      axis=0
    )

  def compute_tpr(self, tp, fn):
    # tp: true positive, 2-dimensional for 2 groups
    # fn: false negative, 2-dimensional for 2 groups
    return np.divide(
      tp,
      tp + fn,
      out=np.zeros_like(tp),
      where=(tp + fn) != 0)

  def reset(self):
    self.timestep = 0
    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

    return self.process_observation(self.env.reset())

  def step(self, action):
    old_delta = self.delta

    # Update instance variables before we step the environment
    group_id = np.argmax(self.env.state.group)
    if action == 1:
      # Check if agent would default
      if self.env.state.will_default:
        self.fp[group_id] += 1
      else:
        self.tp[group_id] += 1
    elif action == 0:
      if self.env.state.will_default:
        self.tn[group_id] += 1
      else:
        self.fn[group_id] += 1
    self.tpr = self.compute_tpr(tp=self.tp,
                                fn=self.fn)
    self.old_bank_cash = self.env.state.bank_cash

    # Update delta terms
    self.delta = np.abs(self.tpr[0] - self.tpr[1])
    self.delta_delta = self.delta - old_delta


    obs, _, done, info = self.env.step(action)

    r = self.reward_fn(old_bank_cash=self.old_bank_cash,
                       bank_cash=self.env.state.bank_cash,
                       tpr=self.tpr,
                       zeta0=ZETA_0,
                       zeta1=ZETA_1)

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True

    return self.process_observation(obs), r, done, info
