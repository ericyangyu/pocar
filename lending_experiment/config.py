import torch

from lending_experiment.agents.human_designed_policies.threshold_policies import ThresholdPolicy

########## Experiment Setup Parameters ##########
# EXP_DIR = './experiments/advantage_regularized_ppo/'
EXP_DIR = './experiments/advantage_regularized_ppo/'
SAVE_DIR = f'{EXP_DIR}/models/'
# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': './experiments/advantage_regularized_ppo/models/rl_model_100000_steps',

}
# CPO model paths to evaluate
CPO_EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'CPO': 'cpo/save-dir/cpo_agent.pt'
}

########## Env Parameters ##########
DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)
NUM_GROUPS = 2
GROUP_0_PROB = 0.5
BANK_STARTING_CASH= 10000
INTEREST_RATE = 1
CLUSTER_SHIFT_INCREMENT= 0.01
CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS
EP_TIMESTEPS = 2000
MAXIMIZE_REWARD = ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = ThresholdPolicy.EQUALIZE_OPPORTUNITY

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 10_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])])
SAVE_FREQ = 10000
REGULARIZE_ADVANTAGE = True
# Weights for delta bank cash and delta terms in the reward for the lending environment
ZETA_0 = 1
ZETA_1 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)
# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.25
BETA_2 = 0.25
# Threshold for delta
OMEGA = 0.005

########## Eval Parameters ##########
# Weights for delta bank cash and delta terms in the reward for the lending environment
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 2
BURNIN = 0  # Number of steps before applying the threshold policy.

