import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/advantage_regularized_ppo/'
SAVE_DIR = f'{EXP_DIR}/models/'

# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': './experiments/advantage_regularized_ppo/models/final_model',

}
# CPO model paths to evaluate
CPO_EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'CPO': 'cpo/save-dir/cpo_agent.pt'
}

########## Env Parameters ##########
N_LOCATIONS = 5
N_ATTENTION_UNITS = 6
EP_TIMESTEPS = 1000
INCIDENT_RATES = [8, 6, 4, 3, 1.5]
DYNAMIC_RATE = 0.1

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 10_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [128, 128, dict(vf=[128, 64], pi=[128, 64])])  # actor-critic architecture
SAVE_FREQ = 10000  # save frequency in timesteps
REGULARIZE_ADVANTAGE = True  # Regularize advantage?
# Weights for incidents seen, missed incidents, and delta in reward for the attention allocation environment
ZETA_0 = 1
ZETA_1 = 0.25
ZETA_2 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)
# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.15
BETA_2 = 0.15
# Threshold for delta
OMEGA = 0.05
# Number of timesteps remembered in observation history
OBS_HIST_LEN = 8

########## Eval Parameters ##########
# Weights for incidents seen, missed incidents, and delta in reward for the attention allocation environment
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0.25
EVAL_ZETA_2 = 10
# How many timesteps in the past the observation history should include
EVAL_OBS_HIST_LEN = 8
