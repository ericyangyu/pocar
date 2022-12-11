import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/advantage_regularized_ppo/'
SAVE_DIR = f'{EXP_DIR}/models/'

# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': './experiments/advantage_regularized_ppo/models/rl_model_200000_steps',

}
# CPO model paths to evaluate
CPO_EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'CPO': 'cpo/save-dir/cpo_agent.pt'
}

########## Env Parameters ##########
INFECTION_PROBABILITY = 0.5
INFECTED_EXIT_PROBABILITY = 0.005
NUM_TREATMENTS = 1
BURNIN = 1
GRAPH_NAME = 'karate'
EP_TIMESTEPS = 20  # Number of steps in the experiment.

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 10_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [512, 512, dict(vf=[512, 256], pi=[512, 256])])
SAVE_FREQ = 10000
REGULARIZE_ADVANTAGE = True  # Regularize advantage?
# Weights for percent healthy and delta term in reward
ZETA_0 = 1
ZETA_1 = 0
# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.1
BETA_2 = 0.1
# Threshold for delta
OMEGA = 0.05

########## Eval Parameters ##########
# Weights for percent healthy term and delta term in reward
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0.1
