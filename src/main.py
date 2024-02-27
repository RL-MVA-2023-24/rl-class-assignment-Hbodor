import random
import os
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import DQNAgent, DuelingAgent, DuelingAgentWithDoubleQLearning, ProjectAgentWithBatchNorm   # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = DQNAgent()
    #file_name = "learning_rate=0.001_gamma=0.99_gradient_steps=10_update_target_freq=400_update_target_tau=0.005_update_target_strategy=replace_epsilon_decay_period=40000_model_DQN.pt"
    #file_name = "learning_rate=0.001_gamma=0.99_gradient_steps=10_update_target_freq=400_update_target_tau=0.005_update_target_strategy=replace_epsilon_decay_period=40000_model_dueling.pt"
    #file_name = "depth=5_activation=silu_learning_rate=0.001_gamma=0.99_gradient_steps=10_update_target_freq=100_update_target_tau=0.005_update_target_strategy=ema_epsilon_decay_period=40000_criterion=SmoothL1Loss()_model_DQN.pt"
    file_name = "best_model.pt"
    agent.load(file_name)
    # Keep the following lines to evaluate your agent unchanged.
    print("Scoring ...", file_name)
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    print(f"Score agent: {score_agent:.2e}")
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
