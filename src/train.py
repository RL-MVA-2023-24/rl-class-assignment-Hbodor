from gymnasium.wrappers import TimeLimit
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from train_utils import ReplayBuffer, DuelingDQN, DQN

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.$

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class DQNAgent:
    # act greedy
    def __init__(self):
        self.model = None
        self.target_model = None
        self.memory = None
        self.optimizer = None
        self.optimizer2 = None
        self.gamma = None
        self.batch_size = None
        self.nb_actions = None
        self.config = None
        self.path = ""
        self.model_name_prefix = ""
        self.model_name_suffix = "DQN"
        self.best_model = None

    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            action = torch.argmax(Q).item()
        return action


    def save(self, path):
        # prefix based on the values of self.config
        self.path = path + f"{self.model_name_prefix}_{self.model_name_suffix}.pt"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self, file_name = "model2.pt"):
        device = torch.device('cpu')
        self.path = ""
        self.model = self.network({}, device)
        self.model.load_state_dict(torch.load(self.path + file_name, map_location=device))
        self.model.eval()
        return 

    def network(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = config.get('nb_neurons', 256)  # Default to 256 if not specified
        depth = config.get('depth', 5)  # Default depth is 5
        activation = config.get('activation', 'relu')  # Default activation function is ReLU
        # Initialize the DQN model with the specified parameters
        model = DQN(state_dim=state_dim, n_action=n_action, nb_neurons=nb_neurons, depth = depth, activation = activation).network.to(device)
        return model

    ## UTILITY FUNCTIONS
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, max_episode = 200):
        ## CONFIGURE NETWORK
        # DQN config (change here for better results?)
        config = {'nb_actions': env.action_space.n,
                  'activation': 'relu',
                  'depth' : 2,
                    'learning_rate': 0.001,
                    'gamma': 0.99,
                    'buffer_size': 100000,
                    'epsilon_min': 0.01,
                    'epsilon_max': 1.,
                    'epsilon_decay_period': 15000,
                    'epsilon_delay_decay': 1000,
                    'batch_size': 200,
                    'gradient_steps': 5,
                    'update_target_strategy': 'replace', # or 'ema'
                    'update_target_freq': 100,
                    'update_target_tau': 0.005,
                    'criterion': torch.nn.SmoothL1Loss()}
        # Select the keys you want to include in the prefix
        keys = ['depth', 'activation', 'learning_rate', 'gamma', "gradient_steps", "update_target_freq", "update_target_tau", "update_target_strategy", "epsilon_decay_period", "criterion"]

        # Create the prefix by concatenating the key-value pairs
        prefix = '_'.join(f'{key}={config[key]}' for key in keys)

        # Now you can use this prefix when naming your model
        model_name = f'{prefix}_model'
        self.model_name_prefix = model_name
        self.config = config
        # network
        device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.model = self.network(config, device)
        self.target_model = deepcopy(self.model).to(device)

        # 
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        # epsilon greedy strategy
        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_stop = config['epsilon_decay_period'] 
        epsilon_delay = config['epsilon_delay_decay']
        epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop

        # memory buffer
        self.memory = ReplayBuffer(config['buffer_size'], device)

        # learning parameters (loss, lr, optimizer, gradient step)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        
        nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        # target network
        update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        previous_val = 0
        ## INITIATE NETWORK
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0

        ## TRAIN NETWORK
        print("Training the agent for ", max_episode, " episodes.")
        print(f"Model config is {config}")
        while episode < max_episode:
            # update epsilon
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon-epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if update_target_strategy == 'replace':
                if step % update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > 0:
                    validation_score = evaluate_HIV(agent=self, nb_episode=1)
                else :
                    validation_score = 0
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:.2e}'.format(episode_cum_reward),
                      # evaluation score 
                      ", validation score ", '{:.2e}'.format(validation_score),
                      sep='')
                state, _ = env.reset()
                # EARLY STOPPING => works really well
                if validation_score >= previous_val:
                   print("better model")
                   self.path = ""
                   self.save(self.path)
                   previous_val = validation_score
                   self.best_model = deepcopy(self.model).to(device)
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state


        self.model.load_state_dict(self.best_model.state_dict())
        #self.path = f"C:\Users\hamza\OneDrive\Documents\Python Scripts\RL_repos_to_test\rl-class-assignment-Thomas-Risola-main\rl-class-assignment-Thomas-Risola-main"
        self.path = ""
        self.save(self.path)
        return episode_return



class DuelingAgent(DQNAgent):
    def model_definition(self, config, device):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.model_name_suffix = "dueling"
        hidden_dim = config.get('hidden_dim', 256)  # Default to 256 if not specified
        
        # Initialize the Dueling DQN network with the specified dimensions
        model = DuelingDQN(input_dim=state_dim, output_dim=action_dim, hidden_dim=hidden_dim).to(device)
        return model



class DuelingAgentWithDoubleQLearning(DuelingAgent):
    def gradient_step(self):
        self.model_name_suffix = "double_dueling"
        if len(self.memory) > self.batch_size:
            # Sample a batch from the replay buffer
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            
            # Compute Q values for next states using the online model
            Q_next = self.model(Y)
            # Select the best actions in the next states using the online model
            best_actions = Q_next.max(1)[1].unsqueeze(1)
            
            # Compute Q values for the chosen actions in the next states using the target model
            Q_target_next = self.target_model(Y).gather(1, best_actions)
            
            # Compute the target values
            update = R + self.gamma * Q_target_next * (1 - D)
            
            # Compute current Q values
            Q_current = self.model(X).gather(1, A.long().unsqueeze(1))
            
            # Compute loss
            loss = self.criterion(Q_current, update)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Other methods from DuelingAgent remain unchanged


class ProjectAgentWithBatchNorm(DQNAgent):
    def network(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = config.get('nb_neurons', 256)  # Default to 256 if not specified
        self.model_name_suffix = "batch_norm"
        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.BatchNorm1d(nb_neurons),  # Batch normalization layer
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.BatchNorm1d(nb_neurons),  # Batch normalization layer
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.BatchNorm1d(nb_neurons),  # Batch normalization layer
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.BatchNorm1d(nb_neurons),  # Batch normalization layer
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),  # Additional layer if needed
            nn.BatchNorm1d(nb_neurons),  # Batch normalization layer
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        ).to(device)

        return DQN


if __name__ == "__main__":
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = DQNAgent()
    agent.train(max_episode=400)
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score_training.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")