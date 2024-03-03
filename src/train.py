from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


import random
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    # act greedy
    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            action = torch.argmax(Q).item()
        self.model.train()  # Switch back to training mode if necessary
        return action


    def save(self, path):
        self.path = path + f"model_{self.model_name_suffix}.pt"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.path = r"C:\Users\hamza\OneDrive\Documents\Python Scripts\RL_repos_to_test\rl-class-assignment-Thomas-Risola-main\rl-class-assignment-Thomas-Risola-main\model2.pt"
        self.model = self.network({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return 

    ## MODEL ARCHITECTURE
    # this work meh => 100 episode to see something good happening askip
    #train for 100 episode => start validation after 100 ep 
    def network(self, config, device):

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        nb_neurons=256 #go try 256? 512? 1024 ? idea stack one more layer for fun :) :)

        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons), # try this after ?
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)

        return DQN

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

    def gradient_step_v2(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            Q_target_Ymax = self.target_model(Y).max(1)[0].detach()
            Q_Ymax = self.model(Y).max(1)[0].detach()
            next_Q = torch.min(Q_target_Ymax, Q_Ymax)
            update = torch.addcmul(R, 1-D, next_Q, value=self.gamma)
            Q_target_XA = self.target_model(X).gather(1, A.to(torch.long).unsqueeze(1))
            Q_XA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            
            loss = self.criterion(Q_target_XA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
            loss = self.criterion(Q_XA, update.unsqueeze(1))
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step() 
    
    def train(self):
        self.model_name_suffix = "DQN"

        ## CONFIGURE NETWORK
        # DQN config (change here for better results?)
        config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'buffer_size': 100000,
                'epsilon_min': 0.01,
                'epsilon_max': 1.,
                'epsilon_decay_period': 20000,
                'epsilon_delay_decay': 500,
                'batch_size': 1000,
                'gradient_steps': 4,
                'update_target_strategy': 'replace', # or 'ema'
                'update_target_freq': 400,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss()}

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
        epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
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

        max_episode = 2000 #epoch #maximum around 100 i guess

        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0

        ## TRAIN NETWORK
        print("Training the agent for ", max_episode, " episodes.")
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# Assuming the ProjectAgent class and other necessary imports are already defined as provided

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        # Common feature layer
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Outputs V(s)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # Outputs A(s, a)
        )
        
    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine values and advantages to get Q-values
        # Use the mean as an estimator to stabilize learning
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

class DuelingAgent(ProjectAgent):
    def network(self, config, device):
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


class ProjectAgentWithBatchNorm(ProjectAgent):
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