import torch
import random
import numpy as np
import torch.nn as nn
from copy import deepcopy

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



# class DQN(nn.Module):
#     def __init__(self, state_dim, n_action, nb_neurons=256, depth=5, activation='relu'):
#         super(DQN, self).__init__()
#         self.layers = nn.ModuleList([nn.Linear(state_dim, nb_neurons)])
        
#         # Map string to actual PyTorch activation function
#         activation_functions = {
#             'relu': nn.ReLU(),
#             'silu': nn.SiLU(),  # SiLU (Swish) activation function
#             'tanh': nn.Tanh()
#         }
#         self.activation = activation_functions.get(activation, nn.ReLU())  # Default to ReLU
        
#         # Add hidden layers based on the specified depth
#         for _ in range(depth - 1):
#             self.layers.extend([
#                 nn.Linear(nb_neurons, nb_neurons),
#                 deepcopy(self.activation)
#             ])
        
#         # Output layer
#         self.layers.append(nn.Linear(nb_neurons, n_action))
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x) if isinstance(layer, nn.Linear) else self.activation(x)
#         return x



class DQN(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons=256, depth=5, activation='relu'):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(state_dim, nb_neurons)])
        
        # Map string to actual PyTorch activation function
        activation_functions = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),  # SiLU (Swish) activation function
            'tanh': nn.Tanh()
        }
        self.activation = activation_functions.get(activation, nn.ReLU())  # Default to ReLU
        
        # Add hidden layers based on the specified depth
        for _ in range(depth - 1):
            self.layers.extend([
                nn.Linear(nb_neurons, nb_neurons),
                deepcopy(self.activation)
            ])
        
        # Output layer
        self.layers.append(nn.Linear(nb_neurons, n_action))
        self.network = nn.Sequential(*self.layers)
        print(self.network)
    
    def forward(self, x):
        return self.network(x)

class DQN(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons=256):
        super(DQN, self).__init__()
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        )


class DQN(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons=256, depth=5, activation='relu'):
        super(DQN, self).__init__()
        
        # Choose the activation function based on the input parameter
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),  # Also known as Swish
            'tanh': nn.Tanh()
        }
        selected_activation = activations.get(activation, nn.ReLU())
        
        # Initialize layers using nn.Sequential
        layers = []
        # Input layer
        layers.append(nn.Linear(state_dim, nb_neurons))
        layers.append(selected_activation)
        
        # Hidden layers created in a loop based on the specified depth
        for _ in range(1, depth):
            layers.append(nn.Linear(nb_neurons, nb_neurons))
            layers.append(deepcopy(selected_activation))
        
        # Output layer
        layers.append(nn.Linear(nb_neurons, n_action))
        
        # Convert the list of layers into a nn.Sequential module
        self.network = nn.Sequential(*layers)

        
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
