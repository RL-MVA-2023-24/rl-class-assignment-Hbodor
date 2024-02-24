from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Setting up the environment
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env = None, state_size = None, action_size = None, seed = 42):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        
        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.t_step = 0
        
    def act(self, state, use_random=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if use_random or random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        # Learning happens every four steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones))

        # Double DQN update
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Loss and optimization
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)

    def load(self, file_name):
        self.qnetwork_local.load_state_dict(torch.load(file_name))
        self.qnetwork_target.load_state_dict(torch.load(file_name))



if __name__ == "__main__":
    # Agent initialization
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(env, state_size, action_size, seed=0)

    # Training loop
    def train_dqn(n_episodes=2000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []                        # List to save scores from each episode
        eps = eps_start                    # Initialize epsilon for epsilon-greedy action selection
        t = time.time()
        for i_episode in range(1, n_episodes+1):
            state = env.reset()[0]
            score = 0
            for t in range(max_t):
                action = agent.act(state, use_random=(np.random.rand() <= eps))
                next_state, reward, done, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores.append(score)           # Save most recent score
            eps = max(eps_end, eps_decay*eps) # Decrease epsilon
            # Print the most recent score, teh time taken, value of epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores), time.time()-t, eps), end="")
            #print(f"Episode {i_episode}\tAverage Score: {np.mean(scores):.2f}")
            if i_episode % 100 == 0:
                agent.save(f"checkpoint_{i_episode}.pth")
        return scores

    # Run training
    scores = train_dqn()

