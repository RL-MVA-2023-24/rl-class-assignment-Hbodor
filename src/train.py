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

class ProjectAgent:
    def __init__(self, seed = 42):
        env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
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

    def load(self, file_name = "checkpoint_400.pth"):
        print(f"Scoring for {file_name} ...")
        self.qnetwork_local.load_state_dict(torch.load(file_name))
        self.qnetwork_target.load_state_dict(torch.load(file_name))




import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ProjectAgent2:
    def __init__(self, seed=42, tau=0.1, update_every=5):
        env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.seed = random.seed(seed)
        self.tau = tau
        self.update_every = update_every
        self.episode_count = 0  # To track the number of episodes for periodic updates

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
        
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = random.sample(self.memory, self.batch_size)
                self.learn(experiences, self.gamma)

        if done:
            self.episode_count += 1
            if self.episode_count % self.update_every == 0:
                self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)

    def load(self, file_name = "checkpoint_400.pth"):
        print(f"Scoring for {file_name} ...")
        self.qnetwork_local.load_state_dict(torch.load(file_name))
        self.qnetwork_target.load_state_dict(torch.load(file_name))

# Assuming the environment setup and necessary imports are already handled
# Initialize the agent here and proceed with the training loop as described





class RandomAgent():
    """
    Defines an interface for agents in a simulation or decision-making environment.

    An Agent must implement methods to act based on observations, save its state to a file,
    and load its state from a file. This interface uses the Protocol class from the typing
    module to specify methods that concrete classes must implement.

    Protocols are a way to define formal Python interfaces. They allow for type checking
    and ensure that implementing classes provide specific methods with the expected signatures.
    """

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        """
        Determines the next action based on the current observation from the environment.

        Implementing this method requires processing the observation and optionally incorporating
        randomness into the decision-making process (e.g., for exploration in reinforcement learning).

        Args:
            observation (np.ndarray): The current environmental observation that the agent must use
                                       to decide its next action. This array typically represents
                                       the current state of the environment.
            use_random (bool, optional): A flag to indicate whether the agent should make a random
                                         decision. This is often used for exploration. Defaults to False.

        Returns:
            int: The action to be taken by the agent.
        """
        # return random action from the action space of env
        return random.choice(range(env.action_space.n))

    def save(self, path: str) -> None:
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        pass

    def load(self) -> None:
        """
        Loads the agent's state from a file specified by the path (HARDCODED). This not a good practice,
        but it will simplify the grading process.

        This method should deserialize the saved state (e.g., model weights, configuration settings)
        from the file and restore the agent to this state. Implementations must ensure that the
        agent's state is compatible with the `act` method's expectations.

        Note:
            It's important to ensure that neural network models (if used) are loaded in a way that is
            compatible with the execution device (e.g., CPU, GPU). This may require specific handling
            depending on the libraries used for model implementation. WARNING: THE GITHUB CLASSROOM
        HANDLES ONLY CPU EXECUTION. IF YOU USE A NEURAL NETWORK MODEL, MAKE SURE TO LOAD IT IN A WAY THAT
        DOES NOT REQUIRE A GPU.
        """
        pass



if __name__ == "__main__":
    # Agent initialization
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ProjectAgent2(seed=0)

    # Training loop
    def train_dqn(n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []                        # List to save scores from each episode
        eps = eps_start                    # Initialize epsilon for epsilon-greedy action selection
        for i_episode in range(1, n_episodes+1):
            time_now = time.time()
            state = env.reset()[0]
            score = 0
            for t in range(max_t):
                action = agent.act(state, use_random=(np.random.rand() <= eps))
                next_state, reward, done, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break 
            scores.append(score)           # Save most recent score
            eps = max(eps_end, eps_decay*eps) # Decrease epsilon
            # Print the most recent score, teh time taken, value of epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores), time.time() - time_now, eps))
            #print(f"Episode {i_episode}\tAverage Score: {np.mean(scores):.2f}")
            if i_episode in [1, 2, 5, 20]:
                agent.save(f"checkpoint_{i_episode}.pth")
            if i_episode % 100 == 0:
                agent.save(f"checkpoint_{i_episode}.pth")
        return scores

    # Run training
    scores = train_dqn()



