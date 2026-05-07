import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────
# Dueling Q-Network
# ──────────────────────────────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """
    Architecture: Shared features -> V(s) stream & A(s,a) stream.
    Splitting the network allows the agent to learn state values 
    independently of specific actions.
    """
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        
        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # State Value Stream - V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        
        # Advantage Stream - A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        v = self.value_stream(features)
        a = self.advantage_stream(features)
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))


# ──────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay (PER)
# ──────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Stores transitions and prioritizes them based on TD-error.
    Highly efficient for learning from rare events (like traffic bursts).
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0=uniform, 1=full)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance Sampling weights to correct bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5


# ──────────────────────────────────────────────────────────────────────
# D3QN Agent
# ──────────────────────────────────────────────────────────────────────
class D3QNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        # Hardware setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent initialized on: {self.device}")
        
        # Networks
        self.online = DuelingQNetwork(state_dim, action_dim, cfg['hidden']).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim, cfg['hidden']).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg['lr'])
        self.memory = PrioritizedReplayBuffer(cfg['buffer_cap'])
        
        # Hyperparameters
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon_start']
        self.epsilon_decay = cfg['epsilon_decay']
        self.epsilon_end = cfg['epsilon_end']
        self.batch_size = cfg['batch_size']
        self.target_update = cfg['target_update']
        self.beta = 0.4  # Initial importance sampling weight
        self.episode_count = 0

    def select_action(self, state, eval_mode=False):
        # Epsilon-greedy
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.online.advantage_stream[-1].out_features - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online(state_t).argmax().item()

    def train_step(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0.0

        # Increase beta over time (closer to 1.0)
        self.beta = min(1.0, self.beta + 0.001)
        
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Move tensors to GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q-values
        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Logic
        with torch.no_grad():
            # 1. Online network picks the best action for next_state
            next_actions = self.online(next_states).argmax(dim=1)
            # 2. Target network evaluates that action
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Update PER priorities based on TD-error
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Prioritized Loss
        loss = (weights * (current_q - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)