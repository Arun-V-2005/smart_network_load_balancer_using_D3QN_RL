import numpy as np

class NetworkEnvironment:
    """
    Discrete load-balancing environment.

    - Action: pick a server ID to route the *incoming batch* to.
    - Dynamics: non-stationary traffic, burstiness, time-varying capacities, and occasional outages.
    - Episode ends if any queue hits max_queue (hard overload).
    """
    def __init__(self, num_servers=5, max_queue=50, seed=42, episode_horizon=200):
        self.num_servers = num_servers
        self.max_queue = max_queue
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.episode_horizon = int(episode_horizon)

        # Static per-server baseline capacity (requests/step)
        self.base_capacities = self.rng.integers(8, 15, size=num_servers).astype(np.float32)
        self.max_capacity = float(self.base_capacities.max()) * 1.25

        # State: [q_norm, cap_norm, down_flag] * N + [time_phase, prev_action_norm]
        self.state_dim = num_servers * 3 + 2
        self.action_dim = num_servers
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.prev_action = 0
        self.queue_lengths = np.zeros(self.num_servers, dtype=np.float32)
        self.capacities = self.base_capacities.copy()
        self.down_steps_remaining = np.zeros(self.num_servers, dtype=np.int32)
        self.down_flags = np.zeros(self.num_servers, dtype=np.float32)
        return self._get_state()

    def step(self, action):
        action = int(action)
        self.t += 1

        incoming = self._generate_traffic()
        switch_cost = 0.02 if action != self.prev_action else 0.0

        # Route all incoming requests to chosen server.
        self.queue_lengths[action] += incoming
        
        # Update outage processes and time-varying capacities, then process.
        self._update_outages()
        self._update_capacities()
        processed = np.minimum(self.queue_lengths, self.capacities)
        self.queue_lengths -= processed

        # Utilization: fraction of capacity used this step (0..1), safe for down servers.
        utilizations = (processed / np.maximum(self.capacities, 1e-6)).astype(np.float32)
        load_variance = float(np.var(utilizations))
        
        reward = self._calculate_reward(action)
        reward -= float(switch_cost)

        done = bool(np.any(self.queue_lengths >= self.max_queue) or (self.t >= self.episode_horizon))
        info = {
            "capacities": self.capacities.copy(),
            "down_flags": self.down_flags.copy(),
            "queue_lengths": self.queue_lengths.copy(),
            "utilizations": utilizations,
            "load_variance": load_variance,
        }
        self.prev_action = action
        return self._get_state(), reward, done, info

    def _get_state(self):
        q_norm = np.clip(self.queue_lengths / self.max_queue, 0.0, 1.5)
        cap_norm = np.clip(self.capacities / self.max_capacity, 0.0, 1.5)
        down = self.down_flags
        per_server = np.stack([q_norm, cap_norm, down], axis=1).reshape(-1).astype(np.float32)  # (3N,)

        # Time feature to make non-stationarity learnable
        phase = np.float32((self.t % self.episode_horizon) / max(1, self.episode_horizon - 1))
        prev_a = np.float32(self.prev_action / max(1, self.num_servers - 1))
        return np.concatenate([per_server, np.array([phase, prev_a], dtype=np.float32)], axis=0)

    def _generate_traffic(self):
        """
        Scalar arrivals per step with non-stationary rate + burstiness.
        """
        # Slowly varying load (sinusoid) + baseline
        phase = (2.0 * np.pi) * (self.t / max(1, self.episode_horizon))
        lam = 4.0 + 3.0 * (0.5 + 0.5 * np.sin(phase))  # in [4,7]
        arrivals = float(self.rng.poisson(lam=lam))
        if self.rng.random() < 0.12:  # burst chance
            arrivals += float(self.rng.integers(10, 25))
        return np.float32(arrivals)

    def _update_outages(self):
        # decrement existing outages
        self.down_steps_remaining = np.maximum(self.down_steps_remaining - 1, 0)
        self.down_flags = (self.down_steps_remaining > 0).astype(np.float32)

        # new outage events (rare)
        if self.rng.random() < 0.03:
            srv = int(self.rng.integers(self.num_servers))
            self.down_steps_remaining[srv] = int(self.rng.integers(5, 20))
            self.down_flags[srv] = 1.0

    def _update_capacities(self):
        # time-varying capacity with noise; outages reduce to ~0
        noise = self.rng.normal(loc=0.0, scale=0.5, size=self.num_servers).astype(np.float32)
        caps = self.base_capacities + noise
        caps = np.clip(caps, 1.0, self.max_capacity)
        caps = np.where(self.down_flags > 0.0, 0.25, caps)
        self.capacities = caps.astype(np.float32)

    def _calculate_reward(self, action):
        q_norm = self.queue_lengths / self.max_queue
        latency = float(q_norm[action])
        imbalance = float(np.var(q_norm))
        overload = float(np.sum(q_norm > 0.9)) * 0.75
        down_penalty = 1.0 if self.down_flags[action] > 0.0 else 0.0
        return float(-1.2 * latency - 0.6 * imbalance - overload - down_penalty)