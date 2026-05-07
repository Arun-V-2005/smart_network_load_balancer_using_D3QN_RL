import numpy as np

def _split_state(state):
    """
    State layout: [q_norm, cap_norm, down]*N + [phase, prev_action]
    Returns q_norm, cap_norm, down as (N,) arrays.
    """
    state = np.asarray(state, dtype=np.float32)
    n = int((len(state) - 2) // 3)
    per = state[: 3 * n]
    return per[0::3], per[1::3], per[2::3]

class RoundRobin:
    def __init__(self, num_servers):
        self.num_servers = num_servers
        self.index = 0
    def select_action(self, state):
        action = self.index % self.num_servers
        self.index += 1
        return action

class LeastConnection:
    def select_action(self, state):
        q_norm, _, down = _split_state(state)
        # avoid routing to down servers unless all are down
        masked = np.where(down > 0.0, np.inf, q_norm)
        if np.all(np.isinf(masked)):
            masked = q_norm
        return int(np.argmin(masked))

class RandomPolicy:
    def __init__(self, num_servers, seed=0):
        self.num_servers = num_servers
        self.rng = np.random.default_rng(int(seed))
    def select_action(self, state):
        return int(self.rng.integers(self.num_servers))

class MaxCapacity:
    def select_action(self, state):
        _, cap, down = _split_state(state)
        masked = np.where(down > 0.0, -np.inf, cap)
        if np.all(np.isneginf(masked)):
            masked = cap
        return int(np.argmax(masked))