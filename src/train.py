import numpy as np
import torch
import os
from environment import NetworkEnvironment
from agent import D3QNAgent
from baselines import RoundRobin, LeastConnection, RandomPolicy, MaxCapacity
from visualizer import plot_load_distribution, plot_learning_curves, plot_bar

# Advanced Configuration
CFG = {
    'lr': 1e-4, 
    'gamma': 0.99,          # High gamma for long-term planning
    'epsilon_start': 1.0, 
    'epsilon_decay': 0.992, # Slower decay for better exploration
    'epsilon_end': 0.02, 
    'batch_size': 128,      # Large batch to saturate GPU
    'hidden': 256, 
    'target_update': 10,
    'episodes': 2000, 
    'buffer_cap': 20000,    # Larger buffer for PER
    'learning_starts': 2000,
    'per_alpha': 0.6,
    'per_beta_start': 0.4,
    'per_beta_inc': 0.001,
    # Adaptive exploration
    'adaptive_exploration': True,
    'adapt_window': 50,
    'min_adapt_episodes': 100,
    'epsilon_min': 0.02,
    'epsilon_max': 1.0,
    'epsilon_inc': 0.02,
    'epsilon_dec': 0.03,
    'stagnation_tol': 0.01,
    'target_load_var': 0.02,
    'adaptive_mode': 'volatility',  # 'volatility' or 'trend'
    'base_eps': 0.02,
    'vol_coeff': 0.01,
    'load_var_eps': 0.06,
    'outage_eps': 0.07,
    # Optional cyclical pulse (disabled by default)
    'cycle_every': 0,
    'cycle_len': 0,
    'cycle_eps': 0.08,
    # Speed (RTX 3050 friendly)
    'amp': True,          # mixed precision on CUDA
    'pin_memory': True,   # faster CPU->GPU transfer for batches
    'non_blocking': True,
    'tf32': True,         # faster matmuls on Ampere+ (safe for RL)
    'compile': False,     # set True if torch>=2.0 and you want extra speed
}

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def jains_fairness(x: np.ndarray) -> float:
    """
    Jain's fairness index in (0,1], 1.0 is perfectly balanced.
    """
    x = np.asarray(x, dtype=np.float32)
    s1 = float(np.sum(x))
    s2 = float(np.sum(x * x))
    n = float(x.size)
    return (s1 * s1) / (n * s2 + 1e-8)

def evaluate_policy(env, policy, episodes=30, seed0=1000):
    rewards = []
    load_vars = []
    overload_rates = []
    switch_rates = []
    fairness = []
    for ep in range(episodes):
        state = env.reset(seed=seed0 + ep)
        total = 0.0
        lv_sum = 0.0
        overload_steps = 0
        switches = 0
        steps = 0
        done = False
        prev_action = None
        while not done:
            action = policy.select_action(state)
            if prev_action is not None and int(action) != int(prev_action):
                switches += 1
            prev_action = int(action)
            state, r, done, info = env.step(action)
            total += float(r)
            lv_sum += float(info["load_variance"])
            q_norm = info["queue_lengths"] / env.max_queue
            if bool(np.any(q_norm > 0.9)):
                overload_steps += 1
            fairness.append(jains_fairness(np.clip(info["utilizations"], 0.0, 1.0)))
            steps += 1
        rewards.append(total)
        load_vars.append(lv_sum / max(1, steps))
        overload_rates.append(overload_steps / max(1, steps))
        switch_rates.append(switches / max(1, steps - 1))
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_load_variance": float(np.mean(load_vars)),
        "avg_overload_rate": float(np.mean(overload_rates)),
        "avg_switch_rate": float(np.mean(switch_rates)),
        "avg_jain_fairness": float(np.mean(fairness)) if fairness else float("nan"),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "load_variances": np.asarray(load_vars, dtype=np.float32),
    }

def evaluate_d3qn(env, agent, episodes=30, seed0=1000):
    rewards = []
    load_vars = []
    overload_rates = []
    switch_rates = []
    fairness = []
    for ep in range(episodes):
        state = env.reset(seed=seed0 + ep)
        total = 0.0
        lv_sum = 0.0
        overload_steps = 0
        switches = 0
        steps = 0
        done = False
        prev_action = None
        while not done:
            action = agent.select_action(state, eval_mode=True)
            if prev_action is not None and int(action) != int(prev_action):
                switches += 1
            prev_action = int(action)
            state, r, done, info = env.step(action)
            total += float(r)
            lv_sum += float(info["load_variance"])
            q_norm = info["queue_lengths"] / env.max_queue
            if bool(np.any(q_norm > 0.9)):
                overload_steps += 1
            fairness.append(jains_fairness(np.clip(info["utilizations"], 0.0, 1.0)))
            steps += 1
        rewards.append(total)
        load_vars.append(lv_sum / max(1, steps))
        overload_rates.append(overload_steps / max(1, steps))
        switch_rates.append(switches / max(1, steps - 1))
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_load_variance": float(np.mean(load_vars)),
        "avg_overload_rate": float(np.mean(overload_rates)),
        "avg_switch_rate": float(np.mean(switch_rates)),
        "avg_jain_fairness": float(np.mean(fairness)) if fairness else float("nan"),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "load_variances": np.asarray(load_vars, dtype=np.float32),
    }

def rollout_utilization_episode(env, policy, seed=999):
    """
    Runs 1 episode and returns a (T, num_servers) utilization history.
    policy must implement select_action(state) OR be a D3QNAgent (uses eval_mode).
    """
    state = env.reset(seed=seed)
    util_history = []
    done = False
    while not done:
        if hasattr(policy, "select_action") and not isinstance(policy, D3QNAgent):
            action = policy.select_action(state)
        else:
            action = policy.select_action(state, eval_mode=True)
        state, _, done, info = env.step(action)
        util_history.append(np.clip(info["utilizations"], 0.0, 1.0))
    return np.asarray(util_history, dtype=np.float32)

def main():
    os.makedirs("results", exist_ok=True)
    set_seeds(42)
    env = NetworkEnvironment()
    agent = D3QNAgent(env.state_dim, env.action_dim, CFG)

    print(f"--- Starting Training on {agent.device} ---")

    # Baselines (pre-train reference)
    baselines = {
        "Random": RandomPolicy(env.action_dim, seed=1),
        "RoundRobin": RoundRobin(env.action_dim),
        "LeastConnection": LeastConnection(),
        "MaxCapacity": MaxCapacity(),
    }
    baseline_metrics = {k: evaluate_policy(env, p, episodes=25, seed0=2000) for k, p in baselines.items()}
    print("\n--- Baseline Eval (reward / load_var / overload_rate / switch_rate / jain) ---")
    for k, m in baseline_metrics.items():
        print(
            f"{k:14s}: {m['avg_reward']:8.2f} | "
            f"load_var: {m['avg_load_variance']:.4f} | "
            f"overload: {m['avg_overload_rate']:.3f} | "
            f"switch: {m['avg_switch_rate']:.3f} | "
            f"jain: {m['avg_jain_fairness']:.3f}"
        )
    
    ep_rewards = []
    ep_load_vars = []
    for ep in range(CFG['episodes']):
        state = env.reset()
        total_reward = 0
        lv_sum = 0.0
        steps = 0
        outage_seen = False
        
        for step in range(env.episode_horizon):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Use the agent's internal PER memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # The agent handles batching and GPU transfer internally
            loss = agent.train_step()
            
            state = next_state
            total_reward += reward
            lv_sum += float(info["load_variance"])
            outage_seen = outage_seen or bool(np.any(info["down_flags"] > 0.0))
            steps += 1
            if done: break

        if ep % CFG['target_update'] == 0:
            agent.update_target()

        ep_rewards.append(float(total_reward))
        ep_lv = lv_sum / max(1, steps)
        ep_load_vars.append(ep_lv)

        # Adaptive exploration update (replaces fixed decay when enabled)
        if getattr(agent, "adaptive_exploration", False):
            agent.adaptive_epsilon_update(total_reward, ep_lv, episode_outage=outage_seen)
        else:
            agent.decay_epsilon()
            
        if ep % 20 == 0:
            avg_lv = lv_sum / max(1, steps)
            print(f"Episode {ep:3d} | Reward: {total_reward:7.2f} | LoadVar: {avg_lv:.4f} | Eps: {agent.epsilon:.2f}")

    # --- INVISIBLE BENCHMARK & TEST ---
    ep_load_vars_arr = np.asarray(ep_load_vars, dtype=np.float32)
    overall_lv = float(np.mean(ep_load_vars_arr)) if len(ep_load_vars_arr) else float("nan")
    last_k = min(100, len(ep_load_vars_arr))
    last100_lv = float(np.mean(ep_load_vars_arr[-last_k:])) if last_k else float("nan")
    print(f"\n--- Training Load Variance Summary ---")
    print(f"Avg LoadVar (all episodes): {overall_lv:.4f}")
    print(f"Avg LoadVar (last {last_k} episodes): {last100_lv:.4f}")

    print("\n--- Training Complete. Running Test Episode ---")
    test_state = env.reset()
    q_history = []
    
    for _ in range(env.episode_horizon):
        action = agent.select_action(test_state, eval_mode=True)
        test_state, _, done, info = env.step(action)
        q_history.append(np.clip(info["queue_lengths"] / env.max_queue, 0.0, 1.0))
        if done:
            break
    
    plot_load_distribution(q_history, title="D3QN Queue Heatmap (normalized)")

    # Utilization heatmaps for baselines vs D3QN (same seed)
    print("\n--- Saving Utilization Heatmaps (same seed) ---")
    seed_vis = 4242
    for name, pol in baselines.items():
        util_hist = rollout_utilization_episode(env, pol, seed=seed_vis)
        plot_load_distribution(util_hist, title=f"{name} Utilization Heatmap")
    d3qn_util = rollout_utilization_episode(env, agent, seed=seed_vis)
    plot_load_distribution(d3qn_util, title="D3QN Utilization Heatmap")

    # Fair evaluation vs baselines on identical seeds
    d3qn_metrics = evaluate_d3qn(env, agent, episodes=25, seed0=2000)
    print("\n--- Final Eval (reward / load_var / overload_rate / switch_rate / jain) ---")
    for k, m in baseline_metrics.items():
        print(
            f"{k:14s}: {m['avg_reward']:8.2f} | "
            f"load_var: {m['avg_load_variance']:.4f} | "
            f"overload: {m['avg_overload_rate']:.3f} | "
            f"switch: {m['avg_switch_rate']:.3f} | "
            f"jain: {m['avg_jain_fairness']:.3f}"
        )
    print(
        f"{'D3QN':14s}: {d3qn_metrics['avg_reward']:8.2f} | "
        f"load_var: {d3qn_metrics['avg_load_variance']:.4f} | "
        f"overload: {d3qn_metrics['avg_overload_rate']:.3f} | "
        f"switch: {d3qn_metrics['avg_switch_rate']:.3f} | "
        f"jain: {d3qn_metrics['avg_jain_fairness']:.3f}"
    )

    scores = {k: m["avg_reward"] for k, m in baseline_metrics.items()}
    scores["D3QN"] = d3qn_metrics["avg_reward"]

    plot_learning_curves({"D3QN (train)": np.asarray(ep_rewards)}, title="D3QN Training Rewards")
    plot_bar(scores, title="Baselines vs D3QN (Avg Eval Reward)")

if __name__ == "__main__":
    main()