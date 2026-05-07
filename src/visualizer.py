import matplotlib
# Force non-interactive backend to avoid blocking training runs (Windows/headless).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

_SHOW = os.environ.get("SHOW_PLOTS", "0") == "1"

def plot_load_distribution(history, title="Server Utilization Heatmap"):
    data = np.array(history).T  # Shape: (Servers, Steps)
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(label='Utilization (0-1)')
    plt.ylabel('Server ID')
    plt.xlabel('Time Step')
    plt.title(title)
    
    if not os.path.exists("results"): os.makedirs("results")
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    print(f"Plot saved to results/{title.replace(' ', '_').lower()}.png")
    if _SHOW:
        plt.show()
    plt.close()

def plot_learning_curves(curves, title="Average Reward per Episode"):
    """
    curves: dict[str, np.ndarray] mapping label -> rewards over episodes
    """
    plt.figure(figsize=(12, 5))
    for label, y in curves.items():
        y = np.asarray(y, dtype=np.float32)
        plt.plot(y, label=label, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()

    if not os.path.exists("results"):
        os.makedirs("results")
    out = f"results/{title.replace(' ', '_').lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    if _SHOW:
        plt.show()
    plt.close()

def plot_bar(scores, title="Evaluation: Avg Reward (higher is better)"):
    labels = list(scores.keys())
    vals = [scores[k] for k in labels]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals, color=["#888888"] * (len(labels) - 1) + ["#4C78A8"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Avg reward")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)

    if not os.path.exists("results"):
        os.makedirs("results")
    out = f"results/{title.replace(' ', '_').lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    if _SHOW:
        plt.show()
    plt.close()