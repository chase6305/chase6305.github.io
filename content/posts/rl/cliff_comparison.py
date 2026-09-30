"""Reproduce the cliff comparison in the RL article. Requires NumPy and Matplotlib.

The step, epsilon_greedy and train functions match the article example.
"""
import numpy as np

# 4x12：起点 36，终点 47，悬崖 37..46。
# 掉崖奖励 -100 并回到起点，但 episode 不终止；到达终点才终止。
def step(state, action):
    row, col = divmod(state, 12)
    dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
    row, col = np.clip(row + dr, 0, 3), np.clip(col + dc, 0, 11)
    next_state = int(row * 12 + col)
    if 37 <= next_state <= 46:
        return 36, -100.0, False
    return next_state, -1.0, next_state == 47

def epsilon_greedy(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    choices = np.flatnonzero(q[state] == q[state].max())
    return int(rng.choice(choices))

def train(method, episodes=500, seed=42, alpha=0.5, gamma=1.0, epsilon=0.1):
    if method not in ("sarsa", "q_learning"):
        raise ValueError("unknown method")
    rng = np.random.default_rng(seed)
    q = np.zeros((48, 4))
    returns = []
    for _ in range(episodes):
        state, total = 36, 0.0
        action = epsilon_greedy(q, state, rng, epsilon)
        for _ in range(10000):
            next_state, reward, terminated = step(state, action)
            next_action = epsilon_greedy(q, next_state, rng, epsilon)
            bootstrap = (q[next_state, next_action] if method == "sarsa"
                         else q[next_state].max())
            target = reward + gamma * (0.0 if terminated else bootstrap)
            q[state, action] += alpha * (target - q[state, action])
            total += reward
            if terminated:
                break
            state, action = next_state, next_action
        else:
            raise RuntimeError("episode exceeded the demonstration step budget")
        returns.append(total)
    return q, np.array(returns)

def evaluate(q, epsilon, seed, episodes=100, max_steps=200):
    """Frozen Q table, fresh random stream; no learning during evaluation."""
    rng = np.random.default_rng(seed)
    scores, falls, successes = [], [], []
    for _ in range(episodes):
        state, total, count = 36, 0.0, 0
        for _ in range(max_steps):
            action = epsilon_greedy(q, state, rng, epsilon)
            state, reward, terminated = step(state, action)
            total += reward
            count += int(reward == -100.0)
            if terminated:
                break
        successes.append(bool(terminated))
        scores.append(total)
        falls.append(count)
    return float(np.mean(scores)), float(np.mean(falls)), float(np.mean(successes))


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("cliff-results"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    if args.seeds < 2 or args.episodes < 100:
        parser.error("use at least two seeds and 100 training episodes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for method in ("sarsa", "q_learning"):
        records, histories = [], []
        for seed in range(args.seeds):
            q, history = train(method, episodes=args.episodes, seed=seed)
            before = q.copy()
            greedy, greedy_falls, greedy_success = evaluate(q, 0., 100000+seed)
            exploratory, exploratory_falls, exploratory_success = evaluate(q, .1, 200000+seed)
            np.testing.assert_array_equal(q, before)
            records.append({"seed": seed,
                            "training_last_100": float(history[-100:].mean()),
                            "greedy_evaluation": greedy,
                            "exploratory_evaluation": exploratory,
                            "greedy_success_rate": greedy_success,
                            "exploratory_success_rate": exploratory_success,
                            "greedy_falls_per_episode": greedy_falls,
                            "exploratory_falls_per_episode": exploratory_falls})
            histories.append(history)
        summaries = {key: {"mean": float(np.mean([r[key] for r in records])),
                           "seed_std": float(np.std([r[key] for r in records], ddof=1))}
                     for key in records[0] if key != "seed"}
        results[method] = {"summary": summaries, "per_seed": records}
        curves[method] = np.asarray(histories)
        print(method, json.dumps(summaries), flush=True)
    report = {"numpy": np.__version__, "training_seeds": list(range(args.seeds)),
              "training_episodes": args.episodes, "alpha": .5, "gamma": 1.,
              "training_epsilon": .1, "evaluation_episodes_per_seed": 100,
              "greedy_seed_offset": 100000, "exploratory_seed_offset": 200000,
              "training_step_budget": 10000, "evaluation_step_budget": 200,
              "evaluation_budget_semantics": "Truncate after 200 steps; retain return and mark unsuccessful.",
              "results": results,
              "scope": "4x12 deterministic cliff; fixed hyperparameters, not a general ranking."}
    (args.output_dir/"cliff-results.json").write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"sarsa": "#3989ce", "q_learning": "#9866c0"}
    labels = {"sarsa": "SARSA", "q_learning": "Q-learning"}
    for method, history in curves.items():
        # Each point averages only the current and preceding 24 episodes.
        smooth = np.array([np.convolve(row, np.ones(25)/25, mode="valid") for row in history])
        mean, sd = smooth.mean(axis=0), smooth.std(axis=0, ddof=1)
        x = np.arange(25, args.episodes+1)
        axes[0].plot(x, mean, color=colors[method], label=labels[method])
        axes[0].fill_between(x, mean-sd, mean+sd, color=colors[method], alpha=.16)
    axes[0].set(title="Training with exploration (epsilon = 0.1)",
                xlabel="Training episode (25-episode trailing mean)", ylabel="Return per episode")
    axes[0].legend(title="Mean and +/- 1 seed SD")
    keys = ["greedy_evaluation", "exploratory_evaluation"]
    for offset, method in [(-.18, "sarsa"), (.18, "q_learning")]:
        summary = results[method]["summary"]
        axes[1].bar(np.arange(2)+offset, [summary[k]["mean"] for k in keys], width=.34,
                    yerr=[summary[k]["seed_std"] for k in keys], capsize=4,
                    label=labels[method], color=colors[method], alpha=.8)
    axes[1].set(xticks=[0,1], xticklabels=["Greedy (epsilon = 0)", "Exploring (epsilon = 0.1)"],
                ylabel="Frozen-policy return per episode", title="Frozen Q; evaluation cap = 200 steps")
    axes[1].legend(title="Error bars: +/- 1 seed SD")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Cliff walking: {args.seeds} training seeds, {args.episodes} episodes each")
    fig.savefig(args.output_dir/"cliff-comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
