"""Deterministic queue-policy model, not a concurrent queue implementation.

Run: python -B queue_freshness.py --output-dir results
Requires NumPy and Matplotlib only for statistics/plotting. Integer event times
avoid a wall-clock benchmark. Arrival precedes consumption at simultaneous times.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np


POLICIES = ("Reject new + FIFO", "Replace oldest + FIFO", "Reject new + drain + TTL")


def simulate(policy, capacity=8, ttl_ms=20):
    if policy not in POLICIES or capacity < 1 or ttl_ms < 0:
        raise ValueError("invalid policy parameters")
    queue = deque()
    counts = dict(offered=0, overflow_drops=0, consumer_discards=0,
                  expired_drops=0, delivered=0)
    deliveries, backlog = [], []
    for now in range(0, 1001, 5):  # 200 Hz producer, including both endpoints
        sample = (counts["offered"], now)
        counts["offered"] += 1
        if len(queue) == capacity:
            counts["overflow_drops"] += 1
            if policy == "Replace oldest + FIFO":
                queue.popleft()
                queue.append(sample)
        else:
            queue.append(sample)
        # A 50 Hz consumer pauses during [400, 600) ms.
        if now % 20 == 0 and not 400 <= now < 600 and queue:
            if policy == "Reject new + drain + TTL":
                # Bounded by capacity; a concurrent implementation still needs
                # a fixed work budget, not an unbounded "until empty" loop.
                counts["consumer_discards"] += len(queue) - 1
                while len(queue) > 1:
                    queue.popleft()
            sequence, captured = queue.popleft()
            age = now - captured
            if policy == "Reject new + drain + TTL" and age > ttl_ms:
                counts["expired_drops"] += 1
            else:
                deliveries.append((now, sequence, age))
                counts["delivered"] += 1
        backlog.append((now, len(queue)))
    counts["pending"] = len(queue)
    assert counts["offered"] == sum(counts[key] for key in counts if key != "offered")
    sequence = [row[1] for row in deliveries]
    assert all(a < b for a, b in zip(sequence, sequence[1:]))
    assert max(row[1] for row in backlog) <= capacity
    ages = [row[2] for row in deliveries]
    if policy == "Reject new + drain + TTL":
        assert max(ages) <= ttl_ms
    return {"counts": counts, "max_delivered_age_ms": max(ages),
            "p95_delivered_age_ms": float(np.percentile(ages, 95)),
            "deliveries": deliveries, "backlog": backlog}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = {policy: simulate(policy) for policy in POLICIES}
    assert results[POLICIES[0]]["max_delivered_age_ms"] > 200
    assert results[POLICIES[1]]["max_delivered_age_ms"] == 35
    assert results[POLICIES[2]]["counts"]["expired_drops"] == 1
    # A full one-slot FIFO has exactly zero spare capacity after each arrival.
    for policy in POLICIES:
        simulate(policy, capacity=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, constrained_layout=True)
    for (policy, result), color, marker in zip(results.items(),
            ("#d87525", "#6470b5", "#268966"), ("o", "s", "^")):
        values = np.asarray(result["deliveries"])
        # Do not connect across pauses or a rejected stale delivery.
        starts = np.r_[0, np.flatnonzero(np.diff(values[:, 0]) > 20) + 1]
        ends = np.r_[starts[1:], len(values)]
        for segment, (start, end) in enumerate(zip(starts, ends)):
            axes[0].plot(values[start:end, 0], values[start:end, 2], marker=marker,
                         markersize=3, color=color, label=policy if segment == 0 else None)
        backlog = np.asarray(result["backlog"])
        axes[1].step(backlog[:, 0], backlog[:, 1], where="post", color=color, label=policy)
    for ax in axes:
        ax.axvspan(400, 600, color="#e8eaf0", alpha=.85, zorder=0)
        ax.grid(alpha=.2)
    axes[0].set(ylabel="Age of delivered sample [ms]",
                title="Same producer and consumer; different queue policies")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].text(500, 180, "Consumer\npaused", ha="center", color="#475569")
    axes[1].set(xlabel="Simulated time [ms]", ylabel="Pending samples", yticks=range(0, 9, 2))
    axes[1].text(610, 5.1, "Capacity = 8; arrival before consumption", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "queue-freshness.png", dpi=160)
    plt.close(fig)
    report = {"model": {"producer_hz": 200, "consumer_hz": 50, "capacity": 8,
                         "consumer_pause_ms": [400, 600], "ttl_ms": 20,
                         "same_time_order": "arrival, then consumption",
                         "scope": "discrete-event policy model, not concurrency or latency benchmark"},
              "results": results}
    (args.output_dir / "queue-freshness-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: {k: v for k, v in r.items() if k not in {"deliveries", "backlog"}}
                      for p, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
