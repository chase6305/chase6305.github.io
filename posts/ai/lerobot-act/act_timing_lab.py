"""Standard-library ACT timing demonstration; no model or robot dependencies.

Run: python act_timing_lab.py --chunk-size 4 --action-steps 2 --coeff 0.5
Synthetic scalar predictions expose the generation time and execution time.
The offline ensemble illustrates the weighting rule, not LeRobot's online cache.
"""

import argparse
import math
from collections import deque


def predict(start, length):
    """Predict target time plus a generation-dependent bias (teaching only)."""
    return [start + offset + 0.1 * start for offset in range(length)]


def weighted_average(values, coefficient):
    # Inputs are ordered oldest to newest. Shift exponents for stability.
    logits = [-coefficient * i for i in range(len(values))]
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    return sum(w * value for w, value in zip(weights, values)) / sum(weights)


def show_queue(chunk_size, action_steps, ticks):
    print("\nQUEUE: execution time | new prediction? | source | scalar action")
    queue = deque()
    for time in range(ticks):
        refresh = not queue
        if refresh:
            chunk = predict(time, chunk_size)
            queue.extend((time, offset, value)
                         for offset, value in enumerate(chunk[:action_steps]))
        start, offset, value = queue.popleft()
        print(f"t={time:2d} | {str(refresh):5s} | A({start})[{offset}] | {value:.4f}")


def show_ensemble(chunk_size, ticks, coefficient):
    print("\nENSEMBLE: each row combines predictions for ONE execution time")
    for time in range(ticks):
        starts = range(max(0, time - chunk_size + 1), time + 1)
        values = [predict(start, chunk_size)[time - start] for start in starts]
        sources = ", ".join(f"A({start})[{time - start}]" for start in starts)
        result = weighted_average(values, coefficient)
        log_ratio = -coefficient * (len(values) - 1)
        # Display extreme ratios without overflowing exp; this does not affect the mean.
        ratio = f"{math.exp(log_ratio):.4f}" if log_ratio < 700 else "overflow"
        print(f"t={time:2d} | candidates={len(values)} | newest/oldest={ratio}"
              f" | {sources} | action={result:.4f}")


def show_padding():
    print("\nPADDING: episode has actions [0, 1, 2, 3, 4]; start=3, K=4")
    episode = [0.0, 1.0, 2.0, 3.0, 4.0]
    start, length = 3, 4
    # Repeat the last action as padding; do not read the next episode.
    target = [episode[min(start + j, len(episode) - 1)] for j in range(length)]
    is_pad = [start + j >= len(episode) for j in range(length)]
    prediction = [3.5, 3.5, 99.0, 99.0]
    errors = [abs(a - b) for a, b in zip(prediction, target)]
    valid_count = sum(not pad for pad in is_pad)
    masked = sum(e for e, pad in zip(errors, is_pad) if not pad) / max(valid_count, 1)
    print(f"target={target}, action_is_pad={is_pad}")
    print(f"prediction={prediction}")
    print(f"masked L1={masked:.4f}; incorrect unmasked L1={sum(errors) / length:.4f}")


def show_horizon_coverage():
    print("\nHORIZON COVERAGE: K=100; enumerate every start in one complete episode")
    chunk_size = 100
    for episode_length in (50, 100, 300):
        valid_by_offset = [max(episode_length - j, 0) for j in range(chunk_size)]
        total = sum(valid_by_offset)
        print(f"L={episode_length}: mean valid={total / episode_length:.1f}"
              f" | valid fraction={total / (episode_length * chunk_size):.1%}"
              f" | labels at offset 99={valid_by_offset[-1]}")
    print("Overlapping labels are counted repeatedly; these are not independent demonstrations.")


def show_batch_reduction():
    print("\nBATCH LOSS: normalized scalar-action errors, valid entries only")
    errors_by_sample = [[1.0, 1.0, 1.0, 1.0], [3.0]]
    element_mean = sum(map(sum, errors_by_sample)) / sum(map(len, errors_by_sample))
    sample_mean = sum(sum(row) / len(row) for row in errors_by_sample) / len(errors_by_sample)
    print(f"sample A: {errors_by_sample[0]}; sample B: {errors_by_sample[1]}")
    print(f"valid-element mean (ACT): {element_mean:.4f}")
    print(f"equal-sample mean (different objective): {sample_mean:.4f}")


def show_cycle_budget():
    print("\nCYCLE BUDGET: N=1 (no interpolation); illustrative costs, not a hardware benchmark")
    fps, action_steps = 30, 20
    other_ms, forward_ms = 8.0, 50.0
    budget_ms = 1000.0 / fps
    refresh_ms = other_ms + forward_ms
    mean_work_ms = other_ms + forward_ms / action_steps
    print(f"target={fps} Hz; M={action_steps}; cycle budget={budget_ms:.2f} ms")
    print(f"queue-only cycle={other_ms:.2f} ms; refresh cycle={refresh_ms:.2f} ms")
    print(f"mean work per action={mean_work_ms:.2f} ms")
    print(f"refresh exceeds budget by {max(0.0, refresh_ms - budget_ms):.2f} ms")
    print("A low mean does not remove the synchronous refresh stall.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--action-steps", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--coeff", type=float, default=0.5)
    args = parser.parse_args()
    if not 1 <= args.action_steps <= args.chunk_size:
        parser.error("require 1 <= action-steps <= chunk-size")
    if args.ticks < 1 or not math.isfinite(args.coeff):
        parser.error("ticks must be positive and coeff must be finite")
    print("Teaching example only: no ACT network, dataset downloads or robot commands.")
    show_queue(args.chunk_size, args.action_steps, args.ticks)
    # This separate mode predicts every tick, independent of queue action_steps.
    show_ensemble(args.chunk_size, args.ticks, args.coeff)
    show_padding()
    show_horizon_coverage()
    show_batch_reduction()
    show_cycle_budget()
    print("\nWeight direction, candidates ordered oldest to newest: [0.2, 0.5, 0.8]")
    for coefficient in (0.0, 0.5, -0.5):
        result = weighted_average([0.2, 0.5, 0.8], coefficient)
        print(f"m={coefficient:+.1f}: {result:.4f}")


if __name__ == "__main__":
    main()
