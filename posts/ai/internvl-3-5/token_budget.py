"""CPU arithmetic examples, not an InternVL router, trainer or benchmark."""

import math


def positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer")
    return value


def visual_tokens(tiles, high_resolution_tiles=None):
    """tiles includes thumbnails; None means the standard 256-token branch."""
    positive_integer(tiles)
    high = tiles if high_resolution_tiles is None else high_resolution_tiles
    if type(high) is not int or not 0 <= high <= tiles:
        raise ValueError("Invalid count for the 256-token branch")
    return 256 * high + 64 * (tiles - high)


def kv_cache_bytes(*, layers, sequence, kv_heads, head_dim, batch=1, element_bytes=2):
    """Full-attention cache only, without allocator/workspace/activation costs."""
    dimensions = (layers, sequence, kv_heads, head_dim, batch, element_bytes)
    return 2 * math.prod(positive_integer(v) for v in dimensions)


def square_sample_weights(lengths):
    """Total per-sample weights implied by 1/sqrt(N) per supervised token."""
    roots = [math.sqrt(positive_integer(length)) for length in lengths]
    if not roots:
        raise ValueError("No supervised samples")
    return tuple(root / sum(roots) for root in roots)


def sequence_ratio(log_new, log_old):
    """GSPO geometric mean ratio, for one already-unpadded response.

    Scalar arithmetic only: no autograd, clipping objective or policy update.
    """
    log_new, log_old = tuple(log_new), tuple(log_old)
    if not log_new or len(log_new) != len(log_old):
        raise ValueError("Nonempty, aligned log-probabilities are required")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v > 0 for v in log_new + log_old):
        raise ValueError("Log-probabilities must be finite and nonpositive")
    mean_log_ratio = sum(n - o for n, o in zip(log_new, log_old)) / len(log_new)
    try:
        result = math.exp(mean_log_ratio)
    except OverflowError as error:
        raise ValueError("Sequence ratio overflow") from error
    if not math.isfinite(result) or result == 0:
        raise ValueError("Sequence ratio overflow/underflow")
    return result


def group_advantages(rewards):
    """Population-std teaching convention; a zero-variance group gives zeros."""
    rewards = tuple(rewards)
    if len(rewards) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in rewards):
        raise ValueError("Expected at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    try:
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    except OverflowError as error:
        raise ValueError("Non-finite reward variance") from error
    if not math.isfinite(variance):
        raise ValueError("Non-finite reward variance")
    if variance == 0:
        return (0.0,) * len(rewards)
    return tuple((r - mean) / math.sqrt(variance) for r in rewards)


def clipped_surrogate(ratio, advantage, *, epsilon=0.2):
    """One sequence's MAXIMIZATION objective; not a differentiable trainer.

    Symmetric clipping is illustrative, not an InternVL hyperparameter claim.
    A minimization optimizer would use the negative of the group mean.
    """
    values = (ratio, advantage, epsilon)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        raise ValueError("Expected finite numerical ratio, advantage and epsilon")
    if ratio <= 0 or not 0 <= epsilon < 1:
        raise ValueError("Ratio must be positive and epsilon must be in [0,1)")
    clipped = min(max(ratio, 1 - epsilon), 1 + epsilon)
    candidates = (ratio * advantage, clipped * advantage)
    if not all(math.isfinite(v) for v in candidates):
        raise ValueError("Non-finite surrogate")
    return min(candidates)


def self_test():
    assert visual_tokens(13) == 3328
    assert visual_tokens(12, 6) == 1920
    assert visual_tokens(12, 6) / visual_tokens(12) == 0.625
    assert visual_tokens(12, 0) == 768
    cache = kv_cache_bytes(layers=36, sequence=32768, kv_heads=8, head_dim=128)
    assert cache / 2 ** 30 == 4.5
    assert square_sample_weights((100, 400)) == (1 / 3, 2 / 3)
    ratio = sequence_ratio((math.log(0.4), math.log(0.2)),
                           (math.log(0.2), math.log(0.4)))
    assert math.isclose(ratio, 1)
    assert group_advantages((1, 1, 1)) == (0, 0, 0)
    assert group_advantages((0, 1)) == (-1, 1)
    assert clipped_surrogate(1.5, 1) == 1.2
    assert clipped_surrogate(0.5, -1) == -0.8
    assert clipped_surrogate(1.5, -1) == -1.5
    assert clipped_surrogate(0.5, 1) == 0.5
    print("13 CPU arithmetic checks passed; 13 tiles=3328 tokens; KV example=4.5 GiB.")
    print("No model/router training, GPU inference or throughput measurement performed.")


if __name__ == "__main__":
    self_test()
