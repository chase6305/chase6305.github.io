"""Check the token/temporal contracts described in Cosmos 3 v4; no model needed."""
import argparse
import json
from math import isclose


def can_attend(query, key):
    """Tokens are (packed_sample_id, stream, local_position)."""
    sample_q, stream_q, position_q = query
    sample_k, stream_k, position_k = key
    if sample_q != sample_k:
        return False
    if stream_q == 'DM':
        return True
    return stream_k == 'AR' and position_k <= position_q


def token_position(index, tokens_per_second, offset=0.0):
    if tokens_per_second <= 0:
        raise ValueError('tokens_per_second must be positive')
    return offset + 6.0 * index / tokens_per_second


def masked_velocity_mse(predicted, target, target_mask):
    """Average scalar target errors; condition entries are placeholders and ignored."""
    if not (len(predicted) == len(target) == len(target_mask)):
        raise ValueError('prediction, target and mask lengths must match')
    errors = [(p - t) ** 2 for p, t, active in zip(predicted, target, target_mask)
              if active]
    if not errors:
        raise ValueError('at least one target is required')
    return sum(errors) / len(errors)


def flow_step(values, velocities, target_mask, delta_sigma):
    """A simplified Euler step that keeps observed conditions fixed."""
    if not (len(values) == len(velocities) == len(target_mask)):
        raise ValueError('value, velocity and mask lengths must match')
    return [value + delta_sigma * velocity if active else value
            for value, velocity, active in zip(values, velocities, target_mask)]


def masked_flow_example():
    clean, noise, sigma = 2.0, -1.0, 0.5
    values = [5.0, sigma * noise + (1 - sigma) * clean]
    predicted_velocity = [100.0, -2.0]
    # The first entry is only a placeholder: the clean condition has no target loss.
    target_velocity = [0.0, noise - clean]
    target_mask = [False, True]
    return {
        'input': values,
        'predicted_velocity': predicted_velocity,
        'target_mask': target_mask,
        'target_mse': masked_velocity_mse(predicted_velocity, target_velocity, target_mask),
        'after_euler_step': flow_step(values, predicted_velocity, target_mask, -sigma),
    }


def demo():
    tokens = [(0, 'AR', i) for i in range(3)] + [(0, 'DM', i) for i in range(2)]
    mask = [[int(can_attend(q, k)) for k in tokens] for q in tokens]
    alignment = {
        '24fps_video_latent': token_position(6, 24 / 4),
        '16fps_video_latent': token_position(4, 16 / 4),
        '25hz_audio_latent': token_position(25, 25),
        '15hz_action': token_position(15, 15),
    }
    return {'mask': mask, 'positions_at_one_second': alignment,
            'masked_flow_example': masked_flow_example()}


def self_test():
    result = demo()
    assert result['mask'] == [
        [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
    ]
    assert not can_attend((0, 'AR', 2), (0, 'DM', 0))
    assert not can_attend((0, 'DM', 1), (1, 'AR', 0))
    assert not can_attend((0, 'DM', 1), (1, 'DM', 0))
    assert can_attend((0, 'DM', 0), (0, 'DM', 100))
    assert all(isclose(value, 6.0) for value in result['positions_at_one_second'].values())
    assert isclose(token_position(15, 30 / 4), 12.0)
    assert isclose(token_position(6, 6, offset=15000), 15006)
    try:
        token_position(1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid rate accepted')
    clean, noise, sigma = 2.0, -1.0, 0.5
    point = sigma * noise + (1 - sigma) * clean
    velocity = noise - clean
    assert isclose(point + (0 - sigma) * velocity, clean)
    example = result['masked_flow_example']
    assert isclose(example['target_mse'], 1.0)
    assert example['after_euler_step'] == [5.0, 1.5]
    # Counting the condition prediction would instead produce a spurious large loss.
    assert isclose(masked_velocity_mse([100, -2], [0, -3], [True, True]), 5000.5)
    # A large model output at a condition position must not move the observation.
    assert flow_step([5, 0.5], [-1000, -2], [False, True], -0.5) == [5, 1.5]
    result['self_test'] = 'passed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else demo(), indent=2))
