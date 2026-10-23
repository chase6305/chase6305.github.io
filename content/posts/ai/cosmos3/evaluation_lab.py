"""Recompute illustrative HUE and FPS Composite examples; no model results inferred."""
import argparse
import json
from math import isclose
from statistics import fmean


def hue_score(videos):
    """Accept one canonical answer per question, after annotator/QC resolution."""
    if not videos or any(not video for video in videos):
        raise ValueError('each video must contain at least one answered question')
    answers = [answer for video in videos for answer in video]
    if any(answer not in {'Yes', 'No', 'Unclear'} for answer in answers):
        raise ValueError('answers must be Yes, No or Unclear')
    # No and Unclear contribute to the denominator, but not the Yes numerator.
    return 100.0 * answers.count('Yes') / len(answers)


def fps_composite(vq_by_band, mf_by_band):
    """Average VQ*MF within bands, rather than multiply two global means."""
    if not vq_by_band or len(vq_by_band) != len(mf_by_band):
        raise ValueError('quality and fidelity need matching, nonempty FPS bands')
    return fmean(vq * mf for vq, mf in zip(vq_by_band, mf_by_band))


def demo():
    videos = [['Yes'] * 4 + ['No'] + ['Unclear'] * 5, ['Yes'] * 2]
    per_video = [hue_score([video]) for video in videos]
    without_unclear = [[answer for answer in video if answer != 'Unclear']
                       for video in videos]
    quality, fidelity = [10.0, 20.0], [0.9, 0.5]
    return {
        'illustrative_only': True,
        'hue': {
            'question_counts': [len(video) for video in videos],
            'per_video_percent': per_video,
            'correct_model_percent': hue_score(videos),
            'incorrect_mean_of_video_percent': fmean(per_video),
            'incorrect_discard_unclear_percent': hue_score(without_unclear),
        },
        'fps_composite': {
            'quality_by_band': quality,
            'motion_fidelity_by_band': fidelity,
            'correct_mean_of_products': fps_composite(quality, fidelity),
            'incorrect_product_of_means': fmean(quality) * fmean(fidelity),
        },
    }


def self_test():
    result = demo()
    hue = result['hue']
    assert hue['per_video_percent'] == [40.0, 100.0]
    assert isclose(hue['correct_model_percent'], 50.0)
    assert isclose(hue['incorrect_mean_of_video_percent'], 70.0)
    assert isclose(hue['incorrect_discard_unclear_percent'], 600 / 7)
    assert hue_score([['Unclear'] * 10]) == 0.0
    assert hue_score([['Yes', 'No', 'Unclear']]) == hue_score([['Yes'], ['No', 'Unclear']])
    composite = result['fps_composite']
    assert isclose(composite['correct_mean_of_products'], 9.5)
    assert isclose(composite['incorrect_product_of_means'], 10.5)
    result['self_test'] = 'passed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else demo(), indent=2))
