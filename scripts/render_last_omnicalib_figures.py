#!/usr/bin/env python3
"""Render exact LAST covariance example and OmniCalib v2 Table II data."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


last = load("last_geometry", "content/posts/ai/last-action-tokenizer/geometry_sanity.py")
omni = load("omni_observability", "content/posts/calibration/omnicalib/observability_sanity.py")


def text(x, y, label, size=19, anchor="start", weight="normal"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(label))}</text>'


def path(points, stroke, width=2, dash=""):
    data = " ".join(("M" if i == 0 else "L") + f"{x:.4f},{y:.4f}" for i, (x, y) in enumerate(points))
    return f'<path d="{data}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def doc(title, description, body, height=620):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}</style>'
            f'<rect width="1120" height="{height}" fill="white"/>' + "".join(body) + '</svg>\n')


def whitening():
    angle = math.pi / 6
    q = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    transform = last.mm(q, [[3., 0.], [0., 1.]])
    covariance = last.mm(transform, last.transpose(transform))
    w = last.whitening_2d(covariance)
    points = [last.mv(transform, [math.sqrt(2) * math.cos(i * math.tau / 24), math.sqrt(2) * math.sin(i * math.tau / 24)]) for i in range(24)]
    white = [last.mv(w, p) for p in points]
    assert max(abs(x - y) for a, b in zip(last.covariance(points), covariance) for x, y in zip(a, b)) < 1e-12
    assert max(abs(x - y) for a, b in zip(last.covariance(white), last.eye(2)) for x, y in zip(a, b)) < 1e-12
    body = [text(30, 44, 'Local covariance whitening', 29, weight="bold"),
            text(30, 82, 'Constructed 2D samples; population covariance uses a 1/N normalization.', 19)]
    for center, samples, label, color in ((270, points, 'Original residual coordinates', '#5b21b6'), (850, white, 'Whitened coordinates', '#075985')):
        px = lambda value: center + value * 42
        py = lambda value: 310 - value * 42
        body.append(f'<rect x="{center-225}" y="118" width="450" height="380" rx="16" fill="#f8fafc" stroke="#cbd5e1"/>')
        body.extend([path([(center - 210, 310), (center + 210, 310)], '#64748b'), path([(center, 140), (center, 435)], '#64748b')])
        for value in (-4, -2, 2, 4):
            body.append(text(px(value), 334, str(value), 16, 'middle'))
        for value in (-2, 2):
            body.append(text(center - 10, py(value) + 6, str(value), 16, 'end'))
        contour = []
        for i in range(201):
            point = last.mv(transform, [math.cos(i * math.tau / 200), math.sin(i * math.tau / 200)])
            if center == 850:
                point = last.mv(w, point)
            contour.append((px(point[0]), py(point[1])))
        body.append(path(contour, color, 3))
        for point in samples:
            body.append(f'<circle cx="{px(point[0]):.4f}" cy="{py(point[1]):.4f}" r="4.5" fill="white" stroke="{color}" stroke-width="2"/>')
        body.append(text(center, 465, label, 20, 'middle', 'bold'))
    body.append(path([(512, 298), (602, 298), (590, 290), (602, 298), (590, 306)], '#334155', 3))
    body.extend([text(557, 272, 'W', 24, 'middle', 'bold'),
                 text(60, 540, 'Covariance eigenvalues: 9 and 1; principal axis: 30 degrees.', 20),
                 text(60, 578, 'After W = covariance^(-1/2): covariance = I. Pairing semantics still need supervision.', 19)])
    return doc('LAST local whitening arithmetic', 'An exactly constructed anisotropic population is whitened to unit covariance. The contour is the unit Mahalanobis ellipse, and the deterministic samples lie on a radius-square-root-two ring. This figure does not show paper data.', body)


def leg_recovery():
    body = [text(30, 44, 'OmniCalib v2: twelve-joint virtual-injection recovery', 27, weight='bold'),
            text(30, 82, 'Published Table II residuals, with local Hessian standard deviations.', 19)]
    x0, width, top, height = 110, 930, 140, 315
    py = lambda v: top + height * (.2 - v) / .4
    for value in (-.2, -.1, 0., .1, .2):
        body.append(path([(x0, py(value)), (x0 + width, py(value))], '#64748b' if abs(value) == .1 else '#cbd5e1', 2 if abs(value) == .1 else 1, '8 5' if abs(value) == .1 else ''))
        body.append(text(x0 - 14, py(value) + 6, f'{value:+.1f}', 17, 'end'))
    body.append(text(30, 119, 'Residual (deg)', 18))
    labels = [('Hip', 'pitch'), ('Hip', 'roll'), ('Hip', 'yaw'), ('Knee', ''), ('Ankle', 'pitch'), ('Ankle', 'roll')]
    for j, lines in enumerate(labels):
        center = x0 + (j + .5) * width / 6
        for k, label in enumerate(lines):
            body.append(text(center, 487 + k * 23, label, 18, 'middle'))
        for side, offset, color in ((0, -12, '#075985'), (1, 12, '#9a3412')):
            row = omni.ROWS[j + side * 6]
            delta, sigma = row[4:]
            x, y = center + offset, py(delta)
            body.append(path([(x, py(delta - sigma)), (x, py(delta + sigma))], color, 2.5))
            for end in (delta - sigma, delta + sigma):
                body.append(path([(x - 7, py(end)), (x + 7, py(end))], color, 2.5))
            if side == 0:
                body.append(f'<circle cx="{x}" cy="{y}" r="5.5" fill="white" stroke="{color}" stroke-width="2.5"/>')
            else:
                body.append(f'<path d="M{x},{y-7} L{x+7},{y} L{x},{y+7} L{x-7},{y} Z" fill="white" stroke="{color}" stroke-width="2.5"/>')
            if j == 2:
                body.append(text(x + 16, y - 12 if side == 0 else y + 29, f'{delta:+.3f}', 18))
    body.extend([text(65, 555, 'Circle: left leg. Diamond: right leg. Bars: local +/- sigma, not accuracy bounds.', 19),
                 text(65, 590, f'Dashed: +/- 0.1 deg reference. Printed residual RMS: {omni.rms(r[4] for r in omni.ROWS):.6f} deg.', 19)])
    return doc('OmniCalib lower-limb recovery from Table II', 'Both hip-yaw recovery residuals exceed the plus-or-minus 0.1 degree reference despite a twelve-joint RMS of about 0.063 degrees. Local standard deviations are not validated error bounds.', body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    last.check()
    omni.check()
    images = {
        ROOT / 'content/posts/ai/last-action-tokenizer/assets/local-whitening.svg': whitening(),
        ROOT / 'content/posts/calibration/omnicalib/assets/leg-recovery.svg': leg_recovery(),
    }
    import xml.etree.ElementTree as ET
    for path_name, data in images.items():
        ET.fromstring(data)
        if args.write:
            path_name.write_text(data)
        if args.check:
            assert path_name.read_text() == data, f'Stale figure: {path_name}'
    print('PASS: covariance construction, Table II arithmetic, and deterministic SVG assets')


if __name__ == '__main__':
    main()
