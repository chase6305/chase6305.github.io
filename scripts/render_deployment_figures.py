#!/usr/bin/env python3
"""Deterministic coordinate/time/control figures for three deployment articles."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deployment = load('deployment_contracts', 'content/posts/ai/rma-umi-rtc-deployment/deployment_contracts.py')
residual = load('residual_control', 'content/posts/ai/residual-reinforcement-learning/residual_control_sanity.py')
contact = load('contact_calibration', 'content/posts/calibration/interactive-contact-calibration/contact_calibration_sanity.py')


def text(x, y, label, size=19, anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(label))}</text>'


def line(points, color='#64748b', width=2, dash=''):
    d = ' '.join(('M' if i == 0 else 'L')+f'{x:.4f},{y:.4f}' for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def rect(x, y, w, h, fill, stroke='#cbd5e1', radius=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def document(title, description, body, height=680):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}</style>'
            f'<rect width="1120" height="{height}" fill="white"/>'+''.join(body)+'</svg>\n')


def rtc_timeline():
    h, s, d = 16, 5, 4
    weights = deployment.rtc_weights(h, s, d)
    start, cell = 168, 54
    body = [text(30, 43, 'RTC: align the next chunk with the remaining old plan', 27, weight='bold'),
            text(30, 80, 'Constructed schedule: H = 16, s = 5, d = 4; one control step = 20 ms.', 20),
            text(30, 147, 'Old plan', 19, weight='bold'), text(30, 170, '(shifted)', 18),
            text(30, 222, 'New chunk', 19, weight='bold')]
    colors = ['#dbeafe' if i < d else '#ede9fe' if i < h-s else '#dcfce7' for i in range(h)]
    for i in range(h):
        x = start+i*cell
        if i < h-s:
            body.append(rect(x, 122, cell-4, 54, '#f1f5f9'))
            body.append(text(x+(cell-4)/2, 156, i, 20, 'middle'))
        else:
            body.append(rect(x, 122, cell-4, 54, 'white', '#e2e8f0'))
            body.append(text(x+(cell-4)/2, 155, '-', 20, 'middle'))
        body.append(rect(x, 198, cell-4, 54, colors[i], '#64748b'))
        body.append(text(x+(cell-4)/2, 232, i, 20, 'middle'))
    for a, b, label, detail, color in [(0, d, 'Committed prefix', '0-3: 80 ms', '#1d4ed8'),
                                      (d, h-s, 'Editable overlap', '4-10: 140 ms', '#6d28d9'),
                                      (h-s, h, 'New future', '11-15: 100 ms', '#166534')]:
        x1, x2 = start+a*cell, start+b*cell-4
        body.append(line([(x1, 266), (x1, 275), (x2, 275), (x2, 266)], color))
        body.append(text((x1+x2)/2, 307, label, 19, 'middle', 'bold'))
        body.append(text((x1+x2)/2, 333, detail, 18, 'middle'))
    body.append(text(30, 390, 'Guidance weight W', 21, weight='bold'))
    baseline, amplitude = 566, 135
    for value in (0., .5, 1.):
        y = baseline-value*amplitude
        body.append(line([(start-8, y), (start+h*cell-4, y)], '#cbd5e1', 1))
        body.append(text(start-18, y+6, f'{value:g}', 17, 'end'))
    for i, weight in enumerate(weights):
        x = start+i*cell
        if weight:
            body.append(rect(x+5, baseline-weight*amplitude, cell-14, weight*amplitude, colors[i], '#475569', 2))
        else:
            body.append(line([(x+5, baseline), (x+cell-9, baseline)], '#166534', 3))
        body.append(text(x+(cell-4)/2, baseline+27, i, 17, 'middle'))
    body.extend([text(600, 625, 'New-chunk index i', 19, 'middle'),
                 text(30, 660, 'Weights guide generation; they do not average two executed action trajectories.', 20)])
    return document('RTC committed prefix and soft masking', 'The new length-16 action chunk has four committed actions, seven editable overlapping actions and five new future actions. The exact RTC v2 linear-times-exponential guidance weights are plotted below.', body)


def umi_reanchoring():
    old, new, target = (1., 0., math.pi/2), (1., 1., 0.), (2., 1., math.pi/2)
    actions = deployment.relative(old, target), deployment.relative(new, target)
    assert max(abs(a-b) for a, b in zip(actions[0], (1., -1., 0.))) < 1e-12
    assert max(abs(a-b) for a, b in zip(actions[1], (1., 0., math.pi/2))) < 1e-12
    converted = deployment.reanchor(actions[0], old, new)
    assert max(abs(a-b) for a, b in zip(converted, actions[1])) < 1e-12
    assert max(abs(a-b) for a, b in zip(deployment.compose(new, actions[0]), (2., 0., 0.))) < 1e-12
    body = [text(30, 43, 'Reanchor one unchanged world target', 28, weight='bold'),
            text(30, 80, 'Constructed SE(2) example: target = (2 m, 1 m, 90 deg); the world frame is fixed.', 20)]

    def arrow(start, end, color, width=2):
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx/length, dy/length
        bx, by = end[0]-10*ux, end[1]-10*uy
        points = [end, (bx-5*uy, by+5*ux), (bx+5*uy, by-5*ux)]
        return [line([start, end], color, width),
                '<polygon points="'+' '.join(f'{x:.3f},{y:.3f}' for x, y in points)+f'" fill="{color}"/>']

    for x0, anchor, action, name, color in [(35, old, actions[0], 'Old', '#1d4ed8'),
                                            (575, new, actions[1], 'New', '#6d28d9')]:
        body.append(rect(x0, 112, 510, 478, '#f8fafc'))
        body.append(text(x0+255, 151, name+' anchor', 24, 'middle', 'bold'))
        px = lambda x:x0+70+x*120
        py = lambda y:440-y*120
        for x in (1, 2, 3):
            body.append(line([(px(x), py(0)), (px(x), py(2))], '#e2e8f0', 1))
        for y in (1, 2):
            body.append(line([(px(0), py(y)), (px(3), py(y))], '#e2e8f0', 1))
        body.extend(arrow((px(0), py(0)), (px(3)+18, py(0)), '#64748b'))
        body.extend(arrow((px(0), py(0)), (px(0), py(2)-20), '#64748b'))
        for value in (0, 1, 2, 3):
            body.append(text(px(value), py(0)+25, value, 17, 'middle'))
        for value in (1, 2):
            body.append(text(px(0)-15, py(value)+6, value, 17, 'end'))
        body.extend([text(px(3)+24, py(0)+7, 'x_W', 17),
                     text(px(0)+10, py(2)-13, 'y_W', 17)])
        point, goal = (px(anchor[0]), py(anchor[1])), (px(target[0]), py(target[1]))
        body.append(line([point, goal], '#475569', 2, '7 5'))
        for theta, label, axis_color in [(anchor[2], 'x_local', '#b91c1c'),
                                         (anchor[2]+math.pi/2, 'y_local', '#0f766e')]:
            endpoint = (px(anchor[0]+.52*math.cos(theta)), py(anchor[1]+.52*math.sin(theta)))
            body.extend(arrow(point, endpoint, axis_color, 3))
            if math.sin(theta) > .5:
                body.append(text(endpoint[0]+10, endpoint[1]-8, label, 17))
            elif math.cos(theta) < -.5:
                body.append(text((point[0]+endpoint[0])/2, endpoint[1]-14, label, 17, 'middle'))
            else:
                body.append(text(endpoint[0]-2, endpoint[1]+30, label, 17, 'middle'))
        body.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="7" fill="white" stroke="{color}" stroke-width="3"/>')
        body.extend(arrow(goal, (goal[0], goal[1]-52), '#166534', 3))
        body.append(f'<path d="M{goal[0]-7},{goal[1]} L{goal[0]},{goal[1]-7} L{goal[0]+7},{goal[1]} L{goal[0]},{goal[1]+7} Z" fill="#166534"/>')
        body.append(text(goal[0], goal[1]-77, 'Target', 19, 'middle', 'bold'))
        body.append(text(x0+255, 498, f'{name} anchor: ({anchor[0]:g}, {anchor[1]:g}, {math.degrees(anchor[2]):g} deg)', 20, 'middle'))
        body.append(rect(x0+25, 522, 460, 50, '#ede9fe'))
        clean = lambda x:0. if abs(x) < 1e-12 else x
        body.append(text(x0+255, 555, f'A_{name.lower()} = ({clean(action[0]):g}, {clean(action[1]):g}, {clean(math.degrees(action[2])):g} deg)', 24, 'middle', 'bold'))
    body.extend([text(560, 633, 'A_new = inverse(T_new) * T_old * A_old', 24, 'middle', 'bold'),
                 text(560, 673, 'Convert coordinates so both commands reconstruct the same world pose.', 20, 'middle')])
    return document('One world target under two action anchors', 'The old anchor is at position one, zero with ninety-degree rotation. The new anchor is at one, one with zero rotation. The unchanged world target at two, one and ninety degrees has old-relative pose one, minus one, zero degrees, and new-relative pose one, zero, ninety degrees. Dashed lines show displacement; labeled local axes show orientation.', body, 700)


def residual_headroom():
    body = [text(30, 43, 'Residual authority depends on the nominal command', 28, weight='bold'),
            text(30, 80, 'Constructed scalar interface: total command u = b + a, with -1 <= u <= 1.', 20)]
    x0, y0, width, height = 108, 133, 590, 400
    px = lambda b: x0+(b+1.5)/3*width
    py = lambda a: y0+(2.5-a)/5*height
    feasible = [(px(b), py(-1-b)) for b in (-1.5, 1.5)]+[(px(b), py(1-b)) for b in (1.5, -1.5)]
    polygon = ' '.join(f'{x:.4f},{y:.4f}' for x, y in feasible)
    body.append(f'<polygon points="{polygon}" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
    # Intersect with an optional fixed residual radius of 0.4.
    xs = [-1.4+i*2.8/100 for i in range(101)]
    lower = [(px(b), py(residual.feasible_residual(b, radius=.4)[0])) for b in xs]
    upper = [(px(b), py(residual.feasible_residual(b, radius=.4)[1])) for b in reversed(xs)]
    polygon = ' '.join(f'{x:.4f},{y:.4f}' for x, y in lower+upper)
    body.append(f'<polygon points="{polygon}" fill="#ffedd5" stroke="#9a3412" stroke-width="2"/>')
    for value in (-1.5, -1., 0., 1., 1.5):
        body.append(line([(px(value), y0), (px(value), y0+height)], '#cbd5e1', 1))
        body.append(text(px(value), y0+height+30, f'{value:g}', 18, 'middle'))
    for value in (-2., -1., 0., 1., 2.):
        body.append(line([(x0, py(value)), (x0+width, py(value))], '#cbd5e1', 1))
        body.append(text(x0-14, py(value)+6, f'{value:g}', 18, 'end'))
    body.extend([line([(x0, y0+height), (x0+width, y0+height)], '#475569'),
                 line([(x0, y0), (x0, y0+height)], '#475569'),
                 text(x0+width/2, 602, 'Nominal command b', 20, 'middle'),
                 text(30, 116, 'Residual a', 19)])
    body.append(rect(735, 133, 350, 400, '#f8fafc'))
    body.extend([text(755, 170, 'Examples', 24, weight='bold'),
                 text(755, 212, 'b = 0.0:  a in [-1.0, 1.0]', 19),
                 text(755, 248, 'b = 0.9:  a in [-1.9, 0.1]', 19),
                 text(755, 300, 'Also require |a| <= 0.4:', 19, weight='bold'),
                 text(755, 337, 'b = 0.9:  a in [-0.4, 0.1]', 19),
                 text(755, 373, 'b = 1.5:  no feasible residual', 19),
                 text(755, 435, 'Blue: total-action limit', 18),
                 text(755, 467, 'Orange: intersection with', 18),
                 text(755, 494, 'the extra residual radius', 18),
                 text(30, 650, 'Command bounds alone do not verify contacts, collision avoidance or stability.', 20)])
    return document('Residual action headroom', 'An exact feasible strip shows the residual interval imposed by total-command bounds. The orange intersection adds a fixed residual radius of 0.4; a baseline of 1.5 has no feasible residual under that extra restriction.', body)


def contact_observability():
    horizontal, diverse = contact.geometry_rows()
    assert contact.exact_rank(horizontal) == 3 and contact.exact_rank(diverse) == 6
    body = [text(30, 43, 'Contact observability: normals and lever arms both matter', 27, weight='bold'),
            text(30, 80, 'Constructed fixed point-plane correspondences; J = [ n^T, (p x n)^T ].', 20)]
    for x0, rows, title, note, color in [(35, horizontal, 'Horizontal plane only', 'rank = 3', '#1d4ed8'),
                                         (575, diverse, 'Diverse planes and positions', 'rank = 6', '#166534')]:
        body.append(rect(x0, 118, 510, 443, '#f8fafc'))
        body.extend([text(x0+255, 155, title, 22, 'middle', 'bold'),
                     text(x0+255, 191, note, 24, 'middle', 'bold')])
        labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for j, label in enumerate(labels):
            body.append(text(x0+110+j*58, 238, label, 19, 'middle', 'bold'))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                body.append(text(x0+110+j*58, 280+i*39, f'{value:g}', 21, 'middle'))
        for x in (x0+70, x0+435):
            sign = 1 if x == x0+70 else -1
            body.append(line([(x+sign*8, 253), (x, 253), (x, 291+(len(rows)-1)*39),
                              (x+sign*8, 291+(len(rows)-1)*39)], color, 2))
    body.extend([text(60, 598, 'Plane-only null directions: tx, ty and rz (yaw).', 20),
                 text(60, 631, 'Opposite normals can flip a row sign without adding rank.', 20),
                 text(60, 662, 'Finite edges, association changes and sensor errors require a fuller model.', 19)])
    return document('Local rank of contact geometry', 'Three horizontal-plane rows constrain height, roll and pitch. A six-row constructed system with additional side normals and lever arms has full rank. Opposite normals on their own need not add an independent constraint.', body)


def contact_sampling_gap():
    body = [text(30, 43, 'Sparse tool samples can miss geometric penetration', 28, weight='bold'),
            text(30, 80, 'Constructed 2D circle: radius 40 mm, center height 39 mm; exact plane SDF = y.', 20)]
    x0, y0, width, height = 85, 145, 480, 300
    px = lambda x:x0+(x+20)/40*width
    py = lambda y:y0+(6-y)/8*height
    body.append(rect(45, 112, 555, 440, '#f8fafc'))
    body.append(text(322, 137, 'Zoom near the lowest point', 21, 'middle', 'bold'))
    body.append(f'<rect x="{x0}" y="{py(0)}" width="{width}" height="{py(-2)-py(0)}" fill="#fff1f2"/>')
    for v in (-20, -10, 0, 10, 20):
        body.append(line([(px(v), y0+8), (px(v), y0+height)], '#e2e8f0', 1))
        body.append(text(px(v), 478, v, 17, 'middle'))
    for v in (-2, 0, 2, 4, 6):
        body.append(text(x0-13, py(v)+6, v, 17, 'end'))
    arc = [(px(x), py(39-math.sqrt(40**2-x**2))) for x in [-20+i*.4 for i in range(101)]]
    body.append(line(arc, '#1d4ed8', 3))
    body.append(line([(x0, py(0)), (x0+width, py(0))], '#475569', 2))
    body.extend([text(x0+width/2, py(0)-10, 'Plane: y = 0', 18, 'middle'),text(325, 515, 'x (mm)', 19, 'middle'),text(50, 173, 'y (mm)', 18)])
    circle = contact.sampled_circle_distance(8)
    nearest = [(x, y) for x, y in circle['points'] if -20 <= x <= 20 and -2 <= y <= 6]
    assert len(nearest) == 2
    for x, y in nearest:
        body.append(line([(px(x), py(y)), (px(x), py(0))], '#9a3412', 2, '6 4'))
        body.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="6" fill="white" stroke="#9a3412" stroke-width="3"/>')
    body.append(text(325, py(circle['sampled'])-17, 'Sampled min: +2.0448 mm', 19, 'middle'))
    body.append(f'<path d="M{px(0)-6},{py(-1)} L{px(0)},{py(-1)-6} L{px(0)+6},{py(-1)} L{px(0)},{py(-1)+6} Z" fill="#1d4ed8"/>')
    body.append(text(365, py(-1)+24, 'Exact min: -1 mm', 18))
    body.append(rect(625, 112, 460, 440, '#f8fafc'))
    body.extend([text(855, 151, 'Worst-phase uniform samples', 22, 'middle', 'bold'),
                 text(710, 201, 'Points L', 20, 'middle', 'bold'),text(939, 201, 'Sampled min (mm)', 20, 'middle', 'bold')])
    for i, count in enumerate((8, 16, 32, 64)):
        minimum = contact.sampled_circle_distance(count)['sampled']
        y = 257+i*57
        body.extend([text(710, y, count, 23, 'middle'),text(939, y, f'{minimum:+.4f}', 23, 'middle')])
    body.extend([text(650, 505, 'Exact continuous minimum: -1.0000 mm', 19),
                 text(45, 589, 'Inset shows the nearest two of eight samples; the full contour is a circle.', 20),
                 text(45, 625, 'Coverage and SDF approximation error determine the geometric bound.', 20)])
    return document('Finite contact sampling can miss penetration', 'An exact circle penetrates the plane by one millimeter. Eight uniformly spaced samples at the worst phase all miss its lowest point and return a positive minimum of 2.0448 millimeters. A 32-point sample returns a negative minimum.', body, 650)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    images = {
        ROOT/'content/posts/ai/rma-umi-rtc-deployment/assets/rtc-action-timeline.svg':rtc_timeline(),
        ROOT/'content/posts/ai/rma-umi-rtc-deployment/assets/umi-reanchoring.svg':umi_reanchoring(),
        ROOT/'content/posts/ai/residual-reinforcement-learning/assets/residual-headroom.svg':residual_headroom(),
        ROOT/'content/posts/calibration/interactive-contact-calibration/assets/contact-observability.svg':contact_observability(),
        ROOT/'content/posts/calibration/interactive-contact-calibration/assets/contact-sampling-gap.svg':contact_sampling_gap(),
    }
    for path, data in images.items():
        ET.fromstring(data)
        if args.write:
            path.write_text(data)
        if args.check:
            assert path.read_text() == data, f'Stale figure: {path}'
    print('PASS: exact RTC weights, UMI reanchoring, residual intersections, contact ranks and sampling gap, five deterministic SVGs')


if __name__ == '__main__':
    main()
