"""Render cited PDF regions and plot the reported LIBERO-10 results."""
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/ubuntu/Documents/cosmos3.pdf')
ASSETS = ROOT / 'content/posts/ai/cosmos3/assets'
ASSETS.mkdir(parents=True, exist_ok=True)
REGIONS = [
    ('training-curriculum', 20, 8, [58, 288, 536, 427]),
    ('droid-real-robot', 69, 25, [60, 80, 535, 215]),
    ('video-action-consistency', 110, 37, [60, 80, 535, 265]),
    ('audio-event-alignment', 62, 20, [62, 300, 532, 438]),
]
figures = []
for name, page, figure, bounds in REGIONS:
    x0, y0, x1, y1 = bounds
    target = Path('/tmp') / ('cosmos3-' + name)
    subprocess.run([
        'pdftoppm', '-f', str(page), '-l', str(page), '-r', '288',
        '-x', str(x0*4), '-y', str(y0*4), '-W', str((x1-x0)*4),
        '-H', str((y1-y0)*4), '-png', '-singlefile', str(SOURCE), str(target),
    ], check=True)
    with Image.open(str(target) + '.png') as image:
        image.convert('RGB').save(ASSETS / (name + '.webp'), lossless=True, method=6)
        dimensions = list(image.size)
    figures.append({'asset': 'assets/' + name + '.webp', 'paper_figure': figure,
                    'pdf_page': page, 'pdf_crop_points': bounds,
                    'raster_dpi': 288, 'dimensions': dimensions,
                    'processing': 'PDF region rasterization; lossless WebP encoding'})

iterations = [500, 1000, 1500, 2000]
mt = [24.6, 91.4, 95.8, 97.4]
pt = [0.0, 73.8, 93.4, 95.2]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(10, 4.9), constrained_layout=True)
ax.set_facecolor('white')
ax.plot(iterations, mt, color='#376cc4', marker='o', linewidth=2.5,
        label='MT-init: after action mid-training')
ax.plot(iterations, pt, color='#c87528', marker='s', linewidth=2.5,
        label='PT-init: before action mid-training')
for x, y in zip(iterations, mt):
    ax.annotate(f'{y:.1f}', (x, y), xytext=(0, 10), textcoords='offset points',
                ha='center', color='#245698', fontsize=11)
for x, y in zip(iterations, pt):
    ax.annotate(f'{y:.1f}', (x, y), xytext=(0, -20 if y > 20 else 10),
                textcoords='offset points', ha='center', color='#9a551b', fontsize=11)
ax.set(xlabel='Post-training iteration', ylabel='Closed-loop success rate (%)',
       title='LIBERO-10: adaptation from two initializations',
       xticks=iterations, ylim=(-5, 112), yticks=[0, 25, 50, 75, 100])
ax.grid(axis='y', alpha=0.18)
ax.legend(loc='lower right', frameon=True, fontsize=10)
fig.text(0.02, -0.025, 'Source: Cosmos 3 v4, Table 20. 500 rollouts per checkpoint; no uncertainty intervals reported.',
         fontsize=9, color='#526176')
vector_path = ASSETS / 'libero-adaptation.svg'
fig.savefig(vector_path, bbox_inches='tight')
# Matplotlib's multiline paths contain trailing spaces; keep the exported file clean.
vector_path.write_text('\n'.join(line.rstrip() for line in vector_path.read_text().splitlines()) + '\n')
fig.savefig('/tmp/cosmos3-libero-adaptation.png', dpi=180, bbox_inches='tight')
plt.close(fig)
with Image.open('/tmp/cosmos3-libero-adaptation.png') as image:
    image.convert('RGB').save(ASSETS / 'libero-adaptation.webp', lossless=True, method=6)

record = {
    'reviewed_on': '2026-10-23',
    'article': 'content/posts/ai/cosmos3/index.md',
    'source_pdf': str(SOURCE),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'paper': 'https://arxiv.org/abs/2606.02800v4',
    'figures': figures,
    'quantitative_plot': {'asset': 'assets/libero-adaptation.webp',
                          'vector_asset': 'assets/libero-adaptation.svg',
                          'source_table': 20, 'iterations': iterations,
                          'MT_init': mt, 'PT_init': pt, 'units': 'percent',
                          'rollouts_per_checkpoint': 500,
                          'uncertainty': 'not reported; no inferred error bars'},
}
record_path = ROOT / 'docs/image-generation/cosmos3-20261022.json'
if record_path.exists():
    # Keep separately recorded Imagegen prompts and source hashes on a re-render.
    record = {**json.loads(record_path.read_text()), **record}
record_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False, indent=2))
