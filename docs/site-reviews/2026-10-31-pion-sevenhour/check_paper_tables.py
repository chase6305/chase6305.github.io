"""Compare published Markdown tables with the supplied pinned PDF's table rows.

Requires Poppler pdftotext. This source audit does not run any article helper,
model, training or evaluation, and does not infer uncertainty or causality.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import subprocess

PDF_SHA = '2016d07cca61aaa9be936f5a5a701ff033a89e6022f03a9795f639c4d1a2428f'
ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT/'content/posts/ai/pion-muon-vla-rlvr/index.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def page(pdf, number):
    return subprocess.check_output(['pdftotext', '-f', str(number), '-l', str(number),
                                    '-layout', str(pdf), '-'], text=True, encoding='utf-8', timeout=15)


def paper_rows(text, labels, columns):
    rows = {}
    for label in labels:
        matches = re.findall(r'^\s*'+re.escape(label)+r'\s+((?:\d+\.\d+\s*)+)$', text, re.M)
        require(len(matches) == 1, f'Expected exactly one source table row: {label}')
        values = [Decimal(n) for n in matches[0].split()]
        require(len(values) == columns, f'Wrong source column count: {label}')
        rows[label] = values
    return rows


def section(text, heading, next_heading):
    require(text.count(heading) == 1 and text.count(next_heading) == 1, 'Ambiguous article section')
    return text.split(heading, 1)[1].split(next_heading, 1)[0]


def markdown_row(text, label):
    matches = [line for line in text.splitlines() if line.startswith('| '+label+' |')]
    require(len(matches) == 1, f'Expected exactly one article table row: {label}')
    return [Decimal(s.strip().replace(',', '')) for s in matches[0].strip('|').split('|')[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    args = parser.parse_args()
    require(hashlib.sha256(args.pdf.read_bytes()).hexdigest() == PDF_SHA,
            'PDF does not match the supplied arXiv2605.19282v1 file')
    md = ARTICLE.read_text(encoding='utf-8')
    source = {n: page(args.pdf, n) for n in (8, 9, 27, 28, 38, 39)}
    checks = []
    a5 = paper_rows(source[38], ['AdamW', 'Muon', 'Pion (per-head)', 'Pion (default)'], 5)
    adapter = section(md, '### 9.1 ', '### 9.2 ')
    for article_label, paper_label in [('AdamW','AdamW'),('Muon','Muon'),('Pion 整矩阵动作头','Pion (default)')]:
        got = markdown_row(adapter, article_label)
        require(got == a5[paper_label], f'VLA-Adapter mismatch: {article_label}')
        require(sum(got[:4])/4 == got[4], f'Four-suite mean mismatch: {article_label}')
        checks.append({'table':'A5','page':38,'row':paper_label,'values':list(map(str,got))})
    require('96.85%' in md and sum(a5['Pion (per-head)'][:4])/4 == a5['Pion (per-head)'][4],
            'Per-head suite mean mismatch')
    vlanext = paper_rows(source[8], ['AdamW','Muon','Pion (Ours)'], 9)
    vlanext_md = section(md, '### 9.2 ', '### 9.3 ')
    for article_label, paper_label in [('AdamW','AdamW'),('Muon','Muon'),('Pion','Pion (Ours)')]:
        got = markdown_row(vlanext_md, article_label)
        require(got == vlanext[paper_label][:2], f'VLANeXt average mismatch: {article_label}')
        checks.append({'table':'1','page':8,'row':paper_label,'average_values':list(map(str,got))})
    perturbations = re.search(r'分别从 ([\d.]+)、([\d.]+)、([\d.]+) 提升至 ([\d.]+)、([\d.]+)、([\d.]+)', vlanext_md)
    require(perturbations is not None, 'Missing named perturbation comparison')
    reported = list(map(Decimal, perturbations.groups()))
    expected = [vlanext[label][i] for label in ['Muon','Pion (Ours)']for i in [4,7,8]]
    require(reported == expected, 'Language/Noise/Robot column mismatch')
    checks.append({'table':'1','page':8,'columns':['Language','Noise','Robot'],
                   'muon':list(map(str,reported[:3])),'pion':list(map(str,reported[3:]))})
    robot = paper_rows(source[9], ['AdamW','Muon','Pion (Ours)'], 4)
    robot_md = section(md, '### 9.3 ', '## 10. ')
    for article_label,paper_label in [('AdamW','AdamW'),('Muon','Muon'),('Pion','Pion (Ours)')]:
        got = markdown_row(robot_md, article_label)
        require(got == robot[paper_label], f'Real-robot rate mismatch: {article_label}')
        checks.append({'table':'3','page':9,'row':paper_label,'values':list(map(str,got))})
    a3 = ''.join(source[28].split())
    for token in ['Demonstrations200teleoperatedtrajectories','Totaltrainingsteps20,000',
                  'Trialsper(optimizer,task)30(randomizedinitialpose),≤300controlstepseach']:
        require(token in a3, f'Real-robot configuration not found: {token}')
    for token in ['200 条遥操作示范','训练 20,000 步','评估 30 个随机初始配置','最多 300 个控制步']:
        require(token in robot_md, f'Real-robot configuration transcription changed: {token}')
    counts = []
    for rate in robot['Pion (Ours)'][:3]:
        possibilities = [n for n in range(31)
                         if (Decimal(n)*100/30).quantize(Decimal('.1'), rounding=ROUND_HALF_UP) == rate]
        require(len(possibilities) == 1, 'Rounded success rate does not identify a unique integer count')
        counts.append(possibilities[0])
    require(counts == [28,25,24] and '28、25、24 次成功' in robot_md and '77/90' in robot_md,
            'Real-robot count inference changed')
    require((Decimal(sum(counts))*100/90).quantize(Decimal('.1'),rounding=ROUND_HALF_UP) == robot['Pion (Ours)'][3],
            'Real-robot average does not match the inferred counts')
    checks.append({'tables':['3','A3'],'pages':[9,28],'trials_per_task':30,'inferred_pion_counts':counts,
                   'interpretation':'Unique counts implied by reported rounded rates and trial denominator; not raw log validation.'})
    budget_md = section(md, '## 10. ', '## 11. ')
    a2 = source[27].split('Table A2:',1)[1]
    for label,site_label in [('Max prompt / response length','最大 prompt / response 长度（token）'),
                              ('Train batch (prompts)','每批 prompt 数'),
                              ('Rollout group size n','每个 prompt 的 rollout 数'),
                              ('Total training steps','训练步数')]:
        lines=[line.strip()for line in a2.splitlines()if line.strip().startswith(label)]
        require(len(lines)==1, f'Ambiguous budget source row: {label}')
        numbers=lambda s:[int(n.replace(',',''))for n in re.findall(r'\d[\d,]*',s)]
        expected=numbers(lines[0][len(label):])
        site_lines=[line for line in budget_md.splitlines()if line.startswith('| '+site_label+' |')]
        require(len(site_lines)==1, f'Ambiguous article budget row: {site_label}')
        got=numbers(site_lines[0].split('|',2)[2])
        require(got==expected, f'Budget mismatch: {label}')
        checks.append({'table':'A2','page':27,'row':label,'values':got})
    a6 = re.search(r'^\s*S1\s+AdamW\s+AdamW\s+AdamW\s+([\d.]+)\s*$',source[39],re.M)
    require(a6 is not None, 'Missing A6 all-AdamW row')
    discrepancy=re.search(r'全 AdamW 基线为 ([\d.]+)%，主 Figure 5 / Table A5 为 ([\d.]+)%',md)
    require(discrepancy is not None and list(map(Decimal,discrepancy.groups())) == [Decimal(a6[1]),a5['AdamW'][0]],
            'A5/A6 discrepancy not preserved in the article')
    checks.append({'tables':['A5','A6'],'pages':[38,39],'preserved_distinct_baselines':list(discrepancy.groups())})
    print(json.dumps({'pdf_sha256':PDF_SHA,'article_sha256':hashlib.sha256(ARTICLE.read_bytes()).hexdigest(),
                      'table_row_checks':len(checks),'checks':checks,
                      'scope':'Fixed-PDF table transcription and arithmetic only; no raw logs, uncertainty estimates, training reproduction or causal conclusions.'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
