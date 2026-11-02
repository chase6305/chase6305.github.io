"""Check actual PDF text pages for a summary and the start of its content."""
import argparse
import json
from pathlib import Path

PAIRS = [
    ('snr', '怎样估计梯度SNR：有限样本偏差、负校正值与零样本方差', '先定义一次观测是什么'),
    ('trace', '查看归一化输入0.35的逐步变化', '保持总次数为五'),
    ('flat', '平坦谱：有效秩不变，更新幅度为何仍会随维数急降？', '取完整方阵'),
    ('buffers', '多参数异步通信：为何还要检查相邻的缺失梯度', '固定版本的VLA-Adapter循环'),
    ('padding', '两个或三个进程：全部参数有梯度时，尾部补齐也可能覆盖上一批更新', '缺失梯度并非唯一触发点'),
    ('answers', '展开25题答案', '前者沿非零奇异子空间取极分解方向'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = []
    files = sorted(args.directory.glob('pion-*.txt'))
    if not files:
        raise SystemExit('No exported PDF text files found')
    for file in files:
        pages = [''.join(page.split()) for page in file.read_text().split('\f')]
        for name, summary, first in PAIRS:
            title_pages = [n+1 for n, text in enumerate(pages) if summary in text]
            content_pages = [n+1 for n, text in enumerate(pages) if first in text]
            shared = sorted(set(title_pages) & set(content_pages))
            rows.append({'pdf': file.with_suffix('.pdf').name, 'summary': name,
                         'title_pages': title_pages, 'first_content_pages': content_pages,
                         'passed': bool(shared), 'together_pages': shared})
    result = {'pdfs': len(files), 'cases': len(rows),
              'failures': sum(not r['passed'] for r in rows), 'rows': rows,
              'scope': 'Native PDF text pages, not CSS properties. Each full summary and the start of its first content must share a page.'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['pdfs', 'cases', 'failures']}))
    if args.check and result['failures']:
        raise SystemExit('Summary pagination failed; see the saved report')


if __name__ == '__main__':
    main()
