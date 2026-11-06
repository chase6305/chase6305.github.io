#!/usr/bin/env python3
"""Run the article's downloadable pinned-source probe without a second copy."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[3] /
                      'content/posts/ai/machine-learning-for-robotics/suresim_source_probe.py'),
                   run_name='__main__')
