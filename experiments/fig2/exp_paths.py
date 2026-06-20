"""Fig.2 experiment output paths."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG2_EXP_ROOT = os.path.join(PROJECT_ROOT, 'experiments', 'fig2')


def output_dir(slug: str, mkdir: bool = True) -> str:
    d = os.path.join(FIG2_EXP_ROOT, slug, 'outputs')
    if mkdir:
        os.makedirs(d, exist_ok=True)
    return d
