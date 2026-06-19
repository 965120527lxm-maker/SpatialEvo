"""Fig.3 experiment output paths (imported by scripts via sys.path)."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG3_EXP_ROOT = os.path.join(PROJECT_ROOT, 'experiments', 'fig3')


def output_dir(slug: str, mkdir: bool = True) -> str:
    d = os.path.join(FIG3_EXP_ROOT, slug, 'outputs')
    if mkdir:
        os.makedirs(d, exist_ok=True)
    return d
