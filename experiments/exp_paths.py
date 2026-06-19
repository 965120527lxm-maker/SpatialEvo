"""Canonical experiment output paths (scripts stay in scripts/)."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def exp_root(category: str, slug: str) -> str:
    return os.path.join(PROJECT_ROOT, 'experiments', category, slug)


def output_dir(category: str, slug: str, mkdir: bool = True) -> str:
    d = os.path.join(exp_root(category, slug), 'outputs')
    if mkdir:
        os.makedirs(d, exist_ok=True)
    return d


def fig3_output(slug: str, mkdir: bool = True) -> str:
    return output_dir('fig3', slug, mkdir=mkdir)
