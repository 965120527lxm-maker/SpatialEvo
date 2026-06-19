#!/usr/bin/env python3
"""
Move all outputs/ data into experiments/<category>/<slug>/outputs/.

- Registered Fig.3 runs: merge then remove legacy path (no symlinks).
- Orphan fig3_* dirs: experiments/fig3/_legacy/<name>/outputs/
- Other top-level outputs: experiments/<category>/...

After migration, outputs/README.md points to experiments/.
"""

import os
import shutil
import stat
import sys

FIG3_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG3_SCRIPTS)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG3_EXP = os.path.join(PROJECT_ROOT, 'experiments', 'fig3')

# (legacy_rel_path, category, slug)
OTHER_MIGRATIONS = [
    ('outputs/multi_omics', 'multi_omics', 'default'),
    ('outputs/multi_omics_spatialexp_small', 'multi_omics', 'spatialexp_small'),
    ('outputs/multi_omics_spatialexp_gt', 'multi_omics', 'spatialexp_gt'),
    ('outputs/baseline_spatialex', 'baselines', 'spatialex'),
    ('outputs/baselines/fig3_spatialexp', 'baselines', 'fig3_spatialexp'),
    ('outputs/baselines/fig3_spatialexp_small', 'baselines', 'fig3_spatialexp_small'),
    ('outputs/baselines/fig3_spatialexp_gt', 'baselines', 'fig3_spatialexp_gt'),
    ('outputs/archive/fig3_test_conditional', 'archive', 'fig3_test_conditional'),
    ('outputs/summaries', 'summaries', 'reports'),
]

# Import registry legacy paths from sync_experiments
from sync_experiments import EXPERIMENTS  # noqa: E402


def exp_out(category: str, slug: str) -> str:
    return os.path.join(PROJECT_ROOT, 'experiments', category, slug, 'outputs')


def exp_dir(category: str, slug: str) -> str:
    return os.path.join(PROJECT_ROOT, 'experiments', category, slug)


def _merge_into(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        return
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.exists(d):
            if os.path.isdir(s) and os.path.isdir(d):
                _merge_into(s, d)
                try:
                    os.rmdir(s)
                except OSError:
                    pass
            continue
        shutil.move(s, d)


def _remove_path(path):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def migrate_one(legacy_rel, category, slug, dry_run=False):
    legacy = os.path.join(PROJECT_ROOT, legacy_rel)
    target = exp_out(category, slug)
    if not os.path.exists(legacy):
        return 'absent', legacy_rel

    if os.path.islink(legacy):
        if not dry_run:
            os.unlink(legacy)
        return 'unlink', legacy_rel

    if dry_run:
        return 'would_move', legacy_rel

    os.makedirs(exp_dir(category, slug), exist_ok=True)
    if os.path.isdir(target) and os.listdir(target):
        _merge_into(legacy, target)
        _remove_path(legacy)
    else:
        if os.path.exists(target):
            _remove_path(target)
        shutil.move(legacy, target)
    return 'moved', legacy_rel


def migrate_registered(dry_run=False):
    results = []
    for meta in EXPERIMENTS:
        slug = meta['slug']
        for leg in meta['legacy']:
            results.append(migrate_one(leg, 'fig3', slug, dry_run))
    return results


def migrate_orphan_fig3(dry_run=False):
    results = []
    for root in ('outputs/conditional', 'outputs/baselines', 'outputs/oracles', 'outputs/archive'):
        root_abs = os.path.join(PROJECT_ROOT, root)
        if not os.path.isdir(root_abs):
            continue
        for name in os.listdir(root_abs):
            if not name.startswith('fig3_'):
                continue
            legacy_rel = os.path.join(root, name)
            legacy = os.path.join(PROJECT_ROOT, legacy_rel)
            if not os.path.exists(legacy):
                continue
            if os.path.islink(legacy):
                if not dry_run:
                    os.unlink(legacy)
                results.append(('unlink_orphan', legacy_rel))
                continue
            slug = f'_legacy/{name[len("fig3_"):]}'
            results.append(migrate_one(legacy_rel, 'fig3', slug, dry_run))
    return results


def migrate_other(dry_run=False):
    return [migrate_one(leg, cat, slug, dry_run) for leg, cat, slug in OTHER_MIGRATIONS]


def write_legacy_readme(category, slug, source_rel):
    if slug.startswith('_legacy/'):
        d = exp_dir(category, slug)
    else:
        d = exp_dir(category, slug)
    os.makedirs(d, exist_ok=True)
    readme = os.path.join(d, 'README.md')
    if os.path.exists(readme):
        return
    with open(readme, 'w') as f:
        f.write(f'# Migrated from `{source_rel}`\n\nOutputs in `outputs/`.\n')


def cleanup_empty_outputs(dry_run=False):
    outputs_root = os.path.join(PROJECT_ROOT, 'outputs')
    for dirpath, dirnames, filenames in os.walk(outputs_root, topdown=False):
        if dirpath == outputs_root:
            continue
        if not dirnames and not filenames:
            if not dry_run:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass


def write_outputs_readme():
    path = os.path.join(PROJECT_ROOT, 'outputs', 'README.md')
    text = """# outputs/ (deprecated)

All experiment results live under **`experiments/`**:

```text
experiments/fig3/<slug>/outputs/     # Fig.3 panel diagonal integration
experiments/multi_omics/<slug>/outputs/
experiments/baselines/<slug>/outputs/
```

See `experiments/fig3/INDEX.csv` for the full Fig.3 registry.

Re-run migration: `python scripts/fig3/migrate_outputs_to_experiments.py`
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    all_results = []
    all_results += migrate_registered(args.dry_run)
    all_results += migrate_orphan_fig3(args.dry_run)
    all_results += migrate_other(args.dry_run)

    if not args.dry_run:
        cleanup_empty_outputs()
        write_outputs_readme()
        # refresh fig3 run.sh / INDEX without symlinks
        import sync_experiments
        sync_experiments.sync(dry_run=False)

    from collections import Counter
    c = Counter(r[0] for r in all_results)
    print('Migration summary:', dict(c))
    for status, rel in all_results:
        if status not in ('absent',):
            print(f'  {status:12} {rel}')


if __name__ == '__main__':
    main()
