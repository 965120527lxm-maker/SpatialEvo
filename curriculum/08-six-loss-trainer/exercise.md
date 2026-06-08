# Lesson 08: Six-Loss Trainer

## Learning Goal
Implement a cycle-consistent training loop with 6 losses.

## Background
SpatialEx+ trains with 6 losses:
1. `L1(A→A)` — self-reconstruction on slice A
2. `L1(B→B)` — self-reconstruction on slice B
3. `L1(A→B)` — cross-translation A→B
4. `L1(B→A)` — cross-translation B→A
5. `L1(A→B→A)` — cycle consistency
6. `L1(B→A→B)` — cycle consistency

## Exercise
Implement `compute_six_losses` in `starter.py`:
- Inputs: `pred_aa, pred_bb, pred_ab, pred_ba, pred_aba, pred_bab` and ground truths `true_a, true_b`
- Return a dict with all 6 losses plus `"total"`

## Validation
Run `python test.py`. Checks that total = sum of components.
