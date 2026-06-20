# Fig.4 Experiments

Million-cell panel diagonal integration (Tutorial 3 / paper Fig.4).

```bash
# After big_1.npy + big_2.npy land in data/
./experiments/fig4/spatialexp_big/run.sh

# Or auto-wait and launch
ROOT=/path/to/SpatialEx bash scripts/fig4/wait_and_run_fig4.sh
```

Expected inputs under `data/`:
- `big_1.npy` / `big_2.npy` — bundled arrays **or** UNI H&E embeddings (+ matching `.h5ad` sidecar)
- `panel_selection/Big_by_name.csv` — 280-gene panel A/B split

Optional for metrics:
- full-slice h5ad with held-out panels (`--gt_big1`, `--gt_big2`)

Graph cache (skip 10–30 min rebuild on reruns):
- Saved under `<out_dir>/graphs/` as `graph{1,2}_spatial.npz` + `graph_meta.json`
- Use `--rebuild_graphs` to force rebuild
