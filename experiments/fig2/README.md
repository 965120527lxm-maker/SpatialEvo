# Fig.2 Experiments

H&E-to-omics single-cell prediction (Tutorial 1 / paper Fig.2).

```bash
./experiments/fig2/spatialex_breast_cancer/run.sh   # HGNN (SpatialEx)
./experiments/fig2/gt_breast_cancer/run.sh          # Graph Transformer + MFP (128-d)
./experiments/fig2/gt512_breast_cancer/run.sh       # Graph Transformer + MFP (512-d)
./experiments/fig2/gt_dgi_breast_cancer/run.sh     # Graph Transformer + DGI
./experiments/fig2/deeppt_breast_cancer/run.sh
```

Metrics: PCC, SSIM, CMD on Rep1 and Rep2 (313 genes).
