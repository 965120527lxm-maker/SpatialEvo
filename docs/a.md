# Top Marker Genes：HGNN+Cycle vs MLP+Strict MNN（Slice2）

ΔPCC = MLP+MNN PCC − HGNN+Cycle PCC（official split，Slice2 PanelA held-out）

| Gene | HGNN+Cycle | MLP+MNN | ΔPCC |
|------|----------:|--------:|-----:|
| CPA3 | 0.079 | 0.578 | +0.498 |
| CTSG | 0.072 | 0.555 | +0.483 |
| CLEC14A | 0.330 | 0.638 | +0.308 |
| EGFL7 | 0.141 | 0.358 | +0.217 |
| CLDN5 | 0.046 | 0.262 | +0.216 |
| HOXD9 | 0.237 | 0.451 | +0.214 |
| CD93 | 0.409 | 0.614 | +0.205 |
| DERL3 | 0.140 | 0.336 | +0.197 |
| ADIPOQ | 0.461 | 0.647 | +0.186 |
| AQP1 | 0.438 | 0.617 | +0.180 |

数据来源：`experiments/fig3/top_marker_gains_hgnn_vs_mnn.csv`
