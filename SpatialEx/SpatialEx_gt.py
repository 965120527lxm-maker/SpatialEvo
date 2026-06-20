"""Fig.2 Graph Transformer (GT) trainer — same protocol as :class:`SpatialEx`."""

import torch
from tqdm import tqdm

from . import preprocess as pp
from .model_improved import Model_GT
from .SpatialEx import SpatialEx
from .utils import create_optimizer


class SpatialExGT(SpatialEx):
    """Baseline SpatialEx with Graph Transformer backbone instead of HGNN."""

    def __init__(
        self,
        adata1,
        adata2,
        graph1,
        graph2,
        num_layers=2,
        hidden_dim=128,
        num_heads=8,
        dropout=0.1,
        use_mfp=True,
        use_dgi=False,
        mfp_weight=0.1,
        dgi_weight=1.0,
        epochs=500,
        seed=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weight_decay=0,
        optimizer="adam",
        batch_size=4096,
        encoder="transformer",
        lr=0.001,
        loss_fn="mse",
        num_neighbors=7,
        graph_kind="spatial",
        prune=10000,
        save_path=None,
    ):
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = graph1
        self.graph2 = graph2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_mfp = use_mfp
        self.use_dgi = use_dgi
        self.mfp_weight = mfp_weight
        self.dgi_weight = dgi_weight
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr = lr
        self.loss_fn = loss_fn
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.prune = prune
        self.save_path = save_path

        self.in_dim1 = self.adata1.obsm["he"].shape[1]
        self.in_dim2 = self.adata2.obsm["he"].shape[1]
        self.out_dim1 = self.adata1.n_vars
        self.out_dim2 = self.adata2.n_vars

        gt_kw = dict(
            num_layers=num_layers,
            loss_fn=loss_fn,
            device=device,
            num_heads=num_heads,
            dropout=dropout,
            use_mfp=use_mfp,
            use_dgi=use_dgi,
            mfp_weight=mfp_weight,
            dgi_weight=dgi_weight,
        )
        self.module_HA = Model_GT(
            in_dim=self.in_dim1, hidden_dim=hidden_dim, out_dim=self.out_dim1, **gt_kw
        ).to(device)
        self.module_HB = Model_GT(
            in_dim=self.in_dim2, hidden_dim=hidden_dim, out_dim=self.out_dim2, **gt_kw
        ).to(device)
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)

        self.slice1_dataloader = pp.Build_dataloader(
            adata1, graph=graph1, graph_norm="hpnn", feat_norm=False,
            prune=[prune, prune], drop_last=False,
        )
        self.slice2_dataloader = pp.Build_dataloader(
            adata2, graph=graph2, graph_norm="hpnn", feat_norm=False,
            prune=[prune, prune], drop_last=False,
        )

    def train(self):
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        print("\n")
        aux = "DGI" if self.use_dgi else ("MFP" if self.use_mfp else "none")
        print(f"=================================== Start training (GT + {aux}) =========================================")
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            batch_iter = zip(self.slice1_dataloader, self.slice2_dataloader)
            for data1, data2 in batch_iter:
                graph1 = data1[0]["graph"].to(self.device)
                he1 = data1[0]["he"].to(self.device)
                selection1 = data1[0]["selection"]
                graph2 = data2[0]["graph"].to(self.device)
                he2 = data2[0]["he"].to(self.device)
                selection2 = data2[0]["selection"]
                agg_mtx1 = data1[0]["agg_mtx"].to(self.device)
                agg_exp1 = data1[0]["agg_exp"].to(self.device)
                agg_mtx2 = data2[0]["agg_mtx"].to(self.device)
                agg_exp2 = data2[0]["agg_exp"].to(self.device)

                loss1, _ = self.module_HA(graph1, he1, agg_exp1, agg_mtx1, selection1)
                loss2, _ = self.module_HB(graph2, he2, agg_exp2, agg_mtx2, selection2)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_iter.set_description(f"#Epoch: {epoch}: train_loss: {loss.item():.2f}")
