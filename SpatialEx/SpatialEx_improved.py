"""
Improved SpatialEx+ trainer using Graph Transformer and Cross-Attention Translator.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import preprocess as pp
from .utils import create_optimizer
from .model_improved import Model_GT, Model_Plus_GT, CrossAttentionTranslator


class SpatialExP_GT:
    """Improved SpatialEx+ trainer with Graph Transformer and Cross-Attention Translator."""
    
    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 use_agg=True,
                 platform='Xenium',
                 seed=0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 weight_decay=0,
                 optimizer="adam",
                 batch_size=4096,
                 encoder="transformer",
                 hidden_dim=512,
                 num_layers=2,
                 num_heads=8,
                 epochs=1000,
                 lr=0.001,
                 prune=10000,
                 loss_fn="mse",
                 num_neighbors=7,
                 graph_kind='spatial',
                 save_path=None,
                 dropout=0.1,
                 use_mfp=True,
                 translator_hidden_dim=None):
        
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = pp.sparse_mx_to_torch_sparse_tensor(graph1).to(device)
        self.graph2 = pp.sparse_mx_to_torch_sparse_tensor(graph2).to(device)
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.epochs = epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.use_agg = use_agg
        self.platform = platform
        self.num_neighbors = num_neighbors
        self.graph_kind = graph_kind
        self.dropout = dropout
        self.use_mfp = use_mfp
        self.translator_hidden_dim = translator_hidden_dim if translator_hidden_dim is not None else hidden_dim
        
        self.slice1_dataloader = pp.Build_dataloader(
            adata1, graph=graph1, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )
        self.slice2_dataloader = pp.Build_dataloader(
            adata2, graph=graph2, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )
        
        self.HE1 = torch.Tensor(adata1.obsm['he']).to(self.device)
        self.HE2 = torch.Tensor(adata2.obsm['he']).to(self.device)
        self.panelA1 = torch.Tensor(adata1.X).to(self.device)
        self.panelB2 = torch.Tensor(adata2.X).to(self.device)
        
        self.in_dim1 = adata1.obsm['he'].shape[1]
        self.in_dim2 = adata2.obsm['he'].shape[1]
        self.out_dim1 = adata1.n_vars
        self.out_dim2 = adata2.n_vars
        
        self.module_HA = Model_Plus_GT(
            in_dim=self.in_dim1,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim1,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_mfp=self.use_mfp,
            platform=self.platform
        ).to(self.device)
        
        self.module_HB = Model_Plus_GT(
            in_dim=self.in_dim2,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim2,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_mfp=self.use_mfp,
            platform=self.platform
        ).to(self.device)
        
        # Use Cross-Attention Translator instead of simple MLP regression
        self.rm_AB = CrossAttentionTranslator(
            self.out_dim1, self.translator_hidden_dim, self.out_dim2,
            num_heads=max(1, self.num_heads // 2), dropout=self.dropout
        ).to(self.device)
        
        self.rm_BA = CrossAttentionTranslator(
            self.out_dim2, self.translator_hidden_dim, self.out_dim1,
            num_heads=max(1, self.num_heads // 2), dropout=self.dropout
        ).to(self.device)
        
        self.models = [self.module_HA, self.module_HB, self.rm_AB, self.rm_BA]
        self.optimizer = create_optimizer(optimizer, self.models, self.lr, self.weight_decay)
    
    def train(self):
        pp.set_random_seed(self.seed)
        self.module_HA.train()
        self.module_HB.train()
        self.rm_AB.train()
        self.rm_BA.train()
        
        print('\n')
        print('=================================== Start training (Improved GT) =========================================')
        
        if self.platform == 'Xenium':
            for epoch in tqdm(range(self.epochs)):
                batch_iter = zip(self.slice1_dataloader, self.slice2_dataloader)
                for data1, data2 in batch_iter:
                    graph1 = data1[0]['graph'].to(self.device)
                    he1 = data1[0]['he'].to(self.device)
                    panel_1a = data1[0]['exp'].to(self.device)
                    graph2 = data2[0]['graph'].to(self.device)
                    he2 = data2[0]['he'].to(self.device)
                    panel_2b = data2[0]['exp'].to(self.device)
                    agg_mtx1 = data1[0]['agg_mtx'].to(self.device)
                    agg_exp1 = data1[0]['agg_exp'].to(self.device)
                    agg_mtx2 = data2[0]['agg_mtx'].to(self.device)
                    agg_exp2 = data2[0]['agg_exp'].to(self.device)
                    selection1 = data1[0]['selection']
                    selection2 = data2[0]['selection']
                    
                    loss1, _ = self.module_HA(he1, graph1, panel_1a, agg_exp1, agg_mtx1, self.use_agg, selection1)
                    loss2, _ = self.module_HB(he2, graph2, panel_2b, agg_exp2, agg_mtx2, self.use_agg, selection2)
                    
                    panel_2a = self.module_HA.predict(he2, graph2, grad=False)
                    panel_1b = self.module_HB.predict(he1, graph1, grad=False)
                    
                    loss3, _ = self.rm_AB(panel_1a, panel_1b, torch.spmm(agg_mtx1, panel_1b[selection1]), agg_mtx1, self.use_agg)
                    loss4, _ = self.rm_BA(panel_2b, panel_2a, torch.spmm(agg_mtx2, panel_2a[selection2]), agg_mtx2, self.use_agg)
                    
                    loss5, _ = self.rm_AB(panel_2a[selection2], panel_2b, agg_exp2, agg_mtx2, self.use_agg)
                    loss6, _ = self.rm_BA(panel_1b[selection1], panel_1a, agg_exp1, agg_mtx1, self.use_agg)
                    
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        
        elif self.platform == 'Visium':
            for epoch in tqdm(range(self.epochs)):
                loss1, _ = self.module_HA(self.HE1, self.graph1, self.panelA1, use_agg=False)
                loss2, _ = self.module_HB(self.HE2, self.graph2, self.panelB2, use_agg=False)
                
                panelA2 = self.module_HA.predict(self.HE2, self.graph2, grad=False)
                panelB1 = self.module_HB.predict(self.HE1, self.graph1, grad=False)
                
                loss3, _ = self.rm_AB(panelA2, self.panelB2, use_agg=False)
                loss4, _ = self.rm_BA(panelB1, self.panelA1, use_agg=False)
                
                loss5, _ = self.rm_AB(self.panelA1, panelB1, use_agg=False)
                loss6, _ = self.rm_BA(self.panelB2, panelA2, use_agg=False)
                
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def auto_inference(self):
        self.module_HA.eval()
        self.module_HB.eval()
        self.rm_AB.eval()
        self.rm_BA.eval()
        
        panelA1_direct = self.module_HA.predict(self.HE1, self.graph1, grad=False)
        panelB1_indirect = self.rm_AB.predict(panelA1_direct).detach().cpu().numpy()
        
        panelB2_direct = self.module_HB.predict(self.HE2, self.graph2, grad=False)
        panelA2_indirect = self.rm_BA.predict(panelB2_direct).detach().cpu().numpy()
        
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            np.save(self.save_path + 'B1.npy', panelB1_indirect)
            np.save(self.save_path + 'A2.npy', panelA2_indirect)
            print(f'The results have been sucessfully saved in {self.save_path}')
        
        return panelB1_indirect, panelA2_indirect

    def inference_direct(self, he, graph, panel):
        """Directly predict the specified panel with its corresponding backbone.

        Mirrors :meth:`SpatialEx.SpatialExP.inference_direct` so that
        :class:`SpatialExP_GT` can be used as a drop-in replacement in
        evaluation scripts.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        
        if panel == 'panelA':
            self.module_HA.eval()
            omics_direct = self.module_HA.predict(he, graph, grad=False)
        elif panel == 'panelB':
            self.module_HB.eval()
            omics_direct = self.module_HB.predict(he, graph, grad=False)
        else:
            raise ValueError(f"panel must be 'panelA' or 'panelB', got {panel}")
        
        return omics_direct.detach().cpu().numpy()
    
    def inference_indirect(self, he, graph, panel):
        """Indirectly infer the missing panel using a regression mapper.

        Mirrors :meth:`SpatialEx.SpatialExP.inference_indirect`.
        """
        he = torch.Tensor(he).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        
        if panel == 'panelB':
            self.module_HA.eval()
            self.rm_AB.eval()
            panelA1_direct = self.module_HA.predict(he, graph, grad=False)
            omics_indirect = self.rm_AB.predict(panelA1_direct)
            omics_indirect = omics_indirect.detach().cpu().numpy()
        elif panel == 'panelA':
            self.module_HB.eval()
            self.rm_BA.eval()
            panelB2_direct = self.module_HB.predict(he, graph, grad=False)
            omics_indirect = self.rm_BA.predict(panelB2_direct)
            omics_indirect = omics_indirect.detach().cpu().numpy()
        else:
            raise ValueError(f"panel must be 'panelA' or 'panelB', got {panel}")
        
        return omics_indirect
