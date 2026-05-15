import torch
from .preprocess import construct_interaction_sparse,preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed, add_ppr_matrix
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse, MultiViewEncoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import scipy.sparse as sp
import torch

class Enhenced():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
        use_multi_view=True,
        ppr_alpha=0.2,
        contrastive_temperature=0.07 # 新增多视图参数
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype

        # 多视图参数
        self.use_multi_view = use_multi_view
        self.ppr_alpha = ppr_alpha
        self.contrastive_temperature = contrastive_temperature

        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           print("preprocess")
           preprocess(self.adata)
        
        if 'adj' not in adata.obsm.keys():
           if self.datatype in ['Stereo', 'Slide']:
              construct_interaction_KNN(self.adata)
           else:
              print("adj")
              construct_interaction_sparse(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
           print("label_CSL")
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           print("feat")
           get_feature(self.adata)

        # 计算PPR矩阵
        if self.use_multi_view and 'ppr' not in adata.obsm.keys():
            print("ppr")
            add_ppr_matrix(self.adata, alpha=self.ppr_alpha)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']

        graph_neigh_sparse = self.adata.obsm['graph_neigh']
        identity_sparse = sp.eye(graph_neigh_sparse.shape[0], format='csr')
        graph_neigh_with_self = graph_neigh_sparse + identity_sparse
        self.graph_neigh = self.sparse_mx_to_torch_sparse_tensor(graph_neigh_with_self).to(self.device)

        if self.use_multi_view:
            ppr_sparse = self.adata.obsm['ppr']
            self.ppr_matrix = self.sparse_mx_to_torch_sparse_tensor(ppr_sparse).to(self.device)
            print(f"PPR矩阵已加载为稀疏张量，形状: {self.ppr_matrix.shape}, 设备: {self.ppr_matrix.device}")

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        self.adj = preprocess_adj_sparse(self.adata.obsm['adj']).to(self.device)


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)

        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
              
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                
                return self.emb_rec
             else:  
                if self.datatype in ['Stereo', 'Slide']:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                   self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                else:
                   self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                
                return self.adata

    def train_multi_view(self):
        self.model = MultiViewEncoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin multi-view graph contrastive learning...')
        self.model.train()

        n_spots = self.features.shape[0]
        n_genes = self.features.shape[1]
        print(f"Training on {n_spots} spots × {n_genes} genes")

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.features_a = permutation(self.features)

            z_adj, z_ppr, z_adj_corrupted, h_recon, z_adj_act, z_ppr_act, z_adj_corrupted_act, h_recon_act = self.model(
                self.features, self.features_a, self.adj, self.ppr_matrix
            )

            contrastive_loss = self.multi_view_contrastive_loss(
                z_adj_act, z_ppr_act, z_adj_corrupted_act, temperature=self.contrastive_temperature
            )
            n_nodes = z_adj.shape[0]
            recon_loss = F.mse_loss(self.features, h_recon_act)

            # 总损失
            loss = self.lamda1 * recon_loss + self.lamda2 * contrastive_loss

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, Contrast: {contrastive_loss.item():.4f}')

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            z_adj, z_ppr, _, h_recon, z_adj_act, z_ppr_act, z_adj_corrupted_act, h_recon_act  = self.model(
                self.features, self.features_a, self.adj, self.ppr_matrix
            )
            self.emb_rec = z_adj.detach().cpu().numpy()

            self.adata.obsm['emb'] = self.emb_rec
            return self.adata
    
    def loss(self, emb_sp, emb_sc):
        '''\
        Calculate loss

        Parameters
        ----------
        emb_sp : torch tensor
            Spatial spot representation matrix.
        emb_sc : torch tensor
            scRNA cell representation matrix.

        Returns
        -------
        Loss values.

        '''
        map_probs = F.softmax(self.map_matrix, dim=1)
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
           
        loss_recon = F.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
           
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))

        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M

    def multi_view_contrastive_loss(self, z_adj, z_ppr, z_adj_corrupted, temperature=0.07, batch_size=10000):
        """
        批量化多视图对比学习损失
        从所有节点中随机采样一个批次来计算对比损失，避免全矩阵计算
        """
        n_nodes = z_adj.shape[0]

        if n_nodes <= batch_size:
            batch_size = n_nodes
            indices = torch.arange(n_nodes, device=self.device)
        else:
            indices = torch.randperm(n_nodes, device=self.device)[:batch_size]

        z_adj_batch = z_adj[indices]
        z_ppr_batch = z_ppr[indices]
        z_adj_corrupted_batch = z_adj_corrupted[indices]

        def batch_cosine_similarity(x, y):
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return torch.mm(x_norm, y_norm.T)

        pos_sim_matrix = batch_cosine_similarity(z_adj_batch, z_ppr_batch)
        neg_sim_matrix = batch_cosine_similarity(z_adj_batch, z_adj_corrupted_batch)

        pos_sim = torch.diag(pos_sim_matrix)

        pos_sim = torch.clamp(pos_sim, -1.0 + 1e-8, 1.0 - 1e-8)
        neg_sim_matrix = torch.clamp(neg_sim_matrix, -1.0 + 1e-8, 1.0 - 1e-8)

        numerator = torch.exp(pos_sim / temperature)

        mask = torch.eye(batch_size, device=self.device).bool()
        neg_exp = torch.exp(neg_sim_matrix / temperature)
        neg_exp_masked = neg_exp * (~mask).float()

        denominator = numerator.unsqueeze(1) + torch.sum(neg_exp_masked, dim=1, keepdim=True)
        denominator = torch.clamp(denominator, min=1e-8)

        losses = -torch.log(numerator.unsqueeze(1) / denominator)
        return torch.mean(losses)



