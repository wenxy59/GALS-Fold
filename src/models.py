"""
Multi-state GVP-GNN for structure-conditioned RNA sequence design.
Based on Geometric Vector Perceptron (Jing et al., ICLR 2021).
"""

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric

from src.layers import *


class GeometricLongShortRNA(torch.nn.Module):
    """
    Autoregressive GVP-GNN with dual-branch architecture.

    Combines short-range GVP attention for local geometric precision with
    long-range dynamic projection for global context, fused via length-aware gating.

    Args:
        node_in_dim: Input node dimensions (n_scalar, n_vector).
        node_h_dim: Hidden node dimensions.
        edge_in_dim: Input edge dimensions.
        edge_h_dim: Hidden edge dimensions.
        num_layers: Number of encoder/decoder layers.
        drop_rate: Dropout rate.
        heads: Number of attention heads.
        num_anchors: Number of anchors for dynamic projection.
        local_window: Local exclusion window size.
        length_threshold: Sequence length threshold for gating.
        aux_loss_weight: Weight for auxiliary loss.
        out_dim: Output dimension (4 for RNA bases).
    """
    def __init__(
        self,
        node_in_dim = (64, 4),
        node_h_dim = (128, 16),
        edge_in_dim = (32, 1),
        edge_h_dim = (32, 1),
        num_layers = 3,
        drop_rate = 0.1,
        heads = 4,
        num_anchors = 32,
        local_window = 10,
        length_threshold = 150,
        aux_loss_weight = 0.01,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.heads = heads
        self.num_anchors = num_anchors
        self.aux_loss_weight = aux_loss_weight
        self.out_dim = out_dim
        activations = (F.silu, None)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Encoder layers: Dual-branch architecture
        self.encoder_layers = nn.ModuleList(
                GVPDualBranchConvLayer(
                    self.node_h_dim, self.edge_h_dim,
                    heads=self.heads,
                    num_anchors=self.num_anchors,
                    n_feedforward=2,
                    drop_rate=drop_rate,
                    local_window=local_window,
                    length_threshold=length_threshold,
                    activations=activations,
                    vector_gate=True,
                    residual=True,
                    norm_first=True
                )
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        
        decoder_edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, decoder_edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output projection
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))

        self.cached_aux_loss = None

    def forward(self, batch, return_aux_loss=False):
        """Forward pass with optional auxiliary loss computation."""
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        batch_index = batch.batch if hasattr(batch, 'batch') else None

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        total_aux_loss = 0.0
        for layer in self.encoder_layers:
            if return_aux_loss:
                h_V, aux_loss = layer(h_V, edge_index, h_E,
                                     return_aux_loss=True,
                                     batch_index=batch_index)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                h_V = layer(h_V, edge_index, h_E)

        # Pool multi-conformation features
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V

        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)

        logits = self.W_out(h_V)

        if return_aux_loss:
            avg_aux_loss = total_aux_loss / self.num_layers if self.num_layers > 0 else 0.0
            return logits, avg_aux_loss

        return logits

    @torch.no_grad()
    def sample(
            self,
            batch,
            n_samples,
            temperature: Optional[float] = 0.1,
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
        ):
        """
        Sample sequences autoregressively.

        Args:
            batch: Input graph batch.
            n_samples: Number of sequences to sample.
            temperature: Sampling temperature.
            logit_bias: Optional bias for fixing nucleotides [n_nodes, 4].
            return_logits: Whether to return logits.

        Returns:
            seq: Sampled sequences [n_samples, n_nodes].
            logits: Optional logits [n_samples, n_nodes, 4].
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        batch_index = batch.batch if hasattr(batch, 'batch') else None

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E,
                       return_aux_loss=False,
                       batch_index=batch_index)

        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for n_samples
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))

        # Expand edge index for batched decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)

        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Autoregressive decoding
        for i in range(num_nodes):
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])

            edge_mask = edge_index[1] % num_nodes == i
            edge_index_ = edge_index[:, edge_mask]
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True

            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                out = tuple_index(out, node_mask)

                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]

            lgts = self.W_out(out)
            if logit_bias is not None:
                lgts += logit_bias[i]
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        """Pool multi-conformation features via masked mean."""
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        n_conf_true = mask_confs.sum(1, keepdim=True)

        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        h_V = (h_V0.sum(dim=1) / n_conf_true,
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))

        return h_V, h_E

class GVPAttentionShortBranch(torch.nn.Module):
    '''
    Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        node_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        heads = 4,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.heads = heads
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                GVPAttentionConvLayer(self.node_h_dim, self.edge_h_dim, 
                                      heads=self.heads,
                                      activations=activations, vector_gate=True,
                                      drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        
        # Isolate decoder edge dimensions to prevent modifying self.edge_h_dim
        decoder_edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, decoder_edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output (unchanged)
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or not
        
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))
        
        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                    
            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats
            
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
                
            lgts = self.W_out(out)
            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):

        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)
        
        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        # Average pooling multi-conformation features
        h_V = (h_V0.sum(dim=1) / n_conf_true,               # (n_nodes, d_s)
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))  # (n_nodes, d_v, 3)
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],               # (n_edges, d_se)
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))  # (n_edges, d_ve, 3)

        return h_V, h_E


class gRNAde(torch.nn.Module):
    '''
    Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        node_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                gRNAdeLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or not
        
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))
        
        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                    
            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats
            
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
                
            lgts = self.W_out(out)
            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):

        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)
        
        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        # Average pooling multi-conformation features
        h_V = (h_V0.sum(dim=1) / n_conf_true,               # (n_nodes, d_s)
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))  # (n_nodes, d_v, 3)
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],               # (n_edges, d_se)
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))  # (n_edges, d_ve, 3)

        return h_V, h_E