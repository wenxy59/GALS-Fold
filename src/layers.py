################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

import functools
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math
#########################################################################

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position awareness.

    Args:
        max_positions: Maximum sequence length supported.
        embedding_dim: Dimension of positional encoding (should match scalar_dim).
    """
    def __init__(self, max_positions: int = 3000, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

        # Compute sinusoidal positional encodings
        pe = torch.zeros(max_positions, embedding_dim)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embedding_dim // 2])  # Handle odd embedding_dim

        # Register as buffer (not a parameter, but saved in state_dict)
        self.register_buffer('pe', pe)

    def forward(self, n_nodes, batch_index=None, device=None):
        """
        Args:
            n_nodes: Total number of nodes.
            batch_index: [n_nodes] batch assignment for batched graphs.
            device: Target device.

        Returns:
            Tensor of shape [n_nodes, embedding_dim].
        """
        if device is None:
            device = self.pe.device

        if batch_index is not None:
            unique_batches = torch.unique(batch_index)
            local_positions = torch.zeros(n_nodes, dtype=torch.long, device=device)
            for b in unique_batches:
                mask = (batch_index == b)
                local_positions[mask] = torch.arange(mask.sum(), device=device)
            local_positions = local_positions.clamp(max=self.max_positions - 1)
            return self.pe[local_positions]
        else:
            positions = torch.arange(n_nodes, device=device).clamp(max=self.max_positions - 1)
            return self.pe[positions]


#########################################################################

class GVPConvLayer(nn.Module):
    """
    GVP message passing layer with residual updates and feedforward network.

    Args:
        node_dims: Node embedding dimensions (n_scalar, n_vector).
        edge_dims: Edge embedding dimensions (n_scalar, n_vector).
        n_message: Number of GVPs in message function.
        n_feedforward: Number of GVPs in feedforward network.
        drop_rate: Dropout probability.
        autoregressive: If True, uses separate embeddings for src >= dst edges.
        activations: Tuple (scalar_act, vector_act).
        vector_gate: Whether to use vector gating.
    """
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            autoregressive=False, 
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        """
        Args:
            x: Tuple (s, V) node features.
            edge_index: [2, E] edge indices.
            edge_attr: Tuple (s, V) edge features.
            autoregressive_x: Optional separate embeddings for src >= dst edges.
            node_mask: Optional bool mask for selective node updates.
        """
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            if self.norm_first:
                dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            else:
                dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        
        if self.norm_first:
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

class GVPConv(MessagePassing):
    """
    GVP message passing without residual/feedforward (see GVPConvLayer).

    Args:
        in_dims: Input node dimensions (n_scalar, n_vector).
        out_dims: Output node dimensions (n_scalar, n_vector).
        edge_dims: Edge dimensions (n_scalar, n_vector).
        n_layers: Number of GVPs in message function.
        module_list: Preconstructed message function (overrides n_layers).
        aggr: Aggregation method ("add" for autoregressive, "mean" otherwise).
        activations: Tuple (scalar_act, vector_act).
        vector_gate: Whether to use vector gating.
    """
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)
#########################################################################
class GVPAttentionConvLayer(nn.Module):
    """
    GVP attention layer with residual updates and feedforward network.

    Args:
        node_dims: Node dimensions (n_scalar, n_vector).
        edge_dims: Edge dimensions (n_scalar, n_vector).
        heads: Number of attention heads.
        n_feedforward: Number of GVPs in feedforward network.
        drop_rate: Dropout probability.
        activations: Tuple (scalar_act, vector_act).
        vector_gate: Whether to use vector gating.
        residual: Whether to use residual connections.
        norm_first: Whether to apply layer norm before conv and ff.
    """
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            heads=4, 
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(GVPAttentionConvLayer, self).__init__()
        
        # Use GVPAttentionConv as the convolution module
        self.conv = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads, 
            activations=activations, 
            vector_gate=vector_gate
        )
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # Feedforward network
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for _ in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr):
        if self.norm_first:
            dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            dh = self.conv(x, edge_index, edge_attr)
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        return x
#########################################################################
class GVPAttentionConv(MessagePassing):
    """
    GVP-based multi-head attention with edge bias.

    Q, K, V projections and attention scoring are SE(3)-equivariant.
    Edge features bias K and V for geometry-aware local attention.
    """
    def __init__(self, in_dims, out_dims, edge_dims,
                 heads=4,
                 activations=(F.silu, torch.sigmoid), 
                 vector_gate=True):
        
        super(GVPAttentionConv, self).__init__(aggr="add", node_dim=0)
        
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        self.heads = heads
        assert self.so % heads == 0 and self.vo % heads == 0, \
               "Output dimensions must be divisible by number of heads"
        
        self.head_s_dim = self.so // heads
        self.head_v_dim = self.vo // heads
        self.head_dims = (self.head_s_dim, self.head_v_dim)

        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)

        # Q, K, V Projections from GVP
        self.to_q = GVP_(in_dims, (self.so, self.vo))
        self.to_k = GVP_(in_dims, (self.so, self.vo))
        self.to_v = GVP_(in_dims, (self.so, self.vo))

        # Edge Bias Projections
        self.edge_proj_k = GVP_(edge_dims, (self.so, self.vo))
        self.edge_proj_v = GVP_(edge_dims, (self.so, self.vo))

        # Attention Head
        # Input: concatenated Q_i and K_j_biased (per-head)
        attn_in_dims = (2 * self.head_s_dim, 2 * self.head_v_dim)
        # Output: 1 scalar (energy) per head
        self.attn_gvp = GVP(attn_in_dims, (1, 0), 
                            activations=(None, None))

        # Output Projection (Multi-head fusion)
        self.out_proj = GVP_((self.so, self.vo), (self.so, self.vo))
        

    def forward(self, x, edge_index, edge_attr):
        x_s, _ = x
        n_nodes, n_conf = x_s.shape[:2]

        # Project Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Project Edge Biases
        e_k = self.edge_proj_k(edge_attr)
        e_v = self.edge_proj_v(edge_attr)
        
        # Flatten tuples for MessagePassing
        def flatten_heads(t, n, n_conf):
            s, v = t
            s_flat = s.reshape(n, n_conf * self.so)
            v_flat = v.reshape(n, n_conf * self.vo * 3)
            return torch.cat([s_flat, v_flat], dim=-1)

        q_flat = flatten_heads(q, n_nodes, n_conf)
        k_flat = flatten_heads(k, n_nodes, n_conf)
        v_flat = flatten_heads(v, n_nodes, n_conf)
        
        n_edges = edge_attr[0].shape[0]
        e_k_flat = flatten_heads(e_k, n_edges, n_conf)
        e_v_flat = flatten_heads(e_v, n_edges, n_conf)

        # Propagate messages
        message_flat = self.propagate(
            edge_index, 
            q=q_flat, k=k_flat, v=v_flat,
            e_k=e_k_flat, e_v=e_v_flat,
            n_conf=n_conf
        )

        # Un-flatten aggregated message
        s_dim_flat = n_conf * self.so
        s_out = message_flat[..., :s_dim_flat].view(n_nodes, n_conf, self.so)
        v_out = message_flat[..., s_dim_flat:].view(n_nodes, n_conf, self.vo, 3)
        out = (s_out, v_out)

        # Final output projection (fusion)
        final_out = self.out_proj(out)
        return final_out

    def message(self, q_i, k_j, v_j, e_k, e_v, n_conf: int,
                index: torch.Tensor, ptr: Optional[torch.Tensor],
                size_i: Optional[int]) -> torch.Tensor:
        # Un-flatten and reshape to heads
        def unflatten_heads(t_flat, n_conf):
            s_dim_flat = n_conf * self.so
            s = t_flat[..., :s_dim_flat].view(-1, n_conf, self.heads, self.head_s_dim)
            v = t_flat[..., s_dim_flat:].view(-1, n_conf, self.heads, self.head_v_dim, 3)
            return s, v

        q_i = unflatten_heads(q_i, n_conf)
        k_j = unflatten_heads(k_j, n_conf)
        v_j = unflatten_heads(v_j, n_conf)
        e_k = unflatten_heads(e_k, n_conf)
        e_v = unflatten_heads(e_v, n_conf)

        # Apply edge bias
        k_j_biased = tuple_sum(k_j, e_k)
        v_j_biased = tuple_sum(v_j, e_v)

        # Compute attention energy
        attn_input = tuple_cat(q_i, k_j_biased)
        energy_s = self.attn_gvp(attn_input)
        energy_scalar = energy_s.squeeze(-1)
        energy_flat = energy_scalar.view(-1, n_conf * self.heads)
        alpha_flat = softmax(energy_flat, index, ptr=ptr, dim=0)
        alpha = alpha_flat.view(-1, n_conf, self.heads)

        # Apply attention to biased value
        alpha_s = alpha.unsqueeze(-1)
        alpha_v = alpha.unsqueeze(-1).unsqueeze(-1)

        v_s_biased, v_v_biased = v_j_biased
        msg_s = alpha_s * v_s_biased
        msg_v = alpha_v * v_v_biased

        # Flatten message for output
        msg_s_flat = msg_s.contiguous().view(msg_s.shape[0], -1)
        msg_v_flat = msg_v.contiguous().view(msg_v.shape[0], -1)

        return torch.cat([msg_s_flat, msg_v_flat], dim=-1)

########################################################################
class gRNAdeLayer(nn.Module):
    '''
    GVPConvLayer for handling multiple conformations (encoder-only)
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(MultigRNAdeConvLayer, self).__init__()
        self.conv = MultigRNAdeConv(node_dims, node_dims, edge_dims, n_message,
                                 aggr="mean", activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        if self.norm_first:
            dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            dh = self.conv(x, edge_index, edge_attr)
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        return x

class MultigRNAdeConv(MessagePassing):
    '''
    GVPConv for handling multiple conformations
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(MultigRNAdeConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        n_conf = x_s.shape[1]
        
        # x_s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
        x_s = x_s.contiguous().view(x_s.shape[0], x_s.shape[1] * x_s.shape[2])        
        # x_v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
        x_v = x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * x_v.shape[2] * 3)
        
        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)
        
        return _split_multi(message, self.so, self.vo, n_conf)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        # [n_nodes, n_conf * d] -> [n_nodes, n_conf, d]
        s_i = s_i.view(s_i.shape[0], s_i.shape[1]//self.si, self.si)
        s_j = s_j.view(s_j.shape[0], s_j.shape[1]//self.si, self.si)
        # [n_nodes, n_conf * d * 3] -> [n_nodes, n_conf, d, 3]
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//(self.vi * 3), self.vi, 3)
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//(self.vi * 3), self.vi, 3)

        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge_multi(*message)

########################################################################
class GatedDynamicProjection(nn.Module):
    """
    Geometry-aware linear attention with dynamic projection for O(N) complexity.

    Uses scalar-driven clustering for anchor assignment while preserving
    vector information in the retrieval phase via dot-product scoring.

    Args:
        node_dims: Tuple of (scalar_dim, vector_dim).
        num_anchors: Number of anchor points for compression (r << N).
        drop_rate: Dropout rate.
        activations: Tuple of (scalar_activation, vector_activation).
        vector_gate: Whether to use vector gating in GVP.
    """
    def __init__(self, node_dims, num_anchors=32, drop_rate=0.1,
                 activations=(F.silu, torch.sigmoid), vector_gate=True, heads=None):
        super(GatedDynamicProjection, self).__init__()

        if heads is not None:
            warnings.warn(
                "The 'heads' parameter is deprecated and ignored.",
                DeprecationWarning, stacklevel=2
            )

        self.scalar_dim, self.vector_dim = node_dims
        self.num_anchors = num_anchors

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        # Q, K, V projections (preserve vector for geometry-aware retrieval)
        self.query_proj = GVP_(node_dims, node_dims)
        self.key_proj = GVP_(node_dims, node_dims)
        self.value_proj = GVP_(node_dims, node_dims)

        # Scalar-driven clustering: maps nodes to r anchors
        self.compress_scalar_proj = nn.Sequential(
            nn.Linear(self.scalar_dim, self.scalar_dim),
            nn.LayerNorm(self.scalar_dim),
            nn.SiLU(),
            nn.Linear(self.scalar_dim, num_anchors)
        )

        # Gating mechanism
        self.gate_gvp = GVP_(node_dims, (self.scalar_dim, 0), activations=(None, None))
        self.gate_bias = nn.Parameter(torch.full((self.scalar_dim,), -1.0))

        # Output projection
        self.out_proj = GVP_(node_dims, node_dims)
        self.dropout_s = nn.Dropout(drop_rate)
        self.dropout_v = _VDropout(drop_rate)

        # Attention scaling factor
        self.d_total = self.scalar_dim + 3 * self.vector_dim if self.vector_dim > 0 else self.scalar_dim

    def forward(self, x, batch_index=None, return_attention_map=False):
        """
        Args:
            x: Tuple (s, V) where s: [N, (C,) D_s], V: [N, (C,) D_v, 3].
            batch_index: Optional [N] batch assignment for batched graphs.
            return_attention_map: Whether to return implicit attention map.

        Returns:
            Tuple (out_s, out_v) and optionally attention map.
        """
        x_s, x_v = x

        # Average across conformers for long-range branch
        has_multi_conf = len(x_s.shape) == 3
        if has_multi_conf:
            n_nodes, n_conf, _ = x_s.shape
            x_s_avg = x_s.mean(dim=1)
            x_v_avg = x_v.mean(dim=1)
        else:
            n_nodes = x_s.shape[0]
            n_conf = 1
            x_s_avg = x_s
            x_v_avg = x_v

        x_avg = (x_s_avg, x_v_avg)

        # Handle batched graphs
        if batch_index is not None:
            unique_batches = torch.unique(batch_index)
            out_s_list, out_v_list = [], []
            attention_maps = [] if return_attention_map else None

            for b in unique_batches:
                mask = (batch_index == b)
                x_b = (x_s_avg[mask], x_v_avg[mask])

                if return_attention_map:
                    (out_s_b, out_v_b), attn_b = self._process_single_graph(x_b, return_attention_map=True)
                    attention_maps.append((mask, attn_b))
                else:
                    (out_s_b, out_v_b), _ = self._process_single_graph(x_b, return_attention_map=False)

                out_s_list.append(out_s_b)
                out_v_list.append(out_v_b)

            out_s = torch.zeros_like(x_s_avg)
            out_v = torch.zeros_like(x_v_avg)
            for b_idx, b in enumerate(unique_batches):
                mask = (batch_index == b)
                out_s[mask] = out_s_list[b_idx]
                out_v[mask] = out_v_list[b_idx]

            if return_attention_map:
                full_attention_map = torch.zeros(n_nodes, n_nodes, device=x_s_avg.device)
                for mask, attn_b in attention_maps:
                    indices = torch.where(mask)[0]
                    idx_i = indices.unsqueeze(1).expand(-1, len(indices))
                    idx_j = indices.unsqueeze(0).expand(len(indices), -1)
                    full_attention_map[idx_i, idx_j] = attn_b
            else:
                full_attention_map = None
        else:
            if return_attention_map:
                (out_s, out_v), full_attention_map = self._process_single_graph(x_avg, return_attention_map=True)
            else:
                (out_s, out_v), _ = self._process_single_graph(x_avg, return_attention_map=False)
                full_attention_map = None

        # Expand back to multi-conf shape if needed
        if has_multi_conf:
            out_s = out_s.unsqueeze(1).expand(-1, n_conf, -1)
            out_v = out_v.unsqueeze(1).expand(-1, n_conf, -1, -1)

        if return_attention_map:
            return (out_s, out_v), full_attention_map
        return (out_s, out_v)

    def _process_single_graph(self, x, return_attention_map=False):
        """Core linear attention with geometry-aware retrieval."""
        x_s, _ = x

        # Step 1: Q, K, V projections
        Q_s, Q_v = self.query_proj(x)
        K_s, K_v = self.key_proj(x)
        V_s, V_v = self.value_proj(x)

        # Step 2: Scalar-driven clustering -> P matrix [N, r]
        compress_logits = self.compress_scalar_proj(x_s)
        P = F.softmax(compress_logits, dim=-1)

        # Compress K, V via P^T: [r, d]
        K_bar_s = torch.matmul(P.transpose(0, 1), K_s)
        V_bar_s = torch.matmul(P.transpose(0, 1), V_s)
        K_bar_v = torch.einsum('nr,nvc->rvc', P, K_v)
        V_bar_v = torch.einsum('nr,nvc->rvc', P, V_v)

        # Step 3: Geometry-aware retrieval (scalar + vector dot-product)
        S_scalar = torch.matmul(Q_s, K_bar_s.transpose(0, 1))
        S_vector = torch.einsum('nvc,rvc->nr', Q_v, K_bar_v)
        attn_logits = (S_scalar + S_vector) / math.sqrt(self.d_total)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Step 4: Broadcast aggregated values
        linear_out_s = torch.matmul(attn_weights, V_bar_s)
        linear_out_v = torch.einsum('nr,rvc->nvc', attn_weights, V_bar_v)

        # Step 5: Gating
        gate_s = self.gate_gvp(x)
        gate_s = torch.sigmoid(gate_s + self.gate_bias)
        gate_v = torch.sigmoid(gate_s.mean(dim=-1, keepdim=True)).unsqueeze(-1)

        gated_out_s = linear_out_s * gate_s
        gated_out_v = linear_out_v * gate_v

        # Step 6: Output projection
        out_s, out_v = self.out_proj((gated_out_s, gated_out_v))
        out_s = self.dropout_s(out_s)
        out_v = self.dropout_v(out_v)

        attention_map = None
        if return_attention_map:
            attention_map = torch.matmul(P, P.transpose(0, 1))

        return (out_s, out_v), attention_map

#########################################################################
class LocalExclusionLoss(nn.Module):
    """
    Auxiliary loss that penalizes attention to local neighbors.

    Forces the long-range branch to focus on distant dependencies by
    penalizing high attention weights within a local window. Uses
    memory-efficient O(N*w) diagonal band extraction.

    Args:
        local_window: Size of local exclusion zone.
    """
    def __init__(self, local_window=10):
        super(LocalExclusionLoss, self).__init__()
        self.local_window = local_window

    def forward(self, attention_map, batch_index=None):
        """
        Args:
            attention_map: [N, N] implicit attention matrix.
            batch_index: Optional [N] batch assignment.

        Returns:
            Scalar loss value.
        """
        n_nodes = attention_map.shape[0]
        device = attention_map.device
        w = self.local_window

        if batch_index is not None:
            unique_batches = torch.unique(batch_index)
            total_energy = 0.0
            total_pairs = 0

            for b in unique_batches:
                mask = (batch_index == b)
                indices = torch.where(mask)[0]
                n_graph = len(indices)
                attn_submat = attention_map[indices][:, indices]
                energy, pairs = self._compute_band_energy(attn_submat, n_graph, w, device)
                total_energy += energy
                total_pairs += pairs

            loss = total_energy / max(total_pairs, 1.0)
        else:
            energy, pairs = self._compute_band_energy(attention_map, n_nodes, w, device)
            loss = energy / max(pairs, 1.0)

        return loss

    def _compute_band_energy(self, attn_matrix, n, w, device):
        """Sum attention weights in diagonal band (offsets 1 to w)."""
        total_energy = 0.0
        total_pairs = 0

        for offset in range(1, w + 1):
            if offset >= n:
                break
            diag_upper = torch.diagonal(attn_matrix, offset=offset)
            diag_lower = torch.diagonal(attn_matrix, offset=-offset)
            total_energy += diag_upper.sum() + diag_lower.sum()
            total_pairs += 2 * len(diag_upper)

        return total_energy, total_pairs

#########################################################################
class LengthAwareGating(nn.Module):
    """
    Length-aware gating for dual-branch fusion.

    Dynamically adjusts long-range branch contribution based on sequence length:
    short sequences (< threshold) suppress long-range to avoid noise.

    Args:
        length_threshold: Sequence length threshold.
        smoothness: Controls transition smoothness.
        init_alpha: Initial learnable weight (non-zero for gradient flow).
    """
    def __init__(self, length_threshold=150, smoothness=50, init_alpha=0.1):
        super(LengthAwareGating, self).__init__()
        self.length_threshold = length_threshold
        self.smoothness = smoothness
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, seq_length):
        """
        Args:
            seq_length: Sequence length (int or tensor).

        Returns:
            Gating factor gamma in [0, 1].
        """
        if not isinstance(seq_length, torch.Tensor):
            seq_length = torch.tensor(seq_length, dtype=torch.float32,
                                     device=self.alpha.device)
        gamma = torch.sigmoid((seq_length - self.length_threshold) / self.smoothness)
        return gamma

    def fuse(self, short_features, long_features, seq_length):
        """
        Fuse short-range and long-range features.

        Args:
            short_features: Output from short-range branch.
            long_features: Output from long-range branch.
            seq_length: Sequence length (int, float, or per-node tensor).

        Returns:
            Fused features: short + alpha * gamma * long.
        """
        gamma = self.forward(seq_length)

        # Expand gamma to match feature tensor shape
        if isinstance(gamma, torch.Tensor) and len(gamma.shape) > 0:
            n_dims = len(short_features.shape)
            if n_dims == 2:
                gamma = gamma.view(-1, 1)
            elif n_dims == 3:
                gamma = gamma.view(-1, 1, 1)
            elif n_dims == 4:
                gamma = gamma.view(-1, 1, 1, 1)

        fused = short_features + self.alpha * gamma * long_features
        return fused

#########################################################################
class GVPDualBranchConvLayer(nn.Module):
    """
    Dual-branch convolution layer with short-range and long-range branches.

    Combines local geometric precision (GVP attention) with global context
    (geometry-aware dynamic projection) via length-aware gating.

    Args:
        node_dims: Node embedding dimensions (n_scalar, n_vector).
        edge_dims: Edge embedding dimensions (n_scalar, n_vector).
        heads: Number of attention heads for short-range branch.
        num_anchors: Number of anchors for long-range projection.
        n_feedforward: Number of GVPs in feedforward network.
        drop_rate: Dropout rate.
        local_window: Size of local exclusion zone for auxiliary loss.
        length_threshold: Sequence length threshold for gating.
        activations: Tuple of (scalar_activation, vector_activation).
        vector_gate: Whether to use vector gating in GVP.
        residual: Whether to use residual connections.
        norm_first: Whether to apply layer norm before conv and ff.
    """
    def __init__(
            self,
            node_dims,
            edge_dims,
            heads=4,
            num_anchors=32,
            n_feedforward=2,
            drop_rate=.1,
            local_window=10,
            length_threshold=150,
            activations=(F.silu, torch.sigmoid),
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(GVPDualBranchConvLayer, self).__init__()

        self.node_dims = node_dims
        self.scalar_dim, self.vector_dim = node_dims

        # Short-range branch: GVP attention for local geometric precision
        self.short_conv = GVPAttentionConv(
            node_dims, node_dims, edge_dims,
            heads=heads,
            activations=activations,
            vector_gate=vector_gate
        )

        # Long-range branch: geometry-aware dynamic projection
        self.long_proj = GatedDynamicProjection(
            node_dims=node_dims,
            num_anchors=num_anchors,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate
        )

        # Positional encoding for long-range branch
        self.pos_embedding = SinusoidalPositionalEmbedding(
            max_positions=3000,
            embedding_dim=self.scalar_dim
        )
        self.pe_norm = nn.LayerNorm(self.scalar_dim)

        # Length-aware gating for scalar and vector features
        self.scalar_gating = LengthAwareGating(
            length_threshold=length_threshold,
            smoothness=50,
            init_alpha=0.1
        )
        self.vector_gating = LengthAwareGating(
            length_threshold=length_threshold,
            smoothness=50,
            init_alpha=0.1
        )

        # Auxiliary loss module
        self.local_exclusion_loss = LocalExclusionLoss(local_window=local_window)

        # Layer normalization and dropout
        GVP_ = functools.partial(GVP,
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # Feedforward network
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for _ in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

        self.residual = residual
        self.norm_first = norm_first
        self.cached_attention_map = None

    def forward(self, x, edge_index, edge_attr, return_aux_loss=False, batch_index=None):
        """
        Args:
            x: Tuple (s, V) where s: [N, C, D_s], V: [N, C, D_v, 3].
            edge_index: [2, E] edge indices.
            edge_attr: Tuple (s, V) edge features.
            return_aux_loss: Whether to compute local exclusion loss.
            batch_index: Optional [N] batch assignment.

        Returns:
            Updated node features (s, V) and optionally auxiliary loss.
        """
        x_s, x_v = x
        n_nodes = x_s.shape[0]

        # Compute per-node sequence length for batched graphs
        if batch_index is not None:
            ones = torch.ones(n_nodes, dtype=torch.float32, device=x_s.device)
            num_graphs = batch_index.max().item() + 1
            graph_node_counts = scatter_add(ones, batch_index, dim=0, dim_size=num_graphs)
            seq_length_per_node = graph_node_counts[batch_index]
            seq_length = seq_length_per_node.unsqueeze(-1)
        else:
            seq_length = n_nodes

        # Short-range branch
        if self.norm_first:
            dh_short = self.short_conv(self.norm[0](x), edge_index, edge_attr)
        else:
            dh_short = self.short_conv(x, edge_index, edge_attr)

        # Long-range branch with positional encoding
        pe = self.pos_embedding(n_nodes, batch_index=batch_index, device=x_s.device)
        if len(x_s.shape) == 3:
            pe = pe.unsqueeze(1)
        x_s_with_pos = self.pe_norm(x_s + pe)
        x_with_pos = (x_s_with_pos, x_v)

        if return_aux_loss:
            dh_long, attention_map = self.long_proj(
                x_with_pos, batch_index=batch_index, return_attention_map=True
            )
            self.cached_attention_map = attention_map
        else:
            dh_long = self.long_proj(x_with_pos, batch_index=batch_index)

        dh_long_scalar, dh_long_vector = dh_long

        # Fuse branches with length-aware gating
        fused_scalar = self.scalar_gating.fuse(dh_short[0], dh_long_scalar, seq_length)
        fused_vector = self.vector_gating.fuse(dh_short[1], dh_long_vector, seq_length)
        dh = (fused_scalar, fused_vector)

        # Residual + feedforward
        if self.norm_first:
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh

        # Compute auxiliary loss if requested
        if return_aux_loss and self.cached_attention_map is not None:
            aux_loss = self.local_exclusion_loss(
                self.cached_attention_map,
                batch_index=batch_index
            )
            return x, aux_loss

        return x

#########################################################################

class GVP(nn.Module):
    """
    Geometric Vector Perceptron for SE(3)-equivariant transformations.

    Args:
        in_dims: Tuple (n_scalar, n_vector) input dimensions.
        out_dims: Tuple (n_scalar, n_vector) output dimensions.
        h_dim: Intermediate vector channels (optional).
        activations: Tuple (scalar_act, vector_act).
        vector_gate: Whether to use vector gating.
    """
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        """Forward pass. Input/output: tuple (s, V) or scalar tensor."""
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s
    
#########################################################################

class _VDropout(nn.Module):
    """Vector channel dropout (elements of each channel dropped together)."""
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    """Combined dropout for tuples (s, V)."""
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    """Combined LayerNorm for tuples (s, V)."""
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

def tuple_sum(*args):
    """Elementwise sum of tuples (s, V)."""
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    """Elementwise concatenation of tuples (s, V)."""
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    """Index into tuple (s, V) along first dimension."""
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    """Random tuples (s, V) from normal distribution."""
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm clamped above eps. If sqrt=False, returns squared norm."""
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    """Split merged (s, V) tensor back into tuple."""
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v

def _merge(s, v):
    """Merge tuple (s, V) into single tensor."""
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)
