import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import FFN
from .functions import *


class RGAT_Layer(nn.Module):
    def __init__(self, ndim, edim, num_heads=1, feat_drop=0.2):
        super(RGAT_Layer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim), \
            nn.Linear(self.ndim, dim, bias=False), nn.Linear(self.ndim, dim, bias=False)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)
    
    def forward_g(self, g):
        """ @Params:
                g: dgl.graph
                g.ndata['x']: node features
                g.edata['e']: edge features
        """

        node_feat, _ = self.forward(g.ndata['x'], g.edata['e'], g)
        g.ndata['x'] = node_feat
        return g

    def forward(self, x, lgx, graph):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
        # set the same device:
        g = graph.to(x.device)
        # g = graph
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lgx.view(-1, self.num_heads, self.d_k) if lgx.size(-1) == q.size(-1) else \
            lgx.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        # e = e.to('cpu')
        # q, k, v = q.to('cpu'), k.to('cpu'), v.to('cpu')
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            g.edata['e'] = e
            out_x = self.propagate_attention(g)

        # out_x = out_x.to(x.device)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)

        return out_x, lgx

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        # g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x


class Rel_Transformer_Layer(nn.Module):
    def __init__(self, ndim, edim, num_heads=1, feat_drop=0.2):
        super(Rel_Transformer_Layer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        # self.node_former_conv = NodeFormerConv(ndim, edim, num_heads)
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim), \
            nn.Linear(self.ndim, dim), nn.Linear(self.ndim, dim)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)
    
    def forward_g(self, g):
        """ @Params:
                g: dgl.graph
                g.ndata['x']: node features
                g.edata['e']: edge features
        """

        node_feat, _ = self.forward(g.ndata['x'], g.edata['e'], g)
        g.ndata['x'] = node_feat
        return g

    def forward(self, x, lgx, graph, relation_emb, **kwargs):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
        # set the same device:
        g = graph['graph']
        edge_type = graph['graph'].edata['type']
        # g = graph
        # pre-mapping q/k/v affine

        # out_x = self.node_former_conv(x.unsqueeze(0))
        
        in_degree_emb, out_degree_emb, path_len_emb, rel_weight_emb = \
            kwargs['in_degree_emb'], kwargs['out_degree_emb'], kwargs['path_len_emb'], kwargs['relation_weight']
        # degree embedding

        x = x + in_degree_emb(graph['in_degree']) + out_degree_emb(graph['out_degree'])

        dist_2d_emb = path_len_emb(graph['dist']).squeeze(-1)

        edge_type_weight = (relation_emb.weight * rel_weight_emb.weight).sum(dim=-1) / math.sqrt(self.d_k)

        path_2d_emb = (edge_type_weight[graph['path_edge_type']] * graph['path_average_weight']).sum(dim=-1)

        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
    
        q, k, v = q.view(self.num_heads, -1, self.d_k), k.view(self.num_heads, self.d_k, -1), v.view(self.num_heads, -1, self.d_k)
        
        qk_dot = torch.matmul(q, k) / math.sqrt(self.d_k) + path_2d_emb + dist_2d_emb
        qk_attn = torch.softmax(qk_dot, dim=-1)

        out_x = torch.matmul(qk_attn, v)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        
        out_x = self.ffn(out_x)

        return out_x, lgx
