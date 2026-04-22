import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange


class PositionalEncode(nn.Module):
    """Non-learnable positional encoding layer proposed in the Transformer.
    """

    def __init__(self, hidden_size):
        super(PositionalEncode, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, 'B L -> (B L)'), self.inv_freq) #type: ignore
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = rearrange(pos_emb, '(B L) E -> B L E', B=B, L=L)

        return pos_emb


class FourierEncode(nn.Module):
    """A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        Args:
            x (Tensor): input sequence for encoding, (batch_size, seq_len, 1)

        Returns:
            Tensor: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)

        encode = x * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode


class RoPE_Attention_float(nn.Module):
    def __init__(self, hidd_dim):
        super(RoPE_Attention_float, self).__init__()
        
        self.hidd_dim = hidd_dim

        self.wq = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wk = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wv = nn.Linear(self.hidd_dim, self.hidd_dim)
        
        self.Wr = nn.Linear(2, self.hidd_dim // 2, bias=False)
    def forward(self, x, norm_coord, causal_mask, batch_mask):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.apply_rotary_emb(xq, xk, norm_coord)

        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.hidd_dim)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        scores = scores.masked_fill(batch_mask, float('-inf'))
        scores = F.softmax(scores.float(), dim=-1)
        
        output = torch.matmul(scores, xv)
        return output

    def apply_rotary_emb(self, xq, xk, norm_coord):
        _, traj_len, _ = xq.shape

        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        _freqs_cis = self.precompute_freqs_cis(norm_coord)
        
        xq_out = torch.view_as_real(xq_ * _freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * _freqs_cis).flatten(2)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)


    def precompute_freqs_cis(self, norm_coord):
        freqs = self.Wr(norm_coord)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
        return freqs_cis





class RoPE_Encoder(nn.Module):
    def __init__(self, dim, layers, max_seq_len = 10000):
        super().__init__()
        self.dim = dim

        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([RoPE_Encoder_layer(dim) 
                                     for _ in range(layers)])
        
    
    def forward(self, x, norm_coord, mask, src_key_padding_mask):
        _, max_seq_len, _ = x.shape

        src_key_padding_mask = src_key_padding_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, norm_coord, mask, src_key_padding_mask)
        return x

class RoPE_Encoder_layer(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()

        self.dim = dim

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        self.fc_norm = nn.LayerNorm(dim)
        self.RoPE_Attention_float = RoPE_Attention_float(dim)

    def forward(self, x, norm_coord, causal_mask, batch_mask):
        x_attn = self.RoPE_Attention_float(x, norm_coord, causal_mask, batch_mask)
        x = x + x_attn
        x = self.norm(x)

        x_fc = self.fc(x)
        x = x + x_fc
        x = self.fc_norm(x)
        return x
