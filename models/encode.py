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
    """TrajFM rotation (single head): plane angles Wr(x, y). Defaults reproduce v2-base exactly.
    Hybrid / E10 options take planes from the end of the angle vector:
    - time: 5 fixed planes, harmonics 1-3 of 24 h on local hour and 1-2 of 7 d on day-of-week (relative mod period);
    - gap_planes: fixed log-spaced frequencies (periods 10 min to 30 d) on minutes since start (TO-RoPE time gap);
    - anchor_planes: like Wr, but the key gets the negative angle, so scores depend on absolute location;
    - space=False: no Wr; the planes it would use stay unrotated (with nothing else: the 'none' control)."""

    def __init__(self, hidd_dim, space=True, time=False, gap_planes=0, anchor_planes=0):
        super(RoPE_Attention_float, self).__init__()

        self.hidd_dim = hidd_dim

        self.wq = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wk = nn.Linear(self.hidd_dim, self.hidd_dim)
        self.wv = nn.Linear(self.hidd_dim, self.hidd_dim)

        planes = self.hidd_dim // 2
        self.nt, self.ng, self.na = (STRoPE_Attention.N_TIME if time else 0), gap_planes, anchor_planes
        self.ns = planes - self.nt - self.ng - self.na
        assert self.ns >= 0, f"{planes} planes is too few"
        if space:
            self.Wr = nn.Linear(2, self.ns, bias=False)
        if self.na:
            self.Wa = nn.Linear(2, self.na, bias=False)
        self.register_buffer('time_freq', torch.tensor([2 * math.pi * k / 24 for k in (1, 2, 3)] +
                                                       [2 * math.pi * k / 7 for k in (1, 2)])[: self.nt], persistent=False)
        self.register_buffer('gap_freq', 2 * math.pi / torch.logspace(math.log10(10), math.log10(30 * 1440), self.ng),
                             persistent=False)  # not in state dicts: v2-base checkpoints still load

    def forward(self, x, feats, causal_mask, batch_mask):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.apply_rotary_emb(xq, xk, feats)

        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.hidd_dim)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float('-inf'))
        scores = scores.masked_fill(batch_mask, float('-inf'))
        scores = F.softmax(scores.float(), dim=-1)

        output = torch.matmul(scores, xv)
        return output

    def apply_rotary_emb(self, xq, xk, feats):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        aq, ak = self.angles(feats)
        xq_out = torch.view_as_real(xq_ * torch.polar(torch.ones_like(aq), aq)).flatten(2)
        xk_out = torch.view_as_real(xk_ * torch.polar(torch.ones_like(ak), ak)).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def angles(self, feats):
        """(query angles, key angles), each (B, L, hidd_dim // 2)."""
        B, L, _ = feats.shape
        xy = feats[..., 0:2]
        a = [self.Wr(xy) if hasattr(self, 'Wr') else xy.new_zeros(B, L, self.ns)]
        if self.nt:
            t = torch.cat([feats[..., 3:4].expand(B, L, 3), feats[..., 2:3].expand(B, L, 2)], -1)
            a.append(t * self.time_freq)
        if self.ng:
            a.append(feats[..., 5:6] * self.gap_freq)
        q = k = torch.cat(a, -1).float()
        if self.na:
            anchor = self.Wa(xy).float()
            q, k = torch.cat([q, anchor], -1), torch.cat([k, -anchor], -1)
        return q, k





class STRoPE_Attention(nn.Module):
    """Anchored ST-RoPE, multi-head. Each head's 2-D planes are split into
    - space: learned 2-D direction and log-frequency per plane (RoPE-Mixed); wavelengths initialised
      log-spaced over [lmin, lmax] metres; relative (translation-invariant);
    - time: fixed harmonics k=1..3 of 24 h and k=1..2 of 7 d on local hour / day-of-week (ClockRoPE);
      relative modulo the period;
    - anchor: like space, but the key is rotated by the negative angle, so the score depends on the
      sum of the query and key angles, i.e. on absolute location (T-RoPE's sum angle, applied to space).
    Inputs: x (B, L, D); feats (B, L, >=4) = scaled x, y, day-of-week, hour."""

    N_TIME = 5

    def __init__(self, dim, heads=8, time=True, anchor_planes=0, learn_space=True, lmin=50.0, lmax=50000.0,
                 scale=4000.0):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        planes = self.dh // 2
        self.nt, self.na = (self.N_TIME if time else 0), anchor_planes
        self.ns = planes - self.nt - self.na
        assert self.dh % 2 == 0 and self.ns > 0, f"head dim {self.dh} leaves no spatial planes"

        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))

        def planes_init(n):  # log angular frequency per scaled-coordinate unit, random direction
            wl = torch.logspace(math.log10(lmin), math.log10(lmax), n)
            log_w = torch.log(2 * math.pi * scale / wl).expand(heads, n).clone()
            return log_w, torch.rand(heads, n) * math.pi

        log_w, alpha = planes_init(self.ns)
        self.space_log_omega = nn.Parameter(log_w, requires_grad=learn_space)
        self.space_alpha = nn.Parameter(alpha, requires_grad=learn_space)
        if self.na:
            log_w, alpha = planes_init(self.na)
            self.anchor_log_omega, self.anchor_alpha = nn.Parameter(log_w), nn.Parameter(alpha)
        self.register_buffer('time_freq', torch.tensor([2 * math.pi * k / 24 for k in (1, 2, 3)] +
                                                       [2 * math.pi * k / 7 for k in (1, 2)])[: self.nt])

    @staticmethod
    def _angles(xy, log_omega, alpha):
        direction = torch.stack([alpha.cos(), alpha.sin()], -1) * log_omega.exp().unsqueeze(-1)  # (H, n, 2)
        return torch.einsum('bld,hnd->bhln', xy, direction)

    def angles(self, feats):
        """(query angles, key angles), each (B, H, L, planes)."""
        B, L, _ = feats.shape
        xy = feats[..., 0:2].float()
        a = [self._angles(xy, self.space_log_omega, self.space_alpha)]
        if self.nt:
            t = torch.cat([feats[..., 3:4].expand(B, L, 3), feats[..., 2:3].expand(B, L, 2)], -1).float()
            a.append((t * self.time_freq).unsqueeze(1).expand(B, self.h, L, self.nt))
        q, k = torch.cat(a, -1), torch.cat(a, -1)
        if self.na:
            anchor = self._angles(xy, self.anchor_log_omega, self.anchor_alpha)
            q, k = torch.cat([q, anchor], -1), torch.cat([k, -anchor], -1)
        return q, k

    def forward(self, x, feats, causal_mask, batch_mask):
        B, L, _ = x.shape
        split = lambda t: torch.view_as_complex(t.float().reshape(B, L, self.h, self.dh // 2, 2).transpose(1, 2).contiguous())
        q, k = split(self.wq(x)), split(self.wk(x))
        v = self.wv(x).reshape(B, L, self.h, self.dh).transpose(1, 2)
        aq, ak = self.angles(feats)
        q = torch.view_as_real(q * torch.polar(torch.ones_like(aq), aq)).flatten(3)
        k = torch.view_as_real(k * torch.polar(torch.ones_like(ak), ak)).flatten(3)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float('-inf'))
        scores = scores.masked_fill(batch_mask.unsqueeze(1), float('-inf'))
        out = F.softmax(scores, -1) @ v.float()
        return self.wo(out.transpose(1, 2).reshape(B, L, -1).type_as(x))


class RoPE_Encoder(nn.Module):
    """rope: None = the TrajFM rotation (v2-base); else kwargs of STRoPE_Attention plus 'type': 'st'."""

    def __init__(self, dim, layers, rope=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([RoPE_Encoder_layer(dim, rope, dropout) for _ in range(layers)])

    def forward(self, x, feats, mask, src_key_padding_mask):
        src_key_padding_mask = src_key_padding_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, feats, mask, src_key_padding_mask)
        return x

class RoPE_Encoder_layer(nn.Module):
    def __init__(self, dim, rope=None, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()  # Identity at 0: no extra random draws

        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.fc_norm = nn.LayerNorm(dim)
        self.st = rope is not None and rope.get('type') == 'st'  # 'trajfm': RoPE_Attention_float kwargs
        kw = {k: v for k, v in (rope or {}).items() if k not in ('type', 'causal')}
        self.RoPE_Attention_float = (STRoPE_Attention if self.st else RoPE_Attention_float)(dim, **kw)

    def forward(self, x, feats, causal_mask, batch_mask):
        x_attn = self.RoPE_Attention_float(x, feats, causal_mask, batch_mask)
        x = x + self.drop(x_attn)
        x = self.norm(x)

        x_fc = self.fc(x)
        x = x + self.drop(x_fc)
        x = self.fc_norm(x)
        return x
