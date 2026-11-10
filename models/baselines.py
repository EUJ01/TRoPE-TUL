"""C2 controls (model.arch): no rotation, no POI, no modal mixer. Inputs are the TRoPETUL spatial MLP and time
encoders, summed per point; a BiLSTM or a plain Transformer encoder; masked mean pool; the same loss as TRoPETUL
(UserLoss: CE + focal + SupCon on the pooled vector)."""
import copy
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.encode import PositionalEncode, FourierEncode
from models.model import SupConLoss, UserLoss


class PointInputs(nn.Module):
    """TRoPETUL's spatial MLP on x, y plus its four Fourier time encoders (dow, hour, minute, minutes since start)."""

    def __init__(self, embed_size, d_model):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(), nn.Linear(embed_size, d_model))
        self.temporal_modules = nn.ModuleList([FourierEncode(embed_size) for _ in range(4)])
        self.temporal = nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model))

    def forward(self, feats, pad):
        t = torch.cat([m(feats[..., 2 + i]) for i, m in enumerate(self.temporal_modules)], -1)
        return (self.spatial(feats[..., 0:2]) + self.temporal(t)).masked_fill(pad.unsqueeze(-1), 0)


def masked_mean(h, pad):
    return h.masked_fill(pad.unsqueeze(-1), 0).sum(1) / (~pad).sum(1, keepdim=True)


class Control(UserLoss, nn.Module):
    """Shared init: loss settings as TRoPETUL; config keys that only TRoPETUL uses are listed and ignored."""

    def __init__(self, embed_size, d_model, user, alpha, gamma, ce_weight, supcon_weight, focal_weight=None,
                 supcon_temp=0.1, **ignored):
        super().__init__()
        used = [k for k, v in ignored.items() if v not in (None, False)]
        if used:
            print(f"{type(self).__name__}: ignoring {sorted(used)}")
        self.alpha, self.gamma = alpha, gamma
        self.ce_weight, self.supcon_weight = ce_weight, supcon_weight
        self.focal_weight = 1 - ce_weight if focal_weight is None else focal_weight
        self.supcon_loss = SupConLoss(temperature=supcon_temp)
        self.inputs = PointInputs(embed_size, d_model)


class BiLSTMTUL(Control):
    """rope_layer = LSTM layers; d_model // 2 hidden per direction; packed, so padding never enters the recurrence."""

    def __init__(self, d_model, rope_layer, user, dropout=0.1, **kw):
        super().__init__(d_model=d_model, user=user, **kw)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=rope_layer, batch_first=True, bidirectional=True,
                            dropout=dropout if rope_layer > 1 else 0.0)
        self.user_pred_layers = nn.Linear(d_model, user)

    def encode(self, feats, poi_idx, pad):
        x = self.inputs(feats, pad)
        packed = pack_padded_sequence(x, (~pad).sum(1).cpu(), batch_first=True, enforce_sorted=False)
        h = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.size(1))[0]
        pooled = masked_mean(h, pad)
        return self.user_pred_layers(pooled), pooled


def rotate(x, angle):
    """x (..., d) as d / 2 complex planes, each turned by angle (..., d / 2); TrajFM's apply_rotary_emb, per head."""
    x_ = torch.view_as_complex(x.float().contiguous().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * torch.polar(torch.ones_like(angle), angle)).flatten(-2).type_as(x)


class RoPEEncoderLayer(nn.Module):
    """E4 plug-in: nn.TransformerEncoderLayer (post-norm, ReLU, batch_first) with q and k of every head rotated by
    angles Wr(x, y), d_head / 2 planes shared by the heads: the same rotation as the AttnTUL plug-in (attntul/layers.py)."""

    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        assert d_model % nhead == 0 and (d_model // nhead) % 2 == 0, (d_model, nhead)
        self.h, self.dk = nhead, d_model // nhead
        self.qkv, self.out = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        self.Wr = nn.Linear(2, self.dk // 2, bias=False)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop_attn, self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, xy, pad):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each (B, h, L, dk)
        angle = self.Wr(xy).float().unsqueeze(1)  # (B, 1, L, dk / 2)
        q, k = rotate(q, angle), rotate(k, angle)
        # fused kernel, as nn.TransformerEncoderLayer uses (memory); mask True = attend; dropout on the attention weights
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~pad[:, None, None, :],
                                                             dropout_p=self.drop_attn.p if self.training else 0.0)
        x = self.norm1(x + self.drop1(self.out(o.transpose(1, 2).reshape(B, L, D))))
        return self.norm2(x + self.drop2(self.ff(x)))


class TransformerTUL(Control):
    """rope_layer = encoder layers; sinusoidal positions; bidirectional attention with a key padding mask.
    plug_rope (E4, R1.10): RoPEEncoderLayer instead of nn.TransformerEncoderLayer; off = the E3 control unchanged."""

    def __init__(self, d_model, rope_layer, user, dropout=0.1, ff_dim=256, nhead=8, plug_rope=False, **kw):
        super().__init__(d_model=d_model, user=user, **kw)
        self.pos = PositionalEncode(d_model)
        if plug_rope:
            layer = RoPEEncoderLayer(d_model, nhead, ff_dim, dropout)  # deep copies, as nn.TransformerEncoder (same start weights)
            self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(rope_layer)])
        else:
            layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, rope_layer, enable_nested_tensor=False)
        self.user_pred_layers = nn.Linear(d_model, user)

    def encode(self, feats, poi_idx, pad):
        B, L = pad.shape
        x = self.inputs(feats, pad) + self.pos(torch.arange(L, device=pad.device).expand(B, L))
        if hasattr(self, 'layers'):
            for layer in self.layers:
                x = layer(x, feats[..., 0:2], pad)
        else:
            x = self.encoder(x, src_key_padding_mask=pad)
        pooled = masked_mean(x, pad)
        return self.user_pred_layers(pooled), pooled


ARCHS = {'bilstm': BiLSTMTUL, 'transformer': TransformerTUL}
