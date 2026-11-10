import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange

from models.encode import PositionalEncode, FourierEncode, RoPE_Encoder

PAD_TOKEN = 5


def sigmoid_focal_loss(inputs, targets, alpha, gamma):
    """torchvision.ops.sigmoid_focal_loss with reduction='mean' (inlined to drop torchvision)."""
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        return -mean_log_prob_pos.mean()


class UserLoss:
    """user_loss on top of self.encode(feats, poi_idx, pad) -> (logits, pooled); shared by TRoPETUL and the C2 controls.
    Needs alpha, gamma, ce_weight, focal_weight, supcon_weight and supcon_loss."""

    def loss_terms(self, feats, poi_idx, pad, users):
        """((ce, focal, supcon) unweighted, logits, pooled vector); C4 reads the terms' gradients."""
        pred_user, pooled_feats = self.encode(feats, poi_idx, pad)
        ce_loss = F.cross_entropy(pred_user, users)
        focal_loss = sigmoid_focal_loss(pred_user, F.one_hot(users, pred_user.size(1)).float(), self.alpha, self.gamma)
        supcon_loss = self.supcon_loss(pooled_feats, users)
        return (ce_loss, focal_loss, supcon_loss), pred_user, pooled_feats

    def user_loss(self, feats, poi_idx, pad, users):
        """Returns (total loss, logits)."""
        (ce_loss, focal_loss, supcon_loss), pred_user, _ = self.loss_terms(feats, poi_idx, pad, users)
        total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss + self.supcon_weight * supcon_loss
        return total_loss, pred_user


class TRoPETUL(UserLoss, nn.Module):
    """Inputs per point: feats (B, L, 6) = x, y, day-of-week, hour, minute, minutes since start;
    poi_idx (B, L) rows of poi_table, or with cell_embed (B, L, 1+K): column 0 the poi_table row, then the cell id
    per level (fine to coarse, 0 = unknown); pad (B, L) True on padding.

    token_embed: the released constant token embedding plus the constant second positional
    term (F8; off in v2-base). pool_mask: padding excluded from attention pooling (F3).
    rope: None = TrajFM rotation (v2-base); {'type': 'trajfm', 'causal': ...} = TrajFM rotation with a mask choice;
    {'type': 'st', ...} = Anchored ST-RoPE (encode.STRoPE_Attention kwargs, plus 'causal', default True).
    poi_table None (C1, model.poi false): no POI token and no POI layer; column 0 of poi_idx is then unused.
    poi_drop p (2026-10-21): in training, the POI token of each point is zeroed with probability p (0 = off).
    poi_gate (2026-10-21): the POI token is scaled by one learned scalar that starts at 0 (False = no gate).
    poi_gate_cell (2026-10-22): two learned scalars on the POI token, both starting at 0: one for points whose finest
    cell is known, one for points whose finest cell id is 0 (unknown, before drop_fine). Needs cell_embed."""

    def __init__(self, embed_size, d_model, poi_table, rope_layer, user, alpha, gamma, ce_weight, supcon_weight,
                 token_embed=False, pool_mask=True, rope=None, focal_weight=None,
                 dropout=0.1, ff_dim=256, nhead=8, supcon_temp=0.1, head_dropout=0.0, seq_dropout=0.0,
                 cell_embed=None, fourier_xy=None, poi_drop=0.0, poi_gate=False, poi_gate_cell=False, seq_arch='rope', xy_token=True):
        super().__init__()
        assert seq_arch in ('rope', 'bilstm'), seq_arch
        assert 0 <= poi_drop < 1, 'poi_drop must be in [0, 1)'
        assert poi_table is not None or not (poi_drop or poi_gate), 'poi_drop / poi_gate need the POI token (model.poi true)'
        assert not poi_gate_cell or (poi_table is not None and cell_embed), 'poi_gate_cell needs the POI and cell tokens'
        assert not (poi_gate and poi_gate_cell), 'poi_gate and poi_gate_cell: pick one'
        self.poi_drop = poi_drop
        # OURS 2026-11-09 E14 (R1.10): xy_token False drops the xy-MLP token from the mixer; the rotation still reads xy.
        # No module is added or removed, so the default path and its random init are unchanged.
        self.xy_token = xy_token

        self.poi_table = poi_table  # not a parameter or buffer: kept out of the checkpoints; None = no POI token
        self.user = user
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.focal_weight = 1 - ce_weight if focal_weight is None else focal_weight
        self.token_embed = token_embed
        self.pool_mask = pool_mask

        # Spatial Embedding
        self.spatial_embed_layer = nn.Sequential(
            nn.Linear(2, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, d_model)
        )

        # Temporal Embedding
        self.temporal_embed_modules = nn.ModuleList([FourierEncode(embed_size) for _ in range(4)])
        self.temporal_embed_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(embed_size * 4, d_model)
        )

        # POI Embedding
        if poi_table is not None:
            self.poi_embed_layer = nn.Sequential(
                nn.LayerNorm(poi_table.shape[1]),
                nn.Linear(poi_table.shape[1], d_model)
            )

        if token_embed:
            self.token_embed_layer = nn.Sequential(
                nn.Embedding(6, embed_size, padding_idx=PAD_TOKEN),
                nn.LayerNorm(embed_size),
                nn.Linear(embed_size, d_model)
            )
        self.pos_encode_layer = PositionalEncode(d_model)

        # Modal Mixer & Sequence Model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.modal_mixer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.causal = True if rope is None else rope.get('causal', True)
        # seq_arch 'bilstm' (2026-11-07, E3 same-input control, R1.2 / R1.5): every input, the mixer, positions and the
        # pooling stay; only the rotary encoder becomes a BiLSTM (rope_layer layers, d_model // 2 per direction, packed).
        self.seq_arch = seq_arch
        if seq_arch == 'bilstm':
            self.seq_model = nn.LSTM(d_model, d_model // 2, num_layers=rope_layer, batch_first=True, bidirectional=True,
                                     dropout=seq_dropout if rope_layer > 1 else 0.0)
        else:
            self.seq_model = RoPE_Encoder(d_model, layers=rope_layer, rope=rope, dropout=seq_dropout)

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.supcon_loss = SupConLoss(temperature=supcon_temp)
        self.user_pred_layers = nn.Sequential(nn.Linear(d_model, self.user))
        # Identity at 0: the base draws no extra random numbers. On the classifier input only; SupCon sees the undropped vector.
        self.head_drop = nn.Dropout(head_dropout) if head_dropout else nn.Identity()

        # Built after every module above, so their initial weights do not change; off = no modules, no random draws.
        self.cell_embed = None
        if cell_embed:  # one embedding table per cell level, row 0 = unknown cell (stays zero)
            dim = cell_embed.get('dim', 64)
            self.cell_embed = nn.ModuleList(nn.Embedding(r, dim, padding_idx=0) for r in cell_embed['rows'])
            with torch.no_grad():
                for emb in self.cell_embed:
                    emb.weight.normal_(std=d_model ** -0.5)
                    emb.weight[0] = 0
            self.cell_proj = nn.Linear(dim, d_model) if dim != d_model else nn.Identity()
            self.cell_drop = cell_embed.get('drop_fine', 0.0)
        self.fourier_xy = fourier_xy
        if fourier_xy:  # fixed (or learned) sin/cos of xy at wavelengths lmin..lmax metres; replaced last
            n = fourier_xy['n']
            wl = torch.logspace(math.log10(fourier_xy['lmin']), math.log10(fourier_xy['lmax']), n, dtype=torch.float64)
            log_w = torch.log(2 * math.pi * fourier_xy['scale'] / wl).float()  # per scaled-coordinate unit
            if fourier_xy.get('learn', False):
                self.xy_log_omega = nn.Parameter(log_w)
            else:
                self.register_buffer('xy_log_omega', log_w)
            self.spatial_embed_layer[0] = nn.Linear(4 * n, embed_size)
        # zeros: no random draws, so the other weights do not change
        self.poi_gate = nn.Parameter(torch.zeros(1)) if poi_gate and poi_table is not None else None
        self.poi_gate_cell = nn.Parameter(torch.zeros(2)) if poi_gate_cell else None  # [known, unknown finest cell]

    def forward(self, feats, poi_idx, pad):
        causal_mask = gen_causal_mask(feats.size(1)).to(feats.device) if self.causal else None
        modal_h = self.cal_modal_h(feats, poi_idx, pad)
        if self.seq_arch == 'bilstm':  # packed: padding never enters the recurrence
            packed = pack_padded_sequence(modal_h, (~pad).sum(1).cpu(), batch_first=True, enforce_sorted=False)
            return pad_packed_sequence(self.seq_model(packed)[0], batch_first=True, total_length=modal_h.size(1))[0]
        return self.seq_model(modal_h, feats, mask=causal_mask, src_key_padding_mask=pad)

    def cal_modal_h(self, feats, poi_idx, pad):
        B, L, _ = feats.shape
        pad_e = pad.unsqueeze(-1)

        xy = feats[..., 0:2]
        if self.fourier_xy:
            a = xy.float()[..., None] * self.xy_log_omega.exp()  # (B, L, 2, n)
            xy = torch.cat([a.sin(), a.cos()], -1).flatten(-2)  # (B, L, 4n)
        spatial_e = self.spatial_embed_layer(xy).masked_fill(pad_e, 0)
        if self.cell_embed is not None:
            poi_idx, cell_idx = poi_idx[..., 0], poi_idx[..., 1:]
            cell_unk = cell_idx[..., :1] == 0  # finest cell unknown, taken before drop_fine (B, L, 1)
            if self.training and self.cell_drop > 0:  # drop fine levels per point; the coarsest is kept
                drop = torch.rand(cell_idx[..., :-1].shape, device=cell_idx.device) < self.cell_drop
                cell_idx = torch.cat([cell_idx[..., :-1].masked_fill(drop, 0), cell_idx[..., -1:]], -1)
            cell_e = sum(emb(cell_idx[..., k]) for k, emb in enumerate(self.cell_embed))
            cell_e = self.cell_proj(cell_e).masked_fill(pad_e, 0)
        if self.poi_table is not None:
            poi_e = self.poi_embed_layer(self.poi_table[poi_idx.to(self.poi_table.device)].to(poi_idx.device).float()).masked_fill(pad_e, 0)
            if self.training and self.poi_drop > 0:  # drop the POI token per point
                poi_e = poi_e.masked_fill(torch.rand(B, L, 1, device=poi_e.device) < self.poi_drop, 0)
            if self.poi_gate is not None:
                poi_e = poi_e * self.poi_gate
            if self.poi_gate_cell is not None:
                poi_e = poi_e * torch.where(cell_unk, self.poi_gate_cell[1], self.poi_gate_cell[0])
        temporal_e = torch.cat([module(feats[..., 2 + i]) for i, module in enumerate(self.temporal_embed_modules)], -1)
        temporal_e = self.temporal_embed_layer(temporal_e).masked_fill(pad_e, 0)

        if self.token_embed:
            token_e = self.token_embed_layer(pad.long() * PAD_TOKEN)  # token 0 on points
            spatial_e = spatial_e + token_e
            if self.poi_table is not None:
                poi_e = poi_e + token_e
            temporal_e = temporal_e + token_e
            if self.cell_embed is not None:
                cell_e = cell_e + token_e

        tokens = (([spatial_e] if self.xy_token else []) + [temporal_e] + ([poi_e] if self.poi_table is not None else [])
                  + ([cell_e] if self.cell_embed is not None else []))
        modal_e = rearrange(torch.stack(tokens, 2), 'B L F E -> (B L) F E')
        modal_h = rearrange(self.modal_mixer(modal_e), '(B L) F E -> B L F E', B=B).mean(axis=2)

        positions = torch.arange(L, device=feats.device).expand(B, L)
        modal_h = modal_h + self.pos_encode_layer(positions)
        if self.token_embed:
            modal_h = modal_h + self.pos_encode_layer(torch.zeros_like(positions))
        return modal_h

    def pred(self, mem_seq, pad):
        attn_weights = self.attention_pool(mem_seq)  # (B, L, 1)
        if self.pool_mask:
            attn_weights = attn_weights.masked_fill(pad.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled_embedding = torch.sum(mem_seq * attn_weights, dim=1)  # (B, d_model)
        return self.user_pred_layers(self.head_drop(pooled_embedding)), pooled_embedding

    def encode(self, feats, poi_idx, pad):
        return self.pred(self.forward(feats, poi_idx, pad), pad)


def gen_causal_mask(seq_len, include_self=True):
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()
