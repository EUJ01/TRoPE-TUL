import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from torchvision.ops import sigmoid_focal_loss

# Ensure these are correctly importable from your project structure
from models.encode import PositionalEncode, FourierEncode, RoPE_Encoder
from data import FEATURE_TOKEN, PAD_TOKEN, ST_MAP

S_COLS = ST_MAP['spatial']
T_COLS = ST_MAP['temporal']

def load_transfer_feature(model, UTM_region, spatial_middle_coord, poi_embed, poi_coors):
    model.UTM_region = UTM_region
    model.spatial_middle_coord = spatial_middle_coord
    model.poi_embed_mat = poi_embed
    model.poi_coors = poi_coors
    return model

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

class TRoPETUL(nn.Module):
    def __init__(self, embed_size, d_model, poi_embed, poi_coors, rope_layer, UTM_region, spatial_middle_coord, scale, user, alpha, gamma, ce_weight, supcon_weight):
        super().__init__()

        self.poi_coors = poi_coors
        self.UTM_region = UTM_region
        self.spatial_middle_coord = spatial_middle_coord
        self.scale = scale
        self.user = user
        
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        
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
        self.poi_embed_mat = poi_embed
        self.poi_embed_layer = nn.Sequential(
            nn.LayerNorm(poi_embed.shape[1]), 
            nn.Linear(poi_embed.shape[1], d_model)
        )

        # Token & Positional Encoding
        self.token_embed_layer = nn.Sequential(
            nn.Embedding(6, embed_size, padding_idx=5), 
            nn.LayerNorm(embed_size), 
            nn.Linear(embed_size, d_model)
        )
        self.pos_encode_layer = PositionalEncode(d_model)

        # Modal Mixer & Sequence Model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True
        )
        self.modal_mixer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.seq_model = RoPE_Encoder(d_model, layers=rope_layer)

        # --- MODIFICATIONS: Attention Pooling & Contrastive Loss ---
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.supcon_loss = SupConLoss(temperature=0.1)
        self.user_pred_layers = nn.Sequential(nn.Linear(d_model, self.user))

    def forward(self, input_seq, positions):
        L = input_seq.size(1)

        token = input_seq[..., [S_COLS[0], T_COLS[0]], 1].long()
        spatial = input_seq[:, :, S_COLS, 0]
        temporal = input_seq[:, :, T_COLS, 0] 
        temporal_token = tokenize_timestamp(temporal)

        batch_mask = token[..., 0] == PAD_TOKEN
        causal_mask = gen_causal_mask(L).to(input_seq.device)

        modal_h, norm_coord = self.cal_modal_h(spatial, temporal_token, token, positions)
        mem_seq = self.seq_model(modal_h, norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask)
        
        # mem_seq = modal_h #ablation without STRPE
     
        return modal_h, mem_seq

    def cal_modal_h(self, spatial, temporal_token, token, positions):
        B, Len, _ = spatial.shape
        norm_coord = spatial

        token_e = self.token_embed_layer(token)  # (B, L, F, E)
        feature_e_mask = ~torch.isin(token, torch.tensor(FEATURE_TOKEN).to(token.device))  # (B, L, 2)

        # Spatial
        spatial_e = self.spatial_embed_layer(norm_coord)
        spatial_e.masked_fill_(feature_e_mask[..., 0].unsqueeze(-1), 0)
        
        # POI Indexing logic
        from data import TULPadder
        if hasattr(TULPadder, '_current_row_indices') and TULPadder._current_row_indices is not None:
            row_indices = TULPadder._current_row_indices.to(spatial.device)
            if row_indices.shape[0] == B and row_indices.shape[1] >= Len:
                valid_mask = row_indices >= 0
                row_indices = torch.where(valid_mask, row_indices, torch.zeros_like(row_indices))
                row_indices = row_indices[:, :Len]
                poi_e = self.poi_embed_layer(self.poi_embed_mat[row_indices])
            else:
                poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) - spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1)
                poi_e = self.poi_embed_layer(self.poi_embed_mat[poi])
        else:
            poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) - spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1)
            poi_e = self.poi_embed_layer(self.poi_embed_mat[poi])
        
        poi_e.masked_fill_(feature_e_mask[..., 0].unsqueeze(-1), 0)

        # Temporal
        temporal_e = torch.cat([module(temporal_token[..., i]) for i, module in enumerate(self.temporal_embed_modules)], -1)
        temporal_e = self.temporal_embed_layer(temporal_e)
        temporal_e.masked_fill_(feature_e_mask[..., 1].unsqueeze(-1), 0)
        
        #ablation no token
        spatial_e += token_e[:, :, 0]
        poi_e += token_e[:, :, 0]
        temporal_e += token_e[:, :, 1]

        # Clean Stack (No CLS hacking)
        modal_e = rearrange(torch.stack([spatial_e, temporal_e, poi_e], 2), 'B L F E -> (B L) F E')
        # modal_e = rearrange(torch.stack([spatial_e, temporal_e], 2), 'B L F E -> (B L) F E') #ablation no POI embedding
        modal_h = rearrange(self.modal_mixer(modal_e), '(B L) F E -> B L F E', B=B).mean(axis=2)

        # Positional Encoding
        pos_encoding = self.pos_encode_layer(positions[..., 0]) + self.pos_encode_layer(positions[..., 1])
        modal_h += pos_encoding
        return modal_h, norm_coord

    def pred(self, mem_seq, return_features=False):
        # Attention Pooling instead of static averaging or CLS token extraction
        attn_weights = self.attention_pool(mem_seq) # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum of the sequence
        pooled_embedding = torch.sum(mem_seq * attn_weights, dim=1) # (B, d_model)
        
        pred_user = self.user_pred_layers(pooled_embedding)
        
        if return_features:
            return pred_user, pooled_embedding
        return pred_user
    
    def user_loss(self, input_seq, target_seq, positions):
        _, mem_seq = self.forward(input_seq, positions)
        
        # Fetch both logits for classification and features for contrastive loss
        pred_user, pooled_feats = self.pred(mem_seq, return_features=True)

        true_indices = torch.argmax(target_seq, dim=1)
        true_one_hot = F.one_hot(true_indices, num_classes=pred_user.size(1)).float()

        # 1. Cross Entropy Loss
        ce_loss = F.cross_entropy(pred_user, true_indices)

        # 2. Focal Loss
        focal_loss = sigmoid_focal_loss(
            inputs=pred_user,
            targets=true_one_hot,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean"
        )
        
        # 3. Supervised Contrastive Loss
        supcon_loss = self.supcon_loss(pooled_feats, true_indices)

        # Combine Losses (0.5 weight applied to SupCon, tune this based on results)
        total_loss = self.ce_weight * ce_loss + (1 - self.ce_weight) * focal_loss + (self.supcon_weight * supcon_loss)
        # total_loss = self.ce_weight * ce_loss + (1 - self.ce_weight) * focal_loss #ablation no supcon
        # total_loss = self.ce_weight * ce_loss  + (1 - self.ce_weight) * supcon_loss #ablation no focal
        # total_loss = ce_loss #ablation only ce
        
        return total_loss

    @torch.no_grad()
    def test_user(self, input_seq, target_seq, positions):
        B, L_in = input_seq.size(0), input_seq.size(1)
        
        _, mem_seq = self.forward(input_seq, positions[:, :L_in])
        pred_user = self.pred(mem_seq) 

        pred_indices = torch.argmax(pred_user, dim=1) #type: ignore
        true_indices = torch.argmax(target_seq, dim=1)

        acc_1 = accuracy_score(true_indices.cpu(), pred_indices.cpu())
        top_5_preds = torch.topk(pred_user, 5, dim=1).indices #type: ignore
        acc_5 = (top_5_preds.eq(true_indices.unsqueeze(1)).sum(dim=1) > 0).float().mean().item()
        precision, recall, f1, _ = precision_recall_fscore_support(true_indices.cpu(), pred_indices.cpu(), average='macro', zero_division=0) #type: ignore
        mcc = matthews_corrcoef(true_indices.cpu(), pred_indices.cpu())
        
        return {
            'ACC@1': acc_1,
            'ACC@5': acc_5,
            'Macro-P': precision,
            'Macro-R': recall,
            'Macro-F1': f1,
            'MCC' : mcc,
        }, true_indices.cpu(), pred_indices.cpu()

def gen_causal_mask(seq_len, include_self=True):
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()

def tokenize_timestamp(t):
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    return torch.stack([week, hour, minute, d_minute], -1)