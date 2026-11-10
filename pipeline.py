import copy
import os
from itertools import combinations
from time import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef


def param_groups(model, weight_decay, wd_groups):
    """F13: no weight decay on biases, norms, Fourier omega/bias and the rotary frequencies
    (Wr, Wa; ST-RoPE log-omega and directions), and the cell embeddings."""
    if not wd_groups:
        return [{'params': model.parameters(), 'weight_decay': weight_decay}]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        rotary = name.endswith(('.Wr.weight', '.Wa.weight', '_log_omega', '_alpha')) or 'cell_embed' in name
        (no_decay if p.ndim < 2 or rotary else decay).append(p)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


def train_user_model(model, train_dataloader, val_dataloader, device, num_epoch, lr, patience, factor, weight_decay,
                     ckpt_path, best_path, run=None, early_stop=15, clip=1.0, wd_groups=True, resume=False,
                     epoch_hook=None):
    """Selects on validation ACC@1. Writes best weights to best_path and a resume checkpoint to ckpt_path.
    Returns (per-epoch log, best state dict); the model is left holding the best weights.
    epoch_hook(model, epoch): called after each epoch's training pass, before validation (C4)."""
    optimizer = torch.optim.AdamW(param_groups(model, weight_decay, wd_groups), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)

    log, best_acc, bad, start_epoch, best_state = [], -1.0, 0, 0, None
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        log, best_acc, bad, start_epoch = ck['log'], ck['best_acc'], ck['bad'], ck['epoch'] + 1
        best_state = torch.load(best_path, map_location=device)
        print(f"Resuming from epoch {start_epoch + 1}, best val ACC@1 {best_acc * 100:.2f}")

    for epoch in range(start_epoch, num_epoch):
        model.train()
        t0, losses = time(), []
        for batch in train_dataloader:
            feats, poi, pad, users = [t.to(device) for t in batch]
            optimizer.zero_grad()
            loss, _ = model.user_loss(feats, poi, pad, users)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(loss.item())
        epoch_time = time() - t0
        if epoch_hook is not None:
            epoch_hook(model, epoch + 1)

        val_metrics, val_loss, _ = evaluate(model, device, val_dataloader)
        acc = val_metrics['ACC@1']
        scheduler.step(acc)

        entry = {'epoch': epoch + 1, 'time': epoch_time, 'lr': optimizer.param_groups[0]['lr'],
                 'Train_Loss': float(np.mean(losses)), 'Val_loss': val_loss,
                 **{f'val_{k}': round(v * 100, 2) for k, v in val_metrics.items()}}
        log.append(entry)
        if run is not None:
            run.log(entry)

        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
        else:
            bad += 1
        print(f"Epoch {epoch + 1} | {epoch_time:.0f}s | loss {entry['Train_Loss']:.4f} | val loss {val_loss:.4f} "
              f"| val ACC@1 {acc * 100:.2f} (best {best_acc * 100:.2f}, {bad} bad) | val F1 {val_metrics['Macro-F1'] * 100:.2f} | lr {entry['lr']:.2e}", flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'log': log, 'best_acc': best_acc,
                    'bad': bad, 'epoch': epoch}, ckpt_path)
        if bad >= early_stop:
            print(f"Early stop: no val ACC@1 gain for {early_stop} epochs")
            break

    model.load_state_dict(best_state)
    return log, best_state


def grad_share(model, loader, device, clip, seed, path, epoch):
    """C4 (E5): per batch of a fixed loader (its generator re-seeded to `seed`, so the same batches every time), the
    gradients of the unweighted CE / focal / SupCon terms w.r.t. the shared parameters (all but user_pred_layers) and
    the pooled vector. Appends raw and weighted norms, pairwise cosines and whether the clip fires on the weighted sum
    (all parameters, as clip_grad_norm_) to path. Never touches .grad or the optimizer; RNG state restored."""
    names = ('ce', 'focal', 'supcon')
    w = dict(zip(names, (model.ce_weight, model.focal_weight, model.supcon_weight)))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = torch.cat([torch.full((p.numel(),), n.startswith('user_pred_layers'), dtype=torch.bool, device=p.device)
                      for n, p in params])
    rows = []
    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        loader.generator.manual_seed(seed)
        for b, batch in enumerate(loader):
            feats, poi, pad, users = [t.to(device) for t in batch]
            terms, _, pooled = model.loss_terms(feats, poi, pad, users)
            g = {}
            for k, t in zip(names, terms):
                gs = torch.autograd.grad(t, [p for _, p in params] + [pooled], retain_graph=True, allow_unused=True)
                flat = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, (_, p) in zip(gs, params)])
                g[k] = (flat, gs[-1].flatten())
            row = {'epoch': epoch, 'batch': b, 'n': len(users)}
            for k in names:
                for i, part in enumerate(('shared', 'pooled')):
                    v = g[k][0][~head] if i == 0 else g[k][1]
                    row[f'{k}_{part}_norm'] = v.norm().item()
                    row[f'{k}_{part}_wnorm'] = w[k] * row[f'{k}_{part}_norm']
            for a, c in combinations(names, 2):
                row[f'cos_{a}_{c}_shared'] = torch.cosine_similarity(g[a][0][~head], g[c][0][~head], 0).item()
                row[f'cos_{a}_{c}_pooled'] = torch.cosine_similarity(g[a][1], g[c][1], 0).item()
            row['total_norm'] = sum(w[k] * g[k][0] for k in names).norm().item()
            row['clip_fires'] = bool(clip) and row['total_norm'] > clip
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    m = df.mean(numeric_only=True)
    print(f"Grad share epoch {epoch}: shared |g| weighted ce {m['ce_shared_wnorm']:.3g} focal {m['focal_shared_wnorm']:.3g} "
          f"supcon {m['supcon_shared_wnorm']:.3g} | cos ce-focal {m['cos_ce_focal_shared']:.2f} ce-supcon "
          f"{m['cos_ce_supcon_shared']:.2f} focal-supcon {m['cos_focal_supcon_shared']:.2f} | clip fires "
          f"{df['clip_fires'].mean() * 100:.0f}%", flush=True)


@torch.no_grad()
def evaluate(model, device, dataloader):
    """Whole-set metrics in one forward pass per batch. Returns (metrics, mean batch loss, (labels, logits))."""
    model.eval()
    losses, labels, logits = [], [], []
    for batch in dataloader:
        feats, poi, pad, users = [t.to(device) for t in batch]
        loss, out = model.user_loss(feats, poi, pad, users)
        losses.append(loss.item())
        labels.append(users.cpu())
        logits.append(out.float().cpu())
    labels, logits = torch.cat(labels).numpy(), torch.cat(logits).numpy()
    return compute_metrics(labels, logits), float(np.mean(losses)), (labels, logits)


def save_preds(path, split, val_out, test_out):
    """Per split: labels, fp32 logits, trajectory ids, top-5 (best first, [:, 0] == argmax).
    y_true and top5 repeat the test split for code/competing/score_pred.py."""
    z = {}
    for k, (labels, logits) in (('val', val_out), ('test', test_out)):
        logits = np.asarray(logits, np.float32)
        z.update({f'{k}_labels': np.asarray(labels, np.int64), f'{k}_logits': logits, f'{k}_traj': np.array(split[k]),
                  f'{k}_top5': np.argsort(-logits, axis=1, kind='stable')[:, :5]})
    z['y_true'], z['top5'] = z['test_labels'], z['test_top5']
    np.savez_compressed(path, **z)


def compute_metrics(labels, logits):
    """F12: every metric over the whole set; macro averages over all classes (labels=0..C-1)."""
    n_classes = logits.shape[1]
    preds = logits.argmax(1)
    top5 = np.argpartition(-logits, min(5, n_classes) - 1, axis=1)[:, :5]
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=np.arange(n_classes),
                                                  average='macro', zero_division=0)
    return {
        'ACC@1': float((preds == labels).mean()),
        'ACC@5': float((top5 == labels[:, None]).any(1).mean()),
        'Macro-P': float(p),
        'Macro-R': float(r),
        'Macro-F1': float(f1),
        'MCC': float(matthews_corrcoef(labels, preds)),
    }
