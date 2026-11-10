import os
import json
import argparse
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import geo
import utils
from models.model import TRoPETUL
from models.baselines import ARCHS
from data import (MPerUserSampler, TrajDataset, cap_rows, collate, load_trips, make_split, project, time_features,
                  TRAJ_ID_COL, USER_COL, X_COL, Y_COL, T_COL)
from pipeline import train_user_model, evaluate, grad_share, save_preds

SPLIT_SEED = 42  # the split is frozen; --seed only changes training


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate TRoPETUL")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset paths in the config are relative to this")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        help="CUDA Device ID(s) to use (default: the caller's CUDA_VISIBLE_DEVICES, else 0)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed (init, shuffling)")
    parser.add_argument("--run_name", type=str, default=None, help="Default: save_name + datetime key")
    parser.add_argument("--resume", action="store_true", help="Resume --run_name from its checkpoint")
    parser.add_argument("--scale", type=int, default=4000, help="Scale factor for UTM coordinates (metres)")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="TRoPE-TUL")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    with open(args.config) as fp:
        setting = json.load(fp)
    ds = setting['dataset']
    data_path = lambda key: os.path.join(args.data_dir, ds[key])

    # --- Run directories: one per run, never overwritten ---
    if args.resume and not args.run_name:
        raise SystemExit("--resume needs --run_name")
    run_name = args.run_name or f"{setting['save_name']}_{utils.get_datetime_key()}"
    log_dir = os.path.join(os.environ.get('LOG_SAVE_DIR', 'logs'), run_name)
    model_dir = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
    if os.path.exists(log_dir) and not args.resume:
        raise SystemExit(f"{log_dir} exists; pick another --run_name")
    utils.create_if_noexists(log_dir)
    utils.create_if_noexists(model_dir)
    with open(os.path.join(log_dir, 'settings.json'), 'w') as fp:
        json.dump({'settings': setting, 'args': vars(args)}, fp, indent=2)

    # --- Trajectories and the frozen split ---
    df = load_trips(data_path('trips'))
    traj_ids = pd.unique(df[TRAJ_ID_COL])
    rows_by_traj = df.groupby(TRAJ_ID_COL, sort=False).indices
    traj_users = df[USER_COL].to_numpy()[[rows_by_traj[t][0] for t in traj_ids]]
    user_count = df[USER_COL].nunique()
    assert set(df[USER_COL].unique()) == set(range(user_count)), "user ids must be 0..C-1"

    split_path = data_path('split')
    if os.path.exists(split_path):
        with open(split_path) as fp:
            split = json.load(fp)
    else:
        split = make_split(traj_ids, traj_users, SPLIT_SEED)
        utils.create_if_noexists(os.path.dirname(split_path))
        with open(split_path, 'w') as fp:
            json.dump(split, fp)
        print(f"Wrote split {split_path}")
    assert sorted(split['train'] + split['val'] + split['test']) == sorted(traj_ids.tolist()), "split does not match the data"

    # --- Point features: coordinates centred on the train median (F1), local time (F7) ---
    ts = df[T_COL].to_numpy()
    if ds.get('max_len'):  # C5: first max_len points per trajectory, before the median, cells and datasets see them
        rows_by_traj = cap_rows(rows_by_traj, ts, ds['max_len'])
        print(f"max_len {ds['max_len']}: kept {sum(len(r) for r in rows_by_traj.values())} of {len(ts)} points")
    xy = project(df[X_COL].to_numpy(), df[Y_COL].to_numpy(), ds['utm_zone'], ds['bbox'], ds.get('proj'))
    train_rows = np.concatenate([rows_by_traj[t] for t in split['train']])
    xy_m = xy - np.median(xy[train_rows], 0)
    xy = xy_m / ds.get('scale', args.scale)
    ce = setting['model'].get('cell_embed')
    if ce:  # P2: grid-cell ids per level from training points (clamped points and unseen cells -> 0)
        assert all(a < b for a, b in zip(ce['levels'], ce['levels'][1:])), "cell_embed levels must ascend"
        ok = geo.in_bbox(df[X_COL].to_numpy(), df[Y_COL].to_numpy(), ds['bbox'])
        row_traj = pd.factorize(df[TRAJ_ID_COL])[0]
        cells, cell_rows = geo.cell_ids(xy_m, ok, train_rows, row_traj, ce['levels'],
                                        ce.get('max_rows', 65536), ce.get('min_traj', 2))
    point_feats = np.concatenate([xy, time_features(ts, ds['tz'])], 1).astype(np.float32)
    print(f"Scaled coordinates: max |x| {np.abs(xy[:, 0]).max():.1f}, max |y| {np.abs(xy[:, 1]).max():.1f}")

    # --- POI vectors: a table plus a row -> table index (F4, F9) ---
    # C1 model.poi false and the C2 controls: no table; column 0 stays as a placeholder so the cells keep columns 1+
    model_kw = dict(setting['model'])  # settings.json keeps the user's config; derived values go to the model only
    arch, use_poi = model_kw.pop('arch', 'trope'), model_kw.pop('poi', True)
    use_table = use_poi and arch == 'trope'
    poi_table = np.load(data_path('poi_embed'), mmap_mode='r') if use_table else None
    poi_index = np.load(data_path('poi_index')) if 'poi_index' in ds else np.arange(len(df))
    assert len(poi_index) == len(df) and (not use_table or poi_index.max() < len(poi_table)), "POI index does not match the data"
    if ce:
        poi_index = np.concatenate([poi_index[:, None], cells], 1)
    if use_table:
        poi_dtype = torch.float16 if ds.get('poi_fp16', False) else torch.float32
        # Copied in chunks: the GeoLife file (24 GB fp32) never sits whole in host RAM.
        # poi_on_cpu: the table stays in RAM and the model moves each batch's rows to the GPU (for small cards)
        table = torch.empty(poi_table.shape, dtype=poi_dtype, device='cpu' if ds.get('poi_on_cpu', False) else device)
        for s in range(0, len(poi_table), 200_000):
            table[s:s + 200_000] = torch.from_numpy(np.ascontiguousarray(poi_table[s:s + 200_000])).to(table.device, poi_dtype)
        poi_table = table

    datasets = {k: TrajDataset(point_feats, ts, df[USER_COL].to_numpy(), rows_by_traj, split[k], poi_index)
                for k in ('train', 'val', 'test')}
    del df, xy, xy_m, point_feats

    loader_kwargs = setting['finetune']['dataloader']
    cfg = dict(setting['finetune']['config'])
    g = torch.Generator()
    g.manual_seed(args.seed)
    # ponytail: no worker_init_fn, the loaders draw no random numbers in workers; add one with augmentation.
    sampler = cfg.pop('sampler', None)
    if sampler:  # C6: user-balanced batches, train loader only
        assert sampler['type'] == 'm_per_user', sampler
        bs = MPerUserSampler([u for _, _, u in datasets['train'].items], sampler['m'], loader_kwargs['batch_size'], g)
        train_dataloader = DataLoader(datasets['train'], collate_fn=collate, batch_sampler=bs, generator=g,
                                      **{k: v for k, v in loader_kwargs.items() if k != 'batch_size'})
    else:
        train_dataloader = DataLoader(datasets['train'], collate_fn=collate, shuffle=True, generator=g, **loader_kwargs)
    val_dataloader = DataLoader(datasets['val'], collate_fn=collate, **loader_kwargs)
    test_dataloader = DataLoader(datasets['test'], collate_fn=collate, **loader_kwargs)

    # --- Model ---
    loss_kw = dict(alpha=cfg.pop('focal_alpha'), gamma=cfg.pop('focal_gamma'),
                   ce_weight=cfg.pop('ce_weight'), supcon_weight=cfg.pop('supcon_weight'),
                   focal_weight=cfg.pop('focal_weight', None))
    if ce:
        model_kw['cell_embed'] = {**ce, 'rows': cell_rows}
    if model_kw.get('fourier_xy'):
        model_kw['fourier_xy'] = {**model_kw['fourier_xy'], 'scale': ds.get('scale', args.scale)}
    model = ({'trope': TRoPETUL, **ARCHS}[arch])(poi_table=poi_table, user=user_count, **model_kw, **loss_kw).to(device)
    cuda = torch.cuda.is_available()
    if cuda:  # C3: peak memory from here on (the POI table already counts)
        torch.cuda.reset_peak_memory_stats()

    data_summary = {
        "users": user_count,
        "total_traj": len(traj_ids),
        "points": len(ts),
        "split": {k: len(v) for k, v in split.items()},
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print(json.dumps(data_summary, indent=2))

    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name, id=run_name,
                     resume='allow' if args.resume else None,
                     config={'settings': setting, 'args': vars(args), 'data': data_summary})

    # --- Training (selection on validation ACC@1) ---
    every, hook = cfg.pop('grad_share_every', 0), None
    if every:  # C4: a separate loader with its own generator; grad_share re-seeds it, so every pass sees the same batches
        gs_loader = DataLoader(datasets['train'], collate_fn=collate, shuffle=True, generator=torch.Generator(), **loader_kwargs)
        gs_path = os.path.join(log_dir, 'grad_share.csv')

        def hook(m, epoch):
            if epoch % every == 0:
                grad_share(m, gs_loader, device, cfg.get('clip', 1.0), args.seed, gs_path, epoch)
    train_log, _ = train_user_model(
        model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device,
        ckpt_path=os.path.join(model_dir, f'{run_name}.ckpt'),
        best_path=os.path.join(model_dir, f'{run_name}.best.pt'),
        run=run, resume=args.resume, epoch_hook=hook, **cfg)
    pd.DataFrame(train_log).to_csv(os.path.join(log_dir, 'train_log.csv'), index=False)
    if cuda:
        print(f"Peak GPU memory (train) {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
        torch.cuda.reset_peak_memory_stats()

    # --- Validation again and test once, with the best validation weights; logits saved ---
    val_m, _, val_out = evaluate(model, device, val_dataloader)
    best = max(e['val_ACC@1'] for e in train_log)
    print(f"Val ACC@1 of the saved weights {val_m['ACC@1'] * 100:.2f} (best epoch {best:.2f})")
    if round(abs(round(val_m['ACC@1'] * 100, 2) - best), 2) > 0.01:
        print("WARNING: saved validation logits differ from the best epoch")
    # C3: the test pass is timed after the val pass warmed up; includes batch loading and the metrics (small)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    sync()
    t0 = perf_counter()
    metrics, _, (labels, logits) = evaluate(model, device, test_dataloader)
    sync()
    dt = perf_counter() - t0
    print(f"Inference test: {len(labels)} traj in {dt:.2f} s ({dt / len(labels) * 1e6:.1f} ms per 1000 traj)")
    if cuda:
        print(f"Peak GPU memory (test) {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
    print("\n--- Test (best validation checkpoint) ---")
    for key, value in metrics.items():
        print(f"{key}: {value * 100:.2f}%")
    pd.DataFrame([{"run": run_name, **{k: round(v * 100, 2) for k, v in metrics.items()}}]).to_csv(
        os.path.join(log_dir, 'test_metrics.csv'), index=False)
    save_preds(os.path.join(log_dir, f'{run_name}_preds.npz'), split, val_out, (labels, logits))
    run.summary.update({f'test_{k}': v for k, v in metrics.items()})
    run.finish()

    from sklearn.metrics import confusion_matrix
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    utils.save_confusion_matrix(confusion_matrix(labels, logits.argmax(1)), filename=os.path.join(log_dir, "confusion_matrix.jpg"))
    utils.save_all_pr_curves(*utils.compute_pr_curves(labels, probs), filename=os.path.join(log_dir, "pr_curve.jpg"))
    utils.save_confidence_histogram(probs, save_folder=log_dir, filename="confidence_histogram.png", bins=50)


if __name__ == "__main__":
    main()
