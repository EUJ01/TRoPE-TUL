import os
import json
from time import time
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm
import wandb
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score, 
    precision_recall_fscore_support, 
    matthews_corrcoef
)

def train_user_model(model, train_dataloader, val_dataloader, device, num_epoch, lr, patience, factor, weight_decay, focal_alpha, focal_gamma, ce_weight, supcon_weight, data_summary, MODEL_CACHE_DIR, SAVE_NAME):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    log = []
    saved_model_state_dict = None
    best_loss = float('inf')
    start_epoch = 0
    epochs_no_improve = 0
    
    # --- Checkpoint Resumption ---
    checkpoint_path = os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}.tul")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"📂 Resuming training from epoch {start_epoch}, best val_loss = {best_loss:.4f}")
    
    # --- WandB Initialization ---
    try:
        with open(os.path.join('settings', 'local_test.json'), 'r') as fp:
            config = json.load(fp)[0]
    except (FileNotFoundError, IndexError):
        config = {"model": {}, "finetune": {"dataloader": {}, "config": {}}, "save_name": SAVE_NAME}

    run = wandb.init(
        # entity="SP_001",
        project="TRoPE-TUL",
        config={
            "summary": data_summary,
            "model": config.get("model", {}),
            "dataloader": config.get("finetune", {}).get("dataloader", {}),
            "config": config.get("finetune", {}).get("config", {}),
        },
        id=config.get("save_name", SAVE_NAME),
        resume="allow"
    )
    print(f"🌐 WandB run id: {run.id}")

    # --- Training Loop ---
    bar_desc = 'Training, avg loss: %.7f'
    with trange(start_epoch, num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            
            loss_values = []
            epoch_time = 0
            model.train()
            
            for batch in tqdm(train_dataloader, desc='--> Training', leave=False):
                input_tensor, output_tensor, pos_tensor = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                s_time = time()
                
                loss = model.user_loss(input_tensor, output_tensor, pos_tensor)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_time += (time() - s_time)
                loss_values.append(loss.item())
            
            loss_epoch = np.mean(loss_values)
            bar.set_description(bar_desc % loss_epoch)
            
            # --- Validation ---
            val_metrics, val_loss, _, _, _, _ = test_user_model(model, device, val_dataloader)
            scheduler.step(val_loss)

            print(f"\nEpoch {epoch_i + 1} | Val Loss: {val_loss:.6f}")
            for key, value in val_metrics.items():
                print(f"  - {key}: {value * 100:.2f}%")
            
            # --- Logging ---
            log_entry = {
                'epoch': epoch_i + 1, 
                'time': epoch_time, 
                'Train_Loss': loss_epoch, 
                'Val_loss': val_loss
            }
            log_entry.update({k: round(v * 100, 2) for k, v in val_metrics.items()})
            log.append(log_entry)
            run.log(log_entry)

            # --- Model Saving & Early Stopping ---
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                saved_model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch_i,
                    'model_state_dict': saved_model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }, checkpoint_path)
                print("🌟 New best model saved!")
            else:
                epochs_no_improve += 1
                print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= 10:
                print("\n🛑 Early stopping triggered. Training stopped.\n")
                break

    # Wrap up logging
    log_df = pd.DataFrame(log).set_index('epoch')
    
    if saved_model_state_dict is None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(saved_model_state_dict)
    
    return log_df, saved_model_state_dict


@torch.no_grad()
def test_user_model(model, device, dataloader):
    """Evaluates the model and computes global metrics directly to save redundant forward passes."""
    model.eval()
    
    loss_values = []
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in tqdm(dataloader, desc='Testing/Validating', leave=False):
        input_tensor, output_tensor, pos_tensor = [t.to(device) for t in batch]

        # 1. Compute Loss
        loss = model.user_loss(input_tensor, output_tensor, pos_tensor)
        loss_values.append(loss.item())

        # 2. Extract Logits & Probs
        _, mem_seq = model(input_tensor, pos_tensor[:, :input_tensor.size(1)])
        logits = model.pred(mem_seq) 
        probs = torch.softmax(logits, dim=1)

        # 3. Store Batch Results
        all_labels.extend(torch.argmax(output_tensor, dim=1).cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # --- Compute Global Metrics ---
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    loss_epoch = np.mean(loss_values)
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Standard Metrics
    acc_1 = accuracy_score(all_labels, all_preds)
    
    # ACC@5 logic:
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    acc_5 = np.mean([1 if all_labels[i] in top5_preds[i] else 0 for i in range(len(all_labels))])

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0) #type: ignore
    mcc = matthews_corrcoef(all_labels, all_preds)

    total_metrics = {
        'ACC@1': acc_1,
        'ACC@5': acc_5,
        'Macro-P': precision,
        'Macro-R': recall,
        'Macro-F1': f1,
        'MCC': mcc,
    }

    # PR Curves & Average Precision (One-vs-Rest)
    n_classes = all_probs.shape[1]
    pr_curves, avg_precisions = {}, {}
    
    for i in range(n_classes):
        y_true_bin = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        # Handle edge cases where a class is entirely missing from the validation split
        if y_true_bin.sum() > 0:
            prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
            ap = average_precision_score(y_true_bin, y_score)
        else:
            prec, rec, ap = np.array([0.0]), np.array([0.0]), 0.0
            
        pr_curves[i] = (prec, rec)
        avg_precisions[i] = ap

    return total_metrics, loss_epoch, conf_mat, pr_curves, avg_precisions, all_probs


def pad_batch_arrays(arrs):
    """Pad a batch of arrays with representing feature sequences of different lengths."""
    max_len = max(a.shape[1] for a in arrs)
    arrs = [
        np.concatenate([a, np.repeat(a[:, -1:], repeats=max_len-a.shape[1], axis=1)], axis=1)
        for a in arrs
    ]
    return np.concatenate(arrs, 0)