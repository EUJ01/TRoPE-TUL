import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import warnings

import utils
from models.model import TRoPETUL
from data import TRoPETULDataset, fetch_task_padder, X_COL, Y_COL, coord_transform_GPS_UTM
from pipeline import train_user_model, test_user_model

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate TRoPETUL")
    
    # Environment & System arguments
    parser.add_argument("--config", type=str, default="settings/local_test.json", help="Path to the JSON config file")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA Device ID(s) to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Geographic & Transformation arguments
    parser.add_argument("--scale", type=int, default=4000, help="Scale factor for UTM coordinates")
    parser.add_argument("--utm_region", type=int, default=54, help="UTM region for GPS transformation")
    
    # WandB arguments
    parser.add_argument("--wandb_entity", type=str, default="SP_001", help="WandB entity name")
    parser.add_argument("--wandb_project", type=str, default="TRoPE-TUL", help="WandB project name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- System Setup & Reproducibility ---
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # SDP optimization settings
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # --- Directories & Caching ---
    SETTINGS_CACHE_DIR = os.environ.get('SETTINGS_CACHE_DIR', os.path.join('settings', 'cache'))
    MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
    LOG_SAVE_DIR = os.environ.get('LOG_SAVE_DIR', 'logs')
    utils.create_if_noexists(SETTINGS_CACHE_DIR)
    utils.create_if_noexists(MODEL_CACHE_DIR)
    utils.create_if_noexists(LOG_SAVE_DIR)

    # --- Load Configuration ---
    datetime_key = utils.get_datetime_key()
    with open(args.config, 'r') as fp:
        setting = json.load(fp)[0]
    
    with open(os.path.join(SETTINGS_CACHE_DIR, f'{datetime_key}.json'), 'w') as fp:
        json.dump(setting, fp)

    SAVE_NAME = setting["save_name"]
    print(f"Dataset: {setting['dataset']['train_traj_df']}")

    # --- Load Trajectory Data ---
    train_traj_df = pd.read_hdf(setting['dataset']['train_traj_df'], key='trips')
    user_count = len(train_traj_df['user_id'].unique()) #type: ignore
    
    # Update settings with derived data
    setting["finetune"]["padder"]["params"]["num_users"] = user_count
    alpha = setting["finetune"]["config"]["focal_alpha"]
    gamma = setting["finetune"]["config"]["focal_gamma"]
    ce_weight = setting["finetune"]["config"]["ce_weight"]
    supcon_weight = setting["finetune"]["config"]["supcon_weight"]

    # --- Dataset Creation & Preprocessing ---
    # print(f"Before dataset creation, NaN rows: {len(train_traj_df[train_traj_df['lat'].isna() | train_traj_df['lng'].isna()])}")
    
    dataset = TRoPETULDataset(traj_df=train_traj_df, UTM_region=args.utm_region, scale=args.scale)
    
    # print(f"After dataset creation, NaN rows: {len(train_traj_df[train_traj_df['lat'].isna() | train_traj_df['lng'].isna()])}")

    # --- Load POI Data ---
    poi_df = pd.read_hdf(setting['dataset']['poi_df'], key='pois')
    poi_embed = torch.from_numpy(np.load(setting['dataset']['poi_embed'])).float().to(device)

    poi_coors = poi_df[[X_COL, Y_COL]].to_numpy().copy() #type: ignore
    poi_coors = (coord_transform_GPS_UTM(poi_coors, args.utm_region) - dataset.spatial_middle_coord) / args.scale
    poi_coors = torch.tensor(poi_coors).float().to(device)

    # --- Initialize Model ---
    model = TRoPETUL(
        poi_embed=poi_embed, 
        poi_coors=poi_coors, 
        UTM_region=args.utm_region,
        spatial_middle_coord=dataset.spatial_middle_coord, 
        scale=args.scale, 
        **setting['model'],
        user=user_count,
        alpha=alpha,
        gamma=gamma,
        ce_weight=ce_weight,
        supcon_weight=supcon_weight).to(device)
                    
    total_params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)

    # Data Summaries for Logging
    fsquare = {
        "min_traj_len_points": 5,
        "min_traj_per_user": 30,
        "avg_traj_per_user": 83,
    }
    traj_count = len(dataset.traj_df['traj_id'].unique()) #type: ignore

    data_summary = {
        "users": user_count,
        "total_traj": traj_count,
        "total_params": total_params_trainable,
        "Data_Filtering": fsquare,
    }

    print("\n--- Data Summary ---")
    for key, value in data_summary.items():
        print(f"{key} : {value}")
    print("--------------------\n")
        
    # --- Splitting & DataLoaders ---
    train_dataset, val_test_dataset = utils.stratify_dataset(dataset, test_size=0.2, random_seed=args.seed)
    val_dataset, test_dataset = utils.stratify_dataset(val_test_dataset, test_size=0.5, random_seed=args.seed)

    print(f"Train split: {len(train_dataset)}")
    print(f"Val split: {len(val_dataset)}")
    print(f"Test split: {len(test_dataset)}\n")

    downstreamtask = setting['finetune']['padder']['name']
    padder = fetch_task_padder(padder_name=downstreamtask, padder_params=setting['finetune']['padder']['params'])

    loader_kwargs = setting['finetune']['dataloader']
    train_dataloader = DataLoader(train_dataset, collate_fn=padder, **loader_kwargs)
    val_dataloader = DataLoader(val_dataset, collate_fn=padder, **loader_kwargs)
    test_dataloader = DataLoader(test_dataset, collate_fn=padder, **loader_kwargs)

    # --- Model Loading (Optional) ---
    file_path = os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}.{downstreamtask}")
    if os.path.exists(file_path):
        print(f"Loading existing model checkpoint from {file_path}")
        model.load_state_dict(torch.load(file_path, map_location=device))
    else:
        print("No existing model checkpoint found. Starting fresh.")

    # --- Training ---
    # Note: Ensure your `train_user_model` has wandb setup modified slightly if you want to pass args.wandb_entity 
    # and args.wandb_project natively, or you can rely on your existing JSON logic.
    train_log, saved_model_state_dict = train_user_model(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        device=device, 
        **setting['finetune']['config'],
        data_summary=data_summary,
        MODEL_CACHE_DIR=MODEL_CACHE_DIR,
        SAVE_NAME=SAVE_NAME
    )

    if setting['finetune'].get('save', False):
        torch.save(saved_model_state_dict, os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}.{downstreamtask}'))
        
        log_dir = os.path.join(LOG_SAVE_DIR, SAVE_NAME)
        utils.create_if_noexists(log_dir)
        log_path = os.path.join(log_dir, f'{SAVE_NAME}_{downstreamtask}.csv')
        
        file_exists = os.path.exists(log_path)
        train_log.to_csv(log_path, mode='a', header=not file_exists, index=False)

    # --- Testing ---
    print("\n--- Commencing Final Evaluation ---")
    metrics, _, conf_mat, pr_curves, avg_precisions, all_probs = test_user_model(
        model=model, 
        dataloader=test_dataloader, 
        device=device
    )
    
    for key, value in metrics.items():
        print(f"{key}: {round(value * 100, 2)}%,")

    # Save final test metrics
    df = pd.DataFrame([{
        "Model": f"{SAVE_NAME}",
        **{key: round(value * 100, 2) for key, value in metrics.items()}
    }])

    csv_path = os.path.join(LOG_SAVE_DIR, "test.csv")
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # --- Visualizations ---
    output_folder = os.path.join(LOG_SAVE_DIR, SAVE_NAME)
    utils.create_if_noexists(output_folder)
    
    utils.save_confusion_matrix(conf_mat, filename=os.path.join(output_folder, "confusion_matrix.jpg"))
    utils.save_all_pr_curves(pr_curves, avg_precisions, filename=os.path.join(output_folder, "pr_curve.jpg"))
    utils.save_confidence_histogram(all_probs, save_folder=output_folder, filename="confidence_histogram.png", bins=50)

if __name__ == "__main__":
    main()