import os
import string
import random
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import Subset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)

def get_random_string(length):
    """ Generate a random uppercase string of fixed length. """
    letters = string.ascii_uppercase
    # random.choices is significantly faster than a loop with random.choice
    return ''.join(random.choices(letters, k=length))

def create_if_noexists(path):
    """ Safely create a directory if it doesn't exist. """
    os.makedirs(path, exist_ok=True)

def stratify_dataset(dataset, test_size, random_seed):
    """
    Splits a dataset into train and test subsets while stratifying by user_id.
    Optimized to prevent O(N^2) pandas dataframe lookups.
    """
    # If dataset is a Subset, unwrap it
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_dataset = dataset
        subset_indices = np.arange(len(dataset))

    traj_ids = base_dataset.traj_ids #type: ignore
    df = base_dataset.traj_df #type: ignore

    # Only use traj_ids inside this subset
    selected_traj_ids = traj_ids[subset_indices]

    # OPTIMIZATION: Create a fast O(1) lookup dictionary mapping traj_id -> user_id
    # This prevents scanning the entire dataframe repeatedly in a loop.
    user_mapping = df.drop_duplicates(subset=['traj_id']).set_index('traj_id')['user_id'].to_dict()

    user_array = np.array([user_mapping[tid] for tid in selected_traj_ids])
    indices = np.arange(len(selected_traj_ids))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=user_array,
        random_state=random_seed
    )

    # Map back to original indices
    train_indices = subset_indices[train_idx] #type: ignore
    test_indices = subset_indices[test_idx] #type: ignore

    train_dataset = Subset(base_dataset, train_indices) #type: ignore
    test_dataset = Subset(base_dataset, test_indices) #type: ignore

    return train_dataset, test_dataset

# --------------------------------------------------
# Visualization Functions
# --------------------------------------------------

def save_confusion_matrix(conf_mat, filename="confusion_matrix.jpg"):
    """ Save a dynamically scaled confusion matrix as a JPG. """
    num_classes = conf_mat.shape[0]
    
    # Dynamically scale figure size based on number of classes (min 10x10, max 30x30)
    fig_size = max(10, min(30, int(num_classes * 0.3)))
    
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        conf_mat, 
        annot=False, 
        cmap="Blues",
        xticklabels=range(1, num_classes + 1), #type: ignore
        yticklabels=range(1, num_classes + 1), #type: ignore
        cbar=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Classes 1–{num_classes})")
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def save_all_pr_curves(pr_curves, avg_precisions, filename="all_pr_curves.jpg"):
    """
    Save PR curves for ALL classes in one big figure with subplots.
    
    pr_curves: dict[class_id] -> (precision, recall)
    avg_precisions: dict[class_id] -> float
    """
    n_classes = len(pr_curves)
    
    # Auto grid size (square-like)
    n_cols = int(np.ceil(np.sqrt(n_classes)))
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()

    for i in range(n_classes):
        if i in pr_curves:  # Safety check
            precision, recall = pr_curves[i]
            ap = avg_precisions[i]

            axes[i].plot(recall, precision, lw=1.5)
            axes[i].set_title(f"Class {i+1}\nAP={ap:.2f}", fontsize=8)
            axes[i].set_xlabel("Recall", fontsize=6)
            axes[i].set_ylabel("Precision", fontsize=6)
            axes[i].tick_params(axis="both", labelsize=6)

    # Hide unused subplots if the grid isn't perfectly filled
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Explicitly pass fig to clear memory

def save_confidence_histogram(all_probs, save_folder="histograms", filename="confidence_histogram.png", bins=20):
    """
    Creates and saves a histogram of maximum probabilities (confidence scores) from softmax outputs.

    Parameters:
    -----------
    all_probs : np.ndarray
        Array of shape (samples, num_classes) containing probabilities.
    save_folder : str
        Folder where the histogram image will be saved.
    filename : str
        Name of the output image file.
    bins : int
        Number of bins in the histogram.
    """
    create_if_noexists(save_folder)

    # Get max prob per sample (confidence scores)
    max_probs = np.max(all_probs, axis=1)

    plt.figure(figsize=(8, 6))
    plt.hist(max_probs, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Confidence (Max Probability)")
    plt.ylabel("Number of Samples")
    plt.title("Histogram of Prediction Confidences")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def get_user_labels(dataset):
    """
    Extracts user_id labels sequentially from the dataset.
    Optimized to bypass slow pandas DataFrame building if unwrapping a subset.
    """
    # OPTIMIZATION: Same logic as stratify. If it's a massive dataset, 
    # doing dataset[i] thousands of times is extremely slow.
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_dataset = dataset
        subset_indices = np.arange(len(dataset))

    traj_ids = base_dataset.traj_ids #type: ignore
    df = base_dataset.traj_df #type: ignore
    
    selected_traj_ids = traj_ids[subset_indices]
    user_mapping = df.drop_duplicates(subset=['traj_id']).set_index('traj_id')['user_id'].to_dict()
    
    labels = [user_mapping[tid] for tid in selected_traj_ids]
    return labels