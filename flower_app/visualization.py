"""Visualization utilities for Federated Learning data distribution."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_class_distribution(
    partition_indices: list,
    labels: np.ndarray,
    num_classes: int = 8,
    save_dir: str = "artifacts",
    figsize_heatmap: tuple = (10, 8),
    figsize_bar: tuple = (12, 8),
):
    """Plot the class distribution across federated learning participants.
    
    Args:
        partition_indices: List of index arrays, one per partition/client
        labels: Array of labels for each sample
        num_classes: Number of classes in the dataset
        save_dir: Directory to save the plots
        figsize_heatmap: Figure size for heatmap (width, height)
        figsize_bar: Figure size for bar chart (width, height)
    """
    num_partitions = len(partition_indices)
    
    # Build distribution matrix: rows = clients, cols = classes
    distribution = np.zeros((num_partitions, num_classes), dtype=int)
    
    for client_id, indices in enumerate(partition_indices):
        for idx in indices:
            label = labels[idx]
            distribution[client_id, label] += 1
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Heatmap of class distribution
    fig1, ax1 = plt.subplots(figsize=figsize_heatmap)
    sns.heatmap(
        distribution,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {i}" for i in range(num_classes)],
        yticklabels=[f"Client {i}" for i in range(num_partitions)],
        ax=ax1,
        cbar_kws={"label": "Number of Samples"},
    )
    ax1.set_title("Class Distribution per Client - Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Fault Class", fontsize=11)
    ax1.set_ylabel("Federated Client", fontsize=11)
    
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "class_distribution_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Heatmap saved to: {heatmap_path}")
    
    # Plot 2: Stacked bar chart
    fig2, ax2 = plt.subplots(figsize=figsize_bar)
    x = np.arange(num_partitions)
    bottom = np.zeros(num_partitions)
    
    colors = sns.color_palette("Blues_r", num_classes)
    
    for class_id in range(num_classes):
        values = distribution[:, class_id]
        ax2.bar(x, values, bottom=bottom, label=f"Class {class_id}", color=colors[class_id])
        bottom += values
    
    ax2.set_title("Class Distribution per Client - Stacked Bar Chart", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Federated Client", fontsize=11)
    ax2.set_ylabel("Number of Samples", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Client {i}" for i in range(num_partitions)])
    ax2.legend(title="Fault Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    bar_path = os.path.join(save_dir, "class_distribution_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Bar chart saved to: {bar_path}")
    
    # Print summary statistics
    print("\n=== Distribution Summary ===")
    print(f"Total clients: {num_partitions}")
    print(f"Total classes: {num_classes}")
    print(f"Total samples: {distribution.sum()}")
    print(f"\nSamples per client: {distribution.sum(axis=1)}")
    print(f"Samples per class: {distribution.sum(axis=0)}")
    
    return distribution
