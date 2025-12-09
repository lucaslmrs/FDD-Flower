"""Visualization utilities for Federated Learning data distribution."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_class_distribution(
    partition_indices: list,
    labels: np.ndarray,
    num_classes: int = 8,
    save_path: str = "artifacts/class_distribution.png",
    figsize: tuple = (12, 8),
):
    """Plot the class distribution across federated learning participants.
    
    Args:
        partition_indices: List of index arrays, one per partition/client
        labels: Array of labels for each sample
        num_classes: Number of classes in the dataset
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    num_partitions = len(partition_indices)
    
    # Build distribution matrix: rows = clients, cols = classes
    distribution = np.zeros((num_partitions, num_classes), dtype=int)
    
    for client_id, indices in enumerate(partition_indices):
        for idx in indices:
            label = labels[idx]
            distribution[client_id, label] += 1
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Heatmap of class distribution
    ax1 = axes[0]
    sns.heatmap(
        distribution,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=[f"Class {i}" for i in range(num_classes)],
        yticklabels=[f"Client {i}" for i in range(num_partitions)],
        ax=ax1,
        cbar_kws={"label": "Number of Samples"},
    )
    ax1.set_title("Class Distribution per Client (Heatmap)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fault Class", fontsize=10)
    ax1.set_ylabel("Federated Client", fontsize=10)
    
    # Plot 2: Stacked bar chart
    ax2 = axes[1]
    x = np.arange(num_partitions)
    bottom = np.zeros(num_partitions)
    
    colors = sns.color_palette("husl", num_classes)
    
    for class_id in range(num_classes):
        values = distribution[:, class_id]
        ax2.bar(x, values, bottom=bottom, label=f"Class {class_id}", color=colors[class_id])
        bottom += values
    
    ax2.set_title("Class Distribution per Client (Stacked Bar)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Federated Client", fontsize=10)
    ax2.set_ylabel("Number of Samples", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Client {i}" for i in range(num_partitions)])
    ax2.legend(title="Fault Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    
    plt.suptitle("Federated Learning Data Distribution - Bearing Fault Detection", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Class distribution plot saved to: {save_path}")
    
    # Print summary statistics
    print("\n=== Distribution Summary ===")
    print(f"Total clients: {num_partitions}")
    print(f"Total classes: {num_classes}")
    print(f"Total samples: {distribution.sum()}")
    print(f"\nSamples per client: {distribution.sum(axis=1)}")
    print(f"Samples per class: {distribution.sum(axis=0)}")
    
    return distribution
