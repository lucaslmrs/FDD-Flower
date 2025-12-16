"""Advanced visualization utilities for Personalized Federated Learning observability."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from .task import Net


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def load_results_json(results_path: str = "results.json") -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


# =============================================================================
# a) Convergência e Performance
# =============================================================================

def plot_federated_convergence(results_json: dict, save_dir: str = "artifacts"):
    """Plot federated learning convergence (accuracy vs rounds).
    
    Args:
        results_json: Dictionary with federated_rounds data
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    federated_rounds = results_json.get("federated_rounds", {})
    
    if not federated_rounds:
        print("Warning: No federated rounds data found")
        return
    
    # Extract data
    rounds = []
    accuracies = []
    losses = []
    
    for round_num, metrics in sorted(federated_rounds.items(), key=lambda x: int(x[0])):
        rounds.append(int(round_num))
        accuracies.append(metrics.get("cen_accuracy", 0))
        losses.append(metrics.get("loss", 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax1.plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel("Round", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Centralized Accuracy", fontsize=12, fontweight='bold')
    ax1.set_title("Federated Learning Convergence - Accuracy", fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Add trend line
    z = np.polyfit(rounds, accuracies, 2)
    p = np.poly1d(z)
    ax1.plot(rounds, p(rounds), "--", alpha=0.5, color='red', label='Trend')
    ax1.legend()
    
    # Plot 2: Loss
    ax2.plot(rounds, losses, marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax2.set_xlabel("Round", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Centralized Loss", fontsize=12, fontweight='bold')
    ax2.set_title("Federated Learning Convergence - Loss", fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "federated_convergence.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Federated convergence plot saved to: {path}")


def plot_client_accuracy_evolution(results_json: dict, save_dir: str = "artifacts"):
    """Plot accuracy evolution per client during fine-tuning.
    
    Args:
        results_json: Dictionary with fine_tuning_results data
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fine_tuning_results = results_json.get("fine_tuning_results", {})
    
    if not fine_tuning_results:
        print("Warning: No fine-tuning results data found")
        return
    
    # Extract per-client data
    client_data = {}
    
    for round_key, round_data in sorted(fine_tuning_results.items(), key=lambda x: int(x[0].split('_')[1])):
        round_num = int(round_key.split('_')[1])
        clients = round_data.get("clients", {})
        
        for client_key, metrics in clients.items():
            if client_key not in client_data:
                client_data[client_key] = {'rounds': [], 'val_accuracy': [], 'val_loss': []}
            
            client_data[client_key]['rounds'].append(round_num)
            client_data[client_key]['val_accuracy'].append(metrics.get('val_accuracy', 0))
            client_data[client_key]['val_loss'].append(metrics.get('val_loss', 0))
    
    if not client_data:
        print("Warning: No client data found in fine-tuning results")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = sns.color_palette("husl", len(client_data))
    
    # Plot 1: Validation Accuracy
    for idx, (client_key, data) in enumerate(sorted(client_data.items())):
        ax1.plot(data['rounds'], data['val_accuracy'], marker='o', linewidth=2, 
                markersize=6, label=client_key, color=colors[idx])
    
    ax1.set_xlabel("Fine-tuning Round", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Validation Accuracy", fontsize=12, fontweight='bold')
    ax1.set_title("Client Accuracy Evolution During Fine-tuning", fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Validation Loss
    for idx, (client_key, data) in enumerate(sorted(client_data.items())):
        ax2.plot(data['rounds'], data['val_loss'], marker='s', linewidth=2, 
                markersize=6, label=client_key, color=colors[idx])
    
    ax2.set_xlabel("Fine-tuning Round", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Validation Loss", fontsize=12, fontweight='bold')
    ax2.set_title("Client Loss Evolution During Fine-tuning", fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "client_accuracy_evolution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Client accuracy evolution plot saved to: {path}")


# =============================================================================
# c) Contribuição dos Clientes
# =============================================================================

def plot_client_sample_distribution(save_dir: str = "artifacts"):
    """Plot number of samples per client (bar chart).
    
    Args:
        save_dir: Directory to save the plot
    """
    from .task import _load_csv_data
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    data = _load_csv_data()
    
    # Extract sample counts
    client_ids = []
    train_counts = []
    test_counts = []
    total_counts = []
    
    for client_id in sorted(data['data_by_client'].keys()):
        client_data = data['data_by_client'][client_id]
        train_count = len(client_data['features_train'])
        test_count = len(client_data['features_test'])
        
        client_ids.append(f"Client {client_id}")
        train_counts.append(train_count)
        test_counts.append(test_count)
        total_counts.append(train_count + test_count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(client_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_counts, width, label='Testing', color='#F18F01', alpha=0.8)
    
    # Add total on top
    for i, (train, test, total) in enumerate(zip(train_counts, test_counts, total_counts)):
        ax.text(i, total + max(total_counts) * 0.02, str(total), 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel("Client", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight='bold')
    ax.set_title("Sample Distribution per Client", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "client_sample_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Client sample distribution plot saved to: {path}")


def plot_test_labels_distribution(save_dir: str = "artifacts"):
    """Plot the distribution of labels per client in test data.
    
    This visualization shows how many normal (0) and fault (1) samples
    each client has in their test set, helping to understand data imbalance
    and client-specific fault patterns.
    
    Args:
        save_dir: Directory to save the plot
    """
    from .task import _load_csv_data
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    data = _load_csv_data()
    
    # Extract label distribution from test sets
    client_ids = []
    normal_counts = []
    fault_counts = []
    
    for client_id in sorted(data['data_by_client'].keys()):
        client_data = data['data_by_client'][client_id]
        test_labels = client_data['labels_test']
        
        # Count labels (0 = normal, 1 = fault)
        unique, counts = np.unique(test_labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        
        client_ids.append(f"Client {client_id}")
        normal_counts.append(label_counts.get(0, 0))
        fault_counts.append(label_counts.get(1, 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stacked bar chart
    x = np.arange(len(client_ids))
    width = 0.6
    
    bars1 = ax1.bar(x, normal_counts, width, label='Normal (0)', color='#16A085', alpha=0.8)
    bars2 = ax1.bar(x, fault_counts, width, bottom=normal_counts, label='Fault (1)', color='#E74C3C', alpha=0.8)
    
    # Add value labels on bars
    for i, (normal, fault) in enumerate(zip(normal_counts, fault_counts)):
        # Normal count
        if normal > 0:
            ax1.text(i, normal / 2, str(normal), ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white')
        # Fault count
        if fault > 0:
            ax1.text(i, normal + fault / 2, str(fault), ha='center', va='center',
                    fontweight='bold', fontsize=10, color='white')
        # Total on top
        total = normal + fault
        ax1.text(i, total + max(normal_counts + fault_counts) * 0.02, str(total),
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_xlabel("Client", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Number of Test Samples", fontsize=12, fontweight='bold')
    ax1.set_title("Test Set Label Distribution per Client - Stacked View", 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_ids)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Grouped bar chart
    width = 0.35
    bars1 = ax2.bar(x - width/2, normal_counts, width, label='Normal (0)', color='#16A085', alpha=0.8)
    bars2 = ax2.bar(x + width/2, fault_counts, width, label='Fault (1)', color='#E74C3C', alpha=0.8)
    
    # Add value labels on bars
    for i, normal in enumerate(normal_counts):
        if normal > 0:
            ax2.text(i - width/2, normal + max(normal_counts + fault_counts) * 0.01, 
                    str(normal), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for i, fault in enumerate(fault_counts):
        if fault > 0:
            ax2.text(i + width/2, fault + max(normal_counts + fault_counts) * 0.01, 
                    str(fault), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel("Client", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of Test Samples", fontsize=12, fontweight='bold')
    ax2.set_title("Test Set Label Distribution per Client - Grouped View", 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_ids)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "test_labels_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Test labels distribution plot saved to: {path}")
    
    # Print summary statistics
    total_test_samples = sum(normal_counts) + sum(fault_counts)
    total_normal = sum(normal_counts)
    total_fault = sum(fault_counts)
    
    print("\n=== Test Set Label Distribution Summary ===")
    print(f"Total test samples: {total_test_samples}")
    print(f"Total normal samples (0): {total_normal} ({total_normal/total_test_samples*100:.1f}%)")
    print(f"Total fault samples (1): {total_fault} ({total_fault/total_test_samples*100:.1f}%)")
    print("\nPer-client breakdown:")
    for i, client_id in enumerate(client_ids):
        total = normal_counts[i] + fault_counts[i]
        print(f"  {client_id}: {normal_counts[i]} normal, {fault_counts[i]} fault (total: {total})")


# =============================================================================
# d) Comparação Global vs Personalizado
# =============================================================================

def plot_global_vs_personalized(results_json: dict, save_dir: str = "artifacts"):
    """Compare global model accuracy vs personalized model accuracy.
    
    Args:
        results_json: Dictionary with federated and fine-tuning results
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get last federated round accuracy (global model)
    federated_rounds = results_json.get("federated_rounds", {})
    if federated_rounds:
        last_round = max([int(k) for k in federated_rounds.keys()])
        global_accuracy = federated_rounds[str(last_round)].get("cen_accuracy", 0)
    else:
        global_accuracy = 0
    
    # Get final personalized accuracies
    fine_tuning_results = results_json.get("fine_tuning_results", {})
    
    if not fine_tuning_results:
        print("Warning: No fine-tuning results found")
        return
    
    # Get last fine-tuning round
    last_ft_round = max([int(k.split('_')[1]) for k in fine_tuning_results.keys()])
    last_ft_key = f"round_{last_ft_round}"
    
    clients = fine_tuning_results[last_ft_key].get("clients", {})
    
    client_ids = []
    personalized_accuracies = []
    improvements = []
    
    for client_key in sorted(clients.keys()):
        metrics = clients[client_key]
        pers_acc = metrics.get('val_accuracy', 0)
        
        client_ids.append(client_key.replace('client_', 'Client '))
        personalized_accuracies.append(pers_acc)
        improvements.append(pers_acc - global_accuracy)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Comparison bar chart
    x = np.arange(len(client_ids))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [global_accuracy] * len(client_ids), width, 
                    label='Global Model', color='#A23B72', alpha=0.7)
    bars2 = ax1.bar(x + width/2, personalized_accuracies, width, 
                    label='Personalized Model', color='#2E86AB', alpha=0.7)
    
    ax1.set_xlabel("Client", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Validation Accuracy", fontsize=12, fontweight='bold')
    ax1.set_title("Global vs Personalized Model Comparison", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_ids)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Improvement (delta)
    colors = ['#16A085' if imp > 0 else '#E74C3C' for imp in improvements]
    bars3 = ax2.bar(x, improvements, color=colors, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel("Client", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Accuracy Improvement (Δ)", fontsize=12, fontweight='bold')
    ax2.set_title("Performance Gain from Personalization", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(client_ids)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars3, improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.3f}', ha='center', va='bottom' if imp > 0 else 'top', 
                fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(save_dir, "global_vs_personalized.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Global vs personalized comparison plot saved to: {path}")
    
    # Print summary
    print("\n=== Personalization Summary ===")
    print(f"Global model accuracy: {global_accuracy:.4f}")
    print(f"Average personalized accuracy: {np.mean(personalized_accuracies):.4f}")
    print(f"Average improvement: {np.mean(improvements):+.4f}")
    print(f"Best improvement: {max(improvements):+.4f}")
    print(f"Worst improvement: {min(improvements):+.4f}")


# =============================================================================
# f) Estrutura do Modelo - Frozen vs Trainable Parameters
# =============================================================================

def plot_model_parameters_analysis(save_dir: str = "artifacts"):
    """Analyze and visualize frozen vs trainable parameters during fine-tuning.
    
    Args:
        save_dir: Directory to save the plot
    """
    from .task import Net, freeze_first_half
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model and apply freezing
    net = Net()
    freeze_first_half(net)
    
    # Analyze parameters
    layer_names = []
    frozen_params = []
    trainable_params = []
    
    for name, module in net.network.named_children():
        total_params = sum(p.numel() for p in module.parameters())
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if total_params > 0:  # Only include layers with parameters
            layer_names.append(f"Layer {name}")
            frozen_params.append(frozen)
            trainable_params.append(trainable)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stacked bar chart
    x = np.arange(len(layer_names))
    
    bars1 = ax1.barh(x, frozen_params, label='Frozen', color='#95A5A6', alpha=0.8)
    bars2 = ax1.barh(x, trainable_params, left=frozen_params, 
                     label='Trainable', color='#3498DB', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel("Number of Parameters", fontsize=12, fontweight='bold')
    ax1.set_title("Frozen vs Trainable Parameters per Layer", fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Plot 2: Pie chart of total parameters
    total_frozen = sum(frozen_params)
    total_trainable = sum(trainable_params)
    
    sizes = [total_frozen, total_trainable]
    labels = [f'Frozen\n({total_frozen:,} params)', f'Trainable\n({total_trainable:,} params)']
    colors = ['#95A5A6', '#3498DB']
    explode = (0, 0.1)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title("Total Parameter Distribution", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    path = os.path.join(save_dir, "model_parameters_analysis.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Model parameters analysis plot saved to: {path}")
    print(f"\n=== Parameter Analysis ===")
    print(f"Total frozen parameters: {total_frozen:,}")
    print(f"Total trainable parameters: {total_trainable:,}")
    print(f"Total parameters: {total_frozen + total_trainable:,}")
    print(f"Trainable ratio: {total_trainable / (total_frozen + total_trainable) * 100:.1f}%")


# =============================================================================
# g) Análise de Erros - Confusion Matrices
# =============================================================================

def plot_confusion_matrices(run_dir: str, save_dir: str = None):
    """Generate confusion matrices for global and personalized models.
    
    Args:
        run_dir: Directory containing model checkpoints (e.g., artifacts/run_005)
        save_dir: Directory to save plots (defaults to run_dir if None)
    """
    if save_dir is None:
        save_dir = run_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    from .task import _load_csv_data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = _load_csv_data()
    num_clients = data['num_clients']
    
    # Find global model (last federated round)
    global_models = sorted([f for f in os.listdir(run_dir) if f.startswith("global_model_round_")])
    if not global_models:
        print("Warning: No global model found")
        return
    
    last_global_model = os.path.join(run_dir, global_models[-1])
    
    # Load global model
    global_net = Net()
    global_net.load_state_dict(torch.load(last_global_model, map_location=device))
    global_net.to(device)
    global_net.eval()
    
    # Prepare for aggregated confusion matrix
    all_y_true = []
    all_y_pred_global = []
    all_y_pred_personalized = []
    
    # Per-client confusion matrices data
    client_cms = {}
    
    for partition_id in range(num_clients):
        client_id = data['partition_to_client'][partition_id]
        client_data = data['data_by_client'][client_id]
        
        # Prepare test data
        X_test = torch.tensor(client_data['features_test'], dtype=torch.float32).to(device)
        y_test = client_data['labels_test']
        
        # Global model predictions
        with torch.no_grad():
            outputs_global = global_net(X_test)
            _, y_pred_global = torch.max(outputs_global, 1)
            y_pred_global = y_pred_global.cpu().numpy()
        
        # Load personalized model
        personalized_model_path = os.path.join(run_dir, f"personalized_model_client_{partition_id}")
        
        if os.path.exists(personalized_model_path):
            personalized_net = Net()
            personalized_net.load_state_dict(torch.load(personalized_model_path, map_location=device))
            personalized_net.to(device)
            personalized_net.eval()
            
            with torch.no_grad():
                outputs_pers = personalized_net(X_test)
                _, y_pred_pers = torch.max(outputs_pers, 1)
                y_pred_pers = y_pred_pers.cpu().numpy()
        else:
            y_pred_pers = y_pred_global
        
        # Store for aggregated CM
        all_y_true.extend(y_test)
        all_y_pred_global.extend(y_pred_global)
        all_y_pred_personalized.extend(y_pred_pers)
        
        # Per-client CM
        cm_global = confusion_matrix(y_test, y_pred_global, labels=[0, 1])
        cm_pers = confusion_matrix(y_test, y_pred_pers, labels=[0, 1])
        
        client_cms[client_id] = {
            'global': cm_global,
            'personalized': cm_pers,
            'y_true': y_test,
            'y_pred_global': y_pred_global,
            'y_pred_pers': y_pred_pers
        }
    
    # Plot 1: Aggregated confusion matrices
    cm_global_agg = confusion_matrix(all_y_true, all_y_pred_global, labels=[0, 1])
    cm_pers_agg = confusion_matrix(all_y_true, all_y_pred_personalized, labels=[0, 1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Global CM
    sns.heatmap(cm_global_agg, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax1.set_ylabel("True", fontsize=12, fontweight='bold')
    ax1.set_title("Global Model - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    
    # Personalized CM
    sns.heatmap(cm_pers_agg, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'],
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax2.set_ylabel("True", fontsize=12, fontweight='bold')
    ax2.set_title("Personalized Models - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix_aggregated.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Aggregated confusion matrix plot saved to: {path}")
    
    # Plot 2: Per-client confusion matrices
    n_clients = len(client_cms)
    fig, axes = plt.subplots(2, n_clients, figsize=(5 * n_clients, 10))
    
    if n_clients == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (client_id, cms) in enumerate(sorted(client_cms.items())):
        # Global model CM
        sns.heatmap(cms['global'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'],
                    ax=axes[0, idx], cbar_kws={'label': 'Count'})
        axes[0, idx].set_title(f"Client {client_id} - Global", fontweight='bold')
        axes[0, idx].set_xlabel("Predicted")
        axes[0, idx].set_ylabel("True")
        
        # Personalized model CM
        sns.heatmap(cms['personalized'], annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'],
                    ax=axes[1, idx], cbar_kws={'label': 'Count'})
        axes[1, idx].set_title(f"Client {client_id} - Personalized", fontweight='bold')
        axes[1, idx].set_xlabel("Predicted")
        axes[1, idx].set_ylabel("True")
    
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix_per_client.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Per-client confusion matrices plot saved to: {path}")
    
    # Plot 3: Error type heatmap (clients × error types)
    error_types = ['TN', 'FP', 'FN', 'TP']
    error_matrix_global = np.zeros((n_clients, 4))
    error_matrix_pers = np.zeros((n_clients, 4))
    
    client_labels = []
    
    for idx, (client_id, cms) in enumerate(sorted(client_cms.items())):
        client_labels.append(f"Client {client_id}")
        
        # Global: TN, FP, FN, TP
        cm_g = cms['global']
        error_matrix_global[idx] = [cm_g[0,0], cm_g[0,1], cm_g[1,0], cm_g[1,1]]
        
        # Personalized
        cm_p = cms['personalized']
        error_matrix_pers[idx] = [cm_p[0,0], cm_p[0,1], cm_p[1,0], cm_p[1,1]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Global error heatmap
    sns.heatmap(error_matrix_global, annot=True, fmt='.0f', cmap='RdYlGn_r',
                xticklabels=error_types, yticklabels=client_labels,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel("Error Type", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Client", fontsize=12, fontweight='bold')
    ax1.set_title("Global Model - Error Distribution", fontsize=14, fontweight='bold', pad=15)
    
    # Personalized error heatmap
    sns.heatmap(error_matrix_pers, annot=True, fmt='.0f', cmap='RdYlGn_r',
                xticklabels=error_types, yticklabels=client_labels,
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel("Error Type", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Client", fontsize=12, fontweight='bold')
    ax2.set_title("Personalized Models - Error Distribution", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    path = os.path.join(save_dir, "error_type_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Error type heatmap plot saved to: {path}")


# =============================================================================
# Main function to generate all visualizations
# =============================================================================

def generate_all_visualizations(results_path: str = "results.json", 
                                run_dir: str = None,
                                save_dir: str = "artifacts"):
    """Generate all visualization plots for observability.
    
    Args:
        results_path: Path to results.json file
        run_dir: Path to run directory with model checkpoints
        save_dir: Directory to save all plots
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS FOR OBSERVABILITY")
    print("="*60 + "\n")
    
    # Load results
    results = load_results_json(results_path)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # a) Convergência e Performance
    print("\n[1/8] Plotting federated convergence...")
    plot_federated_convergence(results, save_dir)
    
    print("\n[2/8] Plotting client accuracy evolution...")
    plot_client_accuracy_evolution(results, save_dir)
    
    # c) Contribuição dos Clientes
    print("\n[3/8] Plotting client sample distribution...")
    plot_client_sample_distribution(save_dir)
    
    print("\n[4/8] Plotting test labels distribution per client...")
    plot_test_labels_distribution(save_dir)
    
    # d) Comparação Global vs Personalizado
    print("\n[5/8] Plotting global vs personalized comparison...")
    plot_global_vs_personalized(results, save_dir)
    
    # f) Estrutura do Modelo
    print("\n[6/8] Plotting model parameters analysis...")
    plot_model_parameters_analysis(save_dir)
    
    # g) Análise de Erros
    if run_dir and os.path.exists(run_dir):
        print("\n[7/8] Plotting confusion matrices...")
        plot_confusion_matrices(run_dir, save_dir)
    else:
        print("\n[7/8] Skipping confusion matrices (run_dir not provided or doesn't exist)")
    
    # Class distribution (already exists)
    print("\n[8/8] Class distribution already generated during training")
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"Plots saved to: {save_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
        results_path = os.path.join(run_dir, "results.json")
        generate_all_visualizations(results_path, run_dir, run_dir)
    else:
        # Use default paths
        generate_all_visualizations("results.json", "artifacts/run_005", "artifacts")
