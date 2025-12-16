"""Flower app for Pick-and-Place Fault Detection using Federated Learning."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import signal processing functions
from flower_app.signal_processing import (
    process_signal_to_features,
    window_signal,
)

# =============================================================================
# Configuration (default values - can be overridden via pyproject.toml)
# =============================================================================

# Path to dataset
DATA_PATH = Path(__file__).parent.parent / "data" / "all_scenarios_concatenated.csv"

# System parameters (defaults)
NUM_CLASSES = 2  # Binary classification: 0 = normal, 1 = fault
NUM_FEATURES = 11  # Number of input features extracted from signal processing

# Signal processing configuration
USE_SIGNAL_PROCESSING = True  # Enable Hilbert + FFT processing
SAMPLING_RATE = 10.0  # Hz - approximate sampling rate from timestamps (~100ms intervals)
WINDOW_SIZE_SECONDS = 20.0  # Window size for signal segmentation (20 seconds)
WINDOW_OVERLAP = 0.0  # No overlap between windows

# Harmonic frequency bands for feature extraction (in Hz)
# Adapted from bearing fault detection - adjust based on Pick-and-Place dynamics
HARMONIC_BANDS = [
    (0.1, 0.5),   # Very low frequency - slow drift/degradation
    (0.5, 1.0),   # Low frequency - position control oscillations
    (1.0, 2.0),   # Mid frequency - velocity variations
    (2.0, 3.0),   # Higher frequency - rapid changes/faults
]

# Raw signal columns to process (velocity and position signals)
SIGNAL_COLUMNS = [
    'x_speed', 'y_speed', 'z_speed',          # Velocity signals
    'x_filt_speed', 'y_filt_speed', 'z_filt_speed',  # Filtered velocity
    'x_pos', 'y_pos', 'z_pos',                # Position signals
    'AVG_SPEED',                               # Average speed
]

# Label column
LABEL_COLUMN = 'has_fault'

# Client ID column for partitioning
CLIENT_ID_COLUMN = 'client_id'


# =============================================================================
# Model Definition - Neural Network for Fault Detection
# =============================================================================

# Neural Network Architecture Configuration
HIDDEN_LAYERS = [256, 128, 64, 32]  # Hidden layer sizes - deeper network for complex patterns
DROPOUT_RATE = 0.4  # Dropout rate for regularization
USE_BATCH_NORM = True  # Whether to use batch normalization

# Training Optimization Configuration
BATCH_SIZE = 128  # Larger batch size for GPU acceleration
NUM_WORKERS = 4  # Number of workers for DataLoader (parallel data loading)
PIN_MEMORY = True  # Pin memory for faster GPU transfer
USE_AMP = True  # Use Automatic Mixed Precision for faster training


class Net(nn.Module):
    """Neural Network model for fault detection.
    
    Architecture:
        - Input layer: num_features
        - Hidden layers: configurable via HIDDEN_LAYERS
        - Batch normalization (optional): after each hidden layer
        - Dropout: for regularization
        - Output layer: num_classes (softmax applied during loss calculation)
    """

    def __init__(self, num_features: int = None, num_classes: int = None,
                 hidden_layers: list = None, dropout_rate: float = None,
                 use_batch_norm: bool = None):
        super(Net, self).__init__()
        
        # Use defaults if not provided
        num_features = num_features if num_features is not None else NUM_FEATURES
        num_classes = num_classes if num_classes is not None else NUM_CLASSES
        hidden_layers = hidden_layers if hidden_layers is not None else HIDDEN_LAYERS
        dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        use_batch_norm = use_batch_norm if use_batch_norm is not None else USE_BATCH_NORM
        
        # Build network layers
        layers = []
        in_features = num_features
        
        for hidden_size in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_size))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation function
            layers.append(nn.ReLU())
            
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Store configuration for reference
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers

    def forward(self, x):
        return self.network(x)
    
    def __repr__(self):
        return (f"Net(num_features={self.num_features}, "
                f"num_classes={self.num_classes}, "
                f"hidden_layers={self.hidden_layers})")


# =============================================================================
# Data Loading Functions
# =============================================================================

# Cache for loaded data
_cached_data = None
_class_weights = None


def _process_signal_features(df, client_id):
    """Process time-series signals using Hilbert + FFT to extract features.
    
    Args:
        df: DataFrame with time-series data for one client/scenario
        client_id: Client identifier for logging
        
    Returns:
        DataFrame with extracted features and labels
    """
    print(f"  Processing signals for client {client_id} using Hilbert + FFT...")
    
    # Sort by timestamp to ensure temporal order
    if 'SourceTimeStamp' in df.columns:
        df = df.sort_values('SourceTimeStamp').reset_index(drop=True)
    
    all_features = []
    all_labels = []
    
    # Check if we should use windowing
    total_samples = len(df)
    expected_samples_per_window = int(WINDOW_SIZE_SECONDS * SAMPLING_RATE)
    
    if USE_SIGNAL_PROCESSING and total_samples >= expected_samples_per_window:
        # Process with windowing
        print(f"    Using {WINDOW_SIZE_SECONDS}s windows (overlap={WINDOW_OVERLAP})")
        
        # Process each signal column
        for col in SIGNAL_COLUMNS:
            if col not in df.columns:
                print(f"    Warning: Column {col} not found, skipping")
                continue
            
            signal = df[col].values
            signal = np.nan_to_num(signal, nan=0.0)
            
            # Create windows
            windows = window_signal(
                signal, 
                WINDOW_SIZE_SECONDS, 
                SAMPLING_RATE, 
                WINDOW_OVERLAP
            )
            
            # Process each window
            for window_idx, window in enumerate(windows):
                # Extract features from this window
                features = process_signal_to_features(
                    window, 
                    SAMPLING_RATE, 
                    HARMONIC_BANDS
                )
                
                # Get label for this window (majority vote or last sample)
                window_start = int(window_idx * WINDOW_SIZE_SECONDS * SAMPLING_RATE * (1 - WINDOW_OVERLAP))
                window_end = min(window_start + expected_samples_per_window, total_samples)
                window_labels = df[LABEL_COLUMN].iloc[window_start:window_end].values
                label = int(np.round(np.mean(window_labels)))  # Majority vote
                
                # Add column identifier to features
                features_with_prefix = {f"{col}_{k}": v for k, v in features.items()}
                all_features.append(features_with_prefix)
                all_labels.append(label)
        
        if not all_features:
            print(f"    Warning: No features extracted with windowing, using entire signal")
            # Fallback to processing entire signal
            return _process_entire_signal(df, client_id)
        
        # Combine features from all columns into single rows
        # Group by window index (every len(SIGNAL_COLUMNS) features belong to same window)
        features_df = pd.DataFrame(all_features)
        
        print(f"    Extracted {len(features_df)} samples from {len(windows)} windows × {len(SIGNAL_COLUMNS)} signals")
        
    else:
        # Process entire signal as one sample (no windowing)
        print(f"    Processing entire signal (no windowing, {total_samples} samples)")
        return _process_entire_signal(df, client_id)
    
    # Create final dataset
    features_df['label'] = all_labels
    return features_df


def _process_entire_signal(df, client_id):
    """Process entire signal as one sample without windowing.
    
    Args:
        df: DataFrame with time-series data
        client_id: Client identifier
        
    Returns:
        DataFrame with one row of features
    """
    all_features = {}
    
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            continue
        
        signal = df[col].values
        signal = np.nan_to_num(signal, nan=0.0)
        
        # Extract features from entire signal
        features = process_signal_to_features(
            signal, 
            SAMPLING_RATE, 
            HARMONIC_BANDS
        )
        
        # Add column prefix
        for k, v in features.items():
            all_features[f"{col}_{k}"] = v
    
    # Get majority label
    label = int(np.round(df[LABEL_COLUMN].mean()))
    
    features_df = pd.DataFrame([all_features])
    features_df['label'] = label
    
    print(f"    Extracted 1 sample with {len(all_features)} features")
    return features_df


def _load_csv_data():
    """Load and cache the CSV data file."""
    global _cached_data, _class_weights
    
    if _cached_data is not None:
        return _cached_data
    
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Get unique client IDs
    unique_clients = df[CLIENT_ID_COLUMN].unique()
    print(f"Unique clients found: {unique_clients}")
    
    # Create client_id to partition_id mapping
    client_to_partition = {client: idx for idx, client in enumerate(sorted(unique_clients))}
    print(f"Client to partition mapping: {client_to_partition}")
    
    # Process each client's data with signal processing
    data_by_client = {}
    all_features_list = []
    all_labels_list = []
    
    for client in unique_clients:
        client_df = df[df[CLIENT_ID_COLUMN] == client].copy()
        
        # Process signals to extract features
        if USE_SIGNAL_PROCESSING:
            features_df = _process_signal_features(client_df, client)
        else:
            # Legacy mode: use raw features directly
            print(f"  Using raw features (signal processing disabled)")
            available_features = [col for col in SIGNAL_COLUMNS if col in client_df.columns]
            features_df = client_df[available_features + [LABEL_COLUMN]].copy()
            features_df.columns = list(features_df.columns[:-1]) + ['label']
        
        # Handle NaN values
        features_df = features_df.fillna(0.0)
        
        # Split into features and labels
        label_col = 'label'
        feature_cols = [col for col in features_df.columns if col != label_col]
        
        features = features_df[feature_cols].values.astype(np.float32)
        labels = features_df[label_col].values.astype(np.int64)
        
        # Split into train (80%) and test (20%)
        n_samples = len(features)
        n_train = int(0.8 * n_samples)
        
        # Shuffle before splitting
        indices = np.random.permutation(n_samples)
        features = features[indices]
        labels = labels[indices]
        
        data_by_client[client] = {
            'features_train': features[:n_train],
            'labels_train': labels[:n_train],
            'features_test': features[n_train:],
            'labels_test': labels[n_train:],
        }
        
        all_features_list.append(features)
        all_labels_list.append(labels)
        
        print(f"  - Client {client}: {n_train} train, {n_samples - n_train} test samples")
    
    # Combine all data for normalization and class weights
    all_features = np.concatenate(all_features_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    
    # Normalize features (z-score normalization)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    
    # Apply normalization to each client's data
    for client in unique_clients:
        data_by_client[client]['features_train'] = (
            data_by_client[client]['features_train'] - mean
        ) / std
        data_by_client[client]['features_test'] = (
            data_by_client[client]['features_test'] - mean
        ) / std
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    _class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"\nDataset statistics:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Features per sample: {all_features.shape[1]}")
    print(f"  - Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"  - Class weights: {class_weights}")
    
    # Update NUM_FEATURES based on actual feature count
    global NUM_FEATURES
    NUM_FEATURES = all_features.shape[1]
    print(f"  - Updated NUM_FEATURES to: {NUM_FEATURES}")
    
    _cached_data = {
        'data_by_client': data_by_client,
        'client_to_partition': client_to_partition,
        'partition_to_client': {v: k for k, v in client_to_partition.items()},
        'num_clients': len(unique_clients),
        'mean': mean,
        'std': std,
        'num_features': NUM_FEATURES,
    }
    
    return _cached_data


def get_class_weights():
    """Get class weights for imbalanced data handling."""
    global _class_weights
    if _class_weights is None:
        _load_csv_data()
    return _class_weights


# Cache for visualization flag
_visualization_done = False


def load_data(partition_id: int, num_partitions: int, alpha: float = None, sampling_rate: int = None):
    """Load partitioned fault detection data for a specific client.
    
    Args:
        partition_id: ID of the partition/client (0 to num_partitions-1)
        num_partitions: Total number of partitions (ignored - uses actual clients)
        alpha: Dirichlet alpha parameter (ignored - using natural partitioning)
        sampling_rate: Sampling rate (ignored - not applicable for CSV data)
        
    Returns:
        Tuple of (trainloader, testloader)
    """
    global _visualization_done
    
    # Load data
    data = _load_csv_data()
    
    # Get client ID for this partition
    if partition_id >= data['num_clients']:
        raise ValueError(f"partition_id {partition_id} >= num_clients {data['num_clients']}")
    
    client_id = data['partition_to_client'][partition_id]
    client_data = data['data_by_client'][client_id]
    
    # Generate visualization plot (only once)
    if not _visualization_done:
        try:
            from .visualization import plot_class_distribution
            
            # Prepare data for visualization
            all_labels = []
            partition_indices = []
            current_idx = 0
            
            for pid in range(data['num_clients']):
                cid = data['partition_to_client'][pid]
                cdata = data['data_by_client'][cid]
                n_samples = len(cdata['labels_train'])
                all_labels.extend(cdata['labels_train'].tolist())
                partition_indices.append(np.arange(current_idx, current_idx + n_samples))
                current_idx += n_samples
            
            plot_class_distribution(
                partition_indices=partition_indices,
                labels=np.array(all_labels),
                num_classes=NUM_CLASSES,
                save_dir="artifacts",
            )
            _visualization_done = True
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
            _visualization_done = True
    
    # Get actual number of features from data
    num_features_actual = client_data['features_train'].shape[1]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(client_data['features_train'], dtype=torch.float32),
        torch.tensor(client_data['labels_train'], dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(client_data['features_test'], dtype=torch.float32),
        torch.tensor(client_data['labels_test'], dtype=torch.long)
    )
    
    # Create DataLoaders with optimizations
    trainloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    testloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    return trainloader, testloader
    
    return trainloader, testloader


def load_test_data():
    """Load the centralized test dataset (all clients combined).
    
    Returns:
        DataLoader for test data
    """
    data = _load_csv_data()
    
    # Combine test data from all clients
    all_features = []
    all_labels = []
    
    for client_id in data['data_by_client']:
        client_data = data['data_by_client'][client_id]
        all_features.append(client_data['features_test'])
        all_labels.append(client_data['labels_test'])
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    test_dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    return DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set with class weights and mixed precision.
    
    Args:
        net: PyTorch model
        trainloader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on (cpu/cuda)
        
    Returns:
        Average training loss
    """
    net.to(device)
    
    # Use class weights for imbalanced data
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Mixed precision training for GPU acceleration
    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    net.train()
    
    running_loss = 0.0
    total_batches = 0
    
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if use_amp:
                # Mixed precision forward and backward pass
                with torch.cuda.amp.autocast():
                    outputs = net(features)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = net(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
    
    avg_trainloss = running_loss / max(total_batches, 1)
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate the model on the test set.
    
    Args:
        net: PyTorch model
        testloader: DataLoader for test data
        device: Device to evaluate on (cpu/cuda)
        
    Returns:
        Tuple of (loss, accuracy)
    """
    net.to(device)
    
    # Use class weights for consistent evaluation
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = net(features)
            total_loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct / max(total_samples, 1)
    avg_loss = total_loss / max(len(testloader), 1)
    
    return avg_loss, accuracy


# =============================================================================
# Weight Management Functions
# =============================================================================

def get_weights(net):
    """Extract parameters from a model as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Copy parameters onto the model from numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v.copy()) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# =============================================================================
# Personalized Federated Learning - Fine-tuning Functions
# =============================================================================

def freeze_first_half(net):
    """Freeze the first half of the network layers for fine-tuning.
    
    With HIDDEN_LAYERS = [256, 128, 64, 32], the network has:
    - 4 hidden blocks × 4 modules each (Linear, BatchNorm, ReLU, Dropout) = 16 modules
    - 1 output layer = 17 modules total
    
    First half (freeze): modules 0-7 (hidden layers 256, 128)
    Second half (train): modules 8-16 (hidden layers 64, 32 + output)
    """
    modules = list(net.network.children())
    total_modules = len(modules)
    freeze_until = total_modules // 2  # Freeze first half
    
    frozen_count = 0
    trainable_count = 0
    
    for idx, module in enumerate(modules):
        if idx < freeze_until:
            # Freeze this layer
            for param in module.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
        else:
            # Keep trainable
            for param in module.parameters():
                param.requires_grad = True
                trainable_count += param.numel()
    
    print(f"Fine-tuning mode: Frozen {frozen_count} params, Trainable {trainable_count} params")
    return net


def fine_tune(net, trainloader, epochs, lr, device):
    """Fine-tune only the unfrozen (second half) layers of the model.
    
    Args:
        net: PyTorch model with some layers frozen
        trainloader: DataLoader for training data
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning (typically smaller)
        device: Device to train on (cpu/cuda)
        
    Returns:
        Average training loss during fine-tuning
    """
    net.to(device)
    
    # Use class weights for imbalanced data
    class_weights = get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    # Only optimize parameters that require gradients (unfrozen layers)
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    # Mixed precision training for GPU acceleration
    use_amp = USE_AMP and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    net.train()
    
    running_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        for batch in trainloader:
            features, labels = batch
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(features)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
    
    avg_loss = running_loss / max(total_batches, 1)
    return avg_loss


def save_personalized_model(net, client_id, run_dir):
    """Save a personalized model for a specific client.
    
    Args:
        net: PyTorch model to save
        client_id: Client identifier
        run_dir: Directory for this run (e.g., artifacts/run_001)
    """
    import os
    os.makedirs(run_dir, exist_ok=True)
    
    model_path = os.path.join(run_dir, f"personalized_model_client_{client_id}")
    torch.save(net.state_dict(), model_path)
    print(f"Saved personalized model for client {client_id} to {model_path}")
    return model_path


def get_next_run_dir(base_dir="artifacts"):
    """Get the next run directory with incremental counter.
    
    Returns:
        Path to the new run directory (e.g., artifacts/run_001)
    """
    import os
    os.makedirs(base_dir, exist_ok=True)
    
    # Find existing run directories
    existing_runs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    
    if not existing_runs:
        next_num = 1
    else:
        # Extract numbers and find max
        nums = []
        for run in existing_runs:
            try:
                num = int(run.split("_")[1])
                nums.append(num)
            except (IndexError, ValueError):
                continue
        next_num = max(nums) + 1 if nums else 1
    
    run_dir = os.path.join(base_dir, f"run_{next_num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")
    return run_dir
