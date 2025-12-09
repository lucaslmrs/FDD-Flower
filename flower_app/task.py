"""Flower app for Bearing Fault Detection using Federated Learning."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy.fft import fft
from scipy.signal import hilbert
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Configuration (default values - can be overridden via pyproject.toml)
# =============================================================================

# Path to dataset (fixed path)
DATA_PATH = Path(__file__).parent.parent / "downloads" / "dataset_for_team"

# System parameters (defaults)
FS = 50000  # Sampling rate (Hz) - configurable via 'sampling-rate'
NUM_CLASSES = 8  # Number of classes - configurable via 'num-classes'
NUM_FEATURES = 4  # Number of input features - configurable via 'num-features'
DIRICHLET_ALPHA = 1.0  # Dirichlet alpha - configurable via 'dirichlet-alpha'


# =============================================================================
# Model Definition - Multiclass Logistic Regression in PyTorch
# =============================================================================

class Net(nn.Module):
    """Multiclass Logistic Regression model for bearing fault detection."""

    def __init__(self, num_features: int = None, num_classes: int = None):
        super(Net, self).__init__()
        num_features = num_features if num_features is not None else NUM_FEATURES
        num_classes = num_classes if num_classes is not None else NUM_CLASSES
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_features_single(signal: np.ndarray, sampling_rate: int = None) -> np.ndarray:
    """Extract features from a single vibration signal using Hilbert Transform + FFT.
    
    Args:
        signal: 1D numpy array with vibration data
        sampling_rate: Sampling rate in Hz (uses FS default if not provided)
        
    Returns:
        1D numpy array with 4 harmonic features
    """
    fs = sampling_rate if sampling_rate is not None else FS
    
    # Apply Hilbert transform to get envelope
    envelope = np.abs(hilbert(signal))
    
    # Centralize (remove mean)
    envelope_centered = envelope - np.mean(envelope)
    
    # Apply FFT
    fft_result = np.abs(fft(envelope_centered))
    
    # Take only positive frequencies (first half)
    n_points = len(fft_result) // 2
    fft_positive = fft_result[:n_points]
    
    # Calculate frequency vector
    T_total = n_points * 2 / fs
    d_f = 1 / T_total
    frequency_vector = np.arange(0, fs - d_f, d_f)
    
    # Ensure frequency_vector matches fft_positive length
    if len(frequency_vector) > len(fft_positive):
        frequency_vector = frequency_vector[:len(fft_positive)]
    
    # Extract harmonics at specific frequency bands
    # These bands correspond to characteristic fault frequencies
    features = []
    
    # First harmonic: 10-20 Hz (related to shaft rotation)
    mask1 = (frequency_vector > 10) & (frequency_vector < 20)
    features.append(np.mean(fft_positive[mask1]) if np.any(mask1) else 0.0)
    
    # Second harmonic: 90-100 Hz (BPFO region)
    mask2 = (frequency_vector > 90) & (frequency_vector < 100)
    features.append(np.mean(fft_positive[mask2]) if np.any(mask2) else 0.0)
    
    # Third harmonic: 126-136 Hz (BPFI region)
    mask3 = (frequency_vector > 126) & (frequency_vector < 136)
    features.append(np.mean(fft_positive[mask3]) if np.any(mask3) else 0.0)
    
    # Fourth harmonic: 20-30 Hz
    mask4 = (frequency_vector > 20) & (frequency_vector < 30)
    features.append(np.mean(fft_positive[mask4]) if np.any(mask4) else 0.0)
    
    return np.array(features, dtype=np.float32)


def extract_features_batch(signals: np.ndarray, sampling_rate: int = None) -> np.ndarray:
    """Extract features from multiple signals.
    
    Args:
        signals: 2D numpy array (n_samples, signal_length)
        sampling_rate: Sampling rate in Hz (uses FS default if not provided)
        
    Returns:
        2D numpy array (n_samples, n_features)
    """
    features = []
    for i in range(signals.shape[0]):
        features.append(extract_features_single(signals[i], sampling_rate))
    return np.array(features)


# =============================================================================
# Data Loading Functions
# =============================================================================

# Cache for loaded data
_cached_data = None


def _load_mat_data():
    """Load and cache the .mat data files."""
    global _cached_data
    
    if _cached_data is not None:
        return _cached_data
    
    # Load training data
    mat_train = scipy.io.loadmat(str(DATA_PATH / "data_train.mat"))
    mat_train_labels = scipy.io.loadmat(str(DATA_PATH / "data_train_labels.mat"))
    
    # Load test data
    mat_test = scipy.io.loadmat(str(DATA_PATH / "data_test.mat"))
    mat_test_labels = scipy.io.loadmat(str(DATA_PATH / "data_test_labels.mat"))
    
    # Extract and flatten signals
    data_train = mat_train['data_train'][0]
    data_train = np.array([a.flatten() for a in data_train])
    
    data_test = mat_test['data_test'][0]
    data_test = np.array([a.flatten() for a in data_test])
    
    # Extract labels (convert from 1-8 to 0-7 for PyTorch)
    labels_train = mat_train_labels['data_train_labels'].flatten() - 1
    labels_test = mat_test_labels['data_test_labels'].flatten() - 1
    
    # Extract features
    print("Extracting features from training data...")
    features_train = extract_features_batch(data_train)
    print("Extracting features from test data...")
    features_test = extract_features_batch(data_test)
    
    # Normalize features (z-score normalization)
    mean = features_train.mean(axis=0)
    std = features_train.std(axis=0) + 1e-8
    features_train = (features_train - mean) / std
    features_test = (features_test - mean) / std
    
    _cached_data = {
        'features_train': features_train,
        'labels_train': labels_train,
        'features_test': features_test,
        'labels_test': labels_test,
        'mean': mean,
        'std': std
    }
    
    return _cached_data


def _dirichlet_partition(labels: np.ndarray, num_partitions: int, alpha: float, seed: int = 42):
    """Partition data indices using Dirichlet distribution.
    
    Args:
        labels: Array of labels for each sample
        num_partitions: Number of partitions (clients)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
        
    Returns:
        List of index arrays, one per partition
    """
    np.random.seed(seed)
    
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    
    # Get indices for each class
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    
    # Initialize partition indices
    partition_indices = [[] for _ in range(num_partitions)]
    
    # For each class, distribute samples according to Dirichlet
    for c in range(n_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_partitions)
        
        # Calculate number of samples per partition for this class
        proportions = (proportions * len(indices)).astype(int)
        
        # Adjust to ensure all samples are assigned
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        # Assign indices to partitions
        start = 0
        for p in range(num_partitions):
            end = start + proportions[p]
            partition_indices[p].extend(indices[start:end].tolist())
            start = end
    
    # Shuffle each partition
    for p in range(num_partitions):
        np.random.shuffle(partition_indices[p])
        partition_indices[p] = np.array(partition_indices[p])
    
    return partition_indices


# Cache for partitioned indices
_partition_cache = None


def load_data(partition_id: int, num_partitions: int, alpha: float = None, sampling_rate: int = None):
    """Load partitioned bearing fault data for a specific client.
    
    Args:
        partition_id: ID of the partition/client (0 to num_partitions-1)
        num_partitions: Total number of partitions
        alpha: Dirichlet alpha parameter for non-IID partitioning
        sampling_rate: Sampling rate in Hz for feature extraction
        
    Returns:
        Tuple of (trainloader, testloader)
    """
    global _partition_cache
    alpha = alpha if alpha is not None else DIRICHLET_ALPHA
    
    # Load data
    data = _load_mat_data()
    features_train = data['features_train']
    labels_train = data['labels_train']
    
    # Create partitions if not cached
    if _partition_cache is None or _partition_cache['num_partitions'] != num_partitions:
        partition_indices = _dirichlet_partition(
            labels_train, num_partitions, alpha
        )
        _partition_cache = {
            'num_partitions': num_partitions,
            'indices': partition_indices
        }
    
    # Get indices for this partition
    indices = _partition_cache['indices'][partition_id]
    
    # Get data for this partition
    X = features_train[indices]
    y = labels_train[indices]
    
    # Split into train (80%) and validation (20%)
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=32)
    
    return trainloader, testloader


def load_test_data():
    """Load the centralized test dataset.
    
    Returns:
        DataLoader for test data
    """
    data = _load_mat_data()
    
    test_dataset = TensorDataset(
        torch.tensor(data['features_test'], dtype=torch.float32),
        torch.tensor(data['labels_test'], dtype=torch.long)
    )
    
    return DataLoader(test_dataset, batch_size=32)


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set.
    
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
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    running_loss = 0.0
    total_batches = 0
    
    for _ in range(epochs):
        for batch in trainloader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
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
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            
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
