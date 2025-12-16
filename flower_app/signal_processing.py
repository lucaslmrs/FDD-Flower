"""Signal processing module for fault detection using Hilbert Transform and FFT.

This module implements signal processing techniques similar to the bearing fault
detection notebook, adapted for time-series sensor data from Pick-and-Place scenarios.
"""

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.fft import fft


def apply_hilbert_transform(signal):
    """Apply Hilbert transform to extract signal envelope.
    
    Args:
        signal: 1D numpy array representing the time-series signal
        
    Returns:
        Absolute value of the Hilbert transform (signal envelope)
    """
    return np.abs(hilbert(signal))


def centralize_signal(signal):
    """Centralize signal by removing its mean.
    
    Args:
        signal: 1D numpy array
        
    Returns:
        Centralized signal (zero mean)
    """
    return signal - np.mean(signal)


def compute_fft(signal, return_positive_half=True):
    """Compute Fast Fourier Transform of the signal.
    
    Args:
        signal: 1D numpy array
        return_positive_half: If True, return only positive frequencies (first half)
        
    Returns:
        Absolute value of FFT coefficients
    """
    fft_result = np.abs(fft(signal))
    
    if return_positive_half:
        # Return only positive frequencies (first half)
        n = len(fft_result) // 2
        return fft_result[:n]
    
    return fft_result


def generate_frequency_vector(signal_length, sampling_rate):
    """Generate frequency vector for FFT interpretation.
    
    Args:
        signal_length: Length of the signal (number of samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Frequency vector in Hz
    """
    N = signal_length
    T_total = N / sampling_rate
    d_f = 1 / T_total
    frequency_vector = np.arange(0, sampling_rate / 2, d_f)
    
    # Ensure same length as positive half of FFT
    return frequency_vector[:signal_length // 2]


def extract_harmonic_features(fft_magnitude, frequency_vector, harmonic_bands):
    """Extract harmonic features from FFT spectrum.
    
    Args:
        fft_magnitude: Magnitude of FFT coefficients
        frequency_vector: Corresponding frequency values in Hz
        harmonic_bands: List of tuples [(f_min1, f_max1), (f_min2, f_max2), ...]
                       defining frequency bands for each harmonic
        
    Returns:
        Dictionary with harmonic features
    """
    features = {}
    
    for idx, (f_min, f_max) in enumerate(harmonic_bands, start=1):
        # Extract magnitude in this frequency band
        mask = (frequency_vector >= f_min) & (frequency_vector <= f_max)
        
        if np.any(mask):
            # Use maximum amplitude in the band as the feature
            features[f'harmonic_{idx}'] = np.max(fft_magnitude[mask])
        else:
            features[f'harmonic_{idx}'] = 0.0
    
    return features


def process_signal_to_features(signal, sampling_rate, harmonic_bands):
    """Complete signal processing pipeline: Hilbert -> Centralize -> FFT -> Features.
    
    Args:
        signal: 1D numpy array of time-series data
        sampling_rate: Sampling rate in Hz
        harmonic_bands: List of frequency band tuples for harmonic extraction
        
    Returns:
        Dictionary of features extracted from the signal
    """
    # Step 1: Apply Hilbert transform to get envelope
    signal_hilbert = apply_hilbert_transform(signal)
    
    # Step 2: Centralize the envelope
    signal_centralized = centralize_signal(signal_hilbert)
    
    # Step 3: Compute FFT
    fft_magnitude = compute_fft(signal_centralized, return_positive_half=True)
    
    # Step 4: Generate frequency vector
    frequency_vector = generate_frequency_vector(len(fft_magnitude) * 2, sampling_rate)
    
    # Ensure frequency vector matches FFT length
    min_len = min(len(fft_magnitude), len(frequency_vector))
    fft_magnitude = fft_magnitude[:min_len]
    frequency_vector = frequency_vector[:min_len]
    
    # Step 5: Extract harmonic features
    features = extract_harmonic_features(fft_magnitude, frequency_vector, harmonic_bands)
    
    # Additional statistical features from the signal
    features['signal_mean'] = np.mean(signal)
    features['signal_std'] = np.std(signal)
    features['signal_max'] = np.max(signal)
    features['signal_min'] = np.min(signal)
    features['signal_rms'] = np.sqrt(np.mean(signal ** 2))
    
    # Features from envelope
    features['envelope_mean'] = np.mean(signal_hilbert)
    features['envelope_std'] = np.std(signal_hilbert)
    features['envelope_max'] = np.max(signal_hilbert)
    
    return features


def window_signal(signal, window_size_seconds, sampling_rate, overlap=0.0):
    """Divide signal into windows with optional overlap.
    
    Args:
        signal: 1D numpy array of time-series data
        window_size_seconds: Window size in seconds
        sampling_rate: Sampling rate in Hz
        overlap: Overlap fraction between windows (0.0 to 0.9)
        
    Returns:
        List of signal windows (numpy arrays)
    """
    window_size = int(window_size_seconds * sampling_rate)
    step = int(window_size * (1 - overlap))
    
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
    
    return windows


def process_dataframe_column(df, column_name, sampling_rate, harmonic_bands, 
                             window_size_seconds=None, overlap=0.0):
    """Process a DataFrame column containing time-series data.
    
    Args:
        df: pandas DataFrame
        column_name: Name of column containing the signal
        sampling_rate: Sampling rate in Hz
        harmonic_bands: List of frequency band tuples
        window_size_seconds: If provided, split signal into windows
        overlap: Overlap fraction for windowing
        
    Returns:
        DataFrame with extracted features
    """
    signal = df[column_name].values
    
    # Handle missing values
    signal = np.nan_to_num(signal, nan=0.0)
    
    if window_size_seconds is not None:
        # Process with windowing
        windows = window_signal(signal, window_size_seconds, sampling_rate, overlap)
        
        feature_list = []
        for window in windows:
            features = process_signal_to_features(window, sampling_rate, harmonic_bands)
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    else:
        # Process entire signal as one sample
        features = process_signal_to_features(signal, sampling_rate, harmonic_bands)
        return pd.DataFrame([features])


# Default harmonic bands (can be customized based on domain knowledge)
DEFAULT_HARMONIC_BANDS = [
    (10, 20),    # First harmonic
    (90, 100),   # Second harmonic
    (126, 136),  # Third harmonic
    (20, 30),    # Fourth harmonic
]
