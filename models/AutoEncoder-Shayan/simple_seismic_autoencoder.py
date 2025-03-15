"""
Seismic Data Autoencoder for Noise Reduction
------------------------------------------
This module implements a simple autoencoder for seismic data processing using scikit-learn.
It supports both SEG-Y and CSV input formats, as well as synthetic data generation.

Key Features:
- SEG-Y and CSV file support
- Synthetic seismic data generation
- Noise reduction using autoencoder
- Data visualization and analysis
- Standardized data preprocessing
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
import segyio
import warnings

class SimpleSeismicAutoencoder:
    """
    A simple autoencoder implementation for seismic data processing.
    
    This class implements an autoencoder using two MLPRegressor networks:
    1. Encoder: Compresses the input data to a lower-dimensional latent space
    2. Decoder: Reconstructs the original data from the latent space
    
    The architecture uses ReLU activation and Adam optimizer for both networks.
    """
    
    def __init__(self, trace_length, latent_dim=32):
        """
        Initialize the Simple Seismic Autoencoder
        
        Parameters:
        -----------
        trace_length : int
            Length of seismic traces (number of samples per trace)
        latent_dim : int
            Dimension of the latent space (compressed representation)
        """
        self.trace_length = trace_length
        self.latent_dim = latent_dim
        # StandardScaler for normalizing input data
        self.scaler = StandardScaler()
        
        # Create encoder network
        # Architecture: input(trace_length) -> 128 -> 64 -> latent_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(128, 64, latent_dim),
            activation='relu',  # ReLU activation for non-linearity
            solver='adam',      # Adam optimizer for better convergence
            max_iter=1000,      # Maximum training iterations
            verbose=True,       # Show training progress
            tol=1e-5           # Convergence tolerance
        )
        
        # Create decoder network
        # Architecture: input(latent_dim) -> 64 -> 128 -> trace_length
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(64, 128, trace_length),
            activation='relu',
            solver='adam',
            max_iter=1000,
            verbose=True,
            tol=1e-5
        )

def read_segy_file(file_path):
    """
    Read SEG-Y file and return traces as numpy array.
    
    This function handles SEG-Y file reading using segyio library:
    1. Opens the file in read mode ignoring geometry
    2. Reads all traces into a numpy array
    3. Normalizes the data to [-1, 1] range
    
    Parameters:
    -----------
    file_path : str
        Path to the SEG-Y file
        
    Returns:
    --------
    numpy.ndarray
        Normalized seismic traces data, shape (n_traces, n_samples)
    
    Raises:
    -------
    Exception
        If there's an error reading the SEG-Y file
    """
    try:
        with segyio.open(file_path, 'r', ignore_geometry=True) as segy:
            # Get dimensions of the data
            n_traces = segy.tracecount
            n_samples = len(segy.samples)
            
            # Allocate memory for all traces
            data = np.zeros((n_traces, n_samples))
            # Read each trace
            for i in range(n_traces):
                data[i, :] = segy.trace[i]
                
        # Normalize data to [-1, 1] range
        data = data / np.max(np.abs(data))
        return data
    except Exception as e:
        raise Exception(f"Error reading SEG-Y file: {str(e)}")

def read_csv_file(file_path):
    """
    Read CSV file containing seismic traces.
    
    This function handles CSV file reading:
    1. Reads the CSV file using pandas
    2. Automatically detects trace orientation
    3. Transposes data if needed (ensures traces are in columns)
    4. Normalizes the data to [-1, 1] range
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    numpy.ndarray
        Normalized seismic traces data, shape (n_traces, n_samples)
    
    Raises:
    -------
    Exception
        If there's an error reading the CSV file
    """
    try:
        # Read CSV file into pandas DataFrame
        data = pd.read_csv(file_path)
        
        # Convert to numpy array
        data = data.values
        
        # Ensure traces are in columns (more samples than traces)
        if data.shape[0] < data.shape[1]:
            data = data.T
            
        # Normalize data to [-1, 1] range
        data = data / np.max(np.abs(data))
        return data
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

def generate_synthetic_seismic(num_traces=100, trace_length=1000):
    """
    Generate synthetic seismic data using Ricker wavelets.
    
    This function creates synthetic seismic traces by:
    1. Generating random reflection points
    2. Creating Ricker wavelets at those points
    3. Combining wavelets to form traces
    4. Normalizing the final data
    
    Parameters:
    -----------
    num_traces : int
        Number of traces to generate
    trace_length : int
        Length of each trace (number of samples)
        
    Returns:
    --------
    numpy.ndarray
        Synthetic seismic data, shape (num_traces, trace_length)
    """
    # Create time axis
    t = np.linspace(0, 1, trace_length)
    # Initialize data array
    data = np.zeros((num_traces, trace_length))
    
    for i in range(num_traces):
        # Generate random number of reflections (3-5)
        num_reflections = np.random.randint(3, 6)
        # Random reflection times between 0.1 and 0.9
        reflection_times = np.random.uniform(0.1, 0.9, num_reflections)
        # Random amplitudes between 0.5 and 1.0
        amplitudes = np.random.uniform(0.5, 1.0, num_reflections)
        
        # Create wavelets at reflection times
        for j in range(num_reflections):
            t0 = reflection_times[j]
            # Ricker wavelet formula
            # Central frequency: 30 Hz
            wavelet = amplitudes[j] * (1 - 2*(np.pi*30*(t-t0))**2) * np.exp(-(np.pi*30*(t-t0))**2)
            data[i, :] += wavelet
    
    # Normalize data to [-1, 1] range
    data = data / np.max(np.abs(data))
    return data

def add_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to seismic data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input seismic data
    noise_level : float
        Standard deviation of the Gaussian noise
        
    Returns:
    --------
    numpy.ndarray
        Noisy seismic data
    """
    # Generate Gaussian noise with specified level
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def plot_results(original, noisy, denoised, index=0):
    """
    Plot comparison of original, noisy, and denoised data.
    
    Creates a figure with three subplots showing:
    1. Original seismic trace
    2. Noisy trace
    3. Denoised trace
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original seismic data
    noisy : numpy.ndarray
        Noisy version of the data
    denoised : numpy.ndarray
        Denoised version of the data
    index : int
        Index of the trace to plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot original data
    plt.subplot(131)
    plt.plot(original[index])
    plt.title('Original')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot noisy data
    plt.subplot(132)
    plt.plot(noisy[index])
    plt.title('Noisy')
    plt.xlabel('Sample')
    
    # Plot denoised data
    plt.subplot(133)
    plt.plot(denoised[index])
    plt.title('Denoised')
    plt.xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png')
    plt.close()

def process_seismic_data(data_path, noise_level=0.2, use_synthetic=False):
    """
    Main function for processing seismic data.
    
    This function handles the complete processing pipeline:
    1. Data loading/generation
    2. Autoencoder creation and training
    3. Noise addition and denoising
    4. Performance evaluation
    5. Result visualization and saving
    
    Parameters:
    -----------
    data_path : str
        Path to the input file (SEG-Y or CSV) or None for synthetic data
    noise_level : float
        Level of noise to add (default: 0.2)
    use_synthetic : bool
        Whether to use synthetic data instead of file input
    """
    print("Starting seismic autoencoder processing...")
    
    # 1. Load or generate data
    if use_synthetic:
        print("\n1. Generating synthetic data...")
        trace_length = 1000
        num_traces = 100
        data = generate_synthetic_seismic(num_traces, trace_length)
        print(f"Generated data shape: {data.shape}")
    else:
        print("\n1. Loading data from file...")
        # Determine file type and read accordingly
        if data_path.lower().endswith(('.sgy', '.segy')):
            data = read_segy_file(data_path)
        elif data_path.lower().endswith('.csv'):
            data = read_csv_file(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide SEG-Y or CSV file.")
        print(f"Loaded data shape: {data.shape}")
        
    trace_length = data.shape[1]
    
    # 2. Create and train autoencoder
    print("\n2. Creating and training autoencoder...")
    autoencoder = SimpleSeismicAutoencoder(trace_length)
    
    # Scale the data using StandardScaler
    scaled_data = autoencoder.scaler.fit_transform(data)
    
    # Train encoder network
    print("Training encoder...")
    encoded = autoencoder.encoder.fit(scaled_data, scaled_data).predict(scaled_data)
    
    # Train decoder network
    print("Training decoder...")
    decoded = autoencoder.decoder.fit(encoded, scaled_data).predict(encoded)
    
    # 3. Test denoising capabilities
    print("\n3. Testing denoising capabilities...")
    # Add noise to original data
    noisy_data = add_noise(data, noise_level=noise_level)
    scaled_noisy = autoencoder.scaler.transform(noisy_data)
    
    # Denoise using trained autoencoder
    encoded_noisy = autoencoder.encoder.predict(scaled_noisy)
    denoised_scaled = autoencoder.decoder.predict(encoded_noisy)
    denoised = autoencoder.scaler.inverse_transform(denoised_scaled)
    
    # Ensure numerical stability
    denoised = np.clip(denoised, -1, 1)
    
    # Calculate and print performance metrics
    original_noise_mse = np.mean((data - noisy_data) ** 2)
    denoised_mse = np.mean((data - denoised) ** 2)
    print(f"Original noise MSE: {original_noise_mse:.6f}")
    print(f"After denoising MSE: {denoised_mse:.6f}")
    
    if denoised_mse < original_noise_mse:
        improvement = (original_noise_mse - denoised_mse) / original_noise_mse * 100
        print(f"Improvement: {improvement:.2f}%")
    else:
        print("No improvement in denoising")
    
    # 4. Visualize results
    print("\n4. Plotting results...")
    plot_results(data, noisy_data, denoised)
    print("Results saved as 'denoising_results.png'")
    
    # 5. Save processed data
    print("\n5. Saving example data...")
    np.savez('example_results.npz', 
             original=data, 
             noisy=noisy_data, 
             denoised=denoised)
    print("Example data saved as 'example_results.npz'")
    
    print("\nAll processing completed successfully!")

if __name__ == "__main__":
    # Set up command-line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description='Process seismic data with autoencoder')
    parser.add_argument('--input', type=str, help='Path to input file (SEG-Y or CSV)')
    parser.add_argument('--noise', type=float, default=0.2, help='Noise level (default: 0.2)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of file input')
    
    args = parser.parse_args()
    
    # Ensure either input file or synthetic data is specified
    if not args.synthetic and not args.input:
        parser.error("Either --input or --synthetic must be specified")
    
    # Process the data with specified parameters
    process_seismic_data(args.input, args.noise, args.synthetic) 