"""
Seismic Data Autoencoder for Noise Reduction (ONNX Optimized)
------------------------------------------------------------
Processes SEG-Y seismic data and exports to ONNX format.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import segyio
import os
import logging
import sys
import skl2onnx
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SeismicAutoencoderONNX:
    """Optimized seismic autoencoder with ONNX deployment."""

    def __init__(self, trace_length, latent_dim=32):
        self.trace_length = trace_length
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()

        # Encoder: Compress to latent_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            tol=1e-5,
            verbose=False
        )

        # Decoder: Reconstruct to trace_length
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(64, 128),
            activation='relu',
            solver='adam',
            max_iter=500,
            tol=1e-5,
            verbose=False
        )
        self.n_traces = None  # To store number of traces for reshaping

    def train(self, data):
        """Train the autoencoder with explicit latent space compression."""
        self.n_traces = data.shape[0]  # Store number of traces
        scaled_data = self.scaler.fit_transform(data)

        # Step 1: Train encoder to compress to latent_dim
        logger.info("Training encoder...")
        latent_target = np.random.randn(self.n_traces, self.latent_dim)  # Shape: (n_traces, latent_dim)
        self.encoder.fit(scaled_data, latent_target)
        encoded = self.encoder.predict(scaled_data)

        # Ensure encoded shape is (n_traces, latent_dim)
        if encoded.shape != (self.n_traces, self.latent_dim):
            logger.info(f"Reshaping encoder output from {encoded.shape} to ({self.n_traces}, {self.latent_dim})")
            encoded = encoded.reshape(self.n_traces, self.latent_dim)

        # Step 2: Train decoder to reconstruct original data from latent space
        logger.info("Training decoder...")
        self.decoder.fit(encoded, scaled_data)

        return self

    def convert_to_onnx(self, output_dir='onnx_models'):
        """Convert models to ONNX format with explicit shape handling."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        encoder_input = [('input', FloatTensorType([None, self.trace_length]))]
        decoder_input = [('input', FloatTensorType([None, self.latent_dim]))]

        try:
            logger.info("Converting encoder to ONNX...")
            encoder_onnx = skl2onnx.convert_sklearn(
                self.encoder, "seismic_encoder",
                encoder_input,
                target_opset=12
            )
            logger.info("Converting decoder to ONNX...")
            decoder_onnx = skl2onnx.convert_sklearn(
                self.decoder, "seismic_decoder",
                decoder_input,
                target_opset=12
            )

            encoder_path = os.path.join(output_dir, 'encoder.onnx')
            decoder_path = os.path.join(output_dir, 'decoder.onnx')

            with open(encoder_path, 'wb') as f:
                f.write(encoder_onnx.SerializeToString())
            with open(decoder_path, 'wb') as f:
                f.write(decoder_onnx.SerializeToString())

            logger.info(f"ONNX models saved: {encoder_path}, {decoder_path}")
            return encoder_path, decoder_path
        except Exception as e:
            logger.error(f"ONNX conversion failed: {str(e)}")
            raise

    def predict(self, input_data, encoder_path, decoder_path):
        """Perform inference using ONNX models with shape correction."""
        try:
            scaled_data = self.scaler.transform(input_data)

            encoder_session = rt.InferenceSession(encoder_path)
            enc_input = encoder_session.get_inputs()[0].name
            enc_output = encoder_session.get_outputs()[0].name
            encoded = encoder_session.run([enc_output], {enc_input: scaled_data.astype(np.float32)})[0]

            # Ensure encoded shape matches (n_traces, latent_dim)
            if encoded.shape != (self.n_traces, self.latent_dim):
                logger.info(f"Reshaping encoded output from {encoded.shape} to ({self.n_traces}, {self.latent_dim})")
                encoded = encoded.reshape(self.n_traces, self.latent_dim)

            decoder_session = rt.InferenceSession(decoder_path)
            dec_input = decoder_session.get_inputs()[0].name
            dec_output = decoder_session.get_outputs()[0].name
            decoded_scaled = decoder_session.run([dec_output], {dec_input: encoded.astype(np.float32)})[0]

            # Ensure decoded shape matches original data shape (n_traces, trace_length)
            if decoded_scaled.shape != (self.n_traces, self.trace_length):
                logger.info(f"Reshaping decoded output from {decoded_scaled.shape} to ({self.n_traces}, {self.trace_length})")
                decoded_scaled = decoded_scaled.reshape(self.n_traces, self.trace_length)

            return self.scaler.inverse_transform(decoded_scaled)
        except Exception as e:
            logger.error(f"ONNX inference failed: {str(e)}")
            raise

def read_segy_file(file_path):
    """Efficiently read SEG-Y file."""
    try:
        with segyio.open(file_path, 'r', ignore_geometry=True) as segy:
            n_traces = segy.tracecount
            n_samples = len(segy.samples)
            data = np.zeros((n_traces, n_samples), dtype=np.float32)
            for i in range(n_traces):
                data[i] = segy.trace[i]
        data /= np.max(np.abs(data))  # Normalize
        return data
    except Exception as e:
        logger.error(f"SEG-Y read failed: {str(e)}")
        raise

def process_seismic_data(data_path):
    """Process seismic data from the specified SEG-Y file."""
    logger.info("Starting seismic data processing...")

    # Load data
    data = read_segy_file(data_path)
    logger.info(f"Data loaded: {data.shape}")

    # Train autoencoder
    autoencoder = SeismicAutoencoderONNX(data.shape[1]).train(data)

    # Convert to ONNX
    encoder_path, decoder_path = autoencoder.convert_to_onnx()

    # Perform inference
    logger.info("Performing ONNX inference...")
    denoised = autoencoder.predict(data, encoder_path, decoder_path)

    # Save results
    np.savez('processed_data.npz', original=data, denoised=denoised)
    logger.info("Results saved as 'processed_data.npz'")

    logger.info("Processing completed successfully!")

if __name__ == "__main__":
    # Hardcode the data path
    DATA_PATH = "D:\Smart-exploration-in-the-oil-and-gas-industry\datasets\Europe\SEAM_Den_Elastic_N23900.sgy"

    try:
        process_seismic_data(DATA_PATH)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)