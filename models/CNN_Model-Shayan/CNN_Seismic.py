# Seismic Data Processing using Convolutional Neural Networks (CNN)
# This script implements CNN models for seismic data analysis including:
# 1. Feature extraction from seismic images
# 2. Classification of seismic events and structures
# 3. Model export to ONNX format for cross-platform compatibility

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import os
import time

# Check and use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# DATA HANDLING
# =============================================================================

class SeismicDataset(Dataset):
    """
    Custom Dataset class for handling seismic data.

    This class prepares seismic data for input to CNN models by:
    - Converting data to PyTorch tensors
    - Adding channel dimension for CNN processing
    - Handling both 2D and 3D seismic data
    - Supporting data augmentation through transforms

    Parameters:
    -----------
    data : numpy.ndarray
        Seismic data samples (can be 2D or 3D)
    labels : numpy.ndarray, optional
        Class labels for supervised learning
    transform : callable, optional
        Optional transform to be applied to each sample
    """

    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Get data sample
        sample = self.data[idx]

        # Apply transformations if specified (e.g., augmentation)
        if self.transform:
            sample = self.transform(sample)

        # Convert numpy array to PyTorch tensor
        sample = torch.from_numpy(sample).float()

        # Add channel dimension for CNN processing
        if len(sample.shape) == 2:  # For 2D data (e.g., seismic section)
            sample = sample.unsqueeze(0)  # Add channel dimension [C, H, W]
        elif len(sample.shape) == 3:  # For 3D data (e.g., seismic volume)
            # Rearrange dimensions from [H, W, D] to [D, H, W] for PyTorch convention
            sample = sample.permute(2, 0, 1)  # [D, H, W] where D is treated as channels

        # Return sample with label if available
        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label).long()
            return sample, label
        else:
            return sample


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class SeismicCNN2D(nn.Module):
    """
    CNN Model for 2D seismic data processing.

    This model is designed for tasks like:
    - Fault detection
    - Seismic facies classification
    - Horizon tracking
    - Reservoir characterization

    Architecture:
    - Multiple convolutional layers for feature extraction
    - Max pooling for downsampling and feature selection
    - Dropout for regularization
    - Fully connected layers for classification

    Parameters:
    -----------
    num_classes : int
        Number of output classes for classification
    input_channels : int
        Number of input channels (default=1 for grayscale seismic data)
    """

    def __init__(self, num_classes=3, input_channels=1):
        super(SeismicCNN2D, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            # Input: [batch_size, input_channels, height, width]
            # Output: [batch_size, 16, height/2, width/2]
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # Batch normalization for training stability
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            # Output: [batch_size, 32, height/4, width/4]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            # Output: [batch_size, 64, height/8, width/8]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth convolutional block
            # Output: [batch_size, 128, height/16, width/16]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of features after convolution
        # This is important for connecting conv layers to fully connected layers
        # For an input of 128x128, after 4 max pooling layers, the size is reduced to 8x8
        self.feature_size = 128 * 8 * 8

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.features(x)

        # Flatten the feature maps
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)
        return x

    def get_feature_extractor(self):
        """
        Return the feature extraction part of the model.
        Useful for transfer learning and feature visualization.
        """
        return self.features


class SeismicCNN3D(nn.Module):
    """
    CNN Model for 3D seismic data processing.

    This model extends the 2D CNN to handle volumetric seismic data,
    which is crucial for complex subsurface feature analysis.

    Applications:
    - 3D fault detection
    - Salt body delineation
    - Stratigraphic feature extraction
    - Geobody identification

    Parameters:
    -----------
    num_classes : int
        Number of output classes for classification
    input_channels : int
        Number of input channels (default=1 for standard 3D seismic data)
    """

    def __init__(self, num_classes=3, input_channels=1):
        super(SeismicCNN3D, self).__init__()

        # Feature extraction layers using 3D convolutions
        self.features = nn.Sequential(
            # First 3D convolutional block
            # Input: [batch_size, input_channels, depth, height, width]
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Second 3D convolutional block
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Third 3D convolutional block
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Fourth 3D convolutional block
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # For a 64x64x64 input volume, after 4 max pooling layers, size is 4x4x4
        self.feature_size = 128 * 4 * 4 * 4

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction with 3D convolutions
        x = self.features(x)

        # Flatten the 3D feature maps
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)
        return x


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=25, scheduler=None, early_stopping_patience=5):
    """
    Train the CNN model with progress tracking and early stopping.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function (e.g., CrossEntropyLoss)
    optimizer : torch.optim.Optimizer
        Optimization algorithm (e.g., Adam)
    num_epochs : int
        Maximum number of training epochs
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping

    Returns:
    --------
    model : torch.nn.Module
        Trained model
    history : dict
        Training history (losses and accuracies)
    """
    # Move model to the selected device (GPU/CPU)
    model.to(device)

    # Initialize history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    no_improve_epochs = 0
    start_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Time':^7}")
    print("-" * 65)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ===== TRAINING PHASE =====
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Process batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # ===== VALIDATION PHASE =====
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0

        # No gradient calculation during validation (saves memory and computations)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Update learning rate if scheduler is provided
        if scheduler:
            scheduler.step(epoch_val_loss)  # For ReduceLROnPlateau

        # Calculate time elapsed for the epoch
        epoch_time = time.time() - epoch_start

        # Print progress
        print(
            f"{epoch + 1:^7} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^9.2f}% | {epoch_val_loss:^9.4f} | {epoch_val_acc:^8.2f}% | {epoch_time:^6.1f}s")

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_seismic_cnn_model.pth')
            print(f"Model improved - saved checkpoint")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Training complete
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Load the best model
    model.load_state_dict(torch.load('best_seismic_cnn_model.pth'))

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return model, history


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the trained model on test data and visualize results.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained neural network model
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    class_names : list
        Names of the classes for display in the confusion matrix

    Returns:
    --------
    metrics : dict
        Performance metrics (accuracy, precision, recall, F1-score)
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and true labels
    y_true = []
    y_pred = []

    # Collect predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate and print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Calculate accuracy
    accuracy = (y_pred == y_true).sum() / len(y_true)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

    return metrics


def extract_features(model, data_loader):
    """
    Extract features from the intermediate layers of the CNN.

    This is useful for:
    - Visualization of learned features
    - Dimensionality reduction
    - Transfer learning
    - Unsupervised learning on extracted features

    Parameters:
    -----------
    model : torch.nn.Module
        Trained CNN model
    data_loader : torch.utils.data.DataLoader
        DataLoader for the data

    Returns:
    --------
    features : numpy.ndarray
        Extracted features
    labels : numpy.ndarray
        Corresponding labels
    """
    # Set model to evaluation mode
    model.eval()

    # Lists to store features and labels
    feature_list = []
    label_list = []

    # Extract features from the last layer before classification
    feature_extractor = model.get_feature_extractor() if hasattr(model, 'get_feature_extractor') else model.features

    # Disable gradient calculations
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move inputs to the device
            inputs = inputs.to(device)

            # Extract features
            outputs = feature_extractor(inputs)

            # Flatten the features
            features = outputs.view(outputs.size(0), -1).cpu().numpy()

            # Store features and labels
            feature_list.extend(features)
            label_list.extend(targets.numpy())

    return np.array(feature_list), np.array(label_list)


def export_to_onnx(model, input_shape, export_path='seismic_cnn_model.onnx'):
    """
    Export the PyTorch model to ONNX format.

    ONNX (Open Neural Network Exchange) is an open format for representing
    deep learning models. Models from PyTorch can be exported to ONNX and then
    converted to different frameworks such as TensorFlow, CoreML, etc.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model to export
    input_shape : tuple
        The shape of the input tensor (batch_size, channels, height, width)
    export_path : str
        Path where the ONNX model will be saved

    Returns:
    --------
    success : bool
        True if export was successful
    """
    try:
        # Create a dummy input tensor based on the specified shape
        dummy_input = torch.randn(input_shape, device=device)

        # Set the model to evaluation mode
        model.eval()

        # Export the model
        torch.onnx.export(
            model,  # model being exported
            dummy_input,  # model input
            export_path,  # where to save the model
            export_params=True,  # store the trained parameter weights
            opset_version=12,  # ONNX version to use
            do_constant_folding=True,  # optimization: fold constants
            input_names=['input'],  # model's input names
            output_names=['output'],  # model's output names
            dynamic_axes={
                'input': {0: 'batch_size'},  # variable batch size
                'output': {0: 'batch_size'}
            }
        )

        print(f"Model successfully exported to ONNX format: {export_path}")

        # Verify the ONNX model
        try:
            import onnx
            # Load the ONNX model
            onnx_model = onnx.load(export_path)
            # Check that the model is well-formed
            onnx.checker.check_model(onnx_model)
            print("ONNX model is well-formed and valid")
        except ImportError:
            print("ONNX package not installed. Skipping model verification.")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")

        return True

    except Exception as e:
        print(f"Failed to export model to ONNX: {e}")
        return False


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_seismic_data(n_samples=1000, width=128, height=128, n_classes=3, noise_level=0.2):
    """
    Generate synthetic seismic data for demonstration and testing.

    This function creates artificial seismic images with different patterns
    simulating geological structures like faults, folds, and reservoirs.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    width : int
        Width of each sample
    height : int
        Height of each sample
    n_classes : int
        Number of classes to generate
    noise_level : float
        Amount of random noise to add (0.0 to 1.0)

    Returns:
    --------
    X : numpy.ndarray
        Generated seismic data with shape (n_samples, width, height)
    y : numpy.ndarray
        Class labels with shape (n_samples,)
    """
    print(f"Generating {n_samples} synthetic seismic samples...")

    # Initialize data arrays
    X = np.zeros((n_samples, width, height))

    # Add random noise base to all samples
    for i in range(n_samples):
        X[i] = np.random.randn(width, height) * noise_level

    # Generate labels randomly
    y = np.random.randint(0, n_classes, size=n_samples)

    # Add distinct features for each class
    for i in range(n_samples):
        # Common structural elements: add horizontal layers to simulate stratigraphy
        num_layers = np.random.randint(5, 15)
        layer_positions = np.sort(np.random.choice(range(height), size=num_layers, replace=False))
        layer_thicknesses = np.random.randint(2, 6, size=num_layers)

        for j, pos in enumerate(layer_positions):
            thickness = layer_thicknesses[j]
            amplitude = np.random.uniform(0.5, 1.5) * (-1 if j % 2 == 0 else 1)
            layer_start = max(0, pos - thickness // 2)
            layer_end = min(height, pos + thickness // 2 + 1)
            X[i, :, layer_start:layer_end] += amplitude

        # Add class-specific features
        if y[i] == 0:  # Class 0: Fault structures
            # Add fault line with displacement
            fault_pos = np.random.randint(width // 4, 3 * width // 4)
            fault_angle = np.random.uniform(-0.2, 0.2)  # Slight angle
            displacement = np.random.randint(5, 15)

            for j in range(height):
                fault_x = min(width - 1, max(0, int(fault_pos + j * fault_angle)))
                # Create displacement in layers
                if fault_x > 0 and fault_x < width - 1:
                    X[i, fault_x:, j] = np.roll(X[i, fault_x:, j], displacement)

            # Add high-amplitude anomaly along fault
            for j in range(height):
                fault_x = min(width - 1, max(0, int(fault_pos + j * fault_angle)))
                if 0 <= fault_x < width:
                    X[i, max(0, fault_x - 1):min(width, fault_x + 2), j] += 2.0

        elif y[i] == 1:  # Class 1: Fold structures
            # Create sinusoidal fold pattern
            amplitude = np.random.uniform(10, 20)
            frequency = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2 * np.pi)

            for x in range(width):
                fold = amplitude * np.sin(frequency * x / width * 2 * np.pi + phase)
                for j in range(height):
                    # Shift the layers according to the fold pattern
                    target_j = int(j + fold)
                    if 0 <= target_j < height:
                        X[i, x, j] = X[i, x, target_j]

        elif y[i] == 2:  # Class 2: Reservoir structures
            # Add elliptical bright spot (potential reservoir)
            center_x = width // 2 + np.random.randint(-width // 4, width // 4)
            center_y = height // 2 + np.random.randint(-height // 4, height // 4)
            radius_x = width // np.random.uniform(4, 8)
            radius_y = height // np.random.uniform(8, 16)

            # Create amplitude anomaly (bright spot)
            for x in range(width):
                for j in range(height):
                    distance = ((x - center_x) / radius_x) ** 2 + ((j - center_y) / radius_y) ** 2
                    if distance <= 1:
                        X[i, x, j] += 2.5 * (1 - distance)  # Higher amplitude in center

            # Add flat reflectors beneath (gas-water contact)
            contact_y = int(center_y + radius_y) + np.random.randint(2, 5)
            if contact_y < height:
                X[i, :, contact_y] += 3.0

    print(f"Synthetic data generation complete.")
    return X, y


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the complete workflow:
    1. Generate/load seismic data
    2. Preprocess and prepare DataLoaders
    3. Create and train CNN model
    4. Evaluate model performance
    5. Visualize features
    6. Export model to ONNX format
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration parameters
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    class_names = ['Fault', 'Fold', 'Reservoir']

    # Create output directory for model artifacts
    os.makedirs('seismic_model_output', exist_ok=True)

    # Generate synthetic seismic data (replace with real data in production)
    print("Generating synthetic seismic data...")
    X, y = generate_synthetic_seismic_data(n_samples=500, width=128, height=128, n_classes=3)
    print(f"Data shape: {X.shape}, Label shape: {y.shape}")

    # Visualize sample data from each class
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sample_idx = np.where(y == i)[0][0]  # Get first sample of each class
        plt.imshow(X[sample_idx], cmap='seismic')
        plt.title(f'Class: {class_names[i]}')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('seismic_model_output/sample_data.png')
    plt.show()

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Create datasets and dataloaders
    train_dataset = SeismicDataset(X_train, y_train)
    val_dataset = SeismicDataset(X_val, y_val)
    test_dataset = SeismicDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model = SeismicCNN2D(num_classes=len(class_names))
    print("Model architecture:")
    print(model)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Train the model
    print("\nTraining model...")
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=7
    )

    # Evaluate the model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(trained_model, test_loader, class_names)

    # Save test metrics to a file
    import json
    with open('seismic_model_output/test_metrics.json', 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'class_reports': metrics['report']
        }, f, indent=4)

    # Extract and visualize features
    print("\nExtracting features from the trained model...")
    features, labels = extract_features(trained_model, test_loader)
    print(f"Extracted features shape: {features.shape}")

    # Visualize features with dimensionality reduction (PCA)
    print("Visualizing features with PCA...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']

    for i, (color, marker) in enumerate(zip(colors, markers)):
        mask = labels == i
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            color=color,
            marker=marker,
            alpha=0.7,
            s=70,
            label=class_names[i]
        )

    plt.title('Feature Visualization (PCA)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('seismic_model_output/feature_visualization.png')
    plt.show()

    # Save model in PyTorch format
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'history': history
    }, 'seismic_model_output/seismic_cnn_model.pt')
    print("Model saved in PyTorch format: seismic_model_output/seismic_cnn_model.pt")

    # Export model to ONNX format
    print("\nExporting model to ONNX format...")
    # For 2D CNN, input shape is [batch_size, channels, height, width]
    input_shape = (1, 1, 128, 128)  # Batch size of 1, 1 channel, 128x128 image
    export_success = export_to_onnx(
        trained_model,
        input_shape,
        export_path='seismic_model_output/seismic_cnn_model.onnx'
    )

    if export_success:
        print("\nONNX export successful. The model can now be used with:")
        print("- ONNX Runtime")
        print("- TensorFlow (via ONNX-TF)")
        print("- TensorRT")
        print("- OpenVINO")
        print("- Windows ML")
        print("- And other ONNX-compatible frameworks")

    # Function to demonstrate inference with ONNX model
    def test_onnx_inference():
        """
        Test the exported ONNX model with sample data.
        """
        try:
            import onnxruntime as ort

            print("\nTesting ONNX model inference...")

            # Get a sample from test data
            sample_data, sample_label = test_dataset[0]
            sample_data = sample_data.unsqueeze(0)  # Add batch dimension

            # Create ONNX Runtime session
            ort_session = ort.InferenceSession('seismic_model_output/seismic_cnn_model.onnx')

            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: sample_data.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Get prediction
            ort_predicted = np.argmax(ort_outputs[0], axis=1)[0]

            print(f"Sample true label: {class_names[sample_label]}")
            print(f"ONNX model prediction: {class_names[ort_predicted]}")

            # Compare with PyTorch model
            with torch.no_grad():
                torch_output = trained_model(sample_data)
                torch_predicted = torch.argmax(torch_output, dim=1).item()

            print(f"PyTorch model prediction: {class_names[torch_predicted]}")

            if ort_predicted == torch_predicted:
                print("✓ ONNX and PyTorch models produce the same prediction")
            else:
                print("✗ ONNX and PyTorch models produce different predictions")

        except ImportError:
            print("\nONNXRuntime not installed. Skipping ONNX inference test.")
            print("To test ONNX inference, install onnxruntime: pip install onnxruntime")

    # Test ONNX inference if export was successful
    if export_success:
        test_onnx_inference()

    print("\n" + "=" * 50)
    print("Seismic CNN Training and Export Completed")
    print("=" * 50)
    print(f"Model files saved in directory: {os.path.abspath('seismic_model_output')}")


if __name__ == "__main__":
    main()