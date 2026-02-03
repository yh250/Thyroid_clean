import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMReadyCNN(nn.Module):
    """
    Returns both the intermediate feature maps from the last convolutional layer
    and the final classification predictions.
    """

    def __init__(self, input_channels, num_classes):
        """
        Initializes the GradCAMReadyCNN model.

        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes for classification.
        """
        super(GradCAMReadyCNN, self).__init__()

        # First Convolutional Block
        # Input: (batch_size, input_channels, 256, 256)
        # Output: (batch_size, 32, 254, 254)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        # Output: (batch_size, 32, 127, 127)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        # Output: (batch_size, 64, 125, 125)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        # Output: (batch_size, 64, 62, 62) (floor division for pooling)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Convolutional Block (Target for Grad-CAM)
        # Output: (batch_size, 128, 60, 60)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU()
        # Output after this conv layer is what we'll use for Grad-CAM features.

        # This pooling layer continues the main path after conv3
        # Output: (batch_size, 128, 30, 30) (floor division for pooling)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        # Input to flatten: (batch_size, 128, 30, 30)
        # Output: (batch_size, 128 * 30 * 30) = (batch_size, 115200)
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout rate 0.3 as in your Keras model
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU()

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple: A tuple containing:
                - conv_features (torch.Tensor): Feature maps from the last convolutional layer (conv3).
                - predictions (torch.Tensor): Final classification predictions (logits before softmax).
        """
        # First convolutional block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Third convolutional block (features for Grad-CAM are captured here)
        x = self.conv3(x)
        x = self.relu3(x)
        conv_features = x  # Capture the output of conv3 (before pool3)

        # Continue with the rest of the network
        x = self.pool3(x)

        # Flatten and Dense layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)  # Dropout is active during training, inactive during eval()
        x = self.fc2(x)
        x = self.relu_fc2(x)

        predictions = self.fc_out(x)

        return conv_features, predictions


"""if __name__ == '__main__':
    # Example usage: Test the model architecture
    input_channels = 1  # For grayscale images (256, 256, 1)
    num_classes = 4

    model = GradCAMReadyCNN(input_channels, num_classes)

    print("--- Model Architecture ---")
    print(model)

    # Create a dummy input tensor: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, input_channels, 256, 256)

    # Perform a forward pass
    conv_out, pred_out = model(dummy_input)

    print(f"\nShape of conv_features output: {conv_out.shape}")
    print(f"Shape of predictions output: {pred_out.shape}")
    print(f"Predictions (first sample): {pred_out[0]}")"""
