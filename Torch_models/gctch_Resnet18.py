import torch
import torch.nn as nn
from torchvision import models

class GradCAMReadyResNet18(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 1, pretrained: bool = True):
        super(GradCAMReadyResNet18, self).__init__()

        # Load pre-trained ResNet18
        if pretrained:
            # Using DEFAULT weights which are ImageNet1K_V1
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("Loaded pre-trained ResNet18 weights.")
        else:
            self.model = models.resnet18(weights=None)
            print("Loaded ResNet18 without pre-trained weights.")

        # ResNet uses ReLU in its basic blocks. We need to iterate through them.
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        print("Set inplace=False for all ReLU layers in the model.")

        # 1. Modify the first convolutional layer for custom input channels
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        print(f"Modified first conv layer to accept {input_channels} input channels.")

        # If loading pretrained weights and converting to grayscale,
        # initialize the new conv1 weights by averaging the original 3 channels.
        if pretrained and input_channels == 1:
            # Original conv1 weights were [out_channels, 3, kH, kW]
            # New conv1 weights are [out_channels, 1, kH, kW]
            # Sum across the input channels (dim=1) and keep the dimension for broadcasting
            # This averages the RGB channel weights into a single grayscale weight.
            with torch.no_grad(): # Perform operation without tracking gradients
                self.model.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
            print("Initialized new conv1 weights from pre-trained RGB by averaging for grayscale.")
        elif not pretrained:
            # If not using pretrained weights, use default Kaiming initialization
            nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')


        # 2. Modify the final classification layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"Modified final classifier layer to output {num_classes} classes.")

        # 3. Define the target layer for Grad-CAM
        # For ResNet18, `layer4` is the last convolutional block whose output we want.
        self.target_layer = self.model.layer4
        print(f"Set target layer for Grad-CAM to: {self.target_layer.__class__.__name__} (self.model.layer4).")


        # Optional: Freeze features for initial training (uncomment if needed)
        # for param in self.model.conv1.parameters(): # Freeze conv1 if just adapted
        #     param.requires_grad = False
        # for param in self.model.bn1.parameters():
        #     param.requires_grad = False
        # for param in self.model.layer1.parameters(): # Freeze block 1
        #     param.requires_grad = False
        # for param in self.model.layer2.parameters(): # Freeze block 2
        #     param.requires_grad = False
        # print("Froze specific ResNet layers for fine-tuning (if uncommented).")


    def forward(self, x):
        # We need to manually pass through the layers to get the output of `target_layer`
        # which is self.model.layer4

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # The output of layer4 is our `conv_features`
        conv_features = self.model.layer4(x)

        # The rest of the forward pass for classification
        x = self.model.avgpool(conv_features) # Use conv_features for pooling
        x = torch.flatten(x, 1)
        predictions = self.model.fc(x)

        return conv_features, predictions