import torch
import torch.nn as nn
from torchvision import models


class GradCAMReadyMobileNetV2(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, pretrained=True):
        super(GradCAMReadyMobileNetV2, self).__init__()

        if pretrained:
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            print("Loaded pre-trained MobileNetV2 weights.")
        else:
            self.model = models.mobilenet_v2(weights=None)
            print("Loaded MobileNetV2 without pre-trained weights.")

        # Store the original features and classifier directly.
        # We will split the 'features' part into two for Grad-CAM.
        self.features_until_target = nn.Sequential(
            *list(self.model.features.children())[:19])  # features[0] to features[18] (inclusive)
        self.features_after_target = nn.Sequential(*list(self.model.features.children())[
                                                    19:])  # features[19] onwards (usually empty or just the final conv_1x1 layer)

        # Explicitly define the global average pooling layer
        # MobileNetV2 uses AdaptiveAvgPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modify the first convolutional layer for grayscale input
        original_conv1 = self.features_until_target[0][0]  # Access the first layer of the first sub-block
        self.features_until_target[0][0] = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        print(f"Modified first conv layer to accept {input_channels} input channels.")

        # Modify the final classification layer
        # MobileNetV2's classifier is a Sequential module with a Dropout and a Linear layer.
        # It's usually self.model.classifier[1] that is the Linear layer.
        self.classifier = self.model.classifier
        num_ftrs = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(f"Modified final classifier layer to output {num_classes} classes.")

        # Define the target layer for Grad-CAM
        # This will be the last module in our 'features_until_target' sequence
        self.target_layer = self.features_until_target[-1]  # This is features[18]

        # Optional: Freeze features for initial training
        # If you freeze, make sure to freeze `self.features_until_target` and `self.features_after_target`
        # for param in self.features_until_target.parameters():
        #     param.requires_grad = False
        # for param in self.features_after_target.parameters():
        #     param.requires_grad = False
        # print("Froze all feature extractor layers.")

    def forward(self, x):
        # 1. Pass through features up to and including the target layer for Grad-CAM
        x = self.features_until_target(x)
        conv_features = x  # This is the output of features[18]

        # 2. Pass through any remaining features (if features[19] onwards exists and is not empty)
        # In standard MobileNetV2, features[19] is usually the final 1x1 conv layer before avgpool.
        x = self.features_after_target(x)

        # 3. Apply the global average pooling
        x = self.avgpool(x)

        # 4. Flatten and pass through the classifier
        x = torch.flatten(x, 1)
        predictions = self.classifier(x)

        return conv_features, predictions