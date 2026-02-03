import torch
import torch.nn as nn
from torchvision import models

class GradCAMReadyAlexNet(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, pretrained=True):
        super(GradCAMReadyAlexNet, self).__init__()

        # Load pre-trained AlexNet
        if pretrained:
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            print("Loaded pre-trained AlexNet weights.")
        else:
            self.model = models.alexnet(weights=None)
            print("Loaded AlexNet without pre-trained weights.")

        # Iterate through features to modify ReLU
        for module in self.model.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        print("Set inplace=False for all ReLU layers in features.")

        # Iterate through classifier to modify ReLU (if any exist, though less common here)
        for module in self.model.classifier.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        print("Set inplace=False for all ReLU layers in classifier.")



        # 1. Modify the first convolutional layer for grayscale input
        original_conv1 = self.model.features[0]
        self.model.features[0] = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        print(f"Modified first conv layer to accept {input_channels} input channels.")

        # 2. Modify the final classification layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        print(f"Modified final classifier layer to output {num_classes} classes.")

        # 3. Define the target layer for Grad-CAM
        self.target_layer = self.model.features[10]

        # Optional: Freeze features for initial training
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        # print("Froze AlexNet feature extractor layers (features.parameters).")


    def forward(self, x):
        conv_features = None
        for i, module in enumerate(self.model.features):
            x = module(x)
            if i == 10:
                conv_features = x

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        predictions = self.model.classifier(x)

        return conv_features, predictions