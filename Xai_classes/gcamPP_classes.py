import torch
import torch.nn.functional as F
import numpy as np

class GradCAMPlusPlus:
    """
     Implements standard GradCAMPlusPlus for a pytorch model
        Arg:
            model: pytorch model applying xai on
            target_layer: layer whose logits are used to create the xai heatmap
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        model_output = self.model(input_tensor)
        if isinstance(model_output, tuple):
            logits = model_output[1]
        else:
            logits = model_output

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        activations = self.activations  # [B, C, H, W]
        gradients = self.gradients      # [B, C, H, W]

        # Compute alpha weights (Grad-CAM++ specific)
        with torch.no_grad():
            grads_pow_2 = gradients ** 2
            grads_pow_3 = grads_pow_2 * gradients

            global_sum = torch.sum(activations, dim=(2, 3), keepdim=True)
            eps = 1e-8

            alpha_numer = grads_pow_2
            alpha_denom = 2 * grads_pow_2 + global_sum * grads_pow_3
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.tensor(eps).to(alpha_denom.device))

            alphas = alpha_numer / alpha_denom
            positive_gradients = F.relu(score[:, None, None, None] * gradients)
            weights = (alphas * positive_gradients).sum(dim=(2, 3), keepdim=True)

        # Compute final Grad-CAM++ heatmap
        weighted_activations = weights * activations
        cam = weighted_activations.sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize and return as NumPy array
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        return heatmap
