import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Implements standard Grad-CAM for a PyTorch model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        # Using lists to store activations and gradients if multiple hooks fire or for clarity
        self.activations = []
        self.gradients = []

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # For GradCAM, we can detach the activations as we don't need their graph history later
        self.activations.append(output.cpu().detach())

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the output of the layer
        # For GradCAM, we can detach the gradients
        self.gradients.append(grad_output[0].cpu().detach())

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        self.activations = [] # Clear previous activations
        self.gradients = []   # Clear previous gradients

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True) # Ensure input has grad history

        # Forward pass to get activations and predictions
        # This will trigger the _save_activations hook
        # The return value of model(input_tensor) will be (conv_features, predictions)
        conv_features_from_model_forward, predictions = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(predictions, dim=1).item()

        score = predictions[:, class_idx]
        self.model.zero_grad() # Zero all gradients for the model

        # Backward pass for the score. This will trigger the _save_gradients hook.
        # retain_graph=True is a good safety measure if other ops might rely on the graph,
        # but for simple GradCAM with hook-captured gradients, it's often not strictly necessary.
        score.backward(retain_graph=True)

        # Retrieve the saved activation and gradient from the lists
        # Assuming batch_size=1, so we take the first (and only) item from the list
        # Move them back to the correct device for computation
        activations = self.activations[0].to(input_tensor.device)
        gradients = self.gradients[0].to(input_tensor.device)

        # Global Average Pooling of gradients (alpha_c^k)
        # Shape of gradients: (batch_size, channels, H, W) -> (batch_size, channels, 1, 1)
        pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weight the activations by the pooled gradients
        # (batch_size, channels, H, W) * (batch_size, channels, 1, 1) -> (batch_size, channels, H, W)
        weighted_activations = activations * pooled_gradients

        # Sum across the channel dimension to get the heatmap
        # Resulting shape: (batch_size, H, W)
        heatmap = torch.sum(weighted_activations, dim=1)
        heatmap = F.relu(heatmap) # Apply ReLU (only positive contributions)

        # Normalize the heatmap to 0-1
        heatmap_single = heatmap[0] # Assuming batch_size=1
        max_val = heatmap_single.max()
        min_val = heatmap_single.min()
        epsilon_norm = 1e-8 # Small constant for numerical stability during normalization
        if max_val > min_val: # Avoid division by zero if heatmap is all same value
            heatmap_single = (heatmap_single - min_val) / (max_val - min_val + epsilon_norm)
        else:
            heatmap_single = torch.zeros_like(heatmap_single) # All zeros if no activation change

        return heatmap_single.cpu().detach().numpy()


"""class GradCAMPlusPlus:
    
    #Implements Grad-CAM++ for a PyTorch model with extensive debug prints.
    #Adjusted for potential zero higher-order gradients issue with ReLU.
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self.hook = self.target_layer.register_forward_hook(self._save_hook_activations)

    def _save_hook_activations(self, module, input, output):
        self._activations = output

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        _, predictions = self.model(input_tensor)
        conv_output = self._activations

        print(f"\n--- GradCAMPlusPlus Debug Output ---")
        print(f"DEBUG: Input tensor requires_grad: {input_tensor.requires_grad}")

        if conv_output is None:
            print(f"DEBUG ERROR: conv_output is None after forward pass. Hook might not have fired or target layer is incorrect.")
            raise RuntimeError("Activations of target layer were not captured by the hook.")

        print(f"DEBUG: Target layer activations (conv_output) shape: {conv_output.shape}")
        print(f"DEBUG: Target layer activations (conv_output) requires_grad: {conv_output.requires_grad}")

        try:
            print(f"DEBUG: conv_output min/max/mean: {conv_output.min().item():.6f}/{conv_output.max().item():.6f}/{conv_output.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG: Could not compute conv_output min/max/mean: {e}")

        # Ensure conv_output explicitly requires graph for higher-order derivatives
        # This is a critical point for Grad-CAM++ to work with ReLU.
        # It should already be true if input_tensor.requires_grad was true.
        # If it wasn't true before, making it true here would disconnect it from original input.
        # The previous version had the correct check. Keeping this as a mental note.
        # if not conv_output.requires_grad:
        #    raise RuntimeError(...)


        if class_idx is None:
            class_idx = torch.argmax(predictions, dim=1).item()

        score = predictions[:, class_idx]
        print(f"DEBUG: Predicted class index: {class_idx}, Score: {score.item():.6f}")

        # --- Compute First-order Gradients ---
        try:
            first_grad = torch.autograd.grad(score, conv_output, retain_graph=True, create_graph=True)[0]
            print(f"DEBUG: Shape of first_grad: {first_grad.shape}")
            print(f"DEBUG: first_grad min/max/mean: {first_grad.min().item():.6f}/{first_grad.max().item():.6f}/{first_grad.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute first_grad: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # --- Compute Second-order Gradients ---
        try:
            grad_2_loss_scalar = torch.sum(first_grad)
            second_grad = torch.autograd.grad(grad_2_loss_scalar, conv_output, retain_graph=True, create_graph=True)[0]
            print(f"DEBUG: Shape of second_grad: {second_grad.shape}")
            print(f"DEBUG: second_grad min/max/mean: {second_grad.min().item():.6f}/{second_grad.max().item():.6f}/{second_grad.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute second_grad: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # --- Compute Third-order Gradients ---
        try:
            grad_3_loss_scalar = torch.sum(second_grad)
            third_grad = torch.autograd.grad(grad_3_loss_scalar, conv_output, retain_graph=False, create_graph=False)[0]
            print(f"DEBUG: Shape of third_grad: {third_grad.shape}")
            print(f"DEBUG: third_grad min/max/mean: {third_grad.min().item():.6f}/{third_grad.max().item():.6f}/{third_grad.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute third_grad: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # --- Grad-CAM++ Alpha Calculation ---
        epsilon = 1e-7
        try:
            alpha_num = second_grad
            # Add a small epsilon to conv_output when used in the denominator
            # This can sometimes help with numerical stability if conv_output itself is zero or near zero.
            # While `alpha_denom = torch.where(alpha_denom != 0.0, ...)` handles zero, this helps with its derivative.
            alpha_denom = (2 * second_grad + third_grad * (conv_output + epsilon)) # Added epsilon here

            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.full_like(alpha_denom, epsilon))
            alphas = alpha_num / (alpha_denom + epsilon)
            print(f"DEBUG: Shape of alphas: {alphas.shape}")
            print(f"DEBUG: alphas min/max/mean: {alphas.min().item():.6f}/{alphas.max().item():.6f}/{alphas.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute alphas: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # --- Compute Weights ---
        try:
            weights = torch.sum(alphas * F.relu(first_grad), dim=(2, 3), keepdim=True)
            print(f"DEBUG: Shape of weights: {weights.shape}")
            print(f"DEBUG: weights min/max/mean: {weights.min().item():.6f}/{weights.max().item():.6f}/{weights.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute weights: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # --- Compute CAM ---
        try:
            cam = torch.sum(weights * conv_output, dim=1, keepdim=True)
            print(f"DEBUG: CAM before ReLU min/max/mean: {cam.min().item():.6f}/{cam.max().item():.6f}/{cam.mean().item():.6f}")
            cam = F.relu(cam)
            print(f"DEBUG: CAM after ReLU min/max/mean: {cam.min().item():.6f}/{cam.max().item():.6f}/{cam.mean().item():.6f}")
        except Exception as e:
            print(f"DEBUG ERROR: Failed to compute CAM: {e}")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Resize CAM
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalize heatmap to [0, 1]
        max_val_norm = np.max(cam)
        min_val_norm = np.min(cam)
        print(f"DEBUG: Final CAM min/max before normalization: {min_val_norm:.6f}/{max_val_norm:.6f}")

        epsilon_norm = 1e-8
        if max_val_norm > min_val_norm:
            cam = (cam - min_val_norm) / (max_val_norm - min_val_norm + epsilon_norm)
            print(f"DEBUG: Heatmap normalized. Min/Max after normalization should be 0/1.")
        else:
            cam = np.zeros_like(cam)
            print("DEBUG: Heatmap was flat (min == max), normalized to all zeros.")

        return cam"""
import torch
import torch.nn.functional as F
import numpy as np

class GradCAMPlusPlus:
    """
    Fixed Grad-CAM++ for models that return both conv features and logits.
    Hook is now optional – direct access preferred.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: A model that returns (conv_output, predictions).
        """
        self.target_layer = target_layer
        self.model = model

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        # Forward pass: expects model to return (conv_output, predictions)
        conv_output, predictions = self.model(input_tensor)
        conv_output.retain_grad()  # Required for higher-order gradients

        if class_idx is None:
            class_idx = torch.argmax(predictions, dim=1).item()

        score = predictions[:, class_idx]

        # --- First-order Gradient ---
        first_grad = torch.autograd.grad(score, conv_output,
                                         retain_graph=True, create_graph=True)[0]

        # --- Second-order Gradient ---
        grad_2_scalar = torch.sum(first_grad)
        second_grad = torch.autograd.grad(grad_2_scalar, conv_output,
                                          retain_graph=True, create_graph=True)[0]

        # --- Third-order Gradient ---
        grad_3_scalar = torch.sum(second_grad)
        third_grad = torch.autograd.grad(grad_3_scalar, conv_output,
                                         retain_graph=False, create_graph=False)[0]

        # --- Compute Alpha ---
        epsilon = 1e-7
        alpha_num = second_grad
        alpha_denom = 2 * second_grad + third_grad * conv_output
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom,
                                  torch.full_like(alpha_denom, epsilon))
        alphas = alpha_num / (alpha_denom + epsilon)

        # --- Weights ---
        weights = torch.sum(alphas * F.relu(first_grad), dim=(2, 3), keepdim=True)

        # --- CAM Computation ---
        cam = torch.sum(weights * conv_output, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # --- Normalize CAM ---
        min_val, max_val = np.min(cam), np.max(cam)
        if max_val > min_val:
            cam = (cam - min_val) / (max_val - min_val + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam