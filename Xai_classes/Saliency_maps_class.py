import torch
import numpy as np

class SaliencyMap:
    """
    Computes a simple saliency map by taking the absolute value of the gradient
    of the output score with respect to the input image.
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): The trained model to interpret.
        """
        self.model = model.eval()  # Set model to evaluation mode

    def __call__(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor (torch.Tensor): Input image tensor with shape (1, C, H, W).
            class_idx (int, optional): Index of the target class. If None, uses predicted class.

        Returns:
            np.ndarray: A 2D normalized saliency map (H, W).
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        logits = output[1] if isinstance(output, tuple) else output

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # Backward pass
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        # Compute saliency (absolute gradient)
        saliency = input_tensor.grad.abs().detach().cpu().squeeze().numpy()

        # Reduce channels if input is RGB
        if saliency.ndim == 3:
            saliency = saliency.sum(axis=0)

        # Normalize to [0, 1]
        min_val = np.min(saliency)
        max_val = np.max(saliency)
        if max_val > min_val:
            saliency = (saliency - min_val) / (max_val - min_val)
        else:
            saliency = np.zeros_like(saliency)

        return saliency
