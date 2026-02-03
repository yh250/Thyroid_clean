import torch
import numpy as np

class SmoothGradSaliencyMap:
    """
    Computes a SmoothGrad-enhanced saliency map.
    Adds noise to input multiple times and averages gradients.
    """

    def __init__(self, model, stdev=0.15, n_samples=25):
        self.model = model.eval()
        self.stdev = stdev
        self.n_samples = n_samples

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad = True

        smooth_saliency = None

        for _ in range(self.n_samples):
            noise = torch.randn_like(input_tensor) * self.stdev
            noisy_input = (input_tensor + noise).requires_grad_(True)

            output = self.model(noisy_input)
            logits = output[1] if isinstance(output, tuple) else output

            if class_idx is None:
                class_idx = torch.argmax(logits, dim=1).item()

            score = logits[:, class_idx]
            self.model.zero_grad()
            score.backward()

            saliency = noisy_input.grad.abs().detach().cpu().squeeze().numpy()

            if saliency.ndim == 3:
                saliency = np.sum(saliency, axis=0)

            if smooth_saliency is None:
                smooth_saliency = saliency
            else:
                smooth_saliency += saliency

        smooth_saliency /= self.n_samples

        # Normalize
        min_val = np.min(smooth_saliency)
        max_val = np.max(smooth_saliency)
        if max_val > min_val:
            smooth_saliency = (smooth_saliency - min_val) / (max_val - min_val)
        else:
            smooth_saliency = np.zeros_like(smooth_saliency)

        return smooth_saliency
