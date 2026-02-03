import torch
import torch.nn.functional as F
import numpy as np

class Occlusion:
    """
    Occlusion Sensitivity:
    Slides a patch over the image, occludes regions, and measures drop in prediction score.
    """

    def __init__(self, model, input_size=(256, 256), patch_size=16, stride=8, device='cpu', gray_value=0.0):
        """
        Args:
            model: Trained model.
            input_size: Tuple (H, W) of input image.
            patch_size: Size of square occlusion patch.
            stride: Step size to slide the patch.
            device: 'cpu' or 'cuda'.
            gray_value: Value to occlude with (e.g., 0.0 or mean pixel).
        """
        self.model = model.eval()
        self.input_size = input_size
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        self.gray_value = gray_value

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        B, C, H, W = input_tensor.shape

        # Get original prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            logits = output[1] if isinstance(output, tuple) else output

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        original_score = logits[0, class_idx].item()

        # Initialize heatmap and counter
        heatmap = torch.zeros((H, W), device=self.device)
        count_map = torch.zeros((H, W), device=self.device)

        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y1 = y
                y2 = min(y + self.patch_size, H)
                x1 = x
                x2 = min(x + self.patch_size, W)

                # Clone and occlude
                occluded = input_tensor.clone()
                occluded[:, :, y1:y2, x1:x2] = self.gray_value

                with torch.no_grad():
                    occ_output = self.model(occluded)
                    occ_logits = occ_output[1] if isinstance(occ_output, tuple) else occ_output
                    occ_score = occ_logits[0, class_idx].item()

                score_drop = original_score - occ_score
                heatmap[y1:y2, x1:x2] += score_drop
                count_map[y1:y2, x1:x2] += 1

        # Average the heatmap values
        heatmap = heatmap / (count_map + 1e-7)
        heatmap = heatmap.cpu().numpy()

        # Normalize
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val > min_val:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap
