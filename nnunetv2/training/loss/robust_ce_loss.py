import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor, pix_weights: Tensor = None) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
            ce_out = super().forward(input, target.long())
            if pix_weights is not None and self.reduction == "none":
                assert pix_weights.shape[1] == 1
                pix_weights = pix_weights[:, 0]
                return torch.mean(torch.mul(ce_out, pix_weights))
            return ce_out
        
        else:
            ce_out = super().forward(input, target.long())
            if pix_weights is not None and self.reduction == "none":
                assert pix_weights.shape[1] == 1
                pix_weights = pix_weights[:, 0]
                return torch.mean(torch.mul(ce_out, pix_weights))
            return ce_out


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduction=False, label_smoothing=label_smoothing)

    def forward(self, inp, target, pix_weights=None):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target, pix_weights)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
