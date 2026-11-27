import torch
import numpy as np

class BitAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.metric = BinaryAccuracy(threshold=threshold)

    def forward(self, predicted_msg, true_msg):

        if isinstance(predicted_msg, torch.Tensor):
             return (predicted_msg == true_msg).sum() / predicted_msg.numel()

        return (predicted_msg == true_msg).sum() / np.size(predicted_msg)
