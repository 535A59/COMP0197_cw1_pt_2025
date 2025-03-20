import torch
import torch.nn as nn

class MyCrossEntropy(nn.Module):
    def __init__(self):
        super(MyCrossEntropy, self).__init__()

    def forward(self, y_pred, t):
        eps = 1e-8
        loss = -torch.mean(t * torch.log(y_pred + eps) + (1 - t) * torch.log(1 - y_pred + eps))
        return loss
    
class MyRootMeanSquare(nn.Module):
    def __init__(self):
        super(MyRootMeanSquare, self).__init__()

    def forward(self, y_pred, t):
        loss = torch.mean((y_pred - t) ** 2)
        loss = torch.clamp(loss, min=1e-8)
        return torch.sqrt(loss)