import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from .train import fit_elm_sgd

class MyExtremeLearningMachine(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_conv_channels: int,
                 kernel_size: int,
                 num_classes: int = 10,
                 std: float = 0.1):
        super(MyExtremeLearningMachine,self).__init__()
        self.fixed_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_conv_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.initialise_fixed_layers(std=std)
        self.batch_norm = nn.BatchNorm2d(hidden_conv_channels)
        for param in self.fixed_conv.parameters():
            param.requires_grad = False
        self.fc = None
        self.num_classes = num_classes

    @torch.no_grad()
    def initialise_fixed_layers(self, std: float = 0.1):
        nn.init.normal_(self.fixed_conv.weight, mean=0.0, std=std)
    
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.fixed_conv(x)
            hidden = self.batch_norm(hidden)
        hidden = F.relu(hidden)
        batch_size = hidden.size(0)
        hidden_flat = hidden.view(batch_size, -1)
        if self.fc is None:
            in_features = hidden_flat.size(1)
            self.fc = nn.Linear(in_features, self.num_classes)
        logits = self.fc(hidden_flat)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=1)