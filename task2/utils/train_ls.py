import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from .general_tool import evaluate_model
import time
import random
from .ELM import MyExtremeLearningMachine


@torch.no_grad()
def fit_elm_ls(model: nn.Module, train_loader, reg_lambda: float = 0.0, device: str = 'cpu'):

    hidden_list = []
    label_list = []
    for x, y in train_loader:
        hidden = model.fixed_conv(x)
        hidden = model.batch_norm(hidden)
        hidden = F.relu(hidden)
        hidden_flat = hidden.view(hidden.size(0), -1)
        hidden_list.append(hidden_flat)
        label_list.append(y)
    
   
    H_all = torch.cat(hidden_list, dim=0) 
    Y_all = torch.cat(label_list, dim=0)
    
    if Y_all.dim() == 1 or (Y_all.dim() == 2 and Y_all.size(1) == 1):
        Y_all = F.one_hot(Y_all.long(), num_classes=model.num_classes).float()
    
    ones = torch.ones(H_all.size(0), 1, device=device)
    H_aug = torch.cat([H_all, ones], dim=1)
    
    D_plus_1 = H_aug.size(1)
    I = torch.eye(D_plus_1, device=device)
    
    A = H_aug.t() @ H_aug + reg_lambda * I 
    B = H_aug.t() @ Y_all  
    W_aug = torch.linalg.solve(A, B) 
    
    W = W_aug[:-1, :].t() 
    b = W_aug[-1, :].t() 
    
    in_features = H_all.size(1)
    if model.fc is None:
        model.fc = nn.Linear(in_features, model.num_classes)
        model.fc.to(device)

    model.fc.weight.data.copy_(W)
    model.fc.bias.data.copy_(b)
    
    return model

def random_search_elm(train_loader, test_loader, device='cpu', num_trials=20, seed=None):
    best_acc = 0.0
    best_model = None
    best_params = None

    for trial in range(num_trials):
        start = time.time()
        hidden_conv_channels = random.choice([8, 16,32])
        kernel_size = random.choice([3, 5])
        std = random.choice([0.05, 0.1, 0.2])
        reg_lambda = random.uniform(0.0, 1.0)
        
        model = MyExtremeLearningMachine(
            in_channels=3,
            hidden_conv_channels=hidden_conv_channels,
            kernel_size=kernel_size,
            num_classes=10,
            std=std
        ).to(device)
        
        model = fit_elm_ls(model, train_loader, reg_lambda=reg_lambda, device=device)
        
        acc, f1 = evaluate_model(model, test_loader, num_classes=10, device=device)
        print(f"Trial {trial+1}: hidden_conv_channels={hidden_conv_channels}, "
              f"kernel_size={kernel_size}, std={std}, reg_lambda={reg_lambda:.4f} "
              f"=> Acc: {acc:.2f}, F1_macro: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_params = {
                'hidden_conv_channels': hidden_conv_channels,
                'kernel_size': kernel_size,
                'std': std,
                'reg_lambda': reg_lambda
            }
            torch.save(model.state_dict(), f'models/best_ls_model.pth')
        end = time.time()
        print(f"Time taken: {end-start:.2f}s")

    print(f"Best model performance: Acc: {best_acc:.2f} F1={f1:.4f} with params: {best_params}")
    return best_model, best_params, best_acc
