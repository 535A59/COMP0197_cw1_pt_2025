import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from .general_tool import evaluate_model

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fit_elm_sgd(model,
                train_loader,
                test_loader,
                num_classes,
                model_name = 'model',
                lr=1e-3,
                epochs=50,
                device='cpu',
                mixup_fn=None):
  
    images, labels = next(iter(train_loader))
    images = images.to(device)
    with torch.no_grad():
        _ = model(images)
    model.to(device)

    optimizer = SGD(params=model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 4) 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if mixup_fn is not None:
                mixed_images, y_a, y_b, lam = mixup_fn(images, labels)
                logits = model(mixed_images)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

            else:
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if mixup_fn is not None:
            print(f"Epoch {epoch}/{epochs}, mixup Loss={epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs}, Loss={epoch_loss:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'models/{model_name}/{model_name}_{epoch}.pth')
            acc, f1 = evaluate_model(model, test_loader, num_classes, device)
            if mixup_fn is not None:
                print(f"[ELM] Acc={acc:.4f}, F1_macro={f1:.4f}")
            else:
                print(f"[ELM+MixUp] Acc={acc:.4f}, F1_macro={f1:.4f}")

def fit_ens_sgd(model,
                train_loader,
                lr=1e-3,
                models=5,
                model_num = 1,
                device='cpu',
                mixup_fn=None):
  
    images, labels = next(iter(train_loader))
    images = images.to(device)
    with torch.no_grad():
        _ = model(images)
    model.to(device)

    optimizer = SGD(params=model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        if mixup_fn is not None:
            mixed_images, y_a, y_b, lam = mixup_fn(images, labels)
            logits = model(mixed_images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    if mixup_fn is not None:
        print(f"Model {model_num}/{models}, mixup Loss={epoch_loss:.4f}")
    else:
        print(f"Model {model_num}/{models}, Loss={epoch_loss:.4f}")

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