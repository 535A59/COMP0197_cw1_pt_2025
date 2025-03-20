import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations_with_replacement
import torch.nn.init as init
from .loss_fun import MyCrossEntropy, MyRootMeanSquare
from .general_tool import polynomial_transform

def fit_logistic_sgd(X, t, M, loss_type='cross_entropy', lr=1e-3, batch_size=32, epochs=50):

    X_poly = polynomial_transform(X, M)
    N, p = X_poly.shape

    w = torch.rand(p, dtype=torch.float32) * 0.05
    w.requires_grad_()

    if loss_type == 'cross_entropy':
        loss_fn = MyCrossEntropy()
    elif loss_type == 'rms':
        loss_fn = MyRootMeanSquare()
    else:
        raise ValueError("Unsupported loss_type. Use 'cross_entropy' or 'rms'.")
    optimizer = torch.optim.SGD([w], lr=lr)

    print_interval = max(1, epochs // 9)

    for epoch in range(epochs):
        idx = torch.randperm(N)
        X_shuffled = X_poly[idx]
        t_shuffled = t[idx].float()

        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i : i+batch_size]
            t_batch = t_shuffled[i : i+batch_size]

            y_linear = X_batch @ w
            y_pred = torch.sigmoid(y_linear)

            loss = loss_fn(y_pred, t_batch)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()  

        if epoch == 0 or (epoch+1) == epochs or (epoch+1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, {loss_type} loss: {loss.item():.6f}")

    return w.detach()