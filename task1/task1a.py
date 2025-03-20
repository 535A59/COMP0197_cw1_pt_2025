from torch import nn
from utils import generate_data, fit_logistic_sgd, evaluate_accuracy
import torch

torch.manual_seed(1)
def Optimise_M(loss_type='cross_entropy', lr=1e-3, batch_size=32, epochs=50,M_nums = 6):
    N_train, N_test, N_valid = 200, 100, 100
    D = 5
    M= 2

    best_M = None
    best_acc = -1.0
    best_w = None

    X_train, t_train_noisy, y_train_true = generate_data(N_train, D, M)
    X_valid, t_valid_noisy, y_valid_true = generate_data(N_valid, D, M)

    for M in range(1,M_nums+1):
        w_hat = fit_logistic_sgd(
            X_train, t_train_noisy, 
            M, loss_type=loss_type, 
            lr=1e-3, batch_size=16, epochs=100
        )


        test_acc = evaluate_accuracy(X_valid, y_valid_true, w_hat, M)
        print(f"[Info] M={M} => Test Accuracy={test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_M = M
            best_w = w_hat



    w_hat = fit_logistic_sgd(
            X_train, t_train_noisy, 
            best_M, loss_type=loss_type, 
            lr=1e-3, batch_size=16, epochs=100
        )
    
    X_test, t_test_noisy,  y_test_true  = generate_data(N_test,  D, M)
    test_acc = evaluate_accuracy(X_test, y_test_true, w_hat, best_M)
    return best_M, best_acc, best_w
    
if __name__ == "__main__":

    CE_M,CE_acc,_= Optimise_M(loss_type='cross_entropy', lr=1e-3, batch_size=16, epochs=50)
    RMS_M,RMS_acc,_ = Optimise_M(loss_type='rms', lr=1e-3, batch_size=16, epochs=50)

    print(f"\nOptimised M for Cross Entropy = {CE_M}, with Test Accuracy={CE_acc:.3f}")
    print(f"\nOptimised M for RMS = {RMS_M}, with Test Accuracy={RMS_acc:.3f}")
