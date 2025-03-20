import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_data, fit_logistic_sgd, evaluate_accuracy, evaluate_noisy_accuracy

def main():
    N_train=200
    N_test=100
    D=5
    M=2
    torch.manual_seed(1)
        
    X_train, t_train_noisy, y_train_true = generate_data(N_train, D, M)
    X_test, t_test_noisy,  y_test_true  = generate_data(N_test,  D, M)

    for M in [1, 2, 3]:
        train_noisy_acc = evaluate_noisy_accuracy(t_train_noisy, y_train_true)
        test_noisy_acc  = evaluate_noisy_accuracy(t_test_noisy,  y_test_true)

        # ========== 训练 (Cross Entropy) ========== 
        w_ce = fit_logistic_sgd(
            X_train, t_train_noisy, 
            M, loss_type='cross_entropy',
            lr=1e-3, batch_size=32, epochs=100
        )
        train_model_acc_ce = evaluate_accuracy(X_train, y_train_true, w_ce, M)
        test_model_acc_ce  = evaluate_accuracy(X_test,  y_test_true,  w_ce, M)
        print(f"[CE] M={M}")
        print(f" - Model Accuracy (Train/Test): {train_model_acc_ce:.3f}/{test_model_acc_ce:.3f}")
        print(f" - Noisy Data Accuracy (Train/Test): {train_noisy_acc:.3f}/{test_noisy_acc:.3f}")

        # ========== 训练 (RMS) ==========
        w_rms = fit_logistic_sgd(
            X_train, t_train_noisy, 
            M, loss_type='rms',
            lr=1e-3, batch_size=32, epochs=100
        )
        train_model_acc_rms = evaluate_accuracy(X_train, y_train_true, w_rms, M)
        test_model_acc_rms  = evaluate_accuracy(X_test,  y_test_true,  w_rms, M)
        print(f"[RMS] M={M}")
        print(f" - Model Accuracy (Train/Test): {train_model_acc_rms:.3f}/{test_model_acc_rms:.3f}")
        print(f" - Noisy Data Accuracy (Train/Test): {train_noisy_acc:.3f}/{test_noisy_acc:.3f}")

    
    print('\n')
    print('Accuracy can be used as a metric for this classification problem because it is an intuitive and simple evaluation metric, and in this data-balanced binary classification scenario, it can directly and clearly reflect the proportion of correctly classified samples, and easy to compare with the results of rest experiments.')
    print('\n')
    print('The accuracy of model prediction measures the ability of the model to generalise and recover the data, reflecting the quality of the model parameters, while the accuracy of the observed noisy training data reflects the degree of noise in the data itself, reflecting the extent to which the observed labels deviate from the true labels. For M=2 the model prediction accuracy significantly exceeds the accuracy of the data itself, indicating that the model effectively extracts the true pattern from the noisy data, whereas for M=1 and M=3, the model is too simple or complex, resulting in a decrease in the model performance, and even failing to exceed the accuracy of the original noise labels.')

if __name__ == "__main__":
    main()