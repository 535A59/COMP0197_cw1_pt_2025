import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import os
import argparse


from utils import(MyExtremeLearningMachine, 
                 MyEnsembleELM,MyMixUp,
                 fit_elm_sgd ,
                 compute_accuracy, 
                 compute_f1_macro, 
                 random_guess_predict, 
                 evaluate_model,
                 visualize_predictions,
                 load_data,
                 visualize_mixup)



torch.manual_seed(42)
os.makedirs('models', exist_ok=True)


def main():
    epochs = 20
    num_classes = 10 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, classes = load_data(batch_size=64) 
    print('In multicategory classification, the random guess benchmark is calculated in two ways: if the categories are balanced, the uniform guess accuracy is 1/K (10 per cent when K = 10); if the categories are not balanced, it is the sum of the proportions of the categories. In this problem, because the categories are balanced, so I use the uniform guess, and then want to use the accuracy to test, for CIFAR10random gauss expected accuracy should be 10%')
    #------------------------------------------
    # A) Random Guess
    #------------------------------------------
    print("=== A) Random Guess Baseline ===")
    test_size = len(test_loader.dataset)
    all_test_labels = []
    for _, labels in test_loader:
        all_test_labels.append(labels)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    rand_preds = random_guess_predict(num_classes, test_size, seed=42)
    acc_rand = compute_accuracy(rand_preds, all_test_labels)
    f1_rand  = compute_f1_macro(rand_preds, all_test_labels, num_classes)
    print(f"[RandomGuess] Acc={acc_rand:.4f}, F1_macro={f1_rand:.4f}")
    #------------------------------------------
    # B) Basic ELM
    #------------------------------------------
    print("\n=== B) Basic ELM ===")
    model_elm = MyExtremeLearningMachine(in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes, std=0.1)
    fit_elm_sgd(model_elm, train_loader, test_loader,num_classes=10,lr=1e-2, epochs=epochs, device=device,model_name='basic_elm')
    acc_elm, f1_elm = evaluate_model(model_elm, test_loader, num_classes, device)
    #------------------------------------------
    # C) ELM + MyMixUp
    #------------------------------------------
    print("\n=== C) ELM + MyMixUp ===")
    mixup_fn = MyMixUp(alpha=1, seed=42)
    model_elm_mixup = MyExtremeLearningMachine(in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes, std=0.1)
    visualize_mixup(train_loader, mixup_fn)
    fit_elm_sgd(model=model_elm_mixup,train_loader=train_loader,test_loader=test_loader,num_classes=10,lr=1e-3,epochs=epochs,device=device,mixup_fn=mixup_fn,model_name='elm_mixup')
    acc_elm_mix, f1_elm_mix = evaluate_model(model_elm_mixup, test_loader, num_classes, device)

    #------------------------------------------
    # D) Ensemble ELM
    #------------------------------------------
    print("\n=== D) MyEnsembleELM ===")
    ensemble_elm = MyEnsembleELM(n_estimators=5,in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes,std=0.1,seed=42)
    ensemble_elm.fit(train_loader, test_loader=test_loader,lr=1e-2, epochs=epochs, device=device,model_name='ensemble_elm')
    acc_ens_v, f1_ens_v = evaluate_model(ensemble_elm, test_loader, num_classes, device, mode='vote')
    acc_ens_m, f1_ens_m = evaluate_model(ensemble_elm, test_loader, num_classes, device, mode='mean')

    #------------------------------------------
    # E) Ensemble + MixUp
    #------------------------------------------
    print("\n=== E) Ensemble + MixUp ===")
    mixup_fn = MyMixUp(alpha=1, seed=42)
    ensemble_elm_mixup = MyEnsembleELM(n_estimators=5,in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes,std=0.1,seed=42)
    ensemble_elm_mixup.fit(train_loader,test_loader=test_loader, lr=1e-2, epochs=epochs, device=device, mixup_fn=mixup_fn,model_name='ensemble_elm_mixup')
    acc_ensm_v, f1_ensm_v = evaluate_model(ensemble_elm_mixup, test_loader, num_classes, device, mode='vote')
    acc_ensm_m, f1_ensm_m = evaluate_model(ensemble_elm_mixup, test_loader, num_classes, device, mode='mean')

    acc_elm = 0
    f1_elm = 0
    acc_elm_mix = 0
    f1_elm_mix = 0
    model_elm = None
    model_elm_mixup = None
    #------------------------------------------
    best_acc = max(acc_rand, acc_elm, acc_elm_mix, acc_ens_v, acc_ens_m, acc_ensm_v, acc_ensm_m)
    best_model = model_elm
    if best_acc == acc_elm_mix:
        best_model = model_elm_mixup
    elif best_acc == acc_ens_v:
        best_model = ensemble_elm
    elif best_acc == acc_ens_m:
        best_model = ensemble_elm
    elif best_acc == acc_ensm_v:
        best_model = ensemble_elm_mixup
    elif best_acc == acc_ensm_m:
        best_model = ensemble_elm_mixup

    visualize_predictions(best_model, test_loader, classes=classes,
                                  device='cpu', out_file='result.png', n_images=36)

    print("\n=== Summary ===")
    print(f"RandomGuess            => Acc={acc_rand:.3f}, F1={f1_rand:.3f}")
    print(f"Basic ELM              => Acc={acc_elm:.3f}, F1={f1_elm:.3f}")
    print(f"ELM + MixUp            => Acc={acc_elm_mix:.3f}, F1={f1_elm_mix:.3f}")
    print(f"Ensemble ELM vote      => Acc={acc_ens_v:.3f}, F1={f1_ens_v:.3f}")
    print(f"Ensemble ELM mean      => Acc={acc_ens_m:.3f}, F1={f1_ens_m:.3f}")
    print(f"Ensemble + MixUp vote  => Acc={acc_ensm_v:.3f}, F1={f1_ensm_v:.3f}")
    print(f"Ensemble + MixUp mean  => Acc={acc_ensm_m:.3f}, F1={f1_ensm_m:.3f}")
    print(f"Best model             => {best_model.__class__.__name__}")


def test(device='cpu',epoch=20):
    print("Running Test...")
    train_loader, test_loader, classes = load_data(batch_size=64) 
    num_classes = 10

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # Load saved models
    model_elm = MyExtremeLearningMachine(in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes, std=0.1)
    with torch.no_grad():
        model_elm(dummy_input)
    model_elm.load_state_dict(torch.load(f'models/basic_elm/basic_elm_{epoch}.pth', map_location=device))

    
    model_elm_mixup = MyExtremeLearningMachine(in_channels=3, hidden_conv_channels=16, kernel_size=3, num_classes=num_classes)
    with torch.no_grad():
        model_elm_mixup(dummy_input)
    mixup = MyMixUp(alpha=1, seed=42)
    visualize_mixup(train_loader, mixup)
    model_elm_mixup.load_state_dict(torch.load(f'models/elm_mixup/elm_mixup_{epoch}.pth', map_location=device))

    ensemble_elm = MyEnsembleELM(n_estimators=5, in_channels=3, hidden_conv_channels=16, kernel_size=3, num_classes=num_classes)
    state_dicts = torch.load(f'models/ensemble_elm/ensemble_elm_{epoch}.pth', map_location=device)
    for model, state_dict in zip(ensemble_elm.models, state_dicts):
        with torch.no_grad():
            model(dummy_input)
        model.load_state_dict(state_dict)

    ensemble_elm_mixup = MyEnsembleELM(n_estimators=5, in_channels=3, hidden_conv_channels=16, kernel_size=3, num_classes=num_classes)
    state_dicts_mixup = torch.load(f'models/ensemble_elm_mixup/ensemble_elm_mixup_{epoch}.pth', map_location=device)
    for model, state_dict in zip(ensemble_elm_mixup.models, state_dicts_mixup):
        with torch.no_grad():
            model(dummy_input)
        model.load_state_dict(state_dict)



    acc_elm, f1_elm = evaluate_model(model_elm, test_loader, num_classes, device)
    print(f"[Test Basic ELM] Acc={acc_elm:.4f}, F1_macro={f1_elm:.4f}")

    acc_elm_mix, f1_elm_mix = evaluate_model(model_elm_mixup, test_loader, num_classes, device)
    print(f"[Test ELM+MixUp] Acc={acc_elm_mix:.4f}, F1_macro={f1_elm_mix:.4f}")

    acc_ens, f1_ens = evaluate_model(ensemble_elm, test_loader, num_classes, device)
    print(f"[Test Ensemble ELM mean] Acc={acc_ens:.4f}, F1_macro={f1_ens:.4f}")
    acc_ens, f1_ens = evaluate_model(ensemble_elm, test_loader, num_classes, device,mode='vote')
    print(f"[Test Ensemble ELM vote] Acc={acc_ens:.4f}, F1_macro={f1_ens:.4f}")

    acc_ens_mix, f1_ens_mix = evaluate_model(ensemble_elm_mixup, test_loader, num_classes, device)
    print(f"[Test Ensemble ELM+MixUp mean] Acc={acc_ens_mix:.4f}, F1_macro={f1_ens_mix:.4f}")
    acc_ens_mix, f1_ens_mix = evaluate_model(ensemble_elm_mixup, test_loader, num_classes, device,mode='vote')
    print(f"[Test Ensemble ELM+MixUp vote] Acc={acc_ens_mix:.4f}, F1_macro={f1_ens_mix:.4f}")

    visualize_predictions(ensemble_elm, test_loader, classes, out_file='result.png', n_images=36)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ELM Training and Testing")
    parser.add_argument('--test-mode', type=str, default=False, help="Mode: 'train' or 'test'")
    parser.add_argument('--epoch-select', type=str, default=20, help="Mode: 5, 10, 15, 20")

    args = parser.parse_args()

    if args.test_mode:
        test(epoch=args.epoch_select)
    else:
        main()