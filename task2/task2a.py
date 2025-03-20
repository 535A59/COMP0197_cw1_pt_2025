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
import time

from utils import(MyExtremeLearningMachine,
                 MyEnsembleELM,
                 evaluate_model,
                 visualize_predictions,
                 load_data,
                 fit_elm_ls,
                 random_search_elm)

def main():
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10
    train_loader, test_loader, classes = load_data(batch_size=64)

    # start = time.time()
    # ensemble_elm = MyEnsembleELM(n_estimators=5,in_channels=3,hidden_conv_channels=16,kernel_size=3,num_classes=num_classes,std=0.1,seed=42)
    # ensemble_elm.fit(train_loader, test_loader=test_loader,lr=1e-2, epochs=20, device=device,model_name='ensemble_elm',save_model=True)
    # acc_ens_m, f1_ens_m = evaluate_model(ensemble_elm, test_loader, num_classes, device, mode='mean')
    # print(f"Ensemble ELM: Acc={acc_ens_m:.4f}, F1_macro={f1_ens_m:.4f}")
    # end = time.time()
    # print(f"Time taken for Ensemble ELM: {end-start:.2f}s")

    start = time.time()
    model = MyExtremeLearningMachine(
            in_channels=3,
            hidden_conv_channels=16,
            kernel_size=3,
            num_classes=10,
            std=0.1
        ).to(device)
    fit_elm_ls(model, train_loader, reg_lambda=0.0)
    end = time.time()
    accuracy,f1= evaluate_model(model, test_loader, num_classes, device)
    print(f"ELM (LS): Acc={accuracy:.4f}, F1_macro={f1:.4f}")
    print(f"Time taken for ELM (LS): {end-start:.2f}s")

    best_model,_,best_para =random_search_elm(train_loader, test_loader, device=device,seed=42)
    accuracy,f1= evaluate_model(best_model, test_loader, num_classes, device)
    visualize_predictions(best_model, test_loader, classes, out_file='new_result.png', n_images=36)



if __name__ == "__main__":
    main()