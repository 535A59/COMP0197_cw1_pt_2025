import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from .train import fit_ens_sgd
from .ELM import MyExtremeLearningMachine
from .MixUp import MyMixUp
from .general_tool import evaluate_model


class MyEnsembleELM:
    def __init__(self, 
                 n_estimators: int,
                 in_channels: int,
                 hidden_conv_channels: int,
                 kernel_size: int,
                 num_classes: int,
                 std: float=0.1,
                 seed=None):
        self.n_estimators = n_estimators
        self.models = []
        self.num_classes = num_classes
        if seed is not None:
            torch.manual_seed(seed)
        if not (1 <= hidden_conv_channels <= 512):
            raise ValueError("hidden_conv_channels must be in [1,512].")
        for _ in range(n_estimators):
            model = MyExtremeLearningMachine(
                in_channels=in_channels,
                hidden_conv_channels=hidden_conv_channels,
                kernel_size=kernel_size,
                num_classes=num_classes,
                std=std
            )
            self.models.append(model)

    def fit(self, train_loader,test_loader,lr=1e-3, epochs=10, device='cpu', mixup_fn=None,model_name='model',save_model=True):
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            for i, model in enumerate(self.models):
                fit_ens_sgd(model, train_loader, lr=lr, models=self.n_estimators,model_num=i+1,device=device, mixup_fn=mixup_fn)
            if epoch % 5 == 0:
                if save_model:
                    torch.save([model.state_dict() for model in self.models], f'models/{model_name}/{model_name}_{epoch}.pth')
                if mixup_fn is not None:
                    acc_ens_v, f1_ens_v = evaluate_model(self, test_loader, self.num_classes , device, mode='vote')
                    print(f"[EnsembleELM_vote] Acc={acc_ens_v:.4f}, F1_macro={f1_ens_v:.4f}")
                    acc_ens_m, f1_ens_m = evaluate_model(self, test_loader, self.num_classes , device, mode='mean')
                    print(f"[EnsembleELM_mean] Acc={acc_ens_m:.4f}, F1_macro={f1_ens_m:.4f}")
                else:
                    acc_ensm_v, f1_ensm_v = evaluate_model(self, test_loader, self.num_classes, device, mode='vote')
                    print(f"[EnsembleELM_vote] Acc={acc_ensm_v:.4f}, F1_macro={f1_ensm_v:.4f}")
                    acc_ensm_m, f1_ensm_m = evaluate_model(self, test_loader, self.num_classes, device, mode='mean')
                    print(f"[EnsembleELM_mean] Acc={acc_ensm_m:.4f}, F1_macro={f1_ensm_m:.4f}")


    def load_models(self, model_paths):
        for model, path in zip(self.models, model_paths):
            model.load_state_dict(torch.load(path))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, mode='mean') -> torch.Tensor:
        if mode == 'mean':
            probs_list = [model.predict_proba(x) for model in self.models]
            all_probs = torch.stack(probs_list, dim=0)
            mean_probs = all_probs.mean(dim=0) 
            return mean_probs
        elif mode == 'vote':
            votes_list = [model.predict(x) for model in self.models] 
            all_votes = torch.stack(votes_list, dim=0)
            mode_pred, _ = torch.mode(all_votes, dim=0)
            batch_size = mode_pred.size(0)
            probs = torch.zeros(batch_size, self.num_classes, device=x.device)
            probs[torch.arange(batch_size), mode_pred] = 1.0
            return probs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mode='mean') -> torch.Tensor:
        probs = self.predict_proba(x, mode=mode)
        return probs.argmax(dim=1)