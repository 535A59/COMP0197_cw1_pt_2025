import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageDraw, ImageFont,ImageFilter
from torch import nn


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:

    correct = (preds == labels).sum().item()
    total = preds.size(0)
    acc = correct / total
    return acc

def compute_f1_macro(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:

    f1_list = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()

        if (tp + fp == 0) or (tp + fn == 0):
            f1_list.append(torch.tensor(0.0))
            continue

        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        if (precision + recall) == 0:
            f1_c = torch.tensor(0.0)
        else:
            f1_c = 2 * precision * recall / (precision + recall)
        f1_list.append(f1_c)

    f1_macro = torch.mean(torch.stack(f1_list)).item()
    return f1_macro

def random_guess_predict(num_classes: int, n_samples: int, seed=None) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        rand_preds = torch.randint(low=0, high=num_classes, size=(n_samples,), generator=g)
    else:
        rand_preds = torch.randint(low=0, high=num_classes, size=(n_samples,))
    return rand_preds

def load_data(batch_size=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size 

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

@torch.no_grad()
def evaluate_model(model, data_loader, num_classes, device='cpu', mode='mean', weights=None):

    all_preds = []
    all_labels = []
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        try:
            preds = model.predict(images, mode=mode, weights=weights)
        except TypeError:
            preds = model.predict(images)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    all_preds  = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc = compute_accuracy(all_preds, all_labels)
    f1  = compute_f1_macro(all_preds, all_labels, num_classes)
    return acc, f1


@torch.no_grad()
def visualize_predictions(model, test_loader, classes, device='cpu', out_file='result.png', n_images=36):

    images_list, labels_gt_list, labels_pred_list = [], [], []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model.predict(images)
        
        images_list.append(images.cpu())
        labels_gt_list.append(labels.cpu())
        labels_pred_list.append(preds.cpu())

        if len(torch.cat(images_list)) >= n_images:
            break

    images_all = torch.cat(images_list)[:n_images]
    labels_gt = torch.cat(labels_gt_list)[:n_images]
    labels_pred = torch.cat(labels_pred_list)[:n_images]

    images_all = images_all * 0.5 + 0.5
    images_all = images_all.clamp(0, 1)

    padding = 60
    grid = make_grid(images_all, nrow=6, padding=padding, pad_value=1.0)
    pil_img = Image.fromarray((grid.permute(1, 2, 0).numpy()*255).astype("uint8"))

    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()  

    total_width, total_height = pil_img.size
    single_w = (total_width - 7*padding) // 6
    single_h = (total_height - 7*padding) // 6

    for idx in range(n_images):
        gt_label = classes[labels_gt[idx]]
        pred_label = classes[labels_pred[idx]]

        row, col = divmod(idx, 6)

        x0 = padding + col*(single_w + padding)
        y0 = padding + row*(single_h + padding)

        text_x = x0 + 2
        text_y = y0 + single_h + 2

        text = f"GT: {gt_label}\nPred: {pred_label}"

        bbox = draw.multiline_textbbox((0,0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        margin = 4
        rect_x1 = text_x + text_w + margin
        rect_y1 = text_y + text_h + margin

        
        draw.rectangle([text_x, text_y, rect_x1, rect_y1], fill=(0,0,0))
        draw.multiline_text(
            (text_x+2, text_y+2),
            text,
            font=font,
            fill=(255,255,255)
        )

    pil_img.save(out_file)
    print(f"Montage saved to: {out_file}")


def visualize_mixup(dataloader, mixup_fn,out_file='mixup.png', device='cpu'):
    images_list = []
    
    for images, label in dataloader:
        images = images.to(device)
        images,_,_,_ = mixup_fn(images,label)
        images_list.append(images.cpu())
        if len(torch.cat(images_list)) >= 16:
            break
    
    images_all = torch.cat(images_list)[:16]
    images_all = (images_all * 0.5) + 0.5
    images_all = images_all.clamp(0, 1)
    grid = make_grid(images_all, nrow=16, padding=0)

    grid_np = (grid.permute(1,2,0).numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(grid_np)
    
    pil_img.save(out_file)
    print(f"16 images in one row saved to: {out_file}")