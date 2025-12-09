import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from model import ClassifierWrapper, save_model

# Hyperparameters you can tune
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_dataloaders(data_dir='data'):
    train_tfms = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tfms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = ImageFolder(Path(data_dir)/'train', transform=train_tfms)
    val_ds = ImageFolder(Path(data_dir)/'val', transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


def train():
    train_loader, val_loader, classes = make_dataloaders()
    model = ClassifierWrapper(n_classes=len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} (train)')
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # validate
        model.eval()
        ys, yps = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model(imgs)
                yps.extend(preds.argmax(dim=1).cpu().numpy().tolist())
                ys.extend(labels.cpu().numpy().tolist())
        val_acc = accuracy_score(ys, yps)
        print(f'Validation accuracy: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model.pt')
            print('Saved best_model.pt')

    print('Training finished. Best val acc:', best_val_acc)

if __name__ == '__main__':
    train()