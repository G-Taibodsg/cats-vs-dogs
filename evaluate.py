# simple evaluation script to run on test set and print classification report / confusion matrix
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

MODEL_PATH = 'best_model.pt'

def evaluate(data_dir='data/test'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(MODEL_PATH, device=device)
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = ImageFolder(data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=32)
    ys, yps = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dl):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            yps.extend(preds)
            ys.extend(labels.numpy().tolist())
    print(classification_report(ys, yps, target_names=ds.classes))
    print('Confusion matrix:')
    print(confusion_matrix(ys, yps))

if __name__ == '__main__':
    evaluate()
