import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# Assumes you have a top-level folder 'raw_data' with subfolders 'cats' and 'dogs'
# Each subfolder contains images. This script will create train/val/test splits.

def prepare_dataset(raw_dir='raw_data', out_dir='data', test_size=0.1, val_size=0.1, seed=42):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # get class names from subfolders, e.g. cats / dogs
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError('No class subfolders found in raw_data (expecting cats/ dogs).')

    # create output directories
    for split in ['train', 'val', 'test']:
        for c in classes:
            (out_dir / split / c).mkdir(parents=True, exist_ok=True)

    # split and copy data
    for c in classes:
        imgs = list((raw_dir / c).glob('*'))
        trainval, test = train_test_split(imgs, test_size=test_size, random_state=seed)
        train, val = train_test_split(
            trainval, test_size=val_size / (1 - test_size), random_state=seed
        )

        for p in train:
            shutil.copy(p, out_dir / 'train' / c / p.name)
        for p in val:
            shutil.copy(p, out_dir / 'val' / c / p.name)
        for p in test:
            shutil.copy(p, out_dir / 'test' / c / p.name)


if __name__ == '__main__':
    prepare_dataset()
