from pathlib import Path
from PIL import Image

def clean_folder(root='data'):
    root = Path(root)
    total = 0
    removed = 0

    for img_path in root.rglob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            total += 1
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                print('Removing bad image:', img_path)
                img_path.unlink()
                removed += 1

    print(f'Checked {total} images, removed {removed} bad images.')

if __name__ == '__main__':
    clean_folder('data')
