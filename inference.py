# inference.py
from PIL import Image
import os
import numpy as np

MODEL_PATH = 'best_model.pt'

def pil_preprocess(pil_image, size=(224, 224)):
    """
    Minimal preprocessing using PIL + numpy:
    - convert to RGB
    - resize to size
    - convert to float32 and normalize using ImageNet mean/std
    - returns a numpy array (C,H,W)
    """
    img = pil_image.convert('RGB').resize(size, Image.BILINEAR)
    arr = np.array(img).astype('float32') / 255.0  # H,W,C, range 0-1
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    arr = (arr - mean) / std
    # to C,H,W
    arr = arr.transpose(2, 0, 1)
    return arr

class InferenceEngine:
    """
    Lazy-load torch + model only when predict() is called.
    If torch/model import fails, predict() returns {'error': True, 'message': ...}.
    """
    def __init__(self, model_path=MODEL_PATH, classes=('cat', 'dog')):
        self.model_path = model_path
        self.classes = classes
        self.device = None
        self.model = None
        self._torch_ok = None
        self.torch = None

    def _ensure_torch_and_model(self):
        """Attempt to import torch and load model. Return (ok:bool, msg:str|None)."""
        if self._torch_ok is not None:
            return self._torch_ok, None if self._torch_ok else "previous import failed"

        try:
            # local imports to avoid module-level import of torch/torchvision
            import torch
            from model import load_model  # model.py imports torch internally
            self.torch = torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if not os.path.exists(self.model_path):
                self._torch_ok = False
                return False, f"Model file not found: {self.model_path}"
            # load_model returns a torch.nn.Module already on correct device
            self.model = load_model(self.model_path, device=self.device)
            self._torch_ok = True
            return True, None
        except Exception as e:
            self._torch_ok = False
            return False, str(e)

    def predict(self, pil_image):
        """
        Input: PIL.Image
        Output: dict with either {'error':True,'message':...} or {'label','prob','probs'}
        """
        ok, msg = self._ensure_torch_and_model()
        if not ok:
            return {'error': True, 'message': f'Torch/model load failed: {msg}'}

        # preprocess with PIL+numpy (avoids torchvision dependency)
        arr = pil_preprocess(pil_image, size=(224, 224))
        # convert to torch tensor on the correct device
        try:
            x = self.torch.from_numpy(arr).unsqueeze(0).to(self.device)  # shape 1,C,H,W
            x = x.type(self.torch.float32)
        except Exception as e:
            return {'error': True, 'message': f'Error converting input to tensor: {e}'}

        try:
            with self.torch.no_grad():
                logits = self.model(x)
                probs = self.torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(probs.argmax())
                return {'label': self.classes[idx], 'prob': float(probs[idx]), 'probs': probs.tolist()}
        except Exception as e:
            return {'error': True, 'message': f'Error during model forward: {e}'}
