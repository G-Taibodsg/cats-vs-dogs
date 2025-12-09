import os
os.environ['QT_OPENGL'] = 'software'
# now try import torch after setting env var
import torch
print('torch ok, version=', torch.__version__, 'cuda=', torch.cuda.is_available())
