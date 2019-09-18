from PIL import Image
from torchvision.utils import make_grid
import torch

def save_org_recon(org, recon, epoch, mode='Train'):
    tensor = torch.cat((org, recon), 1)
    grid = make_grid(tensor, nrow=8, padding=2, pad_value=0,
                     normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(127).clamp(0, 127).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save('Recon/{}_{}.jpg'.format(mode, epoch))