from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from torch.nn import functional as F
from torch.nn.functional import softplus

from models.nn_utils import make_sequential_from_config

import numpy as np

## taken and/or adapted from genesis-v2 implementation of University of Oxford's Applied AI Lab's github account 
## accessed at: https://github.com/applied-ai-lab/genesis

def clamp_preserve_gradients(x, lower, upper):
    # From: http://docs.pyro.ai/en/0.3.3/_modules/pyro/distributions/iaf.html
    return x + (x.clamp(lower, upper) - x).detach()

def flatten(x):
    return x.view(x.size(0), -1)

def unflatten(x):
    return x.view(x.size(0), -1, 1, 1)

def euclidian_norm(x):
    # Clamp before taking sqrt for numerical stability
    return clamp_preserve_gradients((x**2).sum(1), 1e-10, 1e10).sqrt()


def euclidian_distance(embedA, embedB):
    # Unflatten if needed if one is an image and the other a vector
    # Assumes inputs are batches
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return euclidian_norm(embedA - embedB)

def squared_distance(embedA, embedB):
    # Unflatten if needed if one is an image and the other a vector
    # Assumes inputs are batches
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return ((embedA - embedB)**2).sum(1)

def pixel_coords(img_size):
    g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, img_size),
                              torch.linspace(-1, 1, img_size))
    g_1 = g_1.view(1, 1, img_size, img_size)
    g_2 = g_2.view(1, 1, img_size, img_size)
    return torch.cat((g_1, g_2), dim=1)

class ScalarGate(nn.Module):
    def __init__(self, init=0.0):
        super(ScalarGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(init))
    def forward(self, x):
        return self.gate * x


class SemiConv(nn.Module):
    def __init__(self, nin, nout, img_size):
        super(SemiConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1)
        self.gate = ScalarGate()
        coords = pixel_coords(img_size)
        zeros = torch.zeros(1, nout-2, img_size, img_size)
        self.uv = torch.cat((zeros, coords), dim=1)
    def forward(self, x):
        out = self.gate(self.conv(x))
        delta = out[:, -2:, :, :]
        return out + self.uv.to(out.device), delta


@dataclass(eq=False, repr=False)
class ICSBP(nn.Module):
    

    width: int
    height: int
    num_slots: int
    feat_dim: int
    latent_dim: int
    semiconv: bool
    dist_kernel: str

    def __post_init__(self):
        super().__init__()
                
        if self.dist_kernel == 'laplacian':
            sigma_init = 1.0 / (np.sqrt(self.num_slots)*np.log(2))
        elif self.dist_kernel == 'gaussian':
            sigma_init = 1.0 / (self.num_slots*np.log(2))
        elif self.dist_kernel == 'epanechnikov':
            sigma_init = 2.0 / self.num_slots
        else:
            return ValueError("No valid kernel.")
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        # Colour head
        if self.semiconv:
            self.c_head = SemiConv(self.feat_dim, self.latent_dim, self.width)
        else:
            self.c_head = nn.Conv2d(self.feat_dim, self.latent_dim, 1)

    def forward(self, x: Tensor, num_cl_steps: int) -> dict:
        bs, channels, width, height = x.shape

        #print(x.shape)
        #print("-----------------------")
        c_out = self.c_head(x)
        
        if isinstance(c_out, tuple):
            c_f, delta = c_out
        else:
            c_f, delta = c_out, None
            
        # Sample from uniform to select random pixels as seeds
        rand_pixel = torch.empty(bs, 1, *c_f.shape[2:]).to(x.device)
        rand_pixel = rand_pixel.uniform_()
        # Run SBP
        seed_list = []
        log_m_k = []
        log_s_k = [torch.zeros(bs, 1, self.width, self.height).to(x.device)]
        for step in range(num_cl_steps):
            # Determine seed
            scope = F.interpolate(log_s_k[step].exp(), size=c_f.shape[2:],
                                  mode='bilinear', align_corners=False)
            pixel_probs = rand_pixel * scope
            rand_max = pixel_probs.flatten(2).argmax(2).flatten()
            seed = torch.empty((bs, self.latent_dim)).to(x.device)
            for bidx in range(bs):
                seed[bidx, :] = c_f.flatten(2)[bidx, :, rand_max[bidx]]
            seed_list.append(seed)
            # Compute masks
            if self.dist_kernel == 'laplacian':
                distance = euclidian_distance(c_f, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.dist_kernel == 'gaussian':
                distance = squared_distance(c_f, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.dist_kernel == 'epanechnikov':
                distance = squared_distance(c_f, seed)
                alpha = (1 - distance / self.log_sigma.exp()).relu()
            else:
                raise ValueError("No valid kernel.")
            alpha = alpha.unsqueeze(1)
            # Sanity checks
            #if debug:
            # assert alpha.max() <= 1, alpha.max()
            # assert alpha.min() >= 0, alpha.min()
            
            alpha = clamp_preserve_gradients(alpha, 0.01, 0.99)
            # SBP update
            log_a = torch.log(alpha)
            log_neg_a = torch.log(1 - alpha)
            log_m = log_s_k[step] + log_a
            #if dynamic_K and log_m.exp().sum() < 20:
             #   break
            log_m_k.append(log_m)
            log_s_k.append(log_s_k[step] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        # Accumulate stats
        stats = dict(feat=c_f, delta=delta, seeds=seed_list)
        return log_m_k, log_s_k, stats    
    
    

