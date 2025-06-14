from dataclasses import dataclass
from typing import List

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch import distributions as dists
from torch import nn

import numpy as np
from torch.nn import functional as F


from models.base_model import BaseModel
from models.shared.unet import UNet, GNConvBlock
from models.genesis.component_vae import ComponentVAE
from models.shared.geco import GECO
from models.genesis_v2.icsbp import ICSBP

## taken and/or adapted from genesis-v2 implementation of University of Oxford's Applied AI Lab's github account 
## accessed at: https://github.com/applied-ai-lab/genesis

class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)
    def forward(self, x):
        b_sz = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return self.pixel_coords(x)

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))
        self.g_1 = g_1.view((1, 1) + g_1.shape)
        self.g_2 = g_2.view((1, 1) + g_2.shape)
    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1).to(x.device)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1).to(x.device)
        return torch.cat((x, g_1, g_2), dim=1)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)


@dataclass(eq=False, repr=False)
class Genesis_v2(BaseModel):

    geco_goal_constant: float
    geco_step_size: float
    geco_alpha: float
    geco_init: float
    geco_min: float
    geco_speedup: float
    
    feat_dim: int
    pixel_bound: bool
    pixel_std: float
    
    encoder_params: DictConfig
    icsbp_params: DictConfig
    
    input_channels: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.mask_latent_size = self.feat_dim
        self.hidden_state_lstm = self.feat_dim
        
        self.encoder = UNet(self.input_channels,
                            self.feat_dim,
                            num_blocks=int(np.log2(self.height)-1), 
                            filter_start= min(self.feat_dim, 64), 
                            norm=self.encoder_params.norm)
        self.encoder.final_conv = nn.Identity()
        
        self.seg_head = GNConvBlock(self.feat_dim, self.feat_dim)
        
        self.icsbp_params.update(num_slots=self.num_slots)
        self.icsbp_params.update(feat_dim=self.feat_dim)
        self.icsbp = ICSBP(**self.icsbp_params)
        
        self.feat_head = nn.Sequential(
            GNConvBlock(self.feat_dim, self.feat_dim),
            nn.Conv2d(self.feat_dim, 2*self.feat_dim, 1))
        
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*self.feat_dim),
            nn.Linear(2*self.feat_dim, 2*self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.feat_dim, 2*self.feat_dim))
        
        self.h_broadcast, self.w_broadcast = self.height // 16, self.width // 16
        
        self.decoder = nn.Sequential(
            BroadcastLayer(self.height // 16),
            nn.ConvTranspose2d(self.feat_dim+2, 
                               self.feat_dim, 5, 2, 2, 1),
            nn.GroupNorm(8, self.feat_dim), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feat_dim, 
                               self.feat_dim, 5, 2, 2, 1),
            nn.GroupNorm(8, self.feat_dim), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feat_dim,
                               min(self.feat_dim, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(self.feat_dim, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(self.feat_dim, 64),
                               min(self.feat_dim, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(self.feat_dim, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(self.feat_dim, 64), 4, 1))
        
        self.prior_lstm = nn.LSTM(
            self.feat_dim, 4*self.feat_dim
        )
        self.prior_linear = nn.Linear(
            4*self.feat_dim, 2*self.feat_dim
        )

        self.geco_goal_constant *= 3 * self.width * self.height
        self.geco_step_size *= 64**2 / (self.width * self.height)
        self.geco_speedup = self.geco_speedup

        self.geco = GECO(
            self.geco_goal_constant,
            self.geco_step_size,
            self.geco_alpha,
            self.geco_init,
            self.geco_min,
            self.geco_speedup,
        )

    
    @property
    def slot_size(self) -> int:
        return self.feat_dim
    
    @staticmethod
    def sigma_parameterization(s: Tensor) -> Tensor:
        return (s + 4.0).sigmoid() + 1e-4
    
    def get_mask_recon_stack(self, m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_s = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == len(m_r_logits_k) - 1:
                    log_m_r_k.append(log_s)
                else:
                    log_a = F.logsigmoid(logits)
                    log_neg_a = F.logsigmoid(-logits)
                    log_m_r_k.append(log_s + log_a)
                    log_s = log_s +  log_neg_a
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")
    
    def decode_latents(self, z_k):
        # --- Reconstruct components and image ---
        x_r_k, m_r_logits_k = [], []
        for z in z_k:
            dec = self.decoder(z)
            x_r_k.append(dec[:, :3, :, :])
            m_r_logits_k.append(dec[:, 3: , :, :])
        # Optional: Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        # --- Reconstruct masks ---
        log_m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k
    """
    def x_loss(self, x, log_m_k, x_r_k, std, pixel_wise=False):
        # 1.) Sum over steps for per pixel & channel (ppc) losses
        p_xr_stack = dists.Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        err_ppc = -torch.log(log_mx.exp().sum(dim=4))
        # 2.) Sum accross channels and spatial dimensions
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))
    """
    
    def compute_recon_loss(
            self, x: Tensor, x_recon_comp: Tensor, log_masks: Tensor ) -> Tensor:
        
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        #print(x.shape)
        #print(x_recon_comp[0].shape)
        #print(log_masks[0].shape)
        #x = x.unsqueeze(1)
        #recon_dist = dists.Normal(x_recon_comp, self.pixel_std)
        recon_dist = dists.Normal(torch.stack(x_recon_comp, dim=4), self.pixel_std)
        log_p = recon_dist.log_prob(x.unsqueeze(4))
        log_masks = torch.stack(log_masks, dim=4)
        log_mx = log_p + log_masks
        log_mx = -log_mx.logsumexp(dim=4)  # over slots
        
        #print(log_mx.shape)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        return log_mx.mean(dim=0).sum()
    """
      # 1.) Sum over steps for per pixel & channel (ppc) losses
        p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        # TODO(martin): use LogSumExp trick for numerical stability
        err_ppc = -torch.log(log_mx.exp().sum(dim=4))
        # 2.) Sum accross channels and spatial dimensions
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))
    """
    def compute_mask_kl(self, qz_mask: Tensor, z_mask: Tensor) -> Tensor:
        bs = z_mask[0].shape[0]
        dev = z_mask[0].device

        pz_mask = [dists.Normal(torch.zeros(bs,self.feat_dim).to(dev), torch.ones(bs,self.feat_dim).to(dev))]
        #z_mask = torch.stack(z_mask, dim=0)
        zm_seq = torch.cat(
                [zm.view(1, bs, -1) for zm in z_mask[:-1]], dim=0)
        #rnn_input = z_mask[:-1]
        rnn_state, _ = self.prior_lstm(zm_seq)
        pz_mask_mu, pz_mask_sigma = self.prior_linear(rnn_state).chunk(2, dim=2)
        pz_mask_mu = pz_mask_mu.tanh()
        pz_mask_sigma = self.sigma_parameterization(pz_mask_sigma)

        #print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        #print(pz_mask_mu.shape)
        #print(pz_mask_sigma.shape)
        #First prior mask is N(0, 1). Here we append priors for all other slots.
        for i in range(self.num_slots - 1):
            pz_mask.append(dists.Normal(pz_mask_mu[i], pz_mask_sigma[i]))
        
        #print(len(z_mask))
        #print(len(pz_mask))
        #print(z_mask[0].shape)
        #print(pz_mask[1])
        # Compute KL for each slot and return the sum.
        mask_kl = torch.zeros(bs, device=zm_seq.device)
        for i in range(self.num_slots):
            #print(i)
            mask_kl = mask_kl + (
                qz_mask[i].log_prob(z_mask[i]) - pz_mask[i].log_prob(z_mask[i])
            ).sum(dim=1)
        #print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        mask_kl = mask_kl.mean(dim=0).sum()
        return mask_kl

    def forward(self, x: Tensor) -> dict:
        b, _, h, w = x.shape
        
        enc_f = self.encoder(x)
        enc_f = F.relu(enc_f)
        
        ## TODO: implement dynamic K version of ICSBP
        seg_f = self.seg_head(enc_f)
        log_m_k, log_s_k, attn_stats = self.icsbp(seg_f, self.num_slots-1)
        
        comp_stats = dict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        for log_m in log_m_k:
            mask = log_m.exp()
            # Masked sum
            obj_feat = mask * self.feat_head(enc_f)
            obj_feat = obj_feat.sum((2, 3))
            # Normalise
            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-5)
            # Posterior
            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            sigma = F.softplus(sigma_ps + 0.5) + 1e-8
            q_z = dists.Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats['mu_k'].append(mu)
            comp_stats['sigma_k'].append(sigma)
            comp_stats['z_k'].append(z)
            comp_stats['q_z_k'].append(q_z)
        
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        #print(comp_stats["q_z_k"][0])
        #print(comp_stats["z_k"][0].shape)
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats["z_k"])
        
        #log_m_k = torch.stack(log_m_k, dim=1)
        #log_m_r_k = torch.stack(log_m_r_k,dim=1)
        #x_r_k = torch.stack(x_r_k, dim=1)
        
        loss_recon = self.compute_recon_loss(x, x_r_k, log_m_r_k)        
        loss_mask_kl = self.compute_mask_kl(comp_stats["q_z_k"], comp_stats["z_k"])
        
        loss_value = self.geco.loss(loss_recon, loss_mask_kl)
        
        #mx_r_k = x_r_k*log_m_r_k
        
        #instance_seg=torch.argmax(log_m_k, dim=1),
        #instance_seg_r=torch.argmax(log_m_r_k, dim=1)
        
        #print(loss_value.shape, loss_value.dtype)
        #print(log_m_r_k.shape, log_m_r_k.dtype)
        #print(log_m_k.shape, log_m_k.dtype)
        #print(x_r_k.shape, x_r_k.dtype)

        #print(torch.stack(comp_stats["mu_k"],dim=1).shape, torch.stack(comp_stats["mu_k"],dim=1).dtype)
        return {
            "loss": loss_value,  # scalar
            "mask": torch.stack(log_m_r_k, dim=1).exp(), # (B, slots, 1, H, W)
            "attn": torch.stack(log_m_k, dim=1).exp(), # (B, slots, 1, H, W)
            "slot": torch.stack(x_r_k, dim=1),  # (B, slots, 3, H, W)
            "representation": torch.stack(comp_stats["mu_k"],dim=1),  # (B, slots, total latent dim)
            
            "log_s_k" : log_s_k, # (B, 1, 1, H, W)
            "attn_stats": attn_stats,
            "comp_stats": comp_stats,
            #"instance_seg_r": instance_seg_r, # (B, slots, 1, H, W)
            #"instance_seg": instance_seg, # (B, slots, 1, H, W)
    
            "z": comp_stats["z_k"],
            "kl_loss": loss_mask_kl,
            "recon_loss": loss_recon,
            "GECO beta": self.geco.beta,
        }
        
