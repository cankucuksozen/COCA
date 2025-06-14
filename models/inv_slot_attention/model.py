from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import init
import numpy as np


from models.base_model import BaseModel
from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding
from models.shared.resnet import ResNet, BasicBlock, Bottleneck

    
## taken/adapted from google-research's official invariant_slot_attention
## repositories accessed at: 
## https://github.com/google-research/google-research/blob/master/invariant_slot_attention/

class EncoderConfig(Dict):
    encoder_type: str
    encoder_dict: dict
    channels: List[int]
    width: int
    height: int
    input_channels: int = 3
    
class ModelConfig(Dict):
    in_res_w: int
    in_res_h: int
    num_slots: int
    num_attn_iters: int
    input_dims: int
    slot_dims: int
    mlp_dims: int
    zero_position_init: bool
    add_rel_pos_to_values: bool
    inc_scale: bool

class DecoderConfig(Dict):
    slot_dims: int
    broadcast_res: List[int]
    inc_scale: bool
    conv_tranposes: List[bool]
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]


#################################################################################
#################################################################################

class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        input_channels: int = 3,
        channels: List[int] = (32, 32, 32, 32),
        encoder_type: str = "conv",
        encoder_dict: dict = {}
        ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "resnet":
            blockstr = encoder_dict["block"]
            if blockstr == "BasicBlock":
                blocktype = BasicBlock
            elif blockstr == "Bottleneck":
                blocktype = Bottleneck

            layers = encoder_dict["layers"]
            output_channels = channels[-1]
            self.conv_bone = ResNet(blocktype, channels, layers)
            
        elif encoder_type == "conv":
            kernels = encoder_dict["kernels"]
            strides = encoder_dict["strides"]
            paddings = encoder_dict["paddings"]
            batchnorms = encoder_dict["batchnorms"]
            activations = encoder_dict["activations"]
            
            assert len(kernels) == len(strides) == len(paddings) == len(channels) == len(activations)
            self.conv_bone = make_sequential_from_config(
                input_channels,
                channels,
                kernels,
                batchnorms,
                False,
                paddings,
                strides,
                activations,
                try_inplace_activation=True,
            )

    def forward(self, x: Tensor) -> Tensor:
        conv_output = self.conv_bone(x)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        return conv_output.permute(0,2,1)
    
    
def generate_pos_embed(in_res, limits = [-1.0, 1.0]):        
    samples_per_dim = [in_res[0], in_res[1]]
    s = [torch.linspace(limits[0], limits[1], n) for n in samples_per_dim]
    pos_embedding = torch.stack(torch.meshgrid(*s, indexing="ij"), dim=-1)
    return pos_embedding     
    
class Decoder(nn.Module):
    def __init__(
        self,
        slot_dims: int = 64,
        broadcast_res_w: int = 32,
        broadcast_res_h: int = 32,
        inc_scale = False,
        channels: List[int] = (32, 32, 32, 4),
        kernels: List[int] = (5, 5, 5, 3),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 1),
        output_paddings: List[int] = (0, 0, 0, 0),
        conv_transposes: List[bool] = tuple([False] * 4),
        activations: List[str] = tuple(["relu"] * 4),
    ):
        super().__init__()
        
        self.broadcast_res_w = broadcast_res_w
        self.broadcast_res_h = broadcast_res_h

        self.slot_dims = slot_dims
        self.inc_scale = inc_scale
        
        self.pos_embed = generate_pos_embed([broadcast_res_w, broadcast_res_h]).unsqueeze(0)
        self.D_PROJ = nn.Linear(2, slot_dims)
        
        if self.inc_scale:
            self.scales_factor = 1.0

        self.conv_bone = []
        
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes:
            assert len(channels) == len(output_paddings)

        self.conv_bone = make_sequential_from_config(
            slot_dims,
            channels,
            kernels,
            False,
            False,
            paddings,
            strides,
            activations,
            output_paddings,
            conv_transposes,
            try_inplace_activation=True,
        )

    def forward(self, slots_f, slots_pos, slots_scl=None):
        pos_embed = self.pos_embed.to(slots_f.device)
        slots_f = torch.reshape(slots_f, (-1, 1, 1, self.slot_dims,))\
                            .repeat(1, self.broadcast_res_w, self.broadcast_res_h,1)
        
        slots_pos = torch.reshape(slots_pos, (-1,1,1,2))
        if self.inc_scale:
            slots_scl = torch.reshape(slots_scl, (-1,1,1,2))
            
        pos_embed = pos_embed - slots_pos
        if self.inc_scale:
            pos_embed = pos_embed / self.scales_factor
            pos_embed = pos_embed / slots_scl
            
        #print(self.D_PROJ(pos_embed).shape)
        slots_f = slots_f + self.D_PROJ(pos_embed)
        slots_f = torch.permute(slots_f, (0,3,1,2))
            
        output = self.conv_bone(slots_f)
        img, mask = output[:, :3], output[:, -1:]
        
        return img, mask
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x
        

class ISA(nn.Module):
    def __init__(self, in_res_w, in_res_h, num_slots, num_attn_iters,
                    input_dims, slot_dims, mlp_dims,
                    zero_position_init,
                    add_rel_pos_to_values,
                    inc_scale,
                    ablate_non_equivariant = False
                    ):
        
        super(ISA, self).__init__()
        
        self.in_res_w,  self.in_res_w = in_res_w, in_res_h
        self.num_slots, self.num_attn_iters = num_slots, num_attn_iters
        self.input_dims, self.slot_dims, self.mlp_dims = input_dims, slot_dims, mlp_dims
        self.zero_position_init, self.ablate_non_equivariant = zero_position_init, ablate_non_equivariant
        self.add_rel_pos_to_values, self.inc_scale = add_rel_pos_to_values, inc_scale
        
        self.slot_feat_init_fn = nn.init.normal_
        self.slot_pos_init_fn = nn.init.uniform_
        self.slot_scl_init_fn = nn.init.normal_
        
        self.pos_max = 1.0
        self.pos_min = -1.0
        self.abs_grid = nn.Parameter(generate_pos_embed([in_res_w,in_res_h]).flatten(0,1).unsqueeze(0),requires_grad=False)
        self.slots_f = nn.Parameter(self.slot_feat_init_fn(torch.empty(1,self.num_slots,self.slot_dims)),requires_grad=True)
        self.slots_pos = self.slot_pos_init_fn(torch.empty(1,self.num_slots,2), a=self.pos_min, b=self.pos_max)
        
        if inc_scale:
            self.scl_mean = 0.1
            self.scl_std = 0.1
            self.min_scale = 0.001
            self.max_scale = 2.
            self.init_with_fixed_scale = 0.1
            self.scales_factor = 1.0
            self.slots_scl = self.scl_mean + (self.scl_std*self.slot_scl_init_fn(torch.empty(1,self.num_slots,2)))

        self.eps = 1e-8

        self.W_Q = nn.Linear(self.input_dims, self.slot_dims, bias=False)
        self.W_K = nn.Linear(self.input_dims, self.slot_dims, bias=False)
        self.W_V = nn.Linear(self.input_dims, self.slot_dims, bias=False)
        self.G_PRJ = nn.Linear(2, self.slot_dims)
        self.G_ENC = MLP(self.slot_dims, self.mlp_dims, self.slot_dims, False, "pre")
        self.FFN =  MLP(self.slot_dims, self.mlp_dims, self.slot_dims, True, "pre")

        self.LN_I = nn.LayerNorm(self.input_dims)
        self.LN_S = nn.LayerNorm(self.slot_dims)
        self.GRU = nn.GRUCell(self.slot_dims, self.slot_dims)
      
    def forward(self, inputs):
        slots_f = self.slots_f.to(inputs.device).repeat(inputs.size(0),1,1)
        abs_grid, slots_pos = self.abs_grid.to(inputs.device), self.slots_pos.to(inputs.device)
                
        if self.zero_position_init:
            slots_pos *= 0.
        slots_pos = torch.clip(slots_pos, -1., 1.)

        if self.inc_scale:
            slots_scl = self.slots_scl.to(inputs.device)
            if self.init_with_fixed_scale is not None:
                slots_scl = slots_scl * 0. + self.init_with_fixed_scale
            slots_scl = torch.clip(slots_scl, self.min_scale, self.max_scale)
        
        inputs = self.LN_I(inputs)
        
        grid_per_slot = abs_grid.unsqueeze(-3).repeat(1,self.num_slots,1,1)

        k = self.W_K(inputs)
        v = self.W_V(inputs)
        k_rep = k.unsqueeze(-3)
        v_rep = v.unsqueeze(-3)
        
        for i in range(self.num_attn_iters+1):
            if self.ablate_non_equivariant:
                tmp_grid = self.G_PRJ(grid_per_slot)
                k_rel_pos = self.G_ENC(k_rep + tmp_grid)
                if self.add_rel_pos_to_values:
                    v_rel_pos = self.G_ENC(v_rep + tmp_grid)
            else:
                rel_grid = grid_per_slot - slots_pos.unsqueeze(-2)
                if self.inc_scale:
                    rel_grid = rel_grid / self.scales_factor
                    rel_grid = rel_grid / slots_scl.unsqueeze(-2)
                tmp_grid = self.G_PRJ(rel_grid)
                k_rel_pos = self.G_ENC(k_rep + tmp_grid)
                if self.add_rel_pos_to_values:
                    v_rel_pos = self.G_ENC(v_rep + tmp_grid)
        
            slots_n = self.LN_S(slots_f)
            q = self.W_Q(slots_n)
            q = q / np.sqrt(self.slot_dims)

            attn = torch.einsum("bsd,bsnd->bsn", q, k_rel_pos)
            attn = nn.functional.softmax(attn, dim=-2)
            normalizer = torch.sum(attn, dim=-1, keepdim=True) + self.eps
            attn = attn/normalizer

            updates = torch.einsum("bsn,bsnd->bsd", attn, v_rel_pos)
            slots_pos = torch.einsum("bsn,bnp->bsp", attn, abs_grid)

            if self.inc_scale:
                slots_spread = torch.square(grid_per_slot - slots_pos.unsqueeze(-2))
                slots_scl = torch.sqrt(torch.einsum("bsn,bsnp->bsp", attn+self.eps, slots_spread))
                slots_scl = torch.clip(slots_scl, self.min_scale, self.max_scale)

            if i < self.num_attn_iters:
                slots_f_ = self.GRU(slots_f.view(-1,self.slot_dims), updates.view(-1,self.slot_dims))
                slots_f = torch.reshape(slots_f_, slots_f.shape)
                slots_f = self.FFN(slots_f)
            
        out_list = [slots_f, attn, slots_pos]
        if self.inc_scale:
            out_list.append(slots_scl)
            out_list.append(slots_spread)
            
        return out_list


@dataclass(eq=False, repr=False)
class InvariantSlotAttentionAE(BaseModel):

    encoder_params: EncoderConfig
    model_params : ModelConfig
    decoder_params: DecoderConfig
    input_channels: int = 3

    encoder: Encoder = field(init=False)
    decoder: Decoder = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = nn.MSELoss()
            
        self.encoder_params.update(
            width=self.width, height=self.height, input_channels=self.input_channels
        )
        self.encoder = Encoder(**self.encoder_params)
        
        self.model_params.update(
            num_slots = self.num_slots, input_dims=self.encoder_params["channels"][-1]
        )
        self.inv_slot_attention = ISA(**self.model_params)
        
        self.slot_dims = self.model_params.slot_dims
        self.num_tokens = self.model_params.in_res_w*self.model_params.in_res_h
        self.inc_scale = self.model_params.inc_scale
        
        self.decoder_params.update(slot_dims=self.slot_dims,
                                   broadcast_res_w=self.model_params.in_res_w, 
                                   broadcast_res_h=self.model_params.in_res_h,
                                   inc_scale = self.model_params.inc_scale)
        
        self.decoder = Decoder(**self.decoder_params)

    @property
    def slot_size(self) -> int:
        return self.slot_dims
    
    def forward(self, x: Tensor) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        isa_list = self.inv_slot_attention(encoded)
        if self.inc_scale:
            slots_f, attn, slots_pos, slots_scl, slots_spread = isa_list 
            img_slots, masks = self.decoder(slots_f, slots_pos, slots_scl)

        else:
            slots_f, attn, slots_pos = isa_list 
            img_slots, masks = self.decoder(slots_f, slots_pos)
        
        bs = slots_f.size(0)

        img_slots = img_slots.view(bs, self.num_slots, 3, self.height, self.width)
        masks = masks.view(bs, self.num_slots, 1, self.height, self.width)
        masks = masks.softmax(dim=1)

        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=1)
        loss = self.loss_fn(x, recon_img)

        _,num_slots,hw = attn.shape
        h = int(sqrt(hw))
        if h == masks.size(-1):
            attn = attn.view(masks.shape)
        else:
            attn = attn.view(bs,num_slots,h,h)
            attn = nn.functional.interpolate(attn, (masks.size(-2), masks.size(-1)), mode="bilinear")
        
        """
        attn = attn.view(bs,num_slots,-1)
        attn_min,_ = torch.min(attn,dim=-1,keepdim=True)
        attn_max,_ = torch.max(attn,dim=-1,keepdim=True)
        attn = (attn - attn_min)/(attn_max - attn_min)
        
        attn = attn.view(masks.shape)
        """ 
        
        with torch.no_grad():
            recon_slots_output = (img_slots + 1.0) / 2.0
        return {
            "loss": loss,  # scalar
            "mask": masks,  # (B, slots, 1, H, W)
            "attn": attn,
            "slot": recon_slots_output,  # (B, slots, 3, H, W)
            "representation": slots_f,  # (B, slots, latent dim)
            "isa_list": isa_list,
            "reconstruction": recon_img,  # (B, 3, H, W)
        }
