from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, Type

import torch
from torch import Tensor, nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

from models.base_model import BaseModel
from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding


class EncoderConfig(Dict):
    width: int
    height: int
    input_channels: int = 3
    channels: int
    kernel_size: int
    stride: int
    padding:int
    output_channels:int
    
class ModelConfig(Dict):
    temps: List[int]
    channels: List[int]
    num_attns: List[int]
    attn_q_kernels: List[int]
    attn_k_kernels: List[int]
    attn_q_strides: List[int]
    attn_k_strides: List[int]
    attn_q_paddings: List[int]
    attn_k_paddings: List[int]
    num_clusters: List[int]
    kernels: List[List[int]]
    strides: List[List[int]]
    paddings: List[List[int]]

class DecoderConfig(Dict):
    input_channels: int = 3
    width: int
    height: int
    conv_tranposes: List[bool]
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]

class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        input_channels: int = 3,
        channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_channels: int = 32
    ):
        super().__init__()
        self.width = width
        self.height = height        
        
        self.padding = padding
            
        self.conv_bone = nn.Sequential(
            nn.Conv2d(input_channels, 
                      channels, 
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=0,
                      bias = False),
            nn.GroupNorm(8, channels, affine=False, eps=1e-3),
            nn.LeakyReLU(),
            nn.Conv2d(channels, 
                      channels, 
                      kernel_size=1,
                      stride=1, 
                      padding=0,
                      bias = False))    
        
        self.pos_embedding = PositionalEmbedding(
                height, width, channels)
        
        self.lnorm_1 = nn.GroupNorm(8, channels, affine=False, eps=1e-3)
        
        self.conv_fc_1x1 = nn.Conv1d(channels, 
                              output_channels, 
                              kernel_size=1, 
                              bias=False) 
        
        self.conv_mlp_1x1 = nn.Sequential(
            nn.Conv1d(channels, 
                      output_channels, 
                      kernel_size=1,
                      bias = False
                      ),
            nn.LeakyReLU(),
            nn.Conv1d(output_channels, 
                      output_channels, 
                      kernel_size=1,
                      bias = False
                      ))    
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            #if isinstance(m, (nn.Conv1d)):
            #    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        p = self.padding
        x = nn.functional.pad(x, (p,p,p,p), 'replicate')
        conv_output = self.conv_bone(x)
        #conv_output = self.lnorm_1(conv_output.flatten(2,3)).view(x.size(0), -1, conv_output.size(2), conv_output.size(3))
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm_1(conv_output)
        return self.conv_fc_1x1(conv_output) + self.conv_mlp_1x1(conv_output)
    
class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 4),
        kernels: List[int] = (5, 5, 5, 3),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 1),
        output_paddings: List[int] = (0, 0, 0, 0),
        conv_transposes: List[bool] = tuple([False] * 4),
        activations: List[str] = tuple(["relu"] * 4),
    ):
        super().__init__()
        self.conv_bone = []
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes:
            assert len(channels) == len(output_paddings)
        self.pos_embedding = PositionalEmbedding(height, width, input_channels)
        self.width = width
        self.height = height

        self.conv_bone = make_sequential_from_config(
            input_channels,
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pos_embedding(x)
        output = self.conv_bone(x)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask


def scaled_euc_distance(x, y, p=2, scale=None, sqrt=False):
    d = x.size(-1)
    dist = x.unsqueeze(-2) - y.unsqueeze(-3)
    dist = torch.sum(dist**2,dim=-1)
    if scale is not None:
        dist = scale * dist
    if sqrt:
        dist = torch.sqrt(dist)
    return dist

def clamp_preserve_gradients(x, lower, upper):
    # From: http://docs.pyro.ai/en/0.3.3/_modules/pyro/distributions/iaf.html
    return x + (x.clamp(lower, upper) - x).detach()
    
class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, bias = True):
        super(MLP, self).__init__()
        
        self.FC_1 = nn.Linear(input_dims, hidden_dims,bias=bias)
        self.FC_2 = nn.Linear(hidden_dims, output_dims,bias=bias)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
                    
    def forward(self, x):
        x = self.FC_2(F.leaky_relu(self.FC_1(x)))
        return x
    
class GroupNorm(nn.Module):
    def __init__(self, input_dims, num_groups = 4, affine=True, eps=1e-3):
        super(GroupNorm, self).__init__()
        
        self.input_dims = input_dims
        self.num_groups = num_groups
        self.eps = eps

        self.GN = nn.GroupNorm(num_groups, input_dims, affine=affine, eps=eps)    
                    
    def forward(self, x):
        b,n,t,d = x.shape
        x = torch.reshape(x, (b,n*t,d)).transpose(-1,-2)
        x = self.GN(x)
        return torch.reshape(x.transpose(-1,-2), (b,n,t,d))
    
class SelfAttention(nn.Module):
    def __init__(self, index, input_dims, output_dims, num_heads=4):
        super(SelfAttention, self).__init__()
        
        self.index = index
        self.input_dims = input_dims     
        self.output_dims = output_dims  
        self.num_heads = num_heads
            
        self.W_Q = nn.Linear(output_dims, output_dims, bias=False)
        self.W_K = nn.Linear(output_dims, output_dims, bias=False)
        self.W_V = nn.Linear(output_dims, output_dims, bias=False)
        self.W_Z = nn.Linear(output_dims, output_dims)
        
        self.MLP = MLP(output_dims,output_dims,output_dims)
        self.LN_1  = GroupNorm(output_dims, affine=False)
        self.LN_2  = GroupNorm(output_dims, affine=False)
        self.LN_3  = GroupNorm(output_dims, affine=False)
        self.LN_4  = GroupNorm(output_dims, affine=False)


        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.)
        
    def forward(self, ind, in_f, in_f_kv):
        
        b,n,t,d = in_f.shape
                
        _in_f = self.LN_1(in_f)
        _in_f_kv = self.LN_2(in_f_kv)
        
        q = self.W_Q(_in_f)
        k = self.W_K(_in_f_kv)
        v = self.W_V(_in_f_kv)
        
        q = self.LN_3(q)
        k = self.LN_4(k)

        q, k, v = map(lambda z: torch.reshape(z,(b,n,-1,self.num_heads,q.size(-1)//self.num_heads))\
                                          .permute(0,1,3,2,4), [q, k, v])

        q = (1/(np.sqrt(q.size(-1)))) * q
        attn_logits = torch.matmul(q, k.transpose(-1,-2))
        
        attn_weights = F.softmax(attn_logits,dim=-1)
        in_f_u1 = torch.reshape(torch.matmul(attn_weights, v)\
                                .permute(0,1,3,2,4),(b,n,t,-1))
        in_f_u1 = self.W_Z(in_f_u1)
        in_f_u2 = self.MLP(_in_f)
        out_f = in_f + in_f_u1 + in_f_u2

        return out_f, attn_weights
    
class COCA(nn.Module):
    
    def __init__(self, temp, scale, hidden_dims, out_cl):
        super(COCA, self).__init__()
        
        self.temp, self.scale = temp, scale
        self.hidden_dims, self.out_cl = hidden_dims, out_cl
    
        self.LN = GroupNorm(hidden_dims, affine=False)
        self.MLP = MLP(hidden_dims, hidden_dims, hidden_dims)
        self.FC = nn.Linear(hidden_dims, hidden_dims)
        
        self.FC_Q = nn.Linear(hidden_dims, hidden_dims, bias=False)
        self.LN_Q = GroupNorm(hidden_dims, affine=False)
        self.FC_V = nn.Linear(hidden_dims, hidden_dims, bias=False)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)


    def calc_compactness_scores(self, mask, dens, mass, area, iner,\
                                                        qcor, kcor):

        ## parallel compactness scoring on a batch of affinity masks
        ## broadcasts physical attributes, scales them with affinity masks and then
        ## computes compactness scores according to Eq 3 of the original paper
        b,nw,nqn,nkn = mask.shape
        dens = mask * dens.transpose(-1,-2)
        area = mask * area.transpose(-1,-2)
        mass = dens * area
        iner = dens * iner.transpose(-1,-2)
        dist = scaled_euc_distance(qcor, kcor)

        newiner1 = torch.sum(iner,dim=-1,keepdim=True)
        newiner2 = torch.sum(mass*dist,dim=-1,keepdim=True)
        newiner = newiner1 + newiner2

        refiner1 = torch.sum(mass*area, dim=-1,keepdim=True)
        refiner2 = torch.sum(torch.tril(2*\
                    torch.minimum(dens.unsqueeze(-1),dens.unsqueeze(-2))\
                    *(area.unsqueeze(-1) * area.unsqueeze(-2)),diagonal=-1),
                             dim=(-1,-2)).unsqueeze(-1)
        refiner = (refiner1 + refiner2)/(2*np.pi)
        comp = refiner/newiner

        return comp, newiner
    
    def compact_shape_clustering(self, af_m, in_dens, in_mass, in_area,
                                         in_iner, in_qcor, in_kcor, eps=1e-8):
        ## Sequential clustering inspired by SBC
        
        b,nw,nqn,nkn = af_m.shape
        log_sc_m_list = [torch.zeros(b,nw,1,nkn,device=af_m.device)] ## scope initalized
        log_cl_m_list = [] ## cluster masks list initalized
        cl_r_list = []  ## anchor nodes list initalized
        af_m = clamp_preserve_gradients(af_m,0.00001,0.99999)

        ## compute the compactness of all affinity masks, for once
        comp, _ = self.calc_compactness_scores(af_m, in_dens, in_mass, in_area, in_iner, in_qcor, in_kcor)

        ## algorithm 1 from the supplementary material
        for i in range(self.out_cl-1):
            sc_tm = log_sc_m_list[i].exp()
            comp_sc_tm = comp * sc_tm.transpose(-1,-2)
            cl_tr = comp_sc_tm.squeeze(-1).argmax(dim=-1).view(b,nw,1,1)
            cl_tm = torch.gather(af_m, -2, cl_tr.repeat(1,1,1,nkn)) 
            log_cl_tm = torch.log(cl_tm)
            log_n_cl_tm = torch.log(1-cl_tm)
            log_cl_m_list.append(log_sc_m_list[i] + log_cl_tm)
            log_sc_m_list.append(log_sc_m_list[i] + log_n_cl_tm)
            cl_r_list.append(cl_tr)


        ## physical attributes pooling
        sc_lm = log_sc_m_list[-1].exp()
        cl_r = torch.stack(cl_r_list, dim=-2).squeeze(2)
        cl_m = torch.cat([torch.exp(torch.stack(log_cl_m_list, dim=-2).squeeze(2)), sc_lm], dim=-2)
        cl_dens = cl_m * in_dens.transpose(-1,-2)
        cl_area = cl_m * in_area.transpose(-1,-2)
        cl_mass = cl_dens * cl_area
        cl_coor = torch.sum(cl_mass.unsqueeze(-1)*in_kcor.unsqueeze(-3),dim=-2)/(torch.sum(cl_mass,dim=-1,keepdim=True)+eps)
        cl_area = torch.sum(cl_area, dim=-1, keepdim=True)
        cl_mass = torch.sum(cl_mass, dim=-1, keepdim=True)
        cl_dens = (cl_mass / (cl_area+eps))

        ## compute the final compactness and inertia measuruments for output masks
        cl_comp, cl_iner = self.calc_compactness_scores(cl_m, in_dens, in_mass,\
                                                           in_area, in_iner, cl_coor, in_kcor)
        return cl_m, cl_dens, cl_mass, cl_area, cl_iner, cl_coor, cl_comp, cl_r, sc_lm
    
    def forward(self, in_f, in_geo):
        
        b = in_f.size(0)
        in_dens, in_mass, in_area, in_iner, in_coor = torch.split(in_geo, (1,1,1,1,2), dim=-1)

        _in_f = self.LN(in_f) ## pre-norm skip connect
       
        _in_f_m = self.MLP(_in_f)   ## pre-norm skip connect
        _in_f_v = self.FC_V(_in_f)  ## value projection
        _in_f_q = self.FC_Q(_in_f)  ## query projection
        _in_f_q = self.LN_Q(_in_f_q) ## groupnorm on query project
        
        to_cl = _in_f_q

        ## affinity masks calculation
        scl = (self.temp/self.scale)*(1/np.sqrt(_in_f_q.size(-1)))
        affn_dists = scaled_euc_distance(_in_f_q, _in_f_q, scale=scl)
        af_m = F.softmin(affn_dists, dim=-1)
        af_m_max,_ = torch.max(af_m, dim=-1, keepdim=True) 
        af_m_min,_ = torch.min(af_m, dim=-1, keepdim=True) 
        af_m = (af_m-af_m_min) / (af_m_max-af_m_min+1e-8)  

        
        ## run compactness scoring and sequential clustering algorithm
        cl_m, cl_dens, cl_mass, cl_area, cl_iner, cl_coor,\
                            cl_comp, cl_r, sc_m = self.compact_shape_clustering(af_m,\
                                                                in_dens, in_mass, in_area,\
                                                                in_iner, in_coor, in_coor)   
        
        ## pool features according to cluster masks
        cl_m_sum = torch.sum(cl_m, dim=-1,keepdim=True) + 1e-8
        cl_m_mean = cl_m * (1/cl_m_sum)
        cl_f = torch.matmul(cl_m_mean, in_f)
        cl_f_u = torch.matmul(cl_m_mean, _in_f_m) + self.FC(torch.matmul(cl_m_mean, _in_f_v))
        cl_f = cl_f + cl_f_u
            
        cl_geo = torch.cat([cl_dens, cl_mass, cl_area, cl_iner, cl_coor], dim=-1)  
        
        return cl_f, cl_m, cl_geo, cl_r, cl_comp, af_m, sc_m, to_cl
        
class ClusAttention2d(nn.Module):
    
    def __init__(self, in_res, temp, index, max_index, 
                         num_attns,
                         attn_q_kernels, attn_k_kernels,
                         attn_q_strides, attn_k_strides,
                         attn_q_paddings, attn_k_paddings,
                         input_dims, output_dims, 
                         in_cl, in_ks, in_st, in_pd,
                         out_cl, out_ks, out_st, out_pd,
                         skip_cl=-1):
        
        super(ClusAttention2d, self).__init__()
        
        self.in_res, self.temp =  in_res, temp 
        self.index, self.max_index = index, max_index
        self.num_attns = num_attns
        self.attn_q_ks, self.attn_k_ks = attn_q_kernels, attn_k_kernels
        self.attn_q_st, self.attn_k_st = attn_q_strides, attn_k_strides
        self.attn_q_pd, self.attn_k_pd = attn_q_paddings, attn_k_paddings
        self.input_dims, self.output_dims = input_dims, output_dims
        self.in_cl, self.in_ks, self.in_st, self.in_pd = in_cl, in_ks, in_st, in_pd
        self.out_cl, self.out_ks, self.out_st, self.out_pd = out_cl, out_ks, out_st, out_pd
        self.skip_cl = skip_cl    

        ## initialize vit-22b stack
        sa_list = []
        for i in range(num_attns):
            sa_list.append(SelfAttention(i,output_dims, output_dims))
        self.SA = nn.ModuleList(sa_list)

        self.num_nodes = np.prod(out_ks) * in_cl
        self.scale = np.sqrt(np.prod(out_ks))
        self.num_winds = (self.in_res[0] // self.out_ks[0], self.in_res[1] // self.out_ks[1]) 

        ## initialize physical attributes
        if index == 1:
            in_dens = torch.ones(1,np.prod(self.num_winds),np.prod(self.out_ks),1)
            in_mass = torch.ones(1,np.prod(self.num_winds),np.prod(self.out_ks),1)
            in_area = torch.ones(1,np.prod(self.num_winds),np.prod(self.out_ks),1)
            in_iner = torch.ones(1,np.prod(self.num_winds),np.prod(self.out_ks),1)*(1/6)
            in_coor = self.init_coors()
            in_geo = torch.cat([in_dens, in_mass, in_area, in_iner, in_coor], dim=-1)
            self.register_buffer("in_geo", in_geo)
        else:
            self.LN_S1 = GroupNorm(output_dims, affine=False)
            self.MLP_S1 = MLP(output_dims, output_dims, output_dims)
            self.LN_S2 = GroupNorm(output_dims, affine=False)

        self.COCA = COCA(self.temp, self.scale, self.output_dims, self.out_cl)
        
        
    def init_coors(self):
        window_x_coor = torch.arange(0,self.out_ks[1]).view(1,-1)\
                        .repeat(self.out_ks[0],1).view(1,1,np.prod(self.out_ks),1)
        window_y_coor = torch.arange(0,self.out_ks[0]).view(-1,1)\
                        .repeat(1,self.out_ks[1]).view(1,1,np.prod(self.out_ks),1)
        
        x_correct = (self.out_st[1]*torch.arange(self.num_winds[1])).view(1,-1).\
                                                repeat(self.num_winds[0],1).view(1,-1,1,1)
        y_correct = (self.out_st[0]*torch.arange(self.num_winds[0])).view(-1,1).\
                                                repeat(1,self.num_winds[1]).view(1,-1,1,1)
        x_coor = window_x_coor + x_correct
        y_coor = window_y_coor + y_correct
        coors = torch.cat([x_coor, y_coor],dim=-1)
        return coors
    
    
    def unfold(self, x, in_res, ks = [3,3], st = [1,1], pd = [0,0], pad_mode=None):
        b,n,t,d = x.shape
        x = torch.reshape(x.permute(0,2,3,1), (b,t*d, in_res[0],in_res[1]))
        if pad_mode == 'same':
            x = nn.functional.pad(x, (pd[0],pd[1],pd[0],pd[1]), 'replicate')
            x_unf = F.unfold(x, ks, stride=st, padding=0)
        else:
            x_unf = F.unfold(x, ks, stride=st, padding=pd)
        x_unf = torch.reshape(x_unf.permute(0,2,1), (b,-1,t,d,ks[0]*ks[1])).permute(0,1,4,2,3)
        ## B,NW,OKS**2,CL,D
        return x_unf
    
    def forward(self, in_f, in_m=None, in_geo=None, in_f_s=None):
    
        b = in_f.size(0)

        ##  feature refinement
        for i in range(self.num_attns):
            in_f_q = self.unfold(in_f, self.in_res, [self.attn_q_ks,self.attn_q_ks],
                                                     [self.attn_q_st,self.attn_q_st], 
                                                     [self.attn_q_pd,self.attn_q_pd])
            
            in_f_kv = self.unfold(in_f, self.in_res, [self.attn_k_ks,self.attn_k_ks],
                                                     [self.attn_k_st,self.attn_k_st], 
                                                     [self.attn_k_pd,self.attn_k_pd], 
                                                      pad_mode='same')
            
            in_f_q = torch.reshape(in_f_q, (b, in_f_q.size(1), -1, in_f_q.size(-1)))
            in_f_kv = torch.reshape(in_f_kv, (b, in_f_kv.size(1), -1, in_f_kv.size(-1)))
            
            in_f, _ = self.SA[i](i, in_f_q, in_f_kv)
            
            nwa0 = int(self.in_res[0] // self.attn_q_ks)
            nwa1 = int(self.in_res[1] // self.attn_q_ks)
            in_f = torch.reshape(in_f, (b,nwa0,nwa1,self.attn_q_ks,self.attn_q_ks,self.in_cl,-1)).permute(0,1,3,2,4,5,6)
            in_f = torch.reshape(in_f, (b,np.prod(self.in_res),self.in_cl,-1))
                                        
        ##  non-overlapping windows partitioning
        if self.index == 1: 
            in_f = self.unfold(in_f, self.in_res, self.out_ks, self.out_st, self.out_pd)
            in_f = torch.reshape(in_f, (b, in_f.size(1), -1, in_f.size(-1)))
            in_geo = self.in_geo.to(in_f.device).repeat(b,1,1,1)
        else:
            in_f, in_m, in_geo, in_f_s = map(lambda z: self.unfold(z, 
                                        self.in_res, self.out_ks,self.out_st, self.out_pd),
                                             [in_f, in_m, in_geo,in_f_s])
            in_f, in_geo = map(lambda z: torch.reshape(z, 
                                        (z.size(0), z.size(1), -1, z.size(-1))),
                                            [in_f, in_geo])
       
        _, nw, _, _ = in_f.shape
        
        ## affinity masks generation, compactness scoring and sequential clustering is carried out in COCA submodule
        cl_f, cl_m, cl_geo, cl_r, cl_comp, af_m, sc_m, to_cl = self.COCA(in_f, in_geo)

        
        if self.index > 1:

            ##  mask merging
            cl_m_ = torch.reshape(cl_m, (b,nw,self.out_cl,np.prod(self.out_ks),self.in_cl)).permute(0,1,3,2,4)
            cl_m_merge = torch.matmul(cl_m_, in_m)
            cl_m_merge = cl_m_merge.unsqueeze(-1)

            ##  inter-layer skip connections
            in_f_s = in_f_s.unsqueeze(-3)

            cl_m_merge, in_f_s = map(lambda z:
                          torch.reshape(z, (b, nw, self.out_ks[0], self.out_ks[1],
                                            z.size(3), self.in_ks[0], self.in_ks[1], 
                                            self.skip_cl, -1)).permute(0,1,4,2,5,3,6,7,8),
                                            [cl_m_merge.unsqueeze(-1), in_f_s.unsqueeze(-3)])

            cl_m_merge = torch.reshape(cl_m_merge, (b,nw,self.out_cl,
                                                  (self.out_ks[0]*self.in_ks[0]*\
                                                   self.out_ks[1]*self.in_ks[1])*self.skip_cl))
            in_f_s =  torch.reshape(in_f_s, (b,nw,(self.out_ks[0]*self.in_ks[0]*\
                                                   self.out_ks[1]*self.in_ks[1])*self.skip_cl, -1))
                       
            cl_m_merge_sum = torch.sum(cl_m_merge, dim=-1,keepdim=True) + 1e-6
            cl_m_merge_mean = cl_m_merge * (1/cl_m_merge_sum)
            
            cl_f_merge = self.LN_S1(torch.matmul(cl_m_merge_mean, in_f_s))
            cl_f_merge_u = self.MLP_S1(cl_f_merge)
            cl_f_merge = self.LN_S2(cl_f_merge + cl_f_merge_u)
            cl_f = cl_f + cl_f_merge
                    
        return cl_f, cl_m, cl_geo, in_f, cl_r, cl_comp, af_m, sc_m, to_cl
        
        

class ClusAttentionNet(nn.Module):
    def __init__(self, 
                 resolution: List[int] = [32,32],
                 temps: List[int] = [1, 1],
                 input_channels: int = 64,
                 channels: List[int] = [1, 64, 64], 
                 num_attns: List[int] = [2, 2],
                 attn_q_kernels:  List[int] = [1, 1],
                 attn_k_kernels:  List[int] = [3, 5],
                 attn_q_strides:  List[int] = [1, 1],
                 attn_k_strides:  List[int] = [1, 1],
                 attn_q_paddings:  List[int] = [0, 0],
                 attn_k_paddings:  List[int] = [1, 2],
                 num_clusters: List[int] = [1, 2, 4],
                 kernels: List[List[int]] = [[1,1], [4,4], [8,8]], 
                 strides: List[List[int]] = [[1,1], [4,4], [1,1]], 
                 paddings: List[List[int]] = [[0,0], [0,0], [0,0]]):
  
        super(ClusAttentionNet, self).__init__()
       
        self.num_layers = len(channels)-1
        index = 1
        res_list = [resolution]
        clus_attn_list =  []
        
        clus_attn_list.append(ClusAttention2d(res_list[0], temps[0], 
                                             index, self.num_layers,
                                             num_attns[0], 
                                             attn_q_kernels[0], attn_k_kernels[0], 
                                             attn_q_strides[0], attn_k_strides[0],
                                             attn_q_paddings[0], attn_k_paddings[0],
                                             input_channels, channels[1],
                                             num_clusters[0], kernels[0],
                                             strides[0], paddings[0],
                                             num_clusters[1], kernels[1],
                                             strides[1], paddings[1]
                                             ))
        
        res_list.append(self.calc_out_res(res_list[0], kernels[1], strides[1], paddings[1]))
        index = index+1
        
        for i in range(1, self.num_layers):
            clus_attn_list.append(ClusAttention2d(res_list[i], temps[i], 
                                                  index, self.num_layers, 
                                                  num_attns[i],
                                                  attn_q_kernels[i], attn_k_kernels[i], 
                                                  attn_q_strides[i], attn_k_strides[i],
                                                  attn_q_paddings[i], attn_k_paddings[i],
                                                  channels[i], channels[i+1],
                                                  num_clusters[i], kernels[i],
                                                  strides[i], paddings[i],
                                                  num_clusters[i+1],  kernels[i+1],
                                                  strides[i+1], paddings[i+1],
                                                  num_clusters[i-1]))
            index=index+1
            res_list.append(self.calc_out_res(res_list[i], kernels[i+1], strides[i+1], paddings[i+1]))
        
        self.clus_attns = nn.ModuleList(clus_attn_list)
        self.res_list = res_list

        self.LN_ENC = GroupNorm(channels[-1],affine=False)
        self.MLP_ENC = MLP(channels[-1],channels[-1],channels[-1],bias=False)
        
    def calc_out_res(self, in_res, ks, st, pd):
        return [int(((in_res[0]-ks[0]+2*pd[0])/st[0])+1), int(((in_res[1]-ks[1]+2*pd[1])/st[1])+1)]
        
    @torch.no_grad()
    def merge_cluster_masks(self, cl_m_list):
        ### Dendrogram generation by progressively merging cluster masks from each COCA layer.
        cl_m_1_0__ = cl_m_list[0]
        out = []
        for i in range(self.num_layers-1):
            L0 = self.clus_attns[i]
            L = self.clus_attns[i+1]
            cl_m_0 = cl_m_1_0__
            cl_m_1 = cl_m_list[i+1]
            cl_m_0_unf = L.unfold(cl_m_0, self.res_list[i+1], L.out_ks, L.out_st, L.out_pd)
            cl_m_1_ = torch.reshape(cl_m_1, (cl_m_1.size(0),cl_m_1.size(1),\
                        cl_m_1.size(2),np.prod(L.out_ks),-1)).permute(0,1,3,2,4)

            cl_m_1_0 = torch.matmul(cl_m_1_, cl_m_0_unf)
            cl_m_1_0_ = torch.reshape(cl_m_1_0, (cl_m_1_0.size(0), cl_m_1_0.size(1),\
                                        L.out_ks[0],L.out_ks[1], L.out_cl,\
                                        L0.in_ks[0]*L.in_ks[0], L0.in_ks[1]*L.in_ks[1])).permute(0,1,4,2,5,3,6)
            cl_m_1_0__ = torch.reshape(cl_m_1_0_, (cl_m_1_0_.size(0), cl_m_1_0_.size(1),\
                                        cl_m_1_0_.size(2), (L0.in_ks[0]*L.out_ks[0]*L.in_ks[0]*\
                                                            L0.in_ks[1]*L.out_ks[1]*L.in_ks[1])))
            out.append(cl_m_1_0__)
        return out
    
    def forward(self, x):
        out = []
        cl_f_0, cl_m_0, cl_geo_0, in_f_0, cl_r_0, cl_comp_0, af_m_0, sc_m_0, to_cl_0  = self.clus_attns[0](x)
                            
        out.append([cl_f_0, cl_m_0, cl_geo_0, in_f_0, cl_r_0, cl_comp_0,  af_m_0, sc_m_0, to_cl_0])
        
        if self.num_layers > 1:
            for i in range(1,self.num_layers):
                
                cl_f_1, cl_m_1, cl_geo_1, in_f_1, cl_r_1, cl_comp_1, af_m_1, sc_m_1, to_cl_1\
                        = self.clus_attns[i](cl_f_0, cl_m_0, cl_geo_0, in_f_0)
                
      
                out.append([cl_f_1, cl_m_1, cl_geo_1, in_f_1, cl_r_1, cl_comp_1, af_m_1, sc_m_1, to_cl_1])
                
                cl_f_0 = cl_f_1
                cl_m_0 = cl_m_1
                cl_geo_0 = cl_geo_1
                in_f_0 = in_f_1
            
        cl_f_1 = self.LN_ENC(cl_f_1)
        cl_f_1_u = self.MLP_ENC(cl_f_1)
        cl_f_1 = cl_f_1 + cl_f_1_u
                              
        return cl_f_1, out    


@dataclass(eq=False, repr=False)
class ClusAttentionAE(BaseModel):

    encoder_params: EncoderConfig
    model_params : ModelConfig
    decoder_params: DecoderConfig
    h_broadcast: int
    w_broadcast: int
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
        
        self.clus_attention = ClusAttentionNet(
            [self.height, self.width],
            self.model_params.temps,
            self.encoder_params["output_channels"],
            self.model_params.channels,
            self.model_params.num_attns,
            self.model_params.attn_q_kernels,
            self.model_params.attn_k_kernels,
            self.model_params.attn_q_strides,
            self.model_params.attn_k_strides,
            self.model_params.attn_q_paddings,
            self.model_params.attn_k_paddings,
            self.model_params.num_clusters,
            self.model_params.kernels,
            self.model_params.strides,
            self.model_params.paddings
        )
        
        self.decoder_params.update(
            height=self.h_broadcast,
            width=self.w_broadcast,
            input_channels=self.model_params.channels[-1],
        )
        self.decoder = Decoder(**self.decoder_params)

    @property
    def slot_size(self) -> int:
        return self.model_params.channels[-1]

    def spatial_broadcast(self, cluster: Tensor) -> Tensor:
        cluster = cluster.unsqueeze(-1).unsqueeze(-1)
        return cluster.repeat(1, 1, self.h_broadcast, self.w_broadcast)

    def forward(self, x: Tensor) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1).unsqueeze(-2)
        z, clus_out = self.clus_attention(encoded)
        z = z.squeeze(1)
        bs = z.size(0)
        clusters = z.flatten(0, 1)
        clusters = self.spatial_broadcast(clusters)
        img_clusters, masks = self.decoder(clusters)

        img_clusters = img_clusters.view(bs, self.num_slots, 3, self.height, self.width)
        masks = masks.view(bs, self.num_slots, 1, self.height, self.width)
        masks = masks.softmax(dim=1)
        
        recon_clusters_masked = img_clusters * masks
        recon_img = recon_clusters_masked.sum(dim=1)

        loss = self.loss_fn(x, recon_img)        
        with torch.no_grad():
            recon_clusters_output = (img_clusters + 1.0) / 2.0
            cms = [i[1]  for i in clus_out]
            attn_out = self.clus_attention.merge_cluster_masks(cms)
            attn_out = attn_out[-1].view(masks.shape)
        return {
            "loss": loss,  # scalar
            "mask": masks,  # (B, clusters, 1, H, W)
            "attn": attn_out,
            "slot": recon_clusters_output,  # (B, clusters, 3, H, W)
            "representation": z,  # (B, clusters, latent dim)
            "clus_attn_out" : clus_out, 
            "reconstruction": recon_img,  # (B, 3, H, W)
        }
