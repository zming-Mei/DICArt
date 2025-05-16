import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from tqdm import tqdm
import wandb
import math
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from configs.config import get_config
from utils.metrics import rot_diff_degree
from networks.gf_algorithms.discrete_angle import *
from datasets.dataloader import get_data_loaders_from_cfg, process_batch
from networks.gf_algorithms.discrete_number import *


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        if not self.same_channels:
            self.proj = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        if self.same_channels:
            return x + self.net(x)
        else:
            return self.proj(x) + self.net(x)
        
class D3PM(nn.Module):
    def __init__(self, cfg, num_bins=360, T=1000, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.num_bins = num_bins
        self.angle_dimensions = 3  
        self.translation_dimensions = 3  
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.T = T
        self.device = device
        self.mse_weight = 0.8
        self.L1_weight = 0.1
        self.ce_weight = 1
        self.vb_weight = 0.4
        self.eps = 1e-8

        steps = torch.arange(T + 1, dtype=torch.float64, device=device) / T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2)

        self.beta_t = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999)
        self.alpha_t = 1. - self.beta_t
        self.alpha_bar = torch.cumprod(self.alpha_t, dim=0)
        self.build_transition_matrices()

        # Point Cloud Feature Extractors
        if self.cfg.pts_encoder == 'pointnet':
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
        elif self.cfg.pts_encoder == 'pointnet2':
            self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            self.pts_pointnet = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2 = Pointnet2ClsMSG(0)
            self.fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        else:
            raise NotImplementedError

        self.time_embedder = TimestepEmbedder(hidden_size=256, frequency_embedding_size=128).to(device)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.num_dimensions*num_bins + 256 + 1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            ResidualBlock(1024, 1024),  # Added residual block
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.2)  # Increased dropout
        ).to(device)

        self.angles_branch = nn.Sequential(
            nn.Linear(512, 384),  # Wider
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        self.translation_branch = nn.Sequential(
            nn.Linear(512, 384),  # Wider
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        self.angle_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_bins)
            ) for _ in range(self.angle_dimensions)
        ]).to(device)
        
        self.translation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_bins)
            ) for _ in range(self.translation_dimensions)
        ]).to(device)


    def _extract(self, array, t_idx, x_shape=None):

        batch_size = t_idx.shape[0]   
        t_idx = torch.clamp(t_idx, min=0, max=len(array)-1)
        out = array[t_idx].to(t_idx.device)
        out = out.reshape(batch_size, 1)

        if x_shape is not None:
            for _ in range(len(x_shape) - len(out.shape)+1):
                out = out.unsqueeze(-1)  
            out = out.expand(batch_size, *x_shape[1:], 1)
        
        return out
    
    def extract_pts_feature(self, data):
        pts = data['pts']
        if self.cfg.pts_encoder == 'pointnet':
            return self.pts_encoder(pts.permute(0,2,1))
        elif self.cfg.pts_encoder == 'pointnet2':
            return self.pts_encoder(pts)
        
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            feat1 = self.pts_pointnet(pts.permute(0,2,1))
            feat2 = self.pts_pointnet2(pts)
            return self.fusion(torch.cat([feat1, feat2], dim=1))
        else:
            raise NotImplementedError

    def build_transition_matrices(self):
        q_onestep_mats = []
        for beta in self.beta_t:
            mat = torch.ones(self.num_bins, self.num_bins, device=self.device) * beta / self.num_bins
            mat.diagonal().fill_(1 - (self.num_bins - 1) * beta / self.num_bins)
            q_onestep_mats.append(mat)
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_sample(self, x_0, t, noise):
        logits = torch.log(self._at(self.q_mats, t - 1, x_0) + self.eps)  # [bs, num_bins]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))

        return torch.argmax(logits + gumbel_noise, dim=-1)  # [bs]

    
    def q_posterior_logits(self, x_0, x_t, t):
       
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(F.one_hot(x_0, self.num_bins).float() + self.eps)
        else:
            x_0_logits = x_0.clone()
        fact1 = self._at(self.q_one_step_transposed, t - 1, x_t)  # [bs, num_bins]
        softmaxed = F.softmax(x_0_logits, dim=-1)
        if t.min() > 1:
            qmats2 = self.q_mats[t - 2]
            fact2 = torch.einsum("bc,bcd->bd", softmaxed, qmats2)
        else:
            fact2 = softmaxed
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.reshape((t.shape[0], 1))
        return torch.where(t_broadcast == 1, x_0_logits, out)




    def model_predict(self, y_t, t, cond):
        bs = y_t.shape[0]
        y_t_onehot = F.one_hot(y_t, num_classes=self.num_bins).float()  
        y_t_flat = y_t_onehot.view(bs, -1)  
        t_emb = self.time_embedder(t)       # [bs, 128]
        input_feat = torch.cat([y_t_flat, t_emb, cond], dim=1)  
        
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        angles_feat = self.angles_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]
        angle_logits = [head(angles_feat) for head in self.angle_heads]       
        trans_logits = [head(trans_feat) for head in self.translation_heads]
        predicted_x0_logits = torch.stack(angle_logits + trans_logits, dim=1)
        return predicted_x0_logits

    def p_sample(self, y_t, t, cond, noise):
       
        predicted_x0_logits = self.model_predict(y_t, t, cond)  
        pred_q_posterior_logits = torch.stack(
            [self.q_posterior_logits(predicted_x0_logits[:, i, :], y_t[:, i], t)
             for i in range(self.num_dimensions)], dim=1
        )  
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 1).float().reshape((y_t.shape[0], 1, 1))
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)



    def sample(self, pts_feat, steps=1000):
        cond = pts_feat  # 直接使用 [bs, 1024]
        bs = pts_feat.shape[0]
        y_t = torch.randint(0, self.num_bins, (bs, self.num_dimensions), device=self.device)
        for step in reversed(range(1, steps)):
            t = torch.full((bs,), step, dtype=torch.long, device=self.device)
            y_t = self.p_sample(y_t, t, cond, torch.rand((bs, self.num_dimensions, self.num_bins), device=self.device))
        
        return y_t 
    

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)  
        dist2 = dist2.flatten(start_dim=0, end_dim=-2) 
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1) -
            torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()
        
    def loss(self, y, pts_feat):
        cond = pts_feat
        bs = y.shape[0]
        t = torch.randint(1, self.T, (bs,), device=self.device)
        noise = torch.rand((bs, self.num_bins), device=self.device)
        y_t = torch.stack([self.q_sample(y[:,i], t, noise) for i in range(self.num_dimensions)], dim=1)
        
        predicted_x0_logits = self.model_predict(y_t, t, cond)
        predicted_x0 = torch.argmax(predicted_x0_logits , dim=-1)
        # true_q_posterior = self.q_posterior_logits(y, y_t, t)
        # pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, y_t, t)

        true_q_posterior = torch.stack([self.q_posterior_logits(y[:,i], y_t[:,i], t) for i in range(self.num_dimensions)], dim=1)
        pred_q_posterior = torch.stack([self.q_posterior_logits(predicted_x0[:,i], y_t[:,i], t) for i in range(self.num_dimensions)], dim=1)

        vb_loss = self.vb(true_q_posterior, pred_q_posterior)
        angle_ce_loss = sum(F.cross_entropy(predicted_x0_logits[:,i], y[:,i]) for i in range(self.angle_dimensions)) / self.angle_dimensions
        trans_ce_loss = sum(F.cross_entropy(predicted_x0_logits[:,i+self.angle_dimensions], y[:,i+self.angle_dimensions]) 
                            for i in range(self.translation_dimensions)) / self.translation_dimensions
        ce_loss = (0.6*angle_ce_loss + 0.4*trans_ce_loss)
        #l1_loss = F.l1_loss(pred_bins, true_bins)
        pred_bins_arg = predicted_x0_logits.argmax(dim=-1)
        probs = F.softmax(predicted_x0_logits, dim=-1)
        bin_indices = torch.arange(self.num_bins, device=probs.device).float()  # shape: [num_bins]
        pred_bins = torch.sum(probs * bin_indices, dim=-1)  # shape: [bs, num_dimensions]
        true_bins = y.float()
        mse_loss = F.mse_loss(pred_bins, true_bins)
        L1_loss = F.l1_loss(pred_bins, true_bins)
        loss_description = (
            f"CE Loss Contribution: {self.ce_weight * ce_loss:.4f}, "
            f"VB Loss Contribution: {self.vb_weight * vb_loss:.4f}, "
            #f"mse Loss Contribution: {self.mse_weight * mse_loss:.4f}"
            f"L1 Loss Contribution: {self.L1_weight * L1_loss:.4f}"
        )
        total_loss = self.ce_weight*ce_loss  + self.L1_weight*L1_loss + self.vb_weight * vb_loss
        return  total_loss, loss_description

class D3PM_Flow(nn.Module):
    def __init__(self, cfg, num_bins=360, T=1000, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.num_bins = num_bins
        self.angle_dimensions = 3  
        self.translation_dimensions = 3  
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.T = T
        self.device = device
        self.mse_weight = 0.8
        self.L1_weight = 0.1
        self.ce_weight = 1
        self.vb_weight = 0.4
        self.eps = 1e-8

        steps = torch.arange(T + 1, dtype=torch.float64, device=device) / T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2)

        self.beta_t = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999)
        self.alpha_t = 1. - self.beta_t
        self.alpha_bar = torch.cumprod(self.alpha_t, dim=0)
        self.build_transition_matrices()

        # Point Cloud Feature Extractors
        if self.cfg.pts_encoder == 'pointnet':
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
        elif self.cfg.pts_encoder == 'pointnet2':
            self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            self.pts_pointnet = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2 = Pointnet2ClsMSG(0)
            self.fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        else:
            raise NotImplementedError

        self.time_embedder = TimestepEmbedder(hidden_size=256, frequency_embedding_size=128).to(device)
        
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.num_dimensions*num_bins + 256 + 1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            ResidualBlock(1024, 1024),  # Added residual block
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.2)  # Increased dropout
        ).to(device)

        self.angles_branch = nn.Sequential(
            nn.Linear(512, 384),  # Wider
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        self.translation_branch = nn.Sequential(
            nn.Linear(512, 384),  # Wider
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        self.angle_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_bins)
            ) for _ in range(self.angle_dimensions)
        ]).to(device)
        
        self.translation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, num_bins)
            ) for _ in range(self.translation_dimensions)
        ]).to(device)


    def _extract(self, array, t_idx, x_shape=None):

        batch_size = t_idx.shape[0]   
        t_idx = torch.clamp(t_idx, min=0, max=len(array)-1)
        out = array[t_idx].to(t_idx.device)
        out = out.reshape(batch_size, 1)

        if x_shape is not None:
            for _ in range(len(x_shape) - len(out.shape)+1):
                out = out.unsqueeze(-1)  
            out = out.expand(batch_size, *x_shape[1:], 1)
        
        return out
    
    def extract_pts_feature(self, data):
        pts = data['pts']
        if self.cfg.pts_encoder == 'pointnet':
            return self.pts_encoder(pts.permute(0,2,1))
        elif self.cfg.pts_encoder == 'pointnet2':
            return self.pts_encoder(pts)
        
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            feat1 = self.pts_pointnet(pts.permute(0,2,1))
            feat2 = self.pts_pointnet2(pts)
            return self.fusion(torch.cat([feat1, feat2], dim=1))
        else:
            raise NotImplementedError

    def build_transition_matrices(self):
        q_onestep_mats = []
        for beta in self.beta_t:
            mat = torch.ones(self.num_bins, self.num_bins, device=self.device) * beta / self.num_bins
            mat.diagonal().fill_(1 - (self.num_bins - 1) * beta / self.num_bins)
            q_onestep_mats.append(mat)
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_sample(self, x_0, t, noise):
        logits = torch.log(self._at(self.q_mats, t - 1, x_0) + self.eps)  # [bs, num_bins]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))

        return torch.argmax(logits + gumbel_noise, dim=-1)  # [bs]

    
    #flexible flow decider
    def q_posterior_logits(self, x_0, x_t, t):
        batch_size = x_0.shape[0]

        # Convert inputs to appropriate format if needed
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_one_hot = F.one_hot(x_0, self.num_bins).float()
        else:
            x_0_one_hot = x_0.clone()
        
        if x_t.dtype == torch.int64 or x_t.dtype == torch.int32:
            x_t_one_hot = F.one_hot(x_t, self.num_bins).float()
        else:
            x_t_one_hot = x_t.clone()
        
        t_idx = t - 1
        beta_t = self._extract(self.beta_t, t_idx ,x_t.shape)
        alpha_t = self._extract(self.alpha_t, t_idx,x_t.shape)
        alpha_t_minus_1 = self._extract(self.alpha_t, t_idx-1,x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bar, t_idx,x_t.shape)
        alpha_bar_t_minus_1 = self._extract(self.alpha_bar, t_idx-1,x_t.shape)
        
        noise = torch.rand((batch_size,self.num_bins), device=self.device)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        #q_noise = torch.softmax(gumbel_noise, dim=-1)

        q_noise = torch.softmax(noise, dim=-1)
        q_noise_xt = torch.sum(q_noise * x_t_one_hot, dim=-1, keepdim=True)

        q_noise_distribution_xt = beta_t * x_t_one_hot + (1 - beta_t) * q_noise
        lambda_1 = 1 - (1 - beta_t) * (1 - alpha_t_minus_1) * q_noise_xt / (alpha_t + (1 - alpha_t) * q_noise_xt + self.eps)
        lambda_2 = (alpha_bar_t_minus_1 - alpha_bar_t) / (1 - alpha_bar_t + self.eps)

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            is_equal = (x_t == x_0).unsqueeze(-1)
        else:
            is_equal = torch.argmax(x_t_one_hot, dim=-1) == torch.argmax(x_0_one_hot, dim=-1)
            is_equal = is_equal.unsqueeze(-1)
        
        # Case 1: x_t = x_0
        case1_probs = lambda_1 * x_t_one_hot + (1 - lambda_1) * q_noise
        # Case 2: x_t ≠ x_0
        case2_probs = lambda_2* x_0_one_hot + (1 - lambda_2) * q_noise_distribution_xt
        # Combine the two cases
        posterior_probs = torch.where(is_equal, case1_probs, case2_probs)
        posterior_logits = torch.log(posterior_probs + self.eps)

        # Special case for t=1 as in the original code
        t_broadcast = t.reshape((t.shape[0], 1))

        x_0_logits = torch.log(x_0_one_hot + self.eps)
        posterior_logits = torch.where(t_broadcast == 1, x_0_logits, posterior_logits)
        
        return posterior_logits




    def model_predict(self, y_t, t, cond):
        bs = y_t.shape[0]
        y_t_onehot = F.one_hot(y_t, num_classes=self.num_bins).float()  
        y_t_flat = y_t_onehot.view(bs, -1)  
        t_emb = self.time_embedder(t)       # [bs, 128]
        input_feat = torch.cat([y_t_flat, t_emb, cond], dim=1)  
        
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        angles_feat = self.angles_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]
        angle_logits = [head(angles_feat) for head in self.angle_heads]       
        trans_logits = [head(trans_feat) for head in self.translation_heads]
        predicted_x0_logits = torch.stack(angle_logits + trans_logits, dim=1)
        return predicted_x0_logits



    #flexible flow decider
    def p_sample(self, x_t, t, cond, noise):
        """Perform one step of the sampling process using the reparameterized backward transition"""
        batch_size = x_t.shape[0]
        device = x_t.device
        # Use the model to predict x_0
        with torch.no_grad():
            logits = self.model_predict(x_t, t, cond)  # [bs, num_dimensions, num_bins]
        probs = F.softmax(logits, dim=-1)
        if noise is None:
            noise = torch.rand_like(probs)
        x_0_pred = torch.argmax(probs, dim=-1)  
        
        x_t_one_hot = F.one_hot(x_t, self.num_bins).float()
        t_idx = t - 1
        beta_t = self._extract(self.beta_t, t_idx ,x_t.shape)
        alpha_t = self._extract(self.alpha_t, t_idx,x_t.shape)
        alpha_t_minus_1 = self._extract(self.alpha_t, t_idx-1,x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bar, t_idx,x_t.shape)
        alpha_bar_t_minus_1 = self._extract(self.alpha_bar, t_idx-1,x_t.shape)

        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        q_noise = torch.softmax(gumbel_noise, dim=-1)
        #q_noise = torch.softmax(noise, dim=-1)
        q_noise_xt = torch.sum(q_noise * x_t_one_hot, dim=-1, keepdim=True)

        q_noise_distribution_xt = beta_t * x_t_one_hot + (1 - beta_t) * q_noise

        lambda_1 = 1 - (1 - beta_t) * (1 - alpha_t_minus_1) * q_noise_xt / (alpha_t + (1 - alpha_t) * q_noise_xt)
        lambda_2 = (alpha_bar_t_minus_1 - alpha_bar_t) / (1 - alpha_bar_t)
        #print(lambda_1[1][1])
        bt = (x_t == x_0_pred).float()  
        v_1 = (torch.rand(batch_size, self.num_dimensions, 1, device=device) < lambda_1).float()
        v_2 = (torch.rand(batch_size, self.num_dimensions, 1, device=device) < lambda_2).float()
        u_1 = torch.randint(0, self.num_bins, (batch_size, self.num_dimensions), device=device)
        u_2_probs = q_noise_distribution_xt
        u_2 = torch.argmax(u_2_probs, dim=-1)
        # Case 1: bt == 1 → based on x_t and u_1
        x_t_minus_1_bt1 = (v_1.squeeze(-1) * x_t + (1 - v_1.squeeze(-1)) * u_1)
    
        # Case 2: bt == 0 → based on x_0 and u_2
        x_t_minus_1_bt0 = (v_2.squeeze(-1) * x_0_pred + (1 - v_2.squeeze(-1)) * u_2)
        x_t_minus_1 = bt * x_t_minus_1_bt1 + (1 - bt) * x_t_minus_1_bt0
        x_t_minus_1 = x_t_minus_1.long()
        x_t_minus_1 = torch.where(t.unsqueeze(1) == 1, x_0_pred, x_t_minus_1)
        
        return x_t_minus_1


    def sample(self, pts_feat, steps=1000):
        cond = pts_feat  # 直接使用 [bs, 1024]
        bs = pts_feat.shape[0]
        y_t = torch.randint(0, self.num_bins, (bs, self.num_dimensions), device=self.device)
        for step in reversed(range(1, steps)):
            t = torch.full((bs,), step, dtype=torch.long, device=self.device)
            y_t = self.p_sample(y_t, t, cond, torch.rand((bs, self.num_dimensions, self.num_bins), device=self.device))
        
        return y_t 
    

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)  
        dist2 = dist2.flatten(start_dim=0, end_dim=-2) 
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1) -
            torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()
        
    def loss(self, y, pts_feat):
        cond = pts_feat
        bs = y.shape[0]
        t = torch.randint(1, self.T, (bs,), device=self.device)
        noise = torch.rand((bs, self.num_bins), device=self.device)
        y_t = torch.stack([self.q_sample(y[:,i], t, noise) for i in range(self.num_dimensions)], dim=1)
        
        predicted_x0_logits = self.model_predict(y_t, t, cond)
        predicted_x0 = torch.argmax(predicted_x0_logits , dim=-1)
        # true_q_posterior = self.q_posterior_logits(y, y_t, t)
        # pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, y_t, t)

        true_q_posterior = torch.stack([self.q_posterior_logits(y[:,i], y_t[:,i], t) for i in range(self.num_dimensions)], dim=1)
        pred_q_posterior = torch.stack([self.q_posterior_logits(predicted_x0[:,i], y_t[:,i], t) for i in range(self.num_dimensions)], dim=1)

        vb_loss = self.vb(true_q_posterior, pred_q_posterior)
        angle_ce_loss = sum(F.cross_entropy(predicted_x0_logits[:,i], y[:,i]) for i in range(self.angle_dimensions)) / self.angle_dimensions
        trans_ce_loss = sum(F.cross_entropy(predicted_x0_logits[:,i+self.angle_dimensions], y[:,i+self.angle_dimensions]) 
                            for i in range(self.translation_dimensions)) / self.translation_dimensions
        ce_loss = (0.6*angle_ce_loss + 0.4*trans_ce_loss)
        #l1_loss = F.l1_loss(pred_bins, true_bins)
        pred_bins_arg = predicted_x0_logits.argmax(dim=-1)
        probs = F.softmax(predicted_x0_logits, dim=-1)
        bin_indices = torch.arange(self.num_bins, device=probs.device).float()  # shape: [num_bins]
        pred_bins = torch.sum(probs * bin_indices, dim=-1)  # shape: [bs, num_dimensions]
        true_bins = y.float()
        mse_loss = F.mse_loss(pred_bins, true_bins)
        L1_loss = F.l1_loss(pred_bins, true_bins)
        loss_description = (
            f"CE Loss Contribution: {self.ce_weight * ce_loss:.4f}, "
            f"VB Loss Contribution: {self.vb_weight * vb_loss:.4f}, "
            #f"mse Loss Contribution: {self.mse_weight * mse_loss:.4f}"
            f"L1 Loss Contribution: {self.L1_weight * L1_loss:.4f}"
        )
        total_loss = self.ce_weight*ce_loss  + self.L1_weight*L1_loss + self.vb_weight * vb_loss
        return  total_loss, loss_description



class D3PM_Guassion_Flow(nn.Module):
    def __init__(self, cfg, num_bins=360, T=200, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.num_bins = num_bins  # Number of standard tokens (K)
        self.mask_token = num_bins  # Define [MASK] as token with index num_bins
        self.total_tokens = num_bins + 1  # K+1 (including [MASK])
        self.angle_dimensions = 3  # Angle dimensions
        self.translation_dimensions = 3  # Translation dimensions
        self.num_dimensions = self.angle_dimensions + self.translation_dimensions
        self.T = T
        self.device = device
        self.mse_weight = 0.8
        self.L1_weight = 0.1
        self.ce_weight = 1
        self.vb_weight = 0.4
        self.eps = 1e-8

        timesteps = torch.arange(T, dtype=torch.float32, device=device)
        self.gamma_t = 0.3 * timesteps / (T - 1)
        self.gamma_t = torch.clamp(self.gamma_t, min=1e-6)

        self.beta_t = 0.7 * timesteps / (T - 1) 
        self.beta_t = torch.clamp(self.beta_t, min=1e-6)

        
        # Build transition matrices
        self.build_transition_matrices()

        # Point Cloud Feature Extractors
        if self.cfg.pts_encoder == 'pointnet':
            self.pts_encoder = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
        elif self.cfg.pts_encoder == 'pointnet2':
            self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            self.pts_pointnet = PointNetfeat(num_points=self.cfg.num_points, out_dim=1024)
            self.pts_pointnet2 = Pointnet2ClsMSG(0)
            self.fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
        else:
            raise NotImplementedError

        # Time embedding
        self.time_embedder = TimestepEmbedder(hidden_size=256, frequency_embedding_size=128).to(device)
        
        # Shared MLP feature extractor 
        # Note: Adjusted input dimensions to account for [MASK] token
        self.mlp_shared = nn.Sequential(
            nn.Linear(self.num_dimensions*self.total_tokens + 256 + 1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            ResidualBlock(1024, 1024),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.2)
        ).to(device)
        
        # Angle and translation branches
        self.angles_branch = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        self.translation_branch = nn.Sequential(
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        ).to(device)
        
        # Prediction heads (now accounting for [MASK] token)
        self.angle_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, self.total_tokens)  # total_tokens includes [MASK]
            ) for _ in range(self.angle_dimensions)
        ]).to(device)
        
        self.translation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, self.total_tokens)  # total_tokens includes [MASK]
            ) for _ in range(self.translation_dimensions)
        ]).to(device)

    def _extract(self, array, t_idx, x_shape=None):
        batch_size = t_idx.shape[0]   
        t_idx = torch.clamp(t_idx, min=0, max=len(array)-1)
        out = array[t_idx].to(t_idx.device)
        out = out.reshape(batch_size, 1)
        
        # If target shape x_shape is provided, dynamically adjust out shape
        if x_shape is not None:
            # Dynamically expand dimensions
            for _ in range(len(x_shape) - len(out.shape)+1):
                out = out.unsqueeze(-1)
            
            # Use expand method to extend shape
            out = out.expand(batch_size, *x_shape[1:], 1)
        
        return out
    
    def extract_pts_feature(self, data):
        pts = data['pts']
        if self.cfg.pts_encoder == 'pointnet':
            return self.pts_encoder(pts.permute(0,2,1))
        elif self.cfg.pts_encoder == 'pointnet2':
            return self.pts_encoder(pts)
        elif self.cfg.pts_encoder == 'pointnet_and_pointnet2':
            feat1 = self.pts_pointnet(pts.permute(0,2,1))
            feat2 = self.pts_pointnet2(pts)
            return self.fusion(torch.cat([feat1, feat2], dim=1))
        else:
            raise NotImplementedError


    def build_transition_matrices(self):
        K = self.num_bins
        T = self.T
        device = self.device
        # Small epsilon to prevent numerical instability
        eps = 1e-10

        # Initialize transition matrices
        q_onestep_mats = torch.zeros(T, self.total_tokens, self.total_tokens, device=device)
        i_indices = torch.arange(K, device=device)
        j_indices = torch.arange(K, device=device)
        # Compute distance squared for all i, j pairs
        distance = torch.abs(i_indices[:, None] - j_indices[None, :])
        distance_squared = distance ** 2
        # Precompute n_values for denominator
        n_values = torch.arange(-(K-1), K, device=device).float()
        # Compute denominator for each t with epsilon
        denominator = torch.sum(torch.exp(-4 * n_values**2 / ((K-1)**2 * self.beta_t[:, None])), dim=1) + eps
        q_off_diag = torch.exp(-4 * distance_squared / ((K-1)**2 * self.beta_t[:, None, None])) / denominator[:, None, None]
        # Ensure off-diagonal terms are at least eps
        q_off_diag = torch.clamp(q_off_diag, min=eps)
        # Set off-diagonal elements (excluding diagonal)
        mask_off_diag = ~torch.eye(K, dtype=bool, device=device)
        q_onestep_mats[:, :K, :K] = q_off_diag * mask_off_diag.float()
        # Compute row sums of off-diagonal elements
        row_sum = torch.sum(q_onestep_mats[:, :K, :K], dim=2)
        # Set diagonal elements to ensure row sums to 1 - gamma_t, with minimum eps
        diagonal_indices = torch.arange(K, device=device)
        q_onestep_mats[:, diagonal_indices, diagonal_indices] = torch.clamp(1 - row_sum - self.gamma_t[:, None], min=eps)
        # Set transitions to mask token, ensure at least eps
        q_onestep_mats[:, :K, self.mask_token] = torch.clamp(self.gamma_t[:, None], min=eps)
        # Set mask token self-transition
        q_onestep_mats[:, self.mask_token, self.mask_token] = 1.0

        # Transpose for later use
        q_one_step_transposed = q_onestep_mats.transpose(1, 2)
        # Calculate cumulative matrices
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            # Ensure cumulative matrices have minimum value
            q_mat_t = torch.clamp(q_mat_t, min=eps)
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)



    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to total_tokens - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_sample(self, x_0, t, noise):
        """Forward diffusion process: sample x_t from x_0"""
        # Handle the possibility of x_0 being the [MASK] token
        logits = torch.log(self._at(self.q_mats, t - 1, x_0) + self.eps)  # [bs, total_tokens]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))

        return torch.argmax(logits + gumbel_noise, dim=-1)  # [bs]

    def q_posterior_logits(self, x_0, x_t, t):
        """Calculate posterior distribution q(x_{t-1} | x_t, x_0) logits"""
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(F.one_hot(x_0, self.total_tokens).float() + self.eps)
        else:
            x_0_logits = x_0.clone()
        
        fact1 = self._at(self.q_one_step_transposed, t - 1, x_t)  # [bs, total_tokens]
        softmaxed = F.softmax(x_0_logits, dim=-1)
        
        if t.min() > 1:
            qmats2 = self.q_mats[t - 2]
            fact2 = torch.einsum("bc,bcd->bd", softmaxed, qmats2)
        else:
            fact2 = softmaxed
            
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.reshape((t.shape[0], 1))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def model_predict(self, y_t, t, cond):
        """Predict x_0 logits using separate angle and translation features"""
        bs = y_t.shape[0]
        # Use one-hot encoding that includes the [MASK] token
        y_t_onehot = F.one_hot(y_t, num_classes=self.total_tokens).float()  
        y_t_flat = y_t_onehot.view(bs, -1)  
        
        t_emb = self.time_embedder(t)  # [bs, 128]
        input_feat = torch.cat([y_t_flat, t_emb, cond], dim=1)  
        
        shared_feat = self.mlp_shared(input_feat)  # [bs, 512]

        angles_feat = self.angles_branch(shared_feat)  # [bs, 256]
        trans_feat = self.translation_branch(shared_feat)  # [bs, 256]
        
        angle_logits = [head(angles_feat) for head in self.angle_heads]
        trans_logits = [head(trans_feat) for head in self.translation_heads]
        
        predicted_x0_logits = torch.stack(angle_logits + trans_logits, dim=1)
        
        return predicted_x0_logits

    def p_sample(self, y_t, t, cond, noise):
        """Sample y_{t-1} from y_t"""
        predicted_x0_logits = self.model_predict(y_t, t, cond)
        pred_q_posterior_logits = torch.stack(
            [self.q_posterior_logits(predicted_x0_logits[:, i, :], y_t[:, i], t)
             for i in range(self.num_dimensions)], dim=1
        )  
        
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 1).float().reshape((y_t.shape[0], 1, 1))
        gumbel_noise = -torch.log(-torch.log(noise))
        
        return torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)

    def sample(self, pts_feat, steps=200):
        """Generate angle and translation predictions through reverse diffusion"""
        cond = pts_feat  # [bs, 1024]
        bs = pts_feat.shape[0]
        
        # Start from all [MASK] tokens
        y_t = torch.full((bs, self.num_dimensions), self.mask_token, dtype=torch.long, device=self.device)
        
        for step in reversed(range(1, steps)):
            t = torch.full((bs,), step, dtype=torch.long, device=self.device)
            y_t = self.p_sample(y_t, t, cond, torch.rand((bs, self.num_dimensions, self.total_tokens), device=self.device))
        
        # Filter out any remaining [MASK] tokens and replace with random valid tokens
        mask_positions = (y_t == self.mask_token)
        if mask_positions.any():
            random_tokens = torch.randint(0, self.num_bins, mask_positions.sum().item(), device=self.device)
            y_t = y_t.clone()
            y_t[mask_positions] = random_tokens
            
        return y_t

    def vb(self, dist1, dist2):
        """Calculate variational bound loss (KL divergence)"""
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)  
        dist2 = dist2.flatten(start_dim=0, end_dim=-2) 
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1) -
            torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()
        
    def loss(self, y, pts_feat):
        """Calculate loss, processing angles and translations separately"""
        cond = pts_feat
        bs = y.shape[0]
        t = torch.randint(1, self.T, (bs,), device=self.device)
        
        # Make sure y doesn't contain [MASK] tokens for training
        if (y >= self.num_bins).any():
            y = torch.clamp(y, 0, self.num_bins - 1)
            
        noise = torch.rand((bs, self.total_tokens), device=self.device)
        y_t = torch.stack([self.q_sample(y[:,i], t, noise) for i in range(self.num_dimensions)], dim=1)
        
        predicted_x0_logits = self.model_predict(y_t, t, cond)
        predicted_x0 = torch.argmax(predicted_x0_logits, dim=-1)
        
        # Calculate true and predicted posteriors
        true_q_posterior = torch.stack(
            [self.q_posterior_logits(
                F.one_hot(y[:,i], self.total_tokens).float(), 
                y_t[:,i], t) 
             for i in range(self.num_dimensions)], dim=1)
        
        pred_q_posterior = torch.stack(
            [self.q_posterior_logits(
                F.softmax(predicted_x0_logits[:,i], dim=-1), 
                y_t[:,i], t) 
             for i in range(self.num_dimensions)], dim=1)

        # Calculate variational loss
        vb_loss = self.vb(true_q_posterior, pred_q_posterior)
        
        # Calculate cross-entropy loss for angles and translations
        angle_ce_loss = sum(
            F.cross_entropy(
                predicted_x0_logits[:,i], 
                y[:,i],
                # Ignore [MASK] token index in loss calculation
                ignore_index=self.mask_token 
            ) for i in range(self.angle_dimensions)
        ) / self.angle_dimensions
        
        trans_ce_loss = sum(
            F.cross_entropy(
                predicted_x0_logits[:,i+self.angle_dimensions], 
                y[:,i+self.angle_dimensions],
                # Ignore [MASK] token index in loss calculation
                ignore_index=self.mask_token
            ) for i in range(self.translation_dimensions)
        ) / self.translation_dimensions
        
        ce_loss = (0.6*angle_ce_loss + 0.4*trans_ce_loss)
        
        # Calculate MSE and L1 losses
        # First convert softmax probabilities to expected bin values
        probs = F.softmax(predicted_x0_logits, dim=-1)
        # Only consider probabilities for valid tokens (not [MASK])
        valid_probs = probs[:, :, :self.num_bins]
        # Renormalize probabilities
        valid_probs = valid_probs / (valid_probs.sum(dim=-1, keepdim=True) + self.eps)
        
        bin_indices = torch.arange(self.num_bins, device=probs.device).float()
        pred_bins = torch.sum(valid_probs * bin_indices, dim=-1)
        
        true_bins = y.float()
        mse_loss = F.mse_loss(pred_bins, true_bins)
        L1_loss = F.l1_loss(pred_bins, true_bins)
            
        # Construct loss description
        loss_description = (
            f"CE Loss Contribution: {self.ce_weight * ce_loss:.4f}, "
            f"VB Loss Contribution: {self.vb_weight * vb_loss:.4f}, "
            f"L1 Loss Contribution: {self.L1_weight * L1_loss:.4f}"
        )
        
        # Calculate total loss
        total_loss = self.ce_weight * ce_loss + self.L1_weight * L1_loss + self.vb_weight * vb_loss
        return total_loss, loss_description
