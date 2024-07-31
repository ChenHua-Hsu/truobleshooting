import time, functools, torch, os, sys, random, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import display
import tqdm
import data_utils as utils
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint_sequential

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__()
        # Time information incorporated via Gaussian random feature encoding
        # Randomly sampled weights initialisation. Fixed during optimisation i.e. not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, time):
        # Multiply batch of times by network weights
        time_proj = time[:, None] * self.W[None, :] * 2 * np.pi
        # Output [sin(2pi*wt);cos(2pi*wt)]
        gauss_out = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)
        return gauss_out

class Dense(nn.Module):
    """Fully connected layer that reshapes output of embedded conditional variable to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """Dense nn layer output must have same dimensions as input data:
            [batchsize, (dummy)nhits, (dummy)features]
        """
        return self.dense(x)[..., None]

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        """Encoder block:
        Args:
        embed_dim: length of embedding / dimension of the model
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # Need to set batch_first=True
        # Normally in NLP the batch dimension would be the second dimension
        # In most practices it's the first dimension so we match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
        
        self.ffnn_cls = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        #residual = x.clone()
        
        # Mean-field attention
        # Multiheaded self-attention but replacing query with a single mean field approximator
        # attn (query, key, value, key mask)
        cls_attn_out = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        
        cls_attn_res_out = x_cls + cls_attn_out
        cls_norm1_out = self.norm1(self.dropout1(cls_attn_res_out))
        cls_ffnn_out = self.ffnn_cls(cls_norm1_out)
        cls_ffnn_res_out = cls_ffnn_out + cls_norm1_out
        cls_norm2_out = self.norm2(self.dropout2(cls_ffnn_res_out))
        
        # Added to shower hits
        x = x + cls_norm2_out
        x = self.norm3(x)
        ffnn_out = self.ffnn(x)
        x = x + self.dropout3(ffnn_out)
        x = self.norm4(x)
        return x

class Gen(nn.Module):
    def __init__(self, n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std, **kwargs):
        """Transformer encoder model
        Arguments:
        n_feat_dim = number of features
        embed_dim = dimensionality to embed input
        hidden_dim = dimensionaliy of hidden layer
        num_encoder_blocks = number of encoder blocks
        num_attn_heads = number of parallel attention heads to use
        dropout_gen = regularising layer
        marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
        """
        super().__init__()
        # Embedding: size of input (n_feat_dim) features -> size of output (embed_dim)
        self.embed = nn.Linear(n_feat_dim, embed_dim)
        # Seperate embedding for (time/incident energy) conditional inputs (small NN with fixed weights)
        self.embed_e = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # Boils embedding down to single value
        self.dense_t = Dense(embed_dim, 1)
        self.dense_e = Dense(embed_dim, 1)
        # Module list of encoder blocks
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_attn_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout_gen,
                )
                for i in range(num_encoder_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(embed_dim, n_feat_dim)
        # token simply has same dimension as input feature embedding
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.act = nn.GELU()
        # Swish activation function
        self.act_sig = lambda x: x * torch.sigmoid(x)
        # Standard deviation of SDE
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, e, mask=None):
        """
        x = input data
        t = noise
        e = conditional variable
        mask = padding mask for attention mechanism with 'True' indicating the corresponding key value will be ignored when calculating the attention.
        """
        
        # Embed 4-vector input 
        x = self.embed(x)
        # Embed 'time' condition
        embed_t_ = self.act_sig( self.embed_t(t) )
        # Embed incident particle energy
        embed_e_ = self.act_sig( self.embed_e(e) )
        # 'class' token (mean field)
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        
        
        # Feed input embeddings into encoder block
        for layer in self.encoder:
            # Match dimensions and append to input
            x += self.dense_t(embed_t_).clone()
            x += self.dense_e(embed_e_).clone()
            # Each encoder block takes previous blocks output as input
            # To embed the high class feature,for example, I want to add a input embedding to let it know that if energy is higher it's x,y should lower

            x = layer(x, x_cls, mask) # Block layers
        
        # Rescale models output (helps capture the normalisation of the true scores)
        mean_ , std_ = self.marginal_prob_std(x,t)
        output = self.out(x) / std_[:, None, None]
        return output

############################################
##  Serialized Model (under development)  ##
############################################

class Transformer_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, src_key_padding_mask=None):
        """Encoder block:
        Args:
        embed_dim: length of embedding / dimension of the model
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # Need to set batch_first=True
        # Normally in NLP the batch dimension would be the second dimension
        # In most practices it's the first dimension so we match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)

        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.ffnn2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.embed_e = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.act_sig = lambda x: x * torch.sigmoid(x)
        self.dense_t = nn.Linear(embed_dim, embed_dim)
        self.dense_e = nn.Linear(embed_dim, embed_dim)
        self.dense_t =  Dense(embed_dim, 1)
        self.dense_e =  Dense(embed_dim, 1)
    def forward(self, input_):
        #residual = x.clone()
        x = input_[0]
        t = input_[1]
        e = input_[2]
        x_cls = input_[3]
        src_key_padding_mask = input_[4]
        original_t = input_[5]
        # Mean-field attention
        # Multiheaded self-attention but replacing query with a single mean field approximator
        # attn (query, key, value, key mask)
        embed_t_ = self.act_sig(self.embed_t(t))
        embed_e_ = self.act_sig(self.embed_e(e))
        x += self.dense_t(embed_t_).clone()
        x += self.dense_e(embed_e_).clone()
        attn_out = self.attn(x_cls, x, x, key_padding_mask = src_key_padding_mask)[0]

        attn_res_out = x_cls + attn_out
        norm1_out = self.norm1(self.dropout1(attn_res_out))
        ffnn_out = self.ffnn(norm1_out)
        ffnn_res_out = ffnn_out + norm1_out
        norm2_out = self.norm2(self.dropout2(ffnn_res_out))

        # Added to shower hits
        x = x + norm2_out
        x = self.norm3(x)
        ffnn_out = self.ffnn2(x)
        x = x + self.dropout3(ffnn_out)
        x = self.norm4(x)

        return [x, t, e, x_cls, src_key_padding_mask, original_t]

class Embed_Block(nn.Module):
    def __init__(self, n_feat_dim, embed_dim, hidden_dim, **kwargs):
        super().__init__()
        self.embed = nn.Linear(n_feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.ones(1,1,embed_dim), requires_grad=True)
    def forward(self, input_):

        x = input_[0]
        t = input_[1]
        e = input_[2]
        src_key_padding_mask = input_[3]
        embed_x_ = self.embed(x)
        x_cls_expand = self.cls_token.expand(x.size(0), 1, -1)
        return [embed_x_, t, e, x_cls_expand, src_key_padding_mask, t]

class Output_Block(nn.Module):
    def __init__(self, n_feat_dim, embed_dim, marginal_prob_std):
        super().__init__()
        self.out = nn.Linear(embed_dim, n_feat_dim)
        self.marginal_prob_std = marginal_prob_std
    def forward(self, input_):
        x = input_[0]
        embed_t = input_[1]
        e = input_[2]
        x_cls = input_[3]
        src_key_padding_mask = input_[4]
        original_t = input_[5]
        mean_ , std_ = self.marginal_prob_std(x,original_t)
        output = self.out(x) / std_[:, None, None]
        return output

def get_seq_model(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std, **kwargs):
    module_dict = OrderedDict()
    module_dict['embed1'] = Embed_Block(n_feat_dim, embed_dim, hidden_dim)
    for i in range(num_encoder_blocks):
        module_dict['transformer{}'.format(i)] = Transformer_Block(embed_dim=embed_dim,
                                                                   num_heads=num_attn_heads,
                                                                   hidden_dim=hidden_dim,
                                                                   dropout=dropout_gen,
                                                                   )
    module_dict['output1'] = Output_Block(n_feat_dim=n_feat_dim, embed_dim=embed_dim, marginal_prob_std=marginal_prob_std) 
    seq_model = nn.Sequential(module_dict)
    return seq_model


def loss_fn(model, x, incident_energies, marginal_prob_std , padding_value=0, eps=1e-3, device='cpu', diffusion_on_mask=False, serialized_model=False, cp_chunks=0, weight=None):

    """The loss function for training score-based generative models
    Uses the weighted sum of Denoising Score matching objectives
    Denoising score matching
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    - Perturbation avoids need to calculate trace of Jacobian of model output

    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model
        x: A mini-batch of training data
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
        eps: A tolerance value for numerical stability
    """
    # Generate padding mask for attention mechanism
    # Positions with True are ignored while False values will be unchanged
    attn_padding_mask = (x[:,:,0] == 0).type(torch.bool)
    
    # Tensor of randomised 'time' steps
    random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps


    # Mask to avoid perturbing padded entries
    #input_mask = (x[:,:,0] != 0).unsqueeze(-1)
    mask_tensor = (~attn_padding_mask).float()[...,None]
    
    # Calculate mean and standard deviation of the perturbation kernel
    mean_, std_ = marginal_prob_std(x,random_t)
    
    # Noise
    z = torch.normal(0,1,size=x.shape, device=device)
    if not diffusion_on_mask:
      z = z*mask_tensor
      
    # Add noise, scheduled by perturbation kernel, to input
    perturbed_x = mean_ + std_[:, None, None]*z
    if not diffusion_on_mask:
      perturbed_x = perturbed_x*mask_tensor
      
    # Evaluate model (aim: to estimate the score function of each noise-perturbed distribution)
    if serialized_model:
        if cp_chunks == 0:
            scores = model([perturbed_x, random_t, incident_energies, attn_padding_mask])
        else:
            scores = checkpoint_sequential(model, cp_chunks, [perturbed_x, random_t, incident_energies, attn_padding_mask])
    else:
        scores = model(perturbed_x, random_t, incident_energies, mask=attn_padding_mask)
        # if torch.isnan(scores).any():
        #     # Log if there are any nans in the input
        #     with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #         file.write("Nan in scores\n")
        #     if torch.isnan(perturbed_x).any():
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("Nan in perturbed_x\n")
        #     else:
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("No nans in perturbed_x\n")
        #     if torch.isnan(random_t).any():
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("Nan in random_t\n")
        #     else:
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("No nans in random_t\n")
        #     if torch.isnan(incident_energies).any():
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("Nan in incident_energies\n")
        #     else:
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("No nans in incident_energies\n")
        #     if torch.isnan(attn_padding_mask).any():
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("Nan in attn_padding_mask\n")
        #     else:
        #         with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #             file.write("No nans in attn_padding_mask\n")
        # else:
        #     with open("/home/ken91021615/tdsm_encoder_sweep0516/util/log.txt", "a") as file:
        #         file.write("No nans in scores\n")
    
    # Calculate loss 
    if not weight is None:
      losses = torch.square( scores*std_[:,None,None] + z ) * weight
    else:
      losses = torch.square( scores*std_[:,None,None] + z )

    # Mean of losses across all hits and 4-vectors (normalise by number of hits)
    # try sum
    # Calculate the Denoise Score-matching objective
    # Mean the losses across all hits and 4-vectors (using sum, loss numerical value gets too large)
    losses = torch.mean( losses, dim=(1,2) )

    # Mean loss for batch
    batch_loss = torch.mean( losses )
    
    return batch_loss

class ScoreMatchingLoss(nn.Module):
    """
    Denoising Score-Matiching Objective function:
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    
    Subclass of nn.Module as is standard practice to inherit methods from nn.Module in pytroch
    """
    
    def __init__(self):
        # register components of nn.Module to custom loss
        super(ScoreMatchingLoss, self).__init__()
        
    def forward(self, model, x, incident_energies, marginal_prob_std , padding_value=0, eps=1e-3, device='cpu', diffusion_on_mask=False, serialized_model=False, cp_chunks=0):
        
        '''
        Forward method used to calculate value:
            model: A PyTorch model instance that represents a time-dependent score-based model
            x: A mini-batch of training data
            marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
            eps: A tolerance value for numerical stability
        '''
        
        # Generate padding mask for attention mechanism
        # Positions with True are ignored while False values will be unchanged
        attn_padding_mask = (x[:,:,0] == 0).type(torch.bool)

        # Tensor of randomised 'time' steps
        random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps

        # Mask to avoid perturbing padded entries
        #input_mask = (x[:,:,0] != 0).unsqueeze(-1)
        mask_tensor = (~attn_padding_mask).float()[...,None]

        # Calculate mean and standard deviation of the perturbation kernel
        mean_, std_ = marginal_prob_std(x,random_t)

        # Noise
        z = torch.normal(0,1,size=x.shape, device=device)
        if not diffusion_on_mask:
          z = z*mask_tensor

        # Add noise, scheduled by perturbation kernel, to input
        perturbed_x = mean_ + std_[:, None, None]*z
        if not diffusion_on_mask:
          perturbed_x = perturbed_x*mask_tensor

        # Evaluate model (aim: to estimate the score function of each noise-perturbed distribution)
        if serialized_model:
            if cp_chunks == 0:
                scores = model([perturbed_x, random_t, incident_energies, attn_padding_mask])
            else:
                scores = checkpoint_sequential(model, cp_chunks, [perturbed_x, random_t, incident_energies, attn_padding_mask])
        else:
            scores = model(perturbed_x, random_t, incident_energies, mask=attn_padding_mask)

        # Calculate the Denoise Score-matching objective
        # Mean the losses across all hits and 4-vectors (using sum, loss numerical value gets too large)
        losses = torch.square( scores*std_[:,None,None] + z )
        losses = torch.mean( losses, dim=(1,2) )

        # Mean loss for batch
        batch_loss = torch.mean( losses )

        return batch_loss
    
class ScoreMatchingLoss_addcorrealtion(nn.Module):
    """
    Denoising Score-Matiching Objective function:
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    
    Subclass of nn.Module as is standard practice to inherit methods from nn.Module in pytroch
    """
    
    def __init__(self):
        # register components of nn.Module to custom loss
        super(ScoreMatchingLoss_addcorrealtion, self).__init__()
        
    def forward(self, model, x, incident_energies, marginal_prob_std , padding_value=0, eps=1e-3, alpha = 0.5, device='cpu', diffusion_on_mask=False, serialized_model=False, cp_chunks=0):
        
        '''
        Forward method used to calculate value:
            model: A PyTorch model instance that represents a time-dependent score-based model
            x: A mini-batch of training data
            marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
            eps: A tolerance value for numerical stability
        '''
        
        # Generate padding mask for attention mechanism
        # Positions with True are ignored while False values will be unchanged
        attn_padding_mask = (x[:,:,0] == 0).type(torch.bool)

        # Tensor of randomised 'time' steps
        random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps

        # Mask to avoid perturbing padded entries
        #input_mask = (x[:,:,0] != 0).unsqueeze(-1)
        mask_tensor = (~attn_padding_mask).float()[...,None]

        # Calculate mean and standard deviation of the perturbation kernel
        mean_, std_ = marginal_prob_std(x,random_t)

        # Noise
        z = torch.normal(0,1,size=x.shape, device=device)
        if not diffusion_on_mask:
          z = z*mask_tensor

        # Add noise, scheduled by perturbation kernel, to input
        perturbed_x = mean_ + std_[:, None, None]*z
        if not diffusion_on_mask:
          perturbed_x = perturbed_x*mask_tensor

        # Evaluate model (aim: to estimate the score function of each noise-perturbed distribution)
        if serialized_model:
            if cp_chunks == 0:
                scores = model([perturbed_x, random_t, incident_energies, attn_padding_mask])
            else:
                scores = checkpoint_sequential(model, cp_chunks, [perturbed_x, random_t, incident_energies, attn_padding_mask])
        else:
            scores = model(perturbed_x, random_t, incident_energies, mask=attn_padding_mask)

        # Calculate the Denoise Score-matching objective
        # Mean the losses across all hits and 4-vectors (using sum, loss numerical value gets too large)
        losses = torch.square( scores*std_[:,None,None] + z )
        losses = torch.mean( losses, dim=(1,2) )

        # Mean loss for batch
        batch_loss = torch.mean( losses )

        # Add correlation penalty term between e and theta
        e = scores[:, :, 0]  # energy
        r = scores[:, :, 1]  # r (rho in cylindrical coordinates)
        theta = scores[:, :, 2]  # theta (phi in cylindrical coordinates)
        z = scores[:, :, 3]

        # Calculate correlation coefficient
        e_mean = e.mean(dim=1, keepdim=True)
        theta_mean = theta.mean(dim=1, keepdim=True)
        r_mean = r.mean(dim=1, keepdim=True)
        z_mean = z.mean(dim=1, keepdim=True)
        e_centered = e - e_mean
        r_centered = r - r_mean
        z_centered = z - z_mean
        theta_centered = theta - theta_mean
        e_std = e_centered.std(dim=1)
        theta_std = theta_centered.std(dim=1)
        r_std = r_centered.std(dim=1)
        z_std = z_centered.std(dim=1)

        covariance_et = (e_centered * theta_centered).mean(dim=1)
        correlation_rt = (r_centered * theta_centered).mean(dim=1)
        correlation_zt = (z_centered * theta_centered).mean(dim=1)
        
        correlation_et = covariance_et / (e_std * theta_std + 1e-8)  # Avoid division by zero
        correlation_rt = correlation_rt / (r_std * theta_std + 1e-8)
        correlation_zt = correlation_zt / (z_std * theta_std + 1e-8)

        # Add correlation penalty to the loss (e.g., L2 norm of correlation)
        correlation_penalty_et = torch.mean(correlation_et**2)
        correlation_penalty_rt = torch.mean(correlation_rt**2)
        correlation_penalty_zt = torch.mean(correlation_zt**2)
        #alpha = 0.5 # Weight of the correlation penalty

        # Combine original loss with correlation penalty
        total_loss = batch_loss + alpha*correlation_penalty_et + alpha*correlation_penalty_rt + alpha*correlation_penalty_zt

        return total_loss, batch_loss, correlation_penalty_et



