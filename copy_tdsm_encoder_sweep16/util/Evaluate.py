import time, functools, torch, os, sys, random, fnmatch, psutil
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import util.score_model
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chisquare
from util.XMLHandler import XMLHandler
import math
import tqdm
import seaborn as sn
import pandas as pd
from util.pretty_confusion_matrix import pp_matrix

def digitize(tensor, bin_edges, device, middle='true', dtype=torch.float32):
    bin_edges = torch.tensor(bin_edges, device=torch.device(device))
    bin_indices = torch.bucketize(tensor, bin_edges) - 1
    Mask = (bin_indices >= len(bin_edges) - 1) | (bin_indices == -1)
    bin_indices[bin_indices >= (len(bin_edges)-1)] = len(bin_edges)-2
    bin_indices[bin_indices == -1] = 0

    middle_values = (bin_edges[bin_indices] + bin_edges[bin_indices + 1])/2
    return middle_values, bin_indices, Mask

def digitize_input(sample_list, particle, filename, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        event_np = np.zeros((1,n_r_bin * n_theta_bin * z_bin))
        event_np[0][bin_indices[mask]] = E_[mask]
        trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    
    return trans_event, trans_event_np

def digitize_input_dep_e(sample_list, particle, filename, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        event_np = np.zeros((1,n_r_bin * n_theta_bin * z_bin))
        event_np[0][bin_indices[mask]] = E_[mask]
        E_total = E_[mask].sum()
        trans_event_np.append(E_total)
    trans_event_np = trans_event_np#np.concatenate(trans_event_np, axis=0)
    
    return trans_event, trans_event_np

def digitize_input_r(sample_list, particle, filename, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        mask_r      = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) # Energy cut on 15keV
        # Initialize arrays for sum and count
        energy_sum_r = np.zeros(n_r_bin)
        count_r = np.zeros(n_r_bin)

        # Accumulate the sum and count for each r_bin
        for i, r_bin_index in enumerate(r_bin_indices[mask_r]):
            energy_sum_r[r_bin_index] += E_[mask_r][i]
            count_r[r_bin_index] += 1

        # Calculate the mean for each r_bin, avoiding division by zero
        mean_energy_r = np.divide(energy_sum_r, count_r, where=(count_r != 0))

        # Handle cases where count is zero to avoid NaN
        mean_energy_r[count_r == 0] = 0

        # Store the mean energy in event_np
        event_np = np.zeros((1, n_r_bin))
        event_np[0] = mean_energy_r
        trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    
    return trans_event, trans_event_np

def digitize_input_z_dep(sample_list, particle, filename, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        mask_z      = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~z_Mask) # Energy cut on 15keV
        # Initialize arrays for sum and count
        energy_sum_z = np.zeros(z_bin)
        #count_z = np.zeros(z_bin)

        # Accumulate the sum and count for each r_bin
        for i, r_bin_index in enumerate(z_bin_indices[mask_z]):
            energy_sum_z[r_bin_index] += E_[mask_z][i]
            with open('energy_sum_z1.txt', 'a') as f:
                #f.write(str(energy_sum_z))
                f.write(str(E_[mask_z])) 
            print("energy_sum_z")
            print(E_[mask_z][i].shape)
            #count_z[r_bin_index] += 1

        # Calculate the mean for each r_bin, avoiding division by zero
        #mean_energy_z = np.divide(energy_sum_z, count_z, where=(count_z != 0))

        # Handle cases where count is zero to avoid NaN
        #mean_energy_z[count_z == 0] = 0

        # Store the mean energy in event_np
        event_np = np.zeros((1, z_bin))
        event_np[0] = energy_sum_z
        trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    
    return trans_event, trans_event_np
def digitize_input_z_dep(sample_list, particle, filename, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        mask_z      = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~z_Mask) # Energy cut on 15keV
        # Initialize arrays for sum and count
        energy_sum_z = np.zeros(z_bin)
        #count_z = np.zeros(z_bin)

        
        

        # Initialize an array to store the maximum energy for each z-bin
        max_energy_z = np.full(z_bin, -np.inf)  # Initialize with -inf to ensure any energy value is larger

        # Iterate through the events to find the maximum energy for each z-bin
        for i, z_bin_index in enumerate(z_bin_indices[mask_z]):
            energy = E_[mask_z][i]
            if energy > max_energy_z[z_bin_index]:
                max_energy_z[z_bin_index] = energy

        # Replace -inf with 0 for bins with no entries to avoid issues in division
        max_energy_z[max_energy_z == -np.inf] = 0

        # Calculate the sum of deposited energies in each z-bin
        energy_sum_z = np.zeros(z_bin)
        for i, z_bin_index in enumerate(z_bin_indices[mask_z]):
            energy_sum_z[z_bin_index] += E_[mask_z][i]

        # Calculate the ratio of the maximum energy to the deposited energy
        # Avoid division by zero by using np.divide with the where condition
        ratio_max_to_deposited = np.divide(max_energy_z, energy_sum_z, where=(energy_sum_z != 0))

        # Store the ratio in event_np
        event_np = np.zeros((1, z_bin))
        event_np[0] = ratio_max_to_deposited
        trans_event_np.append(event_np)

        # Accumulate the sum and count for each r_bin
        # for i, r_bin_index in enumerate(z_bin_indices[mask_z]):
        #     energy_sum_z[r_bin_index] += E_[mask_z][i]
        #     with open('energy_sum_z1.txt', 'a') as f:
        #         #f.write(str(energy_sum_z))
        #         f.write(str(E_[mask_z])) 
        #     print("energy_sum_z")
        #     print(E_[mask_z][i].shape)
            #count_z[r_bin_index] += 1


        # Calculate the mean for each r_bin, avoiding division by zero
        #mean_energy_z = np.divide(energy_sum_z, count_z, where=(count_z != 0))

        # Handle cases where count is zero to avoid NaN
        #mean_energy_z[count_z == 0] = 0

        # Store the mean energy in event_np
        # event_np = np.zeros((1, z_bin))
        # event_np[0] = energy_sum_z
        # trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    
    return trans_event, trans_event_np

def digitize_input_r_width(sample_list, particle, filename, dtype=torch.float32, pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5, z_bin+1)
    trans_event = []
    trans_event_np = []

    for event in sample_list:
        r = torch.sqrt(event[:, 1] * event[:, 1] + event[:, 2] * event[:, 2])
        theta = torch.atan(event[:, 2] / event[:, 1]) + torch.pi * torch.where(event[:, 1] < 0, torch.tensor(1, device=torch.device(device)), torch.tensor(0, device=torch.device(device))) - 2 * torch.pi * torch.where(torch.logical_and(event[:, 1] < 0, event[:, 2] < 0), torch.tensor(1, device=torch.device(device)), torch.tensor(0, device=torch.device(device)))
        middle_r, r_bin_indices, r_Mask = digitize(r, r_edge, device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge, device)
        middle_z, z_bin_indices, z_Mask = digitize(event[:, 3], z_edge, device)
        bin_indices = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin

        x = middle_r * torch.cos(middle_theta)
        y = middle_r * torch.sin(middle_theta)
        output_ = torch.stack((event[:, 0], x, y, middle_z), dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:, 1:], dim=0, return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values), device=torch.device(device))
        unique_values_gt2 = unique_values[counts > 1]
        unique_values_idx_gt2 = unique_values_idx[counts > 1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0], 0] = output_[matching_indices, 0].sum()
            output_[matching_indices[1:]] = torch.ones(4, device=torch.device(device)) * pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        r_values = middle_r.cpu().numpy().copy()
        z_Mask = z_Mask.cpu().numpy().copy()
        mask_z = (~r_Mask) & (~z_Mask)  # Mask based on conditions

        # Initialize arrays for sum, count, and sum of squared r values
        r_sum_z = np.zeros(z_bin)
        count_z = np.zeros(z_bin)
        r_squared_sum_z = np.zeros(z_bin)

        # Accumulate the sum, count, and squared sum for each z_bin
        for i, z_bin_index in enumerate(z_bin_indices[mask_z]):
            r_sum_z[z_bin_index] += r_values[mask_z][i]
            count_z[z_bin_index] += 1
            r_squared_sum_z[z_bin_index] += r_values[mask_z][i] ** 2

        # Calculate the mean r for each z_bin, avoiding division by zero
        mean_r_z = np.divide(r_sum_z, count_z, where=(count_z != 0))

        # Calculate the variance for each z_bin, avoiding division by zero
        variance_r_z = np.divide(r_squared_sum_z, count_z, where=(count_z != 0)) - mean_r_z ** 2

        # Handle cases where count is zero to avoid NaN values
        variance_r_z[count_z == 0] = 0

        # Calculate the standard deviation (width) for each z_bin
        std_dev_r_z = np.sqrt(variance_r_z)

        # Store the standard deviation in event_np
        event_np = np.zeros((1, z_bin))
        event_np[0] = std_dev_r_z
        trans_event_np.append(event_np)

    trans_event_np = np.concatenate(trans_event_np, axis=0)
    return trans_event, trans_event_np


def r_e(sample_list, particle, filename,dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    r_out = []
    E_out = []
    z_out = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        r_out.append(r)
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        z_out.append(z_bin_indices)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        E_out.append(E_)
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        event_np = np.zeros((1,n_r_bin * n_theta_bin * z_bin))
        event_np[0][bin_indices[mask]] = E_[mask]
        trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    r_out  = np.array(r_out)
    # bins = np.linspace(0, max(r.max(),r_gen.max()), 20)
    # r_indices = np.digitize(r, bins)
    # r_indices_gen = np.digitize(r_gen, bins)
    # print(r_indices)
    # avg_energy = []
    # avg_energy_gen = []
    # #all_e = np.array(all_e)
    E_out = np.array(E_out)
    z_out = np.array(z_out)
    
    # for i in range(1, len(bins)):
    #     geant_bin = E_[r_indices == i]
    #     gen_bin = E_gen[r_indices_gen == i]

    #     if len(gen_bin) > 0:
    #         avg_energy_gen.append(gen_bin.mean())
    #     else:
    #         avg_energy_gen.append(np.nan)  # Use NaN to indicate no data

    #     if len(geant_bin) > 0:
    #         avg_energy.append(geant_bin.mean())
    #     else:
    #         avg_energy.append(np.nan)
    # fig1, ax = plt.subplots(1,2, figsize=(12,6))
    # print('Plot hit energy vs. r')
    # ax[0].set_ylabel('Hit energy [GeV]')
    # ax[0].set_xlabel('r [cm]')
    # ax[0].hist(bins[1:], weights = avg_energy, label='Geant4',color='gray')
    # ax[0].hist(bins[1:], weights = avg_energy_gen, label='Gen',color='orange')
    # ax[0].legend(loc='upper right')
    # fig1_name = '/home/ken91021615/tdsm_encoder_sweep0516/hit_energy_vs_r.png'
    # fig1.savefig(fig1_name)
    return trans_event, trans_event_np, r_out, E_out,z_out
def r_e_plot(sample_list, particle, filename, r_gen,E_gen,z_gen, dtype=torch.float32,pad_value=-20, device='cpu'):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_r_bin = len(r_edge) - 1
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin+1)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    trans_event_np = []
    r_out = []
    E_out = []
    z_out = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        r_out.append(r)
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1, device=torch.device(device))
                                                                         , torch.tensor(0, device=torch.device(device))) - 2*torch.pi*torch.where(torch.logical_and(event[:,1]<0, event[:,2]<0),
                                                                                                                                                  torch.tensor(1, device=torch.device(device)),
                                                                                                                                                  torch.tensor(0, device=torch.device(device)))
        middle_r,     r_bin_indices, r_Mask         = digitize(r,r_edge,device)
        middle_theta, theta_bin_indices, theta_Mask = digitize(theta, theta_edge,device)
        middle_z,     z_bin_indices, z_Mask         = digitize(event[:,3],z_edge,device)
        z_out.append(z_bin_indices)
        bin_indices   = r_bin_indices + theta_bin_indices * n_r_bin + z_bin_indices * n_theta_bin * n_r_bin 

        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        output_ = output_.to(dtype)

        unique_values, indices, counts = torch.unique(output_[:,1:],dim=0,return_counts=True, return_inverse=True)
        unique_values_idx = torch.arange(0, len(unique_values),device=torch.device(device))
        unique_values_gt2 = unique_values[counts>1]
        unique_values_idx_gt2 = unique_values_idx[counts>1]
        for i, combination in zip(unique_values_idx_gt2, unique_values_gt2):
            matching_indices = (indices == i).nonzero(as_tuple=False).squeeze()
            output_[matching_indices[0],0] = output_[matching_indices,0].sum()
            output_[matching_indices[1:]] = torch.ones(4,device=torch.device(device))*pad_value
        trans_event.append(output_)

        bin_indices = bin_indices.cpu().numpy().copy()
        E_          = output_[:,0].cpu().numpy().copy()
        E_out.append(E_)
        r_Mask      = r_Mask.cpu().numpy().copy()
        theta_Mask  = theta_Mask.cpu().numpy().copy()
        z_Mask      = z_Mask.cpu().numpy().copy()
        mask        = (~(E_ == pad_value)) & (E_ > 1.5e-5) & (~r_Mask) & (~theta_Mask) & (~z_Mask) # Energy cut on 15keV
        event_np = np.zeros((1,n_r_bin * n_theta_bin * z_bin))
        event_np[0][bin_indices[mask]] = E_[mask]
        trans_event_np.append(event_np)
    trans_event_np = np.concatenate(trans_event_np, axis=0)
    r_out  = np.array(r_out)
    r_out = np.array(r_out)
    z_out = np.array(z_out)
    r_gen = np.array(r_gen)
    z_gen = np.array(z_gen)
    r_out = r_out.flatten().tolist()
    z_out = z_out.flatten().tolist()
    r_gen = r_gen.flatten().tolist()
    z_gen = z_gen.flatten().tolist()
    r_out  = np.array(r_out)
    r_out = np.array(r_out)
    z_out = np.array(z_out)
    r_gen = np.array(r_gen)
    z_gen = np.array(z_gen)
    
    for i in range(len(r_gen)):
        if r_gen[i] > r_out.max():
            r_gen[i] = r_out.max()
    for i in range(len(z_gen)):
        if z_gen[i] > z_out.max():
            z_gen[i] = z_out.max()
    bins = np.linspace(0, r_out.max(), 50)
    #bins_z = np.linspace(0, z_out.max(), 50)
    r_indices = np.digitize(r_out, bins)
    r_indices_gen = np.digitize(r_gen, bins)
    print(r_indices)

    avg_energy = []
    avg_energy_gen = []
    avg_energy_z = []
    avg_energy_gen_z = []
    #all_e = np.array(all_e)
    E_out = np.array(E_out)
    E_out = E_out.flatten().tolist()
    E_out = np.array(E_out)
    E_gen = np.array(E_gen)
    E_gen = E_gen.flatten().tolist()
    E_gen = np.array(E_gen)
    
    for i in range(1, len(bins)):
        geant_bin = E_out[r_indices == i]
        gen_bin = E_gen[r_indices_gen == i]

        if len(gen_bin) > 0:
            avg_energy_gen.append(gen_bin.mean())
        else:
            avg_energy_gen.append(np.nan)  # Use NaN to indicate no data

        if len(geant_bin) > 0:
            avg_energy.append(geant_bin.mean())
        else:
            avg_energy.append(np.nan)
    for i in range(1, len(z_out)):
        geant_bin_z = E_out[z_out == i]
        gen_bin_z = E_gen[z_gen == i]

        if len(gen_bin_z) > 0:
            avg_energy_gen_z.append(gen_bin_z.mean())
        else:
            avg_energy_gen_z.append(np.nan)  # Use NaN to indicate no data

        if len(geant_bin_z) > 0:
            avg_energy_z.append(geant_bin_z.mean())
        else:
            avg_energy_z.append(np.nan)
    fig1, ax = plt.subplots(1,2, figsize=(12,6))
    print('Plot hit energy vs. r')
    ax[0].set_ylabel('Hit energy [GeV]')
    ax[0].set_xlabel('r [cm]')
    ax[0].hist(bins[1:], weights = avg_energy, label='Geant4',color='gray')
    ax[0].hist(bins[1:], weights = avg_energy_gen, label='Gen',color='orange')
    ax[0].legend(loc='upper right')
    ax[1].set_ylabel('Hit energy [GeV]')
    ax[1].set_xlabel('layer')
    ax[1].hist(avg_energy_z, label='Geant4',color='gray')
    ax[1].hist(avg_energy_gen_z, label='Gen',color='orange')
    ax[1].legend(loc='upper right')
    fig1_name = '/home/ken91021615/tdsm_encoder_sweep0516/hit_energy_vs_r.png'
    fig1.savefig(fig1_name)
    return trans_event, trans_event_np, r_out, E_out, z_out


class attn_cls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(embed_dim, hidden)
        self.fc2     = nn.Linear(hidden, embed_dim)
        self.act     = nn.GELU()
    def forward(self, x, x_cls, src_key_padding_mask=None):
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x_cls = self.act(self.fc1(x_cls))
        x_cls = self.dropout(x_cls)
        x_cls = self.act(self.fc2(x_cls))
        return x_cls


class Classifier(nn.Module):
    def __init__(self, n_dim, embed_dim, hidden_dim, n_layers, n_layers_cls, n_heads, dropout):
        super().__init__()
        self.embed = nn.Linear(n_dim, embed_dim)
        self.embed_e = nn.Sequential(util.score_model.GaussianFourierProjection(embed_dim=64), nn.Linear(64,64))
        self.dense1  = util.score_model.Dense(64,1)
        self.encoder = nn.ModuleList(
            [   
                util.score_model.Block(
                    embed_dim = embed_dim,
                    num_heads = n_heads,
                    hidden_dim  = hidden_dim,
                    dropout   = dropout
                )
                for i in range(n_layers)
            ]
        )
        self.encoder_cls = nn.ModuleList(
            [
                attn_cls(
                    embed_dim = embed_dim,
                    num_heads = n_heads,
                    hidden    = hidden_dim,
                    dropout   = dropout
                )
                for i in range(n_layers_cls)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, 1)
        self.cls_token  = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.cls_token2 = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.act = nn.GELU()

    def forward(self, x, e, mask=None):

        x = self.embed(x)

        embed_e_ = self.act(self.embed_e(e))
        x += self.dense1(embed_e_).clone()

        x_cls = self.cls_token.expand(x.size(0),1,-1)

        for layer in self.encoder:
            x = layer(x, x_cls=x_cls, src_key_padding_mask = mask)

        x_cls2 = self.cls_token2.expand(x.size(0),1,-1)

        for layer in self.encoder_cls:
            x_cls2 = layer(x, x_cls=x_cls2, src_key_padding_mask = mask)

        return torch.flatten(self.out(x_cls2))

class evaluate_dataset(Dataset):

  def __init__(self, data, inE, label, device = 'cpu'):

    self.data  = data
    self.data_np = None
    self.inE   = torch.tensor(inE, device=device)
    self.label = label
    self.max_nhits = -1
    self.device=device
    self.r = None
    self.e = None
    self.z = None

  def __getitem__(self, index):
    x = self.data[index]
    y = self.inE[index]
    z = self.label[index]
    return (x,y,z)

  def __len__(self):
    return len(self.data)

  def padding(self, value = 0):

    for showers in self.data:
        if len(showers) > self.max_nhits:
            self.max_nhits = len(showers)

    padded_showers = []
    for showers in self.data:
        pad_hits = self.max_nhits-len(showers)
        padded_shower = F.pad(input = showers, pad=(0,0,0,pad_hits), mode='constant', value = value)
        padded_showers.append(padded_shower)

    self.data = padded_showers
    self.padding_value = value

  def digitize(self, particle='electron',xml_bin='binning_dataset_2.xml',pad_value=0.0):
    self.data, self.data_np = digitize_input(self.data, particle, xml_bin, device=self.device, pad_value=pad_value)

  def concat(self, dataset2):

    self.data.extend(dataset2.data)
    self.inE   = torch.concat((self.inE, dataset2.inE))
    self.label = torch.concat((self.label, dataset2.label))

  def r_e_get (self, particle='electron',xml_bin='binning_dataset_2.xml',pad_value=0.0):
    self.data, self.data_np, self.r,self.e,self.z = r_e(self.data, particle, xml_bin, device=self.device, pad_value=pad_value)
    return self.r, self.e,self.z
  def r_e_plot_e (self, particle='electron',xml_bin='binning_dataset_2.xml',r_gen = None,E_gen = None,z_gen= None,pad_value=0.0):
    self.data, self.data_np, self.r,self.e,self.z = r_e_plot(self.data, particle, xml_bin, r_gen =r_gen, E_gen=E_gen,z_gen=z_gen,device=self.device, pad_value=pad_value)



class evaluator:

  def __init__(self, base_dataset_name, gen_dataset_name, padding_value = 0, device='cpu', digitize=True):

    base_dataset = torch.load(base_dataset_name, map_location = torch.device(device))
    gen_dataset  = torch.load(gen_dataset_name,  map_location = torch.device(device))
    dataset_size = min(len(base_dataset[0]),len(gen_dataset[0]))
    base_data    = base_dataset[0][:dataset_size]
    gen_data     = gen_dataset[0][:dataset_size]
    if digitize:
        gen_data     = digitize_input(gen_data,'electron','binning_dataset_2.xml')
    base_inE     = base_dataset[1][:dataset_size]
    gen_inE      = gen_dataset[1][:dataset_size]
    base_label   = torch.ones(dataset_size, device=device)
    gen_label    = torch.zeros(dataset_size, device=device)

    self.padding_value = padding_value
    self.dataset = evaluate_dataset(base_data, base_inE, base_label, device)
    self.dataset.concat(evaluate_dataset(gen_data,  gen_inE,  gen_label,  device))
    self.dataset.padding(self.padding_value)
    self.train_dataset = None
    self.test_dataset  = None
    self.validation_dataset = None
    self.model = None
    self.device=device

  def separate_ttv(self, train_ratio, test_ratio):
    assert (train_ratio + test_ratio) < 1.0
    total_size = len(self.dataset)
    train_size = int(total_size * train_ratio)
    test_size  = int(total_size * test_ratio)
    valid_size = int(total_size - train_size - test_size)
    self.train_dataset, self.test_dataset, self.validation_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size, valid_size])

  def train(self,
            model=None,
            batch_size=150,
            lr = 1e-4,
            jupyternotebook = False,
            mask = True,
            n_epochs = 50,
            indices = [0,1,2,3],
            device = 'cpu',
            output_directory='./'):

    indices = torch.tensor(indices,device=device)

    shower_loader_train = DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True)
    shower_loader_test  = DataLoader(self.test_dataset,  batch_size = batch_size, shuffle=True)
    model.to(device)
    optimiser = Adam(model.parameters(), lr=lr)

    av_training_acc_per_epoch = []
    av_testing_acc_per_epoch  = []

    fig, ax = plt.subplots(ncols=1, figsize=(4,4))

    if jupyternotebook:
        epochs = tqdm.notebook.trange(n_epochs)
        from IPython import display
        dh = display.display(fig, display_id=True)
    else:
        epochs = range(0, n_epochs)

    for epoch in epochs:

        cumulative_epoch_loss = 0.
        cumulative_test_epoch_loss = 0.

        for i, (shower_data, incident_energies, label) in enumerate(shower_loader_train, 0):
            masking = shower_data[:,:,3] == self.padding_value if mask else None
            shower_data = torch.index_select(shower_data,2,indices)
            shower_data = shower_data.to(torch.float32)
            incident_energies = incident_energies.to(torch.float32)
            output_vector = model(shower_data, incident_energies, mask = masking)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(output_vector, torch.flatten(label))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pred   = torch.round(torch.sigmoid(output_vector.detach()))
            target = torch.round(torch.flatten(label).detach())
            if i == 0:
                res_true = target
                res_pred = pred
            else:
                res_true = torch.cat((res_true, target), 0)
                res_pred = torch.cat((res_pred, pred),  0)

        for i, (shower_data, incident_energies, label) in enumerate(shower_loader_test,0):
            with torch.no_grad():
                masking = shower_data[:,:,3] == self.padding_value if mask else None
                shower_data = torch.index_select(shower_data, 2, indices)
                shower_data = shower_data.to(torch.float32)
                incident_energies = incident_energies.to(torch.float32)
                output_vector = model(shower_data, incident_energies, mask=masking)
                pred   = torch.round(torch.sigmoid(output_vector.detach()))
                target = torch.round(torch.flatten(label).detach())
                if i == 0:
                    res_true_test = target
                    res_pred_test = pred
                else:
                    res_true_test = torch.cat((res_true_test, target), 0)
                    res_pred_test = torch.cat((res_pred_test, pred),  0)

        acc_train = accuracy_score(res_true.cpu(), res_pred.cpu())
        acc_test  = accuracy_score(res_true_test.cpu(), res_pred_test.cpu())
        av_training_acc_per_epoch.append(acc_train)
        av_testing_acc_per_epoch.append(acc_test)

        if jupyternotebook:
            epochs.set_description('Accuracy: {:2f}(Train) {:2f}(Test)'.format(acc_train,acc_test))

            fig, ax = plt.subplots(ncols=1, figsize=(4,4))
            plt.title('')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.plot(av_training_acc_per_epoch, label='training')
            plt.plot(av_testing_acc_per_epoch, label='testing')
            plt.legend(loc='lower right')
            dh.update(fig)
            plt.close(fig)
        fig, ax = plt.subplots(ncols=1, figsize=(4,4))
        plt.title('')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.plot(av_training_acc_per_epoch, label='training')
        plt.plot(av_testing_acc_per_epoch, label='testing')
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(output_directory, 'loss_v_epoch.png'))
        plt.close(fig)
    self.model = model
    return model
  def evulate_score(self,
                    model=None,
                    indices = [0,1,2,3],
                    batch_size = 150,
                    mask = True,
                    output_directory = './'):
    if model is None:
        model = self.model
    indices = torch.tensor(indices,device=self.device)
    shower_loader_validation = DataLoader(self.validation_dataset, batch_size = batch_size, shuffle=True)
    for i, (shower_data, incident_energies, label) in enumerate(shower_loader_validation,0):
        with torch.no_grad():
            masking = shower_data[:,:,3] == self.padding_value if mask else None
            shower_data = torch.index_select(shower_data, 2, indices)
            shower_data = shower_data.to(torch.float32)
            incident_energies = incident_energies.to(torch.float32)
            output_vector = model(shower_data, incident_energies, mask=masking)
            pred   = torch.round(torch.sigmoid(output_vector.detach()))
            target = torch.round(torch.flatten(label).detach())
            if i == 0:
                res_true_valid = target
                res_pred_valid = pred
            else:
                res_true_valid = torch.cat((res_true_valid, target), 0)
                res_pred_valid = torch.cat((res_pred_valid, pred),  0)
    confusion_Matrix =  confusion_matrix(res_true_valid.cpu(), res_pred_valid.cpu()) 
    df_cm = pd.DataFrame(confusion_Matrix, index = ['Gen', 'Geant'], columns = ['Gen', 'Geant'])
    plt.figure(figsize = (10,7))
   # sn.heatmap(df_cm, annot=True)
    pp_matrix(df_cm, cmap='Blues')
    plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'))
    return (accuracy_score(res_true_valid.cpu(), res_pred_valid.cpu()))

  def draw_distribution(self, output_directory='./'):

    e_geant4 = []
    e_gen    = []
    x_geant4 = []
    x_gen    = []
    y_geant4 = []
    y_gen    = []
    z_geant4 = []
    z_gen    = []

    nEntries_geant4 = []
    nEntries_gen    = []

    point_clouds_loader = DataLoader(self.dataset, batch_size = 150)
    for i, (shower_data, incident_energies, label) in enumerate(point_clouds_loader, 0):

        valid_event = []
        data_np = shower_data.cpu().numpy().copy()
        label_np = label.cpu().numpy().copy()

        mask = ~(data_np[:,:,3] == self.padding_value)

        for j in range(len(data_np)):
            valid_event = data_np[j][mask[j]]
            if label_np[j] == 0:
                e_gen += ((valid_event).copy()[:,0]).flatten().tolist()
                x_gen += ((valid_event).copy()[:,1]).flatten().tolist()
                y_gen += ((valid_event).copy()[:,2]).flatten().tolist()
                z_gen += ((valid_event).copy()[:,3]).flatten().tolist()
                nEntries_gen.append(len(valid_event))
            else:
                e_geant4 += ((valid_event).copy()[:,0]).flatten().tolist()
                x_geant4 += ((valid_event).copy()[:,1]).flatten().tolist()
                y_geant4 += ((valid_event).copy()[:,2]).flatten().tolist()
                z_geant4 += ((valid_event).copy()[:,3]).flatten().tolist()
                nEntries_geant4.append(len(valid_event))

    fig, ax = plt.subplots(2,3, figsize=(15,10))
    self.get_plot(ax[0][0], "# entries", "Hit entries", nEntries_geant4, nEntries_gen, np.arange(min(nEntries_geant4),max(nEntries_geant4),1))
    self.get_plot(ax[0][1], "# entries", "Hit energies", e_geant4, e_gen, np.arange(min(e_geant4),max(e_geant4),(max(e_geant4)-min(e_geant4))/100.))
    self.get_plot(ax[0][2], "# entries", "x", x_geant4, x_gen, np.arange(min(x_geant4),max(x_geant4),0.1))
    self.get_plot(ax[1][0], "# entries", "y", y_geant4, y_gen, np.arange(min(y_geant4),max(y_geant4),0.1))
    self.get_plot(ax[1][1], "# entries", "z", z_geant4, z_gen, np.arange(min(z_geant4),max(z_geant4),0.1))
    fig.savefig(os.path.join(output_directory, 'distribution.png'))

  def get_plot(self, ax, y_label, x_label, x_geant4, x_gen, bins):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    x_geant4_binned,_ = np.histogram(x_geant4, bins=bins)
    x_gen_binned,_    = np.histogram(x_gen,    bins=bins)
    #chi2 = ((x_geant4_binned - x_gen_binned)**2/(x_geant4_binned)).sum()/len(x_gen_binned)
    ax.hist(x_gen, bins = bins, label="gen", alpha=0.5)
    ax.hist(x_geant4, bins = bins, label="geant4", alpha=0.5)
    ax.legend(loc='upper right')



