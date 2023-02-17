import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from common.NeuralNets import Mlp
from common.utils import create_dataset, fit, evaluate

# Cuda for GPU 
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

# Data preprocessing
df = pd.read_csv('data/S3E7/train.csv')
# Cleaning data
df = df.drop(['id'], axis=1) # Drop id
# TODO: See data visualization ipynb
input_dim = df.shape[1]-1

# Create dataset
train_ds, val_ds = create_dataset(df, 'booking_status')

# Model and hyperparams
net = Mlp(input_dim = input_dim, output_dim = 1, layer_dims=[64, 16], device=device)
net = net.to(device)
optimizer = torch.optim.Adam
loss_function = torch.nn.MSELoss()

"""
# Training
epochs = 100
history = fit(epochs, net, train_ds, val_ds, device, opt = optimizer)
val_loss = evaluate(net, val_ds, device)
val_loss_list = [vl['val_loss'] for vl in val_loss]
"""