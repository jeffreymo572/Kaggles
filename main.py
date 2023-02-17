import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

from common.NeuralNets import Mlp
from common.utils import create_dataset, fit, evaluate

# Data preprocessing
df = pd.read_csv('data/S3E7/train.csv')
train_ds, val_ds = create_dataset(df, 'booking_status')

# Cleaning data

# Model

net = Mlp(input_dim = 18, output_dim = 1, layer_dims=[32, 16, 22])
optimizer = torch.optim.Adam
loss_function = torch.nn.MSELoss()


# Training
epochs = 1000
history = fit(epochs, net, train_ds, val_ds, opt = optimizer)
val_loss = evaluate(net, val_ds)
val_loss_list = [vl['val_loss'] for vl in val_loss]
