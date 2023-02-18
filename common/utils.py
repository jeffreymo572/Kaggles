import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader

from common.NeuralNets import Mlp
# Wandb for tracking
import wandb
wandb.init(
    project="KaggleS3E7",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "MLP[18,32,16,16,1]",
    "dataset": "Kaggle S3E7",
    "epochs": 1000,
    "data cleaning": "None"
    }
)

def dataframe_to_arrays(dataframe: pd.DataFrame, target_col_name: str):
    f"""
    Splits Pandas Dataframe into training and testing np.array sets. Only works for 
    numerical data

    args:
        dataframe(pd.Dataframe[NxM]): Pandas dataframe object with no categorical data

    returns:
        inputs_array(np.array[N,M-1]): Numpy array of input data
        target_array(np.array[N,1]): Numpy array of target data
    """
    # TODO: Split target array with col name, current only uses last column
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1.iloc[:,:-1].to_numpy() # [N,M-1]
    targets_array = dataframe1.iloc[:,-1:].to_numpy() # [N,1]
    return inputs_array, targets_array

def create_dataset(df:pd.DataFrame, target_col_name:str, val_percent:int = 0.1) -> torch.Tensor:
    r"""
    Creates a dataset that is split into inputs and targets then split into
    training and validation set

    args:
        df(pd.Dataframe[N,M-1]): Pandas dataframe with training data. See 
                                 dataframe_to_arrays for dataframe requirements
        target_col_name(str): Column name for target given input
        val_percent(float): Float representing percent of validation split

    returns:
        train_ds((inputs[i], targets[i])): Training dataset
        val_ds((inputs[i], targets[i])): Validation dataset 
    """
    num_rows = df.shape[0]

    # Spliting into inputs and targets
    inputs_array, target_array = dataframe_to_arrays(df, target_col_name)
    inputs = torch.from_numpy(inputs_array).type(torch.float32) # torch.Tensor([N,M-1])
    targets = torch.from_numpy(target_array).type(torch.float32) # torch.Tensor([N,1])
    dataset = TensorDataset(inputs, targets) # tuple(inputs[i], targets[i])
    loader = DataLoader(dataset, pin_memory=True, num_workers=4, shuffle=True)
    
    # TODO: Creating dataset using Dataloader instead of TensorDataset
    print(len(loader))
    """
    # Creating validation dataset
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size
    train_ds, val_ds = random_split(loader, [train_size, val_size])
    
    return train_ds, val_ds
    """
    return None, None

def evaluate(model: Mlp, val_loader: torch.Tensor, device:str):
    r"""
    Evaluation of model using validation set

    args:
        model(Mlp): Trained model
        val_load(torch.Tensor): Validation set with N tuples of (input,target) pairs
        device(str): Device to use, pref 'cuda:0' for GPU

    returns:
        outputs(dict): Dictionary of outputs from all validation steps with at least
                        the following items:
        * val_loss(float): Validation loss for each validation step
        * accuracy(int): If prediction was correct (1) or not (0) for each validation step
    """
    outputs = []
    for batch in val_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        outputs.append(model.validation_step((x,y)))
    return model.validation_epoch_end(outputs)

def fit(epochs:int, model: Mlp, train_loader: torch.Tensor, val_loader: torch.Tensor, 
        device:str, opt: torch.optim, lr:float=1e-3):
    r"""
    Training loop for model fitting
    """
    history = []
    losses = []
    optimizer = opt(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        val_loss, val_accuracy = evaluate(model, val_loader, device=device)
        model.epoch_end(epoch, val_loss, epochs, 200)
        history.append(val_loss)

        # Tracking with wandb
        # TODO: Test evaluation accuracy on Wandb
        wandb.log({"Evaluation result": val_loss, "Loss": np.mean(losses),
                   "Evaluation Accuracy": val_accuracy/len(batch)})
        losses = []
    
    return history