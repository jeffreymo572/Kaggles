import numpy as np
from statistics import mean

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader

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

def dataframe_to_arrays(dataframe, target_col_name):
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

def create_dataset(df, target_col_name, val_percent = 0.1):
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

    print(len(loader))
    """
    # Creating validation dataset
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size
    train_ds, val_ds = random_split(loader, [train_size, val_size])
    
    return train_ds, val_ds
    """
    return None, None

def evaluate(model, val_loader, device):
    outputs = []
    for batch in val_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        outputs.append(model.validation_step((x,y)))
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, device, opt, len_train_set,
        lr=1e-3):
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
        wandb.log({"Evaluation result": val_loss, "Loss": np.mean(losses),
                   "Evaluation Accuracy": val_accuracy/len(batch)})
        losses = []
    
    return history