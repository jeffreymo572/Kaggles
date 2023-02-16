import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split

def dataframe_to_arrays(dataframe, target_col_name):
    f"""
    Creates np arrays from Pandas Dataframe. Only works for numerical
    data

    args:
        dataframe[pd.Dataframe]: Pandas dataframe object with no categorical data

    returns:
        inputs_array(np.array): Numpy array of input data
        target_array(np.array): Numpy array of target data
    """
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[dataframe1.loc[:-1]].to_numpy()
    targets_array = dataframe1.iloc[: , :-1].to_numpy()
    return inputs_array, targets_array

def create_dataset(df, target_col_name):
    num_rows = df.shape[0]

    # Spliting into inputs and targets
    inputs_array, target_array = dataframe_to_arrays(df, target_col_name)
    inputs = torch.from_numpy(inputs_array).type(torch.float32)
    targets = torch.from_numpy(target_array).type(torch.float32)
    dataset = TensorDataset(inputs, targets)

    # Creating validation dataset
    val_percent = 0.1
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return train_ds, val_ds

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, opt, lr=1e-3):
    history = []
    optimizer = opt(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history