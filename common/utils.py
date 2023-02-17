import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split

# Cuda for GPU 
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

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
        train_ds: Training dataset
        val_ds: Validation dataset 
    """
    num_rows = df.shape[0]

    # Spliting into inputs and targets
    inputs_array, target_array = dataframe_to_arrays(df, target_col_name)
    inputs = torch.from_numpy(inputs_array).type(torch.float32) # torch.Tensor([N,M-1])
    targets = torch.from_numpy(target_array).type(torch.float32) # torch.Tensor([N,1])
    dataset = TensorDataset(inputs, targets)

    # Creating validation dataset
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
        model.epoch_end(epoch, result, epochs, 200)
        history.append(result)
    return history