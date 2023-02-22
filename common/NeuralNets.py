import numpy as np
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  r"""
  Used for better generalization and exploration from paper: 
  https://arxiv.org/abs/1706.10295?context=stat.ML

  Args:
    in_features (int): Input dimensions
    out_features (int): Output dimensions
    std_init (float): std for noise. More std means higher exploration
  """
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class Mlp(nn.Module):
    """
    Simple multi-layer perceptron net (densly connected net)
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        layer_dims (List[int]): Dimensions of hidden layers
        activation (str): type of activations. Not applying to the last layer 
    """
    def __init__(self, input_dim, output_dim, device, layer_dims=[], activation='swish'):
        super(Mlp, self).__init__()

        self.device = device
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim

        if len(layer_dims) != 0:
            self.layers.append(nn.Linear(input_dim, layer_dims[0]))

            for i in range(len(layer_dims)-1):
                if activation == 'relu':
                  self.layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                  self.layers.append(nn.Sigmoid())
                elif activation == 'swish':
                  self.layers.append(nn.SELU())
                self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            self.layers.append(nn.Sigmoid())
            self.layers.append(nn.Linear(layer_dims[-1], output_dim))

        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        # Composing all layers
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: tuple) -> torch.Tensor:
        r"""
        Training step

        args:
          batch((inputs, targets)): Tuple of N (input, target) pairs
        
        return:
          loss(Tensor): Loss as a result of loss function (MSE)
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # Generate predictions
        out = self(inputs).to(self.device)
        # Calcuate loss
        loss = F.binary_cross_entropy_with_logits(out, targets.reshape(-1,1))
        accuracy = 1-torch.absolute(torch.Tensor(out-targets))
        return loss, accuracy

    def validation_step(self, batch: tuple)-> dict:
        r"""
        Model prediction for validation batch

        args:
          batch((inputs, target)): (Inputs, target) tuple

        returns:
          results(dict): Dictionary of the following items
            * val_loss(float): Validation loss
            * accuracy(int): If prediction was correct (1) or not (0)
        """
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(out, targets) # Similar to accuracy
        # Generate accuracy
        error = F.l1_loss(out, targets)
        correct = 1 if out == targets else 0

        return {'val_loss': loss.detach(), 'accuracy': 1-error.detach(), 'correct': correct}

    def validation_epoch_end(self, outputs:dict)->float:
        r"""
        Returns mean and accuracy of validation batch after all validation steps

        args:
          outputs(dict): Data from all validation phase prediction
        returns:
          outputs(dict): Dictionary of outputs from all validation steps with at least
                         the following items:
          * val_loss(float): Mean validation loss of all validation steps
          * accuracy(int): Mean accuracy of all validation steps (see validation_step() 
                          for accuracy calc)
        """
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [a['accuracy'] for a in outputs]
        batch_correctness = [c['correct'] for c in outputs]

        epoch_loss = torch.stack(batch_losses).mean()
        accuracy = torch.stack(batch_accuracy).mean()
        correctness = np.array(batch_correctness).mean()

        return epoch_loss.item(), accuracy.item(), correctness
    
    def epoch_end(self, epoch:int, val_loss:float, num_epochs:int, n:int):
        r"""
        Prints every n epoch given results
        """
        # Print result every 20th epoch
        if (epoch+1) % n == 0 or epoch == num_epochs-1:
            print(f"Epoch [{epoch+1}], val_loss: {val_loss:.4f}")