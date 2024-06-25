import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_batch_norm=False, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob

        layers = []

        # Define the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout(self.dropout_prob))

        # Define the hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

        # Define the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int=4, hidden_size: int=512, output_size: int=1, flatten: bool=False):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten() if flatten else None
        self.input_size = input_size
        layers = [
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU()
        ]
        for i in range(hidden_layers - 1):  # Adjusted the range based on your intent
            layers.extend([
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_size, output_size, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, cuts):
        if self.flatten:
            cuts = self.flatten(cuts)
        cuts = self.model(cuts)
        return cuts