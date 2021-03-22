# Defines the model architecture for the feedforward classifier network
# It contains 2 layers, with a ReLU activation after the first layer and
# dropout applied after the first linear layer
import torch
import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_labels, dropout_rate):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(self.dropout(x))
        out = self.fc2(x)
        return out
