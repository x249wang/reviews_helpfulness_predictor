import torch
import torch.nn as nn

IN_DIM = 768
HIDDEN_DIM = 256
NUM_LABELS = 2
DROPOUT_RATE = 0.1
MODEL_PATH = "assets/model.pth"


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


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = ClassifierModel(IN_DIM, HIDDEN_DIM, NUM_LABELS, DROPOUT_RATE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
model.eval()
