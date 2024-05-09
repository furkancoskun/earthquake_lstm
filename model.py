import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=True,
                            ) 
        self.out_layer = nn.Linear(in_features=hidden_size,
                                   out_features=1)

    def forward(self, inputs:torch.Tensor):
        input_ = inputs.view(len(inputs), 1, -1)
        h0 = torch.randn(self.num_layers, 1, self.hidden_size, device=input_.device, dtype=input_.dtype)
        c0 = torch.randn(self.num_layers, 1, self.hidden_size, device=input_.device, dtype=input_.dtype)
        lstm_out, (hn, cn) = self.lstm(input_, (h0, c0))
        out = self.out_layer(lstm_out)
        return out
