import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, x_n, h_n, y_n, t) -> None:
        super().__init__()

        self.t = t

        self.w_in = nn.Parameter(torch.empty(x_n, h_n), requires_grad=False)
        self.w = nn.Parameter(torch.empty(h_n, h_n), requires_grad=True)
        self.b = nn.Parameter(torch.empty(h_n), requires_grad=True)
        self.w_out = nn.Parameter(torch.empty(h_n, y_n), requires_grad=True)
        self.b_out = nn.Parameter(torch.empty(y_n), requires_grad=True)
    
    def forward(self, x):
        x = torch.tanh(x @ self.w_in + self.b)
        for _ in range(self.t - 1):
            x = torch.tanh(x @ self.w + self.b)
        x = x @ self.w_out + self.b_out
        return x
    
    @torch.no_grad()
    def step(self, x, first=False):
        if first:
            x = torch.tanh(x @ self.w_in + self.b)
        else:
            x = torch.tanh(x @ self.w + self.b)
        return x




