import torch
import torch.nn as nn

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def round_ste(x):
    return RoundSTE.apply(x)

class FakeQuantize(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.qmin = -128
        self.qmax = 127

    def forward(self, x):
        min_val = x.min().detach()
        max_val = x.max().detach()

        scale = (max_val - min_val) / (self.qmax - self.qmin)
        scale = torch.max(scale, torch.tensor(1e-8, device=x.device)) 
        
        zero_point = self.qmin - round_ste(min_val / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        x_q = round_ste(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)

        x_fake = scale * (x_q - zero_point)
        
        return x_fake