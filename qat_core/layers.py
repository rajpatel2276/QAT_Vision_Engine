import torch.nn as nn
import torch.nn.functional as F
from .quantizer import FakeQuantize

class QATConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
        self.weight_quantizer = FakeQuantize(num_bits=8)
        self.activation_quantizer = FakeQuantize(num_bits=8)

    def forward(self, input):
        quantized_input = self.activation_quantizer(input)
        quantized_weight = self.weight_quantizer(self.weight)
        
        return F.conv2d(
            quantized_input, 
            quantized_weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )