import torch.nn.functional as F 
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

class KeywordHGNN(nn.Module):
    def __init__(self, input_dim=128, embedding_dim=64, hidden_dim=64, dropout=0.5):
        super(KeywordHGNN, self).__init__()
        
        self.hgnn1 = HGNN_conv(input_dim, hidden_dim)
        self.hgnn2 = HGNN_conv(hidden_dim, embedding_dim)  
        # self.dim_weight = nn.Parameter(torch.ones(hidden_dim))
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def forward(self, x, H):
        x = F.relu(self.hgnn1(x, H))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgnn2(x, H)  
        
        return x
    
class HGNN_conv(nn.Module):
    def __init__(self, dim_input, dim_out, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(dim_input, dim_out))
        if bias:
            self.bias = Parameter(torch.Tensor(dim_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameter()
    
    def reset_parameter(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x
    
class HGNN(nn.Module):
    def __init__(self, dim_input, dim_hid, dim_class, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(dim_input, dim_hid)
        self.hgc2 = HGNN_conv(dim_hid, dim_class)
    
    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x