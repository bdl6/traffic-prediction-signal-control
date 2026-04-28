import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self):
        adj = F.relu(torch.matmul(self.E1, self.E2.t()))
        adj = F.softmax(adj, dim=1)
        return adj

class GatedDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        filter, gate = x.chunk(2, dim=-1)
        return torch.tanh(filter) * torch.sigmoid(gate)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        return self.linear(x)

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes=12, in_dim=1, hid_dim=32, embed_dim=10, pre_steps=12):
        super().__init__()
        self.adp_adj = AdaptiveAdjacency(num_nodes, embed_dim)
        self.start_conv = nn.Conv1d(in_dim, hid_dim, 1)

        self.tc1 = GatedDilatedConv(hid_dim, hid_dim, 1)
        self.gc1 = GraphConvolution(hid_dim, hid_dim)

        self.tc2 = GatedDilatedConv(hid_dim, hid_dim, 2)
        self.gc2 = GraphConvolution(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, pre_steps)

    def forward(self, x):
        adj = self.adp_adj()
        x = x.transpose(1, 2)
        x = self.start_conv(x)
        x = x.transpose(1, 2)

        x = self.tc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.gc1(x, adj)
        x = self.tc2(x.transpose(1, 2)).transpose(1, 2)
        x = self.gc2(x, adj)

        pred = self.fc(x)
        return pred.transpose(1, 2)