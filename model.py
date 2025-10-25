import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.register_buffer("mask", mask)

    def forward(self, X):
        masked_weight = self.weight * self.mask
        return F.linear(X, masked_weight, self.bias)


class PNet(nn.Module):
    def __init__(
        self,
        input_dim,
        gene_dim,
        pathway_dim,
        hidden_dim=128,
        output_dim=2,
        gene_mask=None,
    ):
        super().__init__()
        # self.l1 = nn.Linear(input_dim, gene_dim)
        self.l2 = MaskedLinear(gene_dim, pathway_dim, gene_mask.T)
        self.geneout = nn.Linear(pathway_dim, output_dim)
        self.l3 = nn.Linear(pathway_dim, hidden_dim)
        self.pathout = nn.Linear(hidden_dim, output_dim)

        self.l4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        g = F.sigmoid(self.geneout(x))
        x = F.relu(self.l3(x))
        p = F.sigmoid(self.pathout(x))
        x = F.sigmoid(self.l4(x))
        x = F.softmax(x + p + g, dim=1)
        return x
