# Customized code from: https://github.com/VITA-Group/Deep_GCN_Benchmarking/blob/main/tricks/tricks/skipConnections.py

import torch
from torch import nn


class FullResidualConnection(nn.Module):
    def __init__(self):
        super(FullResidualConnection, self).__init__()

    def forward(self, Xs: list):
        # Xs: List of size MP, of: [num, hidden_dim]

        assert len(Xs) >= 1

        if len(Xs) == 1:
            return Xs[-1]
        else:
            return Xs[-1] + Xs[-2]


class ResidualConnection(nn.Module):
    def __init__(self, alpha=0.5):
        super(ResidualConnection, self).__init__()

        self.alpha = alpha

    def forward(self, Xs: list):
        # Xs: List of size MP, of: [num, hidden_dim]

        assert len(Xs) >= 1

        return (
            Xs[-1] if len(Xs) == 1 else (1 - self.alpha) * Xs[-1] + self.alpha * Xs[-2]
        )


class InitialConnection(nn.Module):
    def __init__(self, alpha=0.5):
        super(InitialConnection, self).__init__()

        self.alpha = alpha

    def forward(self, Xs: list):
        # Xs: List of size MP, of: [num, hidden_dim]

        assert len(Xs) >= 1

        return (
            Xs[-1] if len(Xs) == 1 else (1 - self.alpha) * Xs[-1] + self.alpha * Xs[0]
        )


class DenseConnection(nn.Module):
    def __init__(self, in_dim, out_dim, aggregation="concat"):
        super(DenseConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregation = aggregation

        if aggregation == "concat":
            self.layer_transform = nn.Linear(in_dim, out_dim, bias=True)
        elif aggregation == "attention":
            # self.layer_att = nn.Linear(in_dim, 1, bias=True)
            # fix: should be linear layer using "out_dim" not "in_dim"
            self.layer_att = nn.Linear(out_dim, 1, bias=True)

    def forward(self, Xs: list):
        # Xs: List of size MP, of: [num, hidden_dim]

        assert len(Xs) >= 1

        if self.aggregation == "concat":
            # [num, hidden_dim * (mp+1)] - Concat all MP features
            X = torch.cat(Xs, dim=-1)
            # [num, hidden_dim] - Linear layout output
            X = self.layer_transform(X)
            return X

        elif self.aggregation == "maxpool":
            # [num, hidden_dim, (mp+1)] - Stack all MP features together, MP dim gets placed at end
            X = torch.stack(Xs, dim=-1)
            # [num, hidden_dim] - Pools max feature per hidden dim, across the MP dimension
            X, _ = torch.max(X, dim=-1, keepdim=False)
            return X

        # implement with the code from https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/dagnn.py
        elif self.aggregation == "attention":
            # (orig note) pps n x k+1 x c
            # [num, (mp+1), hidden_dim] - Stack all MP features together, MP dim gets placed at idx 1
            pps = torch.stack(Xs, dim=1)
            # [num, (mp+1)] - Linear layout output
            retain_score = self.layer_att(pps).squeeze()
            # [num, 1, (mp+1)] - sigmoid & unsqueeze
            retain_score = torch.sigmoid(retain_score).unsqueeze(1)
            # [num, hidden_dim] - matmul for attention
            X = torch.matmul(retain_score, pps).squeeze()
            return X

        else:
            raise Exception("Unknown aggregation")
