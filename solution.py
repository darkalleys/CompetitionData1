import torch.nn as nn
import torch
 
# 假设的模型推理函数，您需要将其替换为您的真实函数
class SVDNet(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(SVDNet, self).__init__()
        self.dim = dim
        self.rank = rank
        self.input_dim = dim * dim * 2  # 实+虚
 
        # 输出 U: [B, 64×32×2]
        self.fc_U = nn.Linear(self.input_dim, dim * rank * 2)
        # 输出 V: [B, 64×32×2]
        self.fc_V = nn.Linear(self.input_dim, dim * rank * 2)
        # 输出 S: [B, 32]
        self.fc_S = nn.Linear(self.input_dim, rank)
 
    def forward(self, x):  # x: [B, 64, 64, 2]
        x = x.view(1, -1)  # Flatten to [B, 64×64×2]
 
        U = self.fc_U(x).view(self.dim, self.rank, 2)
        V = self.fc_V(x).view(self.dim, self.rank, 2)
        S = self.fc_S(x).view(self.rank)
 
        # Normalize U and V columns
        U = self.normalize_columns(U)
        V = self.normalize_columns(V)
 
        return U, S, V
 
    def normalize_columns(self, mat):
        # mat: [B, dim, rank, 2]
        real = mat[..., 0]
        imag = mat[..., 1]
        norm = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        real = real / norm
        imag = imag / norm
        return torch.stack([real, imag], dim=-1)
    