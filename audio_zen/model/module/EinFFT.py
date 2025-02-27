import torch
import torch.nn as nn
import torch.nn.functional as F

class EinFFT(nn.Module):
    def __init__(self, dim, num_blocks=1, sparsity_threshold=0.01, scale=0.02):
        """
        参数：
        - dim: 输入特征的维度（即 num_channels * num_freqs）。
        - num_blocks: 将特征维度划分为的块数，默认值为4。
        - sparsity_threshold: 稀疏性阈值，默认值为0.01。
        - scale: 初始化权重的缩放因子，默认值为0.02。
        """
        super().__init__()
        self.hidden_size = dim  # 输入特征维度
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0, "hidden_size 必须能被 num_blocks 整除"
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale

        # 初始化复数权重和偏置
        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size) * self.scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size) * self.scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, 1) * self.scale  # 调整形状
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, 1) * self.scale  # 调整形状
        )

    def multiply(self, input_tensor, weights):
        # input_tensor: [B, num_blocks, block_size, T]
        # weights: [num_blocks, block_size, block_size]
        # 执行批量矩阵乘法
        # 将 input_tensor 维度调整为 [B, num_blocks, T, block_size]
        input_tensor = input_tensor.permute(0, 1, 3, 2)  # [B, num_blocks, T, block_size]
        # 执行矩阵乘法
        output = torch.einsum('b n t d, n d k -> b n t k', input_tensor, weights)
        # 调整回 [B, num_blocks, block_size, T]
        output = output.permute(0, 1, 3, 2)
        return output

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        x = x.view(B, self.num_blocks, self.block_size, T)  # [B, num_blocks, block_size, T]
        # 对块维度和时间维度应用 2D FFT
        x_fft = torch.fft.fft2(x, dim=(2, 3), norm='ortho')  # [B, num_blocks, block_size, T]
        # 复数线性层操作
        x_real_1 = F.relu(
            self.multiply(x_fft.real, self.complex_weight_1[0]) -
            self.multiply(x_fft.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0]
        )
        x_imag_1 = F.relu(
            self.multiply(x_fft.real, self.complex_weight_1[1]) +
            self.multiply(x_fft.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1]
        )

        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0]) -
            self.multiply(x_imag_1, self.complex_weight_2[1]) +
            self.complex_bias_2[0]
        )
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1]) +
            self.multiply(x_imag_1, self.complex_weight_2[0]) +
            self.complex_bias_2[1]
        )

        # 组合实部和虚部形成复数
        x_combined = torch.stack([x_real_2, x_imag_2], dim=-1).float()

        if self.sparsity_threshold:
            x_combined = F.softshrink(x_combined, lambd=self.sparsity_threshold)
        x_complex = torch.view_as_complex(x_combined)
        # 逆 FFT 返回时域
        x_ifft = torch.fft.ifft2(x_complex, dim=(2, 3), norm='ortho')
        # 获取实部作为输出
        x_out = x_ifft.real
        x_out = x_out.reshape(B, C, T)
        return x_out
