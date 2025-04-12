import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STAR3(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR3, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        # 输入形状: (B, num_windows, D, window_size)
        batch_size, num_windows, channels, d_series = input.shape

        # 将每个窗口展平为 (B*num_windows, D, window_size)，便于并行处理
        input_flat = input.view(batch_size * num_windows, channels, d_series)

        # 设置 FFN
        combined_mean = F.gelu(self.gen1(input_flat))  # (B*num_windows, D, window_size) -> (B*num_windows, D, window_size)
        combined_mean = self.gen2(combined_mean)  # (B*num_windows, D, window_size) -> (B*num_windows, D, d_core)

        # 每组窗口的独立聚合
        combined_mean = combined_mean.view(batch_size, num_windows, channels, -1)  # 恢复为 (B, num_windows, D, d_core)
        if self.training:
            # 随机池化
            ratio = F.softmax(combined_mean, dim=2)  # (B, num_windows, D, d_core)
            ratio = ratio.permute(0, 1, 3, 2).reshape(-1, channels)  # (B*num_windows*d_core, D)
            indices = torch.multinomial(ratio, 1)  # 采样 (B*num_windows*d_core, 1)
            indices = indices.view(batch_size, num_windows, -1, 1).permute(0, 1, 3, 2)  # (B, num_windows, 1, d_core)
            combined_mean = torch.gather(combined_mean, 2, indices)  # (B, num_windows, 1, d_core)
            core = combined_mean.squeeze(2)  # (B, num_windows, d_core)
            combined_mean = combined_mean.repeat(1, 1, channels, 1)  # (B, num_windows, D, d_core)
        else:
            # 加权平均
            weight = F.softmax(combined_mean, dim=2)  # (B, num_windows, D, d_core)
            combined_mean = torch.sum(combined_mean * weight, dim=2, keepdim=True)  # (B, num_windows, 1, d_core)
            core = combined_mean.squeeze(2)  # (B, num_windows, d_core)
            combined_mean = combined_mean.repeat(1, 1, channels, 1)  # (B, num_windows, D, d_core)

        # MLP 融合
        combined_mean_cat = torch.cat([input, combined_mean], -1)  # 拼接 (B, num_windows, D, window_size + d_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat.view(batch_size * num_windows, channels, -1)))  # (B*num_windows, D, window_size)
        combined_mean_cat = self.gen4(combined_mean_cat)  # (B*num_windows, D, window_size)

        # 恢复原始窗口形状 (B, num_windows, D, window_size)
        output = combined_mean_cat.view(batch_size, num_windows, channels, d_series)

        return output, core

def sliding_window(input, window_size, stride):
    B, D, L = input.shape
    windows = [input[:, :, i:i + window_size] for i in range(0, L - window_size + 1, stride)]
    return torch.stack(windows, dim=1)  # 输出形状: (B, num_windows, D, window_size)

if __name__ == '__main__':
    x1 = torch.randn(184, 116, 175)  # 输入张量 (B, D, L)
    window_size = 35
    stride = 35

    # 滑动窗口划分
    x1_windows = sliding_window(x1, window_size, stride)  # (B, num_windows, D, window_size)
    B, num_windows, D, L = x1_windows.shape

    # 初始化 STAR3 模型
    Model = STAR3(d_series=window_size, d_core=D)

    # 并行处理每组窗口
    out, core = Model(x1_windows)  # 输出形状: (B, num_windows, D, window_size) 和 (B, num_windows, d_core)
    print("Output shape:", out.shape)
    print("Core shape:", core.shape)
