import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input):
        # 输入形状: (B, num_windows, D, window_size)
        batch_size, num_windows, channels, d_series = input.shape

        # 将每个窗口组展平为 (B*num_windows, D, window_size)，便于并行计算
        input_flat = input.view(batch_size * num_windows, channels, d_series)

        # 设置 FFN
        combined_mean = F.gelu(self.gen1(input_flat))  # (B*num_windows, D, window_size) -> (B*num_windows, D, window_size)
        combined_mean = self.gen2(combined_mean)  # (B*num_windows, D, window_size) -> (B*num_windows, D, d_core)

        # 每组窗口的独立聚合
        combined_mean = combined_mean.view(batch_size, num_windows, channels, -1)  # 恢复为 (B, num_windows, D, d_core)
        if self.training:
            # 随机池化
            ratio = F.softmax(combined_mean, dim=2)  # (B, num_windows, D, d_core)
            ratio = ratio.permute(0, 1, 3, 2)  # 转换为 (B, num_windows, d_core, D)
            ratio_flat = ratio.reshape(-1, channels)  # 展平成二维张量 (B*num_windows*d_core, D)

            # 使用 multinomial 采样
            indices = torch.multinomial(ratio_flat, 1)  # (B*num_windows*d_core, 1)
            indices = indices.view(batch_size, num_windows, -1, 1).permute(0, 1, 3, 2)  # 调整形状为 (B, num_windows, 1, d_core)

            # 根据采样的索引选择通道元素
            combined_mean = torch.gather(combined_mean, 2, indices)  # 在通道维度上选择采样元素 (B, num_windows, 1, d_core)
            combined_mean = combined_mean.repeat(1, 1, channels, 1)  # 复制通道信息 (B, num_windows, D, d_core)
        else:
            # 加权平均
            weight = F.softmax(combined_mean, dim=2)  # (B, num_windows, D, d_core)
            combined_mean = torch.sum(combined_mean * weight, dim=2, keepdim=True).repeat(1, 1, channels, 1)  # (B, num_windows, D, d_core)

        # MLP 融合
        combined_mean_cat = torch.cat([input, combined_mean], -1)  # 拼接 (B, num_windows, D, window_size + d_core)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat.view(batch_size * num_windows, channels, -1)))  # 融合 (B*num_windows, D, window_size)
        combined_mean_cat = self.gen4(combined_mean_cat)  # (B*num_windows, D, window_size)

        # 恢复原始窗口形状 (B, num_windows, D, window_size)
        output = combined_mean_cat.view(batch_size, num_windows, channels, d_series)

        return output

def sliding_window(input, window_size, stride):
    B, D, L = input.shape
    windows = [input[:, :, i:i+window_size] for i in range(0, L - window_size + 1, stride)]
    return torch.stack(windows, dim=1)  # 输出形状: (B, num_windows, D, window_size)

if __name__ == '__main__':
    x1 = torch.randn(10, 64, 20)#.to(device)  # 输入张量 (B, D, L)
    window_size = 10
    stride = 2

    # 滑动窗口划分
    x1_windows = sliding_window(x1, window_size, stride)  # (B, num_windows, D, window_size)
    B, num_windows, D, L = x1_windows.shape

    # 初始化 STAR 模型
    Model = STAR(d_series=window_size, d_core=D)#.to(device)

    # 并行处理每组窗口
    out = Model(x1_windows)  # 输出形状: (B, num_windows, D, window_size)
    print(out.shape)
