# main_unet_dann.py
import torch
import torch.nn.functional as F
from torch import nn
from .dbt_base import UNet3D  # 这里导入你的 ISdctgcn.py 中的 UNet3D
from .grl import GRL  # 引用梯度反转层
from .domain_discriminator_3center import DomainDiscriminator  # 引用域判别器

class UNet3DWithDomainAdaptation(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        super(UNet3DWithDomainAdaptation, self).__init__()

        # 使用 ISdctgcn 中的 UNet3D 模型作为主干网络
        self.unet = UNet3D(in_channels=in_channels, out_channels=out_channels, init_features=init_features)

        # 添加域对齐部分
        self.grl = GRL(lambda_=1.0)  # 用于域对抗的梯度反转层
        # 假设池化特征后维度为 8x8x8，并且输入通道数是 64 * 8（即从 encoder4 中获取的特征）
        self.domain_discriminator = DomainDiscriminator(in_features=init_features * 8 * 8 * 8)

    def forward(self, x):
        # 原有 UNet3D 的前向传播部分
        output1, output2 = self.unet(x)
        # Encoder 1
        enc1 = self.unet.encoder1(x)
        # Encoder 2
        enc2 = self.unet.encoder2(self.unet.pool1(enc1))
        # Encoder 3
        enc3 = self.unet.encoder3(self.unet.pool2(enc2))
        # Encoder 4
        enc4 = self.unet.encoder4(self.unet.pool3(enc3)) 
        # 对提取的特征进行池化，以适应域判别器的输入
        # 对提取的特征进行池化，以适应域判别器的输入
        pooled_features = F.adaptive_avg_pool3d(enc4, (4, 4, 4))  # 将池化大小调整为 (4, 4, 4)


        # 通过梯度反转层送入域判别器
        domain_features = self.grl(pooled_features)
        domain_output = self.domain_discriminator(domain_features)

        return output1, output2, domain_output

# 测试代码
if __name__ == '__main__':
    model = UNet3DWithDomainAdaptation(in_channels=1, out_channels=3, init_features=64)
    input_tensor = torch.randn(1, 1, 64, 64, 64)  # Batch size of 1, 1 channel, 64x64x64 volume
    output1, output2, domain_output = model(input_tensor)
    print(output1.shape, output2.shape, domain_output.shape)
