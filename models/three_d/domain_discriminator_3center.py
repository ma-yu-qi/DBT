# domain_discriminator.py
import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features, num_domains=3):  # 设置域的数量为 3
        super(DomainDiscriminator, self).__init__()
        self.domain = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_domains),
            # 这里不要用 Softmax，因为交叉熵损失函数会自动计算 softmax
        )

    def forward(self, x):
        # 将输入特征展平为二维 (batch_size, num_features)
        x = x.view(x.size(0), -1)
        return self.domain(x)

# class DomainDiscriminator(nn.Module):
#     def __init__(self, in_features, num_classes=6):
#         super(DomainDiscriminator, self).__init__()
#         self.domain = nn.Sequential(
#             nn.Linear(in_features, 256),    # 第一层全连接，将输入特征映射到256维
#             nn.ReLU(inplace=True),          # ReLU激活函数
#             nn.Linear(256, 128),            # 第二层全连接，将特征从256维降到128维
#             nn.ReLU(inplace=True),          # ReLU激活函数
#             nn.Linear(128, num_classes),    # 第三层全连接，将特征从128维映射到类别数量维度（这里是6）
#             nn.Softmax(dim=1)               # 使用Softmax激活函数，使输出为6分类的概率分布
#         )

#     def forward(self, x):
#         # 将输入特征展平为二维 (batch_size, num_features)
#         x = x.view(x.size(0), -1)
#         return self.domain(x)
