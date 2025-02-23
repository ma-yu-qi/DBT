import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch_geometric.nn import GCNConv  # 引入GCNConv

# DCT和IDCT的PyTorch封装
def torch_dct2(x):
    # 将PyTorch张量转换为NumPy数组，执行DCT
    x_np = x.detach().cpu().numpy()
    x_dct = dct(dct(x_np, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return torch.from_numpy(x_dct).to(x.device, x.dtype)

def torch_idct2(x_dct):
    # 将PyTorch张量转换为NumPy数组，执行IDCT
    x_dct_np = x_dct.detach().cpu().numpy()
    x_idct = idct(idct(x_dct_np, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return torch.from_numpy(x_idct).to(x_dct.device, x_dct.dtype)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Global Average Pooling
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """
        super(UNet3D, self).__init__()


        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.gcn = GCNConv(features * 8, features * 8)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        # Additional Encoders and Decoders for Low-pass and High-pass
        self.encoder1_ = UNet3D._block(in_channels, features, name="enc1_")
        self.pool1_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2_ = UNet3D._block(features, features * 2, name="enc2_")
        self.pool2_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3_ = UNet3D._block(features * 2, features * 4, name="enc3_")
        self.pool3_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4_ = UNet3D._block(features * 4, features * 8, name="enc4_")
        self.pool4_ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck_ = UNet3D._block(features * 8, features * 16, name="bottleneck_")
        self.upconv4_ = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_ = UNet3D._block((features * 8) * 2, features * 8, name="dec4_")
        self.upconv3_ = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_ = UNet3D._block((features * 4) * 2, features * 4, name="dec3_")
        self.upconv2_ = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_ = UNet3D._block((features * 2) * 2, features * 2, name="dec2_")
        self.upconv1_ = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_ = UNet3D._block(features * 2, features, name="dec1_")

        self.encoder1__ = UNet3D._block(in_channels, features, name="enc1__")
        self.pool1__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2__ = UNet3D._block(features, features * 2, name="enc2__")
        self.pool2__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3__ = UNet3D._block(features * 2, features * 4, name="enc3__")
        self.pool3__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4__ = UNet3D._block(features * 4, features * 8, name="enc4__")
        self.pool4__ = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck__ = UNet3D._block(features * 8, features * 16, name="bottleneck__")
        self.upconv4__ = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4__ = UNet3D._block((features * 8) * 2, features * 8, name="dec4__")
        self.upconv3__ = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3__ = UNet3D._block((features * 4) * 2, features * 4, name="dec3__")
        self.upconv2__ = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2__ = UNet3D._block((features * 2) * 2, features * 2, name="dec2__")
        self.upconv1__ = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1__ = UNet3D._block(features * 2, features, name="dec1__")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.conv_ = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # SE block for feature fusion
        self.se = SEBlock(features)

    def lowpass_torch(self, input, limit):
        dct_input = torch_dct2(input)
        # 创建低通滤波器
        pass1 = torch.abs(torch.arange(input.shape[-1], device=input.device)) < limit
        pass2 = torch.abs(torch.arange(input.shape[-2], device=input.device)) < limit
        kernel = torch.outer(pass2, pass1).float()
        
        # 应用低通滤波器
        filtered_dct = dct_input * kernel.unsqueeze(0).unsqueeze(0)
        return torch_idct2(filtered_dct)

    def highpass_torch(self, input, limit):
        dct_input = torch_dct2(input)
        # 创建高通滤波器
        pass1 = torch.abs(torch.arange(input.shape[-1], device=input.device)) > limit
        pass2 = torch.abs(torch.arange(input.shape[-2], device=input.device)) > limit
        kernel = torch.outer(pass2, pass1).float()
        
        # 应用高通滤波器
        filtered_dct = dct_input * kernel.unsqueeze(0).unsqueeze(0)
        return torch_idct2(filtered_dct)

    def forward(self, x):
        low_x = self.lowpass_torch(x,0.1)
        high_x = self.highpass_torch(x,0.05)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        # Flattening for GCN
        b, c, d, h, w = enc4.size()
        enc4_flat = enc4.view(b, c, -1).permute(0, 2, 1).contiguous()
        node_features = enc4_flat.view(-1, c)
        
        # Constructing adjacency matrix (example: using fully connected graph)
        num_nodes = node_features.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

        # Ensure edge_index is on the same device as node_features
        edge_index = edge_index.to(node_features.device)

        # Apply Graph Convolution
        gcn_output = self.gcn(node_features, edge_index)
        gcn_output = gcn_output.view(b, d * h * w, c).permute(0, 2, 1).contiguous().view(b, c, d, h, w)

        bottleneck = self.bottleneck(self.pool4(gcn_output))


        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Low-pass Branch
        enc1_ = self.encoder1_(low_x)
        enc2_ = self.encoder2_(self.pool1_(enc1_))
        enc3_ = self.encoder3_(self.pool2_(enc2_))
        enc4_ = self.encoder4_(self.pool3_(enc3_))
        bottleneck_ = self.bottleneck_(self.pool4_(enc4_))
        dec4_ = self.upconv4_(bottleneck_)
        dec4_ = torch.cat((dec4_, enc4_), dim=1)
        dec4_ = self.decoder4_(dec4_)
        dec3_ = self.upconv3_(dec4_)
        dec3_ = torch.cat((dec3_, enc3_), dim=1)
        dec3_ = self.decoder3_(dec3_)
        dec2_ = self.upconv2_(dec3_)
        dec2_ = torch.cat((dec2_, enc2_), dim=1)
        dec2_ = self.decoder2_(dec2_)
        dec1_ = self.upconv1_(dec2_)
        dec1_ = torch.cat((dec1_, enc1_), dim=1)
        dec1_ = self.decoder1_(dec1_)

        # High-pass Branch
        enc1__ = self.encoder1__(high_x)
        enc2__ = self.encoder2__(self.pool1__(enc1__))
        enc3__ = self.encoder3__(self.pool2__(enc2__))
        enc4__ = self.encoder4__(self.pool3__(enc3__))
        bottleneck__ = self.bottleneck__(self.pool4__(enc4__))
        dec4__ = self.upconv4__(bottleneck__)
        dec4__ = torch.cat((dec4__, enc4__), dim=1)
        dec4__ = self.decoder4__(dec4__)
        dec3__ = self.upconv3__(dec4__)
        dec3__ = torch.cat((dec3__, enc3__), dim=1)
        dec3__ = self.decoder3__(dec3__)
        dec2__ = self.upconv2__(dec3__)
        dec2__ = torch.cat((dec2__, enc2__), dim=1)
        dec2__ = self.decoder2__(dec2__)
        dec1__ = self.upconv1__(dec2__)
        dec1__ = torch.cat((dec1__, enc1__), dim=1)
        dec1__ = self.decoder1__(dec1__)

        # SE Fusion for dec1_ + dec1__
        fusion = dec1_ + dec1__
        se_output = self.se(fusion)

        outputs1 = self.conv(dec1)
        outputs2 = self.conv_(dec1 + se_output)
        return outputs1, outputs2


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
if __name__ == '__main__':
    model = UNet3D(in_channels=1, out_channels=3, init_features=64)
    input_tensor = torch.randn(1, 1, 64, 64, 64)  # Batch size of 1, 1 channel, 64x64x64 volume
    output1, output2 = model(input_tensor)
    print(output1.shape, output2.shape)