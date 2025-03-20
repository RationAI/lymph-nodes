import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self, backbone: str = "convnext_base", pretrained: bool = True
    ) -> None:
        super().__init__()
        self.encoder = models.convnext_base(pretrained=pretrained).features
        self.out_channels = [96, 192, 384, 768]  # ConvNeXt Base feature map sizes

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features  # Returns multi-scale feature maps


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenation with skip connection
        return self.conv(x)


class ConvNeXtU(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.encoder = ConvNeXtEncoder()

        self.decoder4 = DecoderBlock(768, 384)
        self.decoder3 = DecoderBlock(384, 192)
        self.decoder2 = DecoderBlock(192, 96)
        self.decoder1 = DecoderBlock(96, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.encoder(x)

        d4 = self.decoder4(enc_features[-1], enc_features[-2])
        d3 = self.decoder3(d4, enc_features[-3])
        d2 = self.decoder2(d3, enc_features[-4])
        d1 = self.decoder1(d2, enc_features[0])

        out = self.final_conv(d1)
        out = F.interpolate(
            out, size=(512, 512), mode="bilinear", align_corners=False
        )  # Ensure output matches input size
        return out
