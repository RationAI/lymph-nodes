import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        # Load pre-trained ConvNeXt backbone
        self.backbone = models.convnext_base(
            # weights=models.ConvNeXt_Base_Weights.DEFAULT
            weights=None
        )

        # Extract encoder stages
        self.encoder_layers = list(self.backbone.features.children())
        self.stage1 = nn.Sequential(*self.encoder_layers[:2])
        self.stage2 = nn.Sequential(*self.encoder_layers[2:4])
        self.stage3 = nn.Sequential(*self.encoder_layers[4:6])
        self.stage4 = nn.Sequential(*self.encoder_layers[6:8])

        # Bottleneck
        self.bottleneck = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4)
        self.dec1 = self.conv_block(128, 64)
        self.adjust_channels = nn.Conv2d(3, 64, kernel_size=1)

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.stage1(x)
        enc2 = self.stage2(enc1)
        enc3 = self.stage3(enc2)
        enc4 = self.stage4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc1], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), self.adjust_channels(x)], dim=1))

        # Segmentation output
        return self.segmentation_head(dec1)
