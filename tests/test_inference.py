import albumentations
import mlflow
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from openslide import OpenSlide
from PIL import Image

from lymph_nodes.modeling.backbone.convnext import ConvNeXt


CHECKPOINT_PATH = "mlflow-artifacts:/68/92b7f6f5a4834cef830e4c1f80ce5c9d/artifacts/checkpoints/epoch=20-step=42000/checkpoint.ckpt"
SLIDE_PATH = ""
S_COORDS = (53000, 123000)
E_COORDS = (55000, 125000)
LEVEL = 1


# Download the model checkpoint
ckpt_path = mlflow.artifacts.download_artifacts(
    CHECKPOINT_PATH,
    dst_path="test/data",
)


model = ConvNeXt()
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()

to_tensor = ToTensorV2()
normalize = albumentations.Normalize(
    mean=(215.6701, 225.0577, 237.1580),
    std=(32.2908, 24.3962, 14.4058),
    max_pixel_value=1,
    p=1.0,
)

with OpenSlide(SLIDE_PATH) as slide:
    tile_extent = tuple(
        np.round(
            (np.asarray(E_COORDS) - np.asarray(S_COORDS))
            / slide.level_downsamples[LEVEL]
        )
    )
    # Read a region from the slide
    rgba_region = slide.read_region(S_COORDS, LEVEL, tile_extent)
    rgb_region = Image.alpha_composite(
        Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
    ).convert("RGB")

    region = np.array(rgb_region)
    image = to_tensor(normalize(image=region))["image"]

    batch = image.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(batch)

    Image.fromarray((output.detach().cpu().numpy()[0] * 255), mode="L").save(
        "test/data/segmentation.png"
    )
