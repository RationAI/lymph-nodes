"""Brown-staining baseline: brownish_coverage as the score, validated like the models.

    python scripts/training/brown_baseline.py uni2 --folds 0 --gpu none                  # classification tiling
    python scripts/training/brown_baseline.py uni2 --folds 0 --gpu none --segmentation   # segmentation tiling

The model in the data config doesn't matter (brownish_coverage is the same in every
tiling run of a stride). See configs/experiment/training/brown_baseline.yaml.
"""

import sys

from training_jobs import submit_scenario


segmentation = "--segmentation" in sys.argv
if segmentation:
    sys.argv.remove("--segmentation")
submit_scenario(
    "brown_baseline",
    data="segmentation" if segmentation else "tiled",
    trained=False,
    foundation_model=False,
)
