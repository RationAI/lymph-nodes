"""Main scenario as segmentation: per-patch cancer, TMAs plus negative lymph nodes.

See configs/experiment/training/segmentation_cross_tissue.yaml and scripts/training/training_jobs.py
(needs a GPU and HF_TOKEN).
"""

from training_jobs import submit_scenario


submit_scenario("segmentation_cross_tissue", data="segmentation")
