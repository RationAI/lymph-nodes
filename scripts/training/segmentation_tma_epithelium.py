"""Baseline scenario as segmentation: per-patch epithelium on TMAs.

See configs/experiment/training/segmentation_tma_epithelium.yaml and scripts/training/training_jobs.py
(needs a GPU and HF_TOKEN).
"""

from training_jobs import submit_scenario


submit_scenario("segmentation_tma_epithelium", data="segmentation")
