"""Main scenario as segmentation, positive TMAs only: TMA epithelium vs negative lymph nodes.

See configs/experiment/training/segmentation_cross_tissue_tma_positive.yaml and
scripts/training/training_jobs.py (needs a GPU and HF_TOKEN).
"""

from training_jobs import submit_scenario


submit_scenario("segmentation_cross_tissue_tma_positive", data="segmentation")
