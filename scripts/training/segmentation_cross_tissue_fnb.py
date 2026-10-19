"""Main scenario as segmentation with the negative FN Brno lymph nodes as well.

See configs/experiment/training/segmentation_cross_tissue_fnb.yaml and scripts/training/training_jobs.py
(needs a GPU and HF_TOKEN).
"""

from training_jobs import submit_scenario


submit_scenario("segmentation_cross_tissue_fnb", data="segmentation")
