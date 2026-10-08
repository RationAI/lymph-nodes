"""Baseline scenario: epithelium classifier trained on TMAs only.

See configs/experiment/training/tma_epithelium.yaml and scripts/training/training_jobs.py.
"""

from training_jobs import submit_scenario


submit_scenario("tma_epithelium")
