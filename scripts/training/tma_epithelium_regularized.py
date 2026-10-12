"""Baseline scenario, regularized: linear probe on soft cytokeratin targets.

See configs/experiment/training/tma_epithelium_regularized.yaml and
scripts/training/training_jobs.py.
"""

from training_jobs import submit_scenario


submit_scenario("tma_epithelium_regularized")
