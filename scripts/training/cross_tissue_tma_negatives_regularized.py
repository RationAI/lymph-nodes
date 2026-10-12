"""Main scenario with TMA negatives, regularized: linear probe on soft TMA targets.

See configs/experiment/training/cross_tissue_tma_negatives_regularized.yaml and
scripts/training/training_jobs.py.
"""

from training_jobs import submit_scenario


submit_scenario("cross_tissue_tma_negatives_regularized")
