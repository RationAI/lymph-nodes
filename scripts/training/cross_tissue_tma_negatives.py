"""Main scenario with negative TMA tissue as a fourth training partition.

See configs/experiment/training/cross_tissue_tma_negatives.yaml and
scripts/training/training_jobs.py.
"""

from training_jobs import submit_scenario


submit_scenario("cross_tissue_tma_negatives")
