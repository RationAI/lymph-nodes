"""Cross-tissue scenario: lymph node negatives + cytokeratin-positive TMA tiles.

See configs/experiment/training/cross_tissue.yaml and scripts/training/training_jobs.py.
"""

from training_jobs import submit_scenario


submit_scenario("cross_tissue")
