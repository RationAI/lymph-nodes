"""Custom OmegaConf resolvers used by the training configs.

``register_resolvers()`` must run before Hydra composes/resolves the config, so the
entrypoint calls it at module import time, not inside ``main()``.
"""

from collections.abc import Iterable
from random import randint

from omegaconf import OmegaConf


def _as_list(uris: str | Iterable[str]) -> list[str]:
    return [uris] if isinstance(uris, str) else list(uris)


def other_fold_uris(bases: str | Iterable[str], n_folds: int | str, fold: int | str) -> list[str]:
    """``<base>/fold_<i>`` for every fold except the held-out one, for each run.

    Cross-validation fold selection: no plain OmegaConf interpolation can express "all
    fold_i URIs except the held-out one" (no list-minus-element primitive). Takes one
    run URI or a list of them (one per cohort, each split into the same n_folds), since
    OmegaConf can't concatenate the per-run lists either.
    """
    return [
        f"{base}/fold_{i}"
        for base in _as_list(bases)
        for i in range(int(n_folds))
        if i != int(fold)
    ]


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("random_seed", lambda: randint(0, 2**31), use_cache=True)
    OmegaConf.register_new_resolver("other_fold_uris", other_fold_uris)
