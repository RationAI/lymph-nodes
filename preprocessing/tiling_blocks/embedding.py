from omegaconf import DictConfig
from ray.data import Dataset


class FoundationModelEmbedding:
    """Foundation model embedding via a configurable encoder class.

    Requires a pixel column loaded before this step (e.g. via ReadTiles).
    The encoder class receives ``image_col`` and ``embedding_col``
    via ``fn_constructor_kwargs`` so column naming is handled inside the model.
    """

    def __init__(
        self,
        name: str,
        encoder: DictConfig,
        image_col: str ,
        batch_size: int = 256,
        concurrency: int = 1,
        memory_per_worker: int = 16_000_000_000,
    ) -> None:
        self._name = name
        self._encoder_cfg = encoder
        self._image_col = image_col
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._memory_per_worker = memory_per_worker

    def apply(self, tiles: Dataset) -> Dataset:
        from hydra.utils import get_class

        encoder_cls = get_class(self._encoder_cfg["_target_"])
        return tiles.map_batches(
            encoder_cls,
            fn_constructor_kwargs={"image_col": self._image_col, "embedding_col": self._name},
            batch_size=self._batch_size,
            num_gpus=1,
            concurrency=self._concurrency,
            memory=self._memory_per_worker,
        )
