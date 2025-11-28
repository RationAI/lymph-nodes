import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts


class DataSource(Iterable[Path]):
    def __init__(
        self,
        paths: list[str],
    ) -> None:
        self.data = [Path(path) for path in paths]

    def __iter__(self) -> Iterator[Path]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class GlobDataSource(Iterable[Path]):
    def __init__(
        self,
        src_dir: str,
        glob_pattern: str | list[str] = "*",
        exclue_pattern: None | str | list[str] = None,
    ) -> None:
        dir = Path(src_dir)

        self.exclude_paths = (
            list(self._paths(dir, exclue_pattern)) if exclue_pattern else []
        )
        self.include_paths = list(self._paths(dir, glob_pattern))

        self.data = [
            path for path in self.include_paths if path not in self.exclude_paths
        ]

    def __iter__(self) -> Iterator[Path]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _paths(dir: Path, pattern: str | list[str]) -> Iterable[Path]:
        return itertools.chain(
            *(
                dir.rglob(pat)
                for pat in ([pattern] if isinstance(pattern, str) else pattern)
            )
        )


class ChainedDataSources(Iterable[Path]):
    def __init__(self, sources: list[DataSource | GlobDataSource]) -> None:
        self.data_sources = sources

    def __iter__(self) -> Iterator[Path]:
        return itertools.chain(*(iter(ds) for ds in self.data_sources))

    def __len__(self) -> int:
        return sum([len(ds) for ds in self.data_sources])


class SlideDataSource(Iterable[Path]):
    def __init__(self, uris: list[str]) -> None:
        self.datasets = [self._download_dataset(uri) for uri in uris]

    def __iter__(self) -> Iterator[Path]:
        for dataset in self.datasets:
            for slide_path in dataset["slide_path"]:
                yield Path(slide_path)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def _download_dataset(self, uri: str) -> pd.DataFrame:
        artifact_path = download_artifacts(artifact_uri=uri)
        return pd.read_csv(artifact_path)
