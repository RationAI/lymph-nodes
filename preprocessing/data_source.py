import itertools
from collections.abc import Iterable, Iterator, Sized
from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts


class SizedIterable(Sized, Iterable[str]):
    """A base class for data sources that are both sized and iterable."""


class DataSource(SizedIterable):
    def __init__(self, df: pd.DataFrame, path_key: str) -> None:
        self.data = df
        self.path_key = path_key

    def __iter__(self) -> Iterator[str]:
        return iter(self.data[self.path_key])

    def __len__(self) -> int:
        return len(self.data)

    def to_pandas(self) -> pd.DataFrame:
        return self.data


class ListDataSource(DataSource):
    def __init__(
        self,
        paths: list[str] | list[Path],
    ) -> None:
        super().__init__(
            pd.DataFrame({"slide_path": [str(path) for path in paths]}), "slide_path"
        )


class GlobDataSource(ListDataSource):
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

        super().__init__(
            [path for path in self.include_paths if path not in self.exclude_paths]
        )

    @staticmethod
    def _paths(dir: Path, pattern: str | list[str]) -> Iterable[Path]:
        return itertools.chain(
            *(
                dir.rglob(pat)
                for pat in ([pattern] if isinstance(pattern, str) else pattern)
            )
        )


class MLFlowDataSource(DataSource):
    def __init__(self, uri: str, path_key: str = "slide_path") -> None:
        super().__init__(df=self._download_dataset(uri), path_key=path_key)

    @staticmethod
    def _download_dataset(uri: str) -> pd.DataFrame:
        artifact_path = download_artifacts(artifact_uri=uri)
        return pd.read_csv(artifact_path)


class ChainedDataSources(DataSource):
    def __init__(self, sources: list[DataSource]) -> None:
        self.data_sources = sources

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(*(iter(ds) for ds in self.data_sources))

    def __len__(self) -> int:
        return sum([len(ds) for ds in self.data_sources])

    def to_pandas(self) -> pd.DataFrame:
        return pd.concat([ds.to_pandas() for ds in self.data_sources])
