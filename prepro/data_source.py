import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path


class DataSource(Iterable[Path]):
    def __init__(
        self,
        src_dir: str,
        glob_pattern: str | list[str] = "*",
        exclue_pattern: str | list[str] = "*",
    ) -> None:
        dir = Path(src_dir)

        self.exclude_paths = list(self._paths(dir, exclue_pattern))
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


class ChainedDataSources(DataSource):
    def __init__(self, sources: list[DataSource]) -> None:
        self.data_sources = sources

    def __iter__(self) -> Iterator[Path]:
        return itertools.chain(*(iter(ds) for ds in self.data_sources))

    def __len__(self) -> int:
        return sum([len(ds) for ds in self.data_sources])
