# from __future__ import annotations

# import re
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path
# from typing import cast

# import pyarrow as pa
# import pyarrow.parquet as pq
# import torch
# import torch.nn.functional as F
# from datasets import Dataset as HFDataset
# from datasets import load_dataset
# from datasets.table import InMemoryTable
# from mlflow.artifacts import download_artifacts
# from torch.utils.data import Dataset

# from lymph_nodes.typing import MetadataTileEmbeddings, TileEmbeddingsSample


# def _group_from_stem(stem: str) -> str:
#     mmci = re.match(r"^SNB_[A-Z]+_CASE_(\d+)_SLIDE_", stem)
#     if mmci:
#         return f"case_{mmci.group(1)}"
#     fnb = re.match(r"^FNB-?P?(\d+)-(\d+)-", stem)
#     if fnb:
#         return f"{fnb.group(1)}-{fnb.group(2)}"
#     return stem


# def _label_from_filename(stem: str) -> int:
#     if stem.endswith("-1"):
#         return 1
#     if stem.endswith("-0"):
#         return 0
#     return 1


# class TileEmbeddings(Dataset[TileEmbeddingsSample]):
#     def __init__(
#         self,
#         paths: list[str] | None = None,
#         uris: list[str] | None = None,
#         padding: bool = True,
#     ) -> None:
#         self.padding = padding
#         paths = paths or []
#         uris = uris or []

#         with ThreadPoolExecutor() as executor:
#             artifact_paths = list(
#                 executor.map(lambda u: download_artifacts(artifact_uri=u), uris)
#             )
#         all_dirs = [Path(p) for p in (*paths, *artifact_paths)]

#         slide_files = [
#             p / "slides.parquet" for p in all_dirs if (p / "slides.parquet").exists()
#         ]
#         tile_files = [
#             str(p / "tiles.parquet") for p in all_dirs if (p / "tiles.parquet").exists()
#         ]

#         if not slide_files or not tile_files:
#             raise FileNotFoundError(
#                 f"No slides.parquet/tiles.parquet found in: {all_dirs}"
#             )

#         slides_table = pa.concat_tables(
#             [pq.read_table(str(f)) for f in slide_files],
#             promote_options="default",
#         )
#         self._slides_hf: HFDataset = HFDataset(InMemoryTable(slides_table))

#         self._tiles_ds: HFDataset = cast(
#             "HFDataset",
#             load_dataset(
#                 "parquet",
#                 data_files=tile_files,
#                 split="train",
#                 columns=["x", "y", "embedding", "slide_id"],
#             ),
#         )

#         index_map: dict[str, list[int]] = defaultdict(list)
#         for idx, sid in enumerate(self._tiles_ds["slide_id"]):
#             index_map[sid].append(idx)
#         self._index_map = index_map

#         self.slides = [
#             {
#                 "name": row["id"],
#                 "label": _label_from_filename(row["id"]),
#                 "group": _group_from_stem(row["id"]),
#             }
#             for row in self._slides_hf
#             if row["id"] in index_map
#         ]
#         self.labels = [s["label"] for s in self.slides]
#         self.groups = [s["group"] for s in self.slides]

#         if self.padding:
#             self.max_tiles = max(len(index_map[s["name"]]) for s in self.slides)

#     def __len__(self) -> int:
#         return len(self.slides)

#     def __getitem__(self, idx: int) -> TileEmbeddingsSample:
#         slide = self.slides[idx]
#         indices = self._index_map[slide["name"]]
#         rows = self._tiles_ds.select(indices)

#         embeddings = torch.tensor(rows["embedding"], dtype=torch.float32)
#         xs = torch.tensor(rows["x"], dtype=torch.int32)
#         ys = torch.tensor(rows["y"], dtype=torch.int32)

#         if self.padding:
#             pad_amount = self.max_tiles - embeddings.shape[0]
#             if pad_amount > 0:
#                 embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

#         label = torch.tensor(slide["label"], dtype=torch.float32)
#         metadata = MetadataTileEmbeddings(
#             slide_id=slide["name"],
#             slide_name=slide["name"],
#             slide_path=Path(slide["name"]),
#             tiles=None,
#             x=xs,
#             y=ys,
#         )
#         return embeddings, label, metadata
