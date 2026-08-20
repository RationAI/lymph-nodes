from ray.data import Dataset


class ReadTiles:
    """Reads the raw pixel tile from the slide file into a numpy HWC uint8 array."""

    def __init__(self, name: str) -> None:
        self._name = name

    def apply(self, tiles: Dataset) -> Dataset:
        from ratiopath.tiling.read_slide_tiles import read_slide_tiles
        from ray.data.expressions import col as rcol

        return tiles.with_column(
            self._name,
            read_slide_tiles(
                rcol("path"),
                rcol("tile_x"),
                rcol("tile_y"),
                rcol("tile_extent_x"),
                rcol("tile_extent_y"),
                rcol("level"),
            ),
        )