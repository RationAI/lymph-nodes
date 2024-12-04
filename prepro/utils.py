from pathlib import Path

import pyvips


def get_relative_dir_path(path: Path, prefix: Path) -> Path:
    return path.relative_to(prefix).parent


def mpp_to_ppmm(mpp: tuple[float, float]) -> tuple[float, float]:
    """Convert microns per pixel to pixels per millimeter."""
    return 1000 / mpp[0], 1000 / mpp[1]


def vips_read(path: Path, level: int) -> pyvips.Image:
    extenstion = path.suffix
    kwargs = {}

    if extenstion == ".mrxs":
        kwargs = {"level": level}
    elif extenstion == ".tiff":
        kwargs = {"page": level}

    return pyvips.Image.new_from_file(str(path), **kwargs)  # type: ignore [unused-ignore]
