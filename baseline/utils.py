from pathlib import Path

from openslide import OpenSlide


def save_thumbnail(slide_path: str | Path, thumbnail_path: str | Path) -> None:
    with OpenSlide(slide_path) as slide:
        thumbnail = slide.get_thumbnail((1024, 1024))
        thumbnail.save(thumbnail_path)
