from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ParsedFilename:
    case_id: str
    slice_id: str
    staining: str
    tumor: bool | None



def mmci_snb_filename(filename: str) -> ParsedFilename:
    # Regex pattern breakdown:
    # ^SNB_: Starts with 'SNB_'
    # ([A-Z]+): Group 1 (Staining - one or more capital letters)
    # _CASE_: Literal string '_CASE_'
    # (\d+): Group 2 (Case ID - one or more digits)
    # _SLIDE_: Literal string '_SLIDE_'
    # ([A-Z0-9-]+): Group 3 (Slice ID - one or more capital letters, digits, or hyphens)
    # -(0|1): Group 4 (Tumor Indicator - '0' or '1')
    # \.mrxs$: Matches the literal '.mrxs' extension at the end
    pattern = r"^SNB_([A-Z]+)_CASE_(\d+)_SLIDE_([A-Z0-9-]+)-(0|1)\.mrxs$"

    match = re.match(pattern, Path(filename).name)

    if match:
        # Extract the captured groups
        staining, case_id, slice_id, tumor_indicator = match.groups()

        return ParsedFilename(
            case_id=case_id,
            slice_id=slice_id,
            staining=staining,
            tumor=tumor_indicator == "1"
        )
    else:
        raise ValueError(f"Filename does not match expected MMCI pattern: {filename}")


def mmci_tmas_filename(filename: str) -> ParsedFilename:
    """Parses MMCI tissue microarray (TMA) slide filenames.

    configs/data/raw/mmci_tmas.yaml chains three raw source directories together,
    each with its own naming convention (all three confirmed against real slide
    listings):

    - breast/tissue_microarray/dab:       FIN-<block>-<marker>.mrxs
      e.g. FIN-HR2-19-ERA.mrxs, FIN-HR2-19-PGRA.mrxs. <marker> is an IHC marker
      (ERA = estrogen receptor alpha, PGRA = progesterone receptor A, ...).
    - breast/tissue_microarray_TNBC/ckae: TNBC-BF-<n>-PNG.mrxs
    - colorectum/tissue_microarray/dab:   KOS<nn>.mrxs

    Neither the TNBC nor the colorectal pattern has a stain marker visible in the
    slide name itself, so `staining` is set to "CK" (cytokeratin) from directory
    context, not parsed.

    TMAs are composite arrays of many patients' cores per slide — there's no single
    per-slide tumor/non-tumor label the way there is for mmci_snb_filename's
    whole-slide SNB biopsies, so `tumor` is always False here (not a real signal).
    """
    patterns: list[tuple[str, Callable[[re.Match[str]], ParsedFilename]]] = [
        (
            r"^FIN-([A-Z0-9]+-\d+)-([A-Z]+)\.mrxs$",
            lambda m: ParsedFilename(
                case_id="FIN-" + m.group(1) + '-' + m.group(2), slice_id="FIN", staining=m.group(2), tumor=None
            ),
        ),
        (
            r"^TNBC-(BF-\d+)-PNG\.mrxs$",
            lambda m: ParsedFilename(
                case_id="TNBC-" + m.group(1), slice_id="0", staining="CK", tumor=None
            ),
        ),
        (
            r"^(KOS\d+)\.mrxs$",
            lambda m: ParsedFilename(
                case_id=m.group(1), slice_id="0", staining="CK", tumor=None
            ),
        ),
    ]

    name = Path(filename).name
    for pattern, build in patterns:
        match = re.match(pattern, name)
        if match:
            return build(match)

    raise ValueError(f"Filename does not match any known MMCI TMA pattern: {filename}")


def mmci_snb_test_filename(filename: str) -> ParsedFilename:
    """Parses MMCI SNB test/inference cohort slide filenames.

    e.g. SNB_IHC_TEST_CASE-2024_0960-20.mrxs, SNB_IHC_TEST_CASE-2024_1001-19-1.mrxs
    (configs/data/raw/mmci_ihc_hadb_test.yaml, src_dir .../annotated_ihc_test).

    Unlike mmci_snb_filename's labeled cohorts, none of these carry a trailing 0/1
    tumor indicator — this is the unlabeled test/inference cohort, so `tumor` is
    always False here (not a real signal; ground truth for this cohort lives in the
    annotation masks, not the filename). The optional third numeric group (the "-1"
    in the second example above) is folded into `slice_id` alongside the slide
    number, as mmci_snb_filename's slice_id does for its own multi-part slide IDs
    (e.g. "14-15", "6-8-B").
    """
    pattern = r"^SNB_IHC_TEST_CASE-(\d+)_(\d+)-(\d+)(?:-(\d+))?\.mrxs$"

    match = re.match(pattern, Path(filename).name)

    if match:
        year, case_id, slide_id, extra = match.groups()
        slice_id = f"{slide_id}-{extra}" if extra else slide_id

        return ParsedFilename(
            case_id=f"{case_id}-{year}",
            slice_id=slice_id,
            staining="IHC",
            tumor=True,
        )
    else:
        raise ValueError(f"Filename does not match expected MMCI SNB test pattern: {filename}")


def fnb_filename(filename: str) -> ParsedFilename:
    # Regex pattern breakdown:
    # ^FNB-?: Starts with 'FNB' optionally followed by a hyphen.
    # P?: Optional 'P' prefix.
    # (\d+): Group 1 (Patient ID Number - one or more digits).
    # -(\d+): Group 2 (Year)
    # -(\d+): Group 3 (Slide ID 1)
    # -(\d+): Group 4 (Slide ID 2)
    # -([A-Z]+): Group 5 (Staining - one or more capital letters)
    # -(0|1): Group 6 (Tumor Indicator - '0' or '1')
    # \.czi$: Matches the literal '.czi' extension at the end
    pattern = r"^FNB-?P?(\d+)-(\d+)-(\d+)-(\d+)-([A-Z]+)-(0|1)\.czi$"

    match = re.match(pattern, Path(filename).name)

    if match:
        # Extract the captured groups. Group 1 (case_id) is now only digits.
        case_id, year, slice_1, slice_2, staining, tumor_indicator = match.groups()

        return ParsedFilename(
            case_id=case_id + "-" + year,
            slice_id=slice_1 + "-" + slice_2,
            staining=staining,
            tumor=tumor_indicator == "1"
        )
    else:
        raise ValueError(f"Filename does not match expected FNB pattern: {filename}")