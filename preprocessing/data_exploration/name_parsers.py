from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class ParsedFilename:
    case_id: str
    slice_id: str
    staining: str
    tumor: bool



def parse_mmci_filename(filename: str) -> ParsedFilename:
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


def parse_fnb_filename(filename: str) -> ParsedFilename:
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