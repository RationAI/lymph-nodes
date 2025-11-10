from pathlib import Path
from typing import Any, Iterable
import asyncio
import tempfile

import hydra
from aiohttp import ClientSession, ClientTimeout
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


async def put_request(
    session: ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
    data: dict[str, Any],
) -> tuple[int, str]:
    timeout = ClientTimeout(total=request_timeout)

    try:
        async with semaphore, session.put(url, json=data, timeout=timeout) as response:
            result = await response.text()

            print(
                f"Processed {data['wsi_path']}:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )

            return response.status, result
    except TimeoutError:
        slide_name = Path(data["wsi_path"]).name
        print(
            f"Request to {url} timed out after {request_timeout} seconds. Slide: {slide_name}"
        )
        return -1, "Timeout"


async def repeatable_put_request(
    session: ClientSession,
    url: str,
    data: dict[str, Any],
    num_repeats: int,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
) -> None:
    for attempt in range(1, num_repeats + 1):
        status, text = await put_request(session, url, semaphore, request_timeout, data)

        if status == -1 and text == "Timeout":
            return

        if status == 500 and text == "Internal Server Error":
            att_count = f"attempt {attempt}/{num_repeats}"
            print(
                f"Unexpected status 500 received for {data['wsi_path']} ({att_count}):\n\tResponse: {text}\n"
            )
            await asyncio.sleep(2**attempt)
            continue

        print(
            f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )
        return

    print(f"Failed to process {data['wsi_path']}:\n\tAll retry attempts failed\n")


async def generate_report(
    session: ClientSession,
    report_request_timeout: int,
    slides: list[Path],
    output_dir: str,
    save_location: str,
    url: str,
    semaphore: asyncio.Semaphore,
) -> None:
    url = url.rstrip("/") + "/report"

    data = {
        "backgrounds": [str(slide) for slide in slides],
        "mask_dir": output_dir,
        "save_location": save_location,
    }

    try:
        async with (
            semaphore,
            session.put(
                url, json=data, timeout=ClientTimeout(total=report_request_timeout)
            ) as response,
        ):
            result = await response.text()
            print(
                f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )
    except TimeoutError:
        print(
            f"Report generation request to {url} timed out after {report_request_timeout} seconds."
        )


async def qc_main(config: DictConfig, logger: MLFlowLogger) -> None:
    # Resolve slides: prefer explicit list; else discover from slides_dir
    slides_cfg = config.get("slides") or []
    slides: list[Path] = [Path(s) for s in slides_cfg] if slides_cfg else []

    if not slides and config.get("slides_dir"):
        base = Path(str(config.slides_dir)).expanduser()
        pattern = str(config.get("slides_glob", "**/*.czi"))
        if "**" in pattern:
            slides = sorted(base.rglob(pattern))
        else:
            slides = sorted(base.glob(pattern))

    if not slides:
        raise ValueError(
            "No slides provided. Set +slides=[/abs/one.czi,...] or slides_dir=/path/to/dir in config."
        )

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(int(config.request_limit))
    timeout_sec = int(config.request_timeout)

    async with ClientSession() as session:
        # submit per-slide QC
        tasks = [
            repeatable_put_request(
                session=session,
                url=config.url,
                data={
                    "wsi_path": str(slide.resolve()),
                    "output_path": str(output_path / slide.stem),
                    "mask_level": int(config.mask_level),
                    "sample_level": int(config.sample_level),
                    "check_residual": bool(config.get("check_residual", True)),
                    "check_folding": bool(config.get("check_folding", True)),
                    "check_focus": bool(config.get("check_focus", True)),
                },
                num_repeats=int(config.num_repeats),
                semaphore=semaphore,
                request_timeout=timeout_sec,
            )
            for slide in slides
        ]
        await asyncio.gather(*tasks)

        with tempfile.TemporaryDirectory(
            prefix="qc_masks_report_", dir=output_path.as_posix()
        ) as tmp_dir:
            report_path = Path(tmp_dir, "report.html")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            await generate_report(
                session=session,
                report_request_timeout=int(config.report_request_timeout),
                slides=slides,
                output_dir=output_path.as_posix(),
                save_location=report_path.as_posix(),
                url=config.url,
                semaphore=semaphore,
            )

            logger.log_artifacts(local_dir=report_path.parent.as_posix())


@hydra.main(config_path="../configs", config_name="preprocessing/qc", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    asyncio.run(qc_main(config, logger))

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
