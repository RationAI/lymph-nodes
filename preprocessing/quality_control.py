import asyncio
from pathlib import Path
from typing import Any

import hydra
from aiohttp import ClientSession, ClientTimeout
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger


async def put_request(
    session: ClientSession,
    url: str,
    data: dict[str, Any],
    *,
    timeout_sec: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    timeout = ClientTimeout(total=timeout_sec)
    try:
        async with semaphore, session.put(url, json=data, timeout=timeout) as response:
            text = await response.text()
            return response.status, text
    except TimeoutError:
        print(
            f"Failed to process {data.get('wsi_path', data.get('backgrounds', [''])[0])}:\n\tTimeout after {timeout_sec} seconds\n"
        )
        return -1, "Timeout"


async def repeatable_put_request(
    session: ClientSession,
    url: str,
    data: dict[str, Any],
    *,
    max_attempts: int,
    backoff_base: int,
    timeout_sec: int,
    semaphore: asyncio.Semaphore,
) -> None:
    for attempt in range(1, max_attempts + 1):
        status, text = await put_request(
            session, url, data, timeout_sec=timeout_sec, semaphore=semaphore
        )
        if status == -1 and text == "Timeout":
            return
        if status == 500 and text == "Internal Server Error":
            print(
                f"Unexpected status 500 for {data['wsi_path']} (attempt {attempt}/{max_attempts})"
            )
            await asyncio.sleep(backoff_base**attempt)
            continue
        print(
            f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )
        return
    print(f"Failed to process {data['wsi_path']} after {max_attempts} attempts\n")


async def generate_report_for_slide(
    session: ClientSession,
    slide: Path,
    slide_output_dir: Path,
    combined_report: Path,
    *,
    service_url: str,
    compute_metrics: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    url = service_url.rstrip("/") + "/report"

    data = {
        "backgrounds": [str(slide.resolve())],
        "mask_dir": str(slide_output_dir),
        "save_location": str(combined_report),
        "compute_metrics": compute_metrics,
    }

    async with semaphore, session.put(url, json=data) as response:
        result = await response.text()
        print(
            f"Report updated for {slide.name}:\n\tStatus: {response.status} \n\tResponse: {result}\n"
        )


async def process_slide(
    session: ClientSession,
    slide: Path,
    combined_report: Path,
    *,
    output_dir: Path,
    service_url: str,
    network_cfg: dict,
    qc_cfg: dict,
    semaphore: asyncio.Semaphore,
) -> None:
    slide_output_dir = output_dir / slide.stem
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    await repeatable_put_request(
        session=session,
        url=service_url,
        data={
            "wsi_path": str(slide.resolve()),
            "output_path": str(slide_output_dir),
            "mask_level": int(qc_cfg["mask_level"]),
            "sample_level": int(qc_cfg["sample_level"]),
            "check_residual": bool(qc_cfg["check_residual"]),
            "check_folding": bool(qc_cfg["check_folding"]),
            "check_focus": bool(qc_cfg["check_focus"]),
            "wb_correction": bool(qc_cfg["wb_correction"]),
        },
        max_attempts=int(network_cfg["max_retry_attempts"]),
        backoff_base=int(network_cfg["backoff_base"]),
        timeout_sec=int(network_cfg["request_timeout_sec"]),
        semaphore=semaphore,
    )

    await generate_report_for_slide(
        session,
        slide,
        slide_output_dir,
        combined_report,
        service_url=service_url,
        compute_metrics=bool(qc_cfg["compute_metrics"]),
        semaphore=semaphore,
    )


async def async_main(config: DictConfig) -> None:
    logger = MLFlowLogger(
        experiment_name=config.mlflow.experiment_name,
        run_name=config.mlflow.run_name,
    )

    output_dir = Path(config.data.output_dir)
    combined_report = Path(config.data.combined_report)
    combined_report.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(int(config.network.request_limit))
    # Build slides list dynamically: prefer explicit list, otherwise scan directory for AE positives and include extra negatives
    slides: list[Path]
    if "slides" in config.data and config.data.slides:
        slides = [Path(p) for p in config.data.slides]
    else:
        slides_dir = config.data.get("slides_dir", None)
        if slides_dir is None:
            raise ValueError(
                "No slides specified. Provide data.slides (list) or data.slides_dir (directory)."
            )
        root = Path(str(slides_dir))
        if not root.exists():
            raise FileNotFoundError(f"Slides directory not found: {root}")

        ae_positive = [
            p
            for p in sorted(root.iterdir())
            if p.is_file()
            and p.suffix.lower() == ".czi"
            and "-AE-" in p.name
            and p.name.endswith("-AE-1.czi")
        ]

        # Optional: explicitly include a few negatives by name (basename or absolute path)
        extra_negatives = list(config.data.get("extra_negatives", []))
        extra_paths: list[Path] = []
        for item in extra_negatives:
            p = Path(str(item))
            if not p.is_absolute():
                p = root / p
            if p.exists() and p.is_file():
                extra_paths.append(p)
            else:
                print(f"Warning: extra negative not found: {p}")

        slides = ae_positive + extra_paths
        if not slides:
            raise RuntimeError(
                f"No slides matched in {root}. Ensure AE positives exist and/or configure data.extra_negatives."
            )
        print(
            f"Discovered {len(ae_positive)} AE-positive slides and {len(extra_paths)} extra negatives (total {len(slides)})."
        )

    async with ClientSession() as session:
        tasks = [
            process_slide(
                session,
                slide,
                combined_report,
                output_dir=output_dir,
                service_url=config.service.url,
                network_cfg=dict(config.network),
                qc_cfg=dict(config.qc),
                semaphore=semaphore,
            )
            for slide in slides
        ]
        await asyncio.gather(*tasks)

    logger.experiment.log_artifacts(run_id=logger.run_id, local_dir=str(output_dir))



@hydra.main(config_path="./configs", config_name="quality_control", version_base=None)
def main(config: DictConfig) -> None:
    asyncio.run(async_main(config))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
