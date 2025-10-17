from pathlib import Path
import asyncio
from aiohttp import ClientSession, ClientTimeout
from typing import Any
from rationai.mlkit.lightning.loggers import MLFlowLogger

REQUEST_LIMIT = 4
REQUEST_TIMEOUT = 30 * 60
MAX_REQUEST_RETRY_ATTEMPTS = 5
BACKOFF_BASE = 2

URL = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000/"
OUTPUT_DIR = Path("/mnt/data/Projects/lymph_nodes/test")                # Replace with your desired output directory in xOpat

SLIDES = [                              # Replace with your slide paths
    Path("/mnt/data/Projects/Data/MOU/lymph_nodes/dataset2-ihc-2024/positive/SNB_IHC_CASE_141_SLIDE_4-B-1.mrxs"),
]

semaphore = asyncio.Semaphore(REQUEST_LIMIT)


async def put_request(session: ClientSession, url: str, data: dict[str, Any]) -> tuple[int, str]:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with semaphore, session.put(url, json=data, timeout=timeout) as response:
            text = await response.text()
            return response.status, text
    except asyncio.TimeoutError:
        print(f"Failed to process {data['wsi_path']}:\n\tTimeout after {REQUEST_TIMEOUT} seconds\n")
        return -1, "Timeout"


async def repeatable_put_request(session: ClientSession, url: str, data: dict[str, Any]) -> None:
    for attempt in range(1, MAX_REQUEST_RETRY_ATTEMPTS + 1):
        status, text = await put_request(session, url, data)
        if status == -1 and text == "Timeout":
            return
        if status == 500 and text == "Internal Server Error":
            print(f"Unexpected status 500 for {data['wsi_path']} (attempt {attempt}/{MAX_REQUEST_RETRY_ATTEMPTS})")
            await asyncio.sleep(BACKOFF_BASE**attempt)
            continue
        print(f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n")
        return
    print(f"Failed to process {data['wsi_path']} after {MAX_REQUEST_RETRY_ATTEMPTS} attempts\n")


async def generate_report_for_slide(session: ClientSession, slide: Path, slide_output_dir: Path, combined_report: Path) -> None:
    url = URL + "report"

    data = {
        "backgrounds": [str(slide.resolve())],
        "mask_dir": str(slide_output_dir),
        "save_location": str(combined_report),
        "compute_metrics": True,
    }

    async with semaphore, session.put(url, json=data) as response:
        result = await response.text()
        print(f"Report updated for {slide.name}:\n\tStatus: {response.status} \n\tResponse: {result}\n")


async def process_slide(session: ClientSession, slide: Path, combined_report: Path) -> None:
    slide_output_dir = OUTPUT_DIR / slide.stem
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    await repeatable_put_request(
        session=session,
        url=URL,
        data={
            "wsi_path": str(slide.resolve()),
            "output_path": str(slide_output_dir),
            "mask_level": 3,
            "sample_level": 1,
            "check_residual": True,
            "check_folding": True,
            "check_focus": True,
            "wb_correction": True,
        }
    )

    await generate_report_for_slide(session, slide, slide_output_dir, combined_report)


async def main() -> None:
    logger = MLFlowLogger(
        experiment_name="Lymph nodes",      # Replace with your experiment name
        run_name="QC combined run",         # Replace with your run name
    )

    combined_report = OUTPUT_DIR / "final_report.html"  # Replace with your desired report path
    combined_report.parent.mkdir(parents=True, exist_ok=True)

    async with ClientSession() as session:
        tasks = [process_slide(session, slide, combined_report) for slide in SLIDES]
        await asyncio.gather(*tasks)

    logger.experiment.log_artifacts(run_id=logger.run_id, local_dir=str(OUTPUT_DIR))

if __name__ == "__main__":
    asyncio.run(main())
