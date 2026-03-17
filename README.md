# Machine Learning Template

This project provides a machine learning quickstart template using PyTorch Lightning, Hydra, MLflow, and uv.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd machine-learning
    ```
2.  **Install uv:**
    Follow the instructions on the [uv website](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it installed.
3.  **Install dependencies:**
    This command creates a virtual environment and installs all necessary packages defined in [`pyproject.toml`](pyproject.toml).
    ```bash
    uv sync
    ```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management.

-   Configuration files are located in the [`configs/`](configs) directory.
-   The main configuration file is [`configs/lymph_nodes.yaml`](configs/default.yaml).
-   You can override configuration parameters directly from the command line. For example, to change the batch size:
    ```bash
    uv run python +m <lymph_nodes> mode=fit data.batch_size=64
    ```
-   MLflow is configured as the default logger (see [`configs/default.yaml`](configs/default.yaml)). Ensure your MLflow tracking server is running or configure it accordingly.

## Usage

-   **Train the model:**
    ```bash
    uv run python +m <lymph_nodes> mode=fit
    ```

-   **Validate the model:**
    Requires a checkpoint path to be set in the configuration (e.g., `checkpoint=path/to/your/checkpoint.ckpt`) or passed via the command line.
    ```bash
    uv run python +m <lymph_nodes> mode=validate checkpoint=path/to/checkpoint.ckpt
    ```

-   **Test the model:**
    Requires a checkpoint path.
    ```bash
    uv run python +m <lymph_nodes> mode=test checkpoint=path/to/checkpoint.ckpt
    ```

-   **Run prediction:**
    Requires a checkpoint path.
    ```bash
    uv run python +m <lymph_nodes> mode=predict checkpoint=path/to/checkpoint.ckpt
    ```

## Preprocessing: tissue_mask_generation

`tissue_mask_generation` creates binary tissue masks from whole-slide images (WSI) and logs them as artifacts.

-   Input slide is read at the closest level for configured `mpp`.
-   Tissue candidate regions are detected in HSV space using value/saturation thresholds.
-   Morphological closing and opening are applied to remove small holes/noise.
-   Output masks are saved as `.tiff` files and logged under the configured `artifact_path` (default: `tissue_masks`).

Run:

```bash
uv run python preprocessing/tissue_masks.py
```

Key config options are in [`configs/preprocessing/tissue_masks.yaml`](configs/preprocessing/tissue_masks.yaml):

-   `mpp`: target resolution used for selecting slide level.
-   `max_concurrent`: number of slides processed in parallel.
-   `artifact_path`: MLflow artifact destination.


## Linting, Formatting and Type Checking:

```bash
uvx ruff check  # Check and fix linting issues
uvx ruff format # Format code
uvx mypy .     # Run mypy
```