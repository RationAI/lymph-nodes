# Detection of metastases in immunohistochemically stained lymphatic tissue using deep learning methods

This repository contains the code used during development, training, and final evaluation of the models. 

## Description

The repository is divided into following directories:

- The `pipeline` directory contains the pipeline code for training, tuning, and evaluating the models. The RationAI team created a big part of this code. My contribution was adding CNN architectures of `ResNet` and `AlexNet`, and `VGG` configuration `VGG11` and `VGG19`. All models have adapted the classifier and global pooling layer to handle binary classification.

- The `preprocessing` directory contains code for tile preprocessing. I have written the code in this folder. Specifically, the code consists of 2 important classes:

    - `MaskFilter`, which calculates tile labels and filters positive tiles according to GTMCR (ground truth mask coverage ratio).
    - `SlideFilter` filters tiles from unwanted slides.

- The `metrics` directory contains code for the metrics calculation. I have written the code in this folder

- The `configs` directory contains `.yaml` files for the configuration of experiments (done by the Python framework `hydra`). The directory is further divided into `pipeline_conf`, containing pipeline experiments, and `scripts_conf`, containing script configurations. 

## Installation and usage

In the project's root directory, run the command `$ pdm install` to install all dependencies. Keep in mind that the code only works with access to datasets (see section **Note**).
The whole code consists of multiple modular parts called "components". To configure these components, I used the Python framework [hydra](https://hydra.cc). For managing experiments, training models, and the creation of datasets, I used a tool called [MlFlow](https://mlflow.org/docs/latest/index.html). For managing dependencies and code, the interface [PDM](https://pdm-project.org/en/latest/) is used. 4 commands can run the code:
- `$ pdm train <name-of-the-experiment> user=<user-to-mlflow>`: starts training process of a model
- `$ pdm eval <name-of-the-experiment> user=<user-to-mlflow>`: starts generation of heatmap masks for given dataset and model
- `$ pdm preprocessing <name-of-the-experiment> user=<user-to-mlflow>`: activates preprocessing script
- `$ pdm metrics <name-of-the-experiment> user=<user-to-mlflow>`: activates metrics calculation script
Below are provided examples of above-mentioned commands:

```
$ pdm train train_VGG16 user=<user>
$ pdm eval eval_heatmaps_VGG16 user=<user> 
$ pdm preprocessing positive/colorectal_TMA_train user=<user>
$ pdm metrics VGG16_bad user=<user>
```
## Note

Because of NDA (Non-disclosure agreement) all URIs and file system paths for specific models, datasets and MlFlow repository were removed and replaced with `???` notation. 
If the thesis reporter is interested in viewing the functional code, the reporter can contact me via the university email. The code is under the MIT license.