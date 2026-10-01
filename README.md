# Swin Transformer-Based Super-Resolution of Hydraulic Flood Maps

This repository provides supporting materials for the manuscript:

**"Swin Transformer-Based Super-Resolution of Hydraulic Flood Maps for Rapid Urban Flood Hazard Assessment"**

The repository contains preprocessing and evaluation code, representative
low-resolution (LR) and high-resolution (HR) flood-depth data, and trained
SwinIR model weights for 75/150m used in the study.

## Repository Structure

### 1. `code/`

This folder contains the scripts used for preprocessing and evaluation of the
flood super-resolution datasets.

The preprocessing scripts include procedures for preparing the physical
LR-HR flood-depth data, extracting image patches, filtering patches, normalizing
water depths, and converting depth maps to RGB images using the JET colormap.

The evaluation scripts are used to calculate the depth reconstruction and flood extent metrics.

### 2. `representative_data/`

This folder contains representative samples from the physical hydrodynamic
LR-HR dataset used in the study.

The data are organized into:

- `train_HR_depth/` – Fine-resolution training depth data
- `train_HR_png/` – Fine-resolution training data converted to RGB images
- `train_LR_depth/` – Coarse-resolution training depth data
- `train_LR_png/` – Coarse-resolution training data converted to RGB images
- `Val_HR_depth/` – Fine-resolution validation/evaluation depth data
- `Val_HR_png/` – Fine-resolution validation/evaluation RGB images
- `Val_LR_depth/` – Coarse-resolution validation/evaluation depth data
- `Val_LR_png/` – Coarse-resolution validation/evaluation RGB images

The representative data are provided to demonstrate the format and preprocessing
of the LR-HR pairs used for model development and evaluation.

### 3. `trained_weights/`

This folder contains trained SwinIR model weights.

- `weights_75_150m.pth` – Trained SwinIR model for reconstructing 75 m
  flood-depth maps from 150 m coarse-resolution inputs (×2 super-resolution).

## Hydrodynamic Simulation Data

The original hydrodynamic simulation outputs are provided separately through
Zenodo.

The hydrodynamic dataset contains GeoTIFF flood-depth maps generated at:

- 75 m fine-grid resolution
- 150 m coarse-grid resolution
- 300 m coarse-grid resolution

The dataset includes four study areas:

- Greenville
- Louisburg
- Princeville–Tarboro
- Rocky Mount

The 150 m and 300 m simulations serve as the coarse-resolution inputs, while
the corresponding 75 m simulations serve as the fine-resolution reference data
for the physical super-resolution experiments.

**Zenodo dataset:** https://zenodo.org/records/23084845

## Data Splits

For the physical LR-HR experiments, data from Greenville,
Princeville–Tarboro, and Rocky Mount were used for model training.

Louisburg was excluded from model training and was used as the geographically
independent validation/evaluation location.

## Model

The super-resolution model is based on the SwinIR architecture:

Liang et al., "SwinIR: Image Restoration Using Swin Transformer."

The SwinIR implementation used in this study was based on the KAIR repository.

https://github.com/cszn/KAIR

## Citation

If you use these data, code, or trained model weights, please cite the
corresponding manuscript:

Bhattarai et al., "Swin Transformer-Based Super-Resolution of Hydraulic Flood
Maps for Rapid Urban Flood Hazard Assessment."
