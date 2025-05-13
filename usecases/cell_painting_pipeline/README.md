# Cell Painting Image Analysis with Different Dose Rate

This repository gives an overview and provides code for cell painting image 
segmentation.

## 1. Overview

The corresponding pipeline performs cell painting image analysis with two
primary functions:  
- Cell segmentation: generates a segmentation mask and overlays cell boundaries
  on the original image.
- Cell nucleus segmentation: identifies nuclei via segmentation, extracts 
  bounding boxes, and creates masked images of nuclei.

### 1.1. Getting started

The easiest way to get started with CellSAM is with pip
`pip install git+https://github.com/vanvalenlab/cellSAM.git`
(see section [3](#3-target-hpc-platforms) for the environment setup per 
a corresponding HPC platform)

CellSAM requires `python>=3.10`, but otherwise uses pure PyTorch. 
A sample image is included in this repository. 

### 1.2. Cell segmentation

See example python script [`cell_segmentation.py`](src/cell_segmentation.py)

<p align="center">
  <img alt="Cell Segmentation Mask" src="./sample_imgs/r02c05f09p02_cell_mask.png" width="45%">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img alt="Cell Boundary Mask" src="./sample_imgs/r02c05f09p02_cell_boundary.png" width="45%">
</p>

### 1.3. Cell nucleus segmentation

See example python script [`cell_nucleus_segmentation.py`](src/cell_nucleus_segmentation.py)

<p align="center">
  <img alt="Cell Nucleus Box" src="./sample_imgs/r02c05f09p02_nucleus_bbx.png" width="45%">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img alt="Cell Nucleus Mask" src="./sample_imgs/r02c05f09p02_nucleus_mask.png" width="45%">
</p>

For more details for batch inference, see [`test_all.py`](src/test_all.py).

## 2. With RADICAL-Pilot

Prepare conda/virtual environment first and then install and use 
RADICAL-Pilot in it.

Installation of [RADICAL](https://github.com/radical-cybertools) tool(s)
```shell
# activate virtual/conda environment (see section 3)
pip install radical.pilot
```

Execute examples above (sections [1.2](#12-cell-segmentation) and 
[1.3](#13-cell-nucleus-segmentation)) wrapped into RADICAL-Pilot application,
which manages these examples as computing tasks applied to a particular image.
Prototype script is located in the [wfms/](wfms) directory.

```shell
# 1. launch batch or interactive job (see section 3)
# 2. activate virtual/conda environment

cd LUCID/usecases/cell_painting_pipeline/wfms
python3 cell.rp.py  # possible options: --work_dir, --images_dir
```

## 3. Target HPC platforms

### 3.1. Polaris (ALCF/ANL)

Create virtual environment
```shell
export PYTHONNOUSERSITE=True
module use /soft/modulefiles; module load conda
conda create -y -n ve.cellsam python=3.10
conda activate ve.cellsam
```

Install tools
```shell
pip install matplotlib
pip install 'torch==2.6.0' torchvision
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

Get this repository with examples
```shell
git clone https://github.com/radical-collaboration/LUCID.git
```

Run an interactive job
```shell
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=00:30:00 \
     -q debug -A <PROJECT_NAME>
```

Execute examples above (sections [1.2](#12-cell-segmentation) and 
[1.3](#13-cell-nucleus-segmentation)) as `python3 cell_segmentation.py` and 
`python3 cell_nucleus_segmentation.py` respectively, or RADICAL-Pilot 
application (section [2](#2-with-radical-pilot)) as `python3 cell.rp.py`

### 3.2. IC2 (SDCC/BNL)

... TBD ...

