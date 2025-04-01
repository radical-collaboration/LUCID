# Cell Painting image Analysis with Different dose rate

## Description
This repository provides code for cell painting image analysis, we first segment the cell and then compute some measerurement based on the segmentation resutls.

### Cell boundary or nucli segmentation with CellSAM
CellSAM is described in more detail in the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3), and is publicly deployed at [cellsam.deepcell.org](https://cellsam.deepcell.org/). CellSAM achieves state-of-the-art performance on segmentation across a variety of cellular targets (bacteria, tissue, yeast, cell culture, etc.) and imaging modalities (brightfield, fluorescence, phase, etc.). Feel free to [reach out](mailto:ulisrael@caltech.edu) for support/questions! The full dataset used to train CellSAM is available [here](https://storage.googleapis.com/cellsam-data/dataset.tar.gz).

### Getting started
The easiest way to get started with CellSAM is with pip
`pip install git+https://github.com/vanvalenlab/cellSAM.git`

CellSAM requires `python>=3.10`, but otherwise uses pure PyTorch. A sample image is included in this repository. Segmentation can be performed as follows

```
import numpy as np
from cellSAM import segment_cellular_image
img = np.load("sample_imgs/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')
```

For more details, see `cellsam_introduction.ipynb`.

### Calcualte the measerurement

