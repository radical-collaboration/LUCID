# Cell Painting image Analysis with Different dose rate

## Description
This repository provides code for cell painting image segmentation.

### Cell boundary or nucleus segmentation with CellSAM

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


