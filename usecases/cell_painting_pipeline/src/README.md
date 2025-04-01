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
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from cellSAM.cellsam_pipeline import cellsam_pipeline

img = np.array(Image.open("/hpcgpfs01/scratch/xyu1/cell_data/cellpaint/rpe_images/week_two/rpe_control/Compound_2/r02c05f09p02.png"))

# Image is 3-channel RGB where Channel 1 (G) represents a nuclear stain
# and Channel 2 (B) a membrane stain. Channel 0 (R) is blank.
print("Channel sums (R, G, B):", img.sum(axis=(0, 1)))

# Run the cellSAM pipeline to generate the mask
mask = cellsam_pipeline(
    img,
    low_contrast_enhancement=False,
    use_wsi=False,
    gauge_cell_size=False,
)[0]
print(mask)

# Normalize the mask to range 0 to 1
plt.figure()
plt.imshow(mask, cmap='viridis')
plt.axis('off')
plt.savefig("r02c05f09p02_mask_new_1.png", dpi=300, bbox_inches='tight')
plt.show()

# Find contours at level 0.5 on the normalized mask
contours = measure.find_contours(mask, level=0.001)
# Find contours for the mask

# Display the mask with boundaries
plt.figure(figsize=(8, 8))
plt.imshow(img)
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')

plt.axis('off')
plt.title("Segmentation Mask with Boundaries")
plt.savefig("r02c05f09p02_mask_boundary_new_4_1.png", dpi=300, bbox_inches='tight')
plt.show()
```

For more details, see `test_all.py`.


