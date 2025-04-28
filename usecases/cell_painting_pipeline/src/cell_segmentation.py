#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import measure

from cellSAM.cellsam_pipeline import cellsam_pipeline

IMAGES_DIR = '../sample_imgs'
IMAGE_FILE_NAME = 'r02c05f09p02.png'


def boundary_overlay_image(image_path):

    # Image is 3-channel RGB, where
    #   Channel 1 (G) represents a nuclear stain,
    #   Channel 2 (B) a membrane stain,
    #   Channel 0 (R) is blank.

    img = np.array(Image.open(image_path))
    # Run the cellSAM pipeline to generate the mask
    mask = cellsam_pipeline(
        img,
        low_contrast_enhancement=False,
        use_wsi=False,
        gauge_cell_size=False)

    image_name = os.path.basename(image_path).rsplit('.', 1)[0]

    # Normalize the mask to range 0 to 1
    plt.figure()
    plt.imshow(mask, cmap='viridis')
    plt.axis('off')
    plt.title('Cell Segmentation Mask')
    plt.savefig(f'{image_name}_cell_mask.png',
                dpi=300, bbox_inches='tight')
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
    plt.title('Segmentation Mask with Boundaries')
    plt.savefig(f'{image_name}_cell_mask_boundary.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        dest='image_path',
        type=str,
        # for test purposes this option is not forced
        required=False)
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = get_args()
    boundary_overlay_image(image_path=args.image_path or
                                      f'{IMAGES_DIR}/{IMAGE_FILE_NAME}')

