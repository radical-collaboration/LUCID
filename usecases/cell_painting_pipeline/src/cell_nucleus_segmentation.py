#!/usr/bin/env python3

import argparse
import os
import sys
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from scipy.ndimage import binary_dilation

from cellSAM import segment_cellular_image, get_model

IMAGES_DIR = '../sample_imgs'
IMAGE_FILE_NAME = 'r02c05f09p02.png'


def nucleus_overlay_image(image_path):

    original_size = 1080
    scaled_size = 1024
    scale_factor = original_size / scaled_size

    # Start time recording
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = np.array(Image.open(image_path))
    # Segment the cellular image
    mask, embedding, bounding_boxes = segment_cellular_image(
        img[:, :, 0],
        bbox_threshold=0.35,
        normalize=True,
        device=str(device))

    if mask is None or not hasattr(mask, 'shape'):
        print(f'Invalid mask for sample {image_path}, skipping sample.')
        return

    bounding_boxes = bounding_boxes.cpu().numpy()

    # End time recording
    end_time = time.time()
    elapsed_time = end_time - start_time

    image_name = os.path.basename(image_path).rsplit('.', 1)[0]

    # Plot and save the mask only
    plt.figure()
    plt.imshow(mask, cmap='viridis')
    plt.axis('off')
    plt.savefig(f'{image_name}_nucleus_mask.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot and save the image with bounding boxes only
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox
        xmin_new = xmin * scale_factor
        ymin_new = ymin * scale_factor
        xmax_new = xmax * scale_factor
        ymax_new = ymax * scale_factor
        width = xmax_new - xmin_new
        height = ymax_new - ymin_new
        rect = patches.Rectangle((xmin_new, ymin_new), width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(f'{image_name}_nucleus_bbx.png', dpi=300, bbox_inches='tight')
    plt.show()

    binary_mask = (mask > 0).astype(np.uint8)

    # If the original image is in RGB, expand the binary mask to 3 channels
    if img.ndim == 3 and img.shape[2] == 3:
        binary_mask = np.stack([binary_mask] * 3, axis=-1)

    # Multiply the original image with the binary mask to get the masked image
    masked_image = img * binary_mask

    # Save the masked image
    Image.fromarray(masked_image).save(f'{image_path}_only_nucleus.png')

    print(f'Elapsed time for processing: {elapsed_time:.2f} seconds')


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
    nucleus_overlay_image(image_path=args.image_path or
                                     f'{IMAGES_DIR}/{IMAGE_FILE_NAME}')

