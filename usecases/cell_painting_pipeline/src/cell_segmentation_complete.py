#!/usr/bin/env python3

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from PIL import Image
from scipy.ndimage import binary_dilation

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from cellSAM import segment_cellular_image, get_model

SCALED_SIZE = 1024
CELL_COUNT_SUFFIX = '_cell_count.csv'


class Segmentation:

    def __init__(self, image_path, output_dir=None):

        self.elapsed_time = 0.

        # set input/output directories
        self.input_dir = os.path.dirname(image_path)
        self.output_dir = output_dir or self.input_dir

        # get image file names
        self.ch2 = {'file': os.path.basename(image_path),
                    'path': image_path}
        self.ch3 = {'file': self.ch2['file'].replace('ch2', 'ch3')}

        image_ch3_path = os.path.join(self.input_dir, self.ch3['file'])
        if not os.path.exists(image_ch3_path):
            raise Exception(f'Matching ch3 file not found '
                            f'for {self.ch2["file"]}, skipping.')

        # load images
        self.ch2['img'] = np.array(Image.open(image_path))
        self.ch3['img'] = np.array(Image.open(image_ch3_path))

    def save_cell_count(self, cell_count):
        output_file = self.ch2['file'] + CELL_COUNT_SUFFIX
        df = pd.DataFrame({'image_path': [self.ch2['file']],
                           'cell_count': [cell_count]})
        df.to_csv(os.path.join(self.output_dir, output_file), index=False)

    def plot(self, mask, bounding_boxes):

        # Plot and save original ch2 image
        plt.figure()
        plt.imshow(self.ch2['img'], cmap='viridis')
        plt.axis('off')
        # plt.title(self.ch2['file'])
        output_file = self.ch2['file'].replace('.tiff', '_ch2_ori.png')
        plt.savefig(os.path.join(self.output_dir, output_file),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save original ch3 image
        plt.figure()
        plt.imshow(self.ch3['img'], cmap='viridis')
        plt.axis('off')
        # plt.title(self.ch3['file'])
        output_file = self.ch3['file'].replace('.tiff', '_ch3_ori.png')
        plt.savefig(os.path.join(self.output_dir, output_file),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save mask
        plt.figure()
        plt.imshow(mask, cmap='viridis')
        plt.axis('off')
        # plt.title(f'{self.ch2["file"]} nucleus masks')
        output_file = self.ch2['file'].replace('.tiff', '_ch2_nucleu_mask.png')
        plt.savefig(os.path.join(self.output_dir, output_file),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save bounding boxes
        fig, ax = plt.subplots()
        ax.imshow(self.ch2['img'])
        scale_factor = self.ch2['img'].shape[0] / SCALED_SIZE
        bounding_boxes = bounding_boxes.cpu().numpy()
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            xmin_new = xmin * scale_factor
            ymin_new = ymin * scale_factor
            xmax_new = xmax * scale_factor
            ymax_new = ymax * scale_factor
            width = xmax_new - xmin_new
            height = ymax_new - ymin_new
            ax.add_patch(patches.Rectangle((xmin_new, ymin_new), width, height,
                                           linewidth=1, edgecolor='r',
                                           facecolor='none'))
        plt.axis('off')
        # plt.title(f'{self.ch2["file"]} nucleu bounding boxes')
        output_file = self.ch2['file'].replace('.tiff', '_ch2_nucleu_bbx.png')
        plt.savefig(os.path.join(self.output_dir, output_file),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Plot and save mask for the ch3 image
        binary_mask = (mask > 0).astype(np.uint8)
        # Expand if image is RGB
        if self.ch2['img'].ndim == 3 and self.ch2['img'].shape[2] == 3:
            binary_mask = np.stack([binary_mask] * 3, axis=-1)
        masked_image_ch3 = self.ch3['img'] * binary_mask
        output_file = self.ch3['file'].replace('.tiff', '_ch3_nucleu_areas.npy')
        np.save(os.path.join(self.output_dir, output_file), masked_image_ch3)
        # Normalize masked image (NOTE: not used?)
        normalized_masked_image_ch3 = (
            (masked_image_ch3 - masked_image_ch3.min()) /
            (masked_image_ch3.max() - masked_image_ch3.min()))
        plt.figure()
        plt.imshow(masked_image_ch3, cmap='viridis')
        plt.axis('off')
        # plt.title(f'{self.ch2["file"]} nucleus areas')
        output_file = self.ch3['file'].replace('.tiff', '_ch3_nucleu_areas.png')
        plt.savefig(os.path.join(self.output_dir, output_file),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):

        print(f'Processing {self.ch2["file"]} '
              f'(image shape: {self.ch2["img"].shape})')
        # Start timing
        start_time = time.time()

        # Segment the ch2 image
        mask, embedding, bounding_boxes = segment_cellular_image(
            self.ch2['img'],
            bbox_threshold=0.35,
            normalize=True,
            device=str(device))

        # Get the number of cells detected (bounding_boxes.shape[0])
        cell_count = bounding_boxes.shape[0]
        print(f'Number of cells in {self.ch2["file"]}: {cell_count}')
        # Save the image path and cell count in a CSV file
        self.save_cell_count(cell_count)

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        if mask is None or not hasattr(mask, 'shape'):
            raise Exception(f'Invalid mask for {self.ch2["file"]}, skipping.')

        self.plot(mask, bounding_boxes)

        print(f'Finished processing {self.ch2["file"]} ' 
              f'Time: {self.elapsed_time:.2f} seconds\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        dest='image_path',
        type=str,
        required=True)
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        type=str,
        required=False)
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = get_args()
    try:
        Segmentation(image_path=args.image_path,
                     output_dir=args.output_dir).run()
    except Exception as e:
        print(e)

