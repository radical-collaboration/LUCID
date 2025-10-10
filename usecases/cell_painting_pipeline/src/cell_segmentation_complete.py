#!/usr/bin/env python3

import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from typing import Optional

from PIL import Image
from scipy.ndimage import binary_dilation, label

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from cellSAM import segment_cellular_image, get_model, get_local_model

SUPPORTED_IMAGE_EXT = ('.png', '.tif', '.tiff')

SCALED_SIZE = 1024
CELL_COUNT_SUFFIX = '_cell_count.csv'


class Segmentation:

    def __init__(self, base_channel: str, target_channel: str,
                 image_path: str, output_dir: Optional[str] = None,
                 model_path: Optional[str] = None,
                 save_bbox: bool = False, bbox_threshold: float = 0.35):

        self.elapsed_time = 0.

        self.save_bbox = save_bbox
        self.bbox_threshold = bbox_threshold

        # set input/output directories
        self.input_dir = os.path.dirname(image_path)
        self.output_dir = output_dir or self.input_dir

        # get image file names
        self.ch_base = {'name': base_channel,
                        'file': os.path.basename(image_path),
                        'path': image_path}
        self.ch_tgt = {'name': target_channel,
                       'file': self.ch_base['file'].replace(base_channel,
                                                            target_channel)}
        self.ch_tgt['path'] = os.path.join(self.input_dir, self.ch_tgt['file'])

        if not os.path.exists(self.ch_tgt['path']):
            raise Exception(f'Matching target channel {target_channel} file '
                            f'not found for {self.ch_base["file"]}, skipping.')

        self.ch_base['file_stub'] = os.path.splitext(self.ch_base['file'])[0]
        self.ch_tgt['file_stub'] = os.path.splitext(self.ch_tgt['file'])[0]

        # load images
        self.ch_base['img'] = np.array(Image.open(self.ch_base['path']))
        self.ch_tgt['img'] = np.array(Image.open(self.ch_tgt['path']))

        # get model
        self.model = get_local_model(model_path) if model_path else get_model()
        self.model.to(str(device))

    def save_cell_count(self, cell_count):
        output_file = self.ch_base['file_stub'] + CELL_COUNT_SUFFIX
        df = pd.DataFrame({'image_path': [self.ch_base['file']],
                           'cell_count': [cell_count]})
        df.to_csv(os.path.join(self.output_dir, output_file), index=False)

    def plot_bbox(self, bounding_boxes):
        # plot and save bounding boxes
        fig, ax = plt.subplots()
        ax.imshow(self.ch_base['img'])
        scale_factor = self.ch_base['img'].shape[0] / SCALED_SIZE
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            xmin_new, ymin_new = xmin * scale_factor, ymin * scale_factor
            width = (xmax - xmin) * scale_factor
            height = (ymax - ymin) * scale_factor
            ax.add_patch(patches.Rectangle((xmin_new, ymin_new), width, height,
                                           linewidth=1, edgecolor='r',
                                           facecolor='none'))
        ax.axis('off')
        file_name = '%(file_stub)s_%(name)s_nucleu_bbx.png' % self.ch_base
        fig.savefig(os.path.join(self.output_dir, file_name),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot(self, mask):

        # plot and save original base-/target-channel image
        for ch in [self.ch_base, self.ch_tgt]:
            file_name = '%(file_stub)s_%(name)s_ori.png' % ch
            plt.imsave(os.path.join(self.output_dir, file_name),
                       ch['img'], cmap='viridis')

        file_name = '%(file_stub)s_%(name)s_nucleu_mask.png' % self.ch_base
        plt.imsave(os.path.join(self.output_dir, file_name),
                   mask, cmap='viridis')

        instance_mask, num_instances = label(mask)
        print(f'Identified {num_instances} instances in the mask.')
        file_name = '%(file_stub)s_%(name)s_instance_mask.npy' % self.ch_base
        np.save(os.path.join(self.output_dir, file_name), instance_mask)

        # apply binary mask to target channel
        binary_mask = (mask > 0).astype(np.uint8)
        # expand if image is RGB
        if self.ch_base['img'].ndim == 3 and self.ch_base['img'].shape[2] == 3:
            binary_mask = np.stack([binary_mask] * 3, axis=-1)

        masked_image = self.ch_tgt['img'] * binary_mask
        file_name = '%(file_stub)s_%(name)s_masked.npy' % self.ch_tgt
        np.save(os.path.join(self.output_dir, file_name), masked_image)
        file_name = '%(file_stub)s_%(name)s_masked.png' % self.ch_tgt
        plt.imsave(os.path.join(self.output_dir, file_name),
                   masked_image, cmap='viridis')

    def run(self):

        print(f'Processing {self.ch_base["file"]} '
              f'(image shape: {self.ch_base["img"].shape})')
        # Start timing
        start_time = time.time()

        # Segment the base-channel image
        mask, embedding, bounding_boxes = segment_cellular_image(
            img=self.ch_base['img'],
            model=self.model,
            bbox_threshold=self.bbox_threshold,
            normalize=True,
            device=str(device))

        if mask is None or not hasattr(mask, 'shape'):
            raise Exception(f'Invalid mask for {self.ch_base["file"]}, '
                            f'skipping.')

        if self.save_bbox:
            if hasattr(bounding_boxes, 'cpu'):
                bounding_boxes = bounding_boxes.cpu().numpy()
            # get the number of cells detected (bounding_boxes.shape[0])
            cell_count = bounding_boxes.shape[0]
            print(f'Number of cells in {self.ch_base["file"]}: {cell_count}')
            # save the image path and cell count in a CSV file
            self.save_cell_count(cell_count)
            self.plot_bbox(bounding_boxes)

        self.plot(mask)

        end_time = time.time()
        self.elapsed_time = end_time - start_time
        print(f'Finished processing {self.ch_base["file"]} ' 
              f'Time: {self.elapsed_time:.2f} seconds\n')


def get_args():
    parser = argparse.ArgumentParser(
        description='Process cellular images with segmentation.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image_path',
        type=str,
        help='Path to the base/target channel image.')
    group.add_argument(
        '--input_dir',
        type=str,
        help='Path to the input directory with base/target channel images.')

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory to save results.')
    parser.add_argument(
        '--base_channel',
        type=str,
        required=True,
        help='Channel used for nucleus segmentation (e.g., "ch2").')
    parser.add_argument(
        '--target_channel',
        type=str,
        required=True,
        help='Channel to apply the segmentation mask to (e.g., "ch8").')
    parser.add_argument(
        '--save_bbox',
        action='store_true',
        help='Save bounding box visualizations and cell count CSV.')
    parser.add_argument(
        '--bbox_threshold',
        type=float,
        default=0.35,
        help='Threshold for selecting bounding boxes in segmentation.')
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='Path to the local cellSAM model.')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = get_args()

    image_path_list = []
    if args.image_path:
        assert args.image_path.endswith(SUPPORTED_IMAGE_EXT)
        image_path_list.append(args.image_path)
    elif args.input_dir:
        for image_path in glob.glob(f'{args.input_dir}/*'):
            f = os.path.basename(image_path).lower()
            if args.base_channel in f and f.endswith(SUPPORTED_IMAGE_EXT):
                image_path_list.append(image_path)

    for image_path in image_path_list:
        try:
            Segmentation(base_channel=args.base_channel,
                         target_channel=args.target_channel,
                         image_path=image_path,
                         output_dir=args.output_dir,
                         model_path=args.model_path,
                         save_bbox=args.save_bbox,
                         bbox_threshold=args.bbox_threshold).run()
        except Exception as e:
            print(e)

