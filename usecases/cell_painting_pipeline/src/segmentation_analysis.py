#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import pandas as pd

from collections import defaultdict

from torch.utils.data import Dataset

# Example transform using torchvision
# from torchvision import transforms
# transform = transforms.Compose([transforms.Resize((256, 256)),
#                                 transforms.ToTensor()])

from matplotlib import pyplot as plt
from PIL import Image

# example: r01c01f01p01-ch3sk1fk1fl1_ch3_nucleu_areas.png
TGT_SUFFIX = '_ch3_nucleu_areas.png'
# example: r01c01f01p01-ch2sk1fk1fl1_ch2_nucleu_mask.png
SEG_SUFFIX = '_ch2_nucleu_mask.png'
# example: r01c01f01p01-ch2sk1fk1fl1.tiff_cell_count.csv
CSV_SUFFIX = '.tiff_cell_count.csv'
FEAT_TYPE_VALUES = ['pixel_avg', 'depth_maximize_pixel_avg', 'rpe_images']


class CellPaintDataset(Dataset):

    def __init__(self,
                 base_dir: str,
                 feat_type: str = None,
                 transform=None):
        """
        Args:
            base_dir (str): Root path to the cell paint data
            feat_type (str, optional): Feature type. Defaults to 'pixel_avg'
            transform (callable, optional): Transform to be applied on the image
        """
        self.base_dir = base_dir
        self.feat_type = feat_type or FEAT_TYPE_VALUES[0]
        assert self.feat_type in FEAT_TYPE_VALUES, 'Unknown feature type'
        self.transform = transform

        # store each record as a dictionary ("run_name" points to the record)
        self.data = {}
        self.samples = defaultdict(dict)

        # Walk through the directory structure and gather file paths
        for curr_dir, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(TGT_SUFFIX):
                    continue
                ch3_name = os.path.splitext(file)[0]
                ch2_name_base = ch3_name.replace('ch3', 'ch2')[:25]
                # Create a record dictionary
                r = {
                    'run_name': ch3_name[:12],
                    'feat_image_path': f'{curr_dir}/{ch3_name}.npy',
                    'seg_image_path': f'{curr_dir}/{ch2_name_base}{SEG_SUFFIX}',
                    'csv_path': f'{curr_dir}/{ch2_name_base}{CSV_SUFFIX}'
                }
                # Load the CSV file to get the cell count
                # (assume the count is in the first row)
                try:
                    df = pd.read_csv(r['csv_path'])
                    r['cell_count'] = df['cell_count'][0]
                except Exception as e:
                    print(f'Error reading CSV file {r["csv_path"]}: {e}')
                    continue

                self.data[r['run_name']] = r  # old name "image_list"
                self.samples[r['run_name'][:9]].\
                    setdefault('list', []).append(r['run_name'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[self._get_run_name(idx)]

    def _get_run_name(self, idx):
        return sorted(self.data.keys())[idx] if idx is not None else None

    def feat_func(self, idx=None, run_name=None, feat_type=None):

        run_name = run_name or self._get_run_name(idx)

        if feat_type == 'pixel_avg':
            record = self.data[run_name]
            img_array = np.load(record['feat_image_path'])
            feature = img_array[img_array > 0].mean()

        elif feat_type == 'depth_maximize_pixel_avg':
            img_list = []
            for rn in self.samples[run_name[:9]]['list']:
                img_list.append(np.load(self.data[rn]['feat_image_path']))

            max_image = np.max(np.stack(img_list), axis=0)
            feature = max_image[max_image > 0].mean()

        else:
            raise NotImplementedError()
        return feature

    def set_sample_feature(self):
        feat_type = 'depth_maximize_pixel_avg'
        for sample_key, sample in self.samples.items():
            sample['feature'] = self.feat_func(run_name=sample_key,
                                               feat_type=feat_type)


class Analysis:

    def __init__(self, base_dir, feat_type, transform=None):
        self.dataset = CellPaintDataset(base_dir=base_dir,
                                        feat_type=feat_type,
                                        transform=transform)
        self.dataset.set_sample_feature()

    def plot(self):

        sample_list = self.dataset.samples
        image_list = self.dataset.data

        # Initialize a dictionary to store the means of the images for each r
        image_means_by_r = {}
        c = '06'
        # Iterate through the values of r
        for r in ['01', '02', '03', '04', '05', '06', '07', '08']:
            # Initialize a list to store the means for the current r
            image_means = []
            # Iterate through the values of f
            for f in range(1, 10):  # f takes values from '01' to '09'
                # Iterate through the images
                for p in ['01']:  # Assuming there are 5 images (p01 to p05)
                    # Load the TIFF image
                    sample_name = f'r{r}c{c}f{f:02d}'
                    if sample_name in sample_list:
                        feature = sample_list[sample_name]['feature']
                        image_means.append(feature)
                    else:
                        # print('no', run_name)
                        pass

            # Store the means for the current r
            image_means_by_r[r] = image_means

        # Calculate the mean and standard deviation for each r
        mean_std_by_r = {r: (np.mean(means), np.std(means)) for r, means in
                         image_means_by_r.items()}

        # Plot the mean and standard deviation for each r
        labels = list(mean_std_by_r.keys())

        means = [mean_std_by_r[r][0] for r in labels]
        stds = [mean_std_by_r[r][1] for r in labels]
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        x = range(len(labels))  # X-axis positions

        plt.figure(figsize=(8, 6))
        plt.bar(x, means, yerr=stds, capsize=5,
                color=['blue', 'orange', 'green', 'red'])
        plt.xticks(x, labels)
        plt.xlabel('r values')
        plt.ylabel('Mean Image Value')
        plt.title(f'c{c} features')
        plt.legend(['Mean ± Std'])
        plt.show()

        # Initialize a dictionary to store the means of the images for each r
        image_means_by_r = {}
        c = '12'
        # Iterate through the values of r
        for r in ['01', '02', '03', '04', '05', '06', '07', '08']:
            # Initialize a list to store the means for the current r
            image_means = []
            # Iterate through the values of f
            for f in range(1, 10):  # f takes values from '01' to '09'
                # Iterate through the images
                for p in ['01']:  # Assuming there are 5 images (p01 to p05)
                    # Load the TIFF image
                    sample_name = f'r{r}c{c}f{f:02d}'
                    if sample_name in sample_list:
                        feature = sample_list[sample_name]['feature']
                        image_means.append(feature)
                    else:
                        # print('no', run_name)
                        pass

            # Store the means for the current r
            image_means_by_r[r] = image_means

        # Calculate the mean and standard deviation for each r
        mean_std_by_r = {r: (np.mean(means), np.std(means)) for r, means in
                         image_means_by_r.items()}

        # Plot the mean and standard deviation for each r
        labels = list(mean_std_by_r.keys())
        means = [mean_std_by_r[r][0] for r in labels]
        stds = [mean_std_by_r[r][1] for r in labels]

        x = range(len(labels))  # X-axis positions

        plt.figure(figsize=(8, 6))
        plt.bar(x, means, yerr=stds, capsize=5,
                color=['blue', 'orange', 'green', 'red'])
        plt.xticks(x, labels)
        plt.xlabel('r values')
        plt.ylabel('Mean Image Value')
        plt.title(f'c{c} features')
        plt.legend(['Mean ± Std'])
        plt.show()

        # Initialize a dictionary to store the means of the images for each c
        image_means_by_c = {}
        r = '06'
        # Iterate through the values of r
        for c in range(1, 13):
            # Initialize a list to store the means for the current c
            image_means = []
            # Iterate through the values of f
            for f in range(1, 10):  # f takes values from '01' to '09'
                # Iterate through the images
                for p in range(1,
                               16):  # Assuming there are 5 images (p01 to p05)
                    # Load the TIFF image
                    run_name = f'r{r}c{c:02d}f{f:02d}p{p:02d}'
                    if run_name in image_list:
                        cell_count = image_list[run_name]['cell_count']
                        image_means.append(cell_count)
                    else:
                        # print('no', run_name)
                        pass

            # Store the means for the current r
            image_means_by_c[c] = image_means

        # Calculate the mean and standard deviation for each r
        mean_std_by_c = {c: (np.mean(means), np.std(means)) for c, means in
                         image_means_by_c.items()}

        # Plot the mean and standard deviation for each r
        labels = list(mean_std_by_c.keys())
        means = [mean_std_by_c[c][0] for c in labels]
        stds = [mean_std_by_c[c][1] for c in labels]

        x = range(len(labels))  # X-axis positions

        plt.figure(figsize=(8, 6))
        plt.bar(x, means, yerr=stds, capsize=5,
                color=['blue', 'orange', 'green', 'red'])
        plt.xticks(x, labels)
        plt.xlabel('c values')
        plt.ylabel('cell count')
        plt.title(f'r{r} cell count')
        plt.legend(['Mean ± Std'])
        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--images_dir',
        dest='images_dir',
        type=str,
        required=False,
        help='directory path of input images')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = get_args()
    try:
        Analysis(base_dir=args.images_dir,
                 feat_type='depth_maximize_pixel_avg').plot()
    except Exception as e:
        print(e)
