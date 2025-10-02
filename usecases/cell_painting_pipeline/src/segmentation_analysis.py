#!/usr/bin/env python3

import argparse
import json
import os
import sys
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
from tqdm import tqdm

FEAT_TYPE_VALUES = ['pixel_avg', 'depth_maximize_pixel_avg', 'rpe_images']


class CellPaintDataset(Dataset):

    def __init__(self,
                 base_dir: str,
                 base_channel: str = 'ch2',
                 target_channel: str = 'ch3',
                 feat_type: str = None,
                 transform=None):
        """
        Args:
            base_dir (str): Root path to the cell paint data
            feat_type (str, optional): Feature type. Defaults to 'pixel_avg'
            transform (callable, optional): Transform to be applied on the image
        """
        
        TGT_SUFFIX = f'-{target_channel}sk1fk1fl1_{target_channel}_masked.npy'
        SEG_SUFFIX = f'-{base_channel}sk1fk1fl1_{base_channel}_nucleu_mask.png'
        CSV_SUFFIX = f'-{base_channel}sk1fk1fl1_cell_count.csv'
        self.base_dir = base_dir
        self.feat_type = feat_type or FEAT_TYPE_VALUES[0]
        assert self.feat_type in FEAT_TYPE_VALUES, 'Unknown feature type'
        self.transform = transform

        # store each record as a dictionary
        self.data = {}
        seen_runs = set()  # updated: track unique run_name prefixes

        # Walk through the directory and gather file paths per run
        for curr_dir, _, files in os.walk(self.base_dir):
            for file in files:
                if not file.endswith(TGT_SUFFIX):
                    continue
                full_base = os.path.splitext(file)[0]
                run_key = full_base[:9]  # updated: r01c01f01 prefix
                if run_key in seen_runs:
                    continue  # updated: skip duplicates
                seen_runs.add(run_key)

                # Gather lists of file paths for each p
                tar_list = []
                seg_list = []
                csv_list = []  
                for p in range(1, 16):
                    p2d = f'p{p:02d}'
                    tar_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{TGT_SUFFIX}")) 
                    seg_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{SEG_SUFFIX}"))
                    csv_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{CSV_SUFFIX}"))

                # Create record dictionary
                record = {
                    'run_name': run_key,
                    'feat_image_path': tar_list,
                    'seg_image_path': seg_list,
                    'csv_path': csv_list,
                }

                # Load CSVs to get cell counts
                try:
                    counts = []
                    for csv_p in csv_list:
                        df = pd.read_csv(csv_p)
                        counts.append(df['cell_count'][0])
                    record['cell_count'] = counts
                except Exception as e:
                    print(f'Error reading CSV {csv_p}: {e}')
                    continue

                self.data[run_key] = record

    def __len__(self):
        return len(self.data)

    def _get_run_name(self, idx):  # added helper for indexing
        return sorted(self.data.keys())[idx] if idx is not None else None

    def __getitem__(self, idx):
        run_name = self._get_run_name(idx)
        record = self.data[run_name]

        feat = self.feat_func(record, feat_type=self.feat_type)
        record['feature'] = feat

        return record

    def feat_func(self, record, feat_type='pixel_avg'):
        """
        Compute feature based on feat_type and channel selection ch (3 or 8).
        """
        paths = record['feat_image_path']
        arrays = []
        for pth in paths:
            arrays.append(np.load(pth))

        if feat_type == 'pixel_avg':
            all_pix = np.concatenate([arr[arr > 0].ravel() for arr in arrays])
            feature = all_pix.mean()

        elif feat_type == 'depth_maximize_pixel_avg':
            stacked = np.stack(arrays)
            max_img = np.max(stacked, axis=0)
            feature = max_img[max_img > 0].mean()

        else:
            raise NotImplementedError()

        return feature


class Analysis:
    def __init__(self, base_dir, base_channel, target_channel, feat_type,
                 week_name, plate_name, plate_experiments, transform=None):
        self.dataset = CellPaintDataset(
            base_dir=base_dir,
            base_channel=base_channel,
            target_channel=target_channel,
            feat_type=feat_type,
            transform=transform,
        )
        # preload or generate data_dict
        self.week_name = week_name
        self.plate_name = plate_name
        self.exp_setting = plate_experiments['exp_setting']
        self.exp_list = plate_experiments['exp_list'][plate_name]
        self.target_channel = target_channel
        self.dye_label = plate_experiments['dye_label']

        self.results_dir = os.path.join(base_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        fp = f'{week_name}_{plate_name}_dataset.pkl'
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                self.data_dict = pickle.load(f)
        else:
            dd = {}
            for rec in tqdm(self.dataset):
                dd[rec['run_name']] = rec
            with open(fp, 'wb') as f:
                pickle.dump(dd, f)
            self.data_dict = dd

    def plot(self):
        # experiment settings
        for exp in self.exp_list:
            all_vals, all_rads, all_rc = [], [], []
            img_by_rad = {rad:[] for rad in self.exp_setting[exp]}
            for rad, cfg in self.exp_setting[exp].items():
                for r in cfg['r']:
                    for c in cfg['c']:
                        for f in range(1,10):
                            sn = f"r{r:02d}c{c:02d}f{f:02d}"
                            if sn in self.data_dict:
                                feat = self.data_dict[sn]['feature']/np.mean(self.data_dict[sn]['cell_count'])
                                all_vals.append(feat); all_rads.append(rad); all_rc.append(f"r{r}c{c}")
                                img_by_rad[rad].append(feat)
            cmap = plt.get_cmap('tab10')
            cmap_map = {rad:cmap(i) for i,rad in enumerate(img_by_rad)}
            colors = [cmap_map[r] for r in all_rads]
            plt.figure(figsize=(12,6))
            x = np.arange(len(all_vals))
            plt.bar(x, all_vals, color=colors)
            plt.title(f"{self.week_name} {self.plate_name} {exp} ({self.dye_label[exp]}): cell average {self.target_channel} nucleus")
            plt.xlabel('Index'); plt.ylabel('Mean of max-projection')
            ticks=[]; prev=None
            for lab in all_rc:
                ticks.append(lab if lab!=prev else ''); prev=lab
            plt.xticks(x,ticks,rotation=90)
            patches=[mpatches.Patch(color=cmap_map[r], label=r) for r in img_by_rad]
            plt.legend(handles=patches, title='rad', bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, f"{self.week_name}_{self.plate_name}_{exp}_{self.target_channel}.png")
            plt.savefig(save_path)
            plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--images_dir',
        dest='images_dir',
        type=str,
        required=True,
        help='directory path of input images')
    parser.add_argument(
        '-w', '--week_name',
        dest='week_name',
        type=str,
        required=True,
        help='week name prefix for data dump')
    parser.add_argument(
        '--plate_name',
        type=str,
        required=True,
        help='Plate name')
    parser.add_argument(
        '--plate_config',
        dest='plate_config',
        type=str,
        required=True,
        help='Configuration file with plate experiments')
    parser.add_argument(
        '--feature_type',
        dest='feature_type',
        type=str,
        required=True,
        help='Feature type')
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
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':

    args = get_args()

    plate_config = args.plate_config
    if '/' not in plate_config:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plate_config = f'{current_dir}/{plate_config}'
    with open(plate_config, 'r') as file:
        plate_experiments = json.load(file)

    try:
        Analysis(base_dir=args.images_dir,
                 base_channel=args.base_channel,
                 target_channel=args.target_channel,
                 feat_type=args.feature_type,
                 week_name=args.week_name,
                 plate_name=args.plate_name,
                 plate_experiments=plate_experiments).plot()
    except Exception as e:
        print(e)

