#!/usr/bin/env python3

import argparse
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

# Updated suffix constants based on new dataset
TGT_SUFFIX = '-ch3sk1fk1fl1_ch3_nucleus_areas.npy'
SEG_SUFFIX = '-ch2sk1fk1fl1_ch2_nucleu_mask.png'
CSV_SUFFIX = '-ch2sk1fk1fl1_cell_count.csv'
CH8_SUFFIX = '-ch8sk1fk1fl1_ch8_nucleus_areas.npy'

FEAT_TYPE_VALUES = ['pixel_avg', 'depth_maximize_pixel_avg', 'rpe_images']


exp_setting = {}
plate3_exp = {
    'rad_2': {'r': [7,8], 'c': [1,2,3,4]},
    'rad_1': {'r': [1,2,3,4], 'c': [1,2]},
    'rad_0.1': {'r':[1,2,3,4],'c':[6,7]},
    'rad_0.01':{'r':[7,8],'c':[9,10,11,12]},
    'rad_0.001':{'r':[1,2,3,4],'c':[11,12]}
}
plate8_exp_green = {
    'rad_2': {'r':[7],'c':[3,4]},
    'rad_1': {'r':[4],'c':[1,2]},
    'rad_0.1':{'r':[4],'c':[6,7]},
    'rad_0.01':{'r':[8],'c':[11,12]},
    'rad_0.001':{'r':[4],'c':[11,12]}
}
plate8_exp_red = {
    'rad_2': {'r':[8],'c':[3,4]},
    'rad_1': {'r':[3],'c':[1,2]},
    'rad_0.1':{'r':[3],'c':[6,7]},
    'rad_0.01':{'r':[7],'c':[11,12]},
    'rad_0.001':{'r':[3],'c':[11,12]}
}
plate8_exp_yellow = {
    'rad_2': {'r':[7,8],'c':[1,2]},
    'rad_1': {'r':[1,2],'c':[1,2]},
    'rad_0.1':{'r':[1,2],'c':[6,7]},
    'rad_0.01':{'r':[7,8],'c':[9,10]},
    'rad_0.001':{'r':[1,2],'c':[11,12]}
}
exp_setting['plate3_exp'] = plate3_exp
exp_setting['plate8_exp_green'] = plate8_exp_green
exp_setting['plate8_exp_red'] = plate8_exp_red
exp_setting['plate8_exp_yellow'] = plate8_exp_yellow
# dye labels
dye_label = {
    'plate3_exp': None,
    'plate8_exp_green':'caspase on ch3',
    'plate8_exp_red':'caspase on ch3 and h2ax on ch8',
    'plate8_exp_yellow':'h2ax on ch3'
}
dye_label_ch8 = {
    'plate3_exp':'normal ch8',
    'plate8_exp_green':'normal ch8',
    'plate8_exp_red':'h2ax on ch8',
    'plate8_exp_yellow':'normal ch8'
}

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
                ch3_list = []
                ch8_list = []
                seg_list = []
                csv_list = []  
                for p in range(1, 16):
                    p2d = f'p{p:02d}'
                    ch3_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{TGT_SUFFIX}")) 
                    ch8_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{CH8_SUFFIX}"))
                    seg_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{SEG_SUFFIX}"))
                    csv_list.append(os.path.join(curr_dir, f"{run_key}{p2d}{CSV_SUFFIX}"))

                # Create record dictionary
                record = {
                    'run_name': run_key,
                    'feat_image_path': ch3_list,
                    'seg_image_path': seg_list,
                    'csv_path': csv_list,
                    'ch8_paths': ch8_list,
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

        # compute features for ch3 (default) and ch8
        feat3 = self.feat_func(record, feat_type=self.feat_type, ch=3)
        record['feature'] = feat3

        feat8 = self.feat_func(record, feat_type=self.feat_type, ch=8)
        record['feature8'] = feat8

        return record

    def feat_func(self, record, feat_type='pixel_avg', ch=3):
        """
        Compute feature based on feat_type and channel selection ch (3 or 8).
        """
        paths = record['feat_image_path'] if ch == 3 else record['ch8_paths']
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
    def __init__(self, base_dir, feat_type, week_name, transform=None):
        self.dataset = CellPaintDataset(base_dir=base_dir,
                                        feat_type=feat_type,
                                        transform=transform)
        # preload or generate data_dict
        self.week_name = week_name
        fp = f'{week_name}_dataset.pkl'
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
        # choose experiments
        if 'pl8' in self.week_name:
            exp_list = ['plate8_exp_yellow','plate8_exp_green','plate8_exp_red']
        elif 'plc' in self.week_name:
            exp_list = ['platec_exp_yellow','platec_exp_green','platec_exp_red']
        else:
            exp_list = ['plate3_exp']
        for exp in exp_list:
            # ch3 plot
            all_vals, all_rads, all_rc = [], [], []
            img_by_rad = {rad:[] for rad in exp_setting[exp]}
            for rad, cfg in exp_setting[exp].items():
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
            plt.title(f"{self.week_name} {exp} ({dye_label[exp]}): cell average ch3 nucleus")
            plt.xlabel('Index'); plt.ylabel('Mean of max-projection')
            ticks=[]; prev=None
            for lab in all_rc:
                ticks.append(lab if lab!=prev else ''); prev=lab
            plt.xticks(x,ticks,rotation=90)
            patches=[mpatches.Patch(color=cmap_map[r], label=r) for r in img_by_rad]
            plt.legend(handles=patches, title='rad', bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"./draw_result/{self.week_name}_{exp}_ch3.png")
            plt.close()
            # ch8 plot for red experiments
            if 'red' in exp:
                all_vals, all_rads, all_rc = [], [], []
                img_by_rad = {rad:[] for rad in exp_setting[exp]}
                for rad, cfg in exp_setting[exp].items():
                    for r in cfg['r']:
                        for c in cfg['c']:
                            for f in range(1,10):
                                sn = f"r{r:02d}c{c:02d}f{f:02d}"
                                if sn in self.data_dict:
                                    feat = self.data_dict[sn]['feature8']/np.mean(self.data_dict[sn]['cell_count'])
                                    all_vals.append(feat); all_rads.append(rad); all_rc.append(f"r{r}c{c}")
                                    img_by_rad[rad].append(feat)
                cmap = plt.get_cmap('tab10')
                cmap_map = {rad:cmap(i) for i,rad in enumerate(img_by_rad)}
                colors = [cmap_map[r] for r in all_rads]
                plt.figure(figsize=(12,6))
                x = np.arange(len(all_vals))
                plt.bar(x, all_vals, color=colors)
                plt.title(f"{self.week_name} {exp} ({dye_label_ch8[exp]}): cell average ch8 nucleus")
                plt.xlabel('Index'); plt.ylabel('Mean of max-projection')
                ticks=[]; prev=None
                for lab in all_rc:
                    ticks.append(lab if lab!=prev else ''); prev=lab
                plt.xticks(x,ticks,rotation=90)
                patches=[mpatches.Patch(color=cmap_map[r], label=r) for r in img_by_rad]
                plt.legend(handles=patches, title='rad', bbox_to_anchor=(1.05,1), loc='upper left')
                plt.tight_layout()
                plt.savefig(f"./draw_result/{self.week_name}_{exp}_ch8.png")
                plt.close()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--images_dir',dest='images_dir',type=str,required=True,help='directory path of input images')
    parser.add_argument('-w','--week_name',dest='week_name',type=str,required=True,help='week name prefix for data dump')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    args = get_args()
    try:
        Analysis(base_dir=args.images_dir,
                 feat_type='depth_maximize_pixel_avg',
                 week_name=args.week_name).plot()
    except Exception as e:
        print(e)