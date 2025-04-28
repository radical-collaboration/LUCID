import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.segmentation import find_boundaries
from cellSAM.cellsam_pipeline import cellsam_pipeline  # Using the pipeline from your test files
from cellSAM import segment_cellular_image, get_model
import os
import pandas as pd
import shutil
import torch

# Existing classes
class CellPaint:
    def __init__(self, image=None, csv=None, csv_data=None):
        self.image = image      # path to the main image
        self.csv = csv          # path to the CSV file
        self.csv_data = csv_data  # path to the additional CSV-related image

    def __repr__(self):
        return (f"CellPaint(image={self.image!r}, "
                f"csv={self.csv!r}, csv_data={self.csv_data!r})")


class CellPaintDataset(Dataset):
    def __init__(self, directory, root_directory, transform=None, 
                 csv_data_transform=None, load_images=False):
        """
        Args:
            directory (str): The top-level directory to scan.
            root_directory (str): The base directory used for computing relative paths.
            transform (callable, optional): Transformation to apply to the main image.
            csv_data_transform (callable, optional): Transformation to apply to the csv_data image.
            load_images (bool): If True, images are loaded via PIL; otherwise, file paths are returned.
        """
        self.transform = transform
        self.csv_data_transform = csv_data_transform
        self.load_images = load_images
        self.root_directory = root_directory

        # We'll store our samples as a list of (relative_path, CellPaint) tuples.
        self.data = []
        cell_paint_data = {}

        for root, _, files in os.walk(directory):
            for file in files:
                name, ext = os.path.splitext(file)
                # Get the full path without extension
                full_relative_path = os.path.join(root, name)
                # Compute a relative path (used as an identifier) based on root_directory
                relative_path = os.path.relpath(full_relative_path, root_directory)
                
                # Process only .png files
                if ext.lower() == ".png":
                    if relative_path not in cell_paint_data:
                        cell_paint_data[relative_path] = CellPaint()
                    # Naming convention:
                    #  - Main image: name + '.png'
                    #  - CSV file: name + '.csv'
                    #  - Additional image: name + '_data.png'
                    if not name.endswith("_data"):
                        cell_paint_data[relative_path].image = os.path.join(root, name + '.png')
                        cell_paint_data[relative_path].csv = os.path.join(root, name + '.csv')
                        cell_paint_data[relative_path].csv_data = os.path.join(root, name + '_data.csv')
                    else:
                        cell_paint_data[relative_path].csv_data = os.path.join(root, file)
                        
        # Convert dictionary to list for indexing
        for rel_path, cell in cell_paint_data.items():
            self.data.append((rel_path, cell))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rel_path, cell = self.data[idx]
        sample = {
            'relative_path': rel_path,
            'image': cell.image,
            'csv': cell.csv,
            'csv_data': cell.csv_data
        }
        
        # Optionally load images and CSV data
        if self.load_images:
            if sample['image'] is not None and os.path.exists(sample['image']):
                image = Image.open(sample['image']).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                sample['image'] = image

            if sample['csv_data'] is not None and os.path.exists(sample['csv_data']):
                csv_data_img = Image.open(sample['csv_data']).convert("RGB")
                if self.csv_data_transform:
                    csv_data_img = self.csv_data_transform(csv_data_img)
                sample['csv_data'] = csv_data_img

            if sample['csv'] is not None and os.path.exists(sample['csv']):
                try:
                    sample['csv'] = pd.read_csv(sample['csv'])
                except Exception as e:
                    print(f"Error loading CSV file {sample['csv']}: {e}")
                    sample['csv'] = None
                    
        return sample

# ------------------------
# NEW FUNCTIONALITY INTEGRATING SEGMENTATION
# ------------------------



def process_sample(sample, device):
    """
    For a given sample, this function:
      - Loads the original image.
      - Checks if segmentation outputs already exist (mask file).
      - If not, runs the segmentation pipeline to obtain a nucleus mask and bounding boxes.
      - Draws bounding boxes on the original image.
      - Finds cell boundaries and creates an overlay.
      - Saves the outputs in a directory where "cell_data" is replaced by "image_segmentation"
        in the original image's directory path.
      - Saves a CSV file (with suffix _cell_count.csv) that contains the original image path 
        and the number of cells detected (i.e. bounding_boxes.shape[0]).
      - Does not display any figures, only saves them.
    """

    import os
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd
    from PIL import Image
    from skimage.segmentation import find_boundaries

    # Get the original image path
    img_path = sample['image']
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # original_dir is the directory containing the original image.
    original_dir = os.path.dirname(img_path)
    # Create segment_dir by replacing "cell_data" with "image_segmentation" in original_dir.
    segment_dir = original_dir.replace("cell_data", "all_weeks_nucleus")
    os.makedirs(segment_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Set up the mask filename and check if it already exists
    mask_filename = os.path.join(segment_dir, base_name + "_mask_nucleus.png")
    if os.path.exists(mask_filename):
        print(f"Mask file already exists for sample {mask_filename}, skipping segmentation.")
        return
    
    # Copy original image to segment_dir
    original_copy_path = os.path.join(segment_dir, base_name + ".png")
    shutil.copy(img_path, original_copy_path)
    


    # Load original image as a numpy array
    img = np.array(Image.open(img_path))

    # Determine scaling factor (assuming model input size is 1024x1024)
    original_size = img.shape[0]  # assuming square images
    scaled_size = 1024
    scale_factor = original_size / scaled_size

    # Run the cell segmentation pipeline
    mask, embedding, bounding_boxes = segment_cellular_image(img[:, :, 0], bbox_threshold=0.25, normalize=True, device=str(device))

    # Check if mask is valid (i.e. not None and has the expected attributes)
    if mask is None or not hasattr(mask, 'shape'):
        print(f"Invalid mask for sample {img_path}, skipping sample.")
        return

    if hasattr(bounding_boxes, "cpu"):
        bounding_boxes = bounding_boxes.cpu().numpy()

    # --- Save nucleus mask ---
    plt.figure()
    plt.imshow(mask, cmap='viridis')
    plt.axis('off')
    plt.savefig(mask_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save bounding boxes image ---
    bbx_filename = os.path.join(segment_dir, base_name + "_bbx_nucleus.png")
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax = bbox
        xmin_new = xmin * scale_factor
        ymin_new = ymin * scale_factor
        width = (xmax - xmin) * scale_factor
        height = (ymax - ymin) * scale_factor
        rect = patches.Rectangle(
            (xmin_new, ymin_new), width, height,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(bbx_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    binary_mask = (mask > 0).astype(np.uint8)

    # If the original image is in RGB, expand the binary mask to 3 channels
    if img.ndim == 3 and img.shape[2] == 3:
        binary_mask = np.stack([binary_mask] * 3, axis=-1)
    
    # Multiply the original image with the binary mask to get the masked image
    masked_image = img * binary_mask
    
    # Save the masked image
    mask_img_filename = os.path.join(segment_dir, base_name + "_only_nucleus.png")
    Image.fromarray(masked_image).save(mask_img_filename)


#    # --- Save boundary overlay image ---
#    mask_bd = cellsam_pipeline(
#        img,
#        low_contrast_enhancement=False,
#        use_wsi=False,
#        gauge_cell_size=False
#    )[0]
#
#    boundaries = find_boundaries(mask_bd, mode='thick')
#    overlay_img = img.copy()
#    overlay_img[boundaries] = [255, 0, 0]  # highlight boundaries in red
#    boundary_overlay_filename = os.path.join(segment_dir, base_name + "_cell_boundary.png")
#    plt.figure()
#    plt.imshow(overlay_img)
#    plt.axis('off')
#    plt.savefig(boundary_overlay_filename, dpi=300, bbox_inches='tight')
#    plt.close()
#
#    # --- Save binary boundary mask ---
#    boundary_mask_filename = os.path.join(segment_dir, base_name + "_mask_cell_boundary.png")
#    plt.figure()
#    plt.imshow(mask_bd, cmap='viridis')
#    plt.axis('off')
#    plt.savefig(boundary_mask_filename, dpi=300, bbox_inches='tight')
#    plt.close()

    # Get the number of cells detected (bounding_boxes.shape[0])
    cell_count = bounding_boxes.shape[0]
    print(f"Number of cells in {base_name}: {cell_count}")

    # Save the image path and cell count in a CSV file with suffix _cell_count.csv.
    cell_count_csv = os.path.join(segment_dir, base_name + "_cell_count.csv")
    df = pd.DataFrame({
        "image_path": [img_path],
        "cell_count": [cell_count]
    })
    df.to_csv(cell_count_csv, index=False)

    print(f"Processed sample: {base_name} - outputs saved in {segment_dir}")


    
# ------------------------
# MAIN EXECUTION: Process all samples in the dataset
# ------------------------
if __name__ == '__main__':
    # Parse command-line arguments for the directory paths
    parser = argparse.ArgumentParser(description='Run CellPaintDataset segmentation.')
    parser.add_argument('--data_dir', type=str,
                        default='/hpcgpfs01/scratch/xyu1/cell_data/cellpaint/rpe_images/week_two/',
                        help='Path to the dataset directory.')
    parser.add_argument('--root_dir', type=str,
                        default='/hpcgpfs01/scratch/xyu1/',
                        help='Base directory for computing relative paths.')
    args = parser.parse_args()

    # Use the provided directory paths
    dataset = CellPaintDataset(
        directory=args.data_dir,
        root_directory=args.root_dir,
        transform=None,
        csv_data_transform=None,
        load_images=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset size: {len(dataset)}")

    # Process each sample in the dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        process_sample(sample, device)
