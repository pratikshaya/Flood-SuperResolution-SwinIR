import os
import numpy as np
import rasterio
from matplotlib import cm
from PIL import Image

# ==========================================================
# INPUT FOLDERS
# ==========================================================

LR_FOLDER = "/scratch/users/bnbhattarai/FLO_SR_Project/Our_Own_Flood_Data/after_review/Coarse_Grid/150m/"
HR_FOLDER = "/scratch/users/bnbhattarai/FLO_SR_Project/Our_Own_Flood_Data/after_review/Fine_Grid/"

OUTPUT_ROOT = "/scratch/users/bnbhattarai/FLO_SR_Project/Our_Own_Flood_Data/after_review/preprocessed_dataset"

# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================

TRAIN_CITIES = [
    "Greenville",
    "Prince_Tarboro",
    "RockyMount"
]

VAL_CITIES = [
    "Louisburg"
]

# ==========================================================
# PATCH PARAMETERS
# ==========================================================

###This is for 4 times super resolution

#LR_PATCH = 60
#HR_PATCH = 240
#SCALE = 4
#LR_STRIDE = 30

###This is for 2 times super resolution

LR_PATCH = 120
HR_PATCH = 240
SCALE = 2
LR_STRIDE = 60

# ==========================================================
# FLOOD FILTER
# ==========================================================

FLOOD_THRESHOLD = 0.1      # ft
MIN_WATER_RATIO = 0.10     # 10%

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

train_lr_dir = os.path.join(OUTPUT_ROOT, "train_LR")
train_hr_dir = os.path.join(OUTPUT_ROOT, "train_HR")

val_lr_dir = os.path.join(OUTPUT_ROOT, "val_LR")
val_hr_dir = os.path.join(OUTPUT_ROOT, "val_HR")

# ==========================================================
# DEPTH PATCH OUTPUTS
# ==========================================================

train_hr_depth_dir = os.path.join(OUTPUT_ROOT, "train_HR_depth")
train_lr_depth_dir = os.path.join(OUTPUT_ROOT, "train_LR_depth")

val_hr_depth_dir = os.path.join(OUTPUT_ROOT, "val_HR_depth")
val_lr_depth_dir = os.path.join(OUTPUT_ROOT, "val_LR_depth")

for d in [
    train_lr_dir,
    train_hr_dir,
    val_lr_dir,
    val_hr_dir,
    train_hr_depth_dir,
    train_lr_depth_dir,
    val_hr_depth_dir,
    val_lr_depth_dir
]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# FILE PAIRS
# ==========================================================
### This is for 75/300m pairs
'''
pairs = [
    ("Greenville", "Greenville_300m_19.68ft.tif", "Greenville_75m_4.9ft.tif"),
    ("Louisburg", "Louisburg_300m_19.68ft.tif", "Louisburg_75m_4.9ft.tif"),
    ("Prince_Tarboro", "Prince_Tarboro_300m_19.68ft.tif", "Prince_Tarboro_75m_4.9ft.tif"),
    ("RockyMount", "RockyMount_300m_19.68ft.tif", "RockyMount_75m_4.9ft.tif"),
]

'''
###This is for 75/150m pairs
pairs = [
    ("Greenville", "Greenville_150m_9.8ft.tif", "Greenville_75m_4.9ft.tif"),
    ("Louisburg", "Louisberg_150m_9.8ft.tif", "Louisburg_75m_4.9ft.tif"),
    ("Prince_Tarboro", "Prince_Tarboro_150m_9.8ft.tif", "Prince_Tarboro_75m_4.9ft.tif"),
    ("RockyMount", "RockyMount_150m_9.8ft.tif", "RockyMount_75m_4.9ft.tif"),
]

# ==========================================================
# CLEAN READING FUNCTION (IMPORTANT FIX)
# ==========================================================

def read_clean_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata

    arr = arr.astype(np.float32)

    # handle nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    # handle HEC-RAS -9999 explicitly
    arr = np.where(arr == -9999, np.nan, arr)

    # remove negatives
    arr[arr < 0] = np.nan

    arr = np.nan_to_num(arr)

    return arr

# ==========================================================
# READ ALL DATA FIRST (for robust normalization)
# ==========================================================

print("=" * 60)
print("Computing robust global depth statistics (percentile-based)")
print("=" * 60)

all_values = []

for folder in [LR_FOLDER, HR_FOLDER]:
    for fname in os.listdir(folder):

        if not fname.lower().endswith(".tif"):
            continue

        path = os.path.join(folder, fname)
        arr = read_clean_tif(path)

        vals = arr[arr > 0]
        all_values.append(vals)

all_values = np.concatenate(all_values)

global_min = 0.0
global_max = np.percentile(all_values, 99)  # IMPORTANT FIX

print("Global Min Depth =", global_min)
print("Global Max (99th percentile) =", global_max)

np.save(os.path.join(OUTPUT_ROOT, "normalization_parameters.npy"),
    {
        "global_min": global_min,
        "global_max": global_max
    }
)

# ==========================================================
# DEPTH ? RGB
# ==========================================================

cmap = cm.get_cmap("jet")

def depth_to_rgb(depth):

    depth = np.nan_to_num(depth)
    depth[depth < 0] = 0

    norm = depth / (global_max + 1e-8)
    norm = np.clip(norm, 0, 1)

    rgb = cmap(norm)[:, :, :3]
    return (rgb * 255).astype(np.uint8)

# ==========================================================
# PATCH EXTRACTION
# ==========================================================

total_saved = 0

for city, lr_name, hr_name in pairs:

    print("\n" + "=" * 60)
    print("Processing:", city)
    print("=" * 60)

    lr_path = os.path.join(LR_FOLDER, lr_name)
    hr_path = os.path.join(HR_FOLDER, hr_name)

    lr = read_clean_tif(lr_path)
    hr = read_clean_tif(hr_path)

    h_lr, w_lr = lr.shape

    if city in TRAIN_CITIES:
        out_lr = train_lr_dir
        out_hr = train_hr_dir
        out_lr_depth = train_lr_depth_dir
        out_hr_depth = train_hr_depth_dir
    else:
        out_lr = val_lr_dir
        out_hr = val_hr_dir
        out_lr_depth = val_lr_depth_dir
        out_hr_depth = val_hr_depth_dir

    city_count = 0

    for y in range(0, h_lr - LR_PATCH + 1, LR_STRIDE):
        for x in range(0, w_lr - LR_PATCH + 1, LR_STRIDE):

            lr_patch = lr[y:y+LR_PATCH, x:x+LR_PATCH]
            hr_patch = hr[y*SCALE:(y+LR_PATCH)*SCALE,
                          x*SCALE:(x+LR_PATCH)*SCALE]

            if hr_patch.shape != (HR_PATCH, HR_PATCH):
                continue

            # ==================================================
            # WATER COVERAGE FILTER (FIXED)
            # ==================================================

            lr_water_ratio = np.count_nonzero(
                lr_patch > FLOOD_THRESHOLD
            ) / lr_patch.size

            hr_water_ratio = np.count_nonzero(
                hr_patch > FLOOD_THRESHOLD
            ) / hr_patch.size

            if lr_water_ratio < MIN_WATER_RATIO:
                continue

            if hr_water_ratio < MIN_WATER_RATIO:
                continue

            # ==================================================
            # RGB CONVERSION
            # ==================================================

            lr_rgb = depth_to_rgb(lr_patch)
            hr_rgb = depth_to_rgb(hr_patch)

            patch_name = f"{city}_{city_count:06d}.png"

            Image.fromarray(lr_rgb).save(os.path.join(out_lr, patch_name))
            Image.fromarray(hr_rgb).save(os.path.join(out_hr, patch_name))

            depth_name = patch_name.replace(".png", ".npy")
            np.save(os.path.join(out_lr_depth, depth_name),lr_patch.astype(np.float32))
            np.save(os.path.join(out_hr_depth, depth_name),hr_patch.astype(np.float32))

            city_count += 1
            total_saved += 1

    print(f"Saved patches for {city}: {city_count}")

print("\n" + "=" * 60)
print("FINISHED")
print("Total Saved Patch Pairs:", total_saved)
print("=" * 60)
