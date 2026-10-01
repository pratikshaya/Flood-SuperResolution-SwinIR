import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colormaps
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ==========================================================
# INPUT PATHS
# ==========================================================

HR_DEPTH_DIR = "/scratch/users/bnbhattarai/FLO_SR_Project/Our_Own_Flood_Data/after_review/preprocessed_dataset/val_HR_depth"

SR_DIR = "/scratch/users/bnbhattarai/KAIR/results/swinir_physical_verification_x2/" # this is for swinir

#SR_DIR = "/scratch/users/bnbhattarai/FLO_SR_Project/result/after_review/2X/SR/" #this is for FLO_SR

NORM_FILE = "/scratch/users/bnbhattarai/FLO_SR_Project/Our_Own_Flood_Data/after_review/preprocessed_dataset/normalization_parameters.npy"

OUTPUT_CSV = "evaluation_metrics.csv"

# ==========================================================
# LOAD NORMALIZATION PARAMETERS
# ==========================================================

params = np.load(NORM_FILE, allow_pickle=True).item()

global_min = params["global_min"]
global_max = params["global_max"]

print("global_min =", global_min)
print("global_max =", global_max)

# ==========================================================
# REBUILD JET COLORMAP
# ==========================================================

cmap = colormaps["jet"]

lut_size = 256

lut = (cmap(np.linspace(0, 1, lut_size))[:, :3] * 255).astype(np.uint8)

# ==========================================================
# RGB -> DEPTH
# ==========================================================

def rgb_to_depth(rgb):

    rgb = rgb.astype(np.uint8)

    flat = rgb.reshape(-1, 3)

    diff = (
        (flat[:, None, :] - lut[None, :, :]) ** 2
    ).sum(axis=2)

    idx = np.argmin(diff, axis=1)

    norm = idx / (lut_size - 1)

    depth = norm * global_max

    return depth.reshape(rgb.shape[:2])

# ==========================================================
# CSI
# ==========================================================

def compute_csi(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    denom = TP + FP + FN

    if denom == 0:
        return np.nan

    return TP / denom

# ==========================================================
# HISTOGRAM SIMILARITY
# ==========================================================

def histogram_similarity(hr, sr):

    hist_hr, _ = np.histogram(
        hr,
        bins=50,
        range=(0, global_max),
        density=True
    )

    hist_sr, _ = np.histogram(
        sr,
        bins=50,
        range=(0, global_max),
        density=True
    )

    similarity = np.sum(
        np.minimum(hist_hr, hist_sr)
    )

    return similarity

# ==========================================================
# THRESHOLD FOR FLOOD EXTENT
# ==========================================================

FLOOD_THRESHOLD = 2

# ==========================================================
# FIND MATCHING FILES
# ==========================================================

depth_files = sorted(
    glob.glob(os.path.join(HR_DEPTH_DIR, "*.npy"))
)

results = []

# ==========================================================
# LOOP
# ==========================================================

for depth_file in depth_files:

    name = os.path.basename(depth_file)

    sr_file = os.path.join(
        SR_DIR,
        name.replace(".npy", ".png")
    )

    if not os.path.exists(sr_file):

        print("Missing:", sr_file)
        continue

    # ------------------------------------------------------
    # HR DEPTH
    # ------------------------------------------------------

    hr_depth = np.load(depth_file)

    # ------------------------------------------------------
    # SR RGB
    # ------------------------------------------------------

    sr_rgb = np.array(
        Image.open(sr_file).convert("RGB")
    )

    sr_depth = rgb_to_depth(sr_rgb)

    # ------------------------------------------------------
    # MAE
    # ------------------------------------------------------

    mae = mean_absolute_error(
        hr_depth.flatten(),
        sr_depth.flatten()
    )

    # ------------------------------------------------------
    # RMSE
    # ------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            hr_depth.flatten(),
            sr_depth.flatten()
        )
    )

    # ------------------------------------------------------
    # FLOOD EXTENT
    # ------------------------------------------------------

    hr_flood = (
        hr_depth > FLOOD_THRESHOLD
    ).astype(np.uint8)

    sr_flood = (
        sr_depth > FLOOD_THRESHOLD
    ).astype(np.uint8)

    precision = precision_score(
        hr_flood.flatten(),
        sr_flood.flatten(),
        zero_division=0
    )

    recall = recall_score(
        hr_flood.flatten(),
        sr_flood.flatten(),
        zero_division=0
    )

    f1 = f1_score(
        hr_flood.flatten(),
        sr_flood.flatten(),
        zero_division=0
    )

    csi = compute_csi(
        hr_flood,
        sr_flood
    )

    # ------------------------------------------------------
    # FLOOD AREA ERROR
    # ------------------------------------------------------

    hr_area = int(np.sum(hr_flood))
    sr_area = int(np.sum(sr_flood))
    print(name,hr_area,sr_area)

    if hr_area > 0:

        flood_area_error = (
            abs(sr_area - hr_area)
            / hr_area
        ) * 100

    else:

        flood_area_error = np.nan

    # ------------------------------------------------------
    # WATER VOLUME ERROR
    # ------------------------------------------------------

    hr_volume = np.sum(hr_depth)

    sr_volume = np.sum(sr_depth)

    if hr_volume > 0:

        volume_error = (
            abs(sr_volume - hr_volume)
            / hr_volume
        ) * 100

    else:

        volume_error = np.nan

    # ------------------------------------------------------
    # HISTOGRAM SIMILARITY
    # ------------------------------------------------------

    hist_sim = histogram_similarity(
        hr_depth,
        sr_depth
    )

    results.append([
        name,
        mae,
        rmse,
        precision,
        recall,
        f1,
        csi,
        flood_area_error,
        volume_error,
        hist_sim
    ])

# ==========================================================
# SAVE RESULTS
# ==========================================================

columns = [
    "Patch",
    "MAE",
    "RMSE",
    "Precision",
    "Recall",
    "F1",
    "CSI",
    "FloodAreaErrorPercent",
    "VolumeErrorPercent",
    "HistogramSimilarity"
]

df = pd.DataFrame(
    results,
    columns=columns
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ==========================================================
# PRINT AVERAGES
# ==========================================================

print("\n==============================")
print("AVERAGE RESULTS")
print("==============================")

for c in columns[1:]:

    print(
        f"{c}: "
        f"{df[c].mean():.4f}"
    )

print("\nSaved:")
print(OUTPUT_CSV)