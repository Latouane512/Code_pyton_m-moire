# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:49:12 2025

@author: ASUS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb
from skimage.util import img_as_float
from scipy.ndimage import binary_opening, binary_closing
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import rasterio
from rasterio.transform import from_origin
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# === PARAMÈTRES GÉNÉRAUX ===
root_dir = 'C:/Users/ASUS/Desktop/Point_ponctuel_bon'
output_dir = 'C:/Users/ASUS/Desktop/Contraster/Kmean_superpixel_slic/Out_segmentation_RGB_3_prediction_teste'
resize_dim = (4000, 3000)
k = 4
n_segments = 4000
compactness = 10.0
color_space = 'RGB'  # Choix possible : 'LAB', 'HSV', 'RGB'

# 🎨 Palette RGB définissant les classes (ordre attendu)
palette_rgb = np.array([
    [255, 230, 150],  # 0: Sable clair
    [30, 160, 60],    # 1: Végétation
    [90, 60, 40],     # 2: Substrat foncé / roche
    [190, 210, 220],  # 3: Reflets de surface
], dtype=np.uint8)

cluster_names = [
    "Sable clair",
    "Végétation",
    "Substrat foncé / roche",
    "Reflets de surface"
]

def circular_structuring_element(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (x**2 + y**2) <= radius**2

structure = circular_structuring_element(radius=10)

def map_clusters_to_palette(kmeans_centers, color_space):
    # Convertir les centres KMeans en RGB pour pouvoir comparer avec la palette
    if color_space == 'LAB':
        rgb_centers = (lab2rgb(kmeans_centers.reshape(1, -1, 3)) * 255).astype(np.uint8).reshape(-1, 3)
    elif color_space == 'HSV':
        rgb_centers = (hsv_to_rgb(kmeans_centers) * 255).astype(np.uint8)
    else:
        rgb_centers = (kmeans_centers * 255).astype(np.uint8)

    distances = cdist(rgb_centers, palette_rgb)
    row_ind, col_ind = linear_sum_assignment(distances)
    return {cluster_idx: palette_idx for cluster_idx, palette_idx in zip(row_ind, col_ind)}

def process_image(image_path, save_image_path, save_figure_path=None):
    try:
        print(f"\n🔍 Traitement de : {image_path}")
        image = Image.open(image_path).resize(resize_dim)
        image_np = np.array(image)
        image_float = img_as_float(image_np)

        # Conversion de couleur
        if color_space == 'LAB':
            image_conv = rgb2lab(image_float)
        elif color_space == 'HSV':
            image_conv = rgb_to_hsv(image_float)
        else:
            image_conv = image_float

        # --- 1. Superpixels ---
        segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=0)
        n_spx = segments.max() + 1

        # --- 2. Moyennes superpixels ---
        features = np.zeros((n_spx, 3))
        for i in range(n_spx):
            mask = (segments == i)
            for c in range(3):
                features[i, c] = image_conv[:, :, c][mask].mean()

        # --- 3. KMeans ---
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        kmeans_centers = kmeans.cluster_centers_

        # --- 4. Mapping automatique des clusters vers la palette ---
        inverse_map = map_clusters_to_palette(kmeans_centers, color_space)

        print("📌 Centres KMeans (convertis RGB) :")
        for idx, pal_idx in inverse_map.items():
            print(f"  Cluster {idx} → {cluster_names[pal_idx]}")

        # --- 5. Filtrage morphologique ---
        final_labels = np.zeros_like(segments)
        for cluster_idx in range(k):
            mask = np.isin(segments, np.where(labels == cluster_idx))
            mask_open = binary_opening(mask, structure=structure)
            mask_clean = binary_closing(mask_open, structure=structure)
            final_labels[mask_clean] = cluster_idx

        # --- 6. Reconstruction image segmentée ---
        final_image = np.zeros_like(image_np)
        for i in range(n_spx):
            mapped_color = palette_rgb[inverse_map[labels[i]]]
            final_image[segments == i] = mapped_color

        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        Image.fromarray(final_image).save(save_image_path)
        print(f"✅ Image segmentée sauvegardée : {save_image_path}")

        # --- 7. Export TIFF des labels ---
        save_label_path = save_image_path.replace('.jpg', '_labels.tif').replace('.png', '_labels.tif')
        transform = from_origin(0, 0, 1, 1)
        with rasterio.open(
            save_label_path,
            'w',
            driver='GTiff',
            height=final_labels.shape[0],
            width=final_labels.shape[1],
            count=1,
            dtype=rasterio.uint8,
            transform=transform,
            crs=None
        ) as dst:
            dst.write(final_labels.astype(np.uint8), 1)
        print(f"📁 Label TIFF sauvegardé : {save_label_path}")

        # --- 8. Affichage avec légende ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title("Image originale")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(final_image)
        plt.title("Image segmentée")
        plt.axis('off')

        patches = [
            mpatches.Patch(color=np.array(color)/255.0, label=cluster_names[i])
            for i, color in enumerate(palette_rgb)
        ]
        plt.figlegend(handles=patches, loc='lower center', ncol=3, fontsize='medium', frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_figure_path:
            plt.savefig(save_figure_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    except Exception as e:
        print(f"❌ Erreur : {e}")

# === BOUCLE DE TRAITEMENT ===
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(image_path, root_dir)
            save_image_path = os.path.join(output_dir, rel_path)
            save_figure_path = os.path.splitext(save_image_path)[0] + '_legend.png'
            process_image(image_path, save_image_path, save_figure_path)
