# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:17:30 2025

@author: ASUS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, lab2rgb
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import rasterio
from rasterio.transform import from_origin

# === PARAMÈTRES ===
root_dir = 'C:/Users/ASUS/Desktop/Point_ponctuel_bon'
output_dir = 'C:/Users/ASUS/Desktop/Contraster/Kmean_4000x3000/Out_segmentation_RGB_4_prediction'
resize_dim = (4000, 3000)
k = 4
color_space = 'RGB'  # Choix possible : 'LAB', 'HSV', 'RGB'

# === PALETTE FIXE ET NOMS ASSOCIÉS ===
palette_rgb = np.array([
    [250, 240, 190],  # Sable clair
    [50, 160, 60],    # Végétation
    [90, 60, 40],     # Substrat foncé / roche
    [190, 210, 220],  # Reflets de surface
], dtype=np.uint8)

cluster_names_palette = [
    "Sable clair",
    "Végétation",
    "Substrat foncé / roche",
    "Reflets de surface"
]

# === STRUCTURANT CIRCULAIRE ===
def circular_structuring_element(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (x**2 + y**2) <= radius**2

structure = circular_structuring_element(radius=6)

# === AFFICHAGE AVEC LÉGENDE ===
def display_with_legend(image, segmented_image, filtered_image, cluster_colors, cluster_labels, title=None):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.title(f"Image originale : {image_name}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f"Segmentée (K-Means, {color_space})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image)
    plt.title("Filtrée (morpho circulaire)")
    plt.axis('off')

    patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=cluster_labels[i])
        for i, color in enumerate(cluster_colors)
    ]
    plt.figlegend(handles=patches, loc='center left', bbox_to_anchor=(0.88, 0.5), fontsize='medium')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if title:
        plt.savefig(title, dpi=150)
        print(f"✅ Figure enregistrée : {title}")
    else:
        plt.show()

# === CONVERSIONS RGB → LAB / HSV / RGB ===
def convert_rgb_to_color_space(image_rgb, space):
    norm = image_rgb / 255.0
    if space == 'LAB':
        return rgb2lab(norm)
    elif space == 'HSV':
        return rgb_to_hsv(norm)
    elif space == 'RGB':
        return norm  # normalisé entre 0 et 1
    else:
        raise ValueError(f"Color space non supporté : {space}")

def convert_color_space_to_rgb(image_cs, space):
    if space == 'LAB':
        rgb_norm = lab2rgb(image_cs)
    elif space == 'HSV':
        rgb_norm = hsv_to_rgb(image_cs)
    elif space == 'RGB':
        rgb_norm = image_cs  # déjà en RGB normalisé
    else:
        raise ValueError(f"Color space non supporté : {space}")
    rgb_uint8 = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)
    return rgb_uint8

# === TRAITEMENT D'UNE IMAGE ===
def process_and_save_image(image_path, save_image_path, save_figure_path=None):
    try:
        print(f"🔍 Traitement de : {image_path}")
        image = np.array(Image.open(image_path).resize(resize_dim))

        # Conversion RGB → espace choisi
        image_cs = convert_rgb_to_color_space(image, color_space)

        # KMeans clustering sur l'espace choisi
        pixels = image_cs.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Conversion palette RGB → espace choisi
        palette_norm = palette_rgb / 255.0
        if color_space == 'LAB':
            palette_cs = rgb2lab(palette_norm[np.newaxis, :, :])[0]
        elif color_space == 'HSV':
            palette_cs = rgb_to_hsv(palette_norm[np.newaxis, :, :])[0]
        elif color_space == 'RGB':
            palette_cs = palette_norm
        else:
            raise ValueError(f"Color space non supporté : {color_space}")

        # Distance Euclidienne entre centres de clusters et palette fixe
        dist_matrix = cdist(cluster_centers, palette_cs)

        # Matching optimal avec Hungarian
        row_idx, col_idx = linear_sum_assignment(dist_matrix)

        # Palette réordonnée selon matching
        rgb_colors = palette_rgb[col_idx]
        cluster_names_matched = [cluster_names_palette[j] for j in col_idx]

        print("Distances entre centres et palette :")
        print(np.round(dist_matrix, 3))

        print("Correspondance clusters → palette :")
        for ci, pi in zip(row_idx, col_idx):
            print(f"  Cluster {ci} → {cluster_names_palette[pi]} (distance {dist_matrix[ci, pi]:.3f})")

        # Image segmentée avec palette assignée
        segmented_image = np.zeros_like(image)
        for cluster_idx in range(k):
            segmented_image[labels.reshape(image.shape[:2]) == cluster_idx] = rgb_colors[cluster_idx]

        # Morphologie circulaire sur labels
        filtered_labels = np.zeros_like(labels)
        for cluster_idx in range(k):
            mask = (labels == cluster_idx).reshape(image.shape[:2])
            mask_open = binary_opening(mask, structure=structure)
            mask_clean = binary_closing(mask_open, structure=structure)
            filtered_labels[mask_clean.flatten()] = cluster_idx

        filtered_segmented_image = np.zeros_like(image)
        for cluster_idx in range(k):
            filtered_segmented_image[filtered_labels.reshape(image.shape[:2]) == cluster_idx] = rgb_colors[cluster_idx]

        # === Sauvegarde image filtrée couleur ===
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        Image.fromarray(filtered_segmented_image).save(save_image_path)
        print(f"📸 Image filtrée enregistrée : {save_image_path}")

        # === Sauvegarde raster des labels cluster (TIFF) ===
        tiff_labels_path = os.path.splitext(save_image_path)[0] + "_labels.tif"
        height, width = image.shape[:2]
        transform = from_origin(0, 0, 1, 1)  # adapter selon géoréférencement si besoin

        with rasterio.open(
            tiff_labels_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.uint8,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(filtered_labels.reshape(height, width).astype(rasterio.uint8), 1)

        print(f"📊 Raster labels sauvegardé : {tiff_labels_path}")

        # Affichage
        display_with_legend(image, segmented_image, filtered_segmented_image, rgb_colors, cluster_names_matched, title=save_figure_path)

    except Exception as e:
        print(f"❌ Erreur pour {image_path} : {e}")

# === BOUCLE SUR LES IMAGES ===
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(image_path, root_dir)
            save_image_path = os.path.join(output_dir, rel_path)
            save_figure_path = os.path.splitext(save_image_path)[0] + '_legend.png'
            process_and_save_image(image_path, save_image_path, save_figure_path)
