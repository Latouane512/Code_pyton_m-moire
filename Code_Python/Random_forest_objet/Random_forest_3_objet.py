# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:30:58 2025

@author: ASUS
"""

import os
import numpy as np
import rasterio
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2hsv, rgb2lab
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import collections

# === PARAMÈTRES UTILISATEUR ===
color_space = "RGB"  # Choix possibles : "RGB", "LAB", "HSV"

image_path = "C:/Users/ASUS/Desktop/Point_ponctuel_bon/LAUT01_B1_20240523/DJI_20240523114851_0117_V1.JPG" # Image de base
label_path = ""    # Digitalisation
output_dir = "C:/Users/ASUS/Desktop/Couche_ref/A_RASTER_prediction_objet/LAT01_B1/3" # Chemin fichier enregistrement
os.makedirs(output_dir, exist_ok=True)

output_class_raster = os.path.join(output_dir, "classification_obia_RGB_3.tif")
output_color_raster = os.path.join(output_dir, "classification_colored_RGB_3.tif")



# === CHARGEMENT IMAGE RGB ===
with rasterio.open(image_path) as src:
    image = src.read([1, 2, 3])  # (3, H, W)
    profile = src.profile
    height, width = src.height, src.width

image = np.moveaxis(image, 0, -1)  # (H, W, 3)
image_float = img_as_float(image)

# === CHARGEMENT RASTER LABEL ===
with rasterio.open(label_path) as src:
    labels_raster = src.read(1)

# === VÉRIFICATION DIMENSIONS ===
if (image.shape[0] != labels_raster.shape[0]) or (image.shape[1] != labels_raster.shape[1]):
    raise ValueError(
        f"Erreur : dimensions différentes entre image ({image.shape[0]}x{image.shape[1]}) "
        f"et raster labels ({labels_raster.shape[0]}x{labels_raster.shape[1]})"
    )
else:
    print(f"[OK] Dimensions validées : {image.shape[0]}x{image.shape[1]}")

print("Classes présentes dans le raster de vérité terrain :", np.unique(labels_raster))

# === SEGMENTATION SLIC ===
segments = slic(image_float, n_segments=4000, compactness=10, start_label=1)
num_segments = segments.max()
print(f"[INFO] Nombre de segments : {num_segments}")

# === CONVERSION COLORIMÉTRIQUE ===
if color_space == "HSV":
    image_space = rgb2hsv(image_float)
elif color_space == "LAB":
    image_space = rgb2lab(image_float)
elif color_space == "RGB":
    image_space = image_float
else:
    raise ValueError("color_space doit être 'RGB', 'HSV' ou 'LAB'")

# === EXTRACTION DES FEATURES ET LABELS PAR SEGMENT ===
features = []
segment_labels = []

for seg_id in range(1, num_segments + 1):
    mask = segments == seg_id
    values = image_space[mask]
    feature_vector = np.mean(values, axis=0)
    features.append(feature_vector)

    label_mask = labels_raster[mask]
    vals, counts = np.unique(label_mask, return_counts=True)
    majority_label = vals[np.argmax(counts)]
    segment_labels.append(majority_label)

features = np.array(features)
segment_labels = np.array(segment_labels)

print("Distribution des labels par segment :", collections.Counter(segment_labels))

# === ENTRAÎNEMENT DU CLASSIFIEUR ===
train_mask = segment_labels >= 0  # On inclut toutes les classes, y compris 0
X_train = features[train_mask]
y_train = segment_labels[train_mask]

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_enc)
print(f"[OK] Random Forest entraîné avec {len(X_train)} échantillons.")

# === PRÉDICTION SUR TOUS LES SEGMENTS ===
y_pred_enc = clf.predict(features)
y_pred = encoder.inverse_transform(y_pred_enc)

# === RECONSTITUTION RASTER PRÉDIT ===
pred = np.zeros_like(labels_raster)
for seg_id, class_id in zip(range(1, num_segments + 1), y_pred):
    pred[segments == seg_id] = class_id

# === SAUVEGARDE RASTER CLASSIFIÉ ===
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(output_class_raster, "w", **profile) as dst:
    dst.write(pred.astype(rasterio.uint8), 1)
print(f"[OK] Raster classifié sauvegardé : {output_class_raster}")

# === PALETTE FIXE ===
color_map = {
    0: [194, 178, 128],  # sable
    1: [34, 139, 34],    # végétation aquatique
    2: [128, 128, 128]   # substrat minéral
}

colored = np.zeros((height, width, 3), dtype=np.uint8)
for class_id, color in color_map.items():
    colored[pred == class_id] = color

# === SAUVEGARDE RASTER COLORÉ ===
profile.update(count=3, dtype=rasterio.uint8)
with rasterio.open(output_color_raster, "w", **profile) as dst:
    for i in range(3):
        dst.write(colored[:, :, i], i + 1)
print(f"[OK] Image colorée sauvegardée : {output_color_raster}")
