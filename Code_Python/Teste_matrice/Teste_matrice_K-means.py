# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:17:21 2025

@author: ASUS
"""

import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# === CHEMINS ===
prediction_labels_tiff = "C:/Users/ASUS/Desktop/Dossier_complet_TER/Image_segment/Kmean_superpixel_slic/3/Out_segmentation_HSV_3_prediction_teste/LAUT01_B1_20240523/DJI_20240523114851_0117_V1.jpg"  # le TIFF sauvegardé précédemment
ground_truth_raster = "C:/Users/ASUS/Desktop/Couche_ref/A_RASTER/RASTER_3_LAT01_B1.tif"
rapport_txt_path = "C:/Users/ASUS/Desktop/Couche_ref/TESTE/K-mean_superpixel_slic/LAT01_B1/rapport_classification_3_HSV.txt"
matrice_csv_path = "C:/Users/ASUS/Desktop/Couche_ref/TESTE/K-mean_superpixel_slic/LAT01_B1/matrice_confusion_3_HSV.csv"

os.makedirs(os.path.dirname(rapport_txt_path), exist_ok=True)
os.makedirs(os.path.dirname(matrice_csv_path), exist_ok=True)

# === LECTURE DES RASTERS ===
with rasterio.open(prediction_labels_tiff) as pred_src:
    pred = pred_src.read(1)

with rasterio.open(ground_truth_raster) as gt_src:
    gt = gt_src.read(1)

# === VÉRIFICATION DES DIMENSIONS ===
if pred.shape != gt.shape:
    raise ValueError(f"Dimensions différentes : prédiction = {pred.shape}, vérité terrain = {gt.shape}")

# === MASQUE DES PIXELS VALABLES ===
mask = (gt >= 0) & (pred >= 0)
y_true = gt[mask].flatten()
y_pred = pred[mask].flatten()

# === LISTE DES CLASSES ===
labels = sorted(np.unique(np.concatenate((y_true, y_pred))))

# === MATRICE DE CONFUSION ===
cm = confusion_matrix(y_true, y_pred, labels=labels)

# === AFFICHAGE MATRICE DE CONFUSION ===
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.show()

# === RAPPORT DE CLASSIFICATION ===
report = classification_report(y_true, y_pred, labels=labels)
accuracy = accuracy_score(y_true, y_pred)

# === SAUVEGARDE DU RAPPORT ===
with open(rapport_txt_path, "w", encoding="utf-8") as f:
    f.write("=== Rapport de classification ===\n\n")
    f.write(report)
    f.write(f"\nTaux de réussite global (accuracy) : {accuracy:.2%}\n")

print(f"[OK] Rapport sauvegardé : {rapport_txt_path}")

# === SAUVEGARDE MATRICE DE CONFUSION EN CSV ===
cm_df = pd.DataFrame(cm, index=[f"Réel_{l}" for l in labels], columns=[f"Prédit_{l}" for l in labels])
cm_df.to_csv(matrice_csv_path)
print(f"[OK] Matrice confusion exportée : {matrice_csv_path}")
