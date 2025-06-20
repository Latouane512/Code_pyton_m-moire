# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:43:49 2025

@author: ASUS
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# === PARAMÈTRES ===
prediction_raster = "C:/Users/ASUS/Desktop/Dossier_complet_TER/Image_segment/Random_forest_objet/LAT01_A1/4/classification_obia_RGB_4.tif"
ground_truth_raster = "C:/Users/ASUS/Desktop/Couche_ref/A_RASTER/RASTER_4_LAT01_A1.tif"
rapport_txt_path = "C:/Users/ASUS/Desktop/Couche_ref/TESTE/RANDOM_objet/LAT01_A1/4/rapport_classification_4_RGB.txt"
matrice_csv_path = "C:/Users/ASUS/Desktop/Couche_ref/TESTE//RANDOM_objet/LAT01_A1/4/matrice_confusion_4_RGB.csv"

os.makedirs(os.path.dirname(rapport_txt_path), exist_ok=True)
os.makedirs(os.path.dirname(matrice_csv_path), exist_ok=True)

# === LECTURE DES RASTERS ===
with rasterio.open(prediction_raster) as pred_src:
    pred = pred_src.read(1)

with rasterio.open(ground_truth_raster) as gt_src:
    gt = gt_src.read(1)

# === AFFICHAGE DES CLASSES UNIQUES
print("Valeurs uniques - prédiction :", np.unique(pred))
print("Valeurs uniques - vérité terrain :", np.unique(gt))

# === MASQUE DES PIXELS VALABLES ===
mask = (gt >= 0) & (pred >= 0)
y_true = gt[mask].flatten()
y_pred = pred[mask].flatten()

# === LISTE DES CLASSES À ÉVALUER ===
labels = [0, 1, 2,3]


# === MATRICE DE CONFUSION ===
cm = confusion_matrix(y_true, y_pred, labels=labels)

# === AFFICHAGE HEATMAP ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.show()

# === RAPPORT DE CLASSIFICATION
report = classification_report(y_true, y_pred, labels=labels)
accuracy = accuracy_score(y_true, y_pred)

# === SAUVEGARDE DU RAPPORT TEXTE ===
with open(rapport_txt_path, "w", encoding="utf-8") as f:
    f.write("=== Rapport de classification ===\n\n")
    f.write(report)
    f.write("\n")
    f.write(f"Taux de réussite global (accuracy) : {accuracy:.2%}\n")

print(f"[OK] Rapport de classification sauvegardé : {rapport_txt_path}")

# === SAUVEGARDE MATRICE DE CONFUSION EN CSV ===
cm_df = pd.DataFrame(cm, index=[f"Réel_{l}" for l in labels],
                         columns=[f"Prédit_{l}" for l in labels])
cm_df.to_csv(matrice_csv_path)
print(f"[OK] Matrice de confusion exportée : {matrice_csv_path}")
