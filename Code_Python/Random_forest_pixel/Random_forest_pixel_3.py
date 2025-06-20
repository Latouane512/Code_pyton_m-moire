# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 20:56:14 2025

@author: ASUS
"""

import numpy as np
import cv2
import rasterio
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import time

# === 1. Paramètres ===
color_space = "HSV"  # Choix possible : "RGB", "LAB", "HSV"
image_path = "C:/Users/ASUS/Desktop/Point_ponctuel_bon/LAUT01_A1_20240523/DJI_20240523115611_0337_V1.JPG"
label_path = "C:/Users/ASUS/Desktop/Couche_ref/A_RASTER/RASTER_3_LAT01_A1.tif"
output_dir = "C:/Users/ASUS/Desktop/Couche_ref/A_RASTER_prediction/LAT01_A1"
os.makedirs(output_dir, exist_ok=True)

# === 2. Charger l'image RGB ===
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
print(f"Image RGB chargée : {width} x {height}")

# === 3. Conversion d'espace colorimétrique ===
if color_space == "LAB":
    image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
elif color_space == "HSV":
    image_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
else:
    image_converted = image.copy()
print(f"Image convertie en espace colorimétrique : {color_space}")

# === 4. Charger le raster des labels (.tif) ===
with rasterio.open(label_path) as src_lbl:
    labels = src_lbl.read(1)
    profile_lbl = src_lbl.profile

if labels.shape != (height, width):
    raise ValueError("L'image et le raster des labels n'ont pas la même taille !")

# === 5. Préparation des données ===
X = image_converted.reshape(-1, 3)
y = labels.flatten()
mask = np.isin(y, [0, 1, 2])
X_train = X[mask]
y_train = y[mask]
print(f"Nombre de pixels d'entraînement : {len(y_train)}")

# === 6. Entraînement Random Forest ===
start = time.time()
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
print(f"Modèle entraîné en {time.time() - start:.2f} sec.")

# === 7. Prédiction ===
y_pred = clf.predict(X)
pred = y_pred.reshape((height, width))

# === 8. Génération image couleur depuis prédictions ===
color_map = {
    0: [194, 178, 128],   # sable
    1: [34, 139, 34],     # végétation aquatique
    2: [128, 128, 128]    # substrat minéral
}
colored = np.zeros((height, width, 3), dtype=np.uint8)
for class_id, color in color_map.items():
    colored[pred == class_id] = color

# === 9. Affichage ===
plt.figure(figsize=(12, 8))
plt.imshow(colored)
plt.title(f"Prédiction Random Forest ({color_space})")
plt.axis("off")
plt.tight_layout()
plt.show()

# === 10. Sauvegarde ===
pred_uint8 = pred.astype(np.uint8)
colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

save_rgb = cv2.imwrite(os.path.join(output_dir, "prediction_HSV_3.png"), colored_bgr)
save_labels = cv2.imwrite(os.path.join(output_dir, "prediction_labels_HSV_3.png"), pred_uint8)

print("Sauvegarde 'prediction_rgb.png' :", "OK" if save_rgb else "ÉCHEC")
print("Sauvegarde 'prediction_labels.png' :", "OK" if save_labels else "ÉCHEC")
