# Classification d'images UAV — K-Means et Random Forest (Pixel et OBIA)

Ce projet Python porte sur la classification non supervisée et supervisée d’images drone, en utilisant différentes approches : K-Means, SLIC K-Means, Random Forest pixel par pixel, et Random Forest orientée-objet (OBIA).

## 📂 Contenu du projet

Le projet comprend plusieurs scripts et traitements réalisés sur des images UAV :

### 🔹 Classification non supervisée
- **K-Means classique** sur image RGB, HSV, ou LAB
- **K-Means après segmentation SLIC** (sur-segmentation en super-pixels)

### 🔹 Classification supervisée
- **Random Forest pixel par pixel** à partir d’un raster de labels (vérité terrain)
- **Random Forest orientée-objet (OBIA)** :
  - Segmentation avec SLIC
  - Extraction des caractéristiques d’objets (super-pixels)
  - Apprentissage et classification avec Random Forest

## 🧪 Objectifs du projet

- Comparer différentes méthodes de classification sur la base d’images UAV
- Étudier l’influence de l’espace colorimétrique (RGB, HSV, LAB)
- Évaluer les performances via matrices de confusion (si données de validation disponibles)
- Explorer les apports de l’approche orientée-objet par rapport à l’approche pixel

## 🛠️ Technologies et bibliothèques utilisées

- Python 3
- `scikit-learn` (KMeans, RandomForestClassifier)
- `scikit-image` (SLIC)
- `numpy`, `matplotlib`, `pandas`
- `rasterio`, `opencv-python`, `skimage`

