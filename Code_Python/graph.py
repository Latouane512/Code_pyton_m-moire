# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:42:38 2025

@author: ASUS
"""

import matplotlib.pyplot as plt

# Données
color_spaces = ['LAB', 'RGB', 'HSV']
x = [1, 2, 3]

A1_3 = [40.47, 38.95, 38.21]
A1_4 = [25.54, 24.73, 28.03]
B1_3 = [46.64, 9.59, 38.75]
B1_4 = [33.46, 7.49, 13.81]

# Couleurs flashy
colors = ['#ff1493', '#da70d6', '#ff69b4', '#c71585']

plt.figure(figsize=(10, 6))
plt.plot(x, A1_3, marker='o', label='Lat01_A1 - 3 classes', color=colors[0], linewidth=2)
plt.plot(x, A1_4, marker='o', label='Lat01_A1 - 4 classes', color=colors[1], linewidth=2)
plt.plot(x, B1_3, marker='o', label='Lat01_B1 - 3 classes', color=colors[2], linewidth=2)
plt.plot(x, B1_4, marker='o', label='Lat01_B1 - 4 classes', color=colors[3], linewidth=2)

plt.xticks(x, color_spaces)
plt.ylabel('Précision globale (%)')
plt.ylim(0, 55)
plt.title("Précision globale des classifications SLIC (3 et 4 classes)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
