# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:08:23 2024

@author: alber
"""

import nibabel as nib
import numpy as np
from skimage import measure
from mayavi import mlab
from matplotlib import cm

def visualizza_file_nifti_3d(file_path):
    # Carica il file NIfTI
    img = nib.load(file_path)
    
    # Estrai i dati e le dimensioni dei voxel
    data = img.get_fdata()
    header = img.header
    voxel_dimensions = header.get_zooms()
    
    # Definisci i colori per i vari organi (13 classi possibili)
    colors = cm.rainbow(np.linspace(0, 1, 13))
    
    # Visualizzazione 3D con Mayavi
    unique_labels = np.unique(data)  # Trova le etichette presenti nel volume
    for label in unique_labels:
        if label == 0:  # Ignora il background
            continue
        
        # Estrai il blob corrispondente alla classe corrente
        blob = (data == label)
        verts, faces, _, _ = measure.marching_cubes(blob, level=0)
        
        # Adatta le coordinate dei vertici in base alle dimensioni dei voxel
        verts[:, 0] *= voxel_dimensions[0]
        verts[:, 1] *= voxel_dimensions[1]
        verts[:, 2] *= voxel_dimensions[2]
        
        # Colore corrispondente alla classe (etichette da 1 a 13)
        color = tuple(colors[int(label) - 1][:3])  # I colori partono da indice 0
        
        # Disegna la mesh triangolare con il colore specifico
        mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, color=color)
    
    # Mostra la visualizzazione
    mlab.show()

# Esempio di utilizzo
visualizza_file_nifti_3d('output_classification.nii.gz')
