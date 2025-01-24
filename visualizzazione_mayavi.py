import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import measure, morphology
from skimage.transform import resize
from scipy.ndimage import binary_closing, binary_opening
from scipy import ndimage
from mayavi import mlab
from skimage import io, filters, exposure
from skimage.restoration import denoise_bilateral
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


def plot_organs_mask(organs_mask, slice_axis=2, animated=False):
    """
    Plotta il volume `organs_mask` mostrando sezioni lungo l'asse specificato.
    
    Args:
        organs_mask (numpy.ndarray): Volume binario da visualizzare.
        slice_axis (int): Asse lungo il quale effettuare le sezioni (0, 1 o 2).
        animated (bool): Se True, mostra un'animazione di tutte le sezioni.
    """
    if not animated:
        # Plotta una singola slice (centrale)
        slice_index = organs_mask.shape[slice_axis] // 2
        plt.figure(figsize=(8, 8))
        if slice_axis == 0:
            slice_data = organs_mask[slice_index, :, :]
        elif slice_axis == 1:
            slice_data = organs_mask[:, slice_index, :]
        else:
            slice_data = organs_mask[:, :, slice_index]
        
        plt.imshow(slice_data, cmap='gray')
        plt.title(f"Slice lungo l'asse {slice_axis}, indice {slice_index}")
        plt.axis('off')
        plt.show()
    else:
        # Crea un'animazione delle slice
        fig, ax = plt.subplots(figsize=(8, 8))
        slice_range = organs_mask.shape[slice_axis]
        
        def update(frame):
            ax.clear()
            if slice_axis == 0:
                slice_data = organs_mask[frame, :, :]
            elif slice_axis == 1:
                slice_data = organs_mask[:, frame, :]
            else:
                slice_data = organs_mask[:, :, frame]
            
            ax.imshow(slice_data, cmap='gray')
            ax.set_title(f"Slice {frame} lungo l'asse {slice_axis}")
            ax.axis('off')
        
        # Assegna l'animazione a una variabile
        anim = FuncAnimation(fig, update, frames=slice_range, interval=100)
        plt.show()

        # Ritorna l'oggetto animazione (utile se vuoi salvarlo)
        return anim
    
    
def filtra_polmoni():
    organs_mask = (data_resized < -600) & (data_resized > -1100)
    animation = plot_organs_mask(organs_mask, slice_axis=2, animated=True)  # Visualizza un'animazione

    
    # Applica prima una chiusura morfologica per eliminare piccoli buchi
    selem = morphology.ball(4)  # Elemento strutturante sferico di raggio 3
    organs_mask_closed = binary_closing(organs_mask, structure=selem)

    # Applica un'apertura morfologica per rimuovere piccole protuberanze
    organs_mask_cleaned = binary_opening(organs_mask_closed, structure=selem)

    labeled_volume, num_features = ndimage.label(organs_mask_cleaned)
    print(f"Numero di blob trovati: {num_features}")

    # Genera la mappa di colori per distinguere i blob
    colors = cm.rainbow(np.linspace(0, 1, num_features))

    """mlab.figure(bgcolor=(1, 1, 1))
    blob_properties = measure.regionprops(labeled_volume)

    filtrati = []
    for i, prop in enumerate(blob_properties):
        if 200000 < prop.area < 700000:
            filtrati.append(i + 1)

    for blob_idx in filtrati:
        blob = (labeled_volume == blob_idx)
        verts, faces, _, _ = measure.marching_cubes(blob, level=0.5)

        # Scala le coordinate dei vertici in base ai voxel
        verts[:, 0] *= voxel_dimensions[0]
        verts[:, 1] *= voxel_dimensions[1]
        verts[:, 2] *= voxel_dimensions[2]

        # Colore per il blob corrente
        color = tuple(colors[blob_idx - 1][:3])
        mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, color=color)

    mlab.axes(extent=[0, data.shape[0] * voxel_dimensions[0],
                      0, data.shape[1] * voxel_dimensions[1],
                      0, data.shape[2] * voxel_dimensions[2]])
    mlab.show()"""




# Carica il file .nii
img = nib.load('estratto.nii')
data = img.get_fdata()
# Ottieni informazioni sulla dimensione dei voxel
header = img.header
voxel_dimensions = header.get_zooms()


data_resized=data

filtra_polmoni()

