import nibabel as nib
import numpy as np
from skimage import measure
from skimage.transform import resize
from mayavi import mlab
from scipy.stats import norm
from matplotlib import cm

# Sostituisci con il percorso del tuo file .nii.gz
file_path = 'FLARE22Train/images/FLARE22_Tr_0001_0000.nii.gz'
label_path = 'FLARE22Train/labels/FLARE22_Tr_0001.nii.gz'

# Carica il file NIfTI
img = nib.load(file_path)
lbl = nib.load(label_path)

# Estrai i dati
data = img.get_fdata()
labeled_volume = lbl.get_fdata()

# Ottieni informazioni sulla dimensione dei voxel
header = img.header
voxel_dimensions = header.get_zooms()
print(f"Dimensioni del voxel (x, y, z): {voxel_dimensions}")

# Ridimensiona i dati se necessario per correggere il rapporto dei voxel
desired_shape = (data.shape[0], data.shape[1], int(data.shape[2] * voxel_dimensions[2] / voxel_dimensions[0]))
rescaled_data = resize(data, desired_shape, mode='constant')

nomi = ["liver", "right kidney", "spleen", "pancreas", "aorta", "stomach", "left kidney"]
indici_organi = [1, 2, 3, 4, 5, 11, 13]  # liver, right kidney, spleen, pancreas, aorta, stomach, left kidney

# Precalcola le gaussiane per HU e assi spaziali per ogni organo
organi_gaussiane = {}
for idx, blob_idx in enumerate(indici_organi):  
    blob = (labeled_volume == blob_idx)
    coords = np.array(np.where(blob)).T  # Ottieni le coordinate (x, y, z)
    hu_values = data[blob]
    
    gauss_hu = norm.fit(hu_values)
    gauss_x = norm.fit(coords[:, 0] * voxel_dimensions[0])
    gauss_y = norm.fit(coords[:, 1] * voxel_dimensions[1])
    gauss_z = norm.fit(coords[:, 2] * voxel_dimensions[2])
    
    organi_gaussiane[nomi[idx]] = {
        'HU': gauss_hu,
        'x': gauss_x,
        'y': gauss_y,
        'z': gauss_z
    }

# Griglie di coordinate con la stessa forma del volume
x_grid = np.linspace(0, data.shape[0] * voxel_dimensions[0], data.shape[0])
y_grid = np.linspace(0, data.shape[1] * voxel_dimensions[1], data.shape[1])
z_grid = np.linspace(0, data.shape[2] * voxel_dimensions[2], data.shape[2])

precalculated_probs = {}

# Precalcola le gaussiane spaziali per ogni organo
for idx, organo in enumerate(nomi):
    gauss_x_params = organi_gaussiane[organo]['x']
    gauss_y_params = organi_gaussiane[organo]['y']
    gauss_z_params = organi_gaussiane[organo]['z']
    
    # Calcola le probabilità gaussiane per tutte le coordinate spaziali
    p_x = norm.pdf(x_grid, loc=gauss_x_params[0], scale=gauss_x_params[1])[:, None, None]
    p_y = norm.pdf(y_grid, loc=gauss_y_params[0], scale=gauss_y_params[1])[None, :, None]
    p_z = norm.pdf(z_grid, loc=gauss_z_params[0], scale=gauss_z_params[1])[None, None, :]
    
    # Salva le probabilità precalcolate
    precalculated_probs[organo] = {'p_x': p_x, 'p_y': p_y, 'p_z': p_z}

# Classificazione basata sulle gaussiane precalcolate
classification = np.zeros(data.shape)
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        for z in range(data.shape[2]):
            if data[x, y, z] <= 0:
                classification[x, y, z] = 0
                continue
            probs = np.zeros(len(indici_organi))
            for i, organo in enumerate(nomi):
                gauss_hu_params = organi_gaussiane[organo]['HU']
                
                # Usa le probabilità gaussiane precalcolate
                p_hu = norm.pdf(data[x, y, z], loc=gauss_hu_params[0], scale=gauss_hu_params[1])
                p_x = precalculated_probs[organo]['p_x'][x,0,0]
                p_y = precalculated_probs[organo]['p_y'][0,y,0]
                p_z = precalculated_probs[organo]['p_z'][0,0,z]
                
                # Calcola la probabilità totale moltiplicando le gaussiane
                probs[i] = p_hu * p_x * p_y #* p_z
            
            max_organo = np.argmax(probs)
            max_prob = probs[max_organo]
            if max_prob > (0.01 ** 4):
                classification[x, y, z] = indici_organi[max_organo]


# Visualizzazione 3D con Mayavi
"""colors = cm.rainbow(np.linspace(0, 1, 13))
#classification=labeled_volume
for idx, blob_idx in enumerate(indici_organi):
    blob = (classification == blob_idx)
    verts, faces, _, _ = measure.marching_cubes(blob, level=0)
    
    verts[:, 0] *= voxel_dimensions[0]
    verts[:, 1] *= voxel_dimensions[1]
    verts[:, 2] *= voxel_dimensions[2]
    
    color = tuple(colors[blob_idx - 1][:3])
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, color=color)

mlab.show()"""

classified_img = nib.Nifti1Image(classification, img.affine, header=img.header)

# Sostituisci con il percorso dove desideri salvare il file
output_path = 'FLARE22Train/output_classification.nii.gz'

# Salva l'immagine NIfTI
nib.save(classified_img, output_path)

print(f"Classificazione salvata con successo in {output_path}")
