import numpy as np
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.segmentation import flood, find_boundaries
from skimage.morphology import binary_dilation
import cv2
from collections import deque

def watershed_segmentation(image_data, label_data, current_slice, tolerance, filter_size):
    # Filtro mediano
    filtered_image = median_filter(image_data[:,:,current_slice], size=filter_size)

    # Crea una maschera con 1 solo nei pixel nel range di tolleranza
    mask = (filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1])

    # Standardizza senza contare i pixel fuori dalla maschera
    standardized_filtered_image = np.where(mask, (filtered_image - np.mean(filtered_image[mask])) / np.std(filtered_image[mask]), -1000)

    # Normalizzazione in [0, 255]
    grayscale_image = ((standardized_filtered_image - np.min(standardized_filtered_image)) / (np.max(standardized_filtered_image) - np.min(standardized_filtered_image)) * 255).astype(np.uint8)
    ret, bin_img = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=1)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    color_img = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    n_mark = np.max(markers)
    vettore_aree = [(np.sum(markers == i), i) for i in range(n_mark + 1)]
    coppia = max(vettore_aree, key=lambda x: x[0])
    
    new_mask = np.ones(markers.shape)
    new_mask[markers == coppia[1]] = 0
    # Keep only the largest connected component
    num_labels, labels_im = cv2.connectedComponents(new_mask.astype(np.uint8))
    sizes = np.bincount(labels_im.ravel())
    sizes[0] = 0  # Background size to zero
    largest_label = sizes.argmax()
    new_mask = (labels_im == largest_label).astype(np.uint8)

    # Chiudi tutti i buchi nella maschera finale
    filled_mask = binary_fill_holes(new_mask)

    return filled_mask

def exploration_segmentation(image_data, label_data, current_slice, tolerance, filter_size, max_voxel_exploration):
    # Filtro mediano
    filtered_image = median_filter(image_data[:,:,current_slice], size=filter_size)
    mask2 = (filtered_image > tolerance[0]) & (filtered_image < tolerance[1])

    # BFS per propagare la maschera
    new_mask = np.zeros_like(image_data[:,:,current_slice], dtype=np.uint8)
    vis = np.zeros_like(image_data[:,:,current_slice], dtype=bool)

    def bfs(x, y, vis):
        queue = deque([(x, y)])
        vis[x, y] = True
        new_mask[x, y] = 1
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        explored_voxels = 0

        while queue:
            if explored_voxels > max_voxel_exploration:
                break

            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < image_data.shape[0] and 0 <= ny < image_data.shape[1] and not vis[nx, ny] and
                    tolerance[0] <= filtered_image[nx, ny] <= tolerance[1]):
                    vis[nx, ny] = True
                    new_mask[nx, ny] = 1
                    queue.append((nx, ny))
                    explored_voxels += 1

    # Select a random seed point from the true label mask (red one)
    seed_points = np.array(np.where(label_data[:,:,current_slice])).T
    
    if len(seed_points) == 0:
        raise ValueError("Nessun seed point trovato nella maschera vera.")

    random_seed = seed_points[np.random.choice(seed_points.shape[0])]
    bfs(random_seed[0], random_seed[1], vis)

    return new_mask

def flood_segmentation(image_data, label_data, current_slice, tolerance_flood, tolerance_boundaries, filter_size, range_min, range_max):
    # Applica il filtro per impostare i pixel fuori dal range a -1000
    filtered_image = np.where((image_data[:,:,current_slice] >= range_min) & (image_data[:,:,current_slice] <= range_max), image_data[:,:,current_slice], -1000)

    # Standardizza dopo il filtro nel range [-1000, 3000]
    standardized_filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 4000 - 1000

    # Filtro mediano
    filtered_image = median_filter(standardized_filtered_image, size=filter_size)

    mask2 = filtered_image > tolerance_boundaries

    # Trova i contorni
    boundaries = find_boundaries(mask2, mode='inner')

    # Dilata i contorni per creare una maschera di esclusione
    exclusion_mask = binary_dilation(boundaries)

    # Flood fill per propagare la maschera senza oltrepassare i contorni
    new_mask = np.zeros_like(image_data[:,:,current_slice], dtype=np.uint8)

    # Select a random seed point from the true label mask (red one)
    seed_points = np.array(np.where(label_data[:,:,current_slice])).T
    
    if len(seed_points) == 0:
        raise ValueError("Nessun seed point trovato nella maschera vera.")

    random_seed = seed_points[np.random.choice(seed_points.shape[0])]
    flooded = flood(filtered_image, (random_seed[0], random_seed[1]), tolerance=tolerance_flood)

    # Applica la maschera di esclusione
    new_mask = np.where(exclusion_mask, 0, flooded)

    # Applica binary filling alla maschera trovata
    filled_mask = binary_fill_holes(new_mask)

    return filled_mask
