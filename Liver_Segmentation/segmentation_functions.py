import numpy as np
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.segmentation import flood, find_boundaries
from skimage.morphology import binary_dilation
import cv2
from collections import deque

def watershed_segmentation(slice_data, tolerance=(90, 200), filter_size=4):
    # Filtro mediano
    filtered_image = median_filter(slice_data, size=filter_size)

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

def exploration_segmentation(slice_data, label_slice, tolerance=(90, 200), filter_size=4, max_voxel_exploration=100000):
    # Filtro mediano
    filtered_image = median_filter(slice_data, size=filter_size)
    mask2 = (filtered_image > tolerance[0]) & (filtered_image < tolerance[1])

    # BFS per propagare la maschera
    new_mask = np.zeros_like(slice_data, dtype=np.uint8)
    vis = np.zeros_like(slice_data, dtype=bool)

    def bfs(seeds, vis):
        queue = deque()
        for seed in seeds:
            x, y = seed
            queue.append((x, y))
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
                if (0 <= nx < slice_data.shape[0] and 0 <= ny < slice_data.shape[1] and not vis[nx, ny] and
                    mask2[nx, ny]):
                    vis[nx, ny] = True
                    new_mask[nx, ny] = 1
                    queue.append((nx, ny))
                    explored_voxels += 1

    # Select a random seed point from the true label mask (red one)
    seed_points = np.array(np.where(label_slice)).T
    
    if len(seed_points) == 0:
        raise ValueError("Nessun seed point trovato nella maschera vera.")

    random_seeds =[]
    for i in range(3):
        random_seeds.append(seed_points[np.random.choice(seed_points.shape[0])])
    bfs(random_seeds, vis)
    """# Chiudi tutti i buchi nella maschera finale
    import matplotlib.pyplot as plt

    # Plot the original slice data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Slice Data")
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')

    # Plot the mask before filling holes
    plt.subplot(1, 3, 2)
    plt.title("Mask Before Filling Holes")
    plt.imshow(new_mask, cmap='gray')
    plt.axis('off')"""

    # Chiudi tutti i buchi nella maschera finale
    filled_mask = binary_fill_holes(new_mask)

    # Plot the filled mask
    """plt.subplot(1, 3, 3)
    plt.title("Filled Mask After Segmentation")
    plt.imshow(filled_mask, cmap='gray')
    plt.axis('off')

    plt.show()"""
    return filled_mask
    

def flood_segmentation(slice_data, label_slice, tolerance=(90, 200), tolerance_flood=30, filter_size=4):

    filtered_image = median_filter(slice_data, size=filter_size)

    filtered_image = np.where((filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1]), filtered_image, -1000)

    mask2 = (filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1])

    # Trova i contorni
    boundaries = find_boundaries(mask2, mode='inner')

    # Dilata i contorni per creare una maschera di esclusione
    exclusion_mask = binary_dilation(boundaries)

    # Flood fill per propagare la maschera senza oltrepassare i contorni
    new_mask = np.zeros_like(slice_data, dtype=np.uint8)

    # Select a random seed point from the true label mask (red one)
    seed_points = np.array(np.where(label_slice)).T
    
    if len(seed_points) == 0:
        raise ValueError("Nessun seed point trovato nella maschera vera.")

    random_seed = seed_points[np.random.choice(seed_points.shape[0])]
    flooded = flood(filtered_image, (random_seed[0], random_seed[1]), tolerance=tolerance_flood)

    # Applica la maschera di esclusione
    new_mask = np.where(exclusion_mask, 0, flooded)

    # Applica binary filling alla maschera trovata
    filled_mask = binary_fill_holes(new_mask)

    return filled_mask
