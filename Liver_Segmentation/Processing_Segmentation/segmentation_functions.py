import numpy as np
from scipy.ndimage import median_filter, binary_fill_holes
from skimage.segmentation import flood, find_boundaries
from skimage.morphology import binary_dilation
import cv2
from collections import deque

def watershed_segmentation(slice_data, tolerance=(90, 200), filter_size=4):
    # Apply median filter to the input slice data
    filtered_image = median_filter(slice_data, size=filter_size)

    # Create a mask with pixels within the tolerance range
    mask = (filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1])

    # Standardize the filtered image without considering pixels outside the mask
    standardized_filtered_image = np.where(mask, (filtered_image - np.mean(filtered_image[mask])) / np.std(filtered_image[mask]), -1000)

    # Normalize the standardized image to [0, 255]
    grayscale_image = ((standardized_filtered_image - np.min(standardized_filtered_image)) / (np.max(standardized_filtered_image) - np.min(standardized_filtered_image)) * 255).astype(np.uint8)
    
    # Apply binary thresholding
    ret, bin_img = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY)
    
    # Fill holes inside objects
    bin_img = binary_fill_holes(bin_img).astype(np.uint8) * 255

    # Morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=1)
    
    # Distance transform and thresholding to find sure foreground
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Identify unknown regions
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    color_img = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    # Find the largest region
    n_mark = np.max(markers)
    vettore_aree = [(np.sum(markers == i), i) for i in range(n_mark + 1)]
    coppia = max(vettore_aree, key=lambda x: x[0])
    
    # Create a new mask for the largest region
    new_mask = np.ones(markers.shape)
    new_mask[markers == coppia[1]] = 0
    
    # Find the largest connected component in the new mask
    num_labels, labels_im = cv2.connectedComponents(new_mask.astype(np.uint8))
    sizes = np.bincount(labels_im.ravel())
    sizes[0] = 0  # Set background size to zero
    largest_label = sizes.argmax()
    new_mask = (labels_im == largest_label).astype(np.uint8)

    return new_mask

def exploration_segmentation(slice_data, label_slice, tolerance=(90, 200), filter_size=4, max_voxel_exploration=100000):
    # Apply median filter to the input slice data
    filtered_image = median_filter(slice_data, size=filter_size)
    mask2 = (filtered_image > tolerance[0]) & (filtered_image < tolerance[1])

    # Initialize new mask and visited array
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

    # Select random seed points from the true label mask
    seed_points = np.array(np.where(label_slice)).T
    
    if len(seed_points) == 0:
        raise ValueError("No seed points found in the true mask.")

    random_seeds = []
    for i in range(3):
        seed = seed_points[np.random.choice(seed_points.shape[0])]
        if mask2[seed[0], seed[1]]:
            random_seeds.append(seed)
    bfs(random_seeds, vis)

    # Fill holes in the final mask
    filled_mask = binary_fill_holes(new_mask)

    return filled_mask
    
def flood_segmentation(slice_data, label_slice, tolerance=(90, 200), tolerance_flood=30, filter_size=4):
    # Apply median filter to the input slice data
    filtered_image = median_filter(slice_data, size=filter_size)

    # Apply tolerance filter to the image
    filtered_image = np.where((filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1]), filtered_image, -1000)
    mask2 = (filtered_image >= tolerance[0]) & (filtered_image <= tolerance[1])

    # Find boundaries of the mask
    boundaries = find_boundaries(mask2, mode='inner')

    # Dilate boundaries to create an exclusion mask
    exclusion_mask = binary_dilation(boundaries)

    # Initialize new mask
    new_mask = np.zeros_like(slice_data, dtype=np.uint8)

    # Select random seed points from the true label mask
    seed_points = np.array(np.where(label_slice)).T
    
    if len(seed_points) == 0:
        raise ValueError("No seed points found in the true mask.")
    
    # Ensure seed points are within the mask2 region
    valid_seed_points = seed_points[mask2[seed_points[:, 0], seed_points[:, 1]]]

    if len(valid_seed_points) == 0:
        raise ValueError("No valid seed points found in the filtered mask.")

    flooded_masks = []
    for _ in range(3):
        random_seed = valid_seed_points[np.random.choice(valid_seed_points.shape[0])]
        flooded = flood(filtered_image, (random_seed[0], random_seed[1]), tolerance=tolerance_flood)
        flooded_masks.append(flooded)

    # Choose the largest mask
    largest_flooded_mask = max(flooded_masks, key=lambda x: np.sum(x))
    flooded = largest_flooded_mask
    combined_mask = np.zeros_like(slice_data, dtype=bool)
    for mask in flooded_masks:
        combined_mask = combined_mask | mask

    # Apply exclusion mask
    new_mask = np.where(exclusion_mask, 0, flooded)

    # Fill holes in the found mask
    filled_mask = binary_fill_holes(new_mask)
    return filled_mask
