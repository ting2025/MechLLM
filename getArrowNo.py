from skimage import io, measure
import numpy as np

def getArrowNo(image_path: str):
    image = io.imread(image_path) 
    mask = np.any(image != [0, 0, 0], axis=-1)
    labels = measure.label(mask, connectivity=2)
    num_patches = labels.max()
    return num_patches


