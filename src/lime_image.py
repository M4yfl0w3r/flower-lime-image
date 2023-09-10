import torch
import numpy as np 

from skimage.segmentation import quickshift

class ImageExplainer:

    def __init__(self):
        pass


    def generate_dataset(self, image: np.ndarray, num_samples: int):
        """Function that generates a new dataset that 
           will be used to train a simple linear model."""
        
        superpixels: np.ndarray = quickshift(image = image, kernel_size = 4)

       

