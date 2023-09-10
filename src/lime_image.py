import torch
import numpy as np 
import matplotlib.pyplot as pyplot

from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class ImageExplainer:

    def __init__(self, image):
        self.image = image 


    def generate_dataset(self, num_samples: int):
        """Function that generates a new dataset that 
           will be used to train a simple linear model."""
        
        superpixels: np.ndarray = quickshift(self.image, kernel_size = 4)
        boundaries: np.ndarray  = mark_boundaries(self.image, superpixels)

        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()
        
