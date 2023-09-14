import numpy as np 
import matplotlib.pyplot as pyplot

from typing import Callable
from scipy.stats import bernoulli
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class ImageExplainer:

    def __init__(self, image: np.ndarray, prediction_fn: Callable[[np.ndarray], np.ndarray]):
        self.image = image 
        self.prediction_fn = prediction_fn

    def generate_dataset(self, num_samples: int) -> np.ndarray:
        """Function that generates a new dataset that will be used to train a simple linear model."""
        
        superpixels: np.ndarray = quickshift(self.image, kernel_size = 5, ratio = 0.5)
        num_superpixels: int    = superpixels.max() + 1
        generated_dataset: list = []

        for _ in range(num_samples):
            interpretable_features: np.ndarray = bernoulli.rvs(p = 0.5, size = num_superpixels)
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)
            replacement_color: int = 170 

            for superpixel_index in range(num_superpixels):
                if interpretable_features[superpixel_index] == 0:
                    superpixel_mask: np.ndarray = (superpixels == superpixel_index)
                    image_with_masked_superpixels[superpixel_mask] = replacement_color

            generated_dataset.append(image_with_masked_superpixels)

        for image in generated_dataset:
            pyplot.imshow(image)
            pyplot.axis('off')
            pyplot.show()

    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.image, superpixels)
        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()
        
