import numpy as np 
import matplotlib.pyplot as pyplot

from typing import Callable
from scipy.stats import bernoulli
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class DatasetElement:
    
    def __init__(self, image: np.ndarray, label: int):
        self.image = image 
        self.label = label

    def draw(self):
        pyplot.imshow(self.image)
        pyplot.title(f"Label = {self.label}")
        pyplot.axis('off')
        pyplot.show()


class ImageExplainer:

    def __init__(self, image: np.ndarray, classifier_fn: Callable[[np.ndarray], int]):
        self.image = image 
        self.classifier_fn = classifier_fn

    def generate_dataset(self, num_samples: int):
        """Function that generates a new dataset that will be used to train a simple linear model."""
        
        superpixels: np.ndarray = quickshift(self.image, kernel_size = 5, ratio = 0.5)
        num_superpixels: int    = superpixels.max() + 1
        replacement_color: int  = 170 
        generated_dataset: list[DatasetElement] = []

        for _ in range(num_samples):
            interpretable_features: np.ndarray = bernoulli.rvs(p = 0.5, size = num_superpixels)
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel_index in range(num_superpixels):
                if interpretable_features[superpixel_index] == 0:
                    superpixel_mask: np.ndarray = (superpixels == superpixel_index)
                    image_with_masked_superpixels[superpixel_mask] = replacement_color

            prediction = self.classifier_fn(image_with_masked_superpixels)
            generated_dataset.append(DatasetElement(image_with_masked_superpixels, prediction))

        return generated_dataset
    
    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.image, superpixels)
        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()
