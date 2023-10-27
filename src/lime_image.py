import numpy as np 
import matplotlib.pyplot as pl

from typing import Callable
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge

from skimage.segmentation import (
    quickshift,
    mark_boundaries 
)

class Params:
    image: np.ndarray = None
    classes: dict[int, str] = None
    classifier_fn: Callable[[np.ndarray], np.ndarray] = None

    num_samples: int = 100
    num_classes: int = 4
    num_features: int = 5
    
    replacement_color: int = 170
    kernel_width: float = 0.25

    # Quickshift config
    kernel_size: int = 3
    ratio: float = 0.2
    max_dist: int = 200


class ImageExplainer:

    def __init__(self, params: Params):
        self.params = params 

        self.predictions: np.ndarray = np.empty(shape = (self.params.num_samples, self.params.num_classes))
        self.perturbations: np.ndarray = np.empty(shape = (self.params.num_samples))
        self.model_weights: np.ndarray = np.empty(shape = self.params.num_samples) 
        self.explanation: np.ndarray = np.empty(shape = self.params.image.shape)

        self.top_label: int = 0

    def explain(self) -> None:
        self.generate_dataset()
        self.calculate_weights()
        self.train_linear_model()

    def generate_dataset(self) -> None:
        """Function that generates a new dataset that will be used to train the simple linear model."""
        
        self.superpixels = quickshift(image = self.params.image, 
                                      kernel_size = self.params.kernel_size, 
                                      ratio = self.params.ratio,
                                      max_dist = self.params.max_dist)

        self.num_superpixels = np.unique(self.superpixels).shape[0]
        self.interpretable_features: np.ndarray = np.empty((self.params.num_samples, self.num_superpixels))

        def _mask_random_superpixels(interpretable_features: np.ndarray) -> np.ndarray:
            image_with_masked_superpixels: np.ndarray = np.copy(self.params.image)

            for superpixel in range(self.num_superpixels):
                if interpretable_features[superpixel] == 0:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel)
                    image_with_masked_superpixels[superpixel_mask] = self.params.replacement_color

            return image_with_masked_superpixels

        perturbations = []
        predictions = []
        features = []
        labels = []

        for _ in range(self.params.num_samples):
            feature: np.ndarray = bernoulli.rvs(p = 0.5, size = self.num_superpixels)
            image_with_masked_superpixels: np.ndarray = _mask_random_superpixels(feature)
            prediction: np.ndarray = self.params.classifier_fn(image_with_masked_superpixels)
            label: int = prediction.argmax()

            labels.append(label)
            features.append(feature)
            predictions.append(prediction)
            perturbations.append(image_with_masked_superpixels)

        unique_elements, counts = np.unique(labels, return_counts = True)
        most_common_value = unique_elements[np.argmax(counts)]

        self.top_label = most_common_value
        self.interpretable_features = np.array(features)
        self.predictions = np.array(predictions)
        self.perturbations = np.array(perturbations)

    def calculate_weights(self) -> None:
        """Function that calculates the simple linear model weights based on differences between original
           and generated images."""
        
        distances: np.ndarray = pairwise_distances(self.interpretable_features,
                                                   np.ones(self.num_superpixels)[np.newaxis, :],
                                                   metric = 'cosine').ravel()
        
        self.model_weights = np.sqrt(np.exp(-(distances ** 2) / self.params.kernel_width ** 2))

    def train_linear_model(self) -> None:
        """Train the simple linear model to see which superpixels are the most important."""
        
        model = Ridge()
        
        model.fit(X = self.interpretable_features,
                  y = self.predictions[:, :, self.top_label].flatten(), 
                  sample_weight = self.model_weights)

        def _mark_most_active_superpixels() -> np.ndarray:
            mask: np.ndarray = np.zeros(self.num_superpixels)
            most_active_superpixels: np.ndarray = np.argsort(model.coef_)[-self.params.num_features:]
            mask[most_active_superpixels] = True
            image_with_masked_superpixels: np.ndarray = np.copy(self.params.image)

            for superpixel in range(self.num_superpixels):
                if not mask[superpixel]:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel)
                    image_with_masked_superpixels[superpixel_mask] = 0

            return image_with_masked_superpixels
        
        self.explanation = _mark_most_active_superpixels()

    def show_explanation(self) -> None:
        fig, (ax1, ax2) = pl.subplots(1, 2)

        fig.suptitle(f'Explained class: {self.params.classes[self.top_label]}')

        ax1.imshow(self.params.image)
        ax1.set_title('Original image')
        ax1.axis('off')

        ax2.imshow(self.explanation)
        ax2.set_title('Explanation \n (Most active superpixels)')
        ax2.axis('off')

        pl.show()

    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.params.image, superpixels)
        pl.imshow(boundaries)
        pl.axis('off')
        pl.show()
