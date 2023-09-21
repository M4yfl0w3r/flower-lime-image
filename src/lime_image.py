import numpy as np 
import matplotlib.pyplot as pyplot

from typing import Callable
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class ImageExplainer:

    def __init__(self, image: np.ndarray, classifier_fn: Callable[[np.ndarray], np.ndarray]):
        self.image = image
        self.classifier_fn = classifier_fn

        self.model_weights: list[float] = []
        self.perturbations_images: list[np.ndarray] = []
        self.perturbations_activations: list[np.ndarray] = []
        self.predictions: list[np.ndarray] = []
        self.superpixels: list[int] = []
        self.num_superpixels: int = 0

    def explain(self, num_samples: int, num_features: int) -> None:
        self.generate_dataset(num_samples)
        self.calculate_weights()
        self.train_linear_model(num_features)

    def generate_dataset(self, num_samples: int) -> None:
        """Function that generates a new dataset that will be used to train the simple linear model."""
        
        self.superpixels: np.ndarray = quickshift(self.image, kernel_size = 5, ratio = 0.5)
        self.num_superpixels: int = self.superpixels.max() + 1
        replacement_color: int = 170

        def _mask_random_superpixels(interpretable_features: np.ndarray) -> np.ndarray:
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel in range(self.num_superpixels):
                if interpretable_features[superpixel] == 0:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel)
                    image_with_masked_superpixels[superpixel_mask] = replacement_color

            return image_with_masked_superpixels
                    
        for _ in range(num_samples):
            interpretable_features: np.ndarray = bernoulli.rvs(p = 0.5, size = self.num_superpixels)
            image_with_masked_superpixels: np.ndarray = _mask_random_superpixels(interpretable_features)
            prediction: np.ndarray = self.classifier_fn(image_with_masked_superpixels)

            self.predictions.append(prediction.argmax())
            self.perturbations_images.append(image_with_masked_superpixels)
            self.perturbations_activations.append(interpretable_features)
    
    def calculate_weights(self) -> None:
        """Function that calculates the simple linear model weights based on differences between original
           and generated images."""
        
        for perturbation in self.perturbations_activations:
            distance: np.ndarray = pairwise_distances(np.array(perturbation).reshape(1, -1),
                                                      np.ones(self.num_superpixels)[np.newaxis, :],
                                                      metric = 'cosine')
        
            self.model_weights.append(distance[0, 0])

    def train_linear_model(self, num_features: int) -> None:
        """Train the simple linear model to see which superpixels are the most important."""
        
        model = Ridge()

        model.fit(X = np.array(self.perturbations_activations),
                  y = np.array(self.predictions),
                  sample_weight = np.array(self.model_weights))

        def _mark_most_active_superpixels() -> np.ndarray:
            mask: np.ndarray = np.zeros(self.num_superpixels)
            most_active_superpixels: np.ndarray = np.argsort(model.coef_)[-num_features:] 
            mask[most_active_superpixels] = 1

            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel_index in range(self.num_superpixels):
                if mask[superpixel_index] == 0:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel_index)
                    image_with_masked_superpixels[superpixel_mask] = 0

            return image_with_masked_superpixels

        explained_image = _mark_most_active_superpixels()
        self.compare_original_image_and_explaination(explained_image)

    def compare_original_image_and_explaination(self, explaination: np.ndarray):
        _, (ax1, ax2) = pyplot.subplots(1, 2)
        ax1.imshow(self.image)
        ax2.imshow(explaination)
        ax1.axis('off')
        ax2.axis('off')
        pyplot.show()

    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.image, superpixels)
        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()

