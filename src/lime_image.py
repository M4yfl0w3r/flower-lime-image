import numpy as np 
import matplotlib.pyplot as pyplot

from typing import Callable
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class DatasetElement:
    
    def __init__(self, image: np.ndarray, label: int):
        self.image = image 
        self.label = label

    def draw(self) -> None:
        pyplot.imshow(self.image)
        pyplot.title(f"Label = {self.label}")
        pyplot.axis('off')
        pyplot.show()


class ImageExplainer:

    def __init__(self, image: np.ndarray, classifier_fn: Callable[[np.ndarray], int]):
        self.image = image 
        self.classifier_fn = classifier_fn

        self.linear_model_weights: list[float] = []
        self.generated_dataset: list[DatasetElement] = []
        self.perturbations: list[int] = []
        self.predictions: list[np.ndarray] = []

    def generate_dataset(self, num_samples: int) -> list[DatasetElement]:
        """Function that generates a new dataset that will be used to train the simple linear model."""
        
        self.superpixels: np.ndarray = quickshift(self.image, kernel_size = 5, ratio = 0.5)
        self.num_superpixels: int = self.superpixels.max() + 1
        replacement_color: int = 170

        for _ in range(num_samples):
            interpretable_features: np.ndarray = bernoulli.rvs(p = 0.5, size = self.num_superpixels)
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel_index in range(self.num_superpixels):
                if interpretable_features[superpixel_index] == 0:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel_index)
                    image_with_masked_superpixels[superpixel_mask] = replacement_color

            prediction: np.array = self.classifier_fn(image_with_masked_superpixels)

            self.predictions.append(prediction.max())
            self.perturbations.append(interpretable_features)
            self.generated_dataset.append(DatasetElement(image_with_masked_superpixels, np.max(prediction)))

        return self.generated_dataset
    
    def calculate_weights(self) -> list[int]:
        """Function that calculates the simple linear model weights based on differences between original
           and generated images."""
        
        for perturbation in self.perturbations:
            cosine_distance: np.ndarray = pairwise_distances(np.array(perturbation).reshape(1, -1),
                                                             np.ones(self.num_superpixels)[np.newaxis, :],
                                                             metric = 'cosine')
        
            self.linear_model_weights.append(cosine_distance[0, 0])

        return self.linear_model_weights

    def train_linear_model(self) -> None:
        """Train the simple linear model to see which superpixels are the most important."""
        
        model = Ridge()

        model.fit(X = np.array(self.perturbations),
                  y = np.array(self.predictions),
                  sample_weight = np.array(self.linear_model_weights))

        mask: np.ndarray = np.zeros(self.num_superpixels)
        most_active_superpixels: np.ndarray = np.argsort(model.coef_)[-5:] 
        mask[most_active_superpixels] = 1

        image_with_masked_superpixels: np.ndarray = np.copy(self.image)

        for superpixel_index in range(self.num_superpixels):
            if mask[superpixel_index] == 0:
                superpixel_mask: np.ndarray = (self.superpixels == superpixel_index)
                image_with_masked_superpixels[superpixel_mask] = 0

        _, (ax1, ax2) = pyplot.subplots(1, 2)
        ax1.imshow(self.image)
        ax2.imshow(image_with_masked_superpixels)
        pyplot.show()

    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.image, superpixels)
        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()
