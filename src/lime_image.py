import numpy as np 
import matplotlib.pyplot as pyplot

from typing import Callable
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries 


class ImageExplainer:

    def __init__(self, 
                 image: np.ndarray, 
                 classifier_fn: Callable[[np.ndarray], np.ndarray], 
                 num_samples: int,
                 num_classes: int):
        
        self.image = image
        self.classifier_fn = classifier_fn

        self.model_weights = [] 
        self.perturbations_images = []
        self.interpretable_features = []

        self.num_samples = num_samples
        
        self.predictions = []
        self.num_superpixels: int = 0
        self.superpixels: list = []
        self.explanation: np.ndarray = np.zeros(shape = self.image.shape)

    def explain(self, num_features: int) -> None:
        self.generate_dataset(self.num_samples)
        self.calculate_weights()
        self.train_linear_model(num_features)

    def generate_dataset(self, num_samples: int) -> None:
        """Function that generates a new dataset that will be used to train the simple linear model."""
        
        self.superpixels: np.ndarray = quickshift(self.image, kernel_size = 6, ratio = 0.2, max_dist = 200)
        self.num_superpixels: int = np.unique(self.superpixels).shape[0]
        replacement_color: int = 170

        def _mask_random_superpixels(interpretable_features: np.ndarray) -> np.ndarray:
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel in range(self.num_superpixels):
                if interpretable_features[0][superpixel] == 0:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel)
                    image_with_masked_superpixels[superpixel_mask] = replacement_color

            return image_with_masked_superpixels

        for _ in range(num_samples):
            self.interpretable_features = bernoulli.rvs(p = 0.5, size = (num_samples, self.num_superpixels))
            image_with_masked_superpixels: np.ndarray = _mask_random_superpixels(self.interpretable_features)
            prediction: np.ndarray = self.classifier_fn(image_with_masked_superpixels)

            self.predictions.append(prediction)
            self.perturbations_images.append(image_with_masked_superpixels)

    def calculate_weights(self) -> None:
        """Function that calculates the simple linear model weights based on differences between original
           and generated images."""
        
        distances: np.ndarray = pairwise_distances(self.interpretable_features,
                                                   np.ones(self.num_superpixels)[np.newaxis, :],
                                                   metric = 'cosine').ravel()
        
        self.model_weights = np.sqrt(np.exp(-(distances ** 2) / .25 ** 2))

    def train_linear_model(self, num_features: int) -> None:
        """Train the simple linear model to see which superpixels are the most important."""
        
        model = Ridge()

        y = []
        
        # TODO: Get the top label
        label = 2 # TODO: 

        # TODO: numpy function
        for prediction in self.predictions:
            y.append(prediction[0][label])

        x = self.interpretable_features
        y = np.array(y)
        
        model.fit(x, y, sample_weight = self.model_weights)

        def _mark_most_active_superpixels() -> np.ndarray:
            mask: np.ndarray = np.zeros(self.num_superpixels)
            most_active_superpixels: np.ndarray = np.argsort(model.coef_)[-num_features:]
            mask[most_active_superpixels] = True
            image_with_masked_superpixels: np.ndarray = np.copy(self.image)

            for superpixel in range(self.num_superpixels):
                if not mask[superpixel]:
                    superpixel_mask: np.ndarray = (self.superpixels == superpixel)
                    image_with_masked_superpixels[superpixel_mask] = 0

            return image_with_masked_superpixels
        
        self.explanation = _mark_most_active_superpixels()

    def show_explanation(self):
        _, (ax1, ax2) = pyplot.subplots(1, 2)

        ax1.imshow(self.image)
        ax1.set_title('Original image')
        ax1.axis('off')

        ax2.imshow(self.explanation)
        ax2.set_title('Explanation \n (Most active superpixels)')
        ax2.axis('off')

        pyplot.show()

    def show_superpixels_boundaries(self, superpixels: np.ndarray) -> None:
        boundaries: np.ndarray = mark_boundaries(self.image, superpixels)
        pyplot.imshow(boundaries)
        pyplot.axis('off')
        pyplot.show()
