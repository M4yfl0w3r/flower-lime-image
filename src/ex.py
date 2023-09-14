#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pyplot

from PIL import Image
from pathlib import Path

from lime_image import ImageExplainer


test_image_path = Path('../assets/test_img.jpg')
image = Image.open(test_image_path)

explainer = ImageExplainer(np.array(image))
dataset = explainer.generate_dataset(num_samples = 2)

