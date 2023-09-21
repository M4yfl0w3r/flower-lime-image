import torch
import numpy as np
import matplotlib.pyplot as pyplot

from PIL import Image
from pathlib import Path
from torchvision import transforms

from model import CNN
from lime_image import ImageExplainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_image_path = Path('../assets/test_img.jpg')
image = Image.open(test_image_path)

model_path = Path('../assets/model.pth')
model = CNN()
model.load_state_dict(torch.load(str(model_path), map_location = torch.device('cpu')))

@torch.no_grad()
def batch_predict(image: np.ndarray) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image: Image = Image.fromarray(image)
    image: torch.Tensor = transform(image).float()
    image: torch.Tensor = image.to(device)
    image: torch.Tensor = image.unsqueeze(0)

    output: torch.Tensor = model(image)
    return output.data.cpu().numpy()


explainer = ImageExplainer(np.array(image), batch_predict)
explainer.explain(num_samples = 1, num_features = 5)
