import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms

from model import CNN
from lime_image import ImageExplainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_image_path = Path('../assets/test_img_2.jpg')
image = Image.open(test_image_path)
image = image.resize((224, 224))

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

    logits: torch.Tensor = model(image)
    probs: torch.Tensor = F.softmax(logits, dim = 1)
    return probs.detach().cpu().numpy()


explainer = ImageExplainer(image = np.array(image), 
                           classifier_fn = batch_predict,
                           num_classes = 4,
                           num_samples = 100)

explainer.explain(num_features = 2)
explainer.show_explanation()