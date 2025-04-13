import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Load a pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Compute gradients for a given input image
from torchvision.transforms import ToTensor
from PIL import Image

# Load an example image and convert it to a tensor
image_path = "sample.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image = ToTensor()(image)
image.requires_grad = True

output = model(image.unsqueeze(0))
class_idx = torch.argmax(output)
output[0, class_idx].backward()

# Visualize saliency map
saliency = image.grad.abs().squeeze().permute(1, 2, 0)
plt.imshow(saliency.numpy(), cmap="hot")
plt.title("Saliency Map")
plt.axis("off")
plt.show()