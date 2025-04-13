import torch
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

# Load a pretrained VGG model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval()

# Define content and style loss functions
def content_loss(target, content):
    return torch.mean((target - content) ** 2)

def gram_matrix(feature_map):
    _, C, H, W = feature_map.size()
    features = feature_map.view(C, H * W)
    return torch.mm(features, features.t()) / (C * H * W)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)

# Train the model to optimize the style transfer
# Initialize the image as a copy of the content image with gradients enabled
from PIL import Image
from torchvision import transforms

# Load and preprocess the content image
content_image_path = "path_to_content_image.jpg"  # Replace with the actual path to your content image
content_image = Image.open(content_image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match VGG input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
content_image = preprocess(content_image).unsqueeze(0)  # Add batch dimension

# Load and preprocess the style image
style_image_path = "path_to_style_image.jpg"  # Replace with the actual path to your style image
style_image = Image.open(style_image_path).convert("RGB")
style_image = preprocess(style_image).unsqueeze(0)  # Add batch dimension

image = content_image.clone().requires_grad_(True)

optimizer = optim.Adam([image], lr=0.01)
for i in range(500):
    target_content = vgg(content_image)
    target_style = vgg(style_image)
    loss = content_loss(target_content, content_image) + 1e6 * style_loss(target_style, style_image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Display final style-transferred image
plt.imshow(image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Stylized Image")
plt.axis("off")
plt.show()