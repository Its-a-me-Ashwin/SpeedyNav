import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define an EfficientNet-based edge detection model using all but the last layer
class EfficientNetEdgeNet(nn.Module):
    def __init__(self):
        super(EfficientNetEdgeNet, self).__init__()
        # Load pretrained EfficientNet (EfficientNet-B0 in this case)
        efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Use all EfficientNet layers except the final classification layer
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])  # Skip the classifier
        
        # Output layer for edge detection
        # Final layer uses the same number of channels as the final EfficientNet output, but outputs 1 channel for edges
        self.edge_out = nn.Conv2d(1280, 1, kernel_size=1)  # EfficientNet-B0 has 1280 channels before the classifier

    def forward(self, x):
        # Pass input through EfficientNet layers
        features = self.features(x)
        
        # Output edge map using final convolution layer
        edge_map = self.edge_out(features)
        
        # Resize to the original input size
        edge_map = F.interpolate(edge_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        return torch.sigmoid(edge_map)

# Load and preprocess images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Display the edge detection result
def display_edge_map(image_tensor, edge_map):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert image tensor to numpy for visualization
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0, 1]
    
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Display the edge map
    edge_np = edge_map.squeeze().detach().cpu().numpy()
    ax[1].imshow(edge_np, cmap='gray')
    ax[1].set_title('Detected Edges')
    ax[1].axis('off')
    
    plt.show()

# Testing EfficientNet-based edge detection model
def test_efficientnet_edge_model(image_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = EfficientNetEdgeNet().to(device)
    model.eval()  # Set model to evaluation mode
    
    for image_path in image_paths:
        image = load_image(image_path).to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            edge_map = model(image)
        
        # Display original image and edge map
        display_edge_map(image.cpu(), edge_map.cpu())

# Example usage
if __name__ == "__main__":
    # List of images for testing
    image_paths = ['path_to_image_1.jpg', 'path_to_image_2.jpg']  # Replace with your image paths
    test_efficientnet_edge_model(image_paths)
